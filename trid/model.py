"""Decoder-only Transformer. No pretrained weights or downloaded model code."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 2048
    dim: int = 256
    layers: int = 6
    heads: int = 8
    hidden_dim: int = 768
    context_length: int = 256
    rope_theta: float = 10000.0
    num_intents: int = 0
    # Opt-in architecture experiments; omitted fields preserve existing checkpoints.
    kv_heads: int | None = None
    qk_norm: bool = False

    def __post_init__(self):
        if self.num_intents < 0:
            raise ValueError("num_intents는 0 이상이어야 합니다.")
        if min(self.vocab_size, self.dim, self.layers, self.heads,
               self.hidden_dim, self.context_length) <= 0:
            raise ValueError("모델 설정은 양수여야 합니다.")
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("dim은 heads로 나누어지고 head 차원은 짝수여야 합니다.")
        if self.kv_heads is not None and (type(self.kv_heads) is not int
                or self.kv_heads < 1 or self.heads % self.kv_heads):
            raise ValueError("kv_heads는 heads를 나누는 양의 정수여야 합니다.")
        if type(self.qk_norm) is not bool:
            raise ValueError("qk_norm은 true 또는 false여야 합니다.")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return normed.to(x.dtype) * self.weight


def rotary(x, offset, theta, tables=None):
    if tables is None:
        size = x.shape[-1]
        inv = theta ** (-torch.arange(0, size, 2, device=x.device, dtype=torch.float32) / size)
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * inv[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    else:
        end = offset + x.shape[-2]
        cos, sin = [table[offset:end].to(x.dtype) for table in tables]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.kv_heads = cfg.heads if cfg.kv_heads is None else cfg.kv_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.theta = cfg.rope_theta
        self.qkv = nn.Linear(cfg.dim, cfg.dim + 2 * self.kv_dim, bias=False)
        # Per-head normalization before RoPE, also used in Qwen3. See
        # reports/qwen-architecture-review.txt for reference versions and scope.
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.output = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # Constant position rotations; avoid rebuilding trigonometric tensors for
        # every query/key projection. Nonpersistent buffers keep old checkpoints compatible.
        inv = self.theta ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        angles = torch.arange(cfg.context_length, dtype=torch.float32)[:, None] * inv[None, :]
        self.register_buffer('rope_cos', angles.cos(), persistent=False)
        self.register_buffer('rope_sin', angles.sin(), persistent=False)

    def forward(self, x, cache=None, use_cache=False):
        b, t, d = x.shape
        projected = self.qkv(x)
        if self.kv_heads == self.heads:
            q, k, v = projected.chunk(3, dim=-1)
        else:
            q, k, v = projected.split((d, self.kv_dim, self.kv_dim), dim=-1)
        q = q.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        k, v = [a.reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2) for a in (k, v)]
        q, k = self.q_norm(q), self.k_norm(k)
        offset = 0 if cache is None else cache[0].shape[-2]
        if offset + t > self.rope_cos.shape[0]:
            # Fine-tuning can extend Trid.cfg after loading an older checkpoint.
            size = max(offset + t, 2 * self.rope_cos.shape[0])
            inv = self.theta ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            angles = torch.arange(size, dtype=torch.float32)[:, None] * inv[None, :]
            self.rope_cos = angles.cos().to(device=x.device, dtype=x.dtype)
            self.rope_sin = angles.sin().to(device=x.device, dtype=x.dtype)
        tables = (self.rope_cos, self.rope_sin)
        q, k = rotary(q, offset, self.theta, tables), rotary(k, offset, self.theta, tables)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=-2), torch.cat((cache[1], v), dim=-2)
        # Keep the cache compact. Expansion is temporary for CPU/MPS SDPA
        # portability; this does not promise a faster attention kernel.
        next_cache = (k, v) if use_cache else None
        if self.kv_heads != self.heads:
            groups = self.heads // self.kv_heads
            k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        # Cached decoding must not apply the top-left causal mask to a one-token query.
        mask = None
        if offset and t > 1:
            mask = torch.arange(k.shape[-2], device=x.device)[None, :] <= (
                offset + torch.arange(t, device=x.device)[:, None])
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                          is_causal=(offset == 0 and t > 1))
        y = y.transpose(1, 2).contiguous().reshape(b, t, d)
        return self.output(y), next_cache


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        a, cache = self.attn(self.attn_norm(x), cache, use_cache)
        x = x + a
        n = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n)), cache


class Trid(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.intent_head = nn.Linear(cfg.dim, cfg.num_intents, bias=False) if cfg.num_intents else None
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attn.output.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, cache=None, use_cache=False, return_hidden=False, hidden_only=False):
        offset = 0 if cache is None else cache[0][0].shape[-2]
        if tokens.shape[1] + offset > self.cfg.context_length:
            raise ValueError(f"문맥 한도 {self.cfg.context_length} 토큰을 초과했습니다.")
        x, new_cache = self.embedding(tokens), []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if cache is None else cache[i], use_cache)
            new_cache.append(c)
        # Tied input/output embeddings reduce model size.
        hidden = self.norm(x)
        if hidden_only:
            return hidden
        result = (F.linear(hidden, self.embedding.weight), new_cache if use_cache else None)
        return (*result, hidden) if return_hidden else result

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

"""Late-interaction (ColBERT-style) retrieval head on the trid backbone
(reports/closed-network-architecture-survey.md §2 ColBERT row). No external embedding model.

Query and passage are encoded separately with a marker prefix (`질문: ` / `문단: `) through the causal backbone
(`hidden_only=True`), projected to DIM dimensions and L2-normalised per token. Score = MaxSim: for each query token
the maximum cosine over passage tokens, summed. Passages are pre-encoded once into a right-padded fp16 array
(PassageIndex) so a query costs one short backbone pass plus one batched matmul; the index can be saved with numpy.
Training: in-batch negatives (every other passage in the batch) plus hard negatives (BM25 top-k non-gold and
same-prompt distractors) with a softmax cross-entropy over MaxSim scores.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from .model import Config, Trid
from .runtime import device_for

DIM = 128
QUERY_MARK, PASSAGE_MARK = '질문: ', '문단: '
QUERY_TOKENS, PASSAGE_TOKENS = 64, 512
STAGE = 'retrieval_head_v1'
WEIGHTS = 'retrieval.safetensors'


class RetrievalHead(nn.Module):
    def __init__(self, cfg, dim=DIM):
        super().__init__()
        self.backbone = Trid(cfg)
        self.proj = nn.Linear(cfg.dim, dim, bias=False)
        self.dim = dim

    def forward(self, tokens, lengths, freeze_backbone=False):
        """tokens: [B, T] right-padded ids (bos first); lengths: [B]. Returns unit vectors [B, T-1, dim] for the
        tokens after <|bos|> and a bool mask [B, T-1] of real positions. Right padding never leaks into real
        positions because the backbone is causal, so no attention mask is needed. With freeze_backbone the
        backbone runs without autograd (only the projection trains)."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not freeze_backbone):
            h = self.backbone(tokens, hidden_only=True)[:, 1:]
        v = F.normalize(self.proj(h), dim=-1)
        mask = torch.arange(1, tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        return v * mask[..., None], mask


class PassageIndex:
    """Pre-encoded passages: vectors [N, L, dim] fp16 (zero at padding) and lengths [N]."""

    def __init__(self, vectors, lengths, texts=None):
        if vectors.ndim != 3 or len(vectors) != len(lengths):
            raise ValueError('vectors must be [N, L, dim] with one length per passage')
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float16)
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.texts = list(texts) if texts is not None else None

    def __len__(self):
        return len(self.lengths)

    @property
    def nbytes(self):
        return self.vectors.nbytes + self.lengths.nbytes

    def save(self, path):
        np.savez(path, vectors=self.vectors, lengths=self.lengths)
        if self.texts is not None:
            Path(path).with_suffix('.texts.json').write_text(json.dumps(self.texts, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        texts_path = Path(path).with_suffix('.texts.json')
        texts = json.loads(texts_path.read_text()) if texts_path.exists() else None
        return cls(z['vectors'], z['lengths'], texts)


def tokenize(tok, text, marker, max_tokens):
    """[bos] + ids of marker+text, truncated to max_tokens total (never below two tokens)."""
    ids = tok.encode(marker + text).ids[:max(1, max_tokens - 1)]
    return [tok.token_to_id('<|bos|>')] + ids


def pack(seqs, pad, device):
    size = max(len(s) for s in seqs)
    tokens = torch.full((len(seqs), size), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        tokens[i, :len(s)] = torch.tensor(s)
    return tokens.to(device), torch.tensor([len(s) for s in seqs], device=device)


def maxsim(q, q_mask, p, p_mask):
    """q: [B, Lq, d], q_mask: [B, Lq]; p: [N, Lp, d], p_mask: [N, Lp]. Returns [B, N] MaxSim scores."""
    sim = torch.einsum('bqd,npd->bnqp', q, p.to(q.dtype))
    sim = sim.masked_fill(~p_mask[None, :, None, :], -1e4)
    best = sim.max(dim=-1).values  # [B, N, Lq]
    return (best * q_mask[:, None, :]).sum(-1)


def encode_batch(model, tok, texts, marker, max_tokens, device):
    """(vectors [B, L, dim] float32 on device, lengths [B] excluding bos) for a batch of raw texts."""
    seqs = [tokenize(tok, t, marker, max_tokens) for t in texts]
    tokens, lengths = pack(seqs, tok.token_to_id('<|pad|>'), device)
    v, mask = model(tokens, lengths)
    return v, mask


@torch.inference_mode()
def encode_passages(model, tok, texts, device, batch_size=8, max_tokens=PASSAGE_TOKENS):
    """PassageIndex for texts; sorted-by-length batching, result in input order, fp16 storage."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    pieces = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        v, mask = encode_batch(model, tok, [texts[i] for i in idx], PASSAGE_MARK, max_tokens, device)
        lengths = mask.sum(-1).tolist()
        v = v.to('cpu', torch.float16).numpy()
        for j, i in enumerate(idx):
            pieces[i] = v[j, :lengths[j]]
    size = max(len(p) for p in pieces)
    vectors = np.zeros((len(pieces), size, model.dim), dtype=np.float16)
    for i, p in enumerate(pieces):
        vectors[i, :len(p)] = p
    return PassageIndex(vectors, [len(p) for p in pieces], texts)


@torch.inference_mode()
def encode_query(model, tok, text, device, max_tokens=QUERY_TOKENS):
    """Unit token vectors [Lq, dim] (float32, cpu) of one query."""
    v, mask = encode_batch(model, tok, [text], QUERY_MARK, max_tokens, device)
    return v[0, :int(mask[0].sum())].to('cpu', torch.float32)


def score(query_vectors, index, chunk=256):
    """MaxSim of one query ([Lq, dim]) against every passage of a PassageIndex -> np.ndarray [N] float32."""
    q = torch.as_tensor(query_vectors, dtype=torch.float32)
    out = np.empty(len(index), dtype=np.float32)
    for start in range(0, len(index), chunk):
        p = torch.from_numpy(index.vectors[start:start + chunk]).to(torch.float32)
        lengths = torch.from_numpy(index.lengths[start:start + chunk]).long()
        p_mask = torch.arange(p.shape[1])[None, :] < lengths[:, None]
        q_mask = torch.ones(1, q.shape[0], dtype=torch.bool)
        out[start:start + chunk] = maxsim(q[None], q_mask, p, p_mask)[0].numpy()
    return out


def search(model, tok, query, index, k, device='cpu'):
    """Top-k [(passage_position, score)] for a query string, best first (stable on ties)."""
    s = score(encode_query(model, tok, query, device), index)
    order = np.argsort(-s, kind='stable')[:k]
    return [(int(i), float(s[i])) for i in order]


def contrastive_loss(model, tok, queries, positives, negatives, device, query_tokens=QUERY_TOKENS,
                     passage_tokens=PASSAGE_TOKENS, temperature=.05, freeze_backbone=False):
    """Softmax CE over MaxSim scores. Columns: the B positives (in-batch negatives for the other rows) followed by
    `negatives` (hard negatives shared by all rows). Scores are divided by the query length so the temperature is
    length-independent. Returns (loss, detached scores [B, B+H])."""
    pad = tok.token_to_id('<|pad|>')
    q, q_mask = model(*pack([tokenize(tok, t, QUERY_MARK, query_tokens) for t in queries], pad, device), freeze_backbone)
    p, p_mask = model(*pack([tokenize(tok, t, PASSAGE_MARK, passage_tokens) for t in list(positives) + list(negatives)],
                            pad, device), freeze_backbone)
    scores = maxsim(q, q_mask, p, p_mask) / q_mask.sum(-1, keepdim=True)
    target = torch.arange(len(queries), device=device)
    return F.cross_entropy(scores / temperature, target), scores.detach()


def load_head(path, device='auto'):
    """(model, tokenizer, device) from a run directory written by scripts/train_retrieval_head.py."""
    root = Path(path)
    meta = json.loads((root / 'training.json').read_text())
    if meta.get('stage') != STAGE:
        raise ValueError('검색 헤드 모델 경로를 지정하세요.')
    model = RetrievalHead(Config(**json.loads((root / 'config.json').read_text())), meta.get('projection_dim', DIM))
    model.load_state_dict(load_file(str(root / WEIGHTS)))
    device = device_for(device)
    return model.to(device).eval(), Tokenizer.from_file(str(root / 'tokenizer.json')), device

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .model import Config, Trid
from .text import prompt_ids


def device_for(name="auto"):
    available = {"cuda": torch.cuda.is_available, "mps": torch.backends.mps.is_available}
    if name == "auto":
        return next((d for d, ok in available.items() if ok()), "cpu")
    if name in available and not available[name]():
        raise ValueError(f"이 컴퓨터에서 {name.upper()}를 사용할 수 없습니다.")
    return name


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def load_run(path, device="auto"):
    root = Path(path)
    cfg = Config(**json.loads((root / "config.json").read_text()))
    model = Trid(cfg)
    model.load_state_dict(load_file(str(root / "model.safetensors")))
    device = device_for(device)
    model.to(device).eval()
    return model, Tokenizer.from_file(str(root / "tokenizer.json")), device


@torch.inference_mode()
def generate(model, tok, prompt, device, max_tokens=48, temperature=0.0, raw_text=False, history=None, constraint=None):
    if not 1 <= max_tokens <= model.cfg.context_length:
        raise ValueError("출력 토큰 수가 유효하지 않습니다.")
    dropped = 0
    if raw_text:
        from .text import SPECIAL
        if any(s in prompt for s in SPECIAL):
            raise ValueError("입력에 예약된 제어 토큰을 사용할 수 없습니다.")
        ids = [tok.token_to_id("<|bos|>")] + tok.encode(prompt).ids
    else:
        history = list(history or [])
        ids = prompt_ids(tok, prompt, history)
        while history and len(ids) + max_tokens > model.cfg.context_length:
            history.pop(0)
            dropped += 1
            ids = prompt_ids(tok, prompt, history)
    if len(ids) + max_tokens > model.cfg.context_length:
        raise ValueError(f"입력 {len(ids)} + 출력 예약 {max_tokens} 토큰이 문맥 한도 {model.cfg.context_length}를 초과합니다.")
    x = torch.tensor([ids], device=device)
    eos = tok.token_to_id("<|eos|>")
    forbidden = [tok.token_to_id(t) for t in ["<|pad|>", "<|bos|>", "<|user|>", "<|assistant|>"]]
    output, cache, first = [], None, None
    sync(device)
    start = time.perf_counter()
    for step in range(max_tokens):
        logits, cache = model(x, cache=cache, use_cache=True)
        scores = logits[:, -1, :].clone()
        scores[:, forbidden] = -float("inf")
        if constraint is not None:
            token = torch.tensor([[constraint.choose(scores[0], eos, max_tokens - step)]], device=device)
        elif temperature > 0:
            token = torch.multinomial((scores / temperature).softmax(-1), 1)
        else:
            token = scores.argmax(-1, keepdim=True)
        value = token.item()
        if first is None:
            sync(device)
            first = time.perf_counter() - start
        output.append(value)
        if value == eos:
            break
        x = token
    sync(device)
    elapsed = time.perf_counter() - start
    return {"answer": tok.decode(output, skip_special_tokens=True).strip(),
            "input_tokens": len(ids), "generated_tokens_including_eos": len(output),
            "first_token_seconds": round(first, 4), "total_seconds": round(elapsed, 4),
            "tokens_per_second_including_prefill": round(len(output) / elapsed, 2),
            "history_turns_dropped": dropped,
            "stopped_on_eos": bool(output and output[-1] == eos)}

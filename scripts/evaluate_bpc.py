"""도메인별 held-out BPC(bits per character): `text` 필드가 있는 JSONL 파일마다 총 NLL(bit) / 총 글자 수.

토큰 손실은 어휘 크기에 따라 비교할 수 없으므로 글자 단위로 정규화한다(Tao et al. §A.5).
문서마다 [BOS]+토큰 열을 context_length 창으로 stride만큼 밀며 평가하고, 겹치는 구간은 한 번만(가장 긴 문맥에서) 센다.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from trid.corpus import sha256, write_json
from trid.runtime import load_run


def read_texts(path, limit):
    with Path(path).open(encoding='utf-8') as f:
        rows = (json.loads(l) for l in f if l.strip())
        texts = [r['text'] for r in rows if r.get('text')]
    return texts[:limit] if limit else texts


def windows(ids, length, stride):
    """(시작, 채점 대상 수) — 창은 ids[s:s+length+1](입력 length개, 목표 length개). 첫 창은 모든 목표를, 이후 창은 새로 들어온 목표만 채점."""
    if len(ids) <= length + 1:
        return [(0, len(ids) - 1)]
    out, start = [(0, length)], 0
    while start + length + 1 < len(ids):
        nxt = min(start + stride, len(ids) - length - 1)
        out.append((nxt, nxt - start))
        start = nxt
    return out


@torch.inference_mode()
def nll_file(model, tok, device, texts, stride, batch_size):
    bos, length = tok.token_to_id('<|bos|>'), model.cfg.context_length
    docs = [[bos] + tok.encode(t).ids for t in texts]
    jobs = [(d, s, n) for d, ids in enumerate(docs) for s, n in windows(ids, length, stride)]
    total, scored = 0.0, 0
    for offset in range(0, len(jobs), batch_size):
        batch = jobs[offset:offset + batch_size]
        x = torch.zeros((len(batch), length), dtype=torch.long)
        y = torch.full_like(x, -100)
        for r, (d, s, n) in enumerate(batch):
            seg = docs[d][s:s + length + 1]
            x[r, :len(seg) - 1] = torch.tensor(seg[:-1])
            y[r, len(seg) - 1 - n:len(seg) - 1] = torch.tensor(seg[-n:])
        logits, _ = model(x.to(device))
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.to(device).reshape(-1), ignore_index=-100, reduction='sum').item()
        scored += int((y != -100).sum().item())
    tokens = sum(len(d) - 1 for d in docs)
    assert scored == tokens, (scored, tokens)
    return total, tokens, len(jobs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inputs', nargs='+', required=True, help='`text` 필드가 있는 JSONL 경로들')
    p.add_argument('--limit', type=int, default=0, help='파일당 앞에서부터 N문서만 (0=전체)')
    p.add_argument('--stride', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('Use a new report path.')
    torch.set_num_threads(a.threads)
    model, tok, device = load_run(a.run, a.device)
    if not 1 <= a.stride <= model.cfg.context_length:
        raise ValueError('stride는 1 이상 context_length 이하여야 합니다.')
    files = {}
    for path in a.inputs:
        texts = read_texts(path, a.limit)
        chars = sum(len(t) for t in texts)
        start = time.perf_counter()
        nats, tokens, n_windows = nll_file(model, tok, device, texts, a.stride, a.batch_size)
        files[path] = {'docs': len(texts), 'chars': chars, 'tokens': tokens, 'windows': n_windows, 'chars_per_token': chars / tokens,
                       'token_loss': nats / tokens, 'perplexity': math.exp(min(nats / tokens, 30)),
                       'bpc': nats / math.log(2) / chars, 'uniform_baseline_bpc': math.log2(model.cfg.vocab_size) * tokens / chars,
                       'seconds': round(time.perf_counter() - start, 1)}
        print(path, json.dumps(files[path], ensure_ascii=False), flush=True)
    state = Path(a.run) / 'state.json'
    write_json(out, {
        'run': a.run, 'weights_sha256': sha256(Path(a.run) / 'model.safetensors'),
        'step': json.loads(state.read_text()).get('step') if state.exists() else None,
        'context_length': model.cfg.context_length, 'stride': a.stride, 'limit': a.limit, 'device': str(device),
        'files': files, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'protocol': '문서마다 BOS+토큰 열을 context_length 창으로 stride만큼 밀며 다음 토큰 NLL을 합산(겹치는 토큰은 가장 긴 문맥에서 한 번만), 비트로 바꿔 텍스트 글자 수(코드포인트)로 나눈다. uniform_baseline_bpc는 어휘 균등분포(log2 vocab × 토큰/글자)이며, 토크나이저가 바뀌면 함께 달라진다.'})
    print(out.read_text(), flush=True)


if __name__ == '__main__':
    main()

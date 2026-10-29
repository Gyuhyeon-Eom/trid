"""Per-source next-token loss on a document mix (valid.jsonl rows grouped by `kind`).

Documents are tokenized with the run tokenizer exactly like trid.corpus.prepare (BOS + text + EOS),
each document is chunked independently to the context length (context resets per chunk), and every
target position is counted exactly once. Only the first --limit-docs-per-source rows of each kind
are used, in file order, so results are deterministic and comparable across runs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from trid.corpus import sha256, write_json
from trid.runtime import load_run, sync
from trid.text import SPECIAL

FAMILY = {"law": "law", "ordin": "law", "admrul": "law", "press_release": "briefing", "wikipedia": "wiki"}


def select_documents(path, limit):
    picked = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            kind, text = row.get("kind"), row.get("text")
            if not isinstance(kind, str) or not isinstance(text, str) or not text.strip() or any(s in text for s in SPECIAL):
                raise ValueError(f"잘못된 문서: {path}:{n}")
            if len(picked[kind]) < limit:
                picked[kind].append(text)
    if not picked:
        raise ValueError("문서가 없습니다.")
    return dict(picked)


def chunks(ids, context):
    for pos in range(0, len(ids) - 1, context):
        window = ids[pos:pos + context + 1]
        yield window[:-1], window[1:]


def evaluate(model, tok, device, texts, context, batch_size):
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    pending, total, count, doc_tokens = [], 0.0, 0, 0

    def flush():
        nonlocal total, count
        x = torch.zeros((len(pending), context), dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, (xs, ys) in enumerate(pending):
            x[i, :len(xs)] = torch.tensor(xs)
            y[i, :len(ys)] = torch.tensor(ys)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum").item()
        count += int((y != -100).sum().item())
        pending.clear()

    with torch.inference_mode():
        for text in texts:
            ids = [bos] + tok.encode(text).ids + [eos]
            doc_tokens += len(ids)
            for xs, ys in chunks(ids, context):
                pending.append((xs, ys))
                if len(pending) == batch_size:
                    flush()
        if pending:
            flush()
    assert count == doc_tokens - len(texts)
    return total, count


def summary(total, count, docs):
    loss = total / count
    return {"docs": docs, "target_tokens": count, "loss": loss, "perplexity": math.exp(min(loss, 30))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data/mix-v3-dry")
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit-docs-per-source", type=int, default=200)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("Use a new report path.")
    if a.batch_size < 1 or a.limit_docs_per_source < 1 or a.context < 2:
        raise ValueError("batch size, document limit, and context must be positive.")
    source = Path(a.data) / f"{a.split}.jsonl"
    torch.set_num_threads(a.threads)
    weight_hash = sha256(Path(a.run) / "model.safetensors")
    model, tok, device = load_run(a.run, a.device)
    if a.context > model.cfg.context_length:
        raise ValueError(f"context {a.context} exceeds model context {model.cfg.context_length}.")
    picked = select_documents(source, a.limit_docs_per_source)
    sources, families, grand = {}, defaultdict(lambda: [0.0, 0, 0]), [0.0, 0, 0]
    sync(device)
    start = time.perf_counter()
    for kind in sorted(picked):
        texts = picked[kind]
        t0 = time.perf_counter()
        total, count = evaluate(model, tok, device, texts, a.context, a.batch_size)
        sync(device)
        sources[kind] = {**summary(total, count, len(texts)), "family": FAMILY.get(kind, kind), "seconds": round(time.perf_counter() - t0, 1)}
        for acc in (families[FAMILY.get(kind, kind)], grand):
            acc[0] += total
            acc[1] += count
            acc[2] += len(texts)
        print(f"{kind:14s} docs={len(texts):4d} tokens={count:8d} loss={total / count:.4f} ppl={math.exp(total / count):.2f}", flush=True)
    report = {"run": a.run, "weights_sha256": weight_hash, "data": str(source), "data_sha256": sha256(source),
              "device": device, "context_length": a.context, "limit_docs_per_source": a.limit_docs_per_source,
              "sources": sources,
              "families": {k: summary(*v) for k, v in sorted(families.items())},
              "overall": summary(*grand),
              "seconds": round(time.perf_counter() - start, 1),
              "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "protocol": "First N documents per kind in file order; BOS+text+EOS tokenized with the run tokenizer; each document chunked independently to context_length; every target position counted once; loss = sum CE / target tokens."}
    write_json(out, report)
    print(json.dumps({"families": report["families"], "overall": report["overall"], "seconds": report["seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

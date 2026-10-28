"""Held-out RAG evaluation (data/korean-rag-eval-v1) of an extractive head run (trid.extractive).

Same report shape and scoring as scripts/evaluate_rag_eval.py (its `score` is imported): strict / value /
lenient_value / source / abstained, broken down by origin, task and answerability. The answer is the decoded
passage span plus the source id of the passage it lies in, or the abstain phrase chosen by the presence head.
Items whose prompt exceeds the model context are counted as failures under `skipped_context`.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rag_eval import score  # noqa: E402
from trid.extractive import load_reader, encode, decode, batch, WEIGHTS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    p.add_argument('--data', default='data/korean-rag-eval-v1'); p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args(); out = Path(a.output)
    if out.exists():
        raise ValueError('Use a new output file')
    root = Path(a.data); test = root / 'test.jsonl'; prov = json.loads((root / 'provenance.json').read_text())
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    if digest != prov['hashes']['test']:
        raise ValueError('Evaluation data changed')
    weights = Path(a.run) / WEIGHTS; model_digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    torch.set_num_threads(4); model, tok, device = load_reader(a.run, a.device)
    context = model.backbone.cfg.context_length; pad = tok.token_to_id('<|pad|>')
    rows = [json.loads(l) for l in test.read_text().splitlines()]
    if a.limit:
        rows = rows[:a.limit]
    results = []; agg = defaultdict(Counter)
    with torch.inference_mode():
        for row in rows:
            try:
                enc = encode(tok, row['prompt'], context=context)
            except ValueError:
                enc = None
            if enc is None:
                s = {'strict': False, 'value': False, 'lenient_value': False, 'source': False, 'abstained': False}
                generated = None; skip = True; extra = {}
            else:
                kind, scores = model(*batch([enc], pad, device))
                out_row = decode((kind[0], scores[0]), enc)
                generated = out_row['answer']; s = score(row, generated); skip = False
                extra = {k: out_row[k] for k in ('presence_label', 'presence_probs', 'span_text', 'source', 'span_score')}
            for key in ('all', row['origin'], row['task'], 'answerable' if row['gold'] else 'unanswerable'):
                c = agg[key]; c['total'] += 1; c['skipped_context'] += skip
                for m, v in s.items():
                    c[m] += v
            results.append({'id': row['id'], 'task': row['task'], 'origin': row['origin'], 'answer': row['answer'],
                            'generated': generated, **s, 'skipped_context': skip, **extra})
    if hashlib.sha256(weights.read_bytes()).hexdigest() != model_digest:
        raise ValueError('Model changed')
    summary = {k: dict(v) for k, v in sorted(agg.items())}
    report = {'run': a.run, 'model_sha256': model_digest, 'weights_file': WEIGHTS, 'data': a.data, 'test_sha256': digest,
              'evaluated_at_unix': time.time(), 'model_context_length': context, 'limit': a.limit or None,
              'decoding': 'extractive', 'span_norm': None, 'protocol': __doc__.strip(), 'summary': summary, 'results': results}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: {m: v.get(m, 0) for m in ('strict', 'value', 'source', 'abstained', 'skipped_context', 'total')}
                      for k, v in summary.items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

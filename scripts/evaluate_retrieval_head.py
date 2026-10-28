"""Evaluate a trained retrieval head (trid.retrieval_head) with the protocol of scripts/benchmark_retrieval.py.

Same corpus (every distinct evidence passage of data/korean-rag-eval-v1 test.jsonl), same queries (the '질문: ' line
of every answerable item, relevant = texts of its `sources`), same metrics (Recall@1/3/5, MRR, CPU latency).
Retrievers: head (MaxSim over pre-encoded fp16 passages), bm25_kiwi (bm25s + kiwipiepy, imported from the benchmark)
and hybrid (reciprocal rank fusion of both, k=60). Writes reports/retrieval-head-<name>.json via --output.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_retrieval import kiwi_tokenizer, bm25_retriever, rrf, metrics, KS, RRF_K, KIWI_TAGS
from trid.corpus import sha256, write_json
from trid.prompts import rag_passages
from trid.retrieval_head import load_head, encode_passages, search, QUERY_TOKENS, PASSAGE_TOKENS, WEIGHTS


def load_eval(root):
    test = root / 'test.jsonl'; prov = json.loads((root / 'provenance.json').read_text())
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    if digest != prov['hashes']['test']:
        raise ValueError('Evaluation data changed')
    rows = [json.loads(l) for l in test.read_text().splitlines()]
    doc_id, docs = {}, []
    for row in rows:
        for _, text in rag_passages(row['prompt']):
            if text not in doc_id:
                doc_id[text] = len(docs); docs.append(text)
    queries = []
    for row in rows:
        if not row['gold']:
            continue
        by_src = dict(rag_passages(row['prompt'])); q = row['prompt'].rsplit('\n질문: ', 1)[1].strip()
        queries.append((row, q, {doc_id[by_src[s]] for s in row['sources']}))
    return digest, docs, queries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='directory written by scripts/train_retrieval_head.py')
    p.add_argument('--output', required=True)
    p.add_argument('--data', default='data/korean-rag-eval-v1')
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); out = Path(a.output)
    if out.exists():
        p.error('Use a new output file')
    torch.set_num_threads(a.threads)
    digest, docs, queries = load_eval(Path(a.data))
    model, tok, device = load_head(a.run, a.device)
    meta = json.loads((Path(a.run) / 'training.json').read_text())
    query_tokens, passage_tokens = meta.get('query_tokens', QUERY_TOKENS), meta.get('passage_tokens', PASSAGE_TOKENS)
    kmax = max(KS); index_time = {}
    t = time.perf_counter(); bm25 = bm25_retriever(docs, kiwi_tokenizer()); index_time['bm25_kiwi'] = time.perf_counter() - t
    t = time.perf_counter(); index = encode_passages(model, tok, docs, device, a.batch, passage_tokens); index_time['head'] = time.perf_counter() - t
    truncated = sum(len(tok.encode('문단: ' + d).ids) > passage_tokens - 1 for d in docs)
    names = ('bm25_kiwi', 'head', 'hybrid')
    per_query = []; agg = {n: defaultdict(Counter) for n in names}; lat = {n: [] for n in names}
    for row, q, rel in queries:
        ranked = {}; rec = {'id': row['id'], 'origin': row['origin'], 'task': row['task'], 'relevant': sorted(rel), 'top': {}}
        t = time.perf_counter(); ranked['bm25_kiwi'] = bm25(q, kmax); lat['bm25_kiwi'].append(time.perf_counter() - t)
        t = time.perf_counter(); ranked['head'] = [i for i, _ in search(model, tok, q, index, kmax, device)]; lat['head'].append(time.perf_counter() - t)
        t = time.perf_counter(); ranked['hybrid'] = rrf([ranked['bm25_kiwi'], ranked['head']], kmax)
        lat['hybrid'].append(time.perf_counter() - t + lat['bm25_kiwi'][-1] + lat['head'][-1])
        for n in names:
            m = metrics(ranked[n], rel); rec['top'][n] = ranked[n][:kmax]
            for key in ('all', row['origin'], row['task']):
                c = agg[n][key]; c['total'] += 1
                for k, v in m.items():
                    c[k] += v
        per_query.append(rec)
    summary = {n: {key: {**{m: round(c[m] / c['total'], 4) for m in c if m != 'total'}, 'total': c['total']}
                   for key, c in sorted(g.items())} for n, g in agg.items()}
    latency = {n: {'mean_ms': round(1e3 * float(np.mean(v)), 2), 'p50_ms': round(1e3 * float(np.median(v)), 2),
                   'p90_ms': round(1e3 * float(np.percentile(v, 90)), 2)} for n, v in lat.items()}
    report = {'data': a.data, 'test_sha256': digest, 'evaluated_at_unix': time.time(), 'device': device, 'threads': a.threads,
              'protocol': __doc__.strip(), 'run': a.run, 'model_sha256': sha256(Path(a.run) / WEIGHTS),
              'training': {k: meta.get(k) for k in ('stage', 'initialization', 'initial_weights_sha256', 'freeze_backbone', 'steps_requested',
                                                    'completed_steps', 'best_step', 'best_validation', 'batch_size', 'train_pairs',
                                                    'valid_pairs', 'train_limit', 'pair_counts', 'hard_negatives', 'status')},
              'query_tokens': query_tokens, 'passage_tokens': passage_tokens, 'passages_truncated': truncated,
              'index_bytes_fp16': index.nbytes, 'kiwi_tags': sorted(KIWI_TAGS), 'rrf_k': RRF_K,
              'licences': {'bm25s': 'MIT https://github.com/xhluca/bm25s', 'kiwipiepy': 'Apache-2.0 https://github.com/bab2min/kiwipiepy'},
              'corpus_passages': len(docs), 'queries': len(queries), 'index_seconds': {k: round(v, 2) for k, v in index_time.items()},
              'latency_per_query': latency, 'summary': summary, 'per_query': per_query}
    write_json(out, report)
    for n in names:
        s = summary[n]['all']; print(n, {m: s[m] for m in ('recall@1', 'recall@3', 'recall@5', 'mrr')}, latency[n])


if __name__ == '__main__':
    main()

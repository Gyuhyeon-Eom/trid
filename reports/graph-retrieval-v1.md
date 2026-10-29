# Production actual-law graph retrieval: frozen development measurement

Production Index.build/search over 3000 documents and 109380 passages. Frozen 80 auto-derived multi-evidence paths (40 one-edge, 40 two-edge) and 20 controls; not human gold.

| Method | Multi all-required | Multi answer-passage | Control answer | Mean candidates | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25@3 | 0.0125 | 0.0875 | 0.6500 | 3.00 | 41.82 | 61.11 |
| bm25@12 | 0.0625 | 0.2250 | 0.9500 | 12.00 | 41.08 | 61.98 |
| graph-seeds3-hops1-budget12 | 0.1375 | 0.2500 | 0.6500 | 4.34 | 41.47 | 59.05 |
| graph-seeds3-hops2-budget12 | 0.1625 | 0.3125 | 0.6500 | 4.70 | 41.94 | 62.24 |
| bm25@5 | 0.0125 | 0.1500 | 0.7000 | 5.00 | 41.41 | 61.76 |
| bm25@20 | 0.1750 | 0.3375 | 0.9500 | 20.00 | 41.17 | 62.46 |
| graph-seeds5-hops1-budget20 | 0.1750 | 0.3750 | 0.7000 | 7.15 | 40.56 | 60.62 |
| graph-seeds5-hops2-budget20 | 0.2250 | 0.4625 | 0.7000 | 7.92 | 41.32 | 62.80 |

Compare BM25@12/20 with graph k3/5 and final caps12/20; also report BM25@3/5. Production API has per_seed=3 but no global cap: truncate its ordered results afterward. Full expansion work is charged to graph latency. Report capped and uncapped candidate counts. Graph can underfill.

Wall time of each independent production Index.search call plus final slicing, including Kiwi query tokenization, ACL sets/masks, scoring, ranking, graph expansion. Warmup once per method; rotate method order by case. Build excluded; no shared scoring timings.

Production tokenize_all / Kiwi KIWI_TAGS, Kiwi num_workers=4; bm25s BM25 defaults

Actual Index.search: article heads, directed local and cross-document references, ACL filtering, per_seed=3, hops=1/2

Index build: 64.930 seconds. Local edges: 55026; external edges: 8390. Public ACL used; empty-tag search returned 0.

## Frozen inputs

No fixtures were regenerated or queries tuned. All three fixture hashes were checked unchanged after measurement.
- `cases.jsonl`: `b370ef1ed852219ce36f5edb29e1e6924e4daa799f038f67873a0a8a1e6f71e5`
- `documents.jsonl`: `801dd26c2d9917a9703cbe031147e9b7a7560f5dcbb74572c4998bae87a61b92`
- `provenance.json`: `5e03c32dfe28932a8127cf171c0b146f0aed928be4ede8f9beb6c1149c75a023`

## Limitations
- Frozen auto-derived parser-reference paths, not human question/evidence gold; graph-dependent selection bias.
- Development corpus was used for parser fixes; no heldout/generalization or legal-answer accuracy claim.
- Queries contain source titles/headings. Terminal reference passage is answer-passage proxy, not human minimal answer span.
- First-available balanced kinds, not random representative sampling. Preparation limited case documents to 30000 characters.
- All documents tagged public. Actual ACL filtering and deny-by-default smoke exercised, not a restricted-tenant security evaluation.
- Snapshot includes only some external targets; production title matching may be ambiguous. No independent external-edge correctness adjudication.
- Single-process CPU measurement while other work may run; no production HTTP/concurrency measurement.

## Reproduce
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/validate_structure.py --manifest data/korean-rag-eval-multihop-v1-source/provenance.json --output /tmp/trid-structure-after.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/benchmark_graph_retrieval.py
```
Snapshot restoration requires the original hash-matching source rows. Benchmark fails on snapshot/evidence drift. `prepare_multihop_eval.py` is explicit original preparation, NOT invoked by measurement and must not be run to change frozen v1 cases.

Supersedes the earlier regex-tokenizer/local-edge exploratory measurement. No test suite was run for this analysis.

"""Production Index.build/search measurement on frozen, auto-derived development cases.

No fixture generation or tuning. Uses production Kiwi/BM25, ACL, head and local/
external reference expansion. Final result caps are applied after Index.search;
latency includes the entire untruncated production search call.
"""
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.service.index import Index
from trid.service.structure import parse_document
from trid.service.types import Document
from validate_structure import digest


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, default=Path('/tmp/trid-law-structure-3000.jsonl'))
    ap.add_argument('--fixtures', type=Path, default=Path('data/korean-rag-eval-multihop-v1-source'))
    ap.add_argument('--report', type=Path, default=Path('reports/graph-retrieval-v1.json'))
    args = ap.parse_args()
    fixture_hashes = {name:file_hash(args.fixtures/name) for name in ('cases.jsonl','documents.jsonl','provenance.json')}
    manifest = json.loads((args.fixtures/'provenance.json').read_text())
    if file_hash(args.snapshot) != manifest['snapshot_sha256']:
        raise ValueError('Snapshot hash does not match frozen development provenance')
    cases = [json.loads(line) for line in (args.fixtures/'cases.jsonl').open()]
    rows = [json.loads(line) for line in args.snapshot.open()]
    source = {r['id']:r for r in rows}
    documents = []
    for row in rows:
        passages, edges = parse_document(row['text'],row['id'],row['title'])
        documents.append(Document(row['id'],row['title'],digest(row['text']),{'public'},row['text'],passages,edges))
    passages = {p.id:p for d in documents for p in d.passages}
    for case in cases:
        for evidence in case['evidence']:
            p = passages[evidence['passage_id']]
            row = source[evidence['doc_id']]
            if (row['text'][evidence['char_start']:evidence['char_end']] != evidence['text']
                    or digest(evidence['text']) != evidence['sha256']
                    or digest(row['text']) != evidence['document_sha256']
                    or p.text != evidence['text'] or list(p.path) != evidence['path']):
                raise ValueError(f"Frozen evidence drift: {case['id']} {p.id}")
    print('production Index.build:',len(documents),'documents,',len(passages),'passages;',len(cases),'frozen cases',flush=True)
    start = time.perf_counter()
    index = Index.build(documents)
    index_seconds = time.perf_counter()-start
    print('production index built in',round(index_seconds,3),'seconds',flush=True)
    local_edges = {edge for d in documents for edge in d.refs}
    cross_edges = set(index.edges)-local_edges
    # All legal source documents are public: exercise the actual ACL filter without
    # inventing restricted documents or changing which frozen evidence is accessible.
    tags = {'public'}
    denied_hits = len(index.search(cases[0]['query'],set(),k=5,hops=2))
    if denied_hits:
        raise ValueError('Production deny-by-default ACL returned results')
    methods = [('bm25@3',3,0,3),('bm25@12',12,0,12),
               ('graph-seeds3-hops1-budget12',3,1,12),('graph-seeds3-hops2-budget12',3,2,12),
               ('bm25@5',5,0,5),('bm25@20',20,0,20),
               ('graph-seeds5-hops1-budget20',5,1,20),('graph-seeds5-hops2-budget20',5,2,20)]
    results = collections.defaultdict(list)
    details = []
    # Warm all methods once on the same existing case; no scored/tuned query changes.
    for _,k,hops,_ in methods:
        index.search(cases[0]['query'],tags,k=k,hops=hops,per_seed=3)
    for case_number, case in enumerate(cases):
        # Rotate method order to reduce systematic cache/order advantage.
        offset = case_number % len(methods)
        for name,k,hops,cap in methods[offset:]+methods[:offset]:
            start = time.perf_counter()
            raw = index.search(case['query'],tags,k=k,hops=hops,per_seed=3)
            hits = raw[:cap]
            elapsed = (time.perf_counter()-start)*1000
            ids = [hit.passage.id for hit in hits]
            metric = dict(case=case['id'],hops=case['hops'],
                          all_required=int(set(case['required_passage_ids']) <= set(ids)),
                          answer_passage=int(case['answer_passage_id'] in ids),
                          returned=len(hits),uncapped_returned=len(raw),latency_ms=elapsed,
                          head_hits=sum(bool(h.via and h.via.startswith('head')) for h in hits),
                          reference_hits=sum(bool(h.via and h.via.startswith('ref')) for h in hits))
            results[name].append(metric)
            details.append(dict(method=name,**metric,retrieved=ids,via=[h.via for h in hits]))
    summary = {}
    for name,_,_,_ in methods:
        summary[name] = {}
        for group,predicate in [('all',lambda v:True),('multihop',lambda v:v['hops']>0),('singlehop_control',lambda v:v['hops']==0)]:
            values = [v for v in results[name] if predicate(v)]
            summary[name][group] = dict(n=len(values),
                all_required_recall=float(np.mean([v['all_required'] for v in values])),
                answer_passage_recall=float(np.mean([v['answer_passage'] for v in values])),
                mean_returned=float(np.mean([v['returned'] for v in values])),
                mean_uncapped_returned=float(np.mean([v['uncapped_returned'] for v in values])),
                mean_head_hits=float(np.mean([v['head_hits'] for v in values])),
                mean_reference_hits=float(np.mean([v['reference_hits'] for v in values])),
                latency_median_ms=float(np.median([v['latency_ms'] for v in values])),
                latency_p95_ms=float(np.percentile([v['latency_ms'] for v in values],95)))
    if fixture_hashes != {name:file_hash(args.fixtures/name) for name in fixture_hashes}:
        raise ValueError('Frozen fixture files changed during measurement')
    report = dict(implementation='Production trid.service.index.Index.build and Index.search',
        corpus_documents=len(documents),corpus_passages=len(passages),local_reference_edges=len(local_edges),
        external_reference_edges=len(cross_edges),cases=len(cases),fixture_sha256=fixture_hashes,
        snapshot_sha256=manifest['snapshot_sha256'],fixtures_unchanged=True,
        implementation_sha256={name:file_hash(Path(name)) for name in ('trid/service/index.py','trid/service/structure.py','trid/korean_quality.py')},
        case_hops=dict(collections.Counter(c['hops'] for c in cases)),case_kinds=dict(collections.Counter(c['kind'] for c in cases)),
        index_seconds=index_seconds,tokenizer='Production tokenize_all / Kiwi KIWI_TAGS, Kiwi num_workers=4; bm25s BM25 defaults',
        graph='Actual Index.search: article heads, directed local and cross-document references, ACL filtering, per_seed=3, hops=1/2',
        acl=dict(document_tags=['public'],query_tags=['public'],empty_tags_returned=denied_hits),
        budget='Compare BM25@12/20 with graph k3/5 and final caps12/20; also report BM25@3/5. Production API has per_seed=3 but no global cap: truncate its ordered results afterward. Full expansion work is charged to graph latency. Report capped and uncapped candidate counts. Graph can underfill.',
        latency='Wall time of each independent production Index.search call plus final slicing, including Kiwi query tokenization, ACL sets/masks, scoring, ranking, graph expansion. Warmup once per method; rotate method order by case. Build excluded; no shared scoring timings.',
        limitations=['Frozen auto-derived parser-reference paths, not human question/evidence gold; graph-dependent selection bias.',
                     'Development corpus was used for parser fixes; no heldout/generalization or legal-answer accuracy claim.',
                     'Queries contain source titles/headings. Terminal reference passage is answer-passage proxy, not human minimal answer span.',
                     'First-available balanced kinds, not random representative sampling. Preparation limited case documents to 30000 characters.',
                     'All documents tagged public. Actual ACL filtering and deny-by-default smoke exercised, not a restricted-tenant security evaluation.',
                     'Snapshot includes only some external targets; production title matching may be ambiguous. No independent external-edge correctness adjudication.',
                     'Single-process CPU measurement while other work may run; no production HTTP/concurrency measurement.'],
        summary=summary,details=details)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# Production actual-law graph retrieval: frozen development measurement','',
           f"Production Index.build/search over {len(documents)} documents and {len(passages)} passages. Frozen 80 auto-derived multi-evidence paths (40 one-edge, 40 two-edge) and 20 controls; not human gold.",'',
           '| Method | Multi all-required | Multi answer-passage | Control answer | Mean candidates | p50 ms | p95 ms |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        a=s['all']; m=s['multihop']; c=s['singlehop_control']
        lines.append(f"| {name} | {m['all_required_recall']:.4f} | {m['answer_passage_recall']:.4f} | {c['answer_passage_recall']:.4f} | {a['mean_returned']:.2f} | {a['latency_median_ms']:.2f} | {a['latency_p95_ms']:.2f} |")
    lines += ['',report['budget'],'',report['latency'],'',report['tokenizer'],'',report['graph'],'',
              f'Index build: {index_seconds:.3f} seconds. Local edges: {len(local_edges)}; external edges: {len(cross_edges)}. Public ACL used; empty-tag search returned {denied_hits}.',
              '', '## Frozen inputs', '', 'No fixtures were regenerated or queries tuned. All three fixture hashes were checked unchanged after measurement.']
    lines += [f'- `{name}`: `{value}`' for name,value in fixture_hashes.items()]
    lines += ['', '## Limitations']+['- '+x for x in report['limitations']]
    lines += ['', '## Reproduce', '```sh',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/validate_structure.py --manifest data/korean-rag-eval-multihop-v1-source/provenance.json --output /tmp/trid-structure-after.json',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/benchmark_graph_retrieval.py',
              '```', 'Snapshot restoration requires the original hash-matching source rows. Benchmark fails on snapshot/evidence drift. `prepare_multihop_eval.py` is explicit original preparation, NOT invoked by measurement and must not be run to change frozen v1 cases.',
              '', 'Supersedes the earlier regex-tokenizer/local-edge exploratory measurement. No test suite was run for this analysis.']
    args.report.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','summary')},ensure_ascii=False),flush=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

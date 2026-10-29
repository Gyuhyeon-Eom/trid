"""Explicit preparation only: overwrites auto-derived development fixtures.
Do not run when measuring the frozen v1 cases. No retrieval evaluation here.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.service.structure import parse_document, H_JO
from validate_structure import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, default=Path('/tmp/trid-law-structure-3000.jsonl'))
    ap.add_argument('--fixtures', type=Path, default=Path('data/korean-rag-eval-multihop-v1-source'))
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.snapshot.open()]
    passages, refs = [], collections.defaultdict(list)
    by_doc = {}
    for row in rows:
        ps, edges = parse_document(row['text'], row['id'], row['title'])
        by_doc[row['id']] = ps
        passages.extend(ps)
        for src, dst in edges:
            refs[src].append(dst)
    by_id = {p.id:p for p in passages}
    source = {r['id']:r for r in rows}
    cases, used = [], set()
    # Interleave kinds; at most one case per source document per case family.
    pools = {kind:[r for r in rows if r['kind']==kind] for kind in ('law','admrul','ordin')}
    ordered = [r for group in zip(*pools.values()) for r in group]
    for hops, quota in ((2,40),(1,40),(0,20)):
        count = 0
        for row in ordered:
            if len(row['text']) > 30000:
                continue
            for p in by_doc[row['id']]:
                if (p.id,hops) in used or not any(H_JO.match(x) for x in p.path[1:]) or len(p.text)<70:
                    continue
                paths = [[p.id]]
                for _ in range(hops):
                    paths = [path+[dst] for path in paths for dst in refs[path[-1]] if dst not in path]
                if not paths:
                    continue
                path = paths[0]
                used.add((p.id,hops))
                heading = re.search(r'제\s*\d+조(?:의\d+)?\(([^)]+)\)', p.text)
                topic = heading[1] if heading else p.text[:45]
                query = f"{row['title']} {topic}에 관한 규정" + (' 및 인용 조항의 요건을 함께 확인하시오.' if hops else '의 내용을 확인하시오.')
                evidence = []
                for pid in path:
                    q = by_id[pid]
                    evidence.append(dict(passage_id=pid, doc_id=q.doc_id, path=list(q.path), char_start=q.char_start, char_end=q.char_end, text=q.text, sha256=digest(q.text), source=row['source'], document_sha256=digest(row['text'])))
                cases.append(dict(id=f'dev-{len(cases)+1:03d}', split='development', annotation='auto-derived parser-reference path; not human gold', kind=row['kind'], query=query, hops=hops, gold_path=path, required_passage_ids=path, answer_passage_id=path[-1], evidence=evidence))
                count += 1
                break
            if count == quota:
                break
        if count != quota:
            raise RuntimeError(f'Only {count}/{quota} cases for hops={hops}')
    args.fixtures.mkdir(parents=True, exist_ok=True)
    selected = {e['doc_id'] for c in cases for e in c['evidence']}
    def jsonl(path, objects):
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in objects))
    jsonl(args.fixtures/'cases.jsonl', cases)
    jsonl(args.fixtures/'documents.jsonl', [r for r in rows if r['id'] in selected])
    manifest = dict(snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(), snapshot_documents=len(rows), snapshot_selection='first 1000 unique complete rows per kind before initial byte boundary', document_manifest=[dict(id=r['id'],sha256=digest(r['text']),source=r['source'],kind=r['kind']) for r in rows], annotation='Auto-derived development only; exact source spans validated, semantic necessity not human adjudicated.', cases=len(cases), selected_documents=len(selected), parser_sha256=hashlib.sha256(Path('trid/service/structure.py').read_bytes()).hexdigest())
    (args.fixtures/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("Prepared", len(cases), "auto-derived development cases")


if __name__ == "__main__":
    main()

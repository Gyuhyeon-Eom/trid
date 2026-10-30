"""Carry review state from older guide versions to the fixed consumer versions by id + content hash (B-019 for C-020).

A review made on an older version stays usable only when the same id exists in the new version, its text hash is
unchanged and the new row sits in a candidate part. Everything else is reported so a stale review cannot bring back
a row the new version holds:
  carry            same id, same sha256, candidate in the new version
  content_changed  same id, different sha256 (e.g. HTML entities decoded in guides-v5); review again
  now_held         row is in the new version's held part (reserved overlap, source note, empty answer, ...)
  missing          id not in the new version
Inputs: review files (JSONL with id + source_sha256) and case files (JSON with cases[].source); the new versions are
the exclusive splits (guides-v6, bojo-faq-v3) whose consumer check must pass first.

  .venv/bin/python scripts/migrate_review_state.py --new guides-v6 bojo-faq-v3 \
      --reviews reports/coordination/C-020/review-faq-ai.jsonl --cases reports/coordination/C-020/faq-cases.json \
      --out reports/coordination/B-019/c020-review-migration.json
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded


def load_new(versions):
    from extract_public_guides import CANDIDATE_PARTS, OUT, check_version
    index = {}
    for v in versions:
        problems = check_version(v)
        if problems:
            raise SystemExit(f'{v} fails the consumer check: {problems[:3]}')
        for part in CANDIDATE_PARTS + ('held',):
            for x in (OUT / v / f'{part}.jsonl').read_text().split('\n'):
                if x:
                    r = json.loads(x)
                    index[r['id']] = {'version': v, 'part': part, 'sha256': r['sha256'], 'holds': r['meta'].get('status_holds', [])}
    return index


def classify(rid, sha, index):
    new = index.get(rid)
    if not new:
        return 'missing', None
    if new['part'] == 'held':
        return 'now_held', new
    if sha and sha != new['sha256']:
        return 'content_changed', new
    return 'carry', new


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--new', nargs='+', required=True)
    p.add_argument('--reviews', nargs='*', default=[])
    p.add_argument('--cases', nargs='*', default=[])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    index = load_new(args.new)
    rows, summary = [], Counter()
    for path in args.reviews:
        for x in Path(path).read_text().split('\n'):
            if x:
                rv = json.loads(x)
                state, new = classify(rv['id'], rv.get('source_sha256'), index)
                rows.append({'kind': 'review', 'file': path, 'id': rv['id'], 'old_sha256': rv.get('source_sha256'),
                             'old_verdict': rv.get('verdict'), 'state': state, 'new': new})
                summary[f'review:{state}'] += 1
    for path in args.cases:
        doc = json.loads(Path(path).read_text())
        for case in doc['cases']:
            # cases do not record the source row hash: a case carries only if its source row is a candidate now;
            # content_changed cannot be detected here, so the source hash recorded by its review is used when present
            src_sha = next((r['old_sha256'] for r in rows if r['kind'] == 'review' and r['id'] == case['source']), None)
            state, new = classify(case['source'], src_sha, index)
            rows.append({'kind': 'case', 'file': path, 'id': case['id'], 'source': case['source'], 'old_sha256': src_sha,
                         'state': state, 'new': new})
            summary[f'case:{state}'] += 1
    out = {'code_sha256': CODE_SHA256, 'new_versions': args.new, 'summary': dict(summary),
           'not_carried': [r for r in rows if r['state'] != 'carry'], 'rows': rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out['summary'], ensure_ascii=False))
    for r in out['not_carried']:
        print(r['kind'], r['id'], r['state'], (r['new'] or {}).get('holds'))


if __name__ == '__main__':
    main()

"""Curate an extracted admin-rule attachment batch into parts with an explicit status per row (local only).

Input: data/law-attachments/<source>/{candidates,quality_held,extraction_held,duplicates}.jsonl, the
check_reserved_overlap.py report over that batch, and a review-decisions JSON (row id -> decision, written by a
reviewer who read the rows; see reports/coordination/B-015/attachment-review-decisions.json).
Output parts (data/law-attachments/<out>/), each row with meta.status:
  candidate_quality_ok         extraction ok, PUBLIC_DOCS quality ok, no reserved overlap
  candidate_table_unverified   extraction ok, carries tables, PUBLIC_DOCS quality not ok (tables break the heuristic);
                               content not reviewed
  candidate_short_reviewed     under the length rule but read in full by a reviewer and found complete
  held_reserved_overlap        full or partial overlap with the 148 signed reserved families
  held_review                  reviewer decision is a hold (OCR defects, missing referenced content, low Hangul)
  held_quality / held_extraction  not reviewed, kept with their text
No row is approved for training here; A/C decide. OCR text for rows decided 'ocr_*' is attached from --ocr-dir.

  .venv/bin/python scripts/curate_admrul_attachments.py --source v4 --out v5 --decisions <json> --ocr-dir <dir>
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
BASE = ROOT / 'data/law-attachments'
HOLD_DECISIONS = {'hold_missing_referenced_content', 'ocr_hold_defects', 'ocr_failed', 'hold_low_hangul', 'hold_table_mismatch'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--decisions', required=True, nargs='+', help='review-decision JSON files; later files add to earlier ones')
    p.add_argument('--ocr-dir')
    args = p.parse_args()
    src, out = BASE / args.source, BASE / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    rep = json.loads((src / 'overlap-148.json').read_text())
    overlap = {m['corpus_id'] for m in rep.get('matches') or []} | {x['corpus_id'] for x in rep.get('partial_review') or []}
    decisions, docs = {}, []
    for path in args.decisions:
        doc = json.loads(Path(path).read_text())
        docs.append((path, doc))
        for rid, d in doc['decisions'].items():
            decisions.setdefault(rid, []).append({'reviewer': doc['reviewer'], 'at': doc['reviewed_at'], **d})
    rows = []
    for part in ('candidates', 'quality_held', 'extraction_held', 'duplicates'):
        for x in (src / f'{part}.jsonl').read_text().split('\n'):
            if x:
                r = json.loads(x)
                r['meta']['source_part'] = part
                rows.append(r)
    parts = {k: [] for k in ('candidate_quality_ok', 'candidate_table_verified', 'candidate_table_unverified',
                             'candidate_short_reviewed', 'held_reserved_overlap', 'held_review', 'held_quality',
                             'held_extraction', 'duplicates')}
    for r in rows:
        ds = decisions.get(r['id'], [])
        kinds = {d['decision'] for d in ds}
        if ds:
            r['meta']['review'] = ds
            if any(k.startswith('ocr_') for k in kinds) and args.ocr_dir:
                path = Path(args.ocr_dir) / f"{r['id'].rsplit(':', 1)[1]}.md"
                if path.exists():
                    r['meta']['ocr_text'] = path.read_text(encoding='utf-8', errors='replace')
                    r['meta']['ocr_engine'] = 'kordoc 4.24.23 --ocr (PP-OCRv5 korean)'
        checks = [d['table_check'] for d in ds if d.get('table_check')]
        # table correctness is its own axis: PUBLIC_DOCS quality says nothing about rows/columns (C-014: a quality-ok
        # file had 0/5 tables matching its source)
        r['meta']['table_check'] = checks[-1] if checks else ('unverified' if r['meta'].get('table_rows') else 'no_tables')
        restrictions = [d['usage_restriction'] for d in ds if d.get('usage_restriction')]
        if restrictions:
            r['meta']['usage_restriction'] = restrictions
        if r['id'] in overlap:
            status = 'held_reserved_overlap'
        elif kinds & HOLD_DECISIONS:
            status = 'held_review'
        elif 'short_complete' in kinds:
            status = 'candidate_short_reviewed'
        elif r['meta']['source_part'] == 'candidates':
            if 'table_verified' in kinds:
                status = 'candidate_table_verified'
            elif r['meta']['quality'] == 'ok':
                status = 'candidate_quality_ok'
            else:
                status = 'candidate_table_unverified'
        elif r['meta']['source_part'] == 'quality_held':
            status = 'held_quality'
        elif r['meta']['source_part'] == 'duplicates':
            status = 'duplicates'
        else:
            status = 'held_extraction'
        r['meta']['status'] = status
        r['clean_version'] = f"{r.get('clean_version')}+{args.out}"
        parts[status].append(r)
    out.mkdir(parents=True)
    man = {}
    for name, sel in parts.items():
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        man[name] = {'rows': len(sel), 'characters': sum(len(r['text']) for r in sel),
                     'table_rows': sum(r['meta'].get('table_rows', 0) for r in sel), 'sha256': hashlib.sha256(body).hexdigest(),
                     'table_check': dict(Counter(r['meta']['table_check'] for r in sel)),
                     'usage_restricted': sum(1 for r in sel if r['meta'].get('usage_restriction')),
                     'doc_type': dict(Counter(r['provenance'].get('doc_type') for r in sel)),
                     'topic_labels': dict(Counter(t for r in sel for t in r['meta']['topics'])),
                     'rows_without_topic': sum(not r['meta']['topics'] for r in sel),
                     'rows_with_two_topics': sum(len(r['meta']['topics']) > 1 for r in sel)}
    manifest = {'source': args.source, 'out': args.out, 'code_sha256': CODE_SHA256,
                'decisions_files': [{'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p, _ in docs],
                'overlap_report_sha256': hashlib.sha256((src / 'overlap-148.json').read_bytes()).hexdigest(),
                'parts': man, 'note': 'no row is approved for training; candidate_* are candidates only; 852 reserved families unchecked'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ('rows', 'characters', 'rows_without_topic', 'rows_with_two_topics')} for k, v in man.items()},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

"""Code / unit / date cases on B-015 public data (C-015), CPU only.

Input: reports/coordination/B-015/public-data-v1.json (B's per-dataset summary: raw CSV path, sha256, encoding, unit and
currency notes) and the case spec reports/coordination/C-015/public-data-cases.json. For every case the builder:
  - checks the raw CSV sha256 against B's summary;
  - finds each named row (all 'match' columns equal) and requires exactly one hit;
  - checks every 'expect' cell value on that row;
  - puts the CSV header and the matched rows, verbatim, in the context, plus B's summary fields that the answer
    relies on (labelled as B's transcription of the dataset page; C did not re-read the page).
These are snapshot reference data, not any institution's live database values.

usage: build_public_data_cases.py --out DIR
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'reports/coordination/B-015/public-data-v1.json'
CASES = ROOT / 'reports/coordination/C-015/public-data-cases.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reviews', type=Path, action='append', default=[])
    a = ap.parse_args()
    summary = json.loads(SUMMARY.read_text())
    reviews = {}
    for path in a.reviews:
        for rec in map(json.loads, path.resolve().open()):
            reviews.setdefault(rec['id'], []).append(dict(rec, file=str(path.resolve().relative_to(ROOT))))
    out = []
    for c in json.loads(CASES.read_text())['cases']:
        meta = summary[c['dataset']]
        raw = (ROOT / meta['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise ValueError(f"{c['id']}: raw CSV sha256 differs from B's summary")
        text = raw.decode(meta['encoding'])
        lines = text.splitlines()
        rows = list(csv.DictReader(io.StringIO(text)))
        picked = []
        for spec in c['rows']:
            hits = [i for i, r in enumerate(rows) if all(r[k] == v for k, v in spec['match'].items())]
            if len(hits) != 1:
                raise ValueError(f"{c['id']}: row {spec['match']} matched {len(hits)} rows")
            row = rows[hits[0]]
            for k, v in spec.get('expect', {}).items():
                if row[k] != v:
                    raise ValueError(f"{c['id']}: {k} is {row[k]!r}, case says {v!r}")
            picked.append(hits[0] + 1)                      # line number in the CSV (header is line 0)
        context = [f"[{meta['title']}] (data.go.kr {c['dataset']}, 원본 CSV sha256 {meta['sha256'][:16]}…)\n"
                   + '\n'.join([lines[0]] + [lines[i] for i in picked])]
        b_fields = {k: meta[k] for k in c.get('b_summary_fields', [])}
        if b_fields:
            context.append('B-015 요약(데이터셋 페이지 기재 사항을 B가 옮긴 것): '
                           + '; '.join(f'{k}={v}' for k, v in b_fields.items()))
        case = dict(id=c['id'], kind='public_data_behavior', behavior=c['behavior'], task_type='code_unit_lookup',
                    source_answer_present=c['behavior'] in ('answer', 'answer_with_condition', 'answer_with_exception'),
                    question=c['question'], context=context, target=c['target'],
                    evidence=dict(dataset=c['dataset'], csv_lines=picked, rows=c['rows'], b_summary_fields=b_fields),
                    doc=f"data-go-kr:{c['dataset']}", title=meta['title'], license=meta['license'],
                    currency=meta['currency'], missing=c.get('missing', []), text_sha256=meta['sha256'],
                    authored='question and explanation written by C (AI); cell values checked against the raw CSV',
                    synthetic_parts='question and explanation (AI-written)',
                    content_sha256=hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest(),
                    review=dict(status='unreviewed', records=[]), training_hold=None)
        matching = [r for r in reviews.get(case['id'], []) if r['content_sha256'] == case['content_sha256']]
        if matching:
            case['review'] = dict(status='ai_reviewed', records=[{k: r[k] for k in ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope', 'file') if k in r} for r in matching])
            if any(r['verdict'] != 'correct' for r in matching):
                case['training_hold'] = 'review: AI review found a defect (see review.records)'
        out.append(case)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'public-data-candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in out))
    print(json.dumps(dict(cases=len(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()

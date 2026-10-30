"""Statute behaviour cases on law.go.kr statutes in B's collection (C-018), CPU only.

Replaces the B-013 국고금 관리법 cases dropped from the pilot because that statute is a source of an evaluation set.
For each case in reports/coordination/C-018/statute-cases.json the builder:
  - reads the statute row from data/law-ko-source/docs.jsonl and its dates (공포일자, 시행일자, 공포번호, 제개정구분)
    from the original XML data/law-ko-source/raw/law/<number>.xml;
  - cuts the named article verbatim (from its heading to the next article, 절 or 장 heading) as the context;
  - checks every quoted span is verbatim in that article;
  - stops if the statute's document id appears in evaluation-side data or in a B-013 reserve-overlap list.

usage: build_statute_cases.py --out DIR [--reviews FILE ...]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_pilot_manifest import eval_side_text  # noqa: E402

DOCS = ROOT / 'data/law-ko-source/docs.jsonl'
RAW_LAW = ROOT / 'data/law-ko-source/raw/law'
CASES = ROOT / 'reports/coordination/C-018/statute-cases.json'
OVERLAPS = [ROOT / 'reports/coordination/B-013/overlap-law-admrul-expc-all-148.json',
            ROOT / 'reports/coordination/B-013/overlap-pack-documents-148.json']


def article(text, heading):
    start = text.index('\n' + heading) + 1
    nxt = re.search(r'\n제\d+조(?:의\d+)?\(|\n제\d+[절장] ', text[start + len(heading):])
    return text[start:start + len(heading) + nxt.start()] if nxt else text[start:]


def raw_dates(doc_id):
    xml = (RAW_LAW / f"{doc_id.rsplit(':', 1)[1]}.xml").read_text(errors='ignore')
    get = lambda k: (re.search(rf'<{k}>(?:<!\[CDATA\[)?([^<\]]*)', xml) or [None, None])[1]
    return {k: get(k) for k in ('공포일자', '시행일자', '공포번호', '제개정구분')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reviews', type=Path, action='append', default=[])
    a = ap.parse_args()
    spec = json.loads(CASES.read_text())['cases']
    wanted = {c['doc'] for c in spec}
    rows = {}
    with DOCS.open() as f:
        for line in f:
            if any(w in line[:200] for w in wanted):
                r = json.loads(line)
                if r['id'] in wanted:
                    rows[r['id']] = r
    eval_text = eval_side_text()
    overlap = set()
    for p in OVERLAPS:
        r = json.loads(p.read_text())
        overlap |= {m['corpus_id'] for m in r.get('matches', [])} | {m['corpus_id'] for m in r.get('partial_review', [])}
    for doc in wanted:
        if doc in eval_text or doc in overlap:
            raise SystemExit(f'{doc}: appears in evaluation-side data or a reserve-overlap list; not usable')
    reviews = {}
    for path in a.reviews:
        for rec in map(json.loads, path.resolve().open()):
            reviews.setdefault(rec['id'], []).append(dict(rec, file=str(path.resolve().relative_to(ROOT))))
    out = []
    for c in spec:
        r = rows[c['doc']]
        art = article(r['text'], c['article'])
        for span in c['evidence']:
            if span not in art:
                raise SystemExit(f"{c['id']}: evidence not verbatim in {c['article']}: {span[:40]!r}")
        dates = raw_dates(c['doc'])
        case = dict(id=c['id'], kind='pack_behavior', behavior=c['behavior'], task_type='statute_application',
                    source_answer_present=False, question=c['question'], context=[art], target=c['target'],
                    evidence=dict(spans=c['evidence']), doc=c['doc'], article=c['article'], title=r['title'],
                    effective=dates['시행일자'], date_fields=dates, source=r['source'], license=r['license'],
                    text_sha256=r.get('sha256'), missing=c.get('missing', []),
                    authored='question and explanation written by C (AI); quoted spans checked verbatim in the article',
                    synthetic_parts='question and explanation (AI-written)',
                    content_sha256=hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest(),
                    review=dict(status='unreviewed', records=[]), training_hold=None)
        matching = [x for x in reviews.get(c['id'], []) if x['content_sha256'] == case['content_sha256']]
        if matching:
            case['review'] = dict(status='ai_reviewed', records=[{k: x[k] for k in ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope', 'file') if k in x} for x in matching])
        out.append(case)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'statute-candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in out))
    print(json.dumps(dict(cases=len(out), docs=sorted(wanted)), ensure_ascii=False))


if __name__ == '__main__':
    main()

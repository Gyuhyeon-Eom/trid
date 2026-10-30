"""Short explanation / lookup / ask-back cases from official institution FAQs (B-018 guides-v6, bojo-faq-v3) (C-020).

Gates, in order, for every case in the spec (any failure stops the build):
  1. source files: only B-018's formal versions guides-v6 and bojo-faq-v3 are read (guides-v1..v4 are deprecated,
     guides-v5 is extraction source only); the sha256 of each named part equals the version's own manifest.json; the
     source row is in a named candidate part and not in that version's held.jsonl (which carries B's reserved overlaps);
  2. per-item review (C-020 review file): the row has a record whose source_sha256 matches, verdict ok or format_issue
     (source formatting only; the C-written answer does not copy it) and task_fit equal to the case's fit;
  3. exposure: the first 40 characters of the source question do not occur in evaluation-side data and no 20-character
     window (step 5, whitespace removed) of the source row occurs there; no exception;
  4. grounding: every must_keep string occurs in the target and in the source row; every number with a unit in the target
     (원, %, 일, 개월, 년, 점, 인, 불, 회, 시간, 분의 N) outside the label lines occurs in the source row (whitespace
     removed, HTML entities decoded);
  5. labelling: the target has a '범위:' line (institution scope) and a '시점:' line, and never says '현행'; the case date
     equals the row's own date (B provenance date, else 수정일/등록일, else 자료발간연도) and is written in the '시점:'
     line, and an edition named there in 「」 is the row's 출처 field; ask-back targets ask for the missing fact
     ('알려 주세요' or '확인이 필요').
The context given to the model is the source question and answer verbatim (source_answer_present=True); the source's
typos and spliced headings are left as they are (B cleans). Different task from copying the official answer verbatim.

AI-written questions and answers; review is AI self-review only; not production approval; values are as of the FAQ
edition or row date and current values are unverified.

usage: build_faq_cases.py --spec reports/coordination/C-020/faq-cases.json --review reports/coordination/C-020/review-faq-ai.jsonl
                          --out DIR [--reviews F ...]
"""
import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_pilot_manifest import eval_side_text  # noqa: E402

PUB = ROOT / 'data/public-data'
VERSIONS = ('guides-v6', 'bojo-faq-v3')
FIT = {'faq_explanation': 'short_explanation', 'faq_lookup': 'lookup', 'faq_ask_back': 'ask_back'}
NUM = re.compile(r'\d[\d,.~]*\s*(?:만원|천만원|억원|백만원|만\s*원|원|%|일|개월|년|점|인|불|회|시간)|100분의\s*\d+')
squash = lambda s: re.sub(r'\s+', '', html.unescape(s or ''))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(spec):
    """{row id: row} of the candidate parts named in the spec and the held ids of their versions; part hashes must equal
    the version manifest."""
    rows, held = {}, set()
    for src in spec['sources']:
        if src['version'] not in VERSIONS:
            raise SystemExit(f"{src['version']}: not a B-018 formal version ({', '.join(VERSIONS)})")
        manifest = json.loads((PUB / src['version'] / 'manifest.json').read_text())
        for part in src['parts']:
            path = PUB / src['version'] / f'{part}.jsonl'
            if sha(path) != manifest['parts'][part]['sha256']:
                raise SystemExit(f"{src['version']}/{part}: sha256 differs from the version manifest")
            rows.update((r['id'], r) for r in map(json.loads, path.open()))
        held |= {json.loads(l)['id'] for l in (PUB / src['version'] / 'held.jsonl').open()}
    return rows, held


def row_date(row):
    f = row['meta'].get('fields', {})
    return row['provenance'].get('date') or f.get('수정일') or f.get('등록일') or f.get('자료발간연도')


def qa(text):
    q, _, a = text.partition('\n답변:')
    return q.replace('질문:', '', 1).strip(), a.strip()


def check(c, row, review, ev, windows):
    cid, fit = c['id'], FIT[c['task_type']]
    if row is None:
        raise ValueError(f'{cid}: source {c["source"]} is not in a B candidate part')
    if review is None or review['source_sha256'] != row['sha256']:
        raise ValueError(f'{cid}: no C-020 review record for this source text')
    if review['verdict'] not in ('ok', 'format_issue') or review['task_fit'] != fit:
        raise ValueError(f"{cid}: review verdict {review['verdict']} / task_fit {review['task_fit']} does not allow {fit}")
    q, _ = qa(row['text'])
    t = squash(row['text'])
    hits = sum(t[i:i + 20] in windows for i in range(0, max(1, len(t) - 19), 5))
    if q[:40] in ev or hits:
        raise ValueError(f'{cid}: evaluation exposure (question prefix or {hits} shared 20-char windows)')
    tgt = squash(c['target'])
    for k in c['must_keep']:
        if squash(k) not in tgt or squash(k) not in t:
            raise ValueError(f'{cid}: must_keep {k!r} not in both target and source')
    lines = c['target'].split('\n')
    body = '\n'.join(l for l in lines if not l.startswith(('범위:', '시점:')))
    for m in NUM.finditer(body):
        if squash(m.group(0)).replace(',', '') not in t.replace(',', ''):
            raise ValueError(f'{cid}: number {m.group(0)!r} in the target is not in the source row')
    when = [l for l in lines if l.startswith('시점:')]
    if not any(l.startswith('범위:') for l in lines) or len(when) != 1 or '현행' in c['target']:
        raise ValueError(f'{cid}: target needs one 범위: line and one 시점: line and must not say 현행')
    if c['date'] != row_date(row) or c['date'] not in when[0]:
        raise ValueError(f"{cid}: date {c['date']} is not the row date {row_date(row)} or not in the 시점: line")
    edition = re.search(r'「([^」]+)」', when[0])
    if edition and squash(edition.group(1)) not in squash(row['meta'].get('fields', {}).get('출처', '')):
        raise ValueError(f'{cid}: edition {edition.group(1)!r} is not the row 출처 field')
    if fit == 'ask_back' and not ('알려 주세요' in c['target'] or '확인이 필요' in c['target']):
        raise ValueError(f'{cid}: ask-back target does not ask for the missing fact')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True, help='C-020 per-item FAQ review file')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reviews', type=Path, action='append', default=[], help='case review records')
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    rows, held = load_sources(spec)
    item_reviews = {r['id']: r for r in map(json.loads, a.review.open())}
    ev = eval_side_text()
    ev_sq = squash(ev)
    windows = {ev_sq[i:i + 20] for i in range(len(ev_sq) - 19)}

    cases = []
    for c in spec['cases']:
        row = None if c['source'] in held else rows.get(c['source'])
        try:
            check(c, row, item_reviews.get(c['source']), ev, windows)
        except ValueError as e:
            raise SystemExit(str(e))
        prov = row['provenance']
        cases.append(dict(
            id=c['id'], kind='faq_behavior', behavior=c['behavior'], task_type=c['task_type'],
            source_answer_present=True, question=c['question'],
            context=[f"[{prov['institution']} FAQ · {row['id']} · {c['date']}]\n{row['text']}"],
            target=c['target'], evidence=dict(must_keep=c['must_keep']),
            doc=row['id'], title=row['title'], source=row['source'], license=row['license'], date=c['date'],
            date_field=c['date_field'], text_sha256=row['sha256'], institution=prov['institution'],
            missing=c.get('missing', []),
            authored='question and explanation written by C (AI); kept items, numbers and labels checked against the FAQ row',
            synthetic_parts='question and explanation (AI-written)',
            content_sha256=hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest(),
            review=dict(status='unreviewed', records=[]), training_hold=None, split_hint=None))

    reviews = {}
    for p in a.reviews:
        for rec in map(json.loads, p.resolve().open()):
            reviews.setdefault(rec['id'], []).append(rec)
    keep = ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope')
    for c in cases:
        m = [x for x in reviews.get(c['id'], []) if x['content_sha256'] == c['content_sha256']]
        if m:
            c['review'] = dict(status='ai_reviewed', records=[{k: x[k] for k in keep if k in x} for x in m])
            if any(x['verdict'] != 'correct' for x in m):
                c['training_hold'] = 'review: AI review found a defect (see review.records)'
    a.out.mkdir(parents=True, exist_ok=True)
    body = ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases)
    (a.out / 'faq-candidates.jsonl').write_text(body)
    meta = dict(cases=len(cases), sha256=hashlib.sha256(body.encode()).hexdigest(),
                by_task_type={t: sum(c['task_type'] == t for c in cases) for t in FIT},
                by_institution={i: sum(c['institution'] == i for c in cases) for i in sorted({c['institution'] for c in cases})},
                reviewed=sum(c['review']['status'] == 'ai_reviewed' for c in cases),
                sources={f"{s['version']}/{p}": sha(PUB / s['version'] / f'{p}.jsonl') for s in spec['sources'] for p in s['parts']},
                note='C-020 follow-up candidate batch; not part of pilot v1/v3; AI self-review only; not production approval')
    (a.out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()

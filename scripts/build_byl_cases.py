"""Table and blank-form cases from B-016 statute appendices (data/law-byl/v2), checked against the original .hwp (C-019).

Gates, in order, for every document named in the spec (any failure stops the build):
  1. B status is candidate_table / candidate_form (not held), and the row is not one of B's 10 reserved partial overlaps;
  2. exposure: the law title, law group id and attachment id do not occur in evaluation-side data, and no 20-character
     window (step 5, whitespace and markup removed) of the attachment text occurs there either; only the printed
     paper-size note of forms (e.g. "210mm×297mm[백상지(80g/㎡) …]") is left out of this window check;
  3. tables only: the table read from the original .hwp (`hwp5proc xml`, cell text = paragraphs) equals the extracted
     table cell by cell (text with whitespace removed, rowspan, colspan, order) after dropping leading title rows of the
     original (one cell spanning the full width whose text sits in the extracted text before the table). Extracted
     tables may be HTML (<table>) or markdown (pipe rows; no spans).
Every case check is then run on BOTH grids: 'cell' (row path, column index, value), 'line_pair' (two cells of one row
holding parallel lines: the n-th line of one belongs to the n-th line of the other) and verbatim 'evidence' spans (in
the extracted text, and whitespace-free in the original). Blank forms are used only for what the form itself says
(field names, filling conditions, required items), never filled with made-up institution data.

AI-written questions and answers; review is AI self-review only; not production approval; values are as of the
attachment's revision and current values are unverified.

usage: build_byl_cases.py --spec reports/coordination/C-019/byl-cases.json --out DIR [--reviews F ...]
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_pilot_manifest import eval_side_text  # noqa: E402

ROWS = ROOT / 'data/law-byl/v2/extracted.jsonl'
RAW = ROOT / 'data/law-byl/raw'
STATUS = ROOT / 'reports/coordination/B-016/law-byl-v2.status.jsonl'
OVERLAP = ROOT / 'reports/coordination/B-016/law-byl-v2.overlap-148.json'
squash = lambda s: re.sub(r'\s+', '', s or '')
plain = lambda s: re.sub(r'</?(?:table|tr|td|th|thead|tbody|u|br|b|i|strong|em|span|p)\b[^>]*>|\s+', '', s or '')
PAPER = re.compile(r'\d+\s*mm\s*×\s*\d+\s*mm\s*\[[^\]]*\]')


def original(path):
    """(tables, full text) of a .hwp: each cell = dict(lines=[paragraph texts], rs, cs); nested tables excluded."""
    root = ET.fromstring(subprocess.run(['hwp5proc', 'xml', str(path)], check=True, capture_output=True).stdout)

    def ptext(e):
        if e.tag in ('TableBody', 'TableControl'):
            return ''
        return (''.join(e.itertext()) if e.tag == 'Text' else '') + ''.join(ptext(ch) for ch in e if e.tag != 'Text')

    def paras(e):
        out = []
        for ch in e:
            if ch.tag in ('TableBody', 'TableControl'):
                continue
            out += [ptext(ch)] if ch.tag == 'Paragraph' else paras(ch)
        return out

    tables = [[[dict(lines=paras(tc), rs=int(tc.get('rowspan', 1)), cs=int(tc.get('colspan', 1)))
                for tc in tr.findall('TableCell')] for tr in body.findall('TableRow')] for body in root.iter('TableBody')]
    text = ''.join(''.join(x.itertext()) for x in root.iter('Text'))
    return tables, text


class _Html(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.cell, self.depth = [], None, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'table':
            self.depth += 1
            if self.depth == 1:
                self.tables.append([])
        elif tag == 'tr' and self.depth == 1:
            self.tables[-1].append([])
        elif tag in ('td', 'th') and self.depth == 1:
            self.cell = dict(raw='', rs=int(a.get('rowspan', 1)), cs=int(a.get('colspan', 1)))
        elif tag == 'br' and self.cell is not None:
            self.cell['raw'] += '\n'

    def handle_endtag(self, tag):
        if tag == 'table':
            self.depth -= 1
        elif tag in ('td', 'th') and self.depth == 1 and self.cell is not None:
            self.tables[-1][-1].append(dict(lines=self.cell['raw'].split('\n'), rs=self.cell['rs'], cs=self.cell['cs']))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None and self.depth == 1:
            self.cell['raw'] += data


def extracted(text):
    """Tables of the extracted text in order: HTML <table> blocks and markdown pipe tables, with their offsets."""
    blocks = [(m.start(), m.end(), 'html') for m in re.finditer(r'<table>.*?</table>', text, re.S)]
    for m in re.finditer(r'(?:^\|.*\|[ \t]*(?:\n|$))+', text, re.M):
        if not any(s <= m.start() < e for s, e, _ in blocks):
            blocks.append((m.start(), m.end(), 'md'))
    out = []
    for s, e, kind in sorted(blocks):
        if kind == 'html':
            p = _Html()
            p.feed(text[s:e])
            out.append((s, e, p.tables[0]))
        else:
            rows = []
            for line in text[s:e].strip().split('\n'):
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                if all(re.fullmatch(r':?-{3,}:?', c) for c in cells if c):
                    continue
                rows.append([dict(lines=c.split('<br>'), rs=1, cs=1) for c in cells])
            out.append((s, e, rows))
    return out


def width(rows):
    return sum(c['cs'] for c in rows[0]) if rows else 0


def body(orig_rows, ext_rows, before):
    """Drop leading full-width single-cell title rows of the original whose text is in the text before the table."""
    w = width(ext_rows)
    i = 0
    while (i < len(orig_rows) and len(orig_rows[i]) == 1 and orig_rows[i][0]['cs'] >= w
           and squash(''.join(orig_rows[i][0]['lines'])) in squash(plain(before))):
        i += 1
    return orig_rows[i:], i


def same(a, b):
    ca = [(squash(''.join(c['lines'])), c['rs'], c['cs']) for r in a for c in r]
    cb = [(squash(''.join(c['lines'])), c['rs'], c['cs']) for r in b for c in r]
    return ca == cb, dict(rows=(len(a), len(b)), cells=(len(ca), len(cb)),
                          text_equal=sum(x[0] == y[0] for x, y in zip(ca, cb)), span_equal=sum(x[1:] == y[1:] for x, y in zip(ca, cb)))


def grid(rows):
    g = {}
    for i, row in enumerate(rows):
        j = 0
        for c in row:
            while (i, j) in g:
                j += 1
            for di in range(c['rs']):
                for dj in range(c['cs']):
                    g[(i + di, j + dj)] = c
            j += c['cs']
    n, m = max(i for i, _ in g) + 1, max(j for _, j in g) + 1
    return [[g.get((i, j), dict(lines=[''])) for j in range(m)] for i in range(n)]


def find_row(g, path):
    want = [squash(p) for p in path]
    for i, row in enumerate(g):
        cells = [squash(''.join(c['lines'])) for c in row]
        if all(w in cells for w in want):
            return i
    return None


def check(case, grids, ext_text, orig_text):
    for chk in case.get('checks', []):
        if chk['type'] == 'absent' and (squash(chk['value']) in squash(plain(ext_text)) or squash(chk['value']) in squash(orig_text)):
            raise ValueError(f"{case['id']}: {chk['value']!r} is present in the document, case says absent")
    for name, g in grids:
        for chk in case.get('checks', []):
            if chk['type'] == 'absent':
                continue
            i = find_row(g, chk['row'])
            if i is None:
                raise ValueError(f"{case['id']}: row {chk['row']} not found in {name} table")
            if chk['type'] == 'cell':
                got = ''.join(g[i][chk['col']]['lines'])
                if squash(got) != squash(chk['value']):
                    raise ValueError(f"{case['id']}: {name} {chk['row']} col {chk['col']} is {got!r}, case says {chk['value']!r}")
            elif chk['type'] == 'line_pair':
                a = [squash(x) for x in g[i][chk['col_a']]['lines'] if squash(x)]
                b = [squash(x) for x in g[i][chk['col_b']]['lines'] if squash(x)]
                if len(a) != len(b) or squash(chk['a']) not in a or squash(chk['b']) not in b \
                        or a.index(squash(chk['a'])) != b.index(squash(chk['b'])):
                    raise ValueError(f"{case['id']}: {name} lines {a} / {b} do not pair {chk['a']!r} with {chk['b']!r}")
    for span in case['evidence']:
        if span not in ext_text:
            raise ValueError(f"{case['id']}: evidence not verbatim in the extracted text: {span[:40]!r}")
        if squash(plain(span)) not in squash(orig_text):
            raise ValueError(f"{case['id']}: evidence not in the original .hwp text: {span[:40]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reviews', type=Path, action='append', default=[])
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    rows = {json.loads(l)['id']: json.loads(l) for l in ROWS.open()}
    status = {json.loads(l)['id']: json.loads(l) for l in STATUS.open()}
    partial = {p['corpus_id'] for p in json.loads(OVERLAP.read_text())['partial_review']}
    ev = eval_side_text()
    ev_plain = squash(plain(ev))
    windows = {ev_plain[i:i + 20] for i in range(len(ev_plain) - 19)}

    report, docs = {}, {}
    for doc in sorted({c['doc'] for c in spec['cases']}):
        r, st = rows[doc], status[doc]
        law = r['meta']['law_title']
        if st['status'] not in ('candidate_table', 'candidate_form') or doc in partial:
            raise SystemExit(f'{doc}: B status {st["status"]} / reserved partial overlap: not usable')
        t = squash(plain(PAPER.sub('', r['text'])))
        hits = sum(t[i:i + 20] in windows for i in range(0, max(1, len(t) - 19), 5))
        if law in ev or r['group'] in ev or doc in ev or hits:
            raise SystemExit(f'{doc}: evaluation exposure (title/id or {hits} shared 20-char windows)')
        fid = doc.rsplit(':', 1)[1]
        path = RAW / r['group'].split(':')[1] / f'{fid}.hwp'
        orig_tables, orig_text = original(path)
        info = dict(title=r['title'], law=law, b_status=st['status'], b_values_check=st['values_check'],
                    raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), exposure='none found', tables=[])
        ext = extracted(r['text'])
        if st['status'] == 'candidate_table':
            if len(ext) != len(orig_tables):
                raise SystemExit(f'{doc}: {len(orig_tables)} original vs {len(ext)} extracted tables')
            for k, ((s, e, et), ot) in enumerate(zip(ext, orig_tables)):
                ob, dropped = body(ot, et, r['text'][:s])
                ok, detail = same(ob, et)
                info['tables'].append(dict(index=k, exact_after_title_rows=ok, original_title_rows_dropped=dropped, **detail))
        docs[doc] = (r, ext, orig_tables, orig_text, info)
        report[doc] = info

    cases = []
    for c in spec['cases']:
        r, ext, orig_tables, orig_text, info = docs[c['doc']]
        grids = []
        if 'table' in c:
            tinfo = info['tables'][c['table']]
            if not tinfo['exact_after_title_rows']:
                raise SystemExit(f"{c['id']}: table {c['table']} does not equal the original; not usable")
            s, e, et = ext[c['table']]
            ob, _ = body(orig_tables[c['table']], et, r['text'][:s])
            grids = [('extracted', grid(et)), ('original', grid(ob))]
            context = [r['text'][max(0, s - 300):e]] + c.get('context_extra', [])
        else:
            context = [r['text']]
        check(c, grids, r['text'], orig_text)
        kind = 'table_behavior' if 'table' in c else 'form_behavior'
        cases.append(dict(
            id=c['id'], kind=kind, behavior=c['behavior'],
            task_type='table_lookup' if kind == 'table_behavior' else 'form_field_lookup',
            source_answer_present=c['behavior'] in ('answer', 'answer_with_condition', 'answer_with_exception'),
            question=c['question'], context=context, target=c['target'],
            evidence=dict(spans=c['evidence'], checks=c.get('checks', []), table=c.get('table')),
            doc=c['doc'], title=r['title'], source=r['source'], license=r['license'], date=r['date'],
            date_field='B law-byl v2 row date (law version date; the attachment header carries its own revision date)',
            text_sha256=r['sha256'], institution=r['meta']['law_title'], missing=c.get('missing', []),
            authored='question and explanation written by C (AI); values checked on the extracted table and on the '
                     'original .hwp' if kind == 'table_behavior' else
                     'question and explanation written by C (AI); quoted form text checked in the extracted text and '
                     'the original .hwp; the form is not filled with any institution data',
            synthetic_parts='question and explanation (AI-written)',
            content_sha256=hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest(),
            review=dict(status='unreviewed', records=[]), training_hold=None, split_hint=None))

    reviews = {}
    for p in a.reviews:
        for rec in map(json.loads, p.resolve().open()):
            reviews.setdefault(rec['id'], []).append(rec)
    keep = ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope', 'dimensions')
    for c in cases:
        m = [x for x in reviews.get(c['id'], []) if x['content_sha256'] == c['content_sha256']]
        if m:
            c['review'] = dict(status='ai_reviewed', records=[{k: x[k] for k in keep if k in x} for x in m])
            if any(x['verdict'] != 'correct' for x in m):
                c['training_hold'] = 'review: AI review found a defect (see review.records)'
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'byl-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (a.out / 'byl-candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    print(json.dumps(dict(documents={d: [t['exact_after_title_rows'] for t in v['tables']] or 'form' for d, v in report.items()},
                          cases=len(cases)), ensure_ascii=False))


if __name__ == '__main__':
    main()

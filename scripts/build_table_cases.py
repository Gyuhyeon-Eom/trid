"""Table-grounded internal-work cases from B-014 administrative-rule attachments (C-014), CPU only.

1. compare  For each listed .hwpx attachment, parse the tables straight from the original file's XML (no kordoc) and
            compare them cell by cell (text, rowSpan, colSpan, order) with the kordoc HTML tables in
            data/law-attachments/v4/candidates.jsonl. kordoc keeping table markup is evidence the structure was kept,
            not proof it is right; only tables that match the original exactly are used for cases.
2. cases    reports/coordination/C-014/table-cases.json: behaviour cases written by C (AI) on a matched table. Each
            case names its table and the rows (path of row-header cells) and the column it relies on; the builder
            checks, on BOTH the kordoc grid and the original-file grid, that the named rows exist and the mark sits in
            the named column, and that every quoted span is verbatim in the kordoc table text.

usage: build_table_cases.py --out DIR
"""
import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / 'data/law-attachments/v4/candidates.jsonl'
RAW = ROOT / 'data/law-attachments/raw'
CASES = ROOT / 'reports/coordination/C-014/table-cases.json'
HP = '{http://www.hancom.co.kr/hwpml/2011/paragraph}'
squash = lambda s: re.sub(r'\s+', '', s or '')


def hwpx_tables(path):
    """Top-level and nested tables in document order; a cell's text excludes nested tables."""
    z = zipfile.ZipFile(path)
    tables = []

    def cell_text(e):
        out = []
        for ch in e:
            if ch.tag == HP + 'tbl':
                continue
            if ch.tag == HP + 't':
                out.append(''.join(ch.itertext()))
                continue
            out.append(cell_text(ch))
        return ''.join(out)

    for name in sorted(n for n in z.namelist() if re.match(r'Contents/section\d+\.xml$', n)):
        for tbl in ET.fromstring(z.read(name)).iter(HP + 'tbl'):
            rows = []
            for tr in tbl.findall(HP + 'tr'):
                row = []
                for tc in tr.findall(HP + 'tc'):
                    span = tc.find(HP + 'cellSpan')
                    row.append(dict(text=cell_text(tc), rs=int(span.get('rowSpan', 1)) if span is not None else 1,
                                    cs=int(span.get('colSpan', 1)) if span is not None else 1))
                rows.append(row)
            tables.append(rows)
    return tables


class _Html(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.stack, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'table':
            self.stack.append(dict(rows=[], start=self.getpos()))
        elif tag == 'tr' and self.stack:
            self.stack[-1]['rows'].append([])
        elif tag in ('td', 'th') and self.stack:
            self.cell = dict(text='', rs=int(a.get('rowspan', 1)), cs=int(a.get('colspan', 1)), depth=len(self.stack))
            self.stack[-1]['rows'][-1].append(self.cell)

    def handle_endtag(self, tag):
        if tag == 'table' and self.stack:
            self.tables.append(self.stack.pop()['rows'])
        elif tag in ('td', 'th'):
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None and self.cell['depth'] == len(self.stack):
            self.cell['text'] += data


def html_tables(text):
    p = _Html()
    p.feed(text)
    return p.tables


def table_spans(text):
    """(start, end) character offsets of each top-level <table>…</table> in the kordoc text."""
    return [(m.start(), m.end()) for m in re.finditer(r'<table>.*?</table>', text, re.S)]


def grid(rows):
    """Expand spans: cell (i, j) -> text of the cell that covers it."""
    g = {}
    for i, row in enumerate(rows):
        j = 0
        for c in row:
            while (i, j) in g:
                j += 1
            for di in range(c['rs']):
                for dj in range(c['cs']):
                    g[(i + di, j + dj)] = c['text'].strip()
            j += c['cs']
    n_rows = max(i for i, _ in g) + 1
    n_cols = max(j for _, j in g) + 1
    return [[g.get((i, j), '') for j in range(n_cols)] for i in range(n_rows)]


def compare(orig, ext):
    """Cell-by-cell agreement of text and spans, in reading order."""
    oc = [(squash(c['text']), c['rs'], c['cs']) for r in orig for c in r]
    ec = [(squash(c['text']), c['rs'], c['cs']) for r in ext for c in r]
    return dict(rows=(len(orig), len(ext)), cells=(len(oc), len(ec)),
                text_equal=sum(a[0] == b[0] for a, b in zip(oc, ec)), span_equal=sum(a[1:] == b[1:] for a, b in zip(oc, ec)),
                exact=oc == ec)


def find_row(g, path):
    """Row index whose leading cells contain every element of path (each matched against squashed cell text)."""
    want = [squash(p) for p in path]
    for i, row in enumerate(g):
        cells = [squash(x) for x in row]
        if all(any(w == c for c in cells) for w in want):
            return i
    return None


def check_case(case, g_ext, g_orig, table_text):
    header = g_ext[0]
    for spec in case['rows']:
        col = header.index(spec['column'])
        for name, g in (('extracted', g_ext), ('original', g_orig)):
            i = find_row(g, spec['path'])
            if i is None:
                raise ValueError(f"{case['id']}: row {spec['path']} not found in {name} table")
            marked = [j for j, x in enumerate(g[i]) if x == '○']
            if marked != [col]:
                raise ValueError(f"{case['id']}: {name} row {spec['path']} marks columns {marked}, case says {spec['column']}")
    for span in case['evidence']:
        if span not in table_text:
            raise ValueError(f"{case['id']}: evidence not verbatim in the table text: {span[:40]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reviews', type=Path, action='append', default=[],
                    help='review records (jsonl: id, content_sha256, verdict, …); applied only on a content_sha256 match')
    a = ap.parse_args()
    reviews = {}
    for path in a.reviews:
        for rec in map(json.loads, path.resolve().open()):
            reviews.setdefault(rec['id'], []).append(dict(rec, file=str(path.resolve().relative_to(ROOT))))
    spec = json.loads(CASES.read_text())
    rows = {json.loads(l)['id']: json.loads(l) for l in V4.open()}
    comparisons, cases = {}, []
    for doc in spec['compare']:
        r = rows[doc]
        fid = doc.rsplit(':', 1)[1]
        path = RAW / r['group'].split(':')[1] / f'{fid}.hwpx'
        orig, ext = hwpx_tables(path), html_tables(r['text'])
        comparisons[doc] = dict(title=r['title'], original_tables=len(orig), extracted_tables=len(ext),
                                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                tables=[compare(o, e) for o, e in zip(orig, ext)] if len(orig) == len(ext) else 'table count differs')
    for c in spec['cases']:
        r = rows[c['doc']]
        cmp = comparisons[c['doc']]
        if cmp['tables'] == 'table count differs' or not cmp['tables'][c['table']]['exact']:
            raise ValueError(f"{c['id']}: table {c['table']} does not match the original exactly; not usable")
        fid = c['doc'].rsplit(':', 1)[1]
        orig = hwpx_tables(RAW / r['group'].split(':')[1] / f'{fid}.hwpx')
        start, end = table_spans(r['text'])[c['table']]
        heading_start = r['text'].rfind('<u>', 0, start)
        context = r['text'][heading_start:end]
        check_case(c, grid(html_tables(r['text'])[c['table']]), grid(orig[c['table']]), r['text'][start:end])
        cases.append(dict(id=c['id'], kind='table_behavior', behavior=c['behavior'], task_type='table_lookup',
                          source_answer_present=c['behavior'] in ('answer_with_condition', 'answer_with_exception', 'answer'),
                          question=c['question'], context=[context], target=c['target'],
                          evidence=dict(spans=c['evidence'], rows=c['rows'], table=c['table'],
                                        context_offsets=[heading_start, end]),
                          doc=c['doc'], title=r['title'], source=r['source'], license=r['license'], date=r['date'],
                          date_field='B v4 row date (attachment rule header)', text_sha256=r['sha256'],
                          institution=r['provenance']['institution'], missing=c.get('missing', []),
                          authored='question and explanation written by C (AI); table rows/marks checked against the '
                                   'kordoc table and the original .hwpx', synthetic_parts='question and explanation (AI-written)',
                          content_sha256=hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest(),
                          review=dict(status='unreviewed', records=[]), training_hold=None))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'table-comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=1))
    for c in cases:
        matching = [r for r in reviews.get(c['id'], []) if r['content_sha256'] == c['content_sha256']]
        if matching:
            c['review'] = dict(status='ai_reviewed', records=[{k: r[k] for k in ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope', 'file') if k in r}
                                                              for r in matching])
            if any(r['verdict'] != 'correct' for r in matching):
                c['training_hold'] = 'review: AI review found a defect (see review.records)'
    (a.out / 'table-candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    (a.out / 'doc-provenance.json').write_text(json.dumps(dict(
        rule='provenance only; not a list for removing documents from existing evaluation sets',
        docs={c['doc']: 'training_candidate_' + ('ai_reviewed' if c['review']['status'] == 'ai_reviewed' else 'unreviewed')
              for c in cases},
        compared_not_used={d: 'tables compared with the original; no cases built' for d in comparisons
                           if d not in {c['doc'] for c in cases}}), ensure_ascii=False, indent=2))
    summary = {d: dict(tables=f"{v['original_tables']}/{v['extracted_tables']}",
                       exact=(sum(t['exact'] for t in v['tables']) if isinstance(v['tables'], list) else 0))
               for d, v in comparisons.items()}
    print(json.dumps(dict(comparison=summary, cases=len(cases)), ensure_ascii=False))


if __name__ == '__main__':
    main()

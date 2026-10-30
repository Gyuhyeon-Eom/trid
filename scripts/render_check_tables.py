"""Check structured 별표 rows against a rendering of the same HWP by a second, independent program (rhwp export-svg).

structure_statute_tables.py builds rows from pyhwp's cell grid and splits cells holding parallel paragraph lists by
paragraph position. That split, and the column order, are what a reader relies on. rhwp lays the file out on pages
and writes every glyph with its x/y position, so a table row drawn on the page is a line of glyphs at one baseline.
Per 별표:
  row      the row's own numbers (cells that belong to this row: not merged from above, not repeated beside a split,
           not joined from several source lines, and <= 24 characters so wrapped prose is not involved), in column
           order, must appear as consecutive numbers on one rendered line. pass / fail / not_checkable (fewer than 2
           own numbers: a single number cannot show a column or row shift).
  header   every column label's first part is found in the rendered text
  unit     the '(…단위…)' text is found in the rendered text
  notes    every note line is found in the rendered text
Text matching ignores whitespace. A row passes only when header, unit and notes of its 별표 pass too. This is an
automatic cross-reader check of row/column placement, not a visual review and not an answer check; it does not look at
any question or answer.

  .venv/bin/python scripts/render_check_tables.py --structured data/law-byl/v2-structure-002 \
      --statutes 공무원보수규정 '공무원수당 등에 관한 규정' --out reports/coordination/B-029/render-check.jsonl
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
# integers only, thousands separators kept: rhwp draws no space glyphs, so '1. 5톤' renders as '1.5톤'; reading both
# sides as integers (1, 5) keeps a decimal point from joining numbers that the source separates by a space
NUM = re.compile(r'\d{1,3}(?:,\d{3})+(?!\d)|\d+')
SHORT = 24


def glyph_lines(svg_dir):
    """Rendered lines [(page, y, [(x, char, size, cell)])] in reading order from rhwp's per-glyph <text> elements;
    `cell` is the clip group rhwp draws each table cell in (None outside tables). Cells of one table row can sit a
    pixel apart (cell padding, vertical centring), so baselines within 0.35 font sizes of the line's first baseline are
    one line; rows of a table are ~2 font sizes apart."""
    svg = '{http://www.w3.org/2000/svg}'
    lines = []
    for page, f in enumerate(sorted(Path(svg_dir).glob('*.svg'))):
        glyphs = []

        def walk(e, cell):
            clip = re.search(r'#(cell-clip-\d+)', e.get('clip-path') or '')
            cell = clip.group(1) if clip else cell
            if e.tag == svg + 'text' and e.text is not None:
                if e.get('y') is not None:
                    x, y = float(e.get('x')), float(e.get('y'))
                else:  # condensed glyphs: transform="translate(x,y) scale(..)"
                    m = re.search(r'translate\(([-\d.]+)[ ,]+([-\d.]+)\)', e.get('transform') or '')
                    if not m:
                        return
                    x, y = float(m.group(1)), float(m.group(2))
                glyphs.append((y, x, e.text, float(e.get('font-size') or 10), cell))
            for k in e:
                walk(k, cell)
        walk(ET.parse(f).getroot(), None)
        cur, start = [], None
        for y, x, ch, size, cell in sorted(glyphs, key=lambda g: (g[0], g[1])):
            if cur and y - start > 0.35 * size:
                lines.append((page, start, sorted(cur, key=lambda g: g[0])))
                cur = []
            if not cur:
                start = y
            cur.append((x, ch, size, cell))
        if cur:
            lines.append((page, start, sorted(cur, key=lambda g: g[0])))
    return lines


def numbers_on(line):
    """Numbers of one rendered line; a new token starts at every change of table cell (rhwp's clip group) and where
    the step to the next glyph exceeds 1.6 font sizes, so values of neighbouring cells never merge (narrow columns of
    a wide 봉급표 are drawn closer than one font size apart)."""
    tokens, cur, last, last_cell = [], '', None, None
    for x, ch, size, cell in line:
        if last is not None and (cell != last_cell or x - last > size * 1.6):
            tokens.append(cur)
            cur = ''
        cur += ch
        last, last_cell = x, cell
    tokens.append(cur)
    return [n.replace(',', '') for t in tokens for n in NUM.findall(t)]


def contains(seq, sub):
    return any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def rendered_cells(svg_dir):
    """[(page, x, y, w, h, text)] per table cell rhwp drew (its clip rectangle and all glyphs in it, whitespace
    removed), for cells whose text wraps over several lines."""
    svg = '{http://www.w3.org/2000/svg}'
    out = []
    for page, f in enumerate(sorted(Path(svg_dir).glob('*.svg'))):
        root = ET.parse(f).getroot()
        boxes = {}
        for cp in root.iter(svg + 'clipPath'):
            rect = cp.find(svg + 'rect')
            if (cp.get('id') or '').startswith('cell-clip-') and rect is not None:
                boxes[cp.get('id')] = tuple(float(rect.get(k)) for k in ('x', 'y', 'width', 'height'))
        texts = {}
        for g in root.iter(svg + 'g'):
            m = re.search(r'#(cell-clip-\d+)', g.get('clip-path') or '')
            if m and m.group(1) in boxes:
                glyphs = []
                for t in g.iter(svg + 'text'):
                    if t.text:
                        pos = (t.get('y'), t.get('x')) if t.get('y') else re.search(
                            r'translate\(([-\d.]+)[ ,]+([-\d.]+)\)', t.get('transform') or '').group(2, 1)
                        glyphs.append((float(pos[0]), float(pos[1]), t.text))
                texts.setdefault(m.group(1), []).extend(glyphs)
        for cid, glyphs in texts.items():
            out.append((page, *boxes[cid], re.sub(r'\s', '', ''.join(ch for _, _, ch in sorted(glyphs)))))
    return out


def cells_in_one_row(values, cells):
    """True when every value (whitespace removed) is the whole text of a rendered cell, all on one page, the cells
    overlapping vertically (one table row) and left to right in the given column order."""
    options = [[c for c in cells if c[5] == v] for v in values]
    if not all(options):
        return False

    def pick(i, page, top, bottom, left):
        if i == len(options):
            return True
        return any(pick(i + 1, page, max(top, y), min(bottom, y + h), x)
                   for p, x, y, w, h, _ in options[i]
                   if (page is None or p == page) and x > left and min(bottom, y + h) > max(top, y) + 1)
    return pick(0, None, float('-inf'), float('inf'), float('-inf'))


def check_doc(d):
    raw = ROOT / d['raw']
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(['rhwp', 'export-svg', str(raw), '-o', tmp], capture_output=True, timeout=600)
        if r.returncode:
            return {'id': d['id'], 'render': 'failed'}, []
        lines = glyph_lines(tmp)
        cells = rendered_cells(tmp)
    flat = re.sub(r'\s', '', ''.join(g[1] for _, _, line in lines for g in line))
    norm = lambda s: re.sub(r'\s', '', s)
    doc = {'id': d['id'], 'title': d['title'], 'render': 'ok', 'pages': len({p for p, _, _ in lines})}
    labels = [lab for t in d['tables'] for lab in t['columns'] if not re.fullmatch(r'열\d+', lab)]
    # every word of a label's top part (a two-line header cell renders as two lines)
    # a wrapped header cell interleaves with its neighbours on the page, so a label is also found as a whole cell
    cell_text = ''.join(f'\n{c[5]}' for c in cells)
    doc['header_missing'] = [lab for lab in labels if any(norm(w) not in flat for w in lab.split(' > ')[0].split())
                             and norm(lab.split(' > ')[0]) not in cell_text]
    doc['unit_found'] = None if not d['unit'] else norm(d['unit']) in flat
    doc['notes_missing'] = [n for t in d['tables'] for n in t['notes'] if norm(n) not in flat]
    side_ok = not doc['header_missing'] and doc['unit_found'] is not False and not doc['notes_missing']
    line_numbers = [numbers_on(g) for _, _, g in lines]
    results, c = [], Counter()
    for i, t in enumerate(d['tables']):
        if not t['rows'] or 'irregular_layout_rows_not_emitted' in t['flags']:
            continue
        for n, row in enumerate(t['rows']):
            repeated = set(row.get('repeated') or [])
            skip = repeated | set(row.get('joined') or [])  # drawn once / on several lines
            own = [v for k, v in row['cells'].items() if k not in skip and len(v) <= SHORT]
            seq = [x.replace(',', '') for v in own for x in NUM.findall(v)]
            # a whole cell per value: wrapped text, joined lines; a split row's parts are not whole cells
            whole = [norm(v) for k, v in row['cells'].items() if k not in repeated] if not row.get('part') else []
            on_line = len(seq) >= 2 and any(contains(ln, seq) for ln in line_numbers)
            as_cells = len(whole) >= 2 and cells_in_one_row(whole, cells)
            if on_line or as_cells:
                status = 'pass' if side_ok else 'fail'
            elif len(seq) >= 2:
                status = 'fail'  # the row's own numbers exist but are not drawn together in this order
            else:
                status = 'not_checkable'  # text-only row whose cells could not be matched one to one
            detail = None
            if status == 'fail':
                detail = ('row values neither on one rendered line in column order nor whole cells of one rendered row'
                          if side_ok else
                          f"header/unit/notes not found in rendering: {doc['header_missing'][:3]} "
                          f"{'unit' if doc['unit_found'] is False else ''} {[x[:30] for x in doc['notes_missing'][:2]]}")
            c[status] += 1
            results.append({'row_id': f"{d['id']}:t{i}:r{n}", 'status': status, 'numbers': seq,
                            'method': 'line' if on_line else 'cells' if as_cells else None, 'detail': detail})
    doc['rows'] = dict(c)
    return doc, results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--structured', required=True, help='structure version dir (structured.jsonl)')
    p.add_argument('--statutes', nargs='+', required=True, help='statute titles (text before " [별표")')
    p.add_argument('--out', required=True, help='row results jsonl; a <out>.docs.json summary is written beside it')
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; outputs are never overwritten')
    docs = [json.loads(x) for x in (ROOT / args.structured / 'structured.jsonl').read_text().splitlines() if x]
    docs = [d for d in docs if d.get('status') in ('structured', 'partly_structured_irregular_table')
            and d['title'].split(' [별표')[0] in args.statutes]
    summaries, rows = [], []
    for d in docs:
        s, r = check_doc(d)
        summaries.append(s)
        rows += r
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    total = Counter(r['status'] for r in rows)
    summary = {'structured': args.structured, 'statutes': args.statutes, 'code_sha256': CODE_SHA256, 'docs': len(docs),
               'rows': dict(total), 'per_doc': summaries}
    out.with_suffix('.docs.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ('docs', 'rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

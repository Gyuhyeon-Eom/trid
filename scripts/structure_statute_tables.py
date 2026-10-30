"""Structure of statute appended tables (법령 별표) read from the HWP file's own table grid, for every 별표 (no row picking).

The search corpus carries each 별표 as kordoc Markdown: one Markdown row per table row, several values of one HWP cell
joined with <br>, merged cells and multi-level headers flattened. A reader then has to line up '가<br>나<br>다<br>라'
with '60<br>60<br>60<br>60' itself, and misreads tables. This step reads the same HWP with an independent reader
(pyhwp `hwp5proc xml`: cells with row/col/rowspan/colspan and their paragraphs) and writes, per 별표:
  caption      full-width rows above the header (title line, '(단위: …)')
  columns      column labels; a multi-level header is joined top-down with ' > '
  rows         one record per body row: {column label: cell text}; a cell merged over several rows is repeated in each
               row it covers; a row whose cells hold the same number k (>= 2) of paragraphs, laid out as parallel
               lists (e.g. 등급 가/나/다/라 next to 일비 60/60/60/60), is split into k rows by paragraph position.
               Parts that continue a value ('(특실)', '(상한액: …') are never split: those cells are joined with a space
  sections     full-width rows between body rows (group labels) are carried into the rows below them
  notes        full-width rows below the body and paragraphs after the table (비고, ※), verbatim, in order
  unit / revised / statute_effective_date   from '(단위: …)', '<개정 YYYY. M. D.>' and the corpus row's provenance
and a linear text form (`structured_text`). Cell text is the source paragraph text (whitespace collapsed); nothing is
computed, normalised or chosen per question. Checks per 별표: the grid is fully covered without overlaps, and the
numbers in the grid + surrounding paragraphs equal the numbers in the kordoc text (collect_statute_tables.value_profile).
A check passing says the values and the HWP grid survived; it does not say an answer read from them is right.

  .venv/bin/python scripts/structure_statute_tables.py --version v2 --out data/law-byl/v2-structure-001
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from collect_statute_tables import value_profile  # noqa: E402
from collect_admrul_attachments import strip_table_markup  # noqa: E402
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
HWP5PROC = ['uv', 'run', '-q', '--no-project', '--python', '3.12', '--with', 'pyhwp', '--with', 'six', '--with', 'lxml',
            'hwp5proc', 'xml']
CONTINUATION = re.compile(r'^[(（<※\[]|[,，]$')


def text_of(elem):
    """Text of an element, not descending into nested tables."""
    out = []

    def walk(e):
        if e.tag == 'TableControl':
            return
        if e.tag == 'Text' and e.text:
            out.append(e.text)
        for k in e:
            walk(k)
    walk(elem)
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def paragraphs(e):
    """Paragraphs under e (a section or a table cell), also inside ColumnSet, not those of nested tables."""
    for k in e:
        if k.tag == 'Paragraph':
            yield k
        elif k.tag != 'TableControl':
            yield from paragraphs(k)


def cell_parts(cell):
    return [t for t in (text_of(p) for p in paragraphs(cell)) if t]


def grid(table):
    """(rows, cols, cells, flags): cells = [{r, c, rs, cs, parts}] from one TableControl."""
    body = table.find('TableBody')
    rows, cols = int(body.get('rows')), int(body.get('cols'))
    cells, flags = [], set()
    for row in body.findall('TableRow'):
        for c in row.findall('TableCell'):
            if c.find('.//TableControl') is not None:
                flags.add('nested_table_text_not_included')
            cells.append({'r': int(c.get('row')), 'c': int(c.get('col')), 'rs': int(c.get('rowspan')),
                          'cs': int(c.get('colspan')), 'parts': cell_parts(c)})
    return rows, cols, cells, flags


def covered(rows, cols, cells):
    """(r, c) -> cell, plus whether every grid position is covered exactly once."""
    at, overlap = {}, False
    for cell in cells:
        for r in range(cell['r'], min(rows, cell['r'] + cell['rs'])):
            for c in range(cell['c'], min(cols, cell['c'] + cell['cs'])):
                overlap |= (r, c) in at
                at[(r, c)] = cell
    return at, len(at) == rows * cols and not overlap


def stacked(row_cells):
    """k when the row's multi-paragraph cells are parallel lists of k values, else None."""
    multi = [c for c in row_cells if len(c['parts']) > 1]
    ks = {len(c['parts']) for c in multi}
    if len(multi) < 2 or len(ks) != 1:
        return None
    if any(CONTINUATION.search(p) for c in multi for p in c['parts']):
        return None
    return ks.pop()


def structure(table):
    rows, cols, cells, flags = grid(table)
    at, complete = covered(rows, cols, cells)
    full = lambda r: all(at.get((r, c)) is at.get((r, 0)) for c in range(cols)) and at.get((r, 0)) is not None
    r = 0
    caption = []
    while r < rows and full(r):
        caption += at[(r, 0)]['parts']
        r += 1
    out = {'grid': {'rows': rows, 'cols': cols, 'complete': complete}, 'caption': caption, 'columns': [],
           'rows': [], 'notes': [], 'flags': sorted(flags),
           'cell_texts': [' '.join(c['parts']) for c in cells]}  # each cell once, for the value check (dropped on output)
    if r >= rows:
        out['flags'].append('no_header_row')
        return out
    depth = max(c['rs'] for c in cells if c['r'] == r)
    if cols == 1:
        depth = 0
    labels = []
    for c in range(cols):
        parts = []
        for h in range(r, r + depth):
            t = ' '.join(at[(h, c)]['parts']) if (h, c) in at else ''
            if t and t not in parts:
                parts.append(t)
        labels.append(' > '.join(parts) or f'열{c + 1}')
    out['columns'], out['header_rows'] = labels, depth
    # Several tables laid side by side in one HWP grid, or a header repeated inside the body, cannot be read as one
    # header + rows: such a table gets no row records (its source text stays the reference)
    irregular = len(set(labels)) < cols
    r += depth
    body_end = rows
    while body_end > r and full(body_end - 1):
        body_end -= 1
    for n in range(body_end, rows):
        out['notes'] += at[(n, 0)]['parts']
    section, used = None, Counter()
    for n in range(r, body_end):
        if full(n):
            section = ' '.join(at[(n, 0)]['parts']) or None
            continue
        row_cells, seen = [], set()
        for c in range(cols):
            cell = at.get((n, c))
            if cell is not None and id(cell) not in seen:
                seen.add(id(cell))
                row_cells.append((c, cell))
        own = [cell for _, cell in row_cells if cell['r'] == n]
        k = stacked(own)
        if any(cell['r'] < n for _, cell in row_cells):
            used['merged_cell_repeated'] += 1
        for i in range(k or 1):
            values = {}
            for c, cell in row_cells:
                parts = cell['parts']
                v = parts[i] if k and cell['r'] == n and len(parts) == k else ' '.join(parts)
                if k and cell['r'] == n and len(parts) not in (1, k):
                    used['stacked_row_cell_count_mismatch'] += 1
                if v:
                    values[labels[c] if cell['cs'] == 1 else ' / '.join(dict.fromkeys(labels[c:c + cell['cs']]))] = v
            if values:
                out['rows'].append({'source_row': n, 'part': (i + 1) if k else None, 'section': section, 'cells': values})
        if k:
            used['row_split_by_paragraph_position'] += 1
        elif any(len(cell['parts']) > 1 for _, cell in row_cells):
            used['multi_paragraph_cell_joined'] += 1
    named = {x for x in labels if not re.fullmatch(r'열\d+', x)} | {p for x in labels for p in x.split(' > ')}
    if any(v in named for row in out['rows'] for v in row['cells'].values()):
        irregular = True  # a header label appears again as a body value: a second table in the same grid
    if used['stacked_row_cell_count_mismatch']:
        irregular = True
    if irregular:
        out['flags'].append('irregular_layout_rows_not_emitted')
        out['rows'] = []
    out['counts'] = dict(used)
    return out


def document(xml):
    """Tables and the paragraphs around them, in document order, from hwp5proc xml output."""
    root = ET.fromstring(xml)
    items = []
    for sec in root.find('BodyText'):
        for p in paragraphs(sec):
            tables = [t for t in p.iter('TableControl')]
            # only top-level tables of this paragraph (a nested table belongs to its cell)
            nested = {id(n) for t in tables for n in t.iter('TableControl') if n is not t}
            for t in tables:
                if id(t) not in nested:
                    items.append(('table', structure(t)))
            t = text_of(p)
            if t:
                items.append(('text', t))
    return items


def linear(row, items):
    head = row['structure_meta']
    lines = [f"[별표 구조] {row['title']}"]
    meta = [f"별표 {head['revised']}" if head['revised'] else None,
            f"법령 시행일 {head['statute_effective_date']}" if head['statute_effective_date'] else None,
            head['unit']]  # the whole parenthetical, e.g. '월지급액, 단위: 원'
    if any(meta):
        lines.append(' | '.join(m for m in meta if m))
    for kind, x in items:
        if kind == 'text':
            lines.append(x)
            continue
        lines += x['caption']
        if 'irregular_layout_rows_not_emitted' in x['flags']:
            lines.append('[이 표는 한 격자에 여러 표가 붙어 있거나 머리행이 본문에 반복되어 행·열 구조로 풀지 않았습니다. 원문 표를 보세요]')
        elif x['columns']:
            lines.append('열: ' + ' | '.join(x['columns']))
        for r in x['rows']:
            prefix = f"[{r['section']}] " if r['section'] else ''
            lines.append('- ' + prefix + ' / '.join(f'{k}: {v}' for k, v in r['cells'].items()))
        lines += [f'비고(표 안): {n}' for n in x['notes']]
    return '\n'.join(lines)


def run(row):
    res = subprocess.run(HWP5PROC + [str(ROOT / row['meta']['raw'])], capture_output=True, timeout=600)
    if res.returncode:
        return {'id': row['id'], 'status': 'reader_failed'}
    items = document(res.stdout.decode('utf-8', 'replace'))
    tables = [x for k, x in items if k == 'table']
    all_text = '\n'.join(x if k == 'text' else '\n'.join(x.pop('cell_texts')) for k, x in items)
    unit = re.search(r'\(((?:[^()]|\([^()]*\))*단위\s*[:：](?:[^()]|\([^()]*\))*)\)', all_text)
    revised = re.search(r'<\s*((?:전부)?개정|제정|신설)\s*(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?)\s*>', all_text)
    meta = {'unit': unit.group(1).strip() if unit else None,
            'revised': f'{revised.group(1)} {revised.group(2)}' if revised else None,
            'statute_effective_date': (row.get('provenance') or {}).get('effective_date')}
    row = {**row, 'structure_meta': meta}
    (a_n, a_u), (b_n, b_u) = value_profile(strip_table_markup(row['text'])), value_profile(all_text)
    checks = {'tables': len(tables), 'grid_complete': all(t['grid']['complete'] for t in tables),
              'numbers_vs_kordoc': 'agree' if a_n == b_n else 'differ',
              'numbers_only_in_kordoc': dict(a_n - b_n), 'numbers_only_in_grid': dict(b_n - a_n)}
    irregular = any('irregular_layout_rows_not_emitted' in t['flags'] for t in tables)
    status = ('no_table' if not tables else
              'structured_check_differs' if not (checks['grid_complete'] and checks['numbers_vs_kordoc'] == 'agree') else
              'partly_structured_irregular_table' if irregular else 'structured')
    return {'id': row['id'], 'title': row['title'], 'source': row['source'], 'raw': row['meta']['raw'],
            'raw_sha256': row['meta'].get('raw_sha256'), 'kordoc_text_sha256': row['sha256'], 'status': status,
            **meta, 'tables': tables, 'checks': checks, 'structured_text': linear(row, items),
            'note': 'HWP grid structure of the whole 별표; not chosen per question; not an answer check'}


def split(structured, review, out):
    """Queryable rows apart from everything else. A row is queryable only from a 'structured' 별표 (grid complete,
    numbers agree), from a table that is regular and has no nested table. Every queryable row is a candidate,
    'unreviewed', unless the 별표 is in the review file with result 'match' (an AI visual review against a rendering of
    the same HWP; not a human approval). Everything else is listed in not-queryable.jsonl with the reason."""
    reviewed = {}
    for x in (ROOT / review).read_text().splitlines() if review else []:
        r = json.loads(x)
        reviewed[r['id']] = r
    rows, held, c = [], [], Counter()
    for line in (ROOT / structured / 'structured.jsonl').read_text().splitlines():
        d = json.loads(line)
        tables = d.get('tables') or []
        if d['status'] != 'structured' and d['status'] != 'partly_structured_irregular_table':
            held.append({'id': d['id'], 'title': d.get('title'), 'table': None, 'reason': d['status']})
            c[f"doc:{d['status']}"] += 1
            continue
        rv = reviewed.get(d['id'])
        for i, t in enumerate(tables):
            why = [f for f in t['flags'] if f in ('irregular_layout_rows_not_emitted', 'nested_table_text_not_included',
                                                  'no_header_row')]
            if why or not t['rows']:
                held.append({'id': d['id'], 'title': d['title'], 'table': i, 'reason': why or ['no_rows']})
                c['table:not_queryable'] += 1
                continue
            state = ('ai_visual_review_match' if rv and rv['result'] == 'match' and i in rv['tables_reviewed']
                     else 'candidate_unreviewed')
            c[f'rows:{state}'] += len(t['rows'])
            for n, r in enumerate(t['rows']):
                rows.append({'id': f"{d['id']}:t{i}:r{n}", 'doc_id': d['id'], 'title': d['title'], 'table': i,
                             'caption': t['caption'], 'unit': d['unit'], 'revised': d['revised'],
                             'statute_effective_date': d['statute_effective_date'], 'section': r['section'],
                             'cells': r['cells'], 'notes': t['notes'], 'review': state,
                             'review_ref': rv.get('review_ref') if rv and state != 'candidate_unreviewed' else None})
    out = ROOT / out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    out.mkdir(parents=True)
    files = {}
    for name, sel in (('queryable-rows', rows), ('not-queryable', held)):
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        files[name] = {'rows': len(sel), 'sha256': hashlib.sha256(body).hexdigest()}
    manifest = {'source': f'{structured}/structured.jsonl',
                'source_sha256': hashlib.sha256((ROOT / structured / 'structured.jsonl').read_bytes()).hexdigest(),
                'review': str(review) if review else None, 'code_sha256': CODE_SHA256, 'files': files, 'counts': dict(c),
                'note': ('queryable-rows are candidates; only rows marked ai_visual_review_match were compared with a '
                         'rendering, by an AI reviewer. Nothing here is approved, and no row was chosen per question.')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', default='v2', help='data/law-byl/<version>/extracted.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--ids', nargs='*', help='only these rows (smoke runs)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--split', metavar='STRUCTURED_DIR', help='write queryable / not-queryable rows of a structure version')
    p.add_argument('--review', help='AI visual review file (jsonl: id, result, tables_reviewed, review_ref) for --split')
    args = p.parse_args()
    if args.split:
        return split(args.split, args.review, args.out)
    out = ROOT / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    rows = [json.loads(x) for x in (ROOT / 'data/law-byl' / args.version / 'extracted.jsonl').read_text().splitlines() if x]
    rows = [r for r in rows if r['kind'] == 'statute_appendix' and '[별표' in r['title'] and r['meta']['raw'].endswith('.hwp')
            and (not args.ids or r['id'] in args.ids)]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, rows))
    out.mkdir(parents=True, exist_ok=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results).encode()
    (out / 'structured.jsonl').write_bytes(body)
    flags = Counter(f for r in results for t in r.get('tables', []) for f in t['flags'])
    counts = Counter(k for r in results for t in r.get('tables', []) for k, v in t.get('counts', {}).items() for _ in range(v))
    manifest = {'source': f'data/law-byl/{args.version}/extracted.jsonl', 'code_sha256': CODE_SHA256, 'rows': len(results),
                'status': dict(Counter(r['status'] for r in results)), 'table_flags': dict(flags), 'row_handling': dict(counts),
                'sha256': hashlib.sha256(body).hexdigest(),
                'note': ('Structure of every HWP 별표 in the source version, from the HWP grid (pyhwp hwp5proc xml); '
                         'not chosen per question and not an answer check. Not approved for training.')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

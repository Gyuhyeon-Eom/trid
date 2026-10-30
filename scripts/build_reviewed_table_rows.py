"""Reviewed table rows of an institution snapshot record, in the reviewed-row schema of structure_statute_tables.py split
(queryable-rows.jsonl), for tables a reviewer compared with the original PDF pages.

A PDF rule (e.g. 한국부동산원 여비규정, inst-rule-reb-002) has no HWP grid; its tables are the kordoc tables (HTML or
Markdown) inside the snapshot record text that A indexes. For every table of each listed 별표, every row is written
(no row picked per question):
  cells        {column label: cell text}; a multi-level header is joined top-down with ' > ', cell line breaks with a
               space; symbols such as '〃' (위와 같음) are kept as written, not resolved
  unit         the nearest '(단위 …)' line above the table inside the 별표
  conditions   text between the previous table (or the 별표 caption) and this table, e.g. '…아래 표에 따른다'
  notes        text after the table up to the next table or the end of the 별표 (비고, * footnotes), verbatim
  source       snapshot id, record source_id, record content_hash, char span of the table in the record text, raw
               sha256, PDF page of the 별표
  canonical    institution, rule, 별표 number as written, caption, 개정 date when the 별표 header gives one
  review       'ai_visual_review_match' with the review file reference (an AI visual comparison, not a human approval)

  .venv/bin/python scripts/build_reviewed_table_rows.py --snapshot runs/search-snapshots/inst-rule-reb-002 \
      --source-id reb.or.kr:1244:115624:file:e49a6aae28155d3a --rule 여비규정 --pages 별표1=10 별표2=11 별표3=12 \
      --review reports/coordination/B-028/reb-yeobi-table-review.jsonl --out reports/coordination/B-029/reb-yeobi-reviewed-rows.jsonl
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
TABLE = re.compile(r'<table>[\s\S]*?</table>|(?:^\|[^\n]*\|[ \t]*(?:\n|$))+', re.M)


def clean(s):
    s = re.sub(r'<br\s*/?>', ' ', s)
    s = html.unescape(re.sub(r'<[^>]+>', '', s))
    s = re.sub(r'\\([\\`*_{}\[\]()#+\-.!$|])', r'\1', s)  # Markdown escapes kordoc adds ('\$', '\*')
    return re.sub(r'\s+', ' ', re.sub(r'^#+\s*', '', s.strip())).strip()


def grid_html(block):
    """[[cell text]] with rowspan/colspan expanded, and the header depth (max rowspan of the first row)."""
    rows, spans = [], {}
    for r, tr in enumerate(re.findall(r'<tr>(.*?)</tr>', block, re.S)):
        out, c = [], 0
        cells = re.findall(r'<(t[hd])([^>]*)>(.*?)</t[hd]>', tr, re.S)
        it = iter(cells)
        while True:
            while (r, c) in spans:
                out.append(spans.pop((r, c)))
                c += 1
            cell = next(it, None)
            if cell is None:
                break
            attrs, text = cell[1], clean(cell[2])
            rs = int((re.search(r'rowspan="(\d+)"', attrs) or [0, 1])[1])
            cs = int((re.search(r'colspan="(\d+)"', attrs) or [0, 1])[1])
            for k in range(cs):
                out.append(text)
                for d in range(1, rs):
                    spans[(r + d, c + k)] = text
            c += cs
        rows.append(out)
    first = re.findall(r'<tr>(.*?)</tr>', block, re.S)[0]
    depth = max(int(x) for x in re.findall(r'rowspan="(\d+)"', first) or ['1'])
    return rows, depth


def grid_markdown(block):
    rows = [[clean(c) for c in line.strip().strip('|').split('|')] for line in block.strip().split('\n')]
    return [r for r in rows if not all(re.fullmatch(r':?-{3,}:?', c) for c in r)], 1


def table_rows(block):
    rows, depth = grid_html(block) if block.startswith('<table>') else grid_markdown(block)
    labels = []
    for c in range(max(len(r) for r in rows)):
        parts = []
        for h in rows[:depth]:
            t = h[c] if c < len(h) else ''
            if t and t not in parts:
                parts.append(t)
        labels.append(' > '.join(parts) or f'열{c + 1}')
    body = [{labels[c]: v for c, v in enumerate(r) if v} for r in rows[depth:]]
    return labels, [b for b in body if b]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--source-id', required=True)
    p.add_argument('--rule', required=True, help='rule name as written in the source, e.g. 여비규정')
    p.add_argument('--pages', nargs='+', required=True, help='별표N=PDF page of that 별표')
    p.add_argument('--review', required=True, help='review jsonl (table, result) of the listed 별표')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; outputs are never overwritten')
    snap = ROOT / args.snapshot
    manifest = json.loads((snap / 'manifest.json').read_text())
    rec = next(json.loads(x) for f in (snap / 'normalized').glob('*.jsonl') for x in f.read_text().splitlines()
               if x and json.loads(x)['source_id'] == args.source_id)
    text = rec['text']
    pages = dict(x.split('=') for x in args.pages)
    reviews = {json.loads(x)['table'].split(']')[0] + ']': json.loads(x) for x in (ROOT / args.review).read_text().splitlines() if x}
    heads = list(re.finditer(r'^(?:#+\s*)?\[\s*(별표\s*\d+(?:의\d+)?)\s*\]\s*(.*)$', text, re.M))
    rows = []
    for h, head in enumerate(heads):
        byl = re.sub(r'\s', '', head.group(1))
        start, end = head.start(), heads[h + 1].start() if h + 1 < len(heads) else len(text)
        rv = reviews.get(f'[{byl}]')
        if byl not in pages or not rv or rv['result'] != 'match':
            continue
        section = text[start:end]
        caption = clean(next((m.group(1) for m in re.finditer(r'<u>(.*?)</u>', section)), head.group(2)))
        revised = re.search(r'<\s*(개정|신설|제정)\s*(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?)\s*>', head.group(0))
        tables = list(TABLE.finditer(section))
        for i, t in enumerate(tables):
            before = section[tables[i - 1].end() if i else head.end() - start:t.start()]
            after = section[t.end():tables[i + 1].start() if i + 1 < len(tables) else len(section)]
            unit = re.findall(r'\(\s*단위\s*[:：]\s*([^)]*)\)', before)
            labels, body = table_rows(t.group(0))
            conditions = [clean(x) for x in before.split('\n') if clean(x) and not re.match(r'\(\s*단위', clean(x))
                          and clean(x) != caption]
            notes = [clean(x) for x in after.split('\n') if clean(x) and not re.fullmatch(r'#+', x.strip())]
            span = [start + t.start(), start + t.end()]
            for n, cells in enumerate(body):
                rows.append({
                    'id': f"{args.source_id}:{byl}:t{i}:r{n}", 'doc_id': args.source_id,
                    'title': f"{rec['issuer']['name']} {args.rule} [{byl}] {caption}", 'table': i,
                    'caption': [f'[{byl}] {head.group(2).strip()}', caption], 'columns': labels,
                    'unit': clean(unit[-1]) if unit else None,
                    'revised': ' '.join(revised.groups()) if revised else None,
                    'statute_effective_date': None, 'rule_effective_date': rec['dates'].get('effective_from'),
                    'section': None, 'cells': cells, 'repeated': [], 'joined': [],
                    'conditions': conditions, 'notes': notes,
                    'canonical': {'institution': rec['issuer']['name'], 'issuer_id': rec['issuer']['id'],
                                  'rule': args.rule, 'byl': byl, 'caption': caption,
                                  'revised': ' '.join(revised.groups()) if revised else None},
                    'source': {'snapshot_id': manifest['snapshot_id'], 'source_id': args.source_id,
                               # the document id A's ingester gives this record (build_workspace_index.api_snapshot)
                               'index_doc_id': f"{args.source_id.rsplit(':', 1)[0]}:{rec['version']}",
                               'content_hash': rec['content_hash'], 'char_span': span,
                               'raw': rec['raw'], 'raw_sha256': rec['raw_sha256'], 'pdf_page': int(pages[byl])},
                    'review': 'ai_visual_review_match', 'review_ref': str(args.review),
                    'review_issues': rv.get('issues', []),
                    'holds': rec.get('holds'), 'index': rec.get('index'), 'currency': rec.get('currency')})
            assert text[span[0]:span[1]] == t.group(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps({'rows': len(rows), 'tables': len({(r['canonical']['byl'], r['table']) for r in rows}),
                      'code_sha256': CODE_SHA256}, ensure_ascii=False))


if __name__ == '__main__':
    main()

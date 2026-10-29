"""Throwaway C-007: assemble the six-type pilot spec from assistant-authored items (AI), spans cut from the source."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/eomgyuhyeon/사업/trid')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_qa import SOURCE, passages  # noqa: E402

ITEMS = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])
template = json.loads(Path('/tmp/c007/spec/template.json').read_text())

wanted = set()
for it in ITEMS['items']:
    wanted |= {it['doc']} if 'doc' in it else {p[0] for p in it.get('parts', [])}
wanted |= {t['doc'] for t in ITEMS.get('tables', {}).values()}
docs = {}
with SOURCE.open() as f:
    for line in f:
        d = json.loads(line)
        if d['id'] in wanted:
            docs[d['id']] = d
            if len(docs) == len(wanted):
                break
assert len(docs) == len(wanted), wanted - set(docs)
cut = {d: passages(doc['text']) for d, doc in docs.items()}


def span(doc, p, start, end):
    text = cut[doc][p]
    i = text.find(start)
    j = text.find(end, i)
    if i < 0 or j < 0:
        raise ValueError(f'{doc} p{p}: span {start!r}..{end!r} not found')
    return text[i:j + len(end)]


NUM = re.compile(r'^[△\-]?[\d,.]+%?$')
tables = {}
for key, t in ITEMS.get('tables', {}).items():
    lines = docs[t['doc']]['text'].split('\n')
    k = next(n for n, l in enumerate(lines) if l.strip().startswith(t['first_row']))
    rows = []
    for l in lines[k:k + t['rows']]:
        toks = l.split()
        n = next(n for n in range(len(toks)) if all(NUM.match(x) for x in toks[n:]))
        rows.append([' '.join(toks[:n])] + toks[n:])
    tables[key] = dict(doc=t['doc'], note=t['note'], caption=t['caption'], header=t['header'], rows=rows)

items = []
for it in ITEMS['items']:
    it = dict(it)
    for field in ('conclusion', 'evidence', 'condition', 'answer'):
        if isinstance(it.get(field), list):   # [start, end] markers
            it[field] = span(it['doc'], it['passage'], *it[field])
    items.append(it)

dev = sorted({d for d in docs if int(hashlib.sha256(('c007-dev' + d).encode()).hexdigest(), 16) % 10 == 0})
spec = dict(template, items=items, tables=tables, dev_docs=dev)
OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
for key, t in tables.items():
    text = '\n'.join([t['caption']] + [' | '.join(t['header'])] + [' | '.join(r) for r in t['rows']])
    ans = [i['answer'] for i in items if i.get('table') == key][0]
    print(key, 'rows', len(t['rows']), 'answer occurrences in table', text.count(ans))
print('items', len(items), 'docs', len(docs), 'dev docs', dev)

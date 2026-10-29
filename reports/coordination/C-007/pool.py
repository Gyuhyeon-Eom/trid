"""Throwaway C-007 candidate pools for the six task types (korea.kr docs allowed for training only)."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/eomgyuhyeon/사업/trid')
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_qa import EXCLUDE, SOURCE, passages, read_jsonl  # noqa: E402

excluded = {d['id'] for p in EXCLUDE for d in read_jsonl(p)}
for w in ('/tmp/c006/qa-01/selected.jsonl', '/tmp/c006/qa-02/selected.jsonl'):
    excluded |= {d['id'] for d in read_jsonl(Path(w))}
spec = json.loads((ROOT / 'reports/coordination/C-002/pilot-spec.json').read_text())
excluded |= {i['doc'] for i in spec['items'] if 'doc' in i} | {p[0] for i in spec['items'] for p in i.get('parts', [])}
excluded |= {t['doc'] for t in spec['tables'].values()} | {d for s in spec['distractors'].values() for d, _ in s}
dev_text = ''
for p in ('/tmp/c004/rag310.json', '/tmp/c004/pilot30.json', '/tmp/c004/probe40.json',
          str(ROOT / 'reports/coordination/C-003/prompts.json')):
    dev_text += ''.join(r['prompt'] for r in json.load(open(p))['results'])
dev_text = re.sub(r'\s+', '', dev_text)

PATTERNS = {
    'condition': re.compile(r'(경우에는|경우에만|요건|자격|이상인|이하인|해당하는 경우|충족)'),
    'exception': re.compile(r'(제외|다만|예외|적용하지 않|해당하지 않)'),
    'numeric_change': re.compile(r'\d[\d,.]*\s*(만원|억원|원|명|%|개월|년|일|시간)'),
    'explanation': re.compile(r'(위해|때문|목적|취지|배경|함으로써)'),
}


def table_block(text):
    lines = text.split('\n')
    best = None
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and 0 < len(lines[j].strip()) <= 25:
            j += 1
        block = lines[i:j]
        if len(block) >= 12 and sum(bool(re.search(r'\d', x)) for x in block) >= 6:
            best = best or '\n'.join(lines[max(0, i - 3):j + 1])
        i = max(j, i + 1)
    return best


pools = {t: [] for t in ('condition', 'exception', 'table', 'combine', 'numeric_change', 'explanation')}
scanned = skipped_dev = 0
with SOURCE.open() as f:
    for line in f:
        doc = json.loads(line)
        h = int(hashlib.sha256(doc['id'].encode()).hexdigest(), 16)
        if h % 8 or doc['id'] in excluded:
            continue
        cut = passages(doc['text'])
        if not cut:
            continue
        scanned += 1
        if any(re.sub(r'\s+', '', p)[:80] in dev_text for p in cut):
            skipped_dev += 1
            continue
        kind = ['condition', 'exception', 'table', 'combine', 'numeric_change', 'explanation'][h // 8 % 6]
        record = dict(doc=doc['id'], title=doc['title'], date=doc.get('date'), source=doc['source'],
                      passages={i: p for i, p in enumerate(cut)})
        if kind == 'table':
            block = table_block(doc['text'])
            if not block:
                continue
            record['raw_table'] = block
        elif kind == 'combine':
            if sum(bool(re.search(r'\d', p)) for p in cut) < 2:
                continue
        else:
            hits = [i for i, p in enumerate(cut) if PATTERNS[kind].search(p)]
            if not hits:
                continue
            record['hit_passages'] = hits
        pools[kind].append(record)

out = Path('/tmp/c007/pool')
out.mkdir(parents=True, exist_ok=True)
for kind, recs in pools.items():
    recs.sort(key=lambda r: hashlib.sha256(('c007' + r['doc']).encode()).hexdigest())
    (out / f'{kind}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in recs[:400]))
print(json.dumps(dict(scanned=scanned, skipped_dev=skipped_dev, excluded=len(excluded),
                      pools={k: len(v) for k, v in pools.items()}), ensure_ascii=False))

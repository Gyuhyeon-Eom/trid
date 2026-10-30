"""Throwaway B-010: export PRISM clean rows for the overlap check and measure what kind of text the reports carry:
statute citations, rule-like clauses, table-like lines, definitions/explanations; print short excerpts."""
import gzip
import json
import re
from collections import Counter

import boto3

B = 'trid-training-148761639313'
s3 = boto3.client('s3')
m = json.loads(s3.get_object(Bucket=B, Key='corpus/state/prism/clean-v2.manifest.json')['Body'].read())
rows = []
for sh in m['shards']:
    for x in gzip.decompress(s3.get_object(Bucket=B, Key=sh['key'])['Body'].read()).decode().split('\n'):
        if x:
            r = json.loads(x)
            r['_part'] = sh['part']
            rows.append(r)
with open('/tmp/trid-corpus/overlap/prism-clean-v1.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps({'id': r['id'], 'group': r['group'], 'source': r['source'], 'text': r['text']}, ensure_ascii=False) + '\n')

LAW = re.compile(r'「[^」]{2,40}(법|령|규칙|조례|규정)」|제\d+조(의\d+)?(제\d+항)?')
RULE = re.compile(r'(하여야 한다|해야 한다|할 수 있다|아니 된다|하지 않는다|으로 한다|로 한다)[.)]?$')
DEF = re.compile(r'(이?란|라 함은)\s.{3,80}(을|를) (말한다|의미한다)|(을|를) 의미한다|으로 정의')
TABLE = re.compile(r'^(\S+\s+){3,}\S+$')
NUMERIC = re.compile(r'\d')
stats, examples = [], {'law': [], 'rule': [], 'def': [], 'table': []}
for r in rows:
    if r['_part'] != 'docs':
        continue
    lines = [l for l in r['text'].split('\n') if l.strip()]
    law = sum(bool(LAW.search(l)) for l in lines)
    rule = sum(bool(RULE.search(l)) for l in lines)
    df = sum(bool(DEF.search(l)) for l in lines)
    table = sum(bool(TABLE.match(l)) and sum(bool(NUMERIC.search(w)) for w in l.split()) >= 3 for l in lines)
    short = sum(len(l) < 20 for l in lines)
    stats.append({'id': r['id'], 'chars': len(r['text']), 'lines': len(lines), 'law_lines': law, 'rule_lines': rule,
                  'definition_lines': df, 'table_like_lines': table, 'short_line_frac': round(short / max(1, len(lines)), 3)})
    for kind, pat in (('law', LAW), ('rule', RULE), ('def', DEF)):
        if len(examples[kind]) < 4:
            hit = next((l for l in lines if pat.search(l) and 40 < len(l) < 220), None)
            if hit:
                examples[kind].append((r['id'], r['title'][:40], hit))
    if len(examples['table']) < 3:
        hit = next((i for i, l in enumerate(lines) if TABLE.match(l) and sum(bool(NUMERIC.search(w)) for w in l.split()) >= 3), None)
        if hit is not None:
            examples['table'].append((r['id'], r['title'][:40], '\n'.join(lines[max(0, hit - 1):hit + 3])))
n = len(stats)
agg = {k: {'docs_with_any': sum(s[k] > 0 for s in stats), 'median_per_doc': sorted(s[k] for s in stats)[n // 2],
           'total': sum(s[k] for s in stats)} for k in ('law_lines', 'rule_lines', 'definition_lines', 'table_like_lines')}
agg['docs'] = n
agg['median_chars'] = sorted(s['chars'] for s in stats)[n // 2]
agg['median_short_line_frac'] = sorted(s['short_line_frac'] for s in stats)[n // 2]
agg['quality_flags'] = dict(Counter(next((f for f in r['flags'] if f.startswith('quality:')), 'quality:ok') for r in rows if r['_part'] == 'docs'))
agg['domains'] = dict(Counter((r['provenance'].get('domain') or '-').split(' > ')[0] for r in rows if r['_part'] == 'docs'))
agg['institutions_top'] = Counter(r['provenance'].get('institution') for r in rows if r['_part'] == 'docs').most_common(8)
json.dump({'aggregate': agg, 'per_doc': stats, 'examples': examples}, open('/tmp/trid-corpus/qa/prism-sample-analysis.json', 'w'),
          ensure_ascii=False, indent=1)
print(json.dumps(agg, ensure_ascii=False, indent=1))
for k, v in examples.items():
    print('==', k)
    for e in v:
        print(' ', e[0], e[1], '|', e[2][:300].replace('\n', ' / '))

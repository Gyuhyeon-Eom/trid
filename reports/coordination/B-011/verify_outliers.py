"""Throwaway B-011: check A-004's five longest press rows against the stored raw (collector item cache, view HTML,
attachment files, docs.jsonl source row) and say where the text came from."""
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import boto3

SRC = Path('/Users/eomgyuhyeon/사업/trid/data/korea-kr-briefing-source')
out = json.load(open('/Users/eomgyuhyeon/사업/trid/reports/coordination/A-004/press-clean-v3-length-outliers.json'))
ids = [r['id'].split(':')[1] for r in out['top5_by_characters']]
s3 = boto3.client('s3')
B = 'trid-training-148761639313'
rows = {}
for r in out['top5_by_characters']:
    body = gzip.decompress(s3.get_object(Bucket=B, Key=r['shard_key'])['Body'].read()).decode().split('\n')[r['shard_line_index_0based']]
    rows[r['id'].split(':')[1]] = json.loads(body)
src = {}
with open(SRC / 'docs.jsonl') as f:
    for line in f:
        m = re.search(r'"id":\s*"korea-kr-press-(\d+)"', line[:200])
        if m and m.group(1) in ids:
            src[m.group(1)] = json.loads(line)
report = []
for nid in ids:
    item = json.load(open(SRC / f'raw/items/{nid}.json'))
    row, doc = rows[nid], src.get(nid, {})
    view = (SRC / f'raw/view/{nid}.html')
    files = sorted(p.name for p in (SRC / 'raw/files').glob(f"{item['file']['file_id']}*")) if item.get('file') else []
    body = item.get('body') or ''
    ctrl = sum(1 for c in body if ord(c) < 32 and c not in '\n\t')
    lines = [l for l in row['text'].split('\n') if l.strip()]
    numeric = sum(1 for l in lines if sum(ch.isdigit() for ch in l) >= max(3, len(l) // 3))
    report.append({
        'id': f'korea-kr-briefing:{nid}', 'title': item['title'],
        'attachments': [(a['name'], a['ext']) for a in item['attachments']],
        'body_source': item['body_source'], 'chosen_file': item.get('file', {}).get('name'),
        'chosen_file_is_press_release': bool(re.search(r'보도|조간|석간|배포|\[\d+\.\d+', item.get('file', {}).get('name', ''))),
        'other_attachments_not_used': [a['name'] for a in item['attachments'] if a['file_id'] != item.get('file', {}).get('file_id')],
        'raw_file_cached': files, 'view_html_bytes': view.stat().st_size if view.exists() else None,
        'item_body_chars': len(body), 'item_body_control_chars': ctrl,
        'source_doc_text_sha256': hashlib.sha256((doc.get('text') or '').encode()).hexdigest() if doc else None,
        'clean_meta_source_doc_sha256': row['meta'].get('source_doc_sha256'),
        'source_doc_sha256_field': doc.get('sha256'),
        'clean_chars': len(row['text']), 'clean_lines': len(lines),
        'numeric_heavy_line_frac': round(numeric / max(1, len(lines)), 3),
        'clean_head': row['text'][:200], 'flags': row['flags']})
json.dump(report, open('/tmp/trid-corpus/qa/press-length-outliers-verify.json', 'w'), ensure_ascii=False, indent=1)
for r in report:
    print(json.dumps({k: v for k, v in r.items() if k != 'clean_head'}, ensure_ascii=False))
    print('   head:', r['clean_head'][:150].replace('\n', ' / '))

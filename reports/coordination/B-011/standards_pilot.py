"""Throwaway B-011: store the three 공공데이터 공통표준 CSVs (data.go.kr 15156379/15156439/15156442, 이용허락범위 제한
없음, 8차 2025-11) as write-once raw and check their internal relations (term -> domain, term words -> word list,
code-valued terms -> 행정표준코드명), units/formats and revision markers."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/Users/eomgyuhyeon/사업/trid/scripts')
import collect_assembly_minutes as am  # noqa: E402

D = Path('/tmp/trid-corpus/inv/dl')
FILES = {'terms': ('15156379', '행정안전부_공공데이터 공통표준용어_20251101.csv'),
         'words': ('15156439', '행정안전부_공공데이터 공통표준단어_20251101.csv'),
         'domains': ('15156442', '행정안전부_공공데이터 공통표준도메인_20251101.csv')}
s3 = am.s3_client()
rows, raw = {}, {}
for kind, (pk, name) in FILES.items():
    body = (D / name).read_bytes()
    key = f'corpus/raw/public-data-standards/{pk}/{name}'
    sha, key = am.put_raw_keep(s3, key, body)
    raw[kind] = {'dataset': f'https://www.data.go.kr/data/{pk}/fileData.do', 'key': key, 'sha256': sha, 'bytes': len(body)}
    rows[kind] = list(csv.DictReader(body.decode('utf-8-sig').splitlines()))
terms, words, domains = rows['terms'], rows['words'], rows['domains']
domain_names = {d['공통표준도메인명'] for d in domains}
word_names = {w['공통표준단어명'] for w in words}
word_abbr = {w['공통표준단어영문약어명'] for w in words}
checks = Counter()
missing_domain, odd_abbr = [], []
for t in terms:
    checks['terms'] += 1
    if t['공통표준도메인명'] in domain_names:
        checks['term_domain_found'] += 1
    else:
        missing_domain.append((t['공통표준용어명'], t['공통표준도메인명']))
    parts = t['공통표준용어영문약어명'].split('_')
    if all(p in word_abbr for p in parts):
        checks['term_abbr_all_words_found'] += 1
    else:
        odd_abbr.append((t['공통표준용어명'], t['공통표준용어영문약어명']))
    if t['행정표준코드명'].strip():
        checks['term_with_admin_code'] += 1
    if t['개정구분명(폐기 또는 변경)'].strip():
        checks[f"term_revision:{t['개정구분명(폐기 또는 변경)'].strip()}"] += 1
units = Counter(d['단위'] for d in domains if d['단위'].strip())
rounds = Counter(re.match(r'\d+차', t['제정차수']).group(0) if re.match(r'\d+차', t['제정차수']) else t['제정차수'] for t in terms)
examples = [t for t in terms if t['행정표준코드명'].strip()][:5]
out = {'raw': raw, 'rows': {k: len(v) for k, v in rows.items()}, 'checks': dict(checks),
       'missing_domain_examples': missing_domain[:10], 'abbr_not_in_word_list_examples': odd_abbr[:10],
       'domain_units': dict(units), 'enactment_rounds': dict(rounds),
       'admin_code_examples': [{k: e[k] for k in ('공통표준용어명', '공통표준도메인명', '허용값', '행정표준코드명', '소관기관명')} for e in examples],
       'licence': '공공데이터포털 이용허락범위 제한 없음 (dataset pages, read 2026-10-30)',
       'revision': '8차(2025-11) 기준; 파일명 20251101; 포털 수정일 2025-12-17 (15156379)'}
s3.put_object(Bucket=am.BUCKET, Key='corpus/state/public-data-standards/raw-v1.manifest.json',
              Body=json.dumps(out, ensure_ascii=False, indent=1).encode())
json.dump(out, open('/tmp/trid-corpus/qa/public-data-standards-check.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'raw'}, ensure_ascii=False, indent=1)[:4000])

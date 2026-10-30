"""Institution public rules (사규 posted on official sites / data.go.kr) as a separate search-index candidate.

Same snapshot layout and record fields as API-S3-001 (reports/coordination/HYBRID-RAG-001-SCHEMA.md) so A's ingester reads
it, plus fields that keep the rule out of evaluation and training:
  holds        ['evaluation', 'training']: an institution's own rule may overlap the 852 unsigned reserved families
  index        'institution_rules_separate': not to be mixed into the evaluation-run index
  currency     the posted version only; nothing here is called current
The source rows come from a guides version (B's data.go.kr file collection), whose text is already PII-redacted
(ko-pii); issuer ids come from the official ALIO registry copy (data/corpus-inventory/alio-agencies.json, no request).
No rule is labelled the institution's non-public internal data: it is a publicly posted rule.

  .venv/bin/python scripts/build_institution_rule_snapshot.py --guides guides-v8 \
      --doc data.go.kr:15110681:file:30bc63ea4a81b527 --snapshot-id inst-rule-lh-pay-001 --out runs/search-snapshots/inst-rule-lh-pay-001
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.korean_quality import PII_HINT  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
REGISTRY = ROOT / 'data/corpus-inventory/alio-agencies.json'


def korean_date(text, label):
    m = re.search(label + r'\s*[:：]\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', text)
    return f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}' if m else None


def record(row, agencies, state):
    """One snapshot record. A rule without '확정일/시행일' labels in its body (board postings, B-027: 한국부동산원) takes its
    effective date from the row provenance (the last 부칙's 시행일, read by B and described in date_type). A 집행내역 row
    is kept apart from rules: kind 'expenditure_record', basis 'actual_spend', its date is the spending month."""
    text = unicodedata.normalize('NFC', row['text']).strip()
    prov = row['provenance']
    inst = prov['institution']
    agency = next((a for a in agencies if a['apbaNa'] == inst), None)
    spend = '집행내역' in (prov.get('doc_type') or '')
    number = None if spend else re.search(r'규정\s*제\s*(\d+)\s*호', text)
    published, effective = korean_date(text, r'확\s*정\s*일'), korean_date(text, r'시\s*행\s*일')
    effective_kind = '시행일(규정 본문)'
    if not spend and not effective and (prov.get('date_type') or '').startswith('마지막 부칙'):
        effective, effective_kind = prov['date'], prov['date_type']
    if spend:
        currency = f"{prov['date']} 집행분 공개 내역(실제 지출). 규정상 기준이 아니며, 이후 정정 여부는 확인하지 않음"
    else:
        currency = (f"{published or effective or prov['date']} 기준 게시본. 이후 개정 여부는 확인하지 않았고 현행이라고 표시하지 않음")
    return {
        'source_id': row['id'], 'version': f'규정 제{number.group(1)}호' if number else row['sha256'][:16],
        # 'institution_rule', not 'rule': A's ingester maps 'rule' to 행정규칙 (a ministry's administrative rule)
        'content_hash': hashlib.sha256(text.encode()).hexdigest(), 'kind': 'expenditure_record' if spend else 'institution_rule',
        'basis': 'actual_spend' if spend else 'rule_text',
        'title': row['title'] + (f' (규정 제{number.group(1)}호)' if number else ''),
        'issuer': {'id': agency['apbaId'] if agency else None, 'name': inst, 'id_scheme': 'alio_apbaId' if agency else None,
                   'type': agency['typeNa'] if agency else prov['institution_type']},
        'applies_to': {'scope': 'institution', 'ids': [agency['apbaId']] if agency else [], 'note': prov['applies_to']},
        'dates': {'published': published, 'effective_from': None if spend else effective, 'effective_to': None,
                  'spending_period': prov['date'] if spend else None, 'retrieved_at': row['collected_at'][:10]},
        'date_kinds': {'published': '확정일(규정 본문)', 'effective_from': effective_kind,
                       'spending_period': prov['date_type'] if spend else None, 'retrieved_at': '수집일'},
        'rights': {'license': row['license'], 'url': row['source'], 'restrictions': None},
        'location': {'url': state.get('provider_url') or row['source'], 'page': None, 'article_path': [],
                     'char_start': 0, 'char_end': len(text),
                     'data_go_kr': row['source'] if 'data.go.kr' in row['source'] else None},
        'text': text, 'structured_facts': [],
        'holds': ['evaluation', 'training'], 'index': 'institution_rules_separate',
        'currency': currency,
        'pii': {'check': 'ko-pii redaction in the guides version + trid.korean_quality.PII_HINT', 'hits': len(PII_HINT.findall(text)),
                'redactions': row['meta'].get('pii_redactions')},
        'table_check': row['meta'].get('table_check'), 'status': 'search_candidate',
        'collection': 'public_file_download (data.go.kr page / institution board), not an API call',
        'raw': row['raw'], 'raw_sha256': row['raw_sha256'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--guides', required=True)
    p.add_argument('--doc', nargs='+', required=True)
    p.add_argument('--snapshot-id', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--producer', default='B', help="task id recorded in the manifest, e.g. 'B/B-027'")
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; snapshots are never overwritten')
    rows = {json.loads(x)['id']: json.loads(x) for part in ('documents', 'held')
            for x in (ROOT / 'data/public-data' / args.guides / f'{part}.jsonl').read_text().splitlines() if x}
    states = {}
    for path in (ROOT / 'data/public-data').glob('files-state*.jsonl'):
        for x in path.read_text().splitlines():
            s = json.loads(x) if x else None
            if s and s['status'] == 'stored':
                states[s['sha256']] = s
    agencies = json.loads(REGISTRY.read_text())['agencies']
    recs = [record(rows[d], agencies, states.get(rows[d]['raw_sha256'], {})) for d in args.doc]
    if any(r['pii']['hits'] for r in recs):
        raise SystemExit(f"PII hint in {[r['source_id'] for r in recs if r['pii']['hits']]}: review before handing over")
    (out / 'normalized').mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in recs).encode()
    (out / 'normalized/records-00000.jsonl').write_bytes(body)
    manifest = {'snapshot_id': args.snapshot_id, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'producer': args.producer, 'code_sha256': CODE_SHA256,
                'kinds': dict(Counter(r['kind'] for r in recs)),
                'sources': [{'provider': 'data.go.kr file data / institution 사규 board', 'guides_version': args.guides,
                             'records': len(recs), 'collection': 'public_file_download, not an API call'}],
                'files': [{'path': 'normalized/records-00000.jsonl', 'sha256': hashlib.sha256(body).hexdigest(), 'records': len(recs)}],
                'records_total': len(recs), 'holds': ['evaluation', 'training'], 'index': 'institution_rules_separate',
                'registry': {'path': str(REGISTRY.relative_to(ROOT)), 'sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest()},
                'notes': 'publicly posted institution rules; not non-public internal data; nothing labelled current'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('snapshot_id', 'records_total', 'files')}, ensure_ascii=False, indent=1))
    for r in recs:
        print(r['source_id'], r['version'], r['issuer'], r['dates'], r['pii'], r['table_check'])


if __name__ == '__main__':
    main()

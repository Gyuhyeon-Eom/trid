"""API-S3-001 snapshot in the HYBRID-RAG-001-SCHEMA.md layout, built from 법제처 DRF Open API responses B already holds.

No API call is made here. The input is the existing collection data/law-ko-source (lawService.do XML responses, one
file per rule, collected earlier with an OC key that is not in this environment); the manifest says so
(collection: existing_collection_no_new_api_call). Selection: 행정규칙 (target admrul) whose title matches --title-regex,
minus every row of the reserved-evaluation overlap list (B-022 v3-overlap-index-separation.jsonl; none of its admrul
rows passed the B-023 span check), so nothing that shares non-statute text with reserved evaluation families is
handed to the search index.

Snapshot layout (one directory = one future S3 prefix):
  manifest.json                    snapshot_id, producer, sources, files [{path, sha256, records|bytes}], records_total,
                                   delta against --previous (new / changed / unchanged / removed by source_id)
  normalized/records-00000.jsonl   one record per rule (schema §2); the text is the body as parsed from the response
  raw/admrul/<일련번호>.xml         the API response as received (audit only; it names the responsible official and
                                   phone number, which are not copied into normalized records)
Record fields beyond the schema: date_kinds, pii {check, hits}, status, collection, listed_current_at_retrieval.
A record whose text matches the PII hint pattern is kept out of normalized/ (status held_pii) and only counted.

  .venv/bin/python scripts/build_api_snapshot.py --snapshot-id api-drf-admrul-travel-001 \
      --out runs/api-snapshots/api-drf-admrul-travel-001 [--previous runs/api-snapshots/<older>]
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.korean_quality import PII_HINT  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
SOURCE = ROOT / 'data/law-ko-source'
OVERLAP = ROOT / 'reports/coordination/B-022/v3-overlap-index-separation.jsonl'
SPAN_CHECK = ROOT / 'reports/coordination/B-023/overlap-102-span-check.jsonl'
LICENSE = 'KR-Copyright-Act-Art7 (law.go.kr)'
ENDPOINT = 'http://www.law.go.kr/DRF/lawService.do?target=admrul'


def field(xml, name):
    m = re.search(rf'<{name}>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{name}>', xml, re.S)
    return m.group(1).strip() if m else None


def iso(d):
    return f'{d[:4]}-{d[4:6]}-{d[6:8]}' if d and re.fullmatch(r'\d{8}', d) else None


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def excluded_ids():
    """Overlap rows still separated after the B-023 span check."""
    ok = {json.loads(x)['corpus_id'] for x in SPAN_CHECK.read_text().splitlines() if x
          and json.loads(x)['decision'] == 'index_ok_all_spans_statute'}
    return {json.loads(x)['corpus_id'] for x in OVERLAP.read_text().splitlines() if x} - ok


def record(doc, xml, retrieved_at):
    text = unicodedata.normalize('NFC', doc['text']).strip()
    serial = field(xml, '행정규칙일련번호')
    published, effective = iso(field(xml, '발령일자')), iso(field(xml, '시행일자'))
    pii_hits = len(PII_HINT.findall(text))
    return {
        'source_id': f"law-go-kr:admrul:{field(xml, '행정규칙ID')}",
        'version': serial,
        'content_hash': sha256_bytes(text.encode()),
        'kind': 'rule',
        'title': field(xml, '행정규칙명'),
        'issuer': {'id': field(xml, '소관부처코드'), 'name': field(xml, '소관부처명'), 'id_scheme': '행정기관코드'},
        'applies_to': {'scope': 'unknown', 'ids': []},
        'dates': {'published': published, 'effective_from': effective, 'effective_to': None, 'retrieved_at': retrieved_at},
        'date_kinds': {'published': '발령일자', 'effective_from': '시행일자', 'retrieved_at': 'raw response file time (UTC)'},
        'rights': {'license': LICENSE, 'url': 'https://www.law.go.kr/LSW/infoUse.do', 'restrictions': None},
        'location': {'url': doc['source'], 'page': None, 'article_path': [], 'char_start': 0, 'char_end': len(text)},
        'text': text,
        'structured_facts': [],
        'rule_kind': field(xml, '행정규칙종류'), 'amendment': field(xml, '제개정구분명'), 'issue_no': field(xml, '발령번호'),
        'listed_current_at_retrieval': field(xml, '현행여부') == 'Y',
        'pii': {'check': 'trid.korean_quality.PII_HINT', 'hits': pii_hits},
        'status': 'held_pii' if pii_hits else 'ok',
        'collection': 'existing_collection_no_new_api_call',
    }


def build(args):
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; snapshots are never overwritten')
    title = re.compile(args.title_regex)
    skip = excluded_ids()
    rows, held, excluded, raw_files = [], 0, 0, []
    with open(SOURCE / 'docs.jsonl') as f:
        for line in f:
            if not line.startswith('{"id": "law-go-kr:admrul:'):
                continue
            doc = json.loads(line)
            if not title.search(doc['title']):
                continue
            if doc['id'] in skip:
                excluded += 1
                continue
            serial = re.search(r'ID=(\d+)', doc['source']).group(1)
            raw = SOURCE / f'raw/admrul/{serial}.xml'
            xml = raw.read_text(encoding='utf-8')
            retrieved = datetime.fromtimestamp(raw.stat().st_mtime, timezone.utc).strftime('%Y-%m-%d')
            r = record(doc, xml, retrieved)
            raw_files.append(raw)
            if r['status'] != 'ok':
                held += 1
                continue
            rows.append(r)
    rows.sort(key=lambda r: r['source_id'])
    (out / 'normalized').mkdir(parents=True)
    (out / 'raw/admrul').mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode()
    (out / 'normalized/records-00000.jsonl').write_bytes(body)
    files = [{'path': 'normalized/records-00000.jsonl', 'sha256': sha256_bytes(body), 'records': len(rows), 'bytes': len(body)}]
    for raw in sorted(raw_files):
        dest = out / 'raw/admrul' / raw.name
        shutil.copyfile(raw, dest)
        files.append({'path': f'raw/admrul/{raw.name}', 'sha256': sha256_bytes(dest.read_bytes()), 'bytes': dest.stat().st_size})
    delta = None
    if args.previous:
        prev_dir = ROOT / args.previous
        prev = {}
        for f in json.loads((prev_dir / 'manifest.json').read_text())['files']:
            if f['path'].startswith('normalized/'):
                for x in (prev_dir / f['path']).read_text().splitlines():
                    p = json.loads(x)
                    prev[p['source_id']] = (p['version'], p['content_hash'])
        cur = {r['source_id']: (r['version'], r['content_hash']) for r in rows}
        delta = {'previous': args.previous, 'new': sorted(cur.keys() - prev.keys()),
                 'changed': sorted(k for k in cur.keys() & prev.keys() if cur[k] != prev[k]),
                 'unchanged': sum(1 for k in cur.keys() & prev.keys() if cur[k] == prev[k]),
                 'removed': sorted(prev.keys() - cur.keys())}
    manifest = {
        'snapshot_id': args.snapshot_id, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'producer': 'B/API-S3-001', 'code_sha256': CODE_SHA256,
        'sources': [{'provider': '법제처 국가법령정보센터 Open API (DRF)', 'endpoint': ENDPOINT, 'records': len(rows),
                     'license': LICENSE, 'collection': 'existing_collection_no_new_api_call',
                     'retrieved_at': 'per record dates.retrieved_at (raw response file time)',
                     'source_dir': str(SOURCE.relative_to(ROOT))}],
        'selection': {'target': 'admrul', 'title_regex': args.title_regex, 'excluded_reserved_overlap': excluded,
                      'held_pii': held, 'overlap_list': str(OVERLAP.relative_to(ROOT)), 'span_check': str(SPAN_CHECK.relative_to(ROOT))},
        'files': files, 'records_total': len(rows), 'raw_bytes': sum(f['bytes'] for f in files[1:]), 'delta': delta,
        'notes': 'Not a new API collection. No value is called current; listed_current_at_retrieval is the 현행여부 flag '
                 'of the response at retrieval time. Public statutory material, not an institution\'s non-public data.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def verify(directory):
    """Re-read every file listed in the manifest and compare sha256 (the readback check used after a copy)."""
    d = ROOT / directory
    manifest = json.loads((d / 'manifest.json').read_text())
    bad = [f['path'] for f in manifest['files'] if sha256_bytes((d / f['path']).read_bytes()) != f['sha256']]
    return {'snapshot_id': manifest['snapshot_id'], 'manifest_sha256': sha256_bytes((d / 'manifest.json').read_bytes()),
            'files': len(manifest['files']), 'mismatched': bad}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snapshot-id')
    p.add_argument('--out')
    p.add_argument('--previous')
    # '급여비용' (요양급여비용 등) contains '여비' but is not travel expense
    p.add_argument('--title-regex', default='출장|(?<!급)여비')
    p.add_argument('--verify', help='only verify an existing snapshot directory')
    args = p.parse_args()
    if args.verify:
        result = verify(args.verify)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 1 if result['mismatched'] else 0
    if not (args.snapshot_id and args.out):
        p.error('--snapshot-id and --out are required unless --verify')
    m = build(args)
    print(json.dumps({k: m[k] for k in ('snapshot_id', 'records_total', 'raw_bytes', 'selection')} |
                     {'delta': {k: (len(v) if isinstance(v, list) else v) for k, v in (m['delta'] or {}).items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

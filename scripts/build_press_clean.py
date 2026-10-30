"""정책브리핑 보도자료 -> corpus contract clean candidate (docs/CORPUS_CONTRACT.md), versioned, from the existing local
collection data/korea-kr-briefing-source/docs.jsonl (read only; never rewritten).

Per document: NFC -> trid.korean_quality.redact (current version, sha recorded) -> exact duplicate drop within the
source (NFC, whitespace removed) -> quarantine of development-evaluation sources (ids or exact text of documents used
by EVAL_SOURCES / frozen QA sets), of bodies that the collector cache shows were taken from an attachment other than
the press release (or from attachment-like files only), and of font-garbled extractions -> split by
sha256("split:"+group). Rows already used by earlier trid training splits keep a 'prior_trid_training' flag (they are
not removed; A decides). Shards are uploaded with S3 SHA-256 checksums, re-verified, then the manifest and
<version>.done are written.

  python scripts/build_press_clean.py --version clean-v1
"""
import argparse
import base64
import datetime
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUCKET = 'trid-training-148761639313'
SOURCE = 'korea-kr-briefing'
INPUT = ROOT / 'data/korea-kr-briefing-source/docs.jsonl'
# Development-evaluation material (A's EVAL_SOURCES in prepare_fineweb2_ko.py and the CPT exclusion list).
EVAL_FILES = ('data/korean-rag-eval-v1/test.jsonl', 'data/institution-qa-frozen-01', 'data/public-document-sft-source')
PRIOR_TRAINING_FILES = ('mix-v3-dry/train.jsonl', 'trid-cpt-kanana-100m/train.jsonl', 'public-document-sft-01/train.jsonl',
                        'public-dialogue-shared-ctx-01/train.jsonl') + tuple(f'public-dialogue-mix-0{i}/train.jsonl' for i in range(1, 7))
PRESS_ID = re.compile(r'korea-kr-press-(\d+)')
SPLITS = (('test', 50), ('valid', 100))
SHARD_ROWS = 20000
ITEMS = ROOT / 'data/korea-kr-briefing-source/raw/items'  # collector cache: which attachment the body came from
# File-name cues of the press release itself vs. other attachments (B-011: A-004's five longest rows were a statistics
# yearbook, survey results, a research report, a white-paper appendix and an arbitration award, not the releases).
PRESS_FILE = re.compile(r'보도자료|보도 자료|조간|석간|즉시\s*보도|배포|\[\d{1,2}\.\d{1,2}|^\(?\d{2}[.-]\d{2}|보도참고')
ATTACHMENT_FILE = re.compile(r'별첨|붙임|첨부|참고자료|부록|백서|보고서|통계|연보|Award|판정|가이드|매뉴얼|안내서|자료집')


def body_origin(news_id):
    """(origin, control-character ratio of the cached body) from the collector cache. origin: html_body, press_file,
    other_attachment (a press-release file existed but another attachment was used), attachment_name (only
    attachment-like files), unclear_name, no_cache."""
    path = ITEMS / f'{news_id}.json'
    if not path.exists():
        return 'no_cache', 0.0
    d = json.loads(path.read_text())
    body = d.get('body') or ''
    head = body[:200000]
    ctrl = sum(1 for c in head if ord(c) < 32 and c not in '\n\t\r') / max(1, len(head))
    chosen = (d.get('file') or {}).get('name') or ''
    if d.get('body_source') in (None, 'html', 'view', 'page'):
        return 'html_body', ctrl
    if PRESS_FILE.search(chosen):
        return 'press_file', ctrl
    if any(PRESS_FILE.search(a.get('name') or '') for a in d.get('attachments') or []):
        return 'other_attachment', ctrl
    return ('attachment_name' if ATTACHMENT_FILE.search(chosen) else 'unclear_name'), ctrl


def split_of(group):
    bucket = int(hashlib.sha256(f'split:{group}'.encode()).hexdigest()[:8], 16) % 10000
    return next((name for name, upper in SPLITS if bucket < upper), 'train')


def dedup_key(text):
    return hashlib.sha256(''.join(text.split()).encode()).hexdigest()


def files_of(name):
    path = ROOT / name
    return [path] if path.is_file() else sorted(path.rglob('*.jsonl')) if path.exists() else []


def eval_exclusions():
    """News ids mentioned anywhere in the evaluation files, and exact-text keys of their long string fields."""
    ids, keys, used = set(), set(), []
    for name in EVAL_FILES:
        for f in files_of(name):
            used.append(str(f.relative_to(ROOT)))
            for line in f.read_text().split('\n'):  # not splitlines(): U+2028/U+0085 occur inside JSON strings
                ids.update(PRESS_ID.findall(line))
                if line.strip():
                    row = json.loads(line)
                    keys.update(dedup_key(v) for v in row.values() if isinstance(v, str) and len(v) >= 200)
    return ids, keys, used


def prior_training_ids():
    ids = set()
    for name in PRIOR_TRAINING_FILES:
        path = ROOT / 'data' / name
        if path.exists():
            with open(path) as f:
                for line in f:
                    m = re.search(r'"id":\s*"korea-kr-press-(\d+)', line)
                    if m:
                        ids.add(m.group(1))
    return ids


def clean(args):
    line, built_at = args
    from trid.korean_quality import redact
    d = json.loads(line)
    news_id = d['id'].rsplit('-', 1)[-1]
    text, pii = redact(unicodedata.normalize('NFC', d.get('text') or ''))
    origin, ctrl = body_origin(news_id)
    m = re.search(r'이 자료는\s*(.+?)의\s*보도자료', d.get('attribution') or '')
    group = f'korea-kr:{news_id}'
    prov = {'institution_type': '중앙부처', 'doc_type': '보도자료', 'published_date': d.get('date')}
    if m:
        prov['institution'] = m.group(1).strip()
    return {'id': f'{SOURCE}:{news_id}', 'group': group, 'kind': 'press_release', 'title': d.get('title') or '',
            'text': text, 'source': d.get('source'), 'license': d.get('license'), 'attribution': d.get('attribution'),
            'date': d.get('date'), 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': built_at,
            'extractor': f"korea-kr-collector:{d.get('body_source', 'unknown')}", 'flags': ['pii_redacted'] if pii else [],
            'split': split_of(group), 'provenance': prov,
            'meta': {'news_id': news_id, 'source_doc_sha256': d.get('sha256'), 'pii_redactions': dict(pii),
                     'body_origin': origin, 'body_control_char_ratio': round(ctrl, 5)}}, dedup_key(text), d


def put_verified(s3, key, body):
    digest = hashlib.sha256(body).digest()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ChecksumSHA256=base64.b64encode(digest).decode())
    return {'key': key, 'sha256': digest.hex(), 'bytes': len(body)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', required=True)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--reserve-quarantine', nargs='*', default=[],
                   help='JSON {doc_id: reason} or text files of doc ids (e.g. reports/coordination/C-008/quarantine-docs.txt) '
                        'whose rows go to quarantine:reserved_overlap')
    args = p.parse_args()
    reserve_q = {}
    for path in args.reserve_quarantine:
        text = (ROOT / path).read_text()
        entries = json.loads(text) if path.endswith('.json') else {x.strip(): 'listed' for x in text.split('\n') if x.strip()}
        for doc_id, reason in entries.items():
            reserve_q[doc_id.rsplit('-', 1)[-1]] = f'{path}: {reason}'
    import boto3
    from trid.korean_quality import redacted_in_url
    s3 = boto3.client('s3')
    state = f'corpus/state/{SOURCE}/{args.version}'
    try:
        s3.head_object(Bucket=BUCKET, Key=f'{state}.done')
        raise SystemExit(f'{args.version} already done; versions are never overwritten')
    except s3.exceptions.ClientError:
        pass
    built_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    eval_ids, eval_keys, eval_used = eval_exclusions()
    prior = prior_training_ids()
    lines = [line for line in INPUT.read_text().split('\n') if line.strip()]
    seen, kept, held, reasons, pii_total = set(), [], [], Counter(), Counter()
    with Pool(args.workers) as pool:
        for row, key, d in pool.imap(clean, ((line, built_at) for line in lines), chunksize=200):
            row['clean_version'] = args.version
            if key in seen:
                reasons['exact_duplicate_in_source'] += 1
                continue
            seen.add(key)
            pii_total.update(row['meta']['pii_redactions'])
            if row['meta']['news_id'] in prior:
                row['flags'].append('prior_trid_training')
            news_id = row['meta']['news_id']
            hold = []
            if news_id in eval_ids or key in eval_keys or dedup_key(d.get('text') or '') in eval_keys:
                hold.append('dev_eval_source')
            if news_id in reserve_q:
                hold.append('reserved_overlap')
                row['meta']['reserved_overlap_evidence'] = reserve_q[news_id]
            origin = row['meta']['body_origin']
            if origin in ('other_attachment', 'attachment_name', 'no_cache'):  # text is not (verifiably) the release
                hold.append(f'body_not_press_release:{origin}')
            if row['meta']['body_control_char_ratio'] > 0.002:  # font-encoding garbage from PDF extraction
                hold.append('garbled_extraction')
            if origin == 'unclear_name':
                row['flags'].append('body_file_name_unverified')
            if redacted_in_url(row['text']):  # privacy is handled (replaced); the URL itself is now damaged
                hold.append('pii_redaction_inside_url')
            row['flags'] += [f'quarantine:{h}' for h in hold]
            (held if hold else kept).append(row)
            for h in hold or ['ok']:
                reasons[h if h == 'ok' else f'quarantine:{h}'] += 1
    shards = []
    for name, rows in (('docs', kept), ('quarantine', held)):
        for i in range(0, max(len(rows), 1), SHARD_ROWS):
            chunk = rows[i:i + SHARD_ROWS]
            body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in chunk).encode(), mtime=0)
            info = put_verified(s3, f'corpus/clean/{SOURCE}/{args.version}/{name}-{i // SHARD_ROWS:05d}.jsonl.gz', body)
            info.update({'part': name, 'rows': len(chunk), 'characters': sum(len(r['text']) for r in chunk)})
            shards.append(info)
    for sh in shards:  # verify before publishing
        head = s3.head_object(Bucket=BUCKET, Key=sh['key'], ChecksumMode='ENABLED')
        if base64.b64decode(head['ChecksumSHA256']).hex() != sh['sha256'] or head['ContentLength'] != sh['bytes']:
            raise RuntimeError(f"verification failed: {sh['key']}")
    kq = ROOT / 'trid/korean_quality.py'
    manifest = {'source': SOURCE, 'clean_version': args.version, 'built_at': built_at,
                'input': str(INPUT.relative_to(ROOT)), 'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                'input_rows': len(lines), 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'korean_quality_sha256': hashlib.sha256(kq.read_bytes()).hexdigest(), 'reasons': dict(reasons),
                'pii_redactions': dict(pii_total), 'eval_files': eval_used, 'eval_ids': len(eval_ids),
                'eval_match': ('quarantine if the news id appears anywhere in an eval file, or the whitespace-free text of '
                               'the document (before or after redaction) equals a string field >= 200 chars of an eval row'),
                'reserve_quarantine_inputs': args.reserve_quarantine,
                'reserve_quarantine_ids': sorted(reserve_q),
                'prior_trid_training_rows': sum('prior_trid_training' in r['flags'] for r in kept),
                'splits': dict(Counter(r['split'] for r in kept)), 'rows': len(kept), 'quarantine_rows': len(held),
                'characters': sum(len(r['text']) for r in kept), 'shards': shards,
                'note': 'candidate for supervised use; not approved for A003; dev splits are not a final test'}
    body = json.dumps(manifest, ensure_ascii=False, indent=1).encode()
    s3.put_object(Bucket=BUCKET, Key=f'{state}.manifest.json', Body=body)
    s3.put_object(Bucket=BUCKET, Key=f'{state}.done', Body=hashlib.sha256(body).hexdigest().encode())
    print(json.dumps({k: v for k, v in manifest.items() if k != 'shards'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

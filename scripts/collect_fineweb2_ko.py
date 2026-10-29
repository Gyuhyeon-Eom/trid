"""FineWeb-2 Korean (HuggingFaceFW/fineweb-2, kor_Hang) -> corpus contract clean-v4 (docs/CORPUS_CONTRACT.md).

  clean     (AWS SageMaker CPU job) for each parquet file: use the immutable raw copy in
            s3://<bucket>/corpus/raw/fineweb2-ko/<repo path> (download the pinned revision from the HF Hub and store
            it there first only if it is missing), apply the clean pipeline in worker processes, upload gzip JSONL
            shards to corpus/clean/fineweb2-ko/<version>/ with server-verified SHA-256 checksums, write the per-file
            manifest to corpus/state/fineweb2-ko/<version>/<file>.manifest.json, re-verify every shard checksum, and
            only then publish <file>.done. Files with a done marker are skipped; an interrupted file is redone from
            scratch under the same deterministic keys, so re-runs never duplicate rows.
  aggregate (anywhere)                       sum the per-file manifests into <version>/{stats,provenance}.json and
            ATTRIBUTION.txt.

clean-v4 per document, in order: NFC -> trid.korean_quality.clean_lines -> drop if empty -> script-ratio check
(FINEWEB2 thresholds) -> drop gambling/adult-service advertising spam (spam_reason) -> licence quarantine (known
paid-report/resale domains; NOT a full licence verification) -> provisional quarantine of rows from hosts of
benchmark-reserved institutions (pending a text-overlap check) -> hold of rows on uncommon TLDs (MT/SEO risk) ->
trid.korean_quality.redact (ko-pii plus the fallbacks in that module, format-checked; mitigation, not a guarantee of
no personal data) -> exact duplicate drop within the file (NFC, whitespace removed) -> split by
sha256("split:"+group): test 0.5 %, valid 0.5 %, train otherwise. These dev splits are not an independent final test.
Cross-source dedup and evaluation decontamination belong to the tokens stage (owner A).

clean-v1 (B-002) lacked the spam rule, the reserved-host quarantine, report.kyobobook.co.kr and the redact() fix for
contacts at the start of a line. clean-v2 (B-004 pilot) had those, but redact() still skipped "02) 737-7378" and
"052- 290-3333" (two separator characters) and numbers followed by 'ㅣ', and sometimes needed a second pass. clean-v3
(B-004 pilot) fixed those but left 050 safe numbers and numbers glued to text, read a corporate registration number as
an RRN and a URL article id as an account, and had no MT/SEO hold. All stay under their own prefixes, not train-ready.

Licence: the FineWeb-2 compilation is ODC-By 1.0; that does not license the underlying web pages, which keep their
authors' copyright and Common Crawl's terms of use. Every row keeps its source URL.
"""
import argparse
import base64
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from functools import cache
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUCKET = 'trid-training-148761639313'
SOURCE = 'fineweb2-ko'
VERSION = 'clean-v4'
REPO = 'HuggingFaceFW/fineweb-2'
LICENSE = 'ODC-By-1.0 (FineWeb-2 compilation only); underlying web text: original authors, Common Crawl terms of use'
SHARD_CHARS = 200_000_000
UUID = re.compile(r'<urn:uuid:([0-9a-f-]+)>')
# Sites that sell student reports/papers: text is commercially licensed by its sellers. Quarantined, not trained on.
PAID_REPORT_SITES = ('happycampus.com', 'reportworld.co.kr', 'reportshop.co.kr', 'report.co.kr', 'thinkzon.com',
                     'allreport.co.kr', 'thinkuniv.com', 'freezine.co.kr', 'happyreport.co.kr', 'campusplus.co.kr',
                     'report.kyobobook.co.kr')
# Hosts of institutions with benchmark-reserved documents (B-003); built from the supervisor's reservation and the ALIO
# agency list. Rows from these hosts are quarantined until the text-overlap check against the reserved texts clears them.
RESERVED_HOSTS_FILE = ROOT / 'reports/coordination/benchmark-reserve-001-hosts.json'
# Gambling / adult-service advertising (FineWeb-2 clean-v1 sample, B-004). Bare words such as 카지노, 도박, 먹튀,
# 온라인 카지노 or 조건만남 also occur in policy texts (사이버도박 단속, 먹튀공탁, 성매매 실태조사), so only advertising
# phrases count and a document is dropped only at >= 3 hits and >= 2 hits per 1,000 characters.
SPAM_ADS = re.compile(r'(?:카지노|토토|바카라|슬롯|홀덤|파워볼)\s*(?:사이트|주소|추천|가입)|우리카지노|에볼루션\s*카지노|더킹\s*카지노'
                      r'|먹튀\s*(?:검증|사이트|제보|확정|폴리스|업체)|안전\s*놀이터'
                      r'|출장\s*(?:마사지|맛사지|안마|샵|업소|만남)|콜걸|대딸방')
SPAM_MIN_HITS, SPAM_PER_1000 = 3, 2.0
# Machine-translated / SEO pages (B-005): in a 40-row AI review of fixed-sample rows on other TLDs (.xyz, .icu, .site,
# .ru, .za, ...) about 25 were spam, MT or adult pages, against about 4 of 40 on .com. There is no reliable text rule,
# so rows from other TLDs are held in quarantine (not dropped) until a reviewed classifier exists. This also holds
# legitimate pages (diaspora news on .ca, wiki mirrors); the hold is a risk decision, not a quality verdict.
COMMON_TLDS = frozenset({'kr', 'com', 'net', 'org', 'co', 'io', 'me', 'tv', 'edu', 'gov'})
SPLITS = (('test', 50), ('valid', 100))  # per-10,000 bucket upper bounds


def split_of(group):
    bucket = int(hashlib.sha256(f'split:{group}'.encode()).hexdigest()[:8], 16) % 10000
    return next((name for name, upper in SPLITS if bucket < upper), 'train')


def host_of(url):
    host = (urlsplit(url).hostname or '').lower()
    return host[4:] if host.startswith('www.') else host


@cache
def reserved_hosts():
    """Fail closed: a clean run without the reservation host list would silently train on reserved-institution pages."""
    return frozenset(json.loads(RESERVED_HOSTS_FILE.read_text())['hosts'])


def quarantine_reason(url):
    host = host_of(url)
    if any(host == s or host.endswith('.' + s) for s in PAID_REPORT_SITES):
        return 'paid_report_site'
    if any(host == s or host.endswith('.' + s) for s in reserved_hosts()):
        return 'benchmark_reserved_host_pending_text_check'
    if host.rsplit('.', 1)[-1] not in COMMON_TLDS:
        return 'uncertain_tld_mt_seo_risk'
    return None


def spam_reason(text):
    hits = len(SPAM_ADS.findall(text))
    return 'spam_gambling_adult_ads' if hits >= SPAM_MIN_HITS and hits * 1000 / len(text) >= SPAM_PER_1000 else None


def dedup_key(text):
    return hashlib.sha256(''.join(text.split()).encode()).hexdigest()


def native_id(record):
    match = UUID.fullmatch(record.get('id') or '')
    return match.group(1) if match else hashlib.sha256((record.get('url') or record.get('text') or '').encode()).hexdigest()[:32]


def clean_record(record, collected_at):
    """-> (row or None, reason, pii Counter). reason: 'ok', 'quarantine:<why>' or a drop reason."""
    from trid.korean_quality import FINEWEB2, clean_lines, redact, script_ratios
    text = unicodedata.normalize('NFC', record.get('text') or '')
    text, lines_dropped = clean_lines(text)
    if not text.strip():
        return None, 'empty_after_line_cleanup', Counter()
    hangul, hanja, latin = script_ratios(text)
    if hangul < FINEWEB2.hangul_min or hanja > FINEWEB2.hanja_max or latin > FINEWEB2.latin_max:
        return None, 'script_ratio', Counter()
    spam = spam_reason(text)
    if spam:
        return None, spam, Counter()
    text, pii = redact(text)
    native = native_id(record)
    url = record.get('url') or ''
    group = f'{SOURCE}:{native}'
    flags = (['lines_dropped'] if lines_dropped else []) + (['pii_redacted'] if pii else [])
    quarantine = quarantine_reason(url)
    if quarantine:
        flags.append(f'quarantine:{quarantine}')
    row = {
        'id': group, 'group': group, 'kind': 'web', 'title': '', 'text': text, 'source': url, 'license': LICENSE,
        'attribution': f'FineWeb-2 ({REPO}, ODC-By 1.0); original: {url}',
        'date': (record.get('date') or '')[:10] or None, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
        'collected_at': collected_at, 'extractor': 'parquet:fineweb2', 'clean_version': VERSION, 'flags': flags,
        'split': split_of(group), 'provenance': {'institution_type': 'web', 'site': host_of(url)},
        'meta': {'dump': record.get('dump'), 'language_score': record.get('language_score'),
                 'minhash_cluster_size': record.get('minhash_cluster_size'), 'crawl_date': record.get('date'),
                 'source_split': 'train', 'pii_redactions': dict(pii)},
    }
    return row, (f'quarantine:{quarantine}' if quarantine else 'ok'), pii


def clean_batch(args):
    records, collected_at = args
    return [clean_record(r, collected_at) for r in records]


class ShardWriter:
    """gzip JSONL shards of ~SHARD_CHARS characters, uploaded with an S3-verified SHA-256 checksum."""

    def __init__(self, s3, prefix, stem, limit=SHARD_CHARS):
        self.s3, self.prefix, self.stem, self.limit = s3, prefix, stem, limit
        self.index, self.shards = 0, []
        self._reset()

    def _reset(self):
        self.buffer, self.chars, self.rows, self.splits = io.BytesIO(), 0, 0, Counter()
        self.gz = gzip.GzipFile(fileobj=self.buffer, mode='wb', compresslevel=6, mtime=0)

    def add(self, row):
        self.gz.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        self.chars += len(row['text'])
        self.rows += 1
        self.splits[row['split']] += 1
        if self.chars >= self.limit:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        self.gz.close()
        body = self.buffer.getvalue()
        digest = hashlib.sha256(body).digest()
        key = f'{self.prefix}-{self.stem}-{self.index:04d}.jsonl.gz'
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=body, ChecksumSHA256=base64.b64encode(digest).decode())
        self.shards.append({'key': key, 'sha256': digest.hex(), 'bytes': len(body), 'rows': self.rows,
                            'characters': self.chars, 'splits': dict(self.splits)})
        self.index += 1
        self._reset()


def verify(s3, shards):
    """Every shard exists with the recorded size and S3-stored SHA-256."""
    for shard in shards:
        head = s3.head_object(Bucket=BUCKET, Key=shard['key'], ChecksumMode='ENABLED')
        stored = base64.b64decode(head.get('ChecksumSHA256', '')).hex()
        if head['ContentLength'] != shard['bytes'] or stored != shard['sha256']:
            raise RuntimeError(f"shard verification failed: {shard['key']}")


def exists(s3, key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_copy(s3, path, revision, workdir):
    """Local path of the raw parquet, from the S3 raw copy (or the pinned HF revision, stored to raw first)."""
    key = f'corpus/raw/{SOURCE}/{path}'
    local = os.path.join(workdir, os.path.basename(path))
    if exists(s3, key):
        s3.download_file(BUCKET, key, local)
        return local, key, 's3-raw'
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(REPO, path, repo_type='dataset', revision=revision, local_dir=workdir)
    s3.upload_file(downloaded, BUCKET, key)
    return downloaded, key, f'hf:{revision}'


def clean_file(s3, path, revision, workdir, workers, job):
    import pyarrow.parquet as pq
    stem = os.path.basename(path).removesuffix('.parquet')
    state = f'corpus/state/{SOURCE}/{VERSION}/{stem}'
    if exists(s3, f'{state}.done'):
        print(f'skip {path} (done)', flush=True)
        return
    local, raw_key, origin = raw_copy(s3, path, revision, workdir)
    raw_sha = file_sha256(local)
    collected_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    prefix = f'corpus/clean/{SOURCE}/{VERSION}'
    kept, quarantined = ShardWriter(s3, f'{prefix}/docs', stem), ShardWriter(s3, f'{prefix}/quarantine', stem)
    reasons, pii_total, seen, read = Counter(), Counter(), set(), 0

    def batches():
        for batch in pq.ParquetFile(local).iter_batches(batch_size=5_000):
            yield batch.to_pylist(), collected_at

    with Pool(workers) as pool:
        source = batches()
        while True:
            # Bounded window: Pool.imap would pull the whole 4.8 GB file into its task queue.
            window = list(islice(source, 2 * workers))
            if not window:
                break
            for results in pool.map(clean_batch, window):
                for row, reason, pii in results:
                    read += 1
                    if row is None:
                        reasons[reason] += 1
                        continue
                    key = dedup_key(row['text'])
                    if key in seen:
                        reasons['exact_duplicate_in_file'] += 1
                        continue
                    seen.add(key)
                    pii_total.update(pii)
                    reasons[reason] += 1
                    (quarantined if reason.startswith('quarantine:') else kept).add(row)
            print(f'{stem}: read {read} | {dict(reasons)}', flush=True)
    kept.flush()
    quarantined.flush()
    verify(s3, kept.shards + quarantined.shards)
    manifest = {
        'source': SOURCE, 'clean_version': VERSION, 'file': path, 'revision': revision, 'raw_key': raw_key,
        'raw_sha256': raw_sha, 'raw_origin': origin, 'code': 'scripts/collect_fineweb2_ko.py',
        'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), 'job': job,
        'korean_quality_sha256': hashlib.sha256((ROOT / 'trid/korean_quality.py').read_bytes()).hexdigest(),
        'reserved_hosts_file': str(RESERVED_HOSTS_FILE.relative_to(ROOT)),
        'reserved_hosts_sha256': hashlib.sha256(RESERVED_HOSTS_FILE.read_bytes()).hexdigest(),
        'spam_rule': {'pattern': SPAM_ADS.pattern, 'min_hits': SPAM_MIN_HITS, 'per_1000_chars': SPAM_PER_1000},
        'collected_at': collected_at, 'documents_read': read, 'reasons': dict(reasons), 'pii_redactions': dict(pii_total),
        'shards': kept.shards, 'quarantine_shards': quarantined.shards,
        'rows': sum(s['rows'] for s in kept.shards), 'characters': sum(s['characters'] for s in kept.shards),
        'splits': dict(sum((Counter(s['splits']) for s in kept.shards), Counter()))}
    s3.put_object(Bucket=BUCKET, Key=f'{state}.manifest.json', Body=json.dumps(manifest, ensure_ascii=False, indent=1).encode())
    s3.put_object(Bucket=BUCKET, Key=f'{state}.done', Body=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest().encode())
    os.remove(local)
    print(json.dumps({k: manifest[k] for k in ('file', 'documents_read', 'rows', 'characters', 'reasons', 'pii_redactions', 'splits')}, ensure_ascii=False), flush=True)


def aggregate(revision, expected_files):
    import boto3
    s3 = boto3.client('s3')
    manifests = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=f'corpus/state/{SOURCE}/{VERSION}/'):
        for item in page.get('Contents', []):
            if item['Key'].endswith('.manifest.json') and exists(s3, item['Key'].replace('.manifest.json', '.done')):
                manifests.append(json.loads(s3.get_object(Bucket=BUCKET, Key=item['Key'])['Body'].read()))
    total = lambda field: sum((Counter(m[field]) for m in manifests), Counter())
    chars = sum(m['characters'] for m in manifests)
    stats = {'source': SOURCE, 'clean_version': VERSION, 'files_done': len(manifests), 'files_expected': expected_files,
             'documents_read': sum(m['documents_read'] for m in manifests), 'rows': sum(m['rows'] for m in manifests),
             'characters': chars, 'estimated_tokens': round(chars / 1.6), 'estimate_note': 'chars / 1.6, not a tokenizer count',
             'reasons': dict(total('reasons')), 'pii_redactions': dict(total('pii_redactions')), 'splits': dict(total('splits')),
             'quarantine_rows': sum(s['rows'] for m in manifests for s in m['quarantine_shards'])}
    provenance = {'source': SOURCE, 'clean_version': VERSION, 'dataset': REPO, 'config': 'kor_Hang', 'revision': revision,
                  'license': LICENSE, 'license_evidence': ['https://huggingface.co/datasets/HuggingFaceFW/fineweb-2',
                                                           'https://commoncrawl.org/terms-of-use'],
                  'paid_report_sites_quarantined': list(PAID_REPORT_SITES), 'pipeline': (__doc__ or '').split('\n\n')[2],
                  'files': sorted(m['file'] for m in manifests), 'jobs': sorted({m['job'] for m in manifests})}
    attribution = ('FineWeb-2 Korean (kor_Hang) by Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-2, '
                   'ODC-By 1.0 (https://opendatacommons.org/licenses/by/1-0/); derived from Common Crawl '
                   '(https://commoncrawl.org/terms-of-use). Underlying pages keep their own copyright; each row records its URL.\n')
    prefix = f'corpus/clean/{SOURCE}/{VERSION}'
    for name, body in (('stats.json', json.dumps(stats, ensure_ascii=False, indent=2)),
                       ('provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2)), ('ATTRIBUTION.txt', attribution)):
        s3.put_object(Bucket=BUCKET, Key=f'{prefix}/{name}', Body=body.encode())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['clean', 'aggregate'])
    p.add_argument('--revision', required=True, help='pinned HF dataset commit sha')
    p.add_argument('--files', nargs='*', default=[], help='repo paths, e.g. data/kor_Hang/train/000_00000.parquet')
    p.add_argument('--expected-files', type=int, default=25)
    p.add_argument('--workdir', default='/opt/fineweb')
    p.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    p.add_argument('--job', default=os.environ.get('JOB_NAME', 'local'))
    args = p.parse_args()
    if args.action == 'clean':
        import boto3
        os.makedirs(args.workdir, exist_ok=True)
        s3 = boto3.client('s3')
        for path in args.files:
            clean_file(s3, path, args.revision, args.workdir, args.workers, args.job)
    else:
        aggregate(args.revision, args.expected_files)


if __name__ == '__main__':
    main()

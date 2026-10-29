"""FineWeb-2 Korean (HuggingFaceFW/fineweb-2, config kor_Hang) -> trid corpus contract shards in S3.

  convert   (AWS, SageMaker CPU job)  download pinned parquet files from the HF Hub, copy the originals to
            s3://<bucket>/corpus/raw/fineweb2-ko/, convert rows to the corpus schema, write gzip JSONL shards to
            s3://<bucket>/corpus/clean/fineweb2-ko/ and a done marker with counts to corpus/state/fineweb2-ko/.
            Files whose done marker exists are skipped, so re-running a job never duplicates shards.
  aggregate (anywhere)                sum the done markers into corpus/clean/fineweb2-ko/{stats,provenance}.json and
            write ATTRIBUTION.txt.

Licence: the FineWeb-2 compilation is ODC-By 1.0; the underlying web text remains the original authors' work and
is subject to Common Crawl's terms of use. Rows record the source URL so documents can be traced or removed.
Text is NFC-normalized; nothing else is altered (FineWeb-2 already applied language ID, quality filters and
MinHash dedup). PII redaction and eval-source exclusion happen later in the mix builder.
"""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import unicodedata

BUCKET = 'trid-training-148761639313'
SOURCE = 'fineweb2-ko'
REPO = 'HuggingFaceFW/fineweb-2'
LICENSE = 'ODC-By-1.0 (FineWeb-2 compilation); underlying web text: original authors, Common Crawl terms of use'
SHARD_CHARS = 200_000_000
UUID = re.compile(r'<urn:uuid:([0-9a-f-]+)>')


def row_of(record, collected_at):
    """FineWeb-2 parquet record (dict) -> corpus contract row."""
    text = unicodedata.normalize('NFC', record['text'] or '')
    native = record.get('id') or ''
    match = UUID.fullmatch(native)
    native = match.group(1) if match else hashlib.sha256((record.get('url') or text).encode()).hexdigest()[:32]
    date = (record.get('date') or '')[:10] or None
    url = record.get('url') or ''
    return {
        'id': f'{SOURCE}:{native}', 'group': f'{SOURCE}:{native}', 'kind': 'web', 'title': '', 'text': text,
        'source': url, 'license': LICENSE,
        'attribution': f'FineWeb-2 ({REPO}, ODC-By 1.0); original: {url}',
        'date': date, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': collected_at,
        'extractor': 'parquet:fineweb2',
        'meta': {'dump': record.get('dump'), 'language_score': record.get('language_score'),
                 'minhash_cluster_size': record.get('minhash_cluster_size'), 'crawl_date': record.get('date')},
    }


class ShardWriter:
    """Accumulates rows and uploads gzip JSONL shards of ~SHARD_CHARS characters."""

    def __init__(self, s3, stem, limit=SHARD_CHARS):
        self.s3, self.stem, self.limit = s3, stem, limit
        self.index, self.buffer, self.chars, self.keys = 0, io.BytesIO(), 0, []
        self.gz = gzip.GzipFile(fileobj=self.buffer, mode='wb', compresslevel=6, mtime=0)

    def add(self, row):
        self.gz.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        self.chars += len(row['text'])
        if self.chars >= self.limit:
            self.flush()

    def flush(self):
        if not self.chars:
            return
        self.gz.close()
        key = f'corpus/clean/{SOURCE}/docs-{self.stem}-{self.index:04d}.jsonl.gz'
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=self.buffer.getvalue())
        self.keys.append(key)
        self.index += 1
        self.buffer, self.chars = io.BytesIO(), 0
        self.gz = gzip.GzipFile(fileobj=self.buffer, mode='wb', compresslevel=6, mtime=0)


def exists(s3, key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def convert(files, revision, workdir):
    import boto3
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    s3 = boto3.client('s3')
    for path in files:
        stem = os.path.basename(path).removesuffix('.parquet')
        done_key = f'corpus/state/{SOURCE}/{stem}.done.json'
        if exists(s3, done_key):
            print(f'skip {path} (done)', flush=True)
            continue
        local = hf_hub_download(REPO, path, repo_type='dataset', revision=revision, local_dir=workdir)
        s3.upload_file(local, BUCKET, f'corpus/raw/{SOURCE}/{path}')
        collected_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        writer, docs, chars, years = ShardWriter(s3, stem), 0, 0, {}
        for batch in pq.ParquetFile(local).iter_batches(batch_size=20_000):
            for record in batch.to_pylist():
                row = row_of(record, collected_at)
                if not row['text']:
                    continue
                writer.add(row)
                docs += 1
                chars += len(row['text'])
                year = (row['date'] or 'unknown')[:4]
                years[year] = years.get(year, 0) + len(row['text'])
            print(f'{stem}: {docs} docs {chars} chars', flush=True)
        writer.flush()
        s3.put_object(Bucket=BUCKET, Key=done_key, Body=json.dumps({
            'file': path, 'revision': revision, 'documents': docs, 'characters': chars, 'chars_by_crawl_year': years,
            'shards': writer.keys, 'collected_at': collected_at}, ensure_ascii=False).encode())
        os.remove(local)
        print(f'done {path}: {docs} docs, {chars} chars, {len(writer.keys)} shards', flush=True)


def aggregate(revision, expected_files):
    import boto3
    s3 = boto3.client('s3')
    markers = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=f'corpus/state/{SOURCE}/'):
        for item in page.get('Contents', []):
            if item['Key'].endswith('.done.json'):
                markers.append(json.loads(s3.get_object(Bucket=BUCKET, Key=item['Key'])['Body'].read()))
    docs = sum(m['documents'] for m in markers)
    chars = sum(m['characters'] for m in markers)
    years = {}
    for m in markers:
        for year, n in m['chars_by_crawl_year'].items():
            years[year] = years.get(year, 0) + n
    stats = {'source': SOURCE, 'files_done': len(markers), 'files_expected': expected_files, 'documents': docs,
             'characters': chars, 'estimated_tokens': round(chars / 1.6), 'estimate_note': 'chars / 1.6, not a tokenizer count',
             'chars_by_crawl_year': dict(sorted(years.items())), 'shards': sum(len(m['shards']) for m in markers)}
    provenance = {
        'source': SOURCE, 'dataset': REPO, 'config': 'kor_Hang', 'split': 'train', 'revision': revision,
        'license': LICENSE, 'license_evidence': ['https://huggingface.co/datasets/HuggingFaceFW/fineweb-2',
                                                 'https://commoncrawl.org/terms-of-use'],
        'collector': 'scripts/collect_fineweb2_ko.py', 'processing': 'NFC normalization only; schema conversion; '
        'FineWeb-2 language ID, quality filtering and MinHash dedup were applied upstream. PII redaction and eval '
        'exclusion are left to scripts/build_pretraining_mix.py.', 'files': sorted(m['file'] for m in markers),
        'raw_prefix': f's3://{BUCKET}/corpus/raw/{SOURCE}/', 'clean_prefix': f's3://{BUCKET}/corpus/clean/{SOURCE}/'}
    attribution = ('FineWeb-2 Korean (kor_Hang) by Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-2, '
                   'licensed under ODC-By 1.0 (https://opendatacommons.org/licenses/by/1-0/). Derived from Common Crawl '
                   '(https://commoncrawl.org/terms-of-use). Each row records its original URL.\n')
    for name, body in (('stats.json', json.dumps(stats, ensure_ascii=False, indent=2)),
                       ('provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2)),
                       ('ATTRIBUTION.txt', attribution)):
        s3.put_object(Bucket=BUCKET, Key=f'corpus/clean/{SOURCE}/{name}', Body=body.encode())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['convert', 'aggregate'])
    p.add_argument('--revision', required=True, help='pinned HF dataset commit sha')
    p.add_argument('--files', nargs='*', default=[], help='repo paths, e.g. data/kor_Hang/train/000_00000.parquet')
    p.add_argument('--expected-files', type=int, default=25)
    p.add_argument('--workdir', default='/opt/fineweb')
    args = p.parse_args()
    if args.action == 'convert':
        convert(args.files, args.revision, args.workdir)
    else:
        aggregate(args.revision, args.expected_files)


if __name__ == '__main__':
    main()

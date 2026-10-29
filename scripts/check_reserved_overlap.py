"""Evaluation-reserved document exclusion for pretraining corpora (BENCHMARK-SCALE-001, A side).

The supervisor owns the reservation. `sign` turns reserved documents into a text-free manifest; `check` finds corpus
rows that must be excluded from training (or, for corpora already trained on, the reserved families whose test items
are contaminated for that checkpoint). Hashes are a leakage filter, not encryption: they do not keep answers secret.

  sign   input JSONL rows {family_id, text, doc_id?, source_url?, group?} -> manifest JSONL rows
         {family_id, doc_id, source_url, group, exact_sha256, minhash[64], ngram8[]} (no text).
  check  corpus JSONL (collector/mix rows: id, group, source, text) against a manifest. A row matches a family when
         its id, group or source URL equals a reserved one, its exact hash (NFC, all whitespace removed) is equal,
         its word 5-shingle MinHash estimate (same Signer as scripts/build_pretraining_mix.py) is >= --minhash, or
         >= --containment of a reserved document's word 8-grams occur in the row (catches excerpts and attachments).
         Writes a JSON report (matches by type, per family, matched corpus ids) and optionally an id list to exclude.

  python scripts/check_reserved_overlap.py sign --input reserved.jsonl --output manifest.jsonl
  python scripts/check_reserved_overlap.py check --manifest manifest.jsonl --corpus data/mix-v3-dry/train.jsonl \
      --report reports/<name>.json --exclude-ids <ids.txt>
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_pretraining_mix import Signer  # noqa: E402  (same MinHash as corpus near-dedup)

NGRAM = 8
SPACE = re.compile(r'\s+')


def exact_hash(text):
    return hashlib.sha256(SPACE.sub('', unicodedata.normalize('NFC', text)).encode()).hexdigest()


def ngrams(text):
    words = unicodedata.normalize('NFC', text).split()
    return {int.from_bytes(hashlib.blake2b(' '.join(words[i:i + NGRAM]).encode(), digest_size=8).digest(), 'little')
            for i in range(len(words) - NGRAM + 1)}


def url_key(url):
    return (url or '').strip().rstrip('/').lower().removeprefix('https://').removeprefix('http://') or None


def sign(a):
    signer = Signer()
    with a.output.open('w') as out:
        for line in a.input.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = unicodedata.normalize('NFC', row['text'])
            out.write(json.dumps({'family_id': row['family_id'], 'doc_id': row.get('doc_id'), 'source_url': row.get('source_url'),
                                  'group': row.get('group'), 'exact_sha256': exact_hash(text),
                                  'minhash': [str(v) for v in signer(text)], 'ngram8': sorted(ngrams(text))}) + '\n')


_ctx = {}


def init(manifest_path, minhash_threshold, containment):
    rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
    _ctx['rows'] = rows
    _ctx['signer'] = Signer()
    _ctx['sigs'] = np.array([[int(v) for v in r['minhash']] for r in rows], dtype=np.uint64)
    owner = defaultdict(set)
    for i, r in enumerate(rows):
        for h in r['ngram8']:
            owner[h].add(i)
    _ctx['owner'] = owner
    _ctx['sizes'] = [max(1, len(r['ngram8'])) for r in rows]
    _ctx['by'] = {k: defaultdict(set) for k in ('id', 'group', 'url', 'exact')}
    for i, r in enumerate(rows):
        for key, value in (('id', r.get('doc_id')), ('group', r.get('group')), ('url', url_key(r.get('source_url'))),
                           ('exact', r['exact_sha256'])):
            if value:
                _ctx['by'][key][value].add(i)
    _ctx['minhash'], _ctx['containment'] = minhash_threshold, containment


def match(lines):
    out = []
    for line in lines:
        row = json.loads(line)
        text = row.get('text') or ''
        hits = defaultdict(set)
        for key, value in (('id', row.get('id')), ('group', row.get('group')), ('url', url_key(row.get('source'))),
                           ('exact', exact_hash(text))):
            for i in _ctx['by'][key].get(value, ()):
                hits[i].add(key)
        if len(_ctx['rows']):
            est = (_ctx['sigs'] == _ctx['signer'](text)[None, :]).mean(axis=1)
            for i in np.flatnonzero(est >= _ctx['minhash']):
                hits[int(i)].add('minhash')
            shared = Counter(i for h in ngrams(text) for i in _ctx['owner'].get(h, ()))
            for i, n in shared.items():
                if n / _ctx['sizes'][i] >= _ctx['containment']:
                    hits[i].add('containment')
        if hits:
            out.append({'corpus_id': row.get('id'), 'kind': row.get('kind'),
                        'families': {_ctx['rows'][i]['family_id']: sorted(v) for i, v in hits.items()}})
    return out


def check_ids(a):
    """Metadata-only reservation (doc_family_id, url; no text): ID/group/URL matches only. Text overlap is NOT checked."""
    reserved = [json.loads(l) for l in a.reservation.read_text().splitlines() if l.strip()]
    if any('minhash' in r or 'text' in r for r in reserved):
        raise ValueError('Use `check` for signed text manifests; check-ids takes metadata-only reservations')
    by_id = {r['doc_family_id']: r['doc_family_id'] for r in reserved}
    by_url = {url_key(r.get('url')): r['doc_family_id'] for r in reserved if r.get('url')}
    matches, scanned = [], 0
    for path in a.corpus:
        with Path(path).open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                scanned += 1
                row = json.loads(line)
                hit = {}
                for key in ('id', 'group'):
                    if row.get(key) in by_id:
                        hit[by_id[row[key]]] = key
                family = by_url.get(url_key(row.get('source')))
                if family:
                    hit[family] = 'url'
                if hit:
                    matches.append({'corpus_id': row.get('id'), 'families': hit})
    report = {'reservation': str(a.reservation), 'reservation_sha256': hashlib.sha256(a.reservation.read_bytes()).hexdigest(),
              'reserved_families': len(reserved), 'mode': 'metadata-only: id/group/url; text overlap not checked',
              'corpus': [{'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in a.corpus],
              'corpus_rows_scanned': scanned, 'matched_corpus_rows': len(matches), 'matches': matches}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ('reserved_families', 'mode', 'corpus_rows_scanned', 'matched_corpus_rows')},
                     ensure_ascii=False))


def chunks(paths, size=500):
    for path in paths:
        with Path(path).open(encoding='utf-8') as f:
            batch = []
            for line in f:
                if line.strip():
                    batch.append(line)
                    if len(batch) == size:
                        yield batch
                        batch = []
            if batch:
                yield batch


def check(a):
    manifest_sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    families = {json.loads(l)['family_id'] for l in a.manifest.read_text().splitlines() if l.strip()}
    matches, scanned = [], 0
    with Pool(a.workers, initializer=init, initargs=(str(a.manifest), a.minhash, a.containment)) as pool:
        for found in pool.imap(match, chunks(a.corpus), chunksize=1):
            matches += found
    for path in a.corpus:
        with Path(path).open(encoding='utf-8') as f:
            scanned += sum(1 for line in f if line.strip())
    by_type = Counter(t for m in matches for types in m['families'].values() for t in types)
    hit_families = Counter(fam for m in matches for fam in m['families'])
    report = {'manifest_sha256': manifest_sha, 'reserved_families': len(families),
              'corpus': [{'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in a.corpus],
              'corpus_rows_scanned': scanned, 'matched_corpus_rows': len(matches),
              'families_with_matches': len(hit_families), 'match_types': dict(by_type),
              'thresholds': {'minhash_estimate': a.minhash, 'ngram8_containment': a.containment},
              'per_family_matched_rows': dict(hit_families), 'matches': matches}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if a.exclude_ids:
        a.exclude_ids.write_text(''.join(f"{m['corpus_id']}\n" for m in matches))
    print(json.dumps({k: report[k] for k in ('reserved_families', 'corpus_rows_scanned', 'matched_corpus_rows',
                                             'families_with_matches', 'match_types')}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('sign')
    s.add_argument('--input', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('check')
    c.add_argument('--manifest', type=Path, required=True)
    c.add_argument('--corpus', type=Path, nargs='+', required=True)
    c.add_argument('--report', type=Path, required=True)
    c.add_argument('--exclude-ids', type=Path)
    c.add_argument('--minhash', type=float, default=0.5)
    c.add_argument('--containment', type=float, default=0.5)
    c.add_argument('--workers', type=int, default=8)
    m = sub.add_parser('check-ids', help='metadata-only reservation (doc_family_id, url): id/group/url matches only')
    m.add_argument('--reservation', type=Path, required=True)
    m.add_argument('--corpus', type=Path, nargs='+', required=True)
    m.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    {'sign': sign, 'check': check, 'check-ids': check_ids}[a.action](a)


if __name__ == '__main__':
    main()

"""Expand the local Wikipedia sample; preserve original holdouts and attribution.

MinHash candidate search is approximate, followed by exact shingle Jaccard.
It does not prove absence of near duplicates or undo exposure in parent weights.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
import shutil
import unicodedata

import numpy as np
from trid.corpus import sha256, split_for, write_json
from trid.text import SPECIAL


def shingles(text):
    words = re.findall(r"\w+", unicodedata.normalize('NFC', text).lower())
    parts = [' '.join(words[i:i+5]) for i in range(max(1, len(words)-4))]
    return {int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), 'little') for x in parts}


class NearDuplicates:
    def __init__(self):
        self.buckets = defaultdict(list)
        self.items = {}
        rng = np.random.default_rng(20261027)
        self.a = rng.integers(1, 2**63, size=16, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self.b = rng.integers(0, 2**63, size=16, dtype=np.uint64)

    def keys(self, values):
        x = np.fromiter(values, dtype=np.uint64)
        with np.errstate(over='ignore'):
            signature = (x[:, None] * self.a + self.b).min(axis=0)
        return [(i, int(signature[i*2]), int(signature[i*2+1])) for i in range(8)]

    def match_and_add(self, ident, text):
        values = shingles(text)
        keys = self.keys(values)
        candidates = {i for key in keys for i in self.buckets[key]}
        for other in sorted(candidates):
            old = self.items[other]
            if min(len(old), len(values)) / max(len(old), len(values)) < .8:
                continue
            overlap = len(values & old) / len(values | old)
            if overlap >= .8:
                return other, overlap
        self.items[ident] = values
        for key in keys:
            self.buckets[key].append(ident)
        return None


def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='data/wiki-ko-pilot')
    p.add_argument('--output', default='data/korean-foundation-v1')
    p.add_argument('--extra-train', type=int, default=12000)
    a = p.parse_args()
    if a.extra_train < 1:
        raise ValueError('--extra-train must be positive.')
    source, out = Path(a.source), Path(a.output)
    if out.exists():
        raise ValueError('Use a new output folder.')
    provenance = json.loads((source/'provenance.json').read_text())
    if sha256(source/'source.parquet') != provenance['source_sha256']:
        raise ValueError('Source parquet changed.')
    rows = {}
    for split in ('train', 'valid', 'test'):
        if sha256(source/f'{split}.jsonl') != provenance['file_sha256'][split]:
            raise ValueError('Source split changed.')
        rows[split] = [json.loads(x) for x in (source/f'{split}.jsonl').read_text().splitlines()]
    existing = {r['sha256'] for part in rows.values() for r in part}
    heap, seen, counts = [], set(existing), Counter()
    for batch in pq.ParquetFile(source/'source.parquet').iter_batches(batch_size=256):
        for row in batch.to_pylist():
            text = unicodedata.normalize('NFC', row['text']).replace('\x00', '').strip()
            if not 300 <= len(text) <= 50000 or len(re.findall('[가-힣]', text))/len(text) < .2 or any(s in text for s in SPECIAL):
                counts['length_language_or_control_rejected'] += 1
                continue
            split, digest = split_for(text)
            if split != 'train' or digest in seen:
                continue
            seen.add(digest)
            # Different digest salt from the split rule; deterministic reservoir.
            priority = int(hashlib.sha256(('extra:'+digest).encode()).hexdigest(), 16)
            record = {'id': row['id'], 'title': row['title'], 'source': row['url'], 'text': text,
                      'sha256': digest, 'contributors': row['url']+'?action=history'}
            item = (-priority, digest, record)
            if len(heap) < a.extra_train:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
            counts['eligible_new_train'] += 1
    rows['train'] += [r for _, _, r in sorted(heap, key=lambda x: x[1])]
    print(f"Selected {len(heap)} additional documents; checking approximate near duplicates.", flush=True)
    index, removed = NearDuplicates(), []
    for split in ('test', 'valid'):
        for i, row in enumerate(rows[split]):
            match = index.match_and_add(f'{split}:{i}', row['text'])
            if match:
                counts['near_duplicates_within_original_holdouts'] += 1
    accepted, exact = [], set()
    for part in ('valid', 'test'):
        exact.update(re.sub(r'\s+', '', r['text']) for r in rows[part])
    for i, row in enumerate(rows['train']):
        key = re.sub(r'\s+', '', row['text'])
        match = ('normalized_exact', 1.0) if key in exact else index.match_and_add(f'train:{i}', row['text'])
        if match:
            removed.append({'id': row['id'], 'sha256': row['sha256'], 'matches': match[0], 'jaccard': match[1]})
        else:
            accepted.append(row)
            exact.add(key)
        if (i+1) % 5000 == 0:
            print(f"Checked {i+1}/{len(rows['train'])}", flush=True)
    out.mkdir(parents=True)
    with (out/'train.jsonl').open('w') as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    for split in ('valid', 'test'):
        shutil.copy2(source/f'{split}.jsonl', out/f'{split}.jsonl')
    for name in ('ATTRIBUTION.txt', 'SOURCE_DATASET_CARD.md'):
        shutil.copy2(source/name, out/name)
    write_json(out/'removed.json', removed)
    write_json(out/'provenance.json', {
        'parent': provenance, 'extra_train_selected': len(heap), 'counts': {'train': len(accepted), 'valid': len(rows['valid']), 'test': len(rows['test'])},
        'filter_counts': dict(counts), 'train_removed': len(removed),
        'selection': 'Original training plus fixed hash-priority sample from same local Wikipedia shard, original split buckets preserved.',
        'deduplication': 'NFC + whitespace-free exact match, then word 5-shingle 16-component MinHash (8 bands of 2), exact Jaccard >= 0.8 on candidates. Approximate; recall not guaranteed.',
        'holdouts': 'Original valid/test files unchanged. No new holdout documents promoted into training. Parent weights may already have seen near duplicates; cannot undo historical exposure.',
        'diversity_limit': 'Wikipedia only; not a broad dialogue or institution corpus.',
        'file_sha256': {s: sha256(out/f'{s}.jsonl') for s in ('train', 'valid', 'test')},
        'real_company_data': False})
    print(json.dumps({'output': str(out), 'train': len(accepted), 'removed': len(removed)}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()

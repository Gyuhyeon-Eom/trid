"""Dense vectors for the workspace search index (HYBRID-RAG-001), resumable and incremental.

Every passage of a workspace index (runs/workspace-index/<v>) is embedded as "<institution> <title>\n<text>" with
BAAI/bge-m3 (MIT, local Hugging Face snapshot; no network at run time). Vectors are cached by the sha256 of that exact
string in a content-addressed store (`--store`, shards of 2,048 float16 unit vectors), so an interrupted run resumes
where it stopped and a new corpus snapshot only embeds passages whose text changed or is new. When every passage
has a vector, `<index>/dense.f16.npy` (rows aligned with the index passages) and `<index>/dense.json` (model,
snapshot revision, weight sha256, vectors sha256, counts) are written. The service loads those two files.

  .venv/bin/python scripts/build_embeddings.py --index runs/workspace-index/v2 --store runs/embeddings/bge-m3
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.service.embed import MAX_TOKENS, MODEL, load_model, passage_input, weight_sha256  # noqa: E402

SHARD = 2048


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Store:
    """Content-addressed float16 unit vectors: shard-NNNNN.npz files holding `keys` and `vectors`."""
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        self.shards = []
        for path in sorted(self.dir.glob('shard-*.npz')):
            z = np.load(path)
            self.shards.append(z['vectors'])
            for i, k in enumerate(z['keys']):
                self.rows[str(k)] = (len(self.shards) - 1, i)

    def add(self, keys, vectors):
        path = self.dir / f'shard-{len(self.shards):05d}.npz'
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, keys=np.array(keys), vectors=vectors.astype(np.float16))
        os.replace(tmp, path)  # a shard is either complete or absent
        self.shards.append(vectors.astype(np.float16))
        for i, k in enumerate(keys):
            self.rows[k] = (len(self.shards) - 1, i)

    def get(self, k):
        s, i = self.rows[k]
        return self.shards[s][i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--store', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--limit', type=int, help='embed at most this many missing passages (smoke runs)')
    a = p.parse_args()
    from trid.service.index import Index
    root = Path(a.index)
    index = Index.load(root)
    meta = json.loads((root / 'meta.json').read_text())
    texts = [passage_input(meta, p) for p in index.passages]
    keys = [key(t) for t in texts]
    store = Store(a.store)
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in store.rows}.items(), key=lambda kt: len(kt[1]))
    print(f'passages {len(texts)} unique inputs {len(set(keys))} to embed {len(missing)}', flush=True)
    if a.limit is not None:
        missing = missing[:a.limit]
    from huggingface_hub import snapshot_download
    model, path = load_model(a.device) if missing else (None, snapshot_download(MODEL, local_files_only=True))
    started = time.time()
    for start in range(0, len(missing), SHARD):
        part = missing[start:start + SHARD]
        vectors = model.encode([t for _, t in part], batch_size=a.batch, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        store.add([k for k, _ in part], vectors)
        done = start + len(part)
        rate = done / (time.time() - started)
        print(f'{done}/{len(missing)} {rate:.1f}/s eta {(len(missing) - done) / rate / 60:.1f} min', flush=True)
    if any(k not in store.rows for k in keys):
        print('incomplete: rerun the same command to continue', flush=True)
        return
    dense = np.stack([store.get(k) for k in keys]).astype(np.float16)
    np.save(root / 'dense.f16.npy', dense)
    info = {'model': MODEL, 'snapshot': Path(path).name, 'weights_sha256': weight_sha256(path),
            'input': '<institution> <title>\\n<passage text>', 'max_tokens': MAX_TOKENS, 'normalized': True,
            'dtype': 'float16', 'passages': len(keys), 'dim': int(dense.shape[1]),
            'passage_keys_sha256': hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
            'vectors_sha256': hashlib.sha256(dense.tobytes()).hexdigest(), 'built_at_unix': time.time()}
    (root / 'dense.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print('wrote', root / 'dense.f16.npy', info['vectors_sha256'][:12], flush=True)


if __name__ == '__main__':
    main()

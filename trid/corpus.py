"""Explicit public corpus download and local-only tokenization.

No model weights are downloaded. Per-document source links are retained.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata
import urllib.request

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from .text import SPECIAL

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SHARD = "20231101.ko/train-00000-of-00003.parquet"
SHARD_SHA = "28ff69a402878eba3be42dfd187f37876b917b72eb9b05896ed7469fa681bb03"
BASE = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISION}"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prefetch(items, work, workers, depth=4):
    """Run `work(item)` on a thread pool and yield (item, future) in input order.
    At most workers*depth items are in flight; consumers call future.result() to raise errors."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        for item in items:
            pending.append((item, pool.submit(work, item)))
            if len(pending) >= workers * depth:
                yield pending.popleft()
        while pending:
            yield pending.popleft()

def split_for(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return ("test" if bucket < 5 else "valid" if bucket < 10 else "train"), digest


def download(url, destination, expected_sha=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha and sha256(destination) != expected_sha:
            raise ValueError(f"기존 파일의 SHA256 불일치: {destination}")
        return
    part = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "trid-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as f:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            f.write(block)
    if expected_sha and sha256(part) != expected_sha:
        raise ValueError("다운로드 SHA256 검증 실패. .part 파일을 확인하세요.")
    os.replace(part, destination)


def fetch_wiki(output, max_documents=12000):
    if max_documents < 100:
        raise ValueError("최소 100개 문서를 지정하세요.")
    import pyarrow.parquet as pq
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("말뭉치 출력 폴더는 새 경로여야 합니다.")
    root.mkdir(parents=True)
    raw = root / "source.parquet"
    print("공개 한국어 Wikipedia 샤드 다운로드 (약 400MB). 모델 가중치는 받지 않습니다.", flush=True)
    download(f"{BASE}/{SHARD}", raw, SHARD_SHA)
    download(f"{BASE}/README.md", root / "SOURCE_DATASET_CARD.md")
    outputs = {s: (root / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "valid", "test")}
    counts, seen = dict.fromkeys(outputs, 0), set()
    try:
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=256, columns=["id", "url", "title", "text"]):
            for row in batch.to_pylist():
                text = unicodedata.normalize("NFC", row["text"]).replace("\x00", "").strip()
                if len(text) < 300 or len(re.findall("[가-힣]", text)) / max(1, len(text)) < .2:
                    continue
                # Keep full documents; no text clipping or title-only fragments.
                if len(text) > 50000 or any(s in text for s in SPECIAL):
                    continue
                split, digest = split_for(text)
                if digest in seen:
                    continue
                seen.add(digest)
                record = {"id": row["id"], "title": row["title"], "source": row["url"],
                          "text": text, "sha256": digest, "contributors": row["url"] + "?action=history"}
                outputs[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                if sum(counts.values()) >= max_documents:
                    break
            if sum(counts.values()) >= max_documents:
                break
    finally:
        for f in outputs.values():
            f.close()
    if min(counts.values()) == 0:
        raise ValueError("비어 있는 split이 있습니다. 더 많은 문서를 확보하세요.")
    provenance = {"dataset": "wikimedia/wikipedia", "configuration": "20231101.ko", "revision": REVISION,
                  "source_url": f"{BASE}/{SHARD}", "source_sha256": SHARD_SHA,
                  "dataset_card_license_labels": ["cc-by-sa-3.0", "gfdl"],
                  "current_wikimedia_license_reference": "https://dumps.wikimedia.org/legal.html",
                  "current_wikimedia_text_license": "CC BY-SA 4.0; see source terms and exceptions",
                  "attribution": "Wikipedia contributors; per-document URL and history retained in JSONL",
                  "changes": "NFC normalization, null-character removal, whitespace trim; filtered and split",
                  "selection": "First eligible documents from first Korean shard; not a representative random sample",
                  "split_method": "normalized full-text SHA256 buckets: train 90%, valid 5%, test 5%; exact dedup only",
                  "counts": counts, "near_duplicate_filtering": False,
                  "file_sha256": {s: sha256(root / f"{s}.jsonl") for s in counts},
                  "real_company_data": False}
    write_json(root / "provenance.json", provenance)
    (root / "ATTRIBUTION.txt").write_text(
        "Source: Korean Wikipedia, 2023-11-01 snapshot, via wikimedia/wikipedia.\n"
        "Authors: Wikipedia contributors. Individual article and history links are retained in each JSONL row.\n"
        "Dataset card labels: CC BY-SA 3.0 and GFDL. Current Wikimedia text terms refer to CC BY-SA 4.0.\n"
        "https://creativecommons.org/licenses/by-sa/3.0/\nhttps://creativecommons.org/licenses/by-sa/4.0/\n"
        "https://dumps.wikimedia.org/legal.html\n"
        "Changes: NFC normalization, null removal, trimming, filtering, exact deduplication and split.\n"
        "Preserve attribution and applicable source licensing for redistributed text.\n"
        "This local pilot is not a declaration that trained weights are free of licensing obligations.\n", encoding="utf-8")
    print(json.dumps({"documents": counts, "output": str(root)}, ensure_ascii=False), flush=True)


def documents(path):
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip() or any(s in text for s in SPECIAL):
                raise ValueError(f"잘못된 문서: {path}:{n}")
            yield text


def prepare(source, output, vocab_size=8192, tokenizer_path=None):
    if not 261 <= vocab_size <= 65535:
        raise ValueError("어휘 수는 261~65535 범위여야 합니다.")
    source, root = Path(source), Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("토큰 데이터 출력 폴더는 새 경로여야 합니다.")
    seen, counts = set(), {}
    for split in ("train", "valid", "test"):
        counts[split] = 0
        for text in documents(source / f"{split}.jsonl"):
            key = hashlib.sha256(unicodedata.normalize("NFC", text).strip().encode()).hexdigest()
            if key in seen:
                raise ValueError("동일 문서가 여러 번 등장합니다. split 누수와 중복을 제거하세요.")
            seen.add(key)
            counts[split] += 1
        if not counts[split]:
            raise ValueError(f"빈 split: {split}")
    if tokenizer_path:
        tok = Tokenizer.from_file(str(tokenizer_path))
        if tok.get_vocab_size() > 65535 or any(tok.token_to_id(s) is None for s in SPECIAL):
            raise ValueError("지원하지 않는 토크나이저입니다.")
    root.mkdir(parents=True, exist_ok=True)
    if tokenizer_path:
        print("기존 토크나이저와 토큰 번호를 유지합니다.", flush=True)
        shutil.copy2(tokenizer_path, root / "tokenizer.json")
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                     special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        print("학습 문서만 사용해 새 BPE 토크나이저를 학습합니다.", flush=True)
        tok.train_from_iterator(documents(source / "train.jsonl"), trainer)
        tok.save(str(root / "tokenizer.json"))
    tokens = {}
    for split in counts:
        total = 0
        with (root / f"{split}.bin").open("wb") as f:
            for text in documents(source / f"{split}.jsonl"):
                ids = [tok.token_to_id("<|bos|>")] + tok.encode(text).ids + [tok.token_to_id("<|eos|>")]
                np.asarray(ids, dtype="<u2").tofile(f)
                total += len(ids)
        tokens[split] = total
        print(f"{split}: {total:,} tokens", flush=True)
    manifest = {"format": "uint16-little-endian", "vocab_size": tok.get_vocab_size(),
                "documents": counts, "tokens": tokens, "source": str(source.resolve()),
                "tokenizer_sha256": sha256(root / "tokenizer.json"),
                "tokenizer_origin": str(Path(tokenizer_path).resolve()) if tokenizer_path else "trained on train split",
                "sha256": {s: sha256(root / f"{s}.bin") for s in counts},
                "source_sha256": {s: sha256(source / f"{s}.jsonl") for s in counts},
                "objective": "continuous next-token prediction; BOS/EOS document separators; all targets supervised"}
    if (source / "provenance.json").exists():
        manifest["source_provenance"] = json.loads((source / "provenance.json").read_text())
    write_json(root / "manifest.json", manifest)

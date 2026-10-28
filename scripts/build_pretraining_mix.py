"""Mix collected pretraining sources into train/valid/test JSONL for trid.corpus.prepare.

Inputs are source directories in the shared collector layout (docs.jsonl rows with text, sha256,
license, attribution, group; optional pilot-stats.json with chars_per_token_sample):
    --source data/law-ko-source:350M --source data/korea-kr-briefing-source:300M ...
Order is priority: when documents duplicate each other, the one from the earlier source (then the
earlier row) is kept.

1. Quality + PII (trid.korean_quality, skip with --no-quality): C4-style line cleanup, ko-pii redaction to
   fixed tokens ([전화] ...), then FineWeb-2 kor_Hang Gopher/FineWeb filters (profile --quality-profile).
   Drop reasons per source go to provenance filter_counts and quality-removed.json.
2. Exact dedup on NFC text with all whitespace removed, across sources.
3. Near dedup: word 5-shingle MinHash (64 components, 16 LSH bands of 4). Bucket pairs whose estimate
   is >= 0.6 get an exact shingle Jaccard check on the re-read texts; >= 0.8 drops the later document.
   Only signatures (512 B/doc) and row offsets stay in memory. Recall is approximate.
4. Split by `group` hash so versions and copies of one item never straddle train/valid/test.
5. Per-source token budget: deterministic hash-priority sample if the source is larger; never upsampled.
"""
import argparse, hashlib, json, re, sys, time, unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.corpus import sha256, write_json
from trid.korean_quality import PROFILES, assess_batch, clean_lines, redact
from trid.text import SPECIAL

PERMS, BANDS = 64, 16
ROWS = PERMS // BANDS
THRESHOLD = 0.8
CANDIDATE = 0.6  # signature estimate that triggers an exact check (64 components: sd ~0.05 at 0.8)
MAX_COMPARE = 32  # kept rows checked per bucket; bounds work on large template buckets
DEFAULT_CHARS_PER_TOKEN = 1.79  # measured on data/korean-foundation-tokens-v2 with the 8k tokenizer
CHUNK = 64  # rows per Kiwi batch (multi-threaded tokenization)
WORD = re.compile(r"\w+")
SPACE = re.compile(r"\s+")
MASK = np.uint64((1 << 61) - 1)


def budget(value):
    """'350M' / '1.2B' / '5000' -> token count."""
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(value[-1:].upper(), 1)
    return int(float(value[:-1] if scale != 1 else value) * scale)


def normalized(row):
    return unicodedata.normalize("NFC", row.get("text") or "").strip()


def prepared(row, quality):
    """Text as it will be written: NFC, then (if quality) line cleanup + PII tokens. (text, lines_dropped, pii)."""
    text = normalized(row)
    if quality is None:
        return text, 0, Counter()
    text, lines = clean_lines(text)
    text, pii = redact(text)
    return text, lines, pii


def read_chunks(f):
    """(offset, line_number, row) batches of CHUNK non-blank rows from docs.jsonl."""
    n = 0
    while True:
        batch = []
        while len(batch) < CHUNK:
            offset, line = f.tell(), f.readline()
            if not line:
                break
            n += 1
            if line.strip():
                batch.append((offset, n, json.loads(line)))
        if not batch:
            return
        yield batch


def parse_source(spec):
    path, _, size = spec.rpartition(":")
    if not path:
        raise argparse.ArgumentTypeError(f"expected DIR:TOKENS, got {spec!r}")
    return Path(path), budget(size)


def shingles(text):
    words = WORD.findall(text.lower())
    return {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def jaccard(a, b):
    a, b = shingles(a), shingles(b)
    return len(a & b) / len(a | b)


class Signer:
    def __init__(self, seed=20261027):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**61 - 1, size=PERMS, dtype=np.uint64)
        self.b = rng.integers(0, 2**61 - 1, size=PERMS, dtype=np.uint64)

    def __call__(self, text):
        parts = shingles(text)
        x = np.fromiter((int.from_bytes(hashlib.blake2b(p.encode(), digest_size=8).digest(), "little") >> 3
                         for p in parts), dtype=np.uint64, count=len(parts))
        with np.errstate(over="ignore"):
            return ((x[:, None] * self.a + self.b) & MASK).min(axis=0)


def near_duplicates(signatures, exact):
    """Rows (priority order) whose exact shingle Jaccard with an earlier kept row sharing an LSH
    bucket is >= THRESHOLD. `exact(i, k)` computes it from the texts; only pairs whose signature
    estimate is >= CANDIDATE are checked. Returns {dropped: (kept, jaccard)}."""
    n = len(signatures)
    dropped, checked = {}, set()
    for band in range(BANDS):
        block = signatures[:, band * ROWS:(band + 1) * ROWS]
        keys = np.frombuffer(np.ascontiguousarray(block).tobytes(), dtype=f"V{8 * ROWS}")
        order = np.argsort(keys, kind="stable")  # stable: equal keys stay in priority order
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            kept = []
            for m in map(int, order[s:e]):
                if m in dropped:
                    continue
                for k in kept[:MAX_COMPARE]:
                    if (k, m) in checked:
                        continue
                    checked.add((k, m))
                    if float((signatures[m] == signatures[k]).mean()) < CANDIDATE:
                        continue
                    j = exact(m, k)
                    if j >= THRESHOLD:
                        dropped[m] = (k, j)
                        break
                else:
                    kept.append(m)
    return dropped


def split_for_group(group, holdout):
    bucket = int(hashlib.sha256(f"split:{group}".encode()).hexdigest()[:8], 16) % 10000
    return "test" if bucket < holdout * 50 else "valid" if bucket < holdout * 100 else "train"


def build(sources, output, holdout_percent, quality="public-docs"):
    """quality: profile name from trid.korean_quality.PROFILES, or None to skip the quality/PII stage."""
    start = time.time()
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output folder.")
    thresholds = PROFILES[quality] if quality else None
    # Pass 1: keep only per-row metadata, file offsets and signatures in memory.
    rows, meta, exact, counts, sigs, quality_removed = [], [], set(), Counter(), [], []
    signer = Signer()
    for index, (path, tokens) in enumerate(sources):
        stats_path = path / "pilot-stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        rate = stats.get("chars_per_token_sample") or stats.get("chars_per_token") or DEFAULT_CHARS_PER_TOKEN
        meta.append({"path": str(path), "token_budget": tokens, "chars_per_token": rate,
                     "docs_sha256": sha256(path / "docs.jsonl")})
        before = len(rows)
        with (path / "docs.jsonl").open("rb") as f:
            for batch in read_chunks(f):
                texts = []
                for offset, n, row in batch:
                    raw = normalized(row)
                    if not raw or any(s in raw for s in SPECIAL):
                        raise ValueError(f"invalid document: {path}/docs.jsonl:{n}")
                    if not row.get("group") or not row.get("license"):
                        raise ValueError(f"missing group/license: {path}/docs.jsonl:{n}")
                    text, lines, pii = prepared(row, thresholds)
                    counts[f"{path.name}:lines_removed"] += lines
                    for label, k in pii.items():
                        counts[f"{path.name}:pii:{label}"] += k
                    texts.append(text)
                verdicts = assess_batch(texts, thresholds) if thresholds else [(True, "ok")] * len(texts)
                for (offset, n, row), text, (ok, reason) in zip(batch, texts, verdicts):
                    if not ok:
                        counts[f"{path.name}:quality:{reason}"] += 1
                        quality_removed.append({"id": row["id"], "source": str(path), "reason": reason})
                        continue
                    key = hashlib.sha256(SPACE.sub("", text).encode()).digest()
                    if key in exact:
                        counts[f"{path.name}:exact_duplicate"] += 1
                        continue
                    exact.add(key)
                    rows.append((index, offset, row["id"], row["group"], row["license"], len(text)))
                    sigs.append(signer(text))
                    if len(rows) % 50000 == 0:
                        print(f"signed {len(rows)}", flush=True)
        print(f"read {path}: {len(rows) - before} docs", flush=True)
    del exact
    signatures = np.vstack(sigs) if sigs else np.empty((0, PERMS), dtype=np.uint64)
    del sigs
    handles = [(Path(m["path"]) / "docs.jsonl").open("rb") for m in meta]

    def text_of(i):
        handles[rows[i][0]].seek(rows[i][1])
        return prepared(json.loads(handles[rows[i][0]].readline()), thresholds)[0]

    try:
        dropped = near_duplicates(signatures, lambda i, k: jaccard(text_of(i), text_of(k)))
    finally:
        for f in handles:
            f.close()
    del signatures
    removed = [{"id": rows[i][2], "source": meta[rows[i][0]]["path"], "matches": rows[k][2],
                "jaccard": round(j, 3)} for i, (k, j) in sorted(dropped.items())]
    for i in dropped:
        counts[f"{Path(meta[rows[i][0]]['path']).name}:near_duplicate"] += 1
    # Per-source budget: hash priority by group first, so a group's rows sit together and are
    # kept or dropped together (except possibly at the budget edge).
    selected, used = [], [0] * len(sources)
    candidates = sorted((r[0], hashlib.sha256(f"mix:{r[3]}".encode()).hexdigest(), i)
                        for i, r in enumerate(rows) if i not in dropped)
    for idx, _, i in candidates:
        est = rows[i][5] / meta[idx]["chars_per_token"]
        if used[idx] + est > meta[idx]["token_budget"]:
            counts[f"{Path(meta[idx]['path']).name}:over_budget"] += 1
            continue
        used[idx] += est
        selected.append(i)
    selected.sort()
    # Pass 2: re-read selected rows by offset and write them in source/row order.
    out.mkdir(parents=True, exist_ok=True)
    files = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "valid", "test")}
    handles = [(Path(m["path"]) / "docs.jsonl").open("rb") for m in meta]
    split_counts, licenses = Counter(), [set() for _ in meta]
    try:
        for i in selected:
            idx, offset, _, group, license, _ = rows[i]
            handles[idx].seek(offset)
            row = json.loads(handles[idx].readline())
            row["text"] = prepared(row, thresholds)[0]
            split = split_for_group(group, holdout_percent)
            files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            split_counts[f"{Path(meta[idx]['path']).name}:{split}"] += 1
            split_counts[split] += 1
            licenses[idx].add(license)
    finally:
        for f in [*files.values(), *handles]:
            f.close()
    for idx, m in enumerate(meta):
        m["estimated_tokens_selected"] = round(used[idx])
        m["shortfall_tokens"] = max(0, m["token_budget"] - round(used[idx]))
        m["licenses"] = sorted(licenses[idx])
    write_json(out / "removed.json", removed)
    write_json(out / "quality-removed.json", quality_removed)
    write_json(out / "provenance.json", {
        "sources": meta, "split_counts": dict(split_counts), "filter_counts": dict(counts),
        "holdout_percent_each": holdout_percent,
        "quality": {
            "profile": quality, "thresholds": asdict(thresholds) if thresholds else None,
            "method": "trid.korean_quality: C4-style line cleanup (javascript/site-menu/1000-char-word lines), "
                      "ko-pii (MIT) redaction of RRN/FRN/PHONE/EMAIL/ACCOUNT/CARD/PASSPORT/DRIVER_LICENSE to fixed "
                      "tokens (hits must match the label's format and not be glued to digits/letters), then FineWeb-2 "
                      "kor_Hang Gopher repetition/quality + FineWeb quality filters with kiwipiepy (Apache-2.0) morpheme "
                      "words and Hangul/Hanja/Latin ratio checks; applied before dedup. Thresholds: "
                      "https://raw.githubusercontent.com/huggingface/fineweb-2/main/configs/kor_Hang.yml (public-docs "
                      "profile: dup_n_grams x2, min_alpha_words 0.7, max_words 1M; see trid/korean_quality.py). "
                      "filter_counts keys: <source>:quality:<reason>, <source>:pii:<label>, <source>:lines_removed."}
        if thresholds else None,
        "deduplication": f"NFC + whitespace-free exact match across sources, then word 5-shingle MinHash "
                         f"({PERMS} components, {BANDS} bands of {ROWS}); bucket pairs with estimate >= {CANDIDATE} "
                         f"get an exact shingle Jaccard check, >= {THRESHOLD} drops the later document by source "
                         f"priority. Recall approximate; precision exact.",
        "split_method": "sha256('split:'+group) buckets, so all versions/copies of a group share a split",
        "selection": "per-source token budget, hash-priority sample, estimated with chars_per_token; no upsampling",
        "file_sha256": {s: sha256(out / f"{s}.jsonl") for s in files},
        "wall_seconds": round(time.time() - start, 1)})
    (out / "ATTRIBUTION.txt").write_text("\n\n".join(
        f"== {m['path']} ==\n" + ((Path(m["path"]) / "ATTRIBUTION.txt").read_text(encoding="utf-8")
                                  if (Path(m["path"]) / "ATTRIBUTION.txt").exists() else "(missing)")
        for m in meta) + "\n", encoding="utf-8")
    if not split_counts["valid"] or not split_counts["test"]:
        raise ValueError("empty valid/test split: raise --holdout-percent or add sources")
    print(json.dumps({"output": str(out), "splits": {s: split_counts[s] for s in files},
                      "near_duplicates": len(dropped), "quality_removed": len(quality_removed),
                      "sources": [{k: m[k] for k in
                      ("path", "token_budget", "estimated_tokens_selected", "shortfall_tokens")} for m in meta]},
                     ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", action="append", type=parse_source, required=True, help="DIR:TOKENS, priority order")
    p.add_argument("--output", required=True)
    p.add_argument("--holdout-percent", type=float, default=1.0, help="percent of groups for valid and for test, each")
    p.add_argument("--quality-profile", choices=sorted(PROFILES), default="public-docs")
    p.add_argument("--no-quality", action="store_true", help="skip the quality/PII stage (no kiwipiepy/ko-pii needed)")
    a = p.parse_args()
    if not 0 < a.holdout_percent <= 10:
        raise ValueError("--holdout-percent must be in (0, 10]")
    build(a.source, a.output, a.holdout_percent, None if a.no_quality else a.quality_profile)


if __name__ == "__main__":
    main()

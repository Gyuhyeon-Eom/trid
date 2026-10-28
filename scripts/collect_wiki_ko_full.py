"""Collect every eligible Korean Wikipedia article (all 3 shards of the 20231101.ko snapshot).

Writes the shared pretraining-source layout used by collect_law_go_kr.py and collect_korea_briefing.py:
DIR/raw/*.parquet, DIR/docs.jsonl, DIR/provenance.json, DIR/ATTRIBUTION.txt, DIR/pilot-stats.json.
Filters match trid.corpus.fetch_wiki (>=300 chars, >=20% Hangul, <=50k chars, no control tokens), so
documents stay comparable with data/korean-foundation-v2 (shard 1 only).
"""
import argparse, hashlib, json, os, random, re, shutil, sys, time, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.corpus import BASE, REVISION, download, sha256, write_json
from trid.text import SPECIAL

SHARDS = {  # path -> sha256, from the Hugging Face tree API at REVISION
    "20231101.ko/train-00000-of-00003.parquet": "28ff69a402878eba3be42dfd187f37876b917b72eb9b05896ed7469fa681bb03",
    "20231101.ko/train-00001-of-00003.parquet": "60795ca350aec5c6813651e181c8e506a1c1b1461d56575b54c25632ed28dd14",
    "20231101.ko/train-00002-of-00003.parquet": "5e16f59e25871042fcd2bb1ca8540085e02a4a88177df95447aefe9420e2be64",
}
CACHED = ROOT / "data/wiki-ko-pilot/source.parquet"  # shard 0, already downloaded for v1/v2
TOKENIZER = ROOT / "runs/trid-korean-medium-base-v1/tokenizer.json"
LICENSE = "CC-BY-SA-4.0/GFDL (Wikipedia)"
ATTRIBUTION = "출처: 한국어 위키백과 기여자 (2023-11-01 스냅숏), 문서별 기록은 source?action=history"
HANGUL = re.compile("[가-힣]")


def keep(text):
    """Cleaned text or None, with the same rules as trid.corpus.fetch_wiki."""
    text = unicodedata.normalize("NFC", text).replace("\x00", "").strip()
    if len(text) < 300 or len(HANGUL.findall(text)) / len(text) < .2:
        return None, "short_or_low_hangul"
    if len(text) > 50000:
        return None, "too_long"
    if any(s in text for s in SPECIAL):
        return None, "control_token"
    return text, None


def fetch_shards(raw):
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, digest in SHARDS.items():
        path = raw / Path(name).name
        if not path.exists():
            if name.endswith("00000-of-00003.parquet") and CACHED.exists() and sha256(CACHED) == digest:
                shutil.copyfile(CACHED, path)
            else:
                print(f"download {name}", flush=True)
                download(f"{BASE}/{name}", path, digest)
        elif sha256(path) != digest:
            raise ValueError(f"체크섬 불일치: {path}")
        paths.append(path)
    return paths


def collect(output, limit=None):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    root = Path(output)
    start = time.time()
    shards = fetch_shards(root / "raw")
    if not (root / "SOURCE_DATASET_CARD.md").exists():
        download(f"{BASE}/README.md", root / "SOURCE_DATASET_CARD.md")
    seen, dropped, kept, chars, sample = set(), {}, 0, 0, []
    rng = random.Random(0)
    part = root / "docs.jsonl.part"
    with part.open("w", encoding="utf-8") as out:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["id", "url", "title", "text"]):
                for row in batch.to_pylist():
                    text, reason = keep(row["text"])
                    if text is not None:
                        digest = hashlib.sha256(text.encode()).hexdigest()
                        if digest in seen:
                            text, reason = None, "exact_duplicate"
                    if text is None:
                        dropped[reason] = dropped.get(reason, 0) + 1
                        continue
                    seen.add(digest)
                    out.write(json.dumps({"id": f"wiki-ko-{row['id']}", "title": row["title"], "source": row["url"],
                                          "text": text, "sha256": digest, "license": LICENSE,
                                          "attribution": ATTRIBUTION, "group": f"wiki:{row['id']}", "date": None,
                                          "kind": "wikipedia", "contributors": row["url"] + "?action=history"},
                                         ensure_ascii=False) + "\n")
                    kept += 1; chars += len(text)
                    if len(sample) < 3000:  # reservoir sample for the chars/token estimate
                        sample.append(text)
                    elif rng.random() < 3000 / kept:
                        sample[rng.randrange(3000)] = text
                    if limit and kept >= limit:
                        break
                if limit and kept >= limit:
                    break
            if limit and kept >= limit:
                break
    os.replace(part, root / "docs.jsonl")
    tok = Tokenizer.from_file(str(TOKENIZER))
    sample_tokens = sum(len(e.ids) for e in tok.encode_batch(sample))
    rate = sum(map(len, sample)) / sample_tokens
    stats = {"documents_kept": kept, "dropped": dropped, "characters": chars,
             "chars_per_token_sample": round(rate, 3), "sample_documents": len(sample),
             "estimated_tokens": round(chars / rate), "tokenizer": str(TOKENIZER.relative_to(ROOT)),
             "exact_duplicate_rate": round(dropped.get("exact_duplicate", 0) / max(1, kept + dropped.get("exact_duplicate", 0)), 5),
             "limit": limit, "wall_seconds": round(time.time() - start, 1)}
    write_json(root / "pilot-stats.json", stats)
    write_json(root / "provenance.json", {
        "dataset": "wikimedia/wikipedia", "configuration": "20231101.ko", "revision": REVISION,
        "shards": {name: {"url": f"{BASE}/{name}", "sha256": digest} for name, digest in SHARDS.items()},
        "dataset_card_license_labels": ["cc-by-sa-3.0", "gfdl"],
        "current_wikimedia_license_reference": "https://dumps.wikimedia.org/legal.html",
        "current_wikimedia_text_license": "CC BY-SA 4.0; see source terms and exceptions",
        "changes": "NFC normalization, null-character removal, whitespace trim; filters as trid.corpus.fetch_wiki",
        "deduplication": "exact text sha256 within this source; near-duplicate filtering happens at mixing time",
        "docs_sha256": sha256(root / "docs.jsonl"), "documents": kept, "real_company_data": False})
    (root / "ATTRIBUTION.txt").write_text(
        "Source: Korean Wikipedia, 2023-11-01 snapshot (all 3 shards), via wikimedia/wikipedia.\n"
        "Authors: Wikipedia contributors. Article and history links are retained in each JSONL row.\n"
        "Dataset card labels: CC BY-SA 3.0 and GFDL. Current Wikimedia text terms refer to CC BY-SA 4.0.\n"
        "https://creativecommons.org/licenses/by-sa/4.0/\nhttps://dumps.wikimedia.org/legal.html\n"
        "Changes: NFC normalization, null removal, trimming, filtering, exact deduplication.\n"
        "Whether ShareAlike reaches trained weights is an open legal question.\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default="data/wiki-ko-full-source")
    p.add_argument("--limit", type=int, help="pilot: stop after N kept documents")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    root = Path(a.output)
    if a.status:
        stats = root / "pilot-stats.json"
        print(stats.read_text() if stats.exists() else json.dumps({"status": "not collected",
              "shards_cached": sorted(x.name for x in (root / "raw").glob("*.parquet")) if (root / "raw").exists() else []}))
        return
    collect(root, a.limit)


if __name__ == "__main__":
    main()

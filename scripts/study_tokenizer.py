"""Tokenizer study for the 0.2 pretraining run: vocab size × pre-tokenizer on a wiki/law/briefing mix.

Trains byte-level BPE tokenizers (HF `tokenizers`) on a mixed sample, holds out documents per domain,
and measures compression, byte-fallback rate, RAG-eval prompt length and gold-span token-boundary
alignment. Writes a JSON report, prints the markdown tables, and optionally copies one tokenizer to
`runs/tokenizers/<name>/` with a provenance file.

Usage (CPU only; heavy, run under `nice`):
  nice -n 10 .venv/bin/python scripts/study_tokenizer.py \
      --law /tmp/trid-tok/law.part --briefing /tmp/trid-tok/briefing.part \
      --work /tmp/trid-tok/tokenizers --output reports/tokenizer-study-v1.json \
      --save l3-8192 --save-dir runs/tokenizers/trid-ko-mix-v1-l3-8192
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from trid.text import SPECIAL  # noqa: E402

# Llama-3 style pre-tokenization regex, Korean variant used in reports/korean-oss-opportunities.md §1.3:
# numbers are split into runs of at most three digits (so `1,500,000` → ` 1`,`,`,`500`,`,`,`000`), and a
# single leading space may attach to a number run (Llama 3 proper uses `\p{N}{1,3}` without the space,
# which would make every ` 15일` cost an extra space token in Korean prose). The English contraction
# alternative of the original pattern is dropped: it never matches Korean text.
LLAMA3_KO_PATTERN = r"[^\r\n\p{L}\p{N}]?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
# GPT-2 regex (what tokenizers' ByteLevel uses, minus English contractions) with only the number rule changed
# to ≤3 digits. Unlike the Llama-3 pattern it never glues a punctuation mark to the following word (`%를`,
# `)은`, `.\n`), which is what breaks token boundaries at the end of values like `102%` or `2026. 12. 18.`.
DIGIT_SPLIT_PATTERN = r" ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def regex_pre_tokenizer(pattern):
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


PRE_TOKENIZERS = {
    # Current trid tokenizer (trid/text.py, trid/corpus.py): GPT-2 regex inside ByteLevel.
    "bl": lambda: pre_tokenizers.ByteLevel(add_prefix_space=False),
    "l3": lambda: regex_pre_tokenizer(LLAMA3_KO_PATTERN),
    "ds": lambda: regex_pre_tokenizer(DIGIT_SPLIT_PATTERN),
}
PRE_TOKENIZER_LABEL = {"bl": "ByteLevel(GPT-2 regex, 현행)", "l3": "Llama-3식 regex, 숫자 ≤3자리",
                       "ds": "GPT-2 regex, 숫자만 ≤3자리"}
PRE_TOKENIZER_PATTERN = {"bl": "GPT-2 (tokenizers ByteLevel use_regex=True)", "l3": LLAMA3_KO_PATTERN, "ds": DIGIT_SPLIT_PATTERN}

DIM = 768
NON_EMBEDDING_PARAMS = 91_245_312 - 8192 * DIM  # README: 91,245,312 total at vocab 8,192, tied embedding
TAO = {  # Tao et al. 2024, arXiv 2407.13623 (NeurIPS 2024)
    "approach1": {"k_nv": 0.08, "a_nv": 0.50, "k_v": 0.20, "a_v": 0.42},  # Fig. 5: N_nv = 0.08·C^0.50, N_v = 0.20·C^0.42
    "approach3": {"E": 5.533, "A1": 1.831, "alpha1": 0.447, "A2": 0.196, "alpha2": 0.671, "B": 2.124, "beta": 0.447},  # §4.3
    "f_english": {"a": 0.0064, "b": -0.1581, "c": 1.2047},  # §2.2 Eq. 3: tokens/char = a·ln²V + b·lnV + c
}


def build_tokenizer(pre, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = PRE_TOKENIZERS[pre]()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    return tok, trainer


def bytes_to_unicode():
    """GPT-2 byte→printable-char table used by HF ByteLevel (inverse recovers token bytes)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def partial_char_token_ids(tok):
    """Ids of vocabulary entries whose bytes are not valid UTF-8 on their own (byte-fallback fragments)."""
    inverse = {v: k for k, v in bytes_to_unicode().items()}
    special = set(SPECIAL)
    ids = set()
    for token, tid in tok.get_vocab().items():
        if token in special:
            continue
        try:
            bytes(inverse[c] for c in token).decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            ids.add(tid)
    return ids


def hangul_syllable_coverage(tok):
    """Distinct precomposed Hangul syllables (가–힣) that are a whole token, with or without a leading space."""
    inverse = {v: k for k, v in bytes_to_unicode().items()}
    found = set()
    for token in tok.get_vocab():
        if token in SPECIAL:
            continue
        try:
            text = bytes(inverse[c] for c in token).decode("utf-8").lstrip(" ")
        except (UnicodeDecodeError, KeyError):
            continue
        if len(text) == 1 and "가" <= text <= "힣":
            found.add(text)
    return len(found)


def digit_group_coverage(tok):
    """How many of the 10/100/1,000 one-/two-/three-digit strings encode to a single token (with a leading space)."""
    out = {}
    for width in (1, 2, 3):
        strings = [f" {n:0{width}d}" for n in range(10 ** width)]
        out[str(width)] = sum(len(e.ids) == 1 for e in tok.encode_batch(strings))
    return out


def token_boundaries(tok, text):
    """Character positions where a token ends (cumulative decoded bytes landing on a UTF-8 char boundary)."""
    byte_to_char = {}
    pos = 0
    for i, ch in enumerate(text):
        byte_to_char[pos] = i
        pos += len(ch.encode("utf-8"))
    byte_to_char[pos] = len(text)
    bounds = {0}
    cursor = 0
    for tid in tok.encode(text).ids:
        token = tok.id_to_token(tid)  # byte-level alphabet: one char per byte
        cursor += len(token.encode("utf-8")) if token in SPECIAL else len(token)
        if cursor in byte_to_char:
            bounds.add(byte_to_char[cursor])
    return bounds


def span_alignment(bounds, text, start, end):
    """A span is extractable when a token starts at it (or at one leading space before it) and a token ends at its end."""
    start_ok = start in bounds or (start > 0 and text[start - 1] == " " and (start - 1) in bounds)
    return start_ok, end in bounds


def scan(path, domain):
    """First pass: metadata and byte offsets for every document line (texts are loaded later by offset)."""
    rows = []
    bad = 0
    with Path(path).open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row["text"]
            except (ValueError, KeyError):
                bad += 1  # the crawl files are copied mid-write; the last line may be truncated
                continue
            if not isinstance(text, str) or not text.strip() or any(s in text for s in SPECIAL):
                bad += 1
                continue
            rows.append({"domain": domain, "id": str(row["id"]), "group": str(row.get("group") or row["id"]),
                         "kind": row.get("kind") or domain, "chars": len(text), "offset": offset})
    return rows, bad


def load_texts(path, rows):
    out = {}
    with Path(path).open("rb") as f:
        for r in sorted(rows, key=lambda r: r["offset"]):
            f.seek(r["offset"])
            out[r["id"]] = json.loads(f.readline())["text"]
    return out


def split_domain(rows, holdout_docs, budget, rng):
    """Hold out whole groups (law versions / press ids) until ≥ holdout_docs, then sample the rest by kind share."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    order = sorted(groups)
    rng.shuffle(order)
    holdout, pool, n = [], [], 0
    for g in order:
        if n < holdout_docs:
            holdout.extend(groups[g])
            n += len(groups[g])
        else:
            pool.extend(groups[g])
    by_kind = defaultdict(list)
    for r in pool:
        by_kind[r["kind"]].append(r)
    total_chars = sum(r["chars"] for r in pool)
    sample = []
    for kind, rs in sorted(by_kind.items()):
        kind_budget = budget * sum(r["chars"] for r in rs) / total_chars
        rng.shuffle(rs)
        used = 0
        for r in rs:
            if used + r["chars"] > kind_budget:
                continue
            sample.append(r)
            used += r["chars"]
    return holdout, sample


def composition(rows):
    kinds = Counter()
    chars = Counter()
    for r in rows:
        kinds[r["kind"]] += 1
        chars[r["kind"]] += r["chars"]
    return {"docs": len(rows), "chars": sum(chars.values()),
            "by_kind": {k: {"docs": kinds[k], "chars": chars[k]} for k in sorted(kinds)}}


def tao_estimates(total_tokens, chars_per_token_by_vocab):
    """Tao et al. 2024 optimal-vocabulary estimates for trid (N_nv fixed, d = 768)."""
    n_total = NON_EMBEDDING_PARAMS + 8192 * DIM
    compute = 6 * n_total * total_tokens
    a1 = TAO["approach1"]
    nv_opt = a1["k_v"] * compute ** a1["a_v"]
    nnv_opt = a1["k_nv"] * compute ** a1["a_nv"]
    out = {"compute_flops": compute, "approach1": {"N_v_opt": nv_opt, "V_opt": nv_opt / DIM, "N_nv_opt": nnv_opt}}
    # Approach 3 (Eq. 8/10) needs tokens-per-char f(V). Fit the paper's quadratic-in-ln(V) form to the measured
    # Korean mix (a is kept from the paper; b, c re-fit by least squares so the curve passes through our points).
    p = TAO["approach3"]
    vs = sorted(chars_per_token_by_vocab)
    xs = [math.log(v) for v in vs]
    ys = [1 / chars_per_token_by_vocab[v] for v in vs]
    a = TAO["f_english"]["a"]
    # least squares for b, c in y - a x² = b x + c (with one point only, keep the paper's b and shift c)
    yy = [y - a * x * x for x, y in zip(xs, ys)]
    mx, my = sum(xs) / len(xs), sum(yy) / len(yy)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, yy)) / sxx if sxx else TAO["f_english"]["b"]
    c = my - b * mx
    f_ko = lambda v: a * math.log(v) ** 2 + b * math.log(v) + c  # noqa: E731
    chars = total_tokens * chars_per_token_by_vocab[min(vs, key=lambda v: abs(v - 8192))]  # 1B-token budget in characters

    def loss(v, fixed):
        nv = v * DIM
        d = chars * f_ko(v) if fixed == "chars" else compute / (6 * (NON_EMBEDDING_PARAMS + nv))
        return (-p["E"] + p["A1"] * NON_EMBEDDING_PARAMS ** -p["alpha1"] + p["A2"] * nv ** -p["alpha2"]
                + p["B"] * d ** -p["beta"])

    grid = [2 ** (k / 8) for k in range(8 * 10, 8 * 18)]  # 1k … 256k
    out["f_korean"] = {"a": a, "b": b, "c": c, "fit_points": {str(v): ys[i] for i, v in enumerate(vs)}}
    out["approach3"] = {}
    for fixed in ("chars", "flops"):
        best = min(grid, key=lambda v: loss(v, fixed))
        out["approach3"][f"fixed_{fixed}"] = {"V_opt": best, "loss_at": {str(v): loss(v, fixed) for v in (4096, 8192, 12288, 16384, 32768)}}
    return out


def evaluate(name, tok, holdout_texts, eval_rows):
    partial = partial_char_token_ids(tok)
    domains = {}
    for domain, texts in holdout_texts.items():
        chars = tokens = fallback = mismatch = 0
        for enc, text in zip(tok.encode_batch(texts), texts):
            chars += len(text)
            tokens += len(enc.ids)
            fallback += sum(1 for i in enc.ids if i in partial)
            mismatch += tok.decode(enc.ids) != text
        domains[domain] = {"docs": len(texts), "chars": chars, "tokens": tokens, "chars_per_token": chars / tokens,
                           "byte_fallback_pct": 100 * fallback / tokens, "roundtrip_mismatches": mismatch}
    prompt_tokens = [len(e.ids) for e in tok.encode_batch([r["prompt"] for r in eval_rows])]
    align = Counter()
    by_task = defaultdict(Counter)
    for r in eval_rows:
        if not r["gold"]:
            continue
        bounds = token_boundaries(tok, r["prompt"])
        text = r["prompt"]
        for variants in r["gold"]:
            starts = ends = both = 0
            found = False
            for value in variants:
                i = text.find(value)
                while i >= 0:
                    found = True
                    s_ok, e_ok = span_alignment(bounds, text, i, i + len(value))
                    starts |= s_ok
                    ends |= e_ok
                    both |= s_ok and e_ok
                    i = text.find(value, i + 1)
            if not found:
                continue
            for counter in (align, by_task[r["task"]]):
                counter["values"] += 1
                counter["start"] += starts
                counter["end"] += ends
                counter["both"] += both
    gold = {"values": align["values"], "start_aligned_pct": 100 * align["start"] / align["values"],
            "end_aligned_pct": 100 * align["end"] / align["values"], "both_aligned_pct": 100 * align["both"] / align["values"],
            "by_task_both_aligned_pct": {t: 100 * c["both"] / c["values"] for t, c in sorted(by_task.items())},
            "by_task_values": {t: c["values"] for t, c in sorted(by_task.items())}}
    vocab = tok.get_vocab_size()
    return {"name": name, "vocab_size": vocab, "domains": domains,
            "eval_prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
            "eval_prompt_tokens_max": max(prompt_tokens),
            "eval_prompts_over_1024": sum(t > 1024 for t in prompt_tokens),
            "gold_boundaries": gold, "partial_char_vocab_entries": len(partial),
            "hangul_syllables_in_vocab": hangul_syllable_coverage(tok),
            "digit_groups_single_token": digit_group_coverage(tok),
            "embedding_params": vocab * DIM, "total_params_tied": NON_EMBEDDING_PARAMS + vocab * DIM}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_tables(report):
    domains = list(next(iter(report["tokenizers"].values()))["domains"])
    lines = ["| 토크나이저 | 어휘 | " + " | ".join(f"{d} 자/토큰" for d in domains) + " | byte-fallback % (" + "/".join(domains)
             + ") | 한글 음절 토큰 | 2·3자리 수 단일토큰 | 평가 프롬프트 토큰(평균/최대) | 값 경계 시작/끝/둘 다 % | 임베딩 params | 전체 params |",
             "|---|---|" + "---|" * (len(domains) + 7)]
    for name, r in report["tokenizers"].items():
        g = r["gold_boundaries"]
        dg = r["digit_groups_single_token"]
        lines.append(f"| {name} | {r['vocab_size']:,} | " + " | ".join(f"{r['domains'][d]['chars_per_token']:.3f}" for d in domains)
                     + " | " + "/".join(f"{r['domains'][d]['byte_fallback_pct']:.1f}" for d in domains)
                     + f" | {r['hangul_syllables_in_vocab']:,} | {dg['2']}/100 · {dg['3']}/1000"
                     + f" | {r['eval_prompt_tokens_mean']:.0f} / {r['eval_prompt_tokens_max']} | {g['start_aligned_pct']:.1f} / {g['end_aligned_pct']:.1f} / {g['both_aligned_pct']:.1f}"
                     + f" | {r['embedding_params']:,} | {r['total_params_tied']:,} |")
    tasks = sorted({t for r in report["tokenizers"].values() for t in r["gold_boundaries"]["by_task_both_aligned_pct"]})
    lines += ["", "| 토크나이저 | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for name, r in report["tokenizers"].items():
        bt = r["gold_boundaries"]["by_task_both_aligned_pct"]
        lines.append(f"| {name} | " + " | ".join(f"{bt.get(t, float('nan')):.0f}" for t in tasks) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wiki", default="data/korean-foundation-v2/train.jsonl")
    p.add_argument("--law", required=True, help="/tmp copy of data/law-ko-source/docs.jsonl.part")
    p.add_argument("--briefing", required=True, help="/tmp copy of data/korea-kr-briefing-source/docs.jsonl.part")
    p.add_argument("--eval", default="data/korean-rag-eval-v1/test.jsonl")
    p.add_argument("--baseline", default="/tmp/trid-base-snap/tokenizer.json", help="current wiki-only tokenizer")
    p.add_argument("--work", required=True, help="directory for trained tokenizers (keep under /tmp)")
    p.add_argument("--output", required=True, help="JSON report path (new file)")
    p.add_argument("--chars", type=int, default=60_000_000, help="training sample size in characters")
    p.add_argument("--mix", default="wiki=0.30,law=0.45,briefing=0.25", help="character share per domain")
    p.add_argument("--holdout", type=int, default=2000, help="held-out documents per domain")
    p.add_argument("--vocabs", default="8192,12288,16384")
    p.add_argument("--pre", default="bl,l3,ds")
    p.add_argument("--tokens", type=float, default=1e9, help="planned pretraining tokens for the Tao et al. estimate")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--save", help="tokenizer name (e.g. l3-8192) to copy into --save-dir with provenance.json")
    p.add_argument("--save-dir")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"Use a new report path: {out}")
    if bool(a.save) != bool(a.save_dir):
        p.error("--save and --save-dir go together")
    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    if abs(sum(mix.values()) - 1) > 1e-6 or set(mix) != {"wiki", "law", "briefing"}:
        p.error("--mix must give wiki, law and briefing shares summing to 1")
    sources = {"wiki": a.wiki, "law": a.law, "briefing": a.briefing}
    rng = random.Random(a.seed)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scanned, bad_lines, holdout_rows, sample_rows = {}, {}, {}, {}
    for domain, path in sources.items():
        rows, bad_lines[domain] = scan(path, domain)
        scanned[domain] = composition(rows)
        holdout_rows[domain], sample_rows[domain] = split_domain(rows, a.holdout, a.chars * mix[domain], rng)
        print(f"{domain}: {len(rows)} docs scanned, holdout {len(holdout_rows[domain])}, sample {composition(sample_rows[domain])}", flush=True)
    holdout_texts, sample_texts = {}, {}
    for domain, path in sources.items():
        texts = load_texts(path, holdout_rows[domain] + sample_rows[domain])
        holdout_texts[domain] = [texts[r["id"]] for r in holdout_rows[domain]]
        sample_texts[domain] = [texts[r["id"]] for r in sample_rows[domain]]
    sample_ids = {d: sorted(r["id"] for r in rows) for d, rows in sample_rows.items()}
    assert all(not set(sample_ids[d]) & {r["id"] for r in holdout_rows[d]} for d in sources)
    sample_sha = hashlib.sha256()
    for domain in sources:
        for text in sample_texts[domain]:
            sample_sha.update(text.encode("utf-8"))
            sample_sha.update(b"\0")
    eval_rows = [json.loads(l) for l in Path(a.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"data ready in {time.time() - t0:.0f}s", flush=True)

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": a.seed, "mix": mix, "chars_budget": a.chars,
        "sources": {d: {"path": str(path), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
                        "docs_scanned": scanned[d], "bad_lines": bad_lines[d]} for d, path in sources.items()},
        "holdout": {d: composition(rows) for d, rows in holdout_rows.items()},
        "sample": {d: composition(rows) for d, rows in sample_rows.items()},
        "sample_sha256": sample_sha.hexdigest(),
        "eval": {"path": a.eval, "sha256": sha256_file(a.eval), "rows": len(eval_rows), "rows_with_gold": sum(bool(r["gold"]) for r in eval_rows)},
        "pre_tokenizers": {k: {"label": PRE_TOKENIZER_LABEL[k], "pattern": PRE_TOKENIZER_PATTERN[k]} for k in PRE_TOKENIZERS},
        "non_embedding_params": NON_EMBEDDING_PARAMS, "dim": DIM, "tokenizers": {}, "training_seconds": {},
    }
    baseline = Tokenizer.from_file(a.baseline)
    report["tokenizers"]["current-wiki-8192"] = evaluate("current-wiki-8192", baseline, holdout_texts, eval_rows)
    report["tokenizers"]["current-wiki-8192"]["path"] = a.baseline
    report["tokenizers"]["current-wiki-8192"]["sha256"] = sha256_file(a.baseline)
    print(f"baseline evaluated in {time.time() - t0:.0f}s", flush=True)
    corpus = [t for d in sources for t in sample_texts[d]]
    for pre in a.pre.split(","):
        for vocab in (int(v) for v in a.vocabs.split(",")):
            name = f"{pre}-{vocab}"
            t1 = time.time()
            tok, trainer = build_tokenizer(pre, vocab)
            tok.train_from_iterator(corpus, trainer, length=len(corpus))
            (work / name).mkdir(exist_ok=True)
            path = work / name / "tokenizer.json"
            tok.save(str(path))
            report["training_seconds"][name] = time.time() - t1
            tok = Tokenizer.from_file(str(path))
            report["tokenizers"][name] = evaluate(name, tok, holdout_texts, eval_rows)
            report["tokenizers"][name].update({"pre_tokenizer": pre, "path": str(path), "sha256": sha256_file(path)})
            r = report["tokenizers"][name]
            print(f"{name}: trained {report['training_seconds'][name]:.0f}s, " + ", ".join(f"{d} {v['chars_per_token']:.3f}" for d, v in r["domains"].items())
                  + f", prompt {r['eval_prompt_tokens_mean']:.0f} tok, gold both-aligned {r['gold_boundaries']['both_aligned_pct']:.1f}%", flush=True)
    cpt = {r["vocab_size"]: sum(d["chars"] for d in r["domains"].values()) / sum(d["tokens"] for d in r["domains"].values())
           for n, r in report["tokenizers"].items() if n.startswith("bl-")}
    report["tao_2024"] = {"citation": "Tao et al., Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies, NeurIPS 2024, arXiv:2407.13623",
                          "planned_tokens": a.tokens, **tao_estimates(a.tokens, cpt)}
    if a.save:
        src = Path(report["tokenizers"][a.save]["path"])
        dest = Path(a.save_dir)
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(src, dest / "tokenizer.json")
        (dest / "sample-ids.json").write_text(json.dumps(sample_ids, ensure_ascii=False), encoding="utf-8")
        provenance = {
            "name": a.save, "created": report["created"], "script": "scripts/study_tokenizer.py", "seed": a.seed,
            "tokenizer_sha256": sha256_file(dest / "tokenizer.json"), "vocab_size": report["tokenizers"][a.save]["vocab_size"],
            "pre_tokenizer": PRE_TOKENIZER_LABEL[report["tokenizers"][a.save]["pre_tokenizer"]],
            "pre_tokenizer_pattern": PRE_TOKENIZER_PATTERN[report["tokenizers"][a.save]["pre_tokenizer"]],
            "special_tokens": SPECIAL, "trainer": {"model": "BPE", "min_frequency": 2, "initial_alphabet": "ByteLevel.alphabet()"},
            "sample": {"mix": mix, "chars_budget": a.chars, "composition": report["sample"], "texts_sha256": report["sample_sha256"],
                       "ids_file": "sample-ids.json", "ids_sha256": sha256_file(dest / "sample-ids.json")},
            "sources": report["sources"], "holdout": report["holdout"], "report": str(out),
            "licenses": {"wiki": "CC BY-SA 4.0 (ko.wikipedia)", "law": "KR-Copyright-Act-Art7 (law.go.kr)", "briefing": "KOGL-Type1 (korea.kr)"},
        }
        (dest / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        report["saved"] = {"name": a.save, "dir": str(dest), "tokenizer_sha256": provenance["tokenizer_sha256"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(markdown_tables(report))
    print(json.dumps(report["tao_2024"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

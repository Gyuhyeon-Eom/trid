# mix-v3 full dry-run measurement

## Outcome

Built and tokenized **328,573 documents / 461,393,814 tokens**. The measured gap to one billion tokens is **538,606,186** (all splits); training-only gap is **548,111,649**. This is corpus preparation, not model training or activation.

All complete rows available at the snapshot boundary from law and briefing were processed, followed by the complete wiki-ko-full-source through quality/dedup and an estimated 200M-token selection budget. No foundation-v2 conversion, source truncation, tokenizer training, upsampling, model download, or live-data modification was used. “All” means eligible snapshot rows after filtering/dedup, not every live or nationwide document.

## Inputs and selection

Live append-only files were opened read-only, frozen at their initial byte length, and only newline-terminated JSON rows copied to `/tmp/trid-mix-v3-snapshot-10777`. Both had zero discarded trailing bytes. Subsequent crawler additions are deliberately excluded. Copied pilot metadata supplies rate/attribution only, not current collection totals.

| Source | Input rows | Input bytes | Quality drops | Exact drops | Near drops | Budget drops | Output docs | Actual tokens | Token share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| law-ko-source | 76,278 | 1,231,536,652 | 4,108 | 1 | 362 | 0 | 71,807 | 201,554,058 | 43.68% |
| korea-kr-briefing-source | 52,837 | 424,897,382 | 13,042 | 8 | 197 | 0 | 39,590 | 50,428,486 | 10.93% |
| wiki-ko-full-source | 379,068 | 1,449,719,043 | 125,845 | 0 | 151 | 35,896 | 217,176 | 209,411,270 | 45.39% |

Law/briefing used a deliberately nonbinding `10^15` token budget to express all eligible documents; their huge mechanical `shortfall_tokens` values in provenance are **not meaningful corpus targets**. Wiki selection estimated 199,999,955 tokens using the source pilot rate 1.779 chars/token; actual tokenization produced 209,411,270 including BOS/EOS. Thus `wiki200M` is an **estimated budget, not an exact 200M ceiling**. No claim of strict actual-token compliance is made.

## Splits and tokenization

| Split | Documents | Tokens |
|---|---:|---:|
| train | 322,120 | 451,888,351 |
| valid | 3,219 | 4,852,855 |
| test | 3,234 | 4,652,608 |

`runs/tokenizers/trid-ko-mix-v1-ds-8192/tokenizer.json` was loaded unchanged with `trid.corpus.prepare(..., tokenizer_path=...)`; no tokenizer retraining. Output format is uint16 little-endian with BOS/EOS per document. Source token composition was measured from binary EOS offsets and the builder’s per-source contiguous document counts within each split, not chars/token estimates.

## Quality, PII and dedup

The public-docs profile runs NFC, C4-style line cleanup, format-constrained ko-pii replacement, then Kiwi/FineWeb-style Korean quality/repetition filters on every input row. Quality drops record the first failed rule, not every failing rule. PII counts below cover preprocessing of **all input rows**, including later-discarded documents; they are detector hits, not verified people or output occurrences.

| Source | Removed lines | PII detector replacements by label |
|---|---:|---|
| law-ko-source | 730 | PHONE=649, RRN=21, ACCOUNT=10, CARD=17, PASSPORT=4, FRN=5 |
| korea-kr-briefing-source | 500 | PHONE=224, RRN=20, CARD=21, ACCOUNT=14, FRN=1 |
| wiki-ko-full-source | 1,747 | PHONE=139, EMAIL=131, RRN=36, PASSPORT=3, ACCOUNT=2, FRN=18, CARD=4 |

Exact dedup uses whitespace-free NFC text across sources in law→briefing→wiki order. Near dedup uses 64-component word-five-shingle MinHash, 16 bands of four, estimate ≥0.6 followed by exact Jaccard ≥0.8; maximum 32 comparison candidates per bucket. This is approximate-recall dedup, **not proof that every near duplicate is absent**, and not semantic/translation dedup. All 710 reported removals met the stored ≥0.8 threshold.

### Independently exercised invariants

- Scanned all 328,573 emitted rows: zero normalized exact duplicates, zero group overlap across splits, correct deterministic split assignment, nonempty text/license. There are 328,348 unique groups.
- Scanned all three token binaries: document BOS/EOS counts match JSONL, separators partition the entire stream, every token ID is below vocabulary size, binary token counts equal manifest.
- Tokenizer copy SHA256 equals original. JSONL and binary provenance hashes are recorded below.
- Rechecked 512 deterministic retained texts with the full public-docs quality assessor: zero failures; rerunning the same PII detector on those texts found zero additional hits. This sample is not random and is **not a full-corpus independent PII audit**. Names/addresses, detector false negatives, and legal suitability are not certified.
- Input row `sha256` fields are retained upstream metadata: **1,194** differ from the cleaned/redacted output text. Do not use them as output-text integrity hashes; use output-file SHA256. This is an explicit metadata limitation, not token corruption.

### Quality drop reasons

| Source | First-failure reason | Documents |
|---|---|---:|
| law-ko-source | duplicated_5_n_grams | 478 |
| law-ko-source | gopher_below_alpha_threshold | 222 |
| law-ko-source | line_punct_ratio | 1,427 |
| law-ko-source | list_ratio | 74 |
| law-ko-source | top_2_gram | 3 |
| law-ko-source | char_dup_ratio | 917 |
| law-ko-source | dup_line_frac | 338 |
| law-ko-source | duplicated_10_n_grams | 90 |
| law-ko-source | top_4_gram | 74 |
| law-ko-source | duplicated_9_n_grams | 65 |
| law-ko-source | hangul_ratio | 158 |
| law-ko-source | latin_ratio | 89 |
| law-ko-source | duplicated_8_n_grams | 62 |
| law-ko-source | duplicated_6_n_grams | 49 |
| law-ko-source | duplicated_7_n_grams | 43 |
| law-ko-source | top_3_gram | 12 |
| law-ko-source | gopher_too_many_ellipsis | 1 |
| law-ko-source | gopher_enough_stop_words | 6 |
| korea-kr-briefing-source | line_punct_ratio | 6,480 |
| korea-kr-briefing-source | dup_line_frac | 2,496 |
| korea-kr-briefing-source | list_ratio | 1,295 |
| korea-kr-briefing-source | hangul_ratio | 366 |
| korea-kr-briefing-source | gopher_below_alpha_threshold | 1,896 |
| korea-kr-briefing-source | char_dup_ratio | 162 |
| korea-kr-briefing-source | latin_ratio | 241 |
| korea-kr-briefing-source | duplicated_6_n_grams | 4 |
| korea-kr-briefing-source | top_4_gram | 17 |
| korea-kr-briefing-source | duplicated_7_n_grams | 7 |
| korea-kr-briefing-source | duplicated_9_n_grams | 12 |
| korea-kr-briefing-source | duplicated_8_n_grams | 10 |
| korea-kr-briefing-source | gopher_too_many_ellipsis | 1 |
| korea-kr-briefing-source | duplicated_10_n_grams | 15 |
| korea-kr-briefing-source | duplicated_5_n_grams | 25 |
| korea-kr-briefing-source | top_3_gram | 8 |
| korea-kr-briefing-source | top_2_gram | 2 |
| korea-kr-briefing-source | gopher_too_many_bullets | 1 |
| korea-kr-briefing-source | gopher_enough_stop_words | 2 |
| korea-kr-briefing-source | hanja_ratio | 2 |
| wiki-ko-full-source | line_punct_ratio | 70,841 |
| wiki-ko-full-source | gopher_below_alpha_threshold | 15,319 |
| wiki-ko-full-source | hangul_ratio | 17,706 |
| wiki-ko-full-source | latin_ratio | 11,097 |
| wiki-ko-full-source | char_dup_ratio | 494 |
| wiki-ko-full-source | list_ratio | 6,563 |
| wiki-ko-full-source | top_2_gram | 168 |
| wiki-ko-full-source | dup_line_frac | 925 |
| wiki-ko-full-source | hanja_ratio | 380 |
| wiki-ko-full-source | top_4_gram | 1,339 |
| wiki-ko-full-source | top_3_gram | 869 |
| wiki-ko-full-source | duplicated_5_n_grams | 63 |
| wiki-ko-full-source | duplicated_7_n_grams | 9 |
| wiki-ko-full-source | lorem_ipsum | 1 |
| wiki-ko-full-source | duplicated_6_n_grams | 11 |
| wiki-ko-full-source | duplicated_9_n_grams | 21 |
| wiki-ko-full-source | gopher_too_many_end_ellipsis | 1 |
| wiki-ko-full-source | duplicated_10_n_grams | 18 |
| wiki-ko-full-source | gopher_too_many_ellipsis | 1 |
| wiki-ko-full-source | duplicated_8_n_grams | 17 |
| wiki-ko-full-source | gopher_short_doc | 1 |
| wiki-ko-full-source | gopher_too_many_hashes | 1 |

## Runtime and reproducibility

Snapshot/input scan: 9.41s; build: 4929.54s; tokenization: 347.29s; verification: 41.86s; total: **5328.11s (88.80 minutes)**. These are wall times including coexistence with the running workload.

The measurement process ran under `nice -n 10`; OMP/OPENBLAS/MKL/VECLIB/NUMEXPR/RAYON thread limits were 1 and TOKENIZERS_PARALLELISM=false. Existing Kiwi uses four worker threads. No MPS calls or model-training commands were issued; existing crawler/training processes were not stopped or reconfigured. This is configured concurrency, not a sampled OS thread-count or interference benchmark.

Equivalent builder invocation (use new output paths to rerun; preserved snapshot required):

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  nice -n 10 .venv/bin/python scripts/build_pretraining_mix.py \
  --source /tmp/trid-mix-v3-snapshot-10777/law-ko-source:1000000000000000 \
  --source /tmp/trid-mix-v3-snapshot-10777/korea-kr-briefing-source:1000000000000000 \
  --source data/wiki-ko-full-source:200M \
  --output data/mix-v3-dry --quality-profile public-docs --holdout-percent 1
```

Then under the same environment/nice limits, call `trid.corpus.prepare("data/mix-v3-dry", "data/mix-v3-dry-tokens", tokenizer_path="runs/tokenizers/trid-ko-mix-v1-ds-8192/tokenizer.json")`. Snapshots are temporary-storage artifacts: preserve them before cleanup for byte-identical reproduction. Live crawl state cannot reproduce this exact input merely by rerunning later.

### Split fix and smoke evidence

Builder previously interpreted the documented per-split percentage as a combined percentage. Thresholds were corrected from 50/100 to 100/200 buckets per requested percent; existing generated mixes intentionally remain unchanged. Only this new mix uses the correction. Regression added to tests/test_build_pretraining_mix.py; this worker did not run test suites.

Observed old-expression reproduction and corrected smoke:
```text
group=holdout:449 bucket=52 requested_percent_each=1.0 expected=test before_expression=valid after=test
PASS corrected threshold smoke
20,000 fixed groups: train=19610 test=201 valid=189
PASS each held-out split is within 0.8–1.2 percent
```

## Provenance SHA256

| Input | SHA256 |
|---|---|
| law-ko-source docs.jsonl snapshot/source | `94240a337823b76b1a78744fa5d03a43be27f77a286e2c36817228fac4f3a105` |
| korea-kr-briefing-source docs.jsonl snapshot/source | `b8f194973f031ddfa7519ee9a927120cb182f0356ddf95af4acced623a7a2800` |
| wiki-ko-full-source docs.jsonl snapshot/source | `02e9ebace76187b8b5b75c6a2b9bad8f8827416698d59b725fb9605bd07b98c5` |

| Artifact | SHA256 |
|---|---|
| scripts/build_pretraining_mix.py | `adeca8463357d6045eb1e001cefc1f7eca52fa1678094ba666ac710ef0502dba` |
| trid/korean_quality.py | `92c49c8554db7d11eec56afa41e861defe3aeea59d5d6dbf4cf1ca172bdf0867` |
| trid/corpus.py | `da2618d40a2cccb5917b22ffd755a71804476038e70a4620d4adb2cef7bfadb3` |
| data/mix-v3-dry/provenance.json | `a85f3d494da782783ed08e8f9fb887f005d0a6a31fdd08d60ddb987b52072d62` |
| data/mix-v3-dry-tokens/manifest.json | `d9f9f64e8f137a16d6d7d4c6961876c325b2b0d499c38c7636b1ee2e3a8d7d84` |
| runs/tokenizers/trid-ko-mix-v1-ds-8192/tokenizer.json | `d6a985c103414498d3f3aec167f583ab53969c528318cb13da96a7d04bc073dc` |
| data/mix-v3-dry/train.jsonl | `8da32dbb7debe5dc684b6ab4b76d00310fada3bc249d2419bc18b98f653cee4f` |
| data/mix-v3-dry-tokens/train.bin | `404544fe1d40cf36d2862aad6935c0d4843fc2362e6c16f80cca4a87a6ede2ee` |
| data/mix-v3-dry/valid.jsonl | `3e1acadd5c3df879baafe105b18eae053c8a21ac4efbf66e420c98df22f1d123` |
| data/mix-v3-dry-tokens/valid.bin | `f78f0891fcb78a2d81cd74ece27d4f11415f03eeddee0e0fc11b7f228c9fc50d` |
| data/mix-v3-dry/test.jsonl | `6c094edfd565ffe67b13530ac3afa7f9a563a43339ce2d3bf82ce67b4710f24d` |
| data/mix-v3-dry-tokens/test.bin | `be2edd00cb0616c2c21f4e6742be4749321971d9a2b5e1f50919bfb675f47ee6` |

Machine-readable detail: `data/mix-v3-dry/measurement.json`, `provenance.json`, `quality-removed.json`, `removed.json`; `data/mix-v3-dry-tokens/manifest.json` embeds source provenance. Attribution files are retained in the mix.

## Remaining gaps and release limitations

- The complete intended snapshot was processed and tokenized; there is no execution blocker or unprocessed selected subset. The remaining 538,606,186-token gap is a real corpus-supply gap, not completed work disguised as a billion-token corpus. Train-only supply is 451,888,351 tokens.
- Law/briefing crawls remain in progress; “all” here excludes later rows and material not collected. Wiki is the existing 2023-11-01 Korean snapshot, not current knowledge. Quality drops can remove legitimate lists/legal boilerplate, so retained volume is not a coverage guarantee.
- Upstream licenses: law corpus marks Korean Copyright Act Article 7; briefing marks KOGL Type 1 text with attribution; Wikipedia marks CC-BY-SA/GFDL. Preserve per-document links, attribution and applicable terms; mixed-corpus preparation does not establish legal clearance of model weights or suitability for public-sector production.
- Near dedup has approximate recall; same-group splitting does not establish legal-entity/temporal/topic-independent evaluation or prevent unrelated IDs from containing similar facts. Existing evaluation/model data were not audited for contamination in this task.
- No air-gap deployment, model quality benchmark, model training, activation, README edit, commit or push was performed.

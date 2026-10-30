"""Span-level check of the rows B-022 proposed for the search index because their reserved-overlap text looked statutory.

B-022 (v3-overlap-index-separation.jsonl) marked 102 law.go.kr rows 'index_ok_shared_text_is_statute': in every reserved
family they overlap, at least 80% of the shared word 8-grams also occur in current statute text (target `law`). A share
is not a span check: 20% of a row's shared text may still be institution wording. Here each row's shared text is cut
into spans (maximal runs of word positions covered by a shared 8-gram) and every span is checked on its own:
  statute_span      the whole span, word for word (NFC, whitespace-normalised), occurs inside one current statute
                    document (law-go-kr:law:*) other than the row itself
  own_statute_span  the row is itself a current statute (law-go-kr:law:*): the span is that statute's own text
  non_statute_span  anything else
A row is 'index_ok_all_spans_statute' only when it has no non_statute_span; otherwise 'separate_from_eval_index'.
Nothing is approved for training and no evaluation set is changed.

  .venv/bin/python scripts/verify_overlap_spans.py --rows reports/coordination/B-022/v3-overlap-index-separation.jsonl \
      --out reports/coordination/B-023/overlap-102-span-check.jsonl
"""
import argparse
import gzip
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
SIGN = ROOT / 'data/benchmark-reserve-001/sign-input.jsonl.gz'
LAW = ROOT / 'data/law-ko-source/docs.jsonl'
N = 8


def words(text):
    return unicodedata.normalize('NFC', text).split()


def grams(w):
    return [' '.join(w[i:i + N]) for i in range(len(w) - N + 1)]


def spans(w, shared):
    """Maximal word ranges [a, b) covered by shared 8-grams."""
    out, cur = [], None
    for i, g in enumerate(grams(w)):
        if g in shared:
            if cur and i <= cur[1]:
                cur[1] = i + N
            else:
                if cur:
                    out.append(tuple(cur))
                cur = [i, i + N]
    if cur:
        out.append(tuple(cur))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; outputs are never overwritten')
    rows = [json.loads(x) for x in (ROOT / args.rows).read_text().split('\n') if x]
    rows = [r for r in rows if r['recommendation'] == 'index_ok_shared_text_is_statute']
    fams = {f for r in rows for f in r['families']}
    sign = defaultdict(set)
    with gzip.open(SIGN, 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['family_id'] in fams:
                sign[r['family_id']].update(grams(words(r['text'])))
    want = {r['corpus_id'] for r in rows}
    text_of, row_spans = {}, {}
    with open(LAW) as f:
        for line in f:
            if any(line.startswith(f'{{"id": "{k}') for k in want):
                d = json.loads(line)
                if d['id'] in want:
                    text_of[d['id']] = words(d['text'])
    for r in rows:
        w = text_of[r['corpus_id']]
        shared = set().union(*(sign[f] for f in r['families'])) & set(grams(w))
        row_spans[r['corpus_id']] = [(a, b, ' '.join(w[a:b])) for a, b in spans(w, shared)]
    # candidate statutes per span: those holding the span's first 8-gram; then a whole-span substring check
    first = defaultdict(list)
    for cid, ss in row_spans.items():
        for k, (a, b, s) in enumerate(ss):
            first[' '.join(s.split()[:N])].append((cid, k))
    statute_hit = defaultdict(set)
    with open(LAW) as f:
        for line in f:
            if not line.startswith('{"id": "law-go-kr:law:'):
                continue
            d = json.loads(line)
            body = ' '.join(words(d['text']))
            for g in first.keys() & set(grams(body.split())):
                for cid, k in first[g]:
                    if cid != d['id'] and row_spans[cid][k][2] in body:
                        statute_hit[(cid, k)].add(d['id'])
    counts, written = Counter(), []
    for r in rows:
        cid, ss = r['corpus_id'], row_spans[r['corpus_id']]
        own = cid.startswith('law-go-kr:law:')
        kinds = []
        for k, (a, b, s) in enumerate(ss):
            kinds.append('statute_span' if statute_hit[(cid, k)] else 'own_statute_span' if own else 'non_statute_span')
        bad = [ss[k][2] for k, c in enumerate(kinds) if c == 'non_statute_span']
        decision = 'index_ok_all_spans_statute' if ss and not bad else 'separate_from_eval_index'
        counts[decision] += 1
        counts.update(f'span:{c}' for c in kinds)
        written.append({'corpus_id': cid, 'kind': r['kind'], 'families': r['families'], 'spans': len(ss),
                        'span_kinds': dict(Counter(kinds)), 'span_words': sum(b - a for a, b, _ in ss),
                        'non_statute_words': sum(b - a for (a, b, _), c in zip(ss, kinds) if c == 'non_statute_span'),
                        'statute_sources': sorted({x for k in range(len(ss)) for x in statute_hit[(cid, k)]})[:5],
                        'non_statute_examples': [s[:160] for s in bad[:3]], 'decision': decision})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in written))
    summary = {'code_sha256': CODE_SHA256, 'rows': len(written), 'counts': dict(sorted(counts.items())),
               'by_kind': {k: dict(Counter(x['decision'] for x in written if x['kind'] == k)) for k in sorted({x['kind'] for x in written})}}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

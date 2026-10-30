"""Split each reserved-overlap pair (corpus row x reserved family) into what is shared statute text and what is not.

For every row that check_reserved_overlap.py reported (full match or partial review), the word 8-grams it shares with
the reserved family's signed text (sign-input.jsonl.gz) are looked up in the current statutes of the local law.go.kr
collection (target `law` only: 법률·시행령·시행규칙). Shared 8-grams found there are '상위 법령 문구'; the rest are
'법령 밖 공유' (institution-specific wording or text copied from administrative rules, manuals, forms). Evidence only:
this does not clear any row for training and does not change any evaluation set.

  .venv/bin/python scripts/classify_reserved_overlap.py --report <check json> --corpus <jsonl> --out <json>
"""
import argparse
import gzip
import json
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGN = ROOT / 'data/benchmark-reserve-001/sign-input.jsonl.gz'
LAW = ROOT / 'data/law-ko-source/docs.jsonl'
TARGET = set()


def grams(text):
    w = unicodedata.normalize('NFC', text).split()
    return {' '.join(w[i:i + 8]) for i in range(len(w) - 7)}


def init(target):
    global TARGET
    TARGET = target


def statute_hits(lines):
    hit = set()
    for line in lines:
        if line.startswith('{"id": "law-go-kr:law:'):
            hit |= grams(json.loads(line)['text']) & TARGET
    return hit


def chunks(f, n=300):
    buf = []
    for line in f:
        buf.append(line)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--report', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rep = json.loads(Path(args.report).read_text())
    want = defaultdict(set)
    for m in rep.get('matches') or []:
        want[m['corpus_id']].update(m['families'])
    for pr in rep.get('partial_review') or []:
        want[pr['corpus_id']].update(pr['families'])
    sign = defaultdict(set)
    with gzip.open(SIGN, 'rt') as f:
        for line in f:
            r = json.loads(line)
            sign[r['family_id']] |= grams(r['text'])
    pairs = []
    with open(args.corpus) as f:
        for line in f:
            r = json.loads(line)
            if r['id'] in want:
                g = grams(r['text'])
                for fam in want[r['id']]:
                    pairs.append((r['id'], fam, g & sign[fam]))
    target = set().union(*(s for _, _, s in pairs)) if pairs else set()
    found = set()
    with open(LAW) as f, Pool(8, initializer=init, initargs=(target,)) as pool:
        for h in pool.imap_unordered(statute_hits, chunks(f)):
            found |= h
    rows = []
    for cid, fam, shared in pairs:
        law = shared & found
        rows.append({'corpus_id': cid, 'family': fam, 'shared_ngram8': len(shared), 'in_statutes': len(law),
                     'statute_share': round(len(law) / max(1, len(shared)), 3),
                     'example_outside_statutes': sorted(shared - law)[:2], 'example_in_statutes': sorted(law)[:1]})
    band = Counter('>=0.8' if r['statute_share'] >= 0.8 else '0.2-0.8' if r['statute_share'] >= 0.2 else '<0.2' for r in rows)
    by_kind = defaultdict(Counter)
    for r in rows:
        by_kind[r['corpus_id'].split(':')[1]]['>=0.8' if r['statute_share'] >= 0.8 else '0.2-0.8' if r['statute_share'] >= 0.2 else '<0.2'] += 1
    summary = {'pairs': len(rows), 'rows': len(want), 'statute_share_band': dict(band),
               'by_corpus_kind': {k: dict(v) for k, v in by_kind.items()},
               'shared_ngrams_total': sum(r['shared_ngram8'] for r in rows),
               'shared_ngrams_in_statutes': sum(r['in_statutes'] for r in rows)}
    Path(args.out).write_text(json.dumps({'summary': summary, 'pairs': rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

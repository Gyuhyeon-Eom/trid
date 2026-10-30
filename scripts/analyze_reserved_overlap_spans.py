"""Where a held document shares word 8-grams with the signed reserved families, by character offset (report only).

Uses only the signature manifest (family_id + ngram8 hashes, no reserved text) and the document text. For each
document: the merged character spans covered by shared 8-grams, their share of the document, which families they hit,
the nearest heading above each span, and whether the span quotes a higher norm (statute/decree/rule article) or not.
Nothing is excluded or approved here; the output is input for a decision by the supervisor and A.

  .venv/bin/python scripts/analyze_reserved_overlap_spans.py --docs data/public-data/guides-v7-src/held.jsonl \
      --ids data.go.kr:15099914:file:4b8c305ba5555885 data.go.kr:15038805:file:a53e94e4ff2f340d --out <json>
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
NGRAM = 8
HEADING = re.compile(r'^\s*(?:#+\s|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|[0-9]{1,2}\.\s|[가-하]\.\s|□|■|제\s*\d+\s*[장절편]|【|<\s*[^>]{2,30}>\s*$)')
HIGHER_NORM = re.compile(r'「[^」]{2,40}(?:법|법률|령|규칙|규정)」|(?:법|시행령|시행규칙)\s*제\d+조')


def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]


def hash8(words):
    return int.from_bytes(hashlib.blake2b(' '.join(words).encode(), digest_size=8).digest(), 'little')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--docs', required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--manifest', default='data/benchmark-reserve-001/signatures-partial-148.jsonl')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    owner = defaultdict(set)
    for x in (ROOT / args.manifest).read_text().split('\n'):
        if x:
            r = json.loads(x)
            for h in r['ngram8']:
                owner[int(h)].add(r['family_id'])
    rows = {json.loads(x)['id']: json.loads(x) for x in (ROOT / args.docs).read_text().split('\n') if x}
    report = {'code_sha256': CODE_SHA256, 'manifest': args.manifest, 'docs': []}
    for rid in args.ids:
        r = rows[rid]
        text = unicodedata.normalize('NFC', r['text'])
        if text != r['text']:
            raise SystemExit(f'{rid}: text is not NFC; offsets would not index the stored text')
        ws = word_spans(text)
        words = [text[s:e] for s, e in ws]
        hit = []
        fam = Counter()
        for i in range(len(words) - NGRAM + 1):
            fs = owner.get(hash8(words[i:i + NGRAM]))
            if fs:
                hit.append((ws[i][0], ws[i + NGRAM - 1][1]))
                fam.update(fs)
        spans = []
        for s, e in sorted(hit):
            if spans and s <= spans[-1][1] + 1:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        lines = text.split('\n')
        line_starts, pos = [], 0
        for line in lines:
            line_starts.append(pos)
            pos += len(line) + 1
        heading_at = []
        for i, line in enumerate(lines):
            if HEADING.match(line) and len(line.strip()) < 80:
                heading_at.append((line_starts[i], line.strip()))
        out_spans = []
        for s, e in spans:
            head = next((h for at, h in reversed(heading_at) if at <= s), None)
            seg = text[s:e]
            out_spans.append({'start': s, 'end': e, 'chars': e - s, 'heading': head,
                              'quotes_higher_norm': bool(HIGHER_NORM.search(seg)), 'sha256': hashlib.sha256(seg.encode()).hexdigest()})
        covered = sum(x['chars'] for x in out_spans)
        report['docs'].append({
            'id': rid, 'title': r['title'], 'institution': r['provenance']['institution'], 'text_sha256': r['sha256'],
            'chars': len(text), 'shared_ngram8_positions': len(hit), 'families': dict(fam.most_common()),
            'spans': len(out_spans), 'covered_chars': covered, 'covered_share': round(covered / len(text), 4),
            'spans_quoting_higher_norm': sum(x['quotes_higher_norm'] for x in out_spans),
            'chars_quoting_higher_norm': sum(x['chars'] for x in out_spans if x['quotes_higher_norm']),
            'top_headings': Counter(x['heading'] for x in out_spans).most_common(12), 'span_list': out_spans})
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for d in report['docs']:
        print(json.dumps({k: d[k] for k in ('id', 'chars', 'spans', 'covered_chars', 'covered_share', 'spans_quoting_higher_norm',
                                            'chars_quoting_higher_norm', 'families', 'top_headings')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Conservative normalization of 법령해석례 text (local, separate version; the source rows stay untouched).

Three fixes, each recorded as an edit with its offset in the source text so every change can be traced and undone:
  footnote   law.go.kr DRF inlines a footnote right after its marker: "기부자 등1)1) <footnote text>\n에게 무상으로".
             The duplicated marker, the footnote text and the newline that ends it are removed from the sentence (which
             then reads "기부자 등[각주1]에게 무상으로", as in the source document), and the footnote text is appended at
             the end as "[각주1] <footnote text>". Nothing is dropped.
  entity     HTML character references (&#8228; &amp; ...) -> the character.
  linebreak  A newline between two Hangul characters is decided from the corpus vocabulary only:
               join (no space) when the joined word occurs >= 2 times elsewhere and at least as often as the rarer half;
               space when both halves occur >= 2 times as words and the joined word never does;
               otherwise the newline is kept (unknown: no guessing, no blanket merge).
Input: data/internal-rules/<version>/{docs,quality_held,quarantine}.jsonl, legal interpretations only.
Output: data/internal-rules/<version>-norm<N>/<part>.jsonl with `text` normalized, `meta.normalization` =
{source_sha256, edits:[{rule, src_start, src_end, replacement}]}, plus manifest.json and a review sample file.

  .venv/bin/python scripts/normalize_interpretations.py --source v3 --out v3-norm1
"""
import argparse
import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAP_MIN = 40  # line length (chars) from which a newline is treated as a wrap; v3 lines before Hangul|Hangul breaks are
#              bimodal: most under 20 (structure) or over 80 (wrapped paragraphs)
LIST_START = re.compile(r'([가-하]\.|\d+\.|\d+\)|[①-⑳]|제\d+조|[<【「(])')
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
BASE = ROOT / 'data/internal-rules'
FOOTNOTE = re.compile(r'(\d{1,2})\)\1\)[ \t]?([^\n]*)\n')
ENTITY = re.compile(r'&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});')
BREAK = re.compile(r'(\S*[가-힣])\n([가-힣]\S*)')
WORD_EDGE = re.compile(r'^[^\w가-힣]+|[^\w가-힣]+$')


def vocabulary(texts):
    freq = Counter()
    for t in texts:
        for w in t.split():
            w = WORD_EDGE.sub('', w)
            if w:
                freq[w] += 1
    return freq


def edits_for(text, freq):
    """List of (start, end, replacement, rule, detail) on the source text, non-overlapping, in order."""
    edits, notes = [], []
    for m in FOOTNOTE.finditer(text):
        n, body = m.group(1), m.group(2).strip()
        edits.append((m.start(), m.end(), f'[각주{n}]', 'footnote', body))
        notes.append(f'[각주{n}] {body}')
    taken = [(s, e) for s, e, *_ in edits]
    inside = lambda pos: any(s <= pos < e for s, e in taken)
    for m in ENTITY.finditer(text):
        if not inside(m.start()):
            rep = html.unescape(m.group(0))
            if rep != m.group(0):
                edits.append((m.start(), m.end(), rep, 'entity', m.group(0)))
    for m in BREAK.finditer(text):
        nl = m.start() + len(m.group(1))
        if inside(nl):
            continue
        a, b = WORD_EDGE.sub('', m.group(1)), WORD_EDGE.sub('', m.group(2))
        joined = freq.get(a + b, 0)
        fa, fb = freq.get(a, 0), freq.get(b, 0)
        line_len = nl - (text.rfind('\n', 0, nl) + 1)
        # Only breaks inside wrapped paragraph lines are decided. Short lines are headings, list items and table cells
        # (v3-norm1 sample: '임용령\n제35조의4', '300,000원\n나.', '입양\n본인' were structure, not wraps), and a
        # following list/article marker is structure too.
        structural = line_len < WRAP_MIN or LIST_START.match(m.group(2))
        if not structural and joined >= 2 and joined >= min(fa, fb):
            edits.append((nl, nl + 1, '', 'linebreak_join', f'{a}|{b} joined={joined} a={fa} b={fb} line={line_len}'))
        elif not structural and joined == 0 and fa >= 2 and fb >= 2:
            edits.append((nl, nl + 1, ' ', 'linebreak_space', f'{a}|{b} joined=0 a={fa} b={fb} line={line_len}'))
        else:
            edits.append((nl, nl, '', 'linebreak_unknown_kept', f'{a}|{b} joined={joined} a={fa} b={fb} line={line_len}'))
    edits.sort(key=lambda e: (e[0], e[1]))
    return edits, notes


def apply(text, edits, notes):
    out, pos = [], 0
    for s, e, rep, rule, _ in edits:
        if rule == 'linebreak_unknown_kept':
            continue
        out.append(text[pos:s])
        out.append(rep)
        pos = e
    out.append(text[pos:])
    body = ''.join(out)
    return body + ('\n\n' + '\n'.join(notes) if notes else '')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--samples', type=int, default=15)
    args = p.parse_args()
    src, out = BASE / args.source, BASE / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    parts = {}
    for name in ('docs', 'quality_held', 'quarantine'):
        rows = [json.loads(x) for x in (src / f'{name}.jsonl').read_text().split('\n') if x]
        parts[name] = [r for r in rows if r['kind'] == 'legal_interpretation']
    freq = vocabulary(r['text'] for rows in parts.values() for r in rows)
    out.mkdir(parents=True)
    counts, examples, man_parts = Counter(), {}, {}
    for name, rows in parts.items():
        lines = []
        for r in rows:
            edits, notes = edits_for(r['text'], freq)
            new = apply(r['text'], edits, notes)
            for s, e, rep, rule, detail in edits:
                counts[rule] += 1
                examples.setdefault(rule, []).append({'id': r['id'], 'before': r['text'][max(0, s - 25):e + 25],
                                                      'after_piece': rep, 'detail': detail})
            counts['rows_changed'] += new != r['text']
            row = dict(r)
            row['text'] = new
            row['sha256'] = hashlib.sha256(new.encode()).hexdigest()
            row['clean_version'] = f"{r.get('clean_version')}+{args.out}"
            row['meta'] = dict(r['meta'], normalization={
                'source_version': args.source, 'source_sha256': r['sha256'], 'code_sha256': CODE_SHA256,
                'edits': [{'rule': rule, 'src_start': s, 'src_end': e, 'replacement': rep}
                          for s, e, rep, rule, _ in edits if rule != 'linebreak_unknown_kept'],
                'unknown_linebreaks_kept': sum(1 for *_, rule, _ in edits if rule == 'linebreak_unknown_kept'),
                'footnotes_moved': len(notes)})
            lines.append(json.dumps(row, ensure_ascii=False))
        body = ('\n'.join(lines) + '\n').encode() if lines else b''
        (out / f'{name}.jsonl').write_bytes(body)
        man_parts[name] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest()}
    rng = random.Random(20261030)
    samples = {k: rng.sample(v, min(args.samples, len(v))) for k, v in examples.items()}
    (out / 'review-samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=1))
    manifest = {'source': args.source, 'out': args.out, 'code_sha256': CODE_SHA256, 'parts': man_parts,
                'edit_counts': dict(counts), 'scope': 'legal interpretations (expc) only',
                'status': 'unverified: edits are applied only after C/B review of review-samples.json per rule'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

"""Footnote relations of 법령해석례 text, without changing any text (local; contract for C, B-016).

law.go.kr inlines each footnote right after its marker ("기부자 등1)1) <footnote text>\n에게 ..."). Instead of moving
text (v3-norm2 moved footnotes to the end of the document, which took premises and definitions out of the 질의요지
section - C-014), this records, per source row:
  sections   name (질의요지, 회답, 이유, 관계법령, 기타), src_start, src_end, sha256 of the source slice
  footnotes  n, marker (src offsets of the 'N)' that stays in the sentence), inline (src offsets of the duplicated
             'N)' + footnote text + the newline that ends it), text, section the marker sits in, kind hint
             (premise: '…전제함' / definition: '…말하며, 이하 같음' / reference: '…참조' / other)
Offsets refer to the unchanged source text (row sha256 recorded); C decides how a case shows a footnote (e.g. inside
the section that references it).

v3-footnotes2 (B-017, C-016 requests), all from the source, nothing inferred:
  ordinal    1-based document order of the footnote. The source restarts numbering inside a section in 7 rows, so 'n'
             alone is ambiguous there; rows carry numbering_restarts=true.
  irregular  'space_before_inline' for the one source row (expc:339505) written '…제10조의7)1) 1) 법제처 …' — marker and
             duplicate separated by a space; the next footnote is 2, so this is footnote 1 (law.go.kr's own HTML view
             renders it as '제10조의(각주: 1) …', dropping the '7)'). Every other footnote has no space there.
  related_law_heading  where the raw XML 이유 places '<관계 법령>': before_text (heading, then the quoted articles; the
             '관계법령' section above), trailing_only (the quoted articles come first and the heading closes the field, so
             the source does not mark where they start: appendix_start stays null), none.

  .venv/bin/python scripts/extract_footnote_relations.py --source v3 --out v3-footnotes2
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
BASE = ROOT / 'data/internal-rules'
RAW_EXPC = ROOT / 'data/law-ko-source/raw/expc'
FOOTNOTE = re.compile(r'(\d{1,2})\)(([ \t]?)\1\)[ \t]?([^\n]*)\n)')
HEADING = re.compile(r'<\s*관계\s*법령\s*>')
SECTION = re.compile(r'【(질의요지|회답|이유|관계법령)】|\n<\s*관계\s*법령\s*>|\n※\s*법령정비')


def sections(text):
    marks = [(m.start(), m.group(1) or ('관계법령' if '관계' in m.group(0) else '법령정비')) for m in SECTION.finditer(text)]
    out = []
    starts = [0] + [s for s, _ in marks]
    names = ['머리'] + [n for _, n in marks]
    for i, (s, n) in enumerate(zip(starts, names)):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        if e > s:
            out.append({'name': n, 'src_start': s, 'src_end': e, 'sha256': hashlib.sha256(text[s:e].encode()).hexdigest()})
    return out


def kind(body):
    if re.search(r'전제(함|한다|로 함)', body):
        return 'premise'
    if re.search(r'(을|를) 말하며|이하 같음|이하 같다', body):
        return 'definition'
    if re.search(r'참조', body):
        return 'reference'
    return 'other'


def related_law_heading(row_id):
    raw = (RAW_EXPC / f"{row_id.rsplit(':', 1)[1]}.xml").read_text(encoding='utf-8')
    m = re.search(r'<이유>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</이유>', raw, re.S)
    hs = list(HEADING.finditer(m.group(1) if m else ''))
    if not hs:
        return 'none'
    return 'trailing_only' if len(hs) == 1 and not m.group(1)[hs[0].end():].strip() else 'before_text'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    src, out = BASE / args.source, BASE / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    out.mkdir(parents=True)
    man = {}
    for part in ('docs', 'quality_held', 'quarantine'):
        rows = [json.loads(x) for x in (src / f'{part}.jsonl').read_text().split('\n') if x]
        rows = [r for r in rows if r['kind'] == 'legal_interpretation']
        c = Counter()
        lines = []
        for r in rows:
            text = r['text']
            secs = sections(text)
            sec_of = lambda pos: next((s['name'] for s in secs if s['src_start'] <= pos < s['src_end']), None)
            fns = []
            for m in FOOTNOTE.finditer(text):
                body = m.group(4).strip()
                fn = {'n': int(m.group(1)), 'ordinal': len(fns) + 1, 'marker': [m.start(), m.start() + len(m.group(1)) + 1],
                      'inline': [m.start(2), m.end(2)], 'text': body, 'section': sec_of(m.start()), 'kind': kind(body)}
                if m.group(3):
                    fn['irregular'] = 'space_before_inline'
                    c['irregular:space_before_inline'] += 1
                fns.append(fn)
                c[f"section:{fn['section']}"] += 1
                c[f"kind:{fn['kind']}"] += 1
            last, restarts = {}, False
            for fn in fns:
                restarts |= fn['n'] <= last.get(fn['section'], 0)
                last[fn['section']] = fn['n']
            heading = related_law_heading(r['id'])
            c['rows'] += 1
            c['rows_with_footnotes'] += bool(fns)
            c['rows_numbering_restarts'] += restarts
            c[f'related_law_heading:{heading}'] += 1
            c['footnotes'] += len(fns)
            rel = {'id': r['id'], 'source_version': args.source, 'source_sha256': r['sha256'], 'sections': secs,
                   'footnotes': fns, 'numbering_restarts': restarts, 'related_law_heading': heading}
            if heading == 'trailing_only':
                rel['appendix_start'] = None  # not marked in the source; not guessed
            lines.append(json.dumps(rel, ensure_ascii=False))
        body = ('\n'.join(lines) + '\n').encode() if lines else b''
        (out / f'{part}.relations.jsonl').write_bytes(body)
        man[part] = {'sha256': hashlib.sha256(body).hexdigest(), **dict(c)}
    manifest = {'source': args.source, 'out': args.out, 'code_sha256': CODE_SHA256, 'parts': man,
                'text_changed': False,
                'contract': 'offsets index the unchanged source text of the row with source_sha256; marker = the N) that '
                            'stays in the sentence; inline = duplicated N) + footnote text + its terminating newline; '
                            'removing inline joins the sentence as in the source document'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

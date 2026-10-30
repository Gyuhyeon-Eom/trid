"""Link structured 별표 rows to the 별표 text A's workspace index already holds, by key, and verify each link.

Row files key a 별표 by its law-byl file id (law-go-kr:law-byl:<flSeq>). The index may hold the same 별표 as that
document, or only inside the statute body (law-go-kr:law:<MST>, from the API snapshot), where it starts at a header line
'■ <법령명> [별표 N]' or '■ <법령명> [별표 N의M]'. The key is (법령명, MST, 별표 번호 as written in the 별표's own header,
caption). Per 별표 this writes:
  byl_doc            the index document with the law-byl id, when the index has it
  body_sections      every body section with that 별표 number in the statute body document of the same MST: char span,
                     passage ids overlapping it, whether the caption is found there, and how many of the row numbers
                     occur in it (all row numbers found = values agree with that body version)
  revision_in_body   the '<개정 …>' tag on the body header line, when the body has one
  status             linked (one body section, caption found, all numbers found), ambiguous (several sections pass),
                     body_empty (the body has only the header line: the rows are the only table text),
                     mismatch (sections exist, none passes), absent (no section with that number)
Passage paths in the index are not used as keys: a body passage path can name a 별표 the text only cites.

  .venv/bin/python scripts/link_byl_to_index.py --rows data/law-byl/v2-structure-002-rows-001/queryable-rows.jsonl \
      --index runs/workspace-index/v5 --statutes '공무원 여비 규정' --out reports/coordination/B-029/byl-index-links.jsonl
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
NUM = re.compile(r'\d{1,3}(?:,\d{3})+(?!\d)|\d+')


def norm(s):
    return re.sub(r'\s', '', s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--structured', required=True, help='structure version the rows came from (for the 별표 header line)')
    p.add_argument('--statutes', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; outputs are never overwritten')
    rows = defaultdict(list)
    for x in (ROOT / args.rows).read_text().splitlines():
        r = json.loads(x)
        if r['title'].split(' [별표')[0] in args.statutes:
            rows[r['doc_id']].append(r)
    docs = {d['doc_id']: d for d in json.loads((ROOT / args.index / 'documents.json').read_text())}
    structured = {json.loads(x)['id']: json.loads(x)
                  for x in (ROOT / args.structured / 'structured.jsonl').read_text().splitlines() if x}
    links, c = [], Counter()
    for doc_id, rs in sorted(rows.items()):
        law = rs[0]['title'].split(' [별표')[0]
        lines = (structured.get(doc_id) or {}).get('structured_text', '').split('\n')
        # the 별표's own header line ('■ 법령명 [별표 6의2] <개정 …>'), not the corpus title ('[별표 0006]')
        head = next((x for x in lines + rs[0]['caption'] if re.match(r'\s*■.*\[별표\s*\d', x)), '')
        number = re.search(r'\[별표\s*(\d+(?:의\d+)?)\]', head)
        number = number.group(1) if number else None
        caption = next((x for x in rs[0]['caption'] if '[별표' not in x and '단위' not in x), '')
        mst = next((m.group(1) for r in rs for m in [re.search(r'law-byl/raw/(\d+)/', (r.get('source') or {}).get('raw') or '')]
                    if m), None)
        # each source cell once: values repeated from a merged cell or beside a split row are left out, and so are
        # deleted rows ('6. 삭제 <2021. 5. 25.>'), which the statute body text omits
        values = [n.replace(',', '') for r in rs if not any(re.search(r'삭제\s*<', v) for v in r['cells'].values())
                  for k, v in r['cells'].items() if k not in (r.get('repeated') or []) for n in NUM.findall(v)]
        body = docs.get(f'law-go-kr:law:{mst}') if mst else None
        sections = []
        if body and number:
            text = body['text']
            heads = list(re.finditer(r'^■\s*' + re.escape(law) + r'\s*\[별표\s*(\d+(?:의\d+)?)\][^\n]*$', text, re.M))
            for i, h in enumerate(heads):
                if h.group(1) != number:
                    continue
                start, end = h.start(), heads[i + 1].start() if i + 1 < len(heads) else len(text)
                sec = text[start:end]
                pool = Counter(n.replace(',', '') for n in NUM.findall(sec))
                need = Counter(values)
                found = sum(min(k, pool[n]) for n, k in need.items())
                rev = re.search(r'<\s*(개정|신설|제정)[^>]*>', h.group(0))
                sections.append({'char_span': [start, end], 'header': h.group(0)[:80],
                                 'revision_in_body': rev.group(0) if rev else None,
                                 'caption_found': norm(caption) in norm(sec) if caption else None,
                                 'numbers_found': f'{found}/{sum(need.values())}',
                                 'body_text_chars': len(norm(sec[len(h.group(0)):])),
                                 'passages': [q['id'] for q in body['passages']
                                              if q['char_start'] < end and q['char_end'] > start]})
        ok = [s for s in sections if s['caption_found'] and s['numbers_found'].split('/')[0] == s['numbers_found'].split('/')[1]]
        if len(ok) > 1:
            # the body may carry an older copy of the same 별표 (공무원 여비 규정 별표 5 twice: '사다리차 또는 승강기의
            # 이용료를 포함한다' and '사다리차 이용료 포함'); the copy sharing more of the rows' wording is this one
            grams = {t[i:i + 4] for r in rs for v in r['cells'].values() for t in [norm(v)] for i in range(len(t) - 3)}
            for s in ok:
                sec = norm(body['text'][s['char_span'][0]:s['char_span'][1]])
                s['row_text_overlap'] = round(sum(g in sec for g in grams) / max(len(grams), 1), 3)
            ok.sort(key=lambda s: -s['row_text_overlap'])
            if ok[0]['row_text_overlap'] - ok[1]['row_text_overlap'] >= 0.05:
                ok = ok[:1]
        status = ('absent' if not sections else 'linked' if len(ok) == 1 else 'ambiguous' if len(ok) > 1 else
                  'body_empty' if all(s['body_text_chars'] < 20 for s in sections) else 'mismatch')
        c[status] += 1
        links.append({'row_doc_id': doc_id, 'law': law, 'mst': mst, 'byl': number, 'caption': caption,
                      'rows': len(rs), 'reviews': dict(Counter(r['review'] for r in rs)),
                      'byl_doc_in_index': doc_id if doc_id in docs else None,
                      'body_doc': body['doc_id'] if body else None, 'body_sections': sections,
                      'linked_section': ok[0]['char_span'] if len(ok) == 1 else None, 'status': status})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in links))
    print(json.dumps({'byl_docs': len(links), 'status': dict(c), 'code_sha256': CODE_SHA256}, ensure_ascii=False))


if __name__ == '__main__':
    main()

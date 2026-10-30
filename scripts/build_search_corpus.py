"""Search corpus handed to A for question-first retrieval over all registered public material (TRID-WORKSPACE-001).

Everything here is public material collected by B (law.go.kr, data.go.kr and the providers it links to). It is not any
institution's non-public internal data. A search corpus is not a training set: each document carries its training
status separately, and nothing in it is approved for training by being here.

Collections (only the candidate parts of B's fixed versions; held parts are never read):
  guides         manuals / 편람 / 사례집 from data.go.kr (guides-v6 documents, guides-v7 documents)
  admrul_attach  law.go.kr 행정규칙 attachments (law-attachments v6 candidate parts)
  statute_byl    law.go.kr 법령 별표·서식 (law-byl v2 rows whose status is candidate_*)
  official_faq   official FAQ / 질의회신 rows as a training consumer sees them (guides-v6-train-view-v1: holds excluded,
                 contact names and 신청번호 replaced)
Each document is cut into passages at headings (and at --max-chars), with the heading path and character offsets into
the document text. A passage is left out when
  - it shares a 20-character window with evaluation-side data (scripts/build_pilot_manifest.EVAL_SIDE; this covers the
    148 signed reserved families, not the 852 unsigned ones), so evaluation items and answers cannot be retrieved;
  - a residual identifier pattern is found (phone/mobile fragments, 주민등록번호 shape, e-mail, masked runs '*****').
Documents with known table or footnote defects are already in held parts upstream and are not read.

  .venv/bin/python scripts/build_search_corpus.py --out data/search-corpus-v1
"""
import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_pilot_manifest import EVAL_SIDE, eval_side_text  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
PUB = ROOT / 'data/public-data'
ATT = ROOT / 'data/law-attachments/v6'
BYL = ROOT / 'data/law-byl/v2'
HEADING = re.compile(r'^\s*(?:(#{1,6})\s+(.+)|([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*.+)|(제\s*\d+\s*[편장절관]\s*.{0,60})|(제\s*\d+\s*조(?:의\d+)?\s*\(.{1,40}\))|'
                     r'(\d{1,2}\.\s+[^\d].{0,60})|(□\s*.{1,60}))\s*$')
RESIDUAL_PII = re.compile(r'(?<![\d-])01[016789][-.\s]?\d{3,4}[-.\s]?[\d*＊]{1,4}|\b\d{6}\s*-\s*[1-4]\d{6}\b|'
                          r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\*{5,}')
squash = lambda s: re.sub(r'\s+', '', html.unescape(s or ''))


def heading_level(m):
    if m.group(1):
        return len(m.group(1)), m.group(2).strip()
    if m.group(3):
        return 1, m.group(3).strip()
    if m.group(4):
        return 1 if re.search(r'[편장]', m.group(4)) else 2, m.group(4).strip()
    if m.group(5):
        return 3, m.group(5).strip()
    if m.group(6):
        return 3, m.group(6).strip()
    return 4, m.group(7).strip()


def passages(text, max_chars):
    """(start, end, heading_path) covering the text in order; cut at headings and at max_chars (on a line break)."""
    out, path, start, pos = [], [], 0, 0
    lines = text.split('\n')
    offsets = []
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    current_path = []
    for i, line in enumerate(lines):
        m = HEADING.match(line) if len(line) < 120 else None
        at = offsets[i]
        if m and at > start:
            out.append((start, at, list(current_path)))
            start = at
        if m:
            level, title = heading_level(m)
            path = [p for p in path if p[0] < level] + [(level, title)]
            current_path = [t for _, t in path]
        elif at - start >= max_chars and at > start:
            out.append((start, at, list(current_path)))
            start = at
    if start < len(text):
        out.append((start, len(text), list(current_path)))
    return [(s, e, h) for s, e, h in out if text[s:e].strip()]


def load_parts(folder, parts):
    for part in parts:
        path = folder / f'{part}.jsonl'
        for x in path.read_text().split('\n'):
            if x:
                yield part, json.loads(x)


def documents():
    """(collection, record) for every candidate document, with the metadata A needs."""
    for v in ('guides-v6', 'guides-v7'):
        for part, r in load_parts(PUB / v, ['documents']):
            p = r['provenance']
            yield 'guides', dict(
                id=r['id'], title=r['title'], text=r['text'], text_sha256=r['sha256'], source_version=f'{v}/{part}',
                institution=p['institution'], institution_type=p.get('institution_type'), doc_type=p.get('doc_type'),
                applies_to=p.get('applies_to'), base_date=p.get('date'), date_type=p.get('date_type'), source_url=r['source'],
                license=r['license'], raw=r['raw'], raw_sha256=r['raw_sha256'], extractor=r['meta'].get('extractor'),
                quality=r['meta'].get('quality'), table_check=r['meta'].get('table_check'), pii_redactions=r['meta'].get('pii_redactions'),
                training_status='not approved: manual with tables unverified (guides candidate, not a training view)')
    for part, r in load_parts(ATT, ['candidate_quality_ok', 'candidate_table_verified', 'candidate_table_unverified',
                                    'candidate_short_reviewed']):
        p, m = r['provenance'], r['meta']
        restriction = m.get('usage_restriction')
        yield 'admrul_attach', dict(
            id=r['id'], title=r['title'], text=r['text'], text_sha256=r['sha256'], source_version=f'law-attachments/v6/{part}',
            institution=p.get('institution'), institution_type='중앙행정기관·소속기관(행정규칙)', doc_type=p.get('doc_type'),
            applies_to=f"{p.get('institution')} 행정규칙의 첨부(별표·서식·지침 등)", base_date=r.get('date'),
            date_type=m.get('date_type'), source_url=r['source'], license=r['license'], raw=m.get('raw'), raw_sha256=m.get('raw_sha256'),
            extractor=r.get('extractor'), quality=m.get('quality'), table_check=m.get('table_check'),
            pii_redactions=m.get('pii_redactions'), usage_restriction=restriction,
            training_status=f'not approved: law-attachments v6 {part}' + ('; usage restricted (time-bound value / future repeal)' if restriction else ''))
    status = {json.loads(x)['id']: json.loads(x) for x in (BYL / 'status.jsonl').read_text().split('\n') if x}
    for part, r in load_parts(BYL, ['extracted']):
        st = status.get(r['id'])
        if not st or not st['status'].startswith('candidate'):
            continue
        p, m = r['provenance'], r['meta']
        yield 'statute_byl', dict(
            id=r['id'], title=r['title'], text=r['text'], text_sha256=r['sha256'], source_version=f'law-byl/v2/{st["status"]}',
            institution=p.get('institution'), institution_type='법령 소관부처', doc_type=p.get('doc_type'),
            applies_to=f"「{m['law_title']}」 적용 대상", base_date=p.get('published_date'), date_type=m.get('date_type'),
            effective_date=p.get('effective_date'), source_url=r['source'], license=r['license'], raw=m.get('raw'),
            raw_sha256=m.get('raw_sha256'), extractor=r.get('extractor'), quality=m.get('quality'),
            table_check=f"values cross-checked ({st['values_check']}), layout {st['layout_check']}", pii_redactions=m.get('pii_redactions'),
            training_status=f'not approved: law-byl v2 {st["status"]}')
    for x in (PUB / 'guides-v6-train-view-v1' / 'rows.jsonl').read_text().split('\n'):
        if not x:
            continue
        r = json.loads(x)
        if r['kind'] != 'qa_item':
            continue
        p = r['provenance']
        yield 'official_faq', dict(
            id=r['id'], title=r['title'], text=r['text'], text_sha256=r['sha256'], source_text_sha256=r['source_sha256'],
            source_version=f"guides-v6-train-view-v1 ({r['version']}/{r['part']})", institution=p['institution'],
            institution_type=p.get('institution_type'), doc_type=p.get('doc_type'), applies_to=p.get('applies_to'),
            base_date=r['prep'].get('year') if not p.get('date') else p.get('date'), date_type=p.get('date_type'),
            source_url=r['source'], license=r['license'], faq_kind=r.get('faq_kind'), replaced=r['replaced'],
            table_check=r.get('table_check'), training_status='not approved: training-view row (holds already excluded)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--max-chars', type=int, default=1500)
    p.add_argument('--public-text', help='JSONL of public statute text whose 20-char windows do not count as exposure')
    p.add_argument('--public-kinds', nargs='+', default=['law'], help='id kinds read from --public-text (id = site:kind:n)')
    args = p.parse_args()
    out = ROOT / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    ev = squash(eval_side_text())
    windows = {ev[i:i + 20] for i in range(len(ev) - 19)}
    public = set()
    if args.public_text:
        # evaluation items quote statutes ('1회에 100만원 또는 매 회계연도에 300만원을 초과하는 금품등…'); a window that is
        # also verbatim statute text (law.go.kr 법령 본문) is public law, not an evaluation item or answer, so it does
        # not make a passage exposed. Only kinds named in --public-kinds are read (default: 법령 only, not 해석례).
        for line in open(ROOT / args.public_text, encoding='utf-8'):
            r = json.loads(line)
            if r['id'].split(':')[1] not in args.public_kinds:
                continue
            t = squash(r['text'])
            public.update(w for w in (t[i:i + 20] for i in range(len(t) - 19)) if w in windows)
        windows -= public
    docs, kept, dropped = [], [], []
    counts = Counter()
    for collection, d in documents():
        text = d.pop('text')
        n_in = n_out = 0
        for s, e, heads in passages(text, args.max_chars):
            seg = text[s:e]
            sq = squash(seg)
            hits = sum(sq[i:i + 20] in windows for i in range(0, max(1, len(sq) - 19), 5))
            pii = RESIDUAL_PII.search(seg)
            pid = f"{d['id']}#{s}"
            if hits or pii:
                reason = 'evaluation_exposure' if hits else 'residual_identifier'
                dropped.append(dict(passage_id=pid, doc_id=d['id'], start=s, end=e, reason=reason, eval_window_hits=hits,
                                    sha256=hashlib.sha256(seg.encode()).hexdigest()))
                counts[f'{collection}:dropped:{reason}'] += 1
                n_out += 1
                continue
            kept.append(dict(passage_id=pid, doc_id=d['id'], collection=collection, start=s, end=e, heading_path=heads,
                             text=seg, sha256=hashlib.sha256(seg.encode()).hexdigest(), has_table=('<table' in seg or '\n| ' in seg),
                             title=d['title'], institution=d['institution'], base_date=d.get('base_date'), source_url=d['source_url']))
            n_in += 1
        counts[f'{collection}:docs'] += 1
        counts[f'{collection}:passages'] += n_in
        docs.append(dict(collection=collection, **d, chars=len(text), passages=n_in, passages_dropped=n_out,
                         search_status='public_search_ok' if n_in else 'no_searchable_passage',
                         nature='공개 자료(공공데이터포털·국가법령정보센터 등); 기관 비공개 내부자료 아님'))
    out.mkdir(parents=True)
    files = {}
    for name, rows in (('documents', docs), ('passages', kept), ('dropped', dropped)):
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        files[name] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest()}
    manifest = {
        'version': out.name, 'code_sha256': CODE_SHA256, 'files': files, 'counts': dict(counts),
        'eval_side_checked': EVAL_SIDE, 'eval_window': '20 chars, step 5, whitespace removed, HTML entities decoded',
        'public_text_windows_not_counted': {'source': args.public_text, 'kinds': args.public_kinds, 'windows': len(public),
                                            'eval_windows_left': len(windows)},
        'by_collection': {c: dict(docs=sum(1 for d in docs if d['collection'] == c),
                                  chars=sum(d['chars'] for d in docs if d['collection'] == c),
                                  institutions=len({d['institution'] for d in docs if d['collection'] == c}))
                          for c in sorted({d['collection'] for d in docs})},
        'not_included': ['held parts of every version (reserved overlap incl. 2026 보수 지침·예산편성 기준, OCR defects, '
                         'table mismatches, deleted 별표, rights holds)', 'internal-rules v3 법령해석례/행정규칙 본문 (not in this batch)',
                         'raw 권익위 board rows outside the training view'],
        'contract': 'search corpus only; training_status is separate and no document is approved for training; '
                    'passages that share text with evaluation-side data are not included; 852 unsigned reserved families '
                    'cannot be checked; public material, not an institution\'s internal non-public data'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('files', 'counts', 'by_collection')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

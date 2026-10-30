"""Search corpus with PDF page positions and per-passage value checks, and active / zero-passage documents separated.

Input: a search corpus version (data/search-corpus-v2). Output: a new version whose passages are the same (same ids,
offsets, text and sha256) with added fields:
  page_start / page_end   1-based PDF pages that hold the passage, found by locating several 24-character anchors of
                          the passage (whitespace, table markup and HTML removed) in the pdftotext page texts of the
                          original file; null for HWP/HWPX/FAQ rows (no pages) or when no anchor is found
  page_method             'pdf_anchor' | 'not_found' | 'no_pages'
  values_check            for PDF passages: every number in the passage (digit runs, commas removed) also occurs on
                          the mapped pages (one page of slack each side) -> 'agree', else 'differ' with the numbers
                          not found; 'unmapped' without pages. For HWPX documents a document-level check against the
                          file's own XML text instead: every <hp:t> run (including runs with nested tab/line-break
                          elements) plus equation scripts (<hp:script>, formula objects that kordoc renders as LaTeX).
  value_review            only on 'differ' passages: each missing number classified against the original PDF text
                          (pdftotext -layout and -raw) and a decision; see review_numbers(). A number found in the
                          original is not thereby verified to sit in the right table row or column.
Markdown image references (![image](images/out/image_052.bmp)) are markup: their file-name digits are not values.
Documents are split into documents.jsonl (at least one passage) and documents-zero-passage.jsonl (every passage was
left out upstream), so a consumer does not count empty documents as searchable.

  .venv/bin/python scripts/annotate_search_pages.py --src data/search-corpus-v2 --out data/search-corpus-v4
"""
import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.korean_quality import PII_LABELS  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
MARKUP = re.compile(r'!\[[^\]]*\]\([^)]*\)|<[^>]+>|[|#*`]|\[각주\d+\]')
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
ANCHOR = 24


def squash(text):
    return re.sub(r'\s+', '', html.unescape(MARKUP.sub(' ', text or '')))


def numbers(text):
    return Counter(n.replace(',', '') for n in NUMBER.findall(text))


def pdf_pages(path, mode='-layout'):
    res = subprocess.run(['pdftotext', mode, str(path), '-'], capture_output=True, timeout=600)
    if res.returncode:
        return None
    return res.stdout.decode('utf-8', 'replace').split('\f')


def hwpx_text(path):
    """Text of every <hp:t> run, nested elements (<hp:tab/>, <hp:lineBreak/>) as spaces, plus equation scripts."""
    with zipfile.ZipFile(path) as z:
        parts = sorted(n for n in z.namelist() if re.match(r'Contents/section\d+\.xml$', n))
        xml = ' '.join(z.read(n).decode('utf-8', 'replace') for n in parts)
    runs = re.findall(r'<hp:t\b[^>]*>(.*?)</hp:t>', xml, re.S) + re.findall(r'<hp:script\b[^>]*>(.*?)</hp:script>', xml, re.S)
    return html.unescape(' '.join(re.sub(r'<[^>]+>', ' ', t) for t in runs))


HOLD = ('merged_tokens', 'doubled_glyphs', 'redaction_residue', 'equation_object', 'image_only_page', 'not_in_pdf')


def number_class(n, text, near, near_raw, mapped, whole):
    """Why number n of a passage is not on its mapped pages. text: passage text with markup removed; near / near_raw:
    pdftotext -layout / -raw of the mapped pages ±1; mapped: -layout of the mapped pages; whole: -layout of the PDF.
      redaction_residue  digits left beside a PII redaction token ('[전화]4039'): privacy review, not a value
      print_production_mark  on or after a '.indd …' print-production line: not document content
      equation_object    inside $…$: a formula object the PDF text layer does not carry
      doubled_glyphs     the line repeats every glyph ('2200○○○○년년'): overprinted text read twice
      letter_spaced      the PDF prints the same token with one space between every character ('0 . 1 3 4 6')
      separator_spacing  the PDF has the token with spaces or a line break after '.', '-', ':', '/' ('2005. 10.23')
      merged_tokens      the PDF has these digits as separate tokens ('1      2      3' or '2019-01-01 00'); the
                         extracted number does not exist in the original
      elsewhere_in_pdf(_short_number)  the number occurs on another page (short numbers: 1-2 digits, weak evidence)
      image_only_page    the mapped pages carry almost no text layer (< 200 characters): not checkable
      not_in_pdf         none of the above"""
    for m in re.finditer(rf'(?<!\d){re.escape(n)}(?!\d)', text):
        start = text.rfind('\n', 0, m.start()) + 1
        end = text.find('\n', m.end())
        line = text[start:end if end >= 0 else len(text)]
        if re.search(r'(?:' + '|'.join(map(re.escape, PII_LABELS.values())) + r')[^\S\n]*$', text[start:m.start()]):
            return 'redaction_residue'
        if '.indd' in text[max(0, start - 80):end if end >= 0 else len(text)]:
            return 'print_production_mark'
        if line[:m.start() - start].count('$') % 2 == 1:
            return 'equation_object'
        if re.search(r'([가-힣○])\1([가-힣○])\2', line):
            return 'doubled_glyphs'
    digits = n.replace('.', '')
    # separators may sit between any two digits, so a 1-2 digit number would match digits of neighbouring cells
    pat = r'(?<!\d)' + r'[\s.\-:,/]*'.join(map(re.escape, digits)) + r'(?!\d)' if len(digits) > 2 else r'(?!x)x'
    for page, layout in ((near, True), (near_raw, False)):
        for m in re.finditer(pat, page or ''):
            s = m.group(0)
            # -layout keeps gap widths, so one space between every character is letter spacing; -raw collapses
            # column gaps to one space, so there it cannot tell letter spacing from separate tokens
            if layout and re.fullmatch(r'(?:\S ){2,}\S', s):
                return 'letter_spaced'
            if re.search(r'\d\s+\d', s):
                continue  # digits split by whitespace alone: separate tokens in this rendering; try the other one
            return 'separator_spacing'
    for page in (near, near_raw):
        if page and re.search(pat, page):
            return 'merged_tokens'
    if re.search(rf'(?<!\d){re.escape(n)}(?!\d)', whole.replace(',', '')):
        return 'elsewhere_in_pdf' if len(digits) > 2 else 'elsewhere_in_pdf_short_number'
    if len(re.sub(r'\s+', '', mapped)) < 200:
        return 'image_only_page'
    return 'not_in_pdf'


def review_numbers(ps, pg, raw_pg):
    lo, hi = max(1, ps['page_start'] - 1), min(len(pg), ps['page_end'] + 1)
    near, near_raw = '\n'.join(pg[lo - 1:hi]), '\n'.join(raw_pg[lo - 1:hi]) if raw_pg else ''
    mapped = '\n'.join(pg[ps['page_start'] - 1:ps['page_end']])
    text = MARKUP.sub(' ', ps['text']).replace(',', '')
    classes = {n: number_class(n, text, near, near_raw, mapped, '\n'.join(pg)) for n in ps['numbers_not_on_pages']}
    hold = sorted(n for n, c in classes.items() if c in HOLD)
    return {'numbers': classes, 'decision': 'hold_for_value_use' if hold else 'show_original_page',
            'privacy_review': any(c == 'redaction_residue' for c in classes.values())}


def locate(seg, pages_sq):
    s = squash(seg)
    if len(s) < ANCHOR:
        anchors = [s] if s else []
    else:
        step = max(1, (len(s) - ANCHOR) // 4)
        anchors = [s[i:i + ANCHOR] for i in range(0, len(s) - ANCHOR + 1, step)][:6]
    found = sorted({i for a in anchors for i, p in enumerate(pages_sq) if a and a in p})
    return (found[0] + 1, found[-1] + 1) if found else (None, None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    src, out = ROOT / args.src, ROOT / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    docs = [json.loads(x) for x in (src / 'documents.jsonl').read_text().split('\n') if x]
    passages = [json.loads(x) for x in (src / 'passages.jsonl').read_text().split('\n') if x]
    by_doc = {}
    for ps in passages:
        by_doc.setdefault(ps['doc_id'], []).append(ps)
    raw_of = {d['id']: d.get('raw') for d in docs}
    pdf_ids = [i for i, r in raw_of.items() if r and r.lower().endswith('.pdf') and i in by_doc]
    with ThreadPoolExecutor(args.workers) as pool:
        pages = dict(zip(pdf_ids, pool.map(lambda i: pdf_pages(ROOT / raw_of[i]), pdf_ids)))
    pages_squashed = {i: [squash(x) for x in pg] for i, pg in pages.items() if pg}
    doc_checks, counts, differ_docs = {}, Counter(), set()
    for d in docs:
        raw = d.get('raw') or ''
        if raw.lower().endswith('.hwpx') and d['id'] in by_doc:
            ref = numbers(hwpx_text(ROOT / raw))
            ext = sum((numbers(MARKUP.sub(' ', ps['text'])) for ps in by_doc[d['id']]), Counter())
            missing = ext - ref
            doc_checks[d['id']] = {'method': 'hwpx_xml_numbers', 'result': 'agree' if not missing else 'differ',
                                   'numbers_not_in_xml': dict(missing.most_common(20))}
    for ps in passages:
        pg = pages.get(ps['doc_id'])
        if pg is None:
            ps.update(page_start=None, page_end=None, page_method='no_pages', values_check='unmapped')
            counts['no_pages'] += 1
            continue
        pages_sq = pages_squashed[ps['doc_id']]
        s, e = locate(ps['text'], pages_sq)
        if s is None:
            ps.update(page_start=None, page_end=None, page_method='not_found', values_check='unmapped')
            counts['pdf_not_found'] += 1
            continue
        lo, hi = max(1, s - 1), min(len(pg), e + 1)
        ref = numbers(' '.join(pg[lo - 1:hi]))
        missing = numbers(MARKUP.sub(' ', ps['text'])) - ref
        ps.update(page_start=s, page_end=e, page_method='pdf_anchor', values_check='agree' if not missing else 'differ')
        if missing:
            ps['numbers_not_on_pages'] = dict(missing.most_common(10))
            differ_docs.add(ps['doc_id'])
        counts['pdf_mapped'] += 1
        counts[f"values_{ps['values_check']}"] += 1
    with ThreadPoolExecutor(args.workers) as pool:
        raw_pages = dict(zip(differ_docs, pool.map(lambda i: pdf_pages(ROOT / raw_of[i], '-raw'), differ_docs)))
    review = Counter()
    for ps in passages:
        if ps.get('values_check') == 'differ':
            ps['value_review'] = review_numbers(ps, pages[ps['doc_id']], raw_pages[ps['doc_id']])
            review.update(f'number:{c}' for c in ps['value_review']['numbers'].values())
            review[f"passage:{ps['value_review']['decision']}"] += 1
            review['passage:privacy_review'] += ps['value_review']['privacy_review']
    active, zero = [], []
    for d in docs:
        d = dict(d)
        if d['id'] in pages:
            d['pdf_pages'] = len(pages[d['id']]) if pages[d['id']] else None
        if d['id'] in doc_checks:
            d['doc_values_check'] = doc_checks[d['id']]
        (active if d['passages'] > 0 else zero).append(d)
    out.mkdir(parents=True)
    files = {}
    for name, rows in (('documents', active), ('documents-zero-passage', zero), ('passages', passages)):
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        files[name] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest()}
    src_man = json.loads((src / 'manifest.json').read_text())
    manifest = {
        'version': out.name, 'source_version': src.name, 'code_sha256': CODE_SHA256, 'files': files,
        'active_documents': len(active), 'zero_passage_documents': len(zero),
        'active_by_collection': dict(Counter(d['collection'] for d in active)),
        'zero_by_collection': dict(Counter(d['collection'] for d in zero)),
        'passages': len(passages), 'page_mapping': dict(counts),
        'hwpx_doc_values': {i: c['result'] for i, c in doc_checks.items()}, 'value_review': dict(sorted(review.items())),
        'unchanged_from_source': 'passage ids, offsets, text and sha256 are those of the source version',
        'carried_contract': src_man.get('contract'), 'eval_side_checked': src_man.get('eval_side_checked')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('files', 'active_documents', 'zero_passage_documents', 'active_by_collection',
                                               'zero_by_collection', 'page_mapping', 'hwpx_doc_values', 'value_review')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

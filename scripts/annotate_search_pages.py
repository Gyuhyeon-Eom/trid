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
                          file's own XML text runs instead (numbers in the extracted text vs numbers in <hp:t> runs).
Documents are split into documents.jsonl (at least one passage) and documents-zero-passage.jsonl (every passage was
left out upstream), so a consumer does not count empty documents as searchable.

  .venv/bin/python scripts/annotate_search_pages.py --src data/search-corpus-v2 --out data/search-corpus-v3
"""
import argparse
import hashlib
import html
import json
import re
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
MARKUP = re.compile(r'<[^>]+>|[|#*`]|\[각주\d+\]')
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
ANCHOR = 24


def squash(text):
    return re.sub(r'\s+', '', html.unescape(MARKUP.sub(' ', text or '')))


def numbers(text):
    return Counter(n.replace(',', '') for n in NUMBER.findall(text))


def pdf_pages(path):
    res = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True, timeout=600)
    if res.returncode:
        return None
    return res.stdout.decode('utf-8', 'replace').split('\f')


def hwpx_text(path):
    with zipfile.ZipFile(path) as z:
        parts = sorted(n for n in z.namelist() if re.match(r'Contents/section\d+\.xml$', n))
        return html.unescape(' '.join(t for n in parts for t in re.findall(r'<hp:t>([^<]*)</hp:t>', z.read(n).decode('utf-8', 'replace'))))


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
    doc_checks, counts = {}, Counter()
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
        counts['pdf_mapped'] += 1
        counts[f"values_{ps['values_check']}"] += 1
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
        'hwpx_doc_values': {i: c['result'] for i, c in doc_checks.items()},
        'unchanged_from_source': 'passage ids, offsets, text and sha256 are those of the source version',
        'carried_contract': src_man.get('contract'), 'eval_side_checked': src_man.get('eval_side_checked')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('files', 'active_documents', 'zero_passage_documents', 'active_by_collection',
                                               'zero_by_collection', 'page_mapping', 'hwpx_doc_values')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

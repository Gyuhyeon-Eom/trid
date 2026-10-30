"""Why numbers in search passages were not found on their PDF pages, and why HWPX manuals hold numbers their XML lacks.

Report only; nothing is corrected or promoted. For every passage with values_check='differ' in a search corpus version
and every number listed as missing, the original PDF text (pdftotext -layout, per page) is searched again:
  whitespace_split   the number is on the mapped pages once whitespace is removed: the PDF text layer splits it
                     across columns/lines ('2019-01-01 00' -> '0100'); the value is in the original
  elsewhere_in_pdf   the number occurs on another page of the same PDF (anchor range too narrow, or a repeated table)
  not_in_pdf         the number is nowhere in the PDF text layer, even with whitespace removed: a possible extraction
                     defect (or text the PDF holds only as an image); these passages are listed for hold
For HWPX documents with doc_values_check='differ', each extracted number missing from the XML text runs is classified:
  list_numbering     it opens a line as an item number ('3.', '3)', '(3)', '가.' sequences) - numbering the HWPX stores
                     as paragraph numbering, not text
  not_in_xml         anything else: a possible extraction defect
None of this verifies that a number is placed in the right table row or column.

  .venv/bin/python scripts/classify_value_mismatches.py --corpus data/search-corpus-v3 --out <json>
"""
import argparse
import hashlib
import html
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
MARKUP = re.compile(r'<[^>]+>|[|#*`]|\[각주\d+\]')
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
LIST_NUMBER = re.compile(r'^\s*(?:[-*]\s*)?\(?(\d{1,3})[.)]\s', re.M)
# a table-of-contents line: item number or title, then the page number at the end ('3. 문서작성 및 처리의 일반원칙 32')
TOC_LINE = re.compile(r'^\s*(?:제\s*\d+\s*[편장절]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d{1,2}\.|[가-하]\.)\s*\S.{1,80}?\s\d{1,4}\s*$')


def in_equation(line, pos):
    return line[:pos].count('$') % 2 == 1


def context_class(text, n):
    """Class from where the number sits in the extracted passage: a print-production mark line ('… .indd 113 12.4.30.')
    or an equation ($…$, kordoc renders formula objects as LaTeX; the PDF text layer does not carry them)."""
    for m in re.finditer(rf'(?<!\d){re.escape(n)}(?!\d)', text):
        start = text.rfind('\n', 0, m.start()) + 1
        end = text.find('\n', m.end())
        line = text[start:end if end >= 0 else len(text)]
        if '.indd' in line:
            return 'print_production_mark'
        if in_equation(line, m.start() - start):
            return 'equation_object'
    return None


def numbers(text):
    return Counter(n.replace(',', '') for n in NUMBER.findall(text))


def pdf_pages(path):
    res = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True, timeout=600)
    return res.stdout.decode('utf-8', 'replace').split('\f') if not res.returncode else None


def digits_only_contains(hay, n):
    """n (commas removed) occurs in hay after whitespace and commas are removed, not inside a longer digit run."""
    h = re.sub(r'[\s,]+', '', hay)
    return re.search(rf'(?<!\d){re.escape(n)}(?!\d)', h) is not None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    corpus = ROOT / args.corpus
    docs = {json.loads(x)['id']: json.loads(x) for x in (corpus / 'documents.jsonl').read_text().split('\n') if x}
    passages = [json.loads(x) for x in (corpus / 'passages.jsonl').read_text().split('\n') if x]
    differ = [ps for ps in passages if ps.get('values_check') == 'differ']
    pages = {d: pdf_pages(ROOT / docs[d]['raw']) for d in {ps['doc_id'] for ps in differ}}
    rows, per_class, per_passage = [], Counter(), Counter()
    for ps in differ:
        pg = pages[ps['doc_id']]
        lo, hi = max(1, ps['page_start'] - 1), min(len(pg), ps['page_end'] + 1)
        near = ' '.join(pg[lo - 1:hi])
        whole = ' '.join(pg)
        classes = {}
        text = MARKUP.sub(' ', ps['text']).replace(',', '')
        for n in ps['numbers_not_on_pages']:
            ctx = context_class(text, n)
            if digits_only_contains(near, n):
                classes[n] = 'whitespace_split'
            elif ctx:
                classes[n] = ctx
            elif re.search(rf'(?<!\d){re.escape(n)}(?!\d)', re.sub(',', '', whole)) or digits_only_contains(whole, n):
                classes[n] = 'elsewhere_in_pdf' if len(n.replace('.', '')) > 2 else 'elsewhere_in_pdf_short_number'
            else:
                classes[n] = 'not_in_pdf'
            per_class[classes[n]] += 1
        order = ('not_in_pdf', 'equation_object', 'elsewhere_in_pdf', 'elsewhere_in_pdf_short_number',
                 'print_production_mark', 'whitespace_split')
        worst = next(c for c in order if c in classes.values())
        per_passage[worst] += 1
        rows.append({'passage_id': ps['passage_id'], 'doc_id': ps['doc_id'], 'pages': [ps['page_start'], ps['page_end']],
                     'numbers': classes, 'passage_class': worst, 'sha256': ps['sha256'],
                     'decision': 'hold_for_value_use' if worst in ('not_in_pdf', 'equation_object') else 'show_original_page'})
    hwpx = []
    for d in docs.values():
        chk = d.get('doc_values_check')
        if not chk or chk['result'] != 'differ':
            continue
        with zipfile.ZipFile(ROOT / d['raw']) as z:
            parts = sorted(n for n in z.namelist() if re.match(r'Contents/section\d+\.xml$', n))
            xml = html.unescape(' '.join(t for n in parts for t in re.findall(r'<hp:t>([^<]*)</hp:t>', z.read(n).decode('utf-8', 'replace'))))
        ref = numbers(xml)
        ext_text = '\n'.join(MARKUP.sub(' ', ps['text']) for ps in passages if ps['doc_id'] == d['id'])
        missing = numbers(ext_text) - ref
        cls = Counter()
        examples = defaultdict(list)
        lines = ext_text.split('\n')
        for line in lines:
            for m in NUMBER.finditer(line):
                n = m.group(0).replace(',', '')
                if missing.get(n, 0) <= 0:
                    continue
                missing[n] -= 1
                if TOC_LINE.match(line) and line.rstrip().endswith(m.group(0)):
                    kind = 'toc_page_reference'
                elif LIST_NUMBER.match(line) and m.start() == LIST_NUMBER.match(line).start(1):
                    kind = 'list_numbering'
                elif in_equation(line, m.start()):
                    kind = 'equation_object'
                else:
                    kind = 'not_in_xml'
                    if len(examples[n]) < 2:
                        examples[n].append(line[max(0, m.start() - 50):m.end() + 30])
                cls[kind] += 1
        hwpx.append({'doc_id': d['id'], 'title': d['title'], 'missing_numbers': sum(cls.values()), 'classes': dict(cls),
                     'not_in_xml_examples': {n: v for n, v in list(examples.items())[:15]}})
    out = {'code_sha256': CODE_SHA256, 'corpus': args.corpus, 'pdf_differ_passages': len(differ),
           'number_classes': dict(per_class), 'passage_classes': dict(per_passage), 'passages': rows, 'hwpx_documents': hwpx,
           'note': 'report only; a number found in the original does not verify its table row/column; nothing corrected'}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ('pdf_differ_passages', 'number_classes', 'passage_classes')}, ensure_ascii=False))
    for h in hwpx:
        print(h['doc_id'], h['missing_numbers'], h['classes'])


if __name__ == '__main__':
    main()

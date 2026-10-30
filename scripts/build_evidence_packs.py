"""Work-case evidence packs (local, no network): for a few internal-work topics (계약, 회계, 여비), quote selected
articles verbatim from the stored law.go.kr collection together with what a case needs to be answerable correctly:
source id and URL, the dates recorded in the stored body XML with their field names (공포일자/시행일자 for statutes,
발령일자/시행일자 for administrative rules), the applicability article, and the condition/exception sentences of each
article (sentences with 다만, 단, 불구하고, 경우에는, 제외). No answers, no test items; a reviewer checks each pack.
  .venv/bin/python scripts/build_evidence_packs.py --out <dir>
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAW = ROOT / 'data/law-ko-source/docs.jsonl'
RAW = ROOT / 'data/law-ko-source/raw'
ARTICLE = re.compile(r'^(제\d+조(?:의\d+)?)\s*\(([^)]{1,60})\)')
CONDITION = re.compile(r'다만|(?:^|\s)단,|불구하고|경우에는|제외한다|제외하고|아니할 수 있다|할 수 있다')
# topic -> [(document title, article heading regex)]; the first article of each pack is its applicability clause
PACKS = {
    '계약': [('국가를 당사자로 하는 계약에 관한 법률', r'적용 범위'),
             ('국가를 당사자로 하는 계약에 관한 법률', r'계약의 방법'),
             ('국가를 당사자로 하는 계약에 관한 법률 시행령', r'수의계약에 의할 수 있는 경우')],
    '회계': [('국고금 관리법', r'적용 범위'),
             ('국고금 관리법', r'지출원인행위'),
             ('수입 및 지출 등에 관한 회계예규', r'회계관계공무원의 등록')],
    '여비': [('공무원 여비 규정', r'^목적$'),
             ('공무원 여비 규정', r'여비의 종류'),
             ('공무원 여비 규정', r'여비의 결제와 정산')],
}
DATE_TAGS = {'law': ('공포일자', '시행일자', '공포번호', '제개정구분'), 'admrul': ('발령일자', '시행일자', '발령번호', '제개정구분명', '소관부처명')}


def raw_dates(kind, item_id):
    path = RAW / kind / f'{item_id}.xml'
    if not path.exists():
        return {}
    head = path.read_text(encoding='utf-8', errors='replace')[:8000]
    out = {}
    for tag in DATE_TAGS[kind]:
        m = re.search(fr'<{tag}>(?:<!\[CDATA\[)?([^<\]]*)', head)
        if m and m.group(1).strip():
            out[tag] = m.group(1).strip()
    return out


def articles(text):
    head, body = None, []
    for line in text.split('\n'):
        m = ARTICLE.match(line)
        if m:
            if head:
                yield head, body
            head, body = m, [line]
        elif head:
            body.append(line)
    if head:
        yield head, body


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    want = {t for items in PACKS.values() for t, _ in items}
    docs = {}
    with open(LAW) as f:
        for line in f:
            if re.match(r'\{"id": "law-go-kr:(law|admrul):', line):
                d = json.loads(line)
                if d['title'] in want:
                    docs.setdefault(d['title'], d)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for topic, items in PACKS.items():
        pack = []
        for title, pattern in items:
            d = docs.get(title)
            if not d:
                pack.append({'document': title, 'status': 'document_not_in_collection'})
                continue
            kind, item_id = d['id'].split(':')[1], d['id'].rsplit(':', 1)[1]
            found = [(m, body) for m, body in articles(d['text']) if re.search(pattern, m.group(2))]
            if not found:
                pack.append({'document': title, 'id': d['id'], 'status': 'article_not_found', 'pattern': pattern})
                continue
            m, body = found[0]
            text = '\n'.join(body)
            pack.append({
                'document': title, 'id': d['id'], 'source': d['source'], 'kind': kind,
                'row_date': d.get('date'), 'row_date_field': '공포일자' if kind == 'law' else '발령일자',
                'raw_dates': raw_dates(kind, item_id),
                'article': f'{m.group(1)}({m.group(2)})', 'text': text,
                'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'conditions_and_exceptions': [s.strip() for s in re.split(r'(?<=[.다])\s+|\n', text) if CONDITION.search(s)],
                'other_matches': len(found) - 1, 'status': 'quoted'})
        report[topic] = pack
    (out / 'evidence-packs.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({t: [(x['document'], x.get('article'), x['status'], x.get('raw_dates')) for x in v] for t, v in report.items()},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

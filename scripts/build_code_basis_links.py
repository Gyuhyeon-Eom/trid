"""공통표준용어 -> 공통표준도메인 -> 공개 행정표준코드명 -> 근거 규정 후보 연결표 (local, no network).

Inputs (all already acquired, rights confirmed):
  - 공공데이터 공통표준용어/도메인 CSV (data.go.kr 15156379, 15156442; 이용허락범위 제한 없음; 8차 2025-11)
  - data/law-ko-source/docs.jsonl (law.go.kr 현행 법령·행정규칙; 저작권법 제7조)
Only terms that name a 행정표준코드 and belong to internal-work topics (HEADINGS below) are linked; the rest are listed as
not attempted. A basis candidate is an article (조) of a statute or administrative rule whose heading matches the
term's concept AND whose body contains at least one of the term's printed value labels (허용값, e.g. '파면'). Label
co-occurrence alone was tried first and produced unrelated articles (e.g. 출장코드 -> 119구조법), so it is not used.
The code values are not taken from 행정표준코드관리시스템 (rights unconfirmed); only labels printed in the standard file.
Nothing here is an institution's DB mapping.

  .venv/bin/python scripts/build_code_basis_links.py --terms <csv> --domains <csv> --out <jsonl>
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAW = ROOT / 'data/law-ko-source/docs.jsonl'
ARTICLE = re.compile(r'^제\d+조(의\d+)?\s*\(([^)]{1,40})\)')
# internal-work code terms -> article-heading pattern of the concept they encode
HEADINGS = {
    '징계종류코드': r'징계의 종류',
    '직종코드': r'공무원의 구분',
    '직군코드': r'직군|직렬',
    '직렬코드': r'직군|직렬',
    '직류코드': r'직류|직렬',
    '직급코드': r'직급',
    '계급코드': r'계급',
    '호봉코드': r'호봉',
    '계약방법코드': r'계약의 방법|계약방법',
    '출장코드': r'출장|여비',
    '급여구분코드': r'보수|급여',
    '재직상태코드': r'휴직|직위해제|재직',
    '비위유형코드': r'징계기준|비위',
    '포상종류코드': r'포상|훈장|표창',
    '세입코드': r'세입',
    '수납구분코드': r'수납',
    '부과구분코드': r'부과',
}


def labels(allowed):
    """'10 : 파면, 20 : 해임 ...' -> ['파면', '해임']; '1: 일반경쟁, 2:지명경쟁' -> ['일반경쟁', '지명경쟁']."""
    out = []
    for part in re.split(r'[,，]', allowed.replace('…', '').replace('...', '')):
        m = re.match(r'\s*[0-9A-Za-z~\-]+\s*:\s*(.+?)\s*$', part)
        if m and 1 < len(m.group(1)) <= 20:
            out.append(m.group(1))
    return out


def articles(text):
    """Yield (heading line, body) per 조."""
    head, body = None, []
    for line in text.split('\n'):
        if ARTICLE.match(line):
            if head:
                yield head, '\n'.join(body)
            head, body = line, [line]
        elif head:
            body.append(line)
    if head:
        yield head, '\n'.join(body)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--terms', required=True)
    p.add_argument('--domains', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-per-term', type=int, default=5)
    args = p.parse_args()
    terms = list(csv.DictReader(open(args.terms, encoding='utf-8-sig')))
    domains = {d['공통표준도메인명']: d for d in csv.DictReader(open(args.domains, encoding='utf-8-sig'))}
    coded = [t for t in terms if t['행정표준코드명'].strip()]
    want = {t['공통표준용어명']: labels(t['허용값']) for t in coded if t['공통표준용어명'] in HEADINGS}
    hits = {name: [] for name in want}
    with open(LAW) as f:
        for line in f:
            if not re.match(r'\{"id": "law-go-kr:(law|admrul):', line):
                continue
            d = json.loads(line)
            for head, body in articles(d['text']):
                heading = ARTICLE.match(head).group(2)
                for name, labs in want.items():
                    if re.search(HEADINGS[name], heading):
                        found = [lab for lab in labs if lab in body]
                        if found:
                            hits[name].append({'law_id': d['id'], 'law_title': d['title'], 'date': d.get('date'),
                                               'article': ARTICLE.match(head).group(0), 'labels_found': found,
                                               'excerpt': body[:400]})
    with open(args.out, 'w') as out:
        for t in coded:
            name = t['공통표준용어명']
            dom = domains.get(t['공통표준도메인명'], {})
            # statutes first, then more labels found
            cands = sorted(hits.get(name, []), key=lambda h: (not h['law_id'].startswith('law-go-kr:law:'), -len(h['labels_found'])))
            status = ('not_attempted_outside_internal_work' if name not in want
                      else 'heading_and_label_match' if cands else 'not_found')
            out.write(json.dumps({
                'term': name, 'term_description': t['공통표준용어설명'], 'abbr': t['공통표준용어영문약어명'],
                'domain': t['공통표준도메인명'], 'data_type': dom.get('데이터타입'), 'length': dom.get('데이터길이'),
                'allowed_values_as_printed': t['허용값'], 'value_labels_used': want.get(name, []),
                'admin_code_name': t['행정표준코드명'], 'owner': t['소관기관명'], 'enacted': t['제정차수'],
                'revision': t['개정구분명(폐기 또는 변경)'] or None, 'heading_pattern': HEADINGS.get(name),
                'basis_candidates': cands[:args.max_per_term], 'basis_candidates_total': len(cands),
                'basis_status': status}, ensure_ascii=False) + '\n')
    print(json.dumps({'coded_terms': len(coded), 'internal_work_terms': len(want),
                      'with_candidate': sum(1 for n in want if hits[n]),
                      'law_sha256': hashlib.sha256(LAW.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

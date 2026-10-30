"""Official names and evidence-backed aliases of administrative agencies and public institutions (B-024 / RETRIEVAL-DECISION-001).

Every alias carries its evidence; no alias is invented from a user question or a guess:
  official_name        행정표준코드 기관코드 (code.go.kr, data.go.kr 15092039, licence 제한 없음): 최하위기관명 and 전체기관명
                       of agencies still in existence (존폐여부 0) in 국가행정조직 / 지방자치단체 / 정부투자기관및기타
  alio_name            ALIO 공공기관 registry copy (data/corpus-inventory/alio-agencies.json): apbaNa -> apbaId
  statutory_abbreviation  a 법령·행정규칙 text that defines the short form itself: '<official name>(이하 "<alias>"라 한다)'
                       (data/law-ko-source, law.go.kr DRF responses); kept only when the alias is defined for one agency in
                       at least --min-docs documents and never for another agency
  predecessor_name     a 기관코드 row whose 이전기관코드 points to an abolished agency: the old name now resolves to the
                       successor, marked as a predecessor (rules issued under the old name stay the old agency's rules)
Each output row: alias, canonical (current official name), code (행정기관코드) and/or alio_id, kind, evidence. An alias that
points to more than one canonical agency is written with ambiguous=true and must not be resolved automatically.

  .venv/bin/python scripts/build_agency_aliases.py --codes data/public-data/raw/15092039/33c7a66a73db18cf.zip \
      --out data/agency-aliases-v1
"""
import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
ALIO = ROOT / 'data/corpus-inventory/alio-agencies.json'
LAW = ROOT / 'data/law-ko-source/docs.jsonl'
# institution-level rows only: departments, teams and sub-district units (보조·보좌기관, 하부행정기구, 사업소) are not agencies
KINDS = ('국가행정조직', '자치행정조직', '교육행정조직', '정부투자기관및기타', '헌법조직', '사법조직')
SUB_UNITS = ('보조기관', '보좌기관', '하부행정기구', '제외기구', '사업소')
# a definition like '(이하 "공단"이라 한다)' only names the body inside that one text; such class words are not aliases
GENERIC = {'공단', '공사', '센터', '재단', '위원회', '연구원', '연구소', '과학원', '본부', '장관', '위원장', '기관', '법인', '진흥원',
           '협회', '조합', '학교', '병원', '청', '원', '회', '부', '처', '시', '도', '군', '구', '지사', '지원단', '사업단', '단',
           '관리원', '공단본부', '대학', '센타', '은행', '기금', '사무국', '사무처', '지원센터', '관리공단'}
DEFINE = re.compile(r'([가-힣ㆍ·]+(?:\s[가-힣ㆍ·]+){0,3})\s*\(\s*이하\s*[“"‘\']([가-힣A-Za-z·ㆍ]{2,12})[”"’\']\s*(?:이?라|으로|로)\s*한다\s*\)')


def norm(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s)).strip()


def load_codes(path):
    z = zipfile.ZipFile(path, metadata_encoding='cp949')
    member = next(n for n in z.namelist() if '유형분류 의미추가' in n)
    lines = z.read(member).decode('cp949').splitlines()
    head = lines[0].split('\t')
    return [dict(zip(head, x.split('\t'))) for x in lines[1:]], member


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--codes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-docs', type=int, default=2)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    codes_path = ROOT / args.codes
    rows, member = load_codes(codes_path)
    by_code = {r['기관코드']: r for r in rows}
    active = [r for r in rows if r['존폐여부'] == '0' and r['유형분류_대_의미'] in KINDS and r['유형분류_중_의미'] not in SUB_UNITS]
    out_rows = []
    canon = {}  # official name -> (canonical, code)
    for r in active:
        for name, kind in ((r['최하위기관명'], 'official_name'), (r['전체기관명'], 'official_full_name')):
            name = norm(name)
            if not name or name == 'NULL':
                continue
            canon.setdefault(name, set()).add((norm(r['최하위기관명']), r['기관코드']))
            out_rows.append({'alias': name, 'canonical': norm(r['최하위기관명']), 'code': r['기관코드'], 'kind': kind,
                             'evidence': {'source': 'code.go.kr 기관코드', 'full_name': norm(r['전체기관명']),
                                          'type': f"{r['유형분류_대_의미']}/{r['유형분류_중_의미']}", 'created': r['생성일자']}})
    alio = json.loads(ALIO.read_text())['agencies']
    alio_by_name = {norm(a['apbaNa']): a for a in alio}
    for a in alio:
        name = norm(a['apbaNa'])
        same = [code for nm, code in canon.get(name, ()) if nm == name]
        code = same[0] if len(same) == 1 else None  # the same institution in the code list: one identity, both ids
        if not same:
            canon.setdefault(name, set()).add((name, f"ALIO:{a['apbaId']}"))
        out_rows.append({'alias': name, 'canonical': name, 'code': code, 'alio_id': a['apbaId'], 'kind': 'alio_name',
                         'evidence': {'source': 'ALIO registry copy', 'type': a['typeNa'], 'ministry': a['jidtNa']}})
    # predecessors: an active row whose 이전기관코드 names an abolished agency
    for r in active:
        old = by_code.get(r['이전기관코드'])
        if old and old['존폐여부'] == '1' and norm(old['최하위기관명']) != norm(r['최하위기관명']):
            out_rows.append({'alias': norm(old['최하위기관명']), 'canonical': norm(r['최하위기관명']), 'code': r['기관코드'],
                             'kind': 'predecessor_name',
                             'evidence': {'source': 'code.go.kr 기관코드 이전기관코드', 'old_code': old['기관코드'],
                                          'old_abolished': old['폐지일자'], 'new_created': r['생성일자']}})
    # statutory abbreviations defined in law.go.kr texts
    defs = defaultdict(lambda: defaultdict(set))  # alias -> (canonical, code) -> {doc ids}
    with open(LAW) as f:
        for line in f:
            if '이하' not in line:
                continue
            d = json.loads(line)
            for m in DEFINE.finditer(d['text']):
                words = norm(m.group(1)).split(' ')
                alias = norm(m.group(2))
                for k in range(len(words)):  # the longest official name that ends the phrase
                    name = ' '.join(words[k:])
                    if name in canon and alias != name and len(alias) < len(name) and alias not in GENERIC:
                        for c, code in canon[name]:
                            if c != alias:
                                defs[alias][(c, code)].add(d['id'])
                        break
    stat = Counter()
    for alias, targets in sorted(defs.items()):
        strong = {c: ids for c, ids in targets.items() if len(ids) >= args.min_docs}
        if not strong:
            stat['below_min_docs'] += 1
            continue
        ambiguous = len(targets) > 1
        for (c, code), ids in strong.items():
            out_rows.append({'alias': alias, 'canonical': c, 'code': None if code.startswith('ALIO:') else code,
                             'alio_id': (alio_by_name.get(c) or {}).get('apbaId'),
                             'kind': 'statutory_abbreviation', 'ambiguous': ambiguous,
                             'evidence': {'source': 'law.go.kr text defines (이하 "…"라 한다)', 'documents': len(ids),
                                          'examples': sorted(ids)[:3],
                                          'other_targets': {f'{o[0]} ({o[1]})': len(v) for o, v in targets.items() if o != (c, code)}}})
            stat['ambiguous' if ambiguous else 'kept'] += 1
    targets_per_alias = defaultdict(set)
    for r in out_rows:
        targets_per_alias[r['alias']].add((r['canonical'], r.get('code') or f"ALIO:{r.get('alio_id')}"))
    for r in out_rows:
        r['ambiguous'] = r.get('ambiguous', False) or len(targets_per_alias[r['alias']]) > 1
    out.mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in out_rows).encode()
    (out / 'aliases.jsonl').write_bytes(body)
    manifest = {'version': out.name, 'code_sha256': CODE_SHA256, 'rows': len(out_rows), 'sha256': hashlib.sha256(body).hexdigest(),
                'by_kind': dict(Counter(r['kind'] for r in out_rows)),
                'ambiguous_aliases': sum(1 for a, t in targets_per_alias.items() if len(t) > 1),
                'statutory_abbreviations': dict(stat),
                'inputs': {'codes': {'path': args.codes, 'member': member, 'sha256': hashlib.sha256(codes_path.read_bytes()).hexdigest()},
                           'alio': {'path': str(ALIO.relative_to(ROOT)), 'sha256': hashlib.sha256(ALIO.read_bytes()).hexdigest()},
                           'law': str(LAW.relative_to(ROOT))},
                'rules': 'aliases only from official name lists, rename links and statute/rule texts that define the short form; '
                         'ambiguous aliases are not resolved automatically; no alias from user questions'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('rows', 'by_kind', 'ambiguous_aliases', 'statutory_abbreviations')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

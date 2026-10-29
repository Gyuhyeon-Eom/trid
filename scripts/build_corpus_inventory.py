"""Benchmark-reservable document-family inventory across collected and catalogued sources (B-002).

One row per document family (all revisions/attachments of one document share an id the supervisor can reserve):
  institutions:alio-rule:<seq>       ALIO 내부규정 (one rule, every revision file)      <- corpus/state/institutions/rules-index.json
  korea-kr:<newsId>                  정책브리핑 보도자료 (one release and its attachments) <- data/korea-kr-briefing-source/docs.jsonl
  assembly-minutes:<id>              국회 회의록 (one meeting)                          <- corpus/state/assembly-minutes/<index>
Fields: doc_family_id, source, doc_type, institution, institution_id, institution_type, ministry, title, date,
effective_date, url, business_group, business_group_rule.

business_group is a RULE-BASED PROPOSAL (title keywords, then supervising ministry/committee), not a reviewed label;
business_group_rule records which rule fired so the supervisor can audit or override it. Unmatched rows stay
'미분류' and are never silently folded into another group. Output: corpus/state/inventory/<version>/doc-families.jsonl.gz
and summary.json in S3, summary copy in data/corpus-inventory/. Reserving a family (excluding it from training) is the
supervisor's decision; this file only makes it possible.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from collect_assembly_minutes import BUCKET, list_keys, read_json_s3, s3_client  # noqa: E402

GROUPS = ('보훈·복지', '주거·부동산', '고용·인사', '계약·조달', '회계·예산', '보건·안전', '교육·연구', '공기업 운영·서비스')
# Order matters: the first matching keyword group wins.
TITLE_RULES = (
    ('계약·조달', r'계약|조달|구매|입찰|용역|물품|자산취득|외주'),
    ('회계·예산', r'회계|예산|재무|결산|자금|출장|여비|경비|수수료|기금운용|세입|세출|지출|재정'),
    ('보건·안전', r'안전|보건|재난|위생|감염|의료|소방|방역|산업재해|중대재해|환경오염'),
    ('교육·연구', r'연구|교육|학술|장학|과학기술|R&D|기술개발|훈련원'),
    ('주거·부동산', r'주택|임대|부동산|토지|주거|분양|공시가격|감정평가|건축|도시재생|전세'),
    ('보훈·복지', r'보훈|복지|돌봄|장애|연금|기초생활|아동|노인|유공자|국민연금|건강보험|급여지원|생계'),
    ('고용·인사', r'인사|복무|징계|채용|보수|직제|취업규칙|근로|임금|노동|고용|일자리|포상|퇴직|휴가|수당|평가'),
    ('공기업 운영·서비스', r'정관|이사회|고객|서비스|요금|약관|민원|경영|감사|위임전결|운영|사업|공급|이용'),
)
MINISTRY_RULES = {
    '국가보훈부': '보훈·복지', '보건복지부': '보훈·복지', '성평등가족부': '보훈·복지', '여성가족부': '보훈·복지',
    '국토교통부': '주거·부동산', '고용노동부': '고용·인사', '인사혁신처': '고용·인사', '조달청': '계약·조달',
    '기획재정부': '회계·예산', '재정경제부': '회계·예산', '기획예산처': '회계·예산', '식품의약품안전처': '보건·안전',
    '질병관리청': '보건·안전', '소방청': '보건·안전', '행정안전부': '보건·안전', '교육부': '교육·연구',
    '과학기술정보통신부': '교육·연구'}
COMMITTEE_RULES = (('보건복지', '보훈·복지'), ('정무', '보훈·복지'), ('국토교통', '주거·부동산'), ('환경노동', '고용·인사'),
                   ('기후에너지환경노동', '고용·인사'), ('예산결산', '회계·예산'), ('기획재정', '회계·예산'),
                   ('행정안전', '보건·안전'), ('교육', '교육·연구'), ('과학기술', '교육·연구'), ('산업통상', '공기업 운영·서비스'))


def business_group(title, ministry=None, committee=None):
    for group, pattern in TITLE_RULES:
        if re.search(pattern, title or ''):
            return group, f'title:{pattern.split("|")[0]}…'
    if committee:
        for key, group in COMMITTEE_RULES:
            if key in committee:
                return group, f'committee:{key}'
    if ministry in MINISTRY_RULES:
        return MINISTRY_RULES[ministry], f'ministry:{ministry}'
    return '미분류', 'none'


ALIO_LICENCE = ('기관 공공저작물(원문별 공공누리 유형 미확인); ALIO 저작권정책: 공공데이터 영리 포함 자유이용, '
                '제3자 권리·비공개정보 제외')
ASSEMBLY_CLASSES = {'1': '본회의', '2': '상임위원회', '3': '특별위원회', '4': '예산결산특별위원회', '5': '국정감사', '6': '국정조사'}
# Every local dataset known to have fed an earlier trid training run (train split) or its dev splits.
PRIOR_TRAINING_FILES = ('mix-v3-dry/train.jsonl', 'trid-cpt-kanana-100m/train.jsonl', 'public-document-sft-01/train.jsonl',
                        'public-dialogue-shared-ctx-01/train.jsonl') + tuple(f'public-dialogue-mix-0{i}/train.jsonl' for i in range(1, 7))
PRIOR_DEV_FILES = tuple(f.replace('/train.jsonl', f'/{s}.jsonl') for f in PRIOR_TRAINING_FILES for s in ('valid', 'test'))


def ids_in(files):
    found = Counter()
    for name in files:
        path = ROOT / 'data' / name
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                m = re.search(r'"id":\s*"korea-kr-press-(\d+)', line)
                if m:
                    found[m.group(1)] += 1
    return found


def alio_rows(s3):
    agencies = {a['apbaId']: a for a in read_json_s3(s3, 'corpus/state/institutions/agencies.json', {'agencies': []})['agencies']}
    index = read_json_s3(s3, 'corpus/state/institutions/rules-index.json', {'rules': {}})
    for seq, r in index['rules'].items():
        agency = agencies.get(r['institution_id'], {})
        group, rule = business_group(r['title'], agency.get('jidtNa'))
        yield {'doc_family_id': f'institutions:alio-rule:{seq}', 'source': 'institutions', 'doc_type': '규정',
               'institution': r['institution'], 'institution_id': r['institution_id'],
               'institution_type': agency.get('typeNa'), 'ministry': agency.get('jidtNa'), 'title': r['title'],
               'rule_category': r['category'], 'date': (r.get('disclosed_date') or '').replace('.', '-') or None,
               # ALIO 재·개정일자 (ruleStDa), not 시행일. Catalogue pages fetched before 2026-10-30 04:00 KST stored it
               # under 'effective_date', and inventory-v1 published it under that wrong name (B-003, B-004).
               'revision_date': (r.get('revision_date') or r.get('effective_date') or '').replace('.', '-') or None,
               'url': f'https://www.alio.go.kr/occasional/ruleDtl.do?seq={seq}', 'licence': ALIO_LICENCE,
               'text_status': 'catalogued', 'text_path': None,
               'training_inclusion': 'unknown',
               'training_inclusion_evidence': 'not collected by B; overlap with FineWeb-2 web text not checked',
               'business_group': group, 'business_group_rule': rule}, index.get('total_count')


def press_rows(prior_train, prior_dev):
    path = ROOT / 'data/korea-kr-briefing-source/docs.jsonl'
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            m = re.search(r'이 자료는\s*(.+?)의\s*보도자료', d.get('attribution') or '')
            ministry = m.group(1).strip() if m else None
            group, rule = business_group(d.get('title'), ministry)
            news_id = d['id'].rsplit('-', 1)[-1]
            if news_id in prior_train:
                inclusion, evidence = 'included_prior_trid_training', 'id in a prior trid training split (PRIOR_TRAINING_FILES)'
            elif news_id in prior_dev:
                inclusion, evidence = 'included_prior_dev_split', 'id in a prior valid/test split (PRIOR_DEV_FILES)'
            else:
                inclusion, evidence = 'unknown', 'not in known prior trid splits; FineWeb-2 overlap not checked'
            yield {'doc_family_id': f'korea-kr:{news_id}', 'source': 'korea-kr-briefing', 'doc_type': '보도자료',
                   'institution': ministry, 'institution_type': '중앙부처' if ministry else None, 'ministry': ministry,
                   'title': d.get('title'), 'date': d.get('date'), 'url': d.get('source'), 'licence': d.get('license'),
                   'text_status': 'secured', 'text_path': 'data/korea-kr-briefing-source/docs.jsonl',
                   'text_sha256': d.get('sha256'), 'training_inclusion': inclusion,
                   'training_inclusion_evidence': evidence, 'business_group': group, 'business_group_rule': rule}


def assembly_rows(s3, index):
    state = read_json_s3(s3, f'corpus/state/assembly-minutes/{index}', {'meetings': {}})
    secured = {k.rsplit('/', 1)[1].split('.')[0] for k in list_keys(s3, 'corpus/raw/assembly-minutes/')}
    for mid, m in state['meetings'].items():
        yield {'doc_family_id': f'assembly-minutes:{mid}', 'source': 'assembly-minutes', 'doc_type': '회의록',
               'institution': '국회', 'institution_type': '국회', 'title': None, 'date': None,
               'meeting_class': ASSEMBLY_CLASSES.get(str(m.get('class'))), 'term': m.get('th'),
               'url': f'https://record.assembly.go.kr/assembly/viewer/minutes/xml.do?id={mid}&type=view',
               'licence': 'KOGL-Type1 (record.assembly.go.kr footer)',
               'text_status': 'secured' if mid in secured else 'catalogued',
               'text_path': f'corpus/raw/assembly-minutes/{mid}.html.gz' if mid in secured else None,
               'training_inclusion': 'unknown',
               'training_inclusion_evidence': 'not in any B clean-v1 output; FineWeb-2 overlap not checked',
               'business_group': '미분류', 'business_group_rule': 'committee name not fetched yet'}


def benchmark_report(rows, families_key, families_sha, alio_total, status):
    by = lambda pred, *keys: dict(Counter(' | '.join(str(r.get(k)) for k in keys) for r in rows if pred(r)))
    every = lambda r: True
    report = {
        'id': 'benchmark-source-inventory', 'owner': 'B (01a18644-74cb-75c1-8098-96d4f5c0e5fe)',
        'instructions': ['BENCHMARK-SCALE-001', 'BENCHMARK-RESERVATION-001'],
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'what_this_is': ('candidate document families the supervisor may reserve for the final test; no questions or '
                         'answers are made here. Counts are documents/families that exist, not reviewed test items.'),
        'doc_family_semantics': {
            'institutions:alio-rule:<seq>': 'one ALIO 내부규정 entry with every revision file and attachment',
            'korea-kr:<newsId>': 'one 정책브리핑 보도자료 with its attachments',
            'assembly-minutes:<id>': 'one 국회 회의록 (one meeting; 부록 included)'},
        'families_file': {'s3': f's3://{BUCKET}/{families_key}', 'sha256': families_sha, 'rows': len(rows),
                          'fields': sorted({k for r in rows for k in r})},
        'counts': {
            'by_source_text_status': by(every, 'source', 'text_status'),
            'by_business_group_doc_type': by(every, 'business_group', 'doc_type'),
            'by_business_group_institution_type': by(every, 'business_group', 'institution_type'),
            'by_training_inclusion': by(every, 'source', 'training_inclusion'),
            'secured_and_not_known_in_training_by_business_group':
                by(lambda r: r['text_status'] == 'secured' and r['training_inclusion'] == 'unknown', 'business_group', 'doc_type'),
            'institutions_per_source': {s: len({r['institution'] for r in rows if r['source'] == s and r['institution']})
                                        for s in sorted({r['source'] for r in rows})},
            'alio_rules_catalogued': sum(r['source'] == 'institutions' for r in rows),
            'alio_rules_total_reported_by_site': alio_total},
        'business_group_note': ('rule-based proposal from title keywords, then ministry/committee; not reviewed. '
                                '미분류 rows are reported, never folded into another group.'),
        'doc_types_not_yet_inventoried': ['공고', '업무편람/매뉴얼', '서식/표', 'FAQ', '지자체 보도자료', 'PRISM 정책연구보고서',
                                          '지방공기업 문서'],
        'blocked_or_pending': [
            {'source': '지방공기업 (클린아이 www.cleaneye.go.kr)', 'state': 'blocked',
             'reason': 'robots.txt "Disallow: /" (2026-10-29T18:36Z); need another official list (행정안전부 공개 현황/data.go.kr)'},
            {'source': 'PRISM', 'state': 'pending', 'reason': 'SPA backed by api.prism.go.kr; endpoints and per-report licence not inventoried yet'},
            {'source': '광역 지자체 17곳 보도자료', 'state': 'pending',
             'reason': 'robots.txt saved per host (data/corpus-inventory/robots/); several hosts Disallow most paths (seoul, daejeon, sejong, gyeongnam, jeju), busan 401, chungbuk unreachable; press-board paths not mapped'},
            {'source': 'ALIO 내부규정 원문(HWP)', 'state': 'pending',
             'reason': ('catalogue only. Text path verified on one file: /occasional/findRuleDtl.json?seq=10794 lists every '
                        'revision file (bFiles), /download/rulefiledown.json?fileNo=69727 returned HWP 5 (49,152 B), '
                        'trid.service.ingest.tool_text (rhwp, MIT) gave 15,447 chars incl. 제정/개정 history. '
                        'Bulk detail/file requests not run.')},
            {'source': '국회 회의록 위원회명·날짜', 'state': 'pending', 'reason': 'list walk stores ids only; committee/date come from the fetched minutes'}],
        'reservation_handling': ('reserve by doc_family_id; reserved families must be excluded from training in every '
                                 'source (A tokens stage) and B will route them to quarantine:benchmark_reserved once the '
                                 'supervisor publishes the reservation manifest. B keeps collecting other documents.'),
        'limitations': [
            'training_inclusion is exact-id evidence only; unknown means not proven absent (FineWeb-2 web text may contain copies).',
            'press releases are one source type from central ministries; they cannot stand in for 8 업무군 coverage.',
            'ALIO counts are catalogue entries (metadata); rule text is not yet collected.'],
        'status': status}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', default='inventory-v1')
    p.add_argument('--assembly-index', default='index.json')
    p.add_argument('--status', default='partial', help="report status; 'ready_for_selection' only when reviewable")
    args = p.parse_args()
    s3 = s3_client()
    rows, alio_total = [], None
    for row, total in alio_rows(s3):
        rows.append(row)
        alio_total = total
    rows += list(press_rows(ids_in(PRIOR_TRAINING_FILES), ids_in(PRIOR_DEV_FILES)))
    rows += list(assembly_rows(s3, args.assembly_index))
    body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(), mtime=0)
    families_key = f'corpus/state/inventory/{args.version}/doc-families.jsonl.gz'
    s3.put_object(Bucket=BUCKET, Key=families_key, Body=body)
    local = ROOT / f'data/corpus-inventory/{args.version}/doc-families.jsonl.gz'
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(body)
    report = benchmark_report(rows, families_key, hashlib.sha256(body).hexdigest(), alio_total, args.status)
    report['families_file']['local'] = str(local.relative_to(ROOT))
    text = json.dumps(report, ensure_ascii=False, indent=1) + '\n'
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/inventory/{args.version}/benchmark-source-inventory.json', Body=text.encode())
    (ROOT / 'reports/coordination/benchmark-source-inventory.json').write_text(text)
    print(json.dumps(report['counts'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

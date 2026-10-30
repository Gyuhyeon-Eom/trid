"""Licensed practical guides and official Q&A sets from data.go.kr (collect_datago_files.py) -> rows with provenance.

Q&A CSVs become one row per question (the CSV row number is kept); manuals/편람 (PDF, HWPX) become one row per file via
kordoc 4.24.23 (tables kept). Every row keeps: institution and its type, the area, who the guidance applies to, the
date and what that date is, the official source page, licence evidence, raw sha256, PII redaction and quality.
Nothing is approved for training; rows with tables need a row/column check first; overlap with the 148 signed
reserved families is checked separately (852 families unsigned, unchecked).

  uv run -q --no-project --python 3.12 --with 'ko-pii==1.22.15' --with kiwipiepy python scripts/extract_public_guides.py --version guides-v1
"""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from collect_admrul_attachments import extract_text, strip_table_markup  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
OUT = ROOT / 'data/public-data'
# Per dataset, from its data.go.kr page and the file itself. applies_to: whose work the text governs.
DATASETS = {
    '15043134': dict(institution='인사혁신처', institution_type='중앙행정기관', area='인사·징계', doc_type='업무편람',
                     applies_to='국가공무원 징계 업무', date='2022-09-01', date_type='데이터셋 파일 기준일(편람 2022년판)'),
    '15038493': dict(institution='행정안전부', institution_type='중앙행정기관', area='문서·결재', doc_type='업무편람',
                     applies_to='행정기관 행정업무(행정업무의 운영 및 혁신에 관한 규정)', date='2025-12-17',
                     date_type='편람 최종본 날짜(파일명), 게시 2026-01-02'),
    '15106457': dict(institution='행정안전부', institution_type='중앙행정기관', area='공용차량 관리', doc_type='매뉴얼',
                     applies_to='행정기관 공용차량 관리·운영', date='2024-11-25', date_type='데이터셋 파일 기준일'),
    '15151524': dict(institution='한국국제협력단', institution_type='공공기관(준정부기관)', area='계약·조달', doc_type='질의응답 세트',
                     applies_to='KOICA 원조조달 계약(답변 출처: 원조조달 업무 매뉴얼). 다른 기관에 그대로 적용되지 않음',
                     date='2025-10-31', date_type='데이터셋 파일 기준일'),
    '15121643': dict(institution='한국산업기술기획평가원', institution_type='공공기관(준정부기관)', area='연구비 집행·회계',
                     doc_type='FAQ', applies_to='산업기술 R&D 과제 연구비(RCMS) 집행', date='2026-07-13', date_type='데이터셋 파일 기준일'),
    '15064076': dict(institution='한전KDN(주)', institution_type='공기업', area='전자입찰 시스템', doc_type='FAQ 목록',
                     applies_to='한전KDN 전자입찰시스템 이용자', date='2024-12-31', date_type='데이터셋 파일 기준일'),
    '15118841': dict(institution='방위사업청', institution_type='중앙행정기관', area='전자조달 시스템', doc_type='FAQ',
                     applies_to='국방전자조달시스템 이용 업체·담당자', date=None, date_type='행별 등록일·수정일'),
    '15169354': dict(institution='국민권익위원회', institution_type='중앙행정기관', area='복무·청탁금지', doc_type='질의응답',
                     applies_to='청탁금지법상 공직자등(공공기관 임직원 포함)', date=None, date_type='행별 자료발간연도'),
    '15038798': dict(institution='행정안전부', institution_type='중앙행정기관', area='지방재정·회계', doc_type='해설사례집',
                     applies_to='지방자치단체 재정 운영', date='2019-09-11', date_type='데이터셋 파일 기준일(2019년판, 이후 개정 미반영)'),
    '15077063': dict(institution='재정경제부', institution_type='중앙행정기관', area='공공기관 경영평가', doc_type='편람',
                     applies_to='공기업·준정부기관 경영실적 평가(2026년도 평가)', date='2026-01-23',
                     date_type='게시일(재공지), 2026년도 편람'),
    '15038527': dict(institution='행정안전부', institution_type='중앙행정기관', area='인사', doc_type='업무 해설서',
                     applies_to='지방공무원 인사(지방자치단체)', date='2025-11-11', date_type='데이터셋 파일 기준일(2025년판)'),
    '15152718': dict(institution='한국국제협력단', institution_type='공공기관(준정부기관)', area='인사·복무', doc_type='질의응답 세트',
                     applies_to='KOICA 직원(복무규정 등 KOICA 내부규정). 다른 기관에 그대로 적용되지 않음',
                     date='2025-10-31', date_type='데이터셋 파일 기준일'),
    '15146890': dict(institution='조달청', institution_type='중앙행정기관', area='계약·조달', doc_type='질의회신 해석사례',
                     applies_to='국가계약법령·조달 제도 (국민신문고 회신)', date=None, date_type='행별 회신일자'),
    '15170150': dict(institution='한국남부발전(주)', institution_type='공기업', area='사내 정보시스템', doc_type='사내 FAQ',
                     applies_to='남부발전 직원·협력업체 시스템 이용', date=None, date_type='행별 등록일자'),
    '15173808': dict(institution='국가철도공단', institution_type='공공기관(준정부기관)', area='전자조달 시스템·계약', doc_type='FAQ·질의응답',
                     applies_to='KR 전자조달시스템 이용자', date=None, date_type='행별 등록일자·답변일자'),
    '15144457': dict(institution='국민권익위원회', institution_type='중앙행정기관', area='복무·청탁금지', doc_type='질의응답 게시판',
                     applies_to='청탁금지법 질의(국민 질의, 기관 답변)', date=None, date_type='행별 등록일자·답변일자'),
    '15131770': dict(institution='한국동서발전(주)', institution_type='공기업', area='실적증명·SRM', doc_type='질의응답 게시글',
                     applies_to='동서발전 실적증명·협력시스템 이용 업체', date=None, date_type='행별 작성 일자'),
    # B-019 slices (guides-v7 and later; v6 is a fixed consumer contract)
    '15120505': dict(institution='한국자산관리공사', institution_type='공공기관(준정부기관)', area='국유재산', doc_type='FAQ',
                     applies_to='캠코 국유재산포털 이용자(국유재산 대부·매각 등)', date=None, date_type='행별 게시일자'),
    '15033217': dict(institution='충청남도', institution_type='지방자치단체', area='계약심사', doc_type='사례집',
                     applies_to='충청남도 사전 계약심사(설계·산출내역·단가)', date='2019-12-31', date_type='데이터셋 파일 기준일'),
    '15174522': dict(institution='조달청', institution_type='중앙행정기관', area='계약·조달 용어', doc_type='용어사전',
                     applies_to='정부조달 용어', date=None, date_type='행별 입력일시'),
    '15099914': dict(institution='인사혁신처', institution_type='중앙행정기관', area='보수·수당·여비', doc_type='업무지침(예규)',
                     applies_to='국가공무원 보수 업무(2026년)', date='2026-01-22', date_type='데이터셋 파일 기준일, 2026년 지침'),
    '15163350': dict(institution='제주특별자치도 감사위원회', institution_type='지방자치단체', area='사전컨설팅 감사', doc_type='사례 목록',
                     applies_to='제주 사전컨설팅 감사', date=None, date_type='행별 작성일'),
    '15138170': dict(institution='한국승강기안전공단', institution_type='공공기관(준정부기관)', area='출장여비', doc_type='기준 단가표',
                     applies_to='승강기안전공단 내부 출장여비', date=None, date_type='행별 적용시작·종료일자'),
    '15038805': dict(institution='행정안전부', institution_type='중앙행정기관', area='예산편성·기금', doc_type='훈령·예규',
                     applies_to='지방자치단체 2026년도 예산편성·기금운용계획', date='2025-07-31',
                     date_type='개정일(훈령 제404호, 예규 제336호)'),
    '15086706': dict(institution='한국동서발전(주)', institution_type='공기업', area='입찰·계약', doc_type='안내자료',
                     applies_to='동서발전 입찰·계약 참여 업체', date='2021-08-26', date_type='게시일'),
    '15043018': dict(institution='한국토지주택공사', institution_type='공기업', area='인사(직무)', doc_type='직무설명서',
                     applies_to='LH 업무직 채용 직무', date='2024-05-14', date_type='데이터셋 파일 기준일'),
    # B-022
    '15152342': dict(institution='한국해양진흥공사', institution_type='공공기관(준정부기관)', area='회계(대손충당금)', doc_type='시스템 기준 테이블',
                     applies_to='해양진흥공사 내부 대손충당금 설정 기준', date='2025-10-29', date_type='데이터셋 파일 기준일'),
    '15110681': dict(institution='한국토지주택공사', institution_type='공기업', area='보수', doc_type='기관 내부규정(사규)',
                     applies_to='LH 직원 보수 (규정 제299호, 2023-12-27 개정). 다른 기관에 그대로 적용되지 않음',
                     date='2023-12-27', date_type='규정 개정일(파일명)'),
    # B-024 (types left as '공공기관' where the subtype was not checked)
    '15152480': dict(institution='주택도시보증공사', institution_type='공공기관', area='보증 심사(PF보증)', doc_type='심사실무 사례집',
                     applies_to='HUG PF보증 심사 실무. 다른 기관에 그대로 적용되지 않음', date='2025-11-10',
                     date_type='사례집 기준일(파일명), 포털 등록 2025-11-17'),
    '15136239': dict(institution='한전케이피에스주식회사', institution_type='공공기관', area='정보화(정보시스템 개발·운영)',
                     doc_type='기관 절차서', applies_to='한전KPS 정보화 사업 관리·개발·운영 절차. 다른 기관에 그대로 적용되지 않음',
                     date='2024-07-31', date_type='절차서 기준일(파일명)'),
    '15039340': dict(institution='행정안전부', institution_type='중앙행정기관', area='자치법규 입법절차', doc_type='업무매뉴얼',
                     applies_to='지방자치단체 자치법규(조례·규칙) 제정·개정 업무', date='2022-02-23', date_type='매뉴얼 기준일(파일명)'),
    '15156787': dict(institution='국토교통부', institution_type='중앙행정기관', area='해외건설 세무', doc_type='업무매뉴얼',
                     applies_to='해외건설 사업자의 세무 업무(국내세법·해외 세무 안내)', date='2022-04-04', date_type='매뉴얼 기준일(파일명)'),
    '15040179': dict(institution='한국환경공단', institution_type='공공기관', area='공사감독(환경시설 설치공사)', doc_type='업무매뉴얼',
                     applies_to='한국환경공단 환경기초시설 설치공사 감독업무. 다른 기관에 그대로 적용되지 않음', date='2016-12-14',
                     date_type='매뉴얼 기준일(파일명), 장별 HWP 묶음'),
}
# datasets kept as raw but not turned into Q&A rows: no answer text (title list) or a numeric table, not guidance
NOT_QA = {'15163350': 'title_list_without_content', '15138170': 'structured_table_not_qa',
          '15152342': 'structured_table_not_qa_with_employee_user_ids'}


def decode(raw):
    """(text, encoding, replaced): strict UTF-8 / CP949 first; a file with a few bad bytes is read as CP949 with
    replacement characters, and rows containing them are held."""
    for enc in ('utf-8-sig', 'cp949'):
        try:
            return raw.decode(enc), enc, False
        except UnicodeDecodeError:
            pass
    return raw.decode('cp949', errors='replace'), 'cp949+replace', True



def zip_name(info):
    """Member name: zipfile decodes names without the UTF-8 flag as CP437; Korean tools write them in CP949."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('cp949')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename

Q_COLS = ('질문', '질의내용', '질문내용', '조달용어명')
A_COLS = ('답변', '회신내용', '정답', '답변내용', '조달용어설명')
DATE_COLS = ('회신일자', '답변일자', '등록일자', '등록일', '자료발간연도', '작성 일자')

def doc_id(s):
    """Document row id. guides-v6 used data.go.kr:<pk> (one file per dataset there); from guides-v7 a dataset may hold
    several files (LH 직무설명서: 12), so the file's sha256 prefix is part of the id."""
    return f"data.go.kr:{s['pk']}:file:{s['sha256'][:16]}"



def qa_from_csv(pk, text):
    """One (title, question, answer, extra fields, holds) per CSV row; column names differ per dataset."""
    for n, it in enumerate(csv.DictReader(io.StringIO(text)), 1):
        spill = it.pop(None, None)  # cells beyond the header: the row does not line up with its columns
        it = {(k or '').strip(): (v or '').strip() for k, v in it.items()}
        q = next((it[c] for c in Q_COLS if it.get(c)), '') or it.get('제목', '') or it.get('주제', '')
        a = next((it[c] for c in A_COLS if it.get(c)), '') or it.get('내용', '')
        holds = ['csv_row_misaligned'] if spill else []
        if '\ufffd' in q + a:
            holds.append('decode_replacement')
        if pk == '15144457' and (it.get('등록일자') or '9999') < '2016':
            holds.append('answer_may_be_user_comment')  # dataset description: before 2016 answers were user comments
        if pk == '15131770':
            holds.append('question_answer_not_separated')  # one free-text post per row
        if pk in NOT_QA:
            holds.append(NOT_QA[pk])
        extra = {k: v for k, v in it.items() if k not in Q_COLS + A_COLS + ('내용',)}
        yield n, (it.get('제목') or q), q, a, extra, holds


MENU_PATH = re.compile(r'[가-힣A-Za-z]+\s*>\s*[가-힣A-Za-z]+')
RULE_CITE = re.compile(r'제\d+조|[가-힣](?:법|법률|시행령|시행규칙|규정|규칙|지침|예규|고시|훈령|기준|매뉴얼|편람)(?=[\s」』)\],.]|$)')
SYSTEM_CATEGORY = re.compile(r'로그인|회원|인증|시스템|브라우저|프로그램|오류|접속|전산|RCMS일반|사용자 정보')


def faq_signals(answer, category=''):
    """Separates system-operation FAQs from rule/procedure answers by visible signals only (heuristic, recorded as such):
    menu paths in the answer ('A > B'), citations of a rule (제N조, …법, …지침, …매뉴얼), a system-like source category."""
    menu, cites, system = len(MENU_PATH.findall(answer)), bool(RULE_CITE.search(answer)), bool(SYSTEM_CATEGORY.search(category))
    if cites and not menu and not system:
        kind = 'rule_or_procedure'
    elif (menu or system) and not cites:
        kind = 'system_operation'
    elif cites:
        kind = 'mixed'
    else:
        kind = 'other'
    return {'menu_paths': menu, 'cites_rule': cites, 'system_category': system, 'faq_kind': kind,
            'faq_kind_method': 'heuristic: menu paths / rule citations / source category'}


def resplit(src_version, out_version, notes_path):
    """Exclusive status per row of an extracted version: every row lands in exactly one part. Overlap with the signed
    reserved families and reviewer source notes (typos are annotated, never corrected) put a row on hold."""
    src, out = OUT / src_version, OUT / out_version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    rep = json.loads((src / 'overlap-148.json').read_text())
    overlap = {m['corpus_id'] for m in rep.get('matches') or []} | {x['corpus_id'] for x in rep.get('partial_review') or []}
    notes = json.loads(Path(notes_path).read_text())['notes'] if notes_path else {}
    parts = {k: [] for k in ('qa_rule_or_procedure', 'qa_mixed', 'qa_other', 'qa_system_operation', 'documents', 'held')}
    for name in ('qa_items', 'documents', 'held'):
        for x in (src / f'{name}.jsonl').read_text().split('\n'):
            if not x:
                continue
            r = json.loads(x)
            holds = [f[5:] for f in r['flags'] if f.startswith('hold:')]
            if r['id'] in overlap:
                holds.append('reserved_overlap')
            if r['id'] in notes:
                r['meta']['source_note'] = notes[r['id']]
                holds.append('source_note')
            if r['kind'] == 'qa_item':
                fields = r['meta'].get('fields') or {}
                category = r['meta'].get('category_path') or ' > '.join(
                    fields[c] for c in ('구분', '문서 구분', '문의분류', '대분류', '중분류', '소분류') if fields.get(c))
                r['meta'].update(faq_signals(r['text'].split('\n답변: ', 1)[-1], category))
            r['meta']['status_holds'] = holds
            r['meta']['clean_version'] = out_version
            part = 'held' if holds else ('documents' if r['kind'] == 'guide_document' else f"qa_{r['meta']['faq_kind']}")
            parts[part].append(r)
    out.mkdir(parents=True)
    man = {}
    for name, sel in parts.items():
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        man[name] = {'rows': len(sel), 'characters': sum(len(r['text']) for r in sel), 'sha256': hashlib.sha256(body).hexdigest(),
                     'by_institution': dict(Counter(r['provenance']['institution'] for r in sel)),
                     'holds': dict(Counter(h for r in sel for h in r['meta']['status_holds']))}
    manifest = {'version': out_version, 'source_version': src_version, 'code_sha256': CODE_SHA256,
                'notes_file': notes_path, 'rows_total': sum(v['rows'] for v in man.values()), 'parts': man,
                'note': 'parts are exclusive; not approved for training; documents carry tables (unverified)'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    problems = check_version(out_version)
    print(json.dumps({**manifest, 'consumer_check': problems or 'ok'}, ensure_ascii=False, indent=1))
    if problems:
        raise SystemExit('consumer check failed')


CANDIDATE_PARTS = ('qa_rule_or_procedure', 'qa_mixed', 'qa_other', 'qa_system_operation', 'documents')


def check_version(version):
    """What a consumer (C) relies on, checked on the written files: every id in exactly one part; no row in a
    candidate part carries a hold, a reserved-overlap id or a reviewer note; manifest row counts and sha256 match."""
    out = OUT / version
    man = json.loads((out / 'manifest.json').read_text())
    src = OUT / man['source_version']
    rep = json.loads((src / 'overlap-148.json').read_text())
    overlap = {m['corpus_id'] for m in rep.get('matches') or []} | {x['corpus_id'] for x in rep.get('partial_review') or []}
    notes = set(json.loads(Path(man['notes_file']).read_text())['notes']) if man.get('notes_file') else set()
    problems, seen = [], {}
    for name, info in man['parts'].items():
        body = (out / f'{name}.jsonl').read_bytes()
        rows = [json.loads(x) for x in body.decode().split('\n') if x]
        if len(rows) != info['rows'] or hashlib.sha256(body).hexdigest() != info['sha256']:
            problems.append(f'{name}: manifest rows/sha do not match the file')
        for r in rows:
            if r['id'] in seen:
                problems.append(f"{r['id']}: in {seen[r['id']]} and {name}")
            seen[r['id']] = name
            if name in CANDIDATE_PARTS and (r['meta']['status_holds'] or any(f.startswith('hold:') for f in r['flags'])
                                            or r['id'] in overlap or r['id'] in notes):
                problems.append(f"{r['id']}: held row in candidate part {name}")
    expected = sum(1 for p in ('qa_items', 'documents', 'held') for x in (src / f'{p}.jsonl').read_text().split('\n') if x)
    if len(seen) != expected:
        problems.append(f'{len(seen)} rows written, {expected} in {src.name}')
    return problems


def main():
    from trid.korean_quality import PUBLIC_DOCS, assess, redact, redacted_in_url
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', required=True)
    p.add_argument('--resplit', metavar='SRC_VERSION', help='exclusive status split of an extracted version')
    p.add_argument('--notes', help='reviewer source notes JSON {"notes": {row id: note}}')
    p.add_argument('--check', action='store_true', help='consumer check of a resplit version (--version); exit 1 on problems')
    p.add_argument('--state-files', nargs='+', default=['files-state.jsonl'],
                   help='collector state files (under data/public-data) whose stored files make up this version')
    args = p.parse_args()
    if args.check:
        problems = check_version(args.version)
        print(json.dumps(problems or 'ok', ensure_ascii=False, indent=1))
        raise SystemExit(1 if problems else 0)
    if args.resplit:
        return resplit(args.resplit, args.version, args.notes)
    out = OUT / args.version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    state = [json.loads(x) for name in args.state_files for x in (OUT / name).read_text().split('\n') if x]
    stored = [s for s in state if s['status'] == 'stored']
    rows = []
    for s in stored:
        ds = DATASETS[s['pk']]
        path = ROOT / s['raw']
        base = {'source': s['page'], 'license': s['licence'], 'raw': s['raw'], 'raw_sha256': s['sha256'], 'collected_at': s['checked_at']}
        if path.suffix.lower() in ('.csv', '.zip'):
            if path.suffix.lower() == '.zip':
                import zipfile
                z = zipfile.ZipFile(path)
                members = [(zip_name(i), z.read(i)) for i in z.infolist() if i.filename.lower().endswith('.csv')]
                if not members:  # e.g. a zipped set of forms: kept as raw, not extracted
                    rows.append((ds, base, doc_id(s), s['file_name'], '', {'extractor': 'none', 'pre_holds': ['archive_without_csv']}))
            else:
                members = [('', path.read_bytes())]
            for member, raw in members:
                text_all, enc, _ = decode(raw)
                prefix = f"data.go.kr:{s['pk']}:" + (f'{member}:' if member else '')
                for n, title, q, a, extra, holds in qa_from_csv(s['pk'], text_all):
                    date = next((extra[c] for c in DATE_COLS if extra.get(c)), None)
                    entities = len(re.findall(r'&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);', q + a))
                    # HTML entities stored literally in the CSV (&nbsp; &#8228; ...) are markup, not wording: decoded
                    # deterministically and counted; spelling, truncation and wording are never changed
                    q, a = html.unescape(q).replace('\xa0', ' '), html.unescape(a).replace('\xa0', ' ')
                    rows.append((ds, base, f'{prefix}{n}', q, a,
                                 {'csv_row': n, 'csv_member': member or None, 'csv_encoding': enc, 'fields': extra,
                                  'row_date': date, 'title': title, 'html_entities_decoded': entities, 'pre_holds': holds}))
        else:
            text, how = extract_text(path)
            rows.append((ds, base, doc_id(s), s['fields'].get('파일데이터명') or s['file_name'], text,
                         {'extractor': how}))
    parts = {'qa_items': [], 'documents': [], 'held': []}
    for ds, base, rid, title, body, meta in rows:
        is_qa = 'csv_row' in meta
        text = unicodedata.normalize('NFC', f'질문: {title}\n답변: {body}' if is_qa else body)
        text, pii = redact(text)
        plain = strip_table_markup(text)
        ok, why = assess(plain, t=PUBLIC_DOCS) if plain.strip() else (False, 'empty')
        holds = list(meta.pop('pre_holds', []))
        if is_qa and not body:
            holds.append('no_answer_content')
        if not plain.strip():
            holds.append('empty')
        if redacted_in_url(text):
            holds.append('pii_redaction_inside_url')
        row = {'id': rid, 'group': rid.rsplit(':', 1)[0] if is_qa else rid, 'kind': 'qa_item' if is_qa else 'guide_document',
               'title': title[:200], 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(), **base,
               'flags': (['pii_redacted'] if pii else []) + [f'hold:{h}' for h in holds],
               'provenance': {k: ds[k] for k in ('institution', 'institution_type', 'area', 'doc_type', 'applies_to', 'date', 'date_type')},
               'meta': {**meta, 'quality': 'ok' if ok else why, 'pii_redactions': dict(pii),
                        'table_rows': sum(1 for l in text.split('\n') if l.startswith('| ')) + text.count('<tr'),
                        'table_check': 'unverified', 'clean_version': args.version, 'code_sha256': CODE_SHA256}}
        if not row['meta']['table_rows']:
            row['meta']['table_check'] = 'no_tables'
        parts['held' if holds else ('qa_items' if is_qa else 'documents')].append(row)
    out.mkdir(parents=True)
    man = {}
    for name, sel in parts.items():
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        man[name] = {'rows': len(sel), 'characters': sum(len(r['text']) for r in sel), 'sha256': hashlib.sha256(body).hexdigest(),
                     'by_dataset': dict(Counter(r['raw'].split('/')[3] for r in sel)),
                     'quality': dict(Counter(r['meta']['quality'] for r in sel)),
                     'pii_rows': sum(1 for r in sel if r['meta']['pii_redactions']),
                     'table_check': dict(Counter(r['meta']['table_check'] for r in sel))}
    with open(out / 'overlap-input.jsonl', 'w') as f:
        for sel in parts.values():
            for r in sel:
                f.write(json.dumps({'id': r['id'], 'group': r['group'], 'source': r['source'], 'text': r['text']}, ensure_ascii=False) + '\n')
    manifest = {'version': args.version, 'code_sha256': CODE_SHA256, 'datasets': DATASETS,
                'not_collected': [{'pk': s['pk'], 'status': s['status'], 'licence': s['licence'],
                                   'title': s['fields'].get('파일데이터명')} for s in state if s['status'] != 'stored'],
                'parts': man, 'note': 'not approved for training; tables unverified; run check_reserved_overlap.py on overlap-input.jsonl'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

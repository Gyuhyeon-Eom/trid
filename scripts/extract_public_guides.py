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
import io
import json
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
}


def decode(raw):
    for enc in ('utf-8-sig', 'cp949'):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            pass
    raise ValueError('unknown encoding')


def main():
    from trid.korean_quality import PUBLIC_DOCS, assess, redact, redacted_in_url
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', required=True)
    args = p.parse_args()
    out = OUT / args.version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    state = [json.loads(x) for x in (OUT / 'files-state.jsonl').read_text().split('\n') if x]
    stored = [s for s in state if s['status'] == 'stored']
    rows = []
    for s in stored:
        ds = DATASETS[s['pk']]
        path = ROOT / s['raw']
        base = {'source': s['page'], 'license': s['licence'], 'raw': s['raw'], 'raw_sha256': s['sha256'], 'collected_at': s['checked_at']}
        if path.suffix == '.csv':
            text_all, enc = decode(path.read_bytes())
            items = list(csv.DictReader(io.StringIO(text_all)))
            for n, it in enumerate(items, 1):
                q = (it.get('질문') or it.get('제목') or '').strip()
                a = (it.get('답변') or it.get('내용') or '').strip()
                extra = {k: v for k, v in it.items() if k not in ('질문', '답변', '제목', '내용')}
                rows.append((ds, base, f"data.go.kr:{s['pk']}:{n}", q, a, {'csv_row': n, 'csv_encoding': enc, 'fields': extra}))
        else:
            text, how = extract_text(path)
            rows.append((ds, base, f"data.go.kr:{s['pk']}", s['fields'].get('파일데이터명') or s['file_name'], text,
                         {'extractor': how}))
    parts = {'qa_items': [], 'documents': [], 'held': []}
    for ds, base, rid, title, body, meta in rows:
        is_qa = 'csv_row' in meta
        text = unicodedata.normalize('NFC', f'질문: {title}\n답변: {body}' if is_qa else body)
        text, pii = redact(text)
        plain = strip_table_markup(text)
        ok, why = assess(plain, t=PUBLIC_DOCS) if plain.strip() else (False, 'empty')
        holds = []
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

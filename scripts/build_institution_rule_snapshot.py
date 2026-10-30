"""Institution public rules (사규 posted on official sites / data.go.kr) as a separate search-index candidate.

Same snapshot layout and record fields as API-S3-001 (reports/coordination/HYBRID-RAG-001-SCHEMA.md) so A's ingester reads
it, plus fields that keep the rule out of evaluation and training:
  holds        ['evaluation', 'training']: an institution's own rule may overlap the 852 unsigned reserved families
  index        'institution_rules_separate': not to be mixed into the evaluation-run index
  currency     the posted version only; nothing here is called current
The source rows come from a guides version (B's data.go.kr file collection), whose text is already PII-redacted
(ko-pii); issuer ids come from the official ALIO registry copy (data/corpus-inventory/alio-agencies.json, no request).
No rule is labelled the institution's non-public internal data: it is a publicly posted rule.

  .venv/bin/python scripts/build_institution_rule_snapshot.py --guides guides-v8 \
      --doc data.go.kr:15110681:file:30bc63ea4a81b527 --snapshot-id inst-rule-lh-pay-001 --out runs/search-snapshots/inst-rule-lh-pay-001
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.korean_quality import PII_HINT  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
REGISTRY = ROOT / 'data/corpus-inventory/alio-agencies.json'


def korean_date(text, label):
    m = re.search(label + r'\s*[:：]\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', text)
    return f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}' if m else None


SPEND_HEAD = re.compile(r"[‘'](\d{2})년\s*(\d{1,2})월\s*(\S+)\s+업무추진비\s*집행내역")


def spend_sections(text, raw_pdf):
    """Officer sections of a 한국부동산원 임원 업무추진비 집행내역 PDF, each checked by two readers: the kordoc table rows
    (date, 내역, 사용처, 대상, 인원, 금액 + 합계 row) and pdftotext -layout lines (a dated line ends with 인원 and 금액; the
    합계 line ends with the total). A section is verified when both readers give the same (date, 인원, 금액) rows and the
    rows add up to the printed 합계 in both. -> [{year, month, position, rows, total, verified, problems}]"""
    import subprocess
    layout = subprocess.run(['pdftotext', '-layout', str(raw_pdf), '-'], capture_output=True, text=True).stdout
    num = lambda s: int(s.replace(',', ''))

    def split(t):
        heads = list(SPEND_HEAD.finditer(t))
        return [(h, t[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(t)]) for i, h in enumerate(heads)]
    pdf = {(h.group(1), h.group(2), h.group(3)): body for h, body in split(layout)}
    out = []
    for h, body in split(text):
        key = (h.group(1), h.group(2), h.group(3))
        rows, total, problems = [], None, []
        for tr in re.findall(r'<tr>(.*?)</tr>', body, re.S):
            cells = [re.sub(r'<br\s*/?>', ' ', c).strip() for c in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
            if len(cells) == 6 and re.fullmatch(r'\d{4}-\d\d-\d\d', cells[0]):
                rows.append({'date': cells[0], '내역': cells[1], '사용처': cells[2], '대상': cells[3],
                             '인원': num(cells[4]), '금액': num(cells[5])})
            elif any(re.sub(r'\s', '', c) == '합계' for c in cells):
                total = num(cells[-1])
        other = pdf.get(key)
        if other is None:
            problems.append('section_missing_in_pdftotext')
        else:
            lines = [re.search(r'(\d{4}-\d\d-\d\d)\s.*?\s(\d+)\s+([\d,]+)\s*$', l) for l in other.splitlines()]
            pdf_rows = sorted((m.group(1), int(m.group(2)), num(m.group(3))) for m in lines if m)
            pdf_total = re.search(r'합\s*계\s+([\d,]+)', other)
            if pdf_rows != sorted((r['date'], r['인원'], r['금액']) for r in rows):
                problems.append('rows_differ_between_readers')
            if not pdf_total or num(pdf_total.group(1)) != total:
                problems.append('total_differs_between_readers')
        if total is None or sum(r['금액'] for r in rows) != total:
            problems.append('rows_do_not_add_up_to_total')
        out.append({'year': 2000 + int(h.group(1)), 'month': int(h.group(2)), 'position': h.group(3), 'rows': rows,
                    'total': total, 'verified': not problems, 'problems': problems})
    return out


def spend_view(sections, title):
    """Text shown to the model: every row with 사용일자·내역·대상·인원·금액; 사용처(업소명) is left out (kept in raw)."""
    lines = [title, '(사용처 업소명은 원문 PDF에 있으나 이 사본에서는 뺐습니다. 금액 단위: 원. 임원 직위별 실제 집행 내역이며 '
                    '규정상 기준이나 직원 1인 식비·평균이 아닙니다.)']
    for s in sections:
        lines += ['', f"## {s['year']}년 {s['month']}월 {s['position']} 업무추진비 집행내역",
                  '| 사용일자 | 내역 | 대상 | 인원(명) | 금액(원) |', '| --- | --- | --- | --- | --- |']
        lines += [f"| {r['date']} | {r['내역']} | {r['대상']} | {r['인원']} | {r['금액']:,} |" for r in s['rows']]
        lines.append(f"| 합계 |  |  |  | {s['total']:,} |" if s['total'] is not None else '| 합계 | (원문에서 읽지 못함) |  |  |  |')
    return '\n'.join(lines)


def spend_facts(sections, agency, url, raw_sha256):
    """One fact per verified officer-month (HYBRID-RAG-001 structured_facts shape plus population / expense_type)."""
    import calendar
    facts = []
    for s in sections:
        if not s['verified']:
            continue
        last = calendar.monthrange(s['year'], s['month'])[1]
        facts.append({'metric': '임원 업무추진비 월 집행액', 'value': s['total'], 'unit': '원',
                      'period': {'from': f"{s['year']}-{s['month']:02d}-01", 'to': f"{s['year']}-{s['month']:02d}-{last:02d}"},
                      'subject': {'id': agency['apbaId'] if agency else None, 'name': agency['apbaNa'] if agency else None},
                      'population': f"임원({s['position']}) 1명 분", 'expense_type': '업무추진비(임원 업무추진비성 경비)',
                      'executions': len(s['rows']), 'basis': 'actual_spending',
                      'not_for': '직원 1인 식비·평균, 규정상 한도로 쓰지 않음',
                      'location': {'url': url, 'raw_sha256': raw_sha256, 'section': f"{s['month']}월 {s['position']}"},
                      'verification': '표 행 합 = 표기 합계, kordoc 표 행과 pdftotext 행(일자·인원·금액) 일치'})
    return facts


def record(row, agencies, state):
    """One snapshot record. A rule without '확정일/시행일' labels in its body (board postings, B-027: 한국부동산원) takes its
    effective date from the row provenance (the last 부칙's 시행일, read by B and described in date_type). A 집행내역 row
    is kept apart from rules: kind 'expenditure_record', basis 'actual_spend', its date is the spending month."""
    text = unicodedata.normalize('NFC', row['text']).strip()
    sections = None
    if '집행내역' in (row['provenance'].get('doc_type') or ''):
        sections = spend_sections(text, ROOT / row['raw'])
        text = spend_view(sections, row['title'])
    prov = row['provenance']
    inst = prov['institution']
    agency = next((a for a in agencies if a['apbaNa'] == inst), None)
    spend = '집행내역' in (prov.get('doc_type') or '')
    number = None if spend else re.search(r'규정\s*제\s*(\d+)\s*호', text)
    published, effective = korean_date(text, r'확\s*정\s*일'), korean_date(text, r'시\s*행\s*일')
    effective_kind = '시행일(규정 본문)'
    if not spend and not effective and (prov.get('date_type') or '').startswith('마지막 부칙'):
        effective, effective_kind = prov['date'], prov['date_type']
    if spend:
        currency = f"{prov['date']} 집행분 공개 내역(실제 지출). 규정상 기준이 아니며, 이후 정정 여부는 확인하지 않음"
    else:
        currency = (f"{published or effective or prov['date']} 기준 게시본. 이후 개정 여부는 확인하지 않았고 현행이라고 표시하지 않음")
    return {
        'source_id': row['id'], 'version': f'규정 제{number.group(1)}호' if number else row['sha256'][:16],
        # 'institution_rule', not 'rule': A's ingester maps 'rule' to 행정규칙 (a ministry's administrative rule)
        'content_hash': hashlib.sha256(text.encode()).hexdigest(), 'kind': 'expenditure_record' if spend else 'institution_rule',
        'basis': 'actual_spend' if spend else 'rule_text',
        'title': row['title'] + (f' (규정 제{number.group(1)}호)' if number else ''),
        'issuer': {'id': agency['apbaId'] if agency else None, 'name': inst, 'id_scheme': 'alio_apbaId' if agency else None,
                   'type': agency['typeNa'] if agency else prov['institution_type']},
        'applies_to': {'scope': 'institution', 'ids': [agency['apbaId']] if agency else [], 'note': prov['applies_to']},
        'dates': {'published': published, 'effective_from': None if spend else effective, 'effective_to': None,
                  'spending_period': prov['date'] if spend else None, 'retrieved_at': row['collected_at'][:10]},
        'date_kinds': {'published': '확정일(규정 본문)', 'effective_from': effective_kind,
                       'spending_period': prov['date_type'] if spend else None, 'retrieved_at': '수집일'},
        'rights': {'license': row['license'], 'url': row['source'], 'restrictions': None},
        'location': {'url': state.get('provider_url') or row['source'], 'page': None, 'article_path': [],
                     'char_start': 0, 'char_end': len(text),
                     'data_go_kr': row['source'] if 'data.go.kr' in row['source'] else None},
        'text': text,
        'structured_facts': (spend_facts(sections, agency, state.get('provider_url') or row['source'], row['raw_sha256'])
                             if sections is not None else []),
        'source_notes': state.get('source_notes', []),
        'holds': ['evaluation', 'training'], 'index': 'institution_rules_separate',
        'currency': currency,
        'pii': {'check': 'ko-pii redaction in the guides version + trid.korean_quality.PII_HINT', 'hits': len(PII_HINT.findall(text)),
                'redactions': row['meta'].get('pii_redactions')},
        'table_check': row['meta'].get('table_check'), 'status': 'search_candidate',
        'collection': 'public_file_download (data.go.kr page / institution board), not an API call',
        'raw': row['raw'], 'raw_sha256': row['raw_sha256'],
        **({'spend_check': {'sections': len(sections), 'verified': sum(s['verified'] for s in sections),
                            'problems': {f"{s['month']}월 {s['position']}": s['problems'] for s in sections if s['problems']},
                            'withheld_from_text': ['사용처'],
                            'population': '임원(직위별)', 'expense_type': '업무추진비(임원 업무추진비성 경비)', 'unit': '원'}}
           if sections is not None else {}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--guides', required=True)
    p.add_argument('--doc', nargs='+', required=True)
    p.add_argument('--snapshot-id', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--producer', default='B', help="task id recorded in the manifest, e.g. 'B/B-027'")
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; snapshots are never overwritten')
    rows = {json.loads(x)['id']: json.loads(x) for part in ('documents', 'held')
            for x in (ROOT / 'data/public-data' / args.guides / f'{part}.jsonl').read_text().splitlines() if x}
    states = {}
    for path in (ROOT / 'data/public-data').glob('files-state*.jsonl'):
        for x in path.read_text().splitlines():
            s = json.loads(x) if x else None
            if s and s['status'] == 'stored':
                states[s['sha256']] = s
    agencies = json.loads(REGISTRY.read_text())['agencies']
    recs = [record(rows[d], agencies, states.get(rows[d]['raw_sha256'], {})) for d in args.doc]
    if any(r['pii']['hits'] for r in recs):
        raise SystemExit(f"PII hint in {[r['source_id'] for r in recs if r['pii']['hits']]}: review before handing over")
    (out / 'normalized').mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in recs).encode()
    (out / 'normalized/records-00000.jsonl').write_bytes(body)
    manifest = {'snapshot_id': args.snapshot_id, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'producer': args.producer, 'code_sha256': CODE_SHA256,
                'kinds': dict(Counter(r['kind'] for r in recs)),
                'sources': [{'provider': 'data.go.kr file data / institution 사규 board', 'guides_version': args.guides,
                             'records': len(recs), 'collection': 'public_file_download, not an API call'}],
                'files': [{'path': 'normalized/records-00000.jsonl', 'sha256': hashlib.sha256(body).hexdigest(), 'records': len(recs)}],
                'records_total': len(recs), 'holds': ['evaluation', 'training'], 'index': 'institution_rules_separate',
                'registry': {'path': str(REGISTRY.relative_to(ROOT)), 'sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest()},
                'notes': 'publicly posted institution rules; not non-public internal data; nothing labelled current'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('snapshot_id', 'records_total', 'files')}, ensure_ascii=False, indent=1))
    for r in recs:
        print(r['source_id'], r['version'], r['issuer'], r['dates'], r['pii'], r['table_check'])


if __name__ == '__main__':
    main()

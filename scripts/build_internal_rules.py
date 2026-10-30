"""Internal-work rules candidate (local, versioned): administrative rules (훈령·예규·고시, law.go.kr admrul) and
legislative interpretations (법령해석례, expc) about public-sector internal work, taken from the existing local
collection data/law-ko-source/docs.jsonl (read only). No network, no AWS.

Per document:
  topic by title (인사, 복무, 보수·수당, 여비, 회계·지출, 예산·기금, 계약·구매, 감사·징계, 결재·문서, 물품·자산;
  import/quarantine-inspection titles excluded) -> NFC -> PII redaction (trid.korean_quality.redact) ->
  quarantine: dev-eval sources (ids or exact text of eval files), benchmark-reserved overlap (check_reserved_overlap.py
  report: full matches and partial-review rows alike), PII redaction inside a URL -> quality verdict (PUBLIC_DOCS) as a
  flag only -> split by sha256("split:"+group).
Rights: 저작권법 제7조 (법령·훈령·예규·고시 and 법령해석 are not protected works); source attribution 법제처 국가법령정보센터.
Output: data/internal-rules/<version>/{docs,quarantine}.jsonl + manifest.json (input/code/redactor hashes).

  .venv/bin/python scripts/build_internal_rules.py --version v1 --overlap-report <check report json>
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
INPUT = ROOT / 'data/law-ko-source/docs.jsonl'
OUT = ROOT / 'data/internal-rules'
# v1 title sample (30 random docs): ~12 off-topic (요양비 보험급여, 의료급여, 보훈급여금, 하천보수원, 사회복무요원·공익근무
# 복무, 병역법, 수입금지식물, 물질안전보건자료 '전보'). v2 drops bare 급여/보수/복무/전보 and excludes those families.
TOPICS = {
    '인사': r'인사관리|인사규정|인사운영|임용|승진|전보인사|채용|정원',
    '복무': r'공무원\s*복무|직원\s*복무|복무규정|복무관리 규정|당직|겸직|근무시간|휴가|유연근무',
    '보수·수당': r'공무원\s*보수|보수규정|보수\s*지급|보수업무|수당|봉급|성과연봉|성과상여',
    '여비': r'여비|출장',
    '회계·지출': r'회계|출납|결산|재무관|지출|국고금|세입세출',
    '예산·기금': r'예산|기금운용|기금관리',
    '계약·구매': r'계약|입찰|조달|구매|검수|낙찰',
    '감사·징계': r'자체감사|감사규정|감사업무|징계|공직기강|비위',
    '결재·문서': r'위임전결|전결|사무관리|문서관리|기록물|행정업무|서무',
    '물품·자산': r'물품관리|물품취급|재물조사|자산관리|재산관리|공유재산|국유재산',
}
EXCLUDE = (r'수입위생|반출입물품|수입식품|수입검사|수입금지|검역|사회복무|공익근무|병역|보험급여|의료급여|보훈급여|요양|'
           r'하천보수|안전보건|생계급여|교육급여|주거급여')
LAW_ID = re.compile(r'law-go-kr:(?:admrul|expc):\d+')
SPLITS = (('test', 50), ('valid', 100))


def split_of(group):
    bucket = int(hashlib.sha256(f'split:{group}'.encode()).hexdigest()[:8], 16) % 10000
    return next((name for name, upper in SPLITS if bucket < upper), 'train')


def topics_of(title, kind='admrul'):
    """Title-match topics. Agency names that contain a topic word (기획예산처, 조달청, 우정사업조달센터) are blanked first.
    For legal interpretations, 계약·구매 needs a public-procurement cue: interpretations about private leases or sale
    contracts (v2 sample: 임대차계약, 주택공급계약, 공매 매각) are not internal work."""
    if re.search(EXCLUDE, title):
        return []
    title = re.sub(r'기획예산처|조달청|우정사업조달센터', '', title)
    found = [k for k, p in TOPICS.items() if re.search(p, title)]
    if kind == 'expc' and '계약·구매' in found and not re.search(r'당사자로 하는 계약|계약담당|조달|입찰|계약예규', title):
        found.remove('계약·구매')
    return found


def overlap_holds(path):
    """{corpus id: reason} from a check_reserved_overlap.py report: full matches and partial-review rows."""
    if not path:
        return {}
    rep = json.loads(Path(path).read_text())
    held = {}
    for m in rep.get('matches') or []:
        held[m['corpus_id']] = 'reserved_overlap:' + '+'.join(sorted({t for v in m['families'].values() for t in v}))
    for p in rep.get('partial_review') or []:
        held.setdefault(p['corpus_id'], 'reserved_overlap_partial')
    return held


RAW = ROOT / 'data/law-ko-source/raw'
RAW_TAGS = {'admrul': ('발령일자', '시행일자', '소관부처명', '행정규칙종류', '제개정구분명', '현행여부', '발령번호'),
            'expc': ('해석일자', '등록일시', '질의기관명', '해석기관명', '안건번호')}


def norm(value):
    digits = re.sub(r'\D', '', value or '')
    return f'{digits[:4]}-{digits[4:6]}-{digits[6:8]}' if len(digits) >= 8 else None


def raw_fields(kind, item_id):
    """Header fields of the stored body XML (data/law-ko-source/raw/<target>/<id>.xml); {} if not stored."""
    path = RAW / kind / f'{item_id}.xml'
    if not path.exists():
        return {}
    head = path.read_text(encoding='utf-8', errors='replace')[:6000]
    out = {}
    for tag in RAW_TAGS[kind]:
        m = re.search(fr'<{tag}>(?:<!\[CDATA\[)?([^<\]]*)', head)
        if m and m.group(1).strip():
            out[tag] = m.group(1).strip()
    return out


def date_type_of(kind, date, raw):
    """Which raw field the row date is: 발령일자 / 해석일자, or unknown when it matches none."""
    if not date:
        return 'unknown'
    field = '발령일자' if kind == 'admrul' else '해석일자'
    return field if norm(raw.get(field)) == date else 'unknown'


def main():
    from build_press_clean import EVAL_FILES, dedup_key, files_of
    from trid.korean_quality import PUBLIC_DOCS, assess_batch, redact, redacted_in_url
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', required=True)
    p.add_argument('--overlap-report', help='check_reserved_overlap.py check report over this candidate set')
    args = p.parse_args()
    out = OUT / args.version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    eval_ids, eval_keys, eval_used = set(), set(), []
    for name in EVAL_FILES:
        for f in files_of(name):
            eval_used.append(str(f.relative_to(ROOT)))
            for line in f.read_text().split('\n'):
                eval_ids.update(LAW_ID.findall(line))
                if line.strip():
                    row = json.loads(line)
                    eval_keys.update(dedup_key(v) for v in row.values() if isinstance(v, str) and len(v) >= 200)
    held_overlap = overlap_holds(args.overlap_report)
    built_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    picked = []
    with open(INPUT) as f:
        for line in f:
            if not re.match(r'\{"id": "law-go-kr:(admrul|expc):', line):
                continue
            d = json.loads(line)
            t = topics_of(d['title'], d['id'].split(':')[1])
            if t:
                picked.append((d, t))
    texts = []
    rows = []
    reasons = Counter()
    for d, t in picked:
        source_ok = hashlib.sha256(d['text'].encode()).hexdigest() == d.get('sha256')
        text, pii = redact(unicodedata.normalize('NFC', d['text']))
        kind = d['id'].split(':')[1]
        hold = []
        if d['id'] in eval_ids or dedup_key(d['text']) in eval_keys or dedup_key(text) in eval_keys:
            hold.append('dev_eval_source')
        if d['id'] in held_overlap:
            hold.append(held_overlap[d['id']])
        if redacted_in_url(text):
            hold.append('pii_redaction_inside_url')
        if not source_ok:
            hold.append('source_hash_mismatch')
        group = d.get('group') or d['id']
        raw = raw_fields(kind, d['id'].rsplit(':', 1)[1])
        date_type = date_type_of(kind, d.get('date'), raw)
        institution = raw.get('소관부처명') if kind == 'admrul' else raw.get('해석기관명')
        prov = {'doc_type': '행정규칙' if kind == 'admrul' else '법령해석례', 'site': 'www.law.go.kr',
                'institution': institution, 'published_date': d.get('date') if date_type in ('발령일자', '해석일자') else None,
                'effective_date': norm(raw.get('시행일자')) if kind == 'admrul' else None}
        rows.append({'id': d['id'], 'group': group, 'kind': 'administrative_rule' if kind == 'admrul' else 'legal_interpretation',
                     'title': d['title'], 'text': text, 'source': d['source'], 'license': d.get('license'),
                     'attribution': d.get('attribution') or '출처: 법제처 국가법령정보센터', 'date': d.get('date'),
                     'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': built_at,
                     'extractor': 'law.go.kr DRF XML (data/law-ko-source)', 'clean_version': args.version,
                     'flags': (['pii_redacted'] if pii else []) + [f'quarantine:{h}' for h in hold],
                     'split': split_of(group), 'provenance': {k: v for k, v in prov.items() if v},
                     'meta': {'topics': t, 'source_sha256': d.get('sha256'), 'pii_redactions': dict(pii),
                              'date_type': date_type, 'raw_fields': raw,
                              'currency': 'listed as current (nw=1) when collected 2026-10-27~29; not re-checked'
                                          if kind == 'admrul' else 'interpretation of the law in force at its date'}})
        texts.append(text)
        for h in hold or ['ok']:
            reasons[h.split(':')[0] if h.startswith('reserved_overlap:') else h] += 1
    for row, (ok, why) in zip(rows, assess_batch(texts, t=PUBLIC_DOCS)):
        row['meta']['quality'] = 'ok' if ok else why
        if not ok:
            row['flags'].append(f'quality:{why}')
    out.mkdir(parents=True)
    parts = {}

    def part_of(r):
        if any(f.startswith('quarantine:') for f in r['flags']):
            return 'quarantine'
        return 'docs' if r['meta']['quality'] == 'ok' else 'quality_held'
    for name in ('docs', 'quality_held', 'quarantine'):
        part = [r for r in rows if part_of(r) == name]
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        parts[name] = {'file': f'{name}.jsonl', 'sha256': hashlib.sha256(body).hexdigest(), 'rows': len(part),
                       'characters': sum(len(r['text']) for r in part),
                       'by_kind': dict(Counter(r['kind'] for r in part)),
                       'by_topic': dict(Counter(t for r in part for t in r['meta']['topics'])),
                       'date_type': dict(Counter(r['meta']['date_type'] for r in part))}
    manifest = {'version': args.version, 'built_at': built_at, 'input': str(INPUT.relative_to(ROOT)),
                'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(), 'code_sha256': CODE_SHA256,
                'korean_quality_sha256': hashlib.sha256((ROOT / 'trid/korean_quality.py').read_bytes()).hexdigest(),
                'overlap_report': args.overlap_report,
                'overlap_report_sha256': hashlib.sha256(Path(args.overlap_report).read_bytes()).hexdigest() if args.overlap_report else None,
                'eval_files': eval_used, 'reasons': dict(reasons), 'parts': parts,
                'quality': dict(Counter(r['meta']['quality'] for r in rows if part_of(r) != 'quarantine')),
                'splits': dict(Counter(r['split'] for r in rows if part_of(r) == 'docs')),
                'rights': '저작권법 제7조 (법령·훈령·예규·고시·법령해석); 출처 표시: 법제처 국가법령정보센터',
                'date_type_rule': ('row date compared with the raw body XML: admrul 발령일자 (collector maps the list '
                                   'field 발령일자), expc 해석일자 (collector maps the list field 회신일자); no match -> unknown'),
                'note': 'local candidate; not uploaded; docs = quality ok, quality_held = PUBLIC_DOCS failure kept with its '
                        'original text (not a training candidate), quarantine = held; topic is a title match, '
                        'not a verified classification'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('eval_files',)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

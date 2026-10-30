"""Statute appended tables and forms (별표·서식) for internal-work statutes, from law.go.kr (local only).

The stored statute body XML (data/law-ko-source/raw/law/<MST>.xml) lists each 별표/서식 with its number, kind, title and
download link (/LSW/flDownload.do?flSeq=N); its 별표내용 field holds only the header line, so the tables (봉급표, 수당
지급표, 여비 지급 구분표, 계약 서식 ...) exist only in the files. Public link, robots.txt Allow: /, no key, no login.
One request in flight, start >= previous end + --interval (log logs/requests-law.go.kr.jsonl). Raw files are written
once to data/law-byl/raw/<MST>/<flSeq>.<ext>. Rights: 법령의 별표·서식 are part of the statute (저작권법 제7조).

Actions:
  catalogue  data/law-byl/catalogue.jsonl for statutes whose title matches STATUTES
  fetch      download catalogue rows not yet fetched (resumable)
  extract    kordoc Markdown (tables kept) + PII + quality + overlap input -> data/law-byl/<version>/
  crosscheck numbers and unit words in each extracted row vs an independent HWP reader (pyhwp hwp5proc xml, run as
             `uv run --with pyhwp --with six --with lxml`); writes <version>/crosscheck.jsonl. Agreement says the
             values survived extraction, not that rows/columns are placed right.
"""
import argparse
import datetime
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from collect_admrul_attachments import extract_text, strip_table_markup  # noqa: E402  (stdlib-only module)
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
RAW_XML = ROOT / 'data/law-ko-source/raw/law'
OUT = ROOT / 'data/law-byl'
STATUTES = re.compile(r'공무원\s*(여비|보수|수당|복무|임용|인재개발|연금)|국가를 당사자로 하는 계약|지방자치단체를 당사자로 하는 계약|'
                      r'국고금 관리법|물품관리법|물품 관리법|국유재산법|공유재산|회계관계직원|국가재정법|지방재정법|지방회계법|예산|'
                      r'보조금 관리|정부기업예산|공공기관의 운영|행정업무의 운영')


def tag(s, name):
    m = re.search(fr'<{name}>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{name}>', s, re.S)
    return html.unescape(m.group(1).strip()) if m else None


def catalogue(args):
    OUT.mkdir(parents=True, exist_ok=True)
    c = Counter()
    with open(OUT / 'catalogue.jsonl', 'w') as out:
        for path in sorted(RAW_XML.glob('*.xml')):
            s = path.read_text(encoding='utf-8', errors='replace')
            title = tag(s, '법령명_한글') or ''
            if not STATUTES.search(title):
                continue
            head = {'law_mst': path.stem, 'law_title': title, '공포일자': tag(s, '공포일자'), '시행일자': tag(s, '시행일자'),
                    '소관부처명': tag(s, '소관부처명')}
            for unit in re.findall(r'<별표단위.*?</별표단위>', s, re.S):
                link = tag(unit, '별표서식파일링크')
                m = re.search(r'flSeq=(\d+)', link or '')
                if not m:
                    continue
                row = {**head, 'flSeq': m.group(1), 'url': f'https://www.law.go.kr{link}' if link.startswith('/') else link,
                       '별표번호': tag(unit, '별표번호'), '별표가지번호': tag(unit, '별표가지번호'), '별표구분': tag(unit, '별표구분'),
                       '별표제목': tag(unit, '별표제목'), 'header_line': (tag(unit, '별표내용') or '').strip()[:200]}
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                c[row['별표구분']] += 1
    print(json.dumps(dict(c), ensure_ascii=False))


def fetch(args):
    import collect_assembly_minutes as am
    rows = [json.loads(x) for x in (OUT / 'catalogue.jsonl').read_text().split('\n') if x]
    state_path = OUT / 'fetch-state.jsonl'
    seen = {json.loads(x)['flSeq'] for x in state_path.read_text().split('\n') if x} if state_path.exists() else set()
    client = am.Client(am.HostLimiter(args.interval, host='law.go.kr'), ROOT / 'logs/requests-law.go.kr.jsonl')
    metrics, done, counts = am.Metrics(), 0, Counter()
    with open(state_path, 'a') as state:
        for r in rows:
            if done >= args.limit:
                break
            if r['flSeq'] in seen:
                continue
            seen.add(r['flSeq'])
            resp = client.request('GET', r['url'], metrics)
            if resp.status_code in (401, 403, 429):
                raise SystemExit(f"stopped: HTTP {resp.status_code} for flSeq {r['flSeq']}")
            body = resp.content
            rec = {'flSeq': r['flSeq'], 'law_mst': r['law_mst'], 'http': resp.status_code, 'bytes': len(body),
                   'content_type': resp.headers.get('Content-Type'), 'content_disposition': resp.headers.get('Content-Disposition'),
                   'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
            kind = ('pdf' if body[:4] == b'%PDF' else 'hwpx' if body[:2] == b'PK' else 'hwp' if body[:8] == bytes.fromhex('d0cf11e0a1b11ae1')
                    else None)
            if resp.status_code == 200 and kind:
                digest = hashlib.sha256(body).hexdigest()
                dest = OUT / 'raw' / r['law_mst'] / f"{r['flSeq']}.{kind}"
                if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                    dest = dest.with_name(f"{r['flSeq']}.{digest[:16]}.{kind}")  # never replace stored raw
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                rec.update({'raw': str(dest.relative_to(ROOT)), 'sha256': digest, 'status': 'stored'})
            else:
                rec['status'] = 'not_a_document'
            state.write(json.dumps(rec, ensure_ascii=False) + '\n')
            state.flush()
            counts[rec['status']] += 1
            done += 1
    print(json.dumps({'requested': done, 'counts': dict(counts), 'metrics': metrics.snapshot()}, ensure_ascii=False))


def extract(args):
    from trid.korean_quality import PUBLIC_DOCS, assess, redact, redacted_in_url, script_ratios
    out = OUT / args.version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    cat = {}
    for x in (OUT / 'catalogue.jsonl').read_text().split('\n'):
        if x:
            r = json.loads(x)
            cat[r['flSeq']] = r
    rows = []
    plain = strip_table_markup
    for x in (OUT / 'fetch-state.jsonl').read_text().split('\n'):
        if not x:
            continue
        st = json.loads(x)
        if st.get('status') != 'stored':
            continue
        meta = cat[st['flSeq']]
        text, how = extract_text(ROOT / st['raw'])
        text = unicodedata.normalize('NFC', text)
        hangul = script_ratios(plain(text))[0] if text.strip() else 0.0
        text, pii = redact(text)
        ok, why = assess(plain(text), t=PUBLIC_DOCS) if text.strip() else (False, 'empty')
        held = []
        if not plain(text).strip() or hangul < 0.3:
            held.append('extraction_failed')
        if (meta['별표제목'] or '').startswith('삭제'):
            held.append('deleted_appendix')  # 별표 slot kept after deletion: header line only, no content
        if redacted_in_url(text):
            held.append('pii_redaction_inside_url')
        norm = lambda d: f'{d[:4]}-{d[4:6]}-{d[6:8]}' if d and len(d) >= 8 else None
        rows.append({
            'id': f"law-go-kr:law-byl:{st['flSeq']}", 'group': f"law:{meta['law_mst']}", 'kind': 'statute_appendix',
            'title': f"{meta['law_title']} [{meta['별표구분']} {meta['별표번호']}] {meta['별표제목']}", 'text': text,
            'source': meta['url'], 'license': 'KR-Copyright-Act-Art7 (law.go.kr)', 'attribution': '출처: 법제처 국가법령정보센터',
            'date': norm(meta['공포일자']), 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': st['fetched_at'],
            'extractor': how, 'clean_version': args.version,
            'flags': (['pii_redacted'] if pii else []) + [f'hold:{h}' for h in held] + ([] if ok else [f'quality:{why}']),
            'provenance': {'institution': meta.get('소관부처명'), 'doc_type': f"법령 {meta['별표구분']}", 'published_date': norm(meta['공포일자']),
                           'effective_date': norm(meta['시행일자']), 'site': 'www.law.go.kr'},
            'meta': {'law_mst': meta['law_mst'], 'law_title': meta['law_title'], 'appendix': {k: meta[k] for k in
                     ('별표번호', '별표가지번호', '별표구분', '별표제목', 'header_line')}, 'date_type': '공포일자 (statute header)',
                     'raw': st['raw'], 'raw_sha256': st['sha256'], 'quality': 'ok' if ok else why,
                     'table_rows': sum(1 for l in text.split('\n') if l.startswith('| ')) + text.count('<tr'),
                     'table_check': 'unverified', 'hangul_ratio_before_redaction': round(hangul, 3), 'pii_redactions': dict(pii)}})
    out.mkdir(parents=True)
    parts = {}
    for name, sel in (('extracted', [r for r in rows if not any(f.startswith('hold:') for f in r['flags'])]),
                      ('extraction_held', [r for r in rows if any(f.startswith('hold:') for f in r['flags'])])):
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        parts[name] = {'rows': len(sel), 'characters': sum(len(r['text']) for r in sel), 'sha256': hashlib.sha256(body).hexdigest(),
                       'table_rows': sum(r['meta']['table_rows'] for r in sel), 'kinds': dict(Counter(r['meta']['appendix']['별표구분'] for r in sel)),
                       'quality': dict(Counter(r['meta']['quality'] for r in sel))}
    with open(out / 'overlap-input.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps({'id': r['id'], 'group': r['group'], 'source': r['source'], 'text': r['text']}, ensure_ascii=False) + '\n')
    manifest = {'version': args.version, 'code_sha256': CODE_SHA256, 'parts': parts,
                'note': 'tables unverified (cell-level comparison not done); forms (서식) are blank templates; '
                        'not approved for training; run check_reserved_overlap.py on overlap-input.jsonl'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
UNITS = ('천원', '백만원', '원', '명', '%', '개월', '일', '시간', 'km', '호봉')


def value_profile(text):
    nums = Counter(n.replace(',', '') for n in NUMBER.findall(text))
    return nums, Counter({u: text.count(u) for u in UNITS if u in text})


def crosscheck_one(path):
    import subprocess
    res = subprocess.run(['uv', 'run', '-q', '--no-project', '--python', '3.12', '--with', 'pyhwp', '--with', 'six', '--with', 'lxml',
                          'hwp5proc', 'xml', str(path)], capture_output=True, timeout=300)
    if res.returncode:
        return None
    xml = res.stdout.decode('utf-8', 'replace')
    # a paragraph is stored as several Text runs (split at formatting changes); join runs, separate paragraphs
    paras = re.split(r'<Paragraph[ >]', xml)
    return html.unescape('\n'.join(''.join(re.findall(r'<Text[^>]*>([^<]*)</Text>', p)) for p in paras))


def crosscheck(args):
    from concurrent.futures import ThreadPoolExecutor
    src = OUT / args.version
    rows = [json.loads(x) for x in (src / 'extracted.jsonl').read_text().split('\n') if x]
    rows = [r for r in rows if r['meta']['raw'].endswith('.hwp')]
    plain = strip_table_markup
    with ThreadPoolExecutor(args.workers) as pool:
        refs = list(pool.map(lambda r: crosscheck_one(ROOT / r['meta']['raw']), rows))
    c = Counter()
    with open(src / 'crosscheck.jsonl', 'w') as out:
        for r, ref in zip(rows, refs):
            if ref is None:
                rec = {'id': r['id'], 'result': 'reader_failed'}
            else:
                (a_n, a_u), (b_n, b_u) = value_profile(plain(r['text'])), value_profile(ref)
                miss, extra = b_n - a_n, a_n - b_n
                rec = {'id': r['id'], 'result': 'agree' if not miss and not extra and a_u == b_u else 'differ',
                       'numbers': sum(b_n.values()), 'missing_in_extract': dict(miss), 'extra_in_extract': dict(extra),
                       'units_extract': dict(a_u), 'units_reader': dict(b_u)}
            c[rec['result']] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps({'rows': len(rows), **c, 'code_sha256': CODE_SHA256}, ensure_ascii=False))



def status(args):
    """Per-row status from extract + overlap report + crosscheck -> <version>/status.jsonl. Nothing here is a training
    approval: values agreeing with a second reader does not verify row/column placement."""
    src = OUT / args.version
    rep = json.loads((src / 'overlap-148.json').read_text())
    overlap = {m['corpus_id'] for m in rep.get('matches') or []} | {x['corpus_id'] for x in rep.get('partial_review') or []}
    check = {}
    for x in (src / 'crosscheck.jsonl').read_text().split('\n'):
        if x:
            r = json.loads(x)
            check[r['id']] = r['result']
    c = Counter()
    with open(src / 'status.jsonl', 'w') as out:
        for part in ('extracted', 'extraction_held'):
            for x in (src / f'{part}.jsonl').read_text().split('\n'):
                if not x:
                    continue
                r = json.loads(x)
                kind = r['meta']['appendix']['별표구분']
                holds = [f[5:] for f in r['flags'] if f.startswith('hold:')]
                if r['id'] in overlap:
                    holds.append('reserved_overlap')
                if part == 'extracted' and check.get(r['id']) != 'agree':
                    holds.append(f"values_{check.get(r['id'], 'not_checked')}")
                st = 'held' if holds else ('candidate_form' if kind == '서식' else 'candidate_table')
                out.write(json.dumps({'id': r['id'], 'status': st, 'holds': holds, 'kind': kind, 'values_check': check.get(r['id']),
                                      'layout_check': 'unverified', 'characters': len(r['text'])}, ensure_ascii=False) + '\n')
                c[st] += 1
                c.update(f'hold:{h}' for h in holds)
    print(json.dumps(dict(c), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['catalogue', 'fetch', 'extract', 'crosscheck', 'status'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=1000)
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--version', default='v1')
    args = p.parse_args()
    {'catalogue': catalogue, 'fetch': fetch, 'extract': extract, 'crosscheck': crosscheck, 'status': status}[args.action](args)


if __name__ == '__main__':
    main()

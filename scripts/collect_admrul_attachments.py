"""Administrative-rule attachments from law.go.kr (local only): the 첨부파일 (전문, 별표, 서식, 업무지침 PDF/HWP/HWPX)
that the stored body XML of each 행정규칙 lists (data/law-ko-source/raw/admrul/<id>.xml, 첨부파일명/첨부파일링크).

Path: the public download link the law.go.kr page itself offers (http://law.go.kr/flDownload.do?flSeq=N); robots.txt
of www.law.go.kr is "Allow: /". No key, no login. One request in flight; each start >= previous end + --interval
(HostLimiter of collect_assembly_minutes, log logs/requests-law.go.kr.jsonl).

Rights: 행정규칙 본문·별표·서식 are not protected works (저작권법 제7조). Files that are not the rule itself - revision
rationale (제개정이유서), old/new comparison tables (신구조문대비표), drafts (개정안) - are listed as `rights_review`,
not downloaded by default.

Actions:
  catalogue  write data/law-attachments/catalogue.jsonl (one row per attachment, rule header fields, topic, class)
  fetch      download up to --limit files of class `rule_text` for internal-work topic rules (raw written once to
             data/law-attachments/raw/<rule id>/<flSeq>.<ext>; an existing file with other bytes is never replaced)
  extract    text + quality + PII + duplicate-with-rule-body + reserved-overlap input ->
             data/law-attachments/<version>/{candidates,quality_held,rights_held}.jsonl + manifest.json
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
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
NOT_RULE = re.compile(r'\(안\)|[가-힣]안\(|[가-힣]안\.|고시안|훈령안|예규안|개정안|제정안|이유서|신구조문|대비표|검토|의견|보도자료|'
                      r'설명자료|안내문|붙임\s*\d|참고')
TABLE_FILE = re.compile(r'별표|별지|서식|기준표|정액표')
RAW_XML = ROOT / 'data/law-ko-source/raw/admrul'
OUT = ROOT / 'data/law-attachments'
HEADER_TAGS = ('행정규칙명', '행정규칙종류', '발령일자', '시행일자', '소관부처명', '현행여부', '발령번호', '제개정구분명')


def field(s, tag):
    m = re.search(fr'<{tag}>(?:<!\[CDATA\[)?([^<\]]*)', s)
    return m.group(1).strip() if m else None


def file_class(name):
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if ext not in ('pdf', 'hwp', 'hwpx'):
        return 'other_format'
    return 'rights_review' if NOT_RULE.search(name) else 'rule_text'


def catalogue(args):
    from build_internal_rules import topics_of
    kept = set()
    with open(ROOT / 'data/law-ko-source/docs.jsonl') as f:
        prefix = '{"id": "law-go-kr:admrul:'
        for line in f:
            if line.startswith(prefix):
                kept.add(line[len(prefix):len(prefix) + 30].split('"')[0])
    OUT.mkdir(parents=True, exist_ok=True)
    c = Counter()
    with open(OUT / 'catalogue.jsonl', 'w') as out:
        for path in sorted(RAW_XML.glob('*.xml')):
            s = path.read_text(encoding='utf-8', errors='replace')
            names = re.findall(r'<첨부파일명>(?:<!\[CDATA\[)?([^<\]]*)', s)
            links = re.findall(r'<첨부파일링크>(?:<!\[CDATA\[)?([^<\]]*)', s)
            header = {t: field(s, t) for t in HEADER_TAGS}
            topics = topics_of(header['행정규칙명'] or '')
            for n, u in zip(names, links):
                n, u = n.strip(), u.strip()
                m = re.search(r'flSeq=(\d+)', u)
                row = {'rule_id': path.stem, 'flSeq': m.group(1) if m else None, 'name': n, 'url': u,
                       'class': file_class(n), 'topics': topics, 'rule_body_in_collection': path.stem in kept, **header}
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                c[(bool(topics), row['class'])] += 1
    print(json.dumps({f'topic={k[0]}:{k[1]}': v for k, v in sorted(c.items())}, ensure_ascii=False))


def fetch(args):
    import collect_assembly_minutes as am
    rows = [json.loads(x) for x in (OUT / 'catalogue.jsonl').read_text().split('\n') if x]
    # scope internal: internal-work topic rules; wide: also attachment-only rules of any topic whose kind is a
    # manual/guideline/instruction (예규, 훈령, 지침, 매뉴얼, 규정, 기타) - not 고시/공고 (mostly statistics approvals,
    # heritage zoning and project notices in the catalogue sample)
    wide_kinds = {'예규', '훈령', '지침', '매뉴얼', '규정', '기타'}
    pick = [r for r in rows if r['class'] == 'rule_text' and r['flSeq'] and
            (r['topics'] or (args.scope == 'wide' and not r['rule_body_in_collection'] and r['행정규칙종류'] in wide_kinds))]
    # What the collection does not already have: (1) rules whose body text is not in data/law-ko-source (attachment
    # only), one file per rule, PDF > HWPX > HWP; (2) 별표/별지/서식 files, HWPX first (cell structure kept), since the
    # rule body text does not carry their tables. Plain full-text copies of rules already collected are skipped.
    import random
    order = {'pdf': 0, 'hwpx': 1, 'hwp': 2}
    first = {}
    for r in pick:
        if not r['rule_body_in_collection']:
            rank = order[r['name'].rsplit('.', 1)[-1].lower()]
            if r['rule_id'] not in first or rank < first[r['rule_id']][0]:
                first[r['rule_id']] = (rank, r)
    tables = sorted((r for r in pick if r['rule_body_in_collection'] and TABLE_FILE.search(r['name'])),
                    key=lambda r: {'hwpx': 0, 'hwp': 1, 'pdf': 2}[r['name'].rsplit('.', 1)[-1].lower()])
    seen_rule = set()
    table_pick = [r for r in tables if not (r['rule_id'] in seen_rule or seen_rule.add(r['rule_id']))]
    attach_only = [r for _, r in first.values()]
    random.Random(args.seed).shuffle(attach_only)
    random.Random(args.seed).shuffle(table_pick)
    todo = [x for pair in zip(attach_only, table_pick) for x in pair] + attach_only[len(table_pick):] + table_pick[len(attach_only):]
    print(f'{len(attach_only)} attachment-only rules, {len(table_pick)} rules with table files', flush=True)
    client = am.Client(am.HostLimiter(args.interval, host='law.go.kr'), ROOT / 'logs/requests-law.go.kr.jsonl')
    metrics, done, counts = am.Metrics(), 0, Counter()
    state_path = OUT / 'fetch-state.jsonl'
    seen = {json.loads(x)['flSeq'] for x in state_path.read_text().split('\n') if x} if state_path.exists() else set()
    with open(state_path, 'a') as state:
        for r in todo:
            if done >= args.limit:
                break
            if r['flSeq'] in seen:
                continue
            ext = r['name'].rsplit('.', 1)[-1].lower()
            dest = OUT / 'raw' / r['rule_id'] / f"{r['flSeq']}.{ext}"
            resp = client.request('GET', f"https://www.law.go.kr/flDownload.do?flSeq={r['flSeq']}", metrics)
            rec = {'flSeq': r['flSeq'], 'rule_id': r['rule_id'], 'http': resp.status_code,
                   'content_type': resp.headers.get('Content-Type'), 'bytes': len(resp.content),
                   'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
            if resp.status_code in (401, 403, 429):
                raise SystemExit(f'stopped: HTTP {resp.status_code} for flSeq {r["flSeq"]}')
            body = resp.content
            if resp.status_code == 200 and body and not body.lstrip()[:15].lower().startswith((b'<!doctype', b'<html')):
                digest = hashlib.sha256(body).hexdigest()
                if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                    dest = dest.with_name(f"{r['flSeq']}.{digest[:16]}.{ext}")  # never replace stored raw
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                rec.update({'raw': str(dest.relative_to(ROOT)), 'sha256': digest, 'status': 'stored'})
            else:
                rec['status'] = 'not_a_file'
            state.write(json.dumps(rec, ensure_ascii=False) + '\n')
            state.flush()
            counts[rec['status']] += 1
            done += 1
    print(json.dumps({'requested': done, 'counts': dict(counts), 'metrics': metrics.snapshot()}, ensure_ascii=False))


KORDOC = ['npx', '--yes', '--package', 'kordoc@4.24.23', '--package', 'pdfjs-dist', 'kordoc']

# kordoc writes merged-cell tables as HTML (<table><tr><td rowspan=..>); the markup stays in the text (it is the table
# structure) but is not language. Only that markup is stripped: statute text uses angle brackets itself
# ("<개정 2026. 9. 29.>", "삭제 <2007.11.22>"), which a generic <[^>]+> pattern removed (B-016).
TABLE_MARKUP = re.compile(r'</?(?:table|thead|tbody|tr|td|th|u|b|i|br|p|span|sup|sub)\b[^>]*>')


def strip_table_markup(text):
    return TABLE_MARKUP.sub(' ', text)



def extract_text(path):
    """PDF/HWP/HWPX -> Markdown with kordoc 4.24.23 (tables as Markdown tables, PDF running headers/footers removed);
    image placeholders dropped. Chosen over pdftotext/hwp5txt after the pilot: hwp5txt returned 12-117 characters for
    별표 HWP files whose tables kordoc reads (B-014).
    kordoc refuses an HWPX zip with more than 500 entries (its ZIP_BOMB guard); a manual with ~1,200 embedded images
    hits that. Such a file is read once more from a copy without BinData/ (the images, which are dropped anyway); the
    text and tables come from Contents/ unchanged, and the extractor label says the images were removed (B-025)."""
    import subprocess
    import tempfile
    import zipfile
    raw = path.read_bytes()
    kind = ('pdf' if raw[:4] == b'%PDF' else 'hwpx' if raw[:2] == b'PK' else 'hwp' if raw[:8] == bytes.fromhex('d0cf11e0a1b11ae1')
            else path.suffix.lstrip('.'))
    if kind not in ('pdf', 'hwp', 'hwpx'):
        return '', f'unsupported:{kind}'
    label = f'kordoc-4.24.23:{kind}'
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / f'f.{kind}', Path(d) / 'out.md'
        src.write_bytes(raw)
        r = subprocess.run(KORDOC + [str(src), '-o', str(out)], capture_output=True, timeout=900, cwd=d)
        if (r.returncode != 0 or not out.exists()) and kind == 'hwpx' and b'ZIP_BOMB' in r.stdout + r.stderr:
            with zipfile.ZipFile(src) as z, zipfile.ZipFile(Path(d) / 'noimg.hwpx', 'w') as w:
                for info in z.infolist():
                    if not info.filename.startswith('BinData/'):
                        w.writestr(info, z.read(info.filename))
            r = subprocess.run(KORDOC + [str(Path(d) / 'noimg.hwpx'), '-o', str(out)], capture_output=True, timeout=900, cwd=d)
            label += '(BinData images removed: zip entry limit)'
        if r.returncode != 0 or not out.exists():
            return '', f'kordoc_failed:{r.returncode}'
        text = out.read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'^!\[image\]\([^)]*\)\s*$', '', text, flags=re.M)
    text = re.sub(r'\[이미지: image\d+\][ \t]*', '', text)  # placeholder of an image that is not in the copy
    return re.sub(r'\n{3,}', '\n\n', text).strip(), label


def extract(args):
    from build_press_clean import dedup_key
    from trid.korean_quality import PUBLIC_DOCS, assess, redact, redacted_in_url, script_ratios
    out = OUT / args.version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    cat = {}
    for x in (OUT / 'catalogue.jsonl').read_text().split('\n'):
        if x:
            r = json.loads(x)
            cat[r['flSeq']] = r
    body_keys = {}
    with open(ROOT / 'data/law-ko-source/docs.jsonl') as f:
        for line in f:
            if line.startswith('{"id": "law-go-kr:admrul:'):
                d = json.loads(line)
                body_keys[dedup_key(d['text'])] = d['id']
    rows, c = [], Counter()
    for x in (OUT / 'fetch-state.jsonl').read_text().split('\n'):
        if not x:
            continue
        st = json.loads(x)
        if st.get('status') != 'stored':
            continue
        meta = cat[st['flSeq']]
        text, how = extract_text(ROOT / st['raw'])
        text = unicodedata.normalize('NFC', text)
        plain = strip_table_markup  # script ratios and quality are measured without table markup
        hangul = script_ratios(plain(text))[0] if text.strip() else 0.0
        text, pii = redact(text)
        ok, why = assess(plain(text), t=PUBLIC_DOCS) if text.strip() else (False, 'empty')
        held = []
        if len(plain(text).strip()) < 300 or hangul < 0.3:
            held.append('extraction_failed')
        if redacted_in_url(text):
            held.append('pii_redaction_inside_url')
        dup_of = body_keys.get(dedup_key(text))
        row_id = f"law-go-kr:admrul-file:{st['flSeq']}"
        rows.append({'id': row_id, 'group': f"admrul:{meta['rule_id']}", 'kind': 'administrative_rule_attachment',
                     'title': f"{meta['행정규칙명']} - {meta['name']}", 'text': text,
                     'source': f"https://www.law.go.kr/flDownload.do?flSeq={st['flSeq']}",
                     'license': 'KR-Copyright-Act-Art7 (law.go.kr)', 'attribution': '출처: 법제처 국가법령정보센터',
                     'date': f"{meta['발령일자'][:4]}-{meta['발령일자'][4:6]}-{meta['발령일자'][6:8]}" if meta.get('발령일자') else None,
                     'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': st['fetched_at'],
                     'extractor': how, 'clean_version': args.version,
                     'flags': (['pii_redacted'] if pii else []) + [f'hold:{h}' for h in held] + ([] if ok else [f'quality:{why}'])
                              + ([f'duplicate_of_rule_body:{dup_of}'] if dup_of else []),
                     'provenance': {'institution': meta.get('소관부처명'), 'doc_type': f"행정규칙 첨부({meta.get('행정규칙종류')})",
                                    'published_date': None, 'effective_date': None, 'site': 'www.law.go.kr'},
                     'meta': {'rule_id': meta['rule_id'], 'file_name': meta['name'], 'topics': meta['topics'],
                              'date_type': '발령일자 (rule header)', 'rule_header': {k: meta.get(k) for k in HEADER_TAGS},
                              'raw': st['raw'], 'raw_sha256': st['sha256'],
                              'table_rows': sum(1 for l in text.split('\n') if l.startswith('| ')) + text.count('<tr'),
                              'quality': 'ok' if ok else why, 'hangul_ratio_before_redaction': round(hangul, 3),
                              'pii_redactions': dict(pii), 'rule_body_in_collection': meta['rule_body_in_collection']}})
    out.mkdir(parents=True)

    def part(r):
        if any(f.startswith('hold:') for f in r['flags']):
            return 'extraction_held'
        if any(f.startswith('duplicate_of_rule_body:') for f in r['flags']):
            return 'duplicates'
        return 'candidates' if r['meta']['quality'] == 'ok' or r['meta']['table_rows'] else 'quality_held'
    parts = {}
    for name in ('candidates', 'quality_held', 'extraction_held', 'duplicates'):
        sel = [r for r in rows if part(r) == name]
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        parts[name] = {'rows': len(sel), 'characters': sum(len(r['text']) for r in sel), 'sha256': hashlib.sha256(body).hexdigest(),
                       'table_rows': sum(r['meta']['table_rows'] for r in sel)}
    with open(out / 'overlap-input.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps({'id': r['id'], 'group': r['group'], 'source': r['source'], 'text': r['text']}, ensure_ascii=False) + '\n')
    manifest = {'version': args.version, 'code_sha256': CODE_SHA256, 'parts': parts,
                'quality': dict(Counter(r['meta']['quality'] for r in rows)),
                'extractors': dict(Counter(r['extractor'] for r in rows)),
                'rule': ('candidates: extraction ok and not a copy of a collected rule body, and (PUBLIC_DOCS quality ok '
                         'or the file carries tables); reserved-overlap check must be run on overlap-input.jsonl before use'),
                'rights': '저작권법 제7조 (행정규칙 본문·별표·서식). Files named as drafts/rationale were not downloaded.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['catalogue', 'fetch', 'extract'])
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--seed', default='20261030')
    p.add_argument('--version', default='v1')
    p.add_argument('--scope', choices=['internal', 'wide'], default='internal')
    args = p.parse_args()
    {'catalogue': catalogue, 'fetch': fetch, 'extract': extract}[args.action](args)


if __name__ == '__main__':
    main()

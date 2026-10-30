"""PRISM (정책연구관리시스템, www.prism.go.kr) policy-research report pilot, through the public web UI's own requests.

Approved path (B-010): the requests the public site makes for a visitor without login, observed in a browser:
  list    POST https://api.prism.go.kr/prism-be-asmt/v1/entire/list-organtheme
          {"asmtNm":"","startDate":"YYYY.MM.DD","endDate":"YYYY.MM.DD","currentPage":n,"pageSize":10}
  detail  POST https://api.prism.go.kr/prism-be-asmt/v1/entire/info {"mainYn":"N","asmtId":...}
  file    POST https://api.prism.go.kr/prism-be-asmt/v1/progress/download-file
          {"asmtId","fileTypeCd","fileSn","fileWkky","pdfTrsfYn"} exactly as listed in the detail's asmtFileList
No login, no key-gated portal API (data.go.kr 15080254), nothing the UI does not request. A 401/403, captcha page or
non-JSON detail stops the run (no retry around it).

Rights: a report is taken only when the detail says it is public (infoRlsCd B0030001) and its 공공누리 record is
제1유형 (D0590002) or 자유이용 (D0590006) with commercial use and modification allowed (koglCmrcUtztnYn = koglWrgsChgYn
= Y). koglAiYn (공공누리 AI 학습용 표시) is recorded as is. Everything else is counted, not downloaded.

Actions:
  fetch  seeded list pages -> details -> report files (D0150004 연구보고서) until --max reports. Raw is written once
         (put_new / put_raw_keep): corpus/raw/prism/<asmtId>/{info.json,<fileTypeCd>-<fileSn>-<fileWkky>.<ext>}.
         Per-report state: corpus/state/prism/pilot-v1/asmt/<asmtId>.json (resumable).
  build  raw -> corpus/clean/prism/<version>/{docs,quarantine}-00000.jsonl.gz + <version>.manifest.json + .done
         (text extraction, NFC, PII redaction, quality assessment, exact duplicates, --hold quarantine).
Requests: one in flight, each start >= previous end + --interval (default 2 s), log logs/requests-prism.go.kr.jsonl.
"""
import argparse
import datetime
import gzip
import hashlib
import html
import io
import json
import random
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from collect_assembly_minutes import (BUCKET, Client, HostLimiter, Metrics, list_keys, put_new,  # noqa: E402
                                      put_raw_keep, put_verified, read_json_s3, s3_client)

SOURCE = 'prism'
API = 'https://api.prism.go.kr/prism-be-asmt/v1'
PAGE = 'https://www.prism.go.kr/homepage/asmt/{}'
STATE = f'corpus/state/{SOURCE}/pilot-v1'
# The server omits its intermediate certificate; verify with certifi + the issuer named in the leaf's AIA
# (http://crt.sectigo.com/SectigoRSADomainValidationSecureServerCA.crt), pinned by fingerprint.
INTERMEDIATE = ROOT / 'scripts/certs/sectigo-rsa-dv-intermediate.pem'
INTERMEDIATE_SHA256 = '7fa4ff68ec04a99d7528d5085f94907f4d1dd1c5381bacdc832ed5c960214676'
KOGL = {'D0590002': 'KOGL-Type1', 'D0590006': 'KOGL-Free'}
REPORT_FILE = 'D0150004'  # 연구보고서 (D0150018 = 활용결과 보고서, not taken)
PUBLIC = 'B0030001'
SPLITS = (('test', 50), ('valid', 100))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded


def split_of(group):
    bucket = int(hashlib.sha256(f'split:{group}'.encode()).hexdigest()[:8], 16) % 10000
    return next((name for name, upper in SPLITS if bucket < upper), 'train')


def make_client(interval):
    import base64
    import certifi
    pem = INTERMEDIATE.read_text()
    der = base64.b64decode(''.join(pem.strip().split('\n')[1:-1]))
    if hashlib.sha256(der).hexdigest() != INTERMEDIATE_SHA256:
        raise SystemExit('intermediate certificate fingerprint mismatch')
    bundle = Path(tempfile.gettempdir()) / 'trid-prism-ca.pem'
    bundle.write_text(Path(certifi.where()).read_text() + pem)
    client = Client(HostLimiter(interval, host='prism.go.kr'), ROOT / 'logs/requests-prism.go.kr.jsonl')
    client.session.verify = str(bundle)
    client.session.headers.update({'Accept': 'application/json, text/plain, */*', 'Origin': 'https://www.prism.go.kr',
                                   'Referer': 'https://www.prism.go.kr/'})
    return client


class Stop(RuntimeError):
    """The site refused or answered with something that is not the public UI's response: stop, do not work around."""


def post_json(client, path, body, metrics):
    r = client.request('POST', f'{API}/{path}', metrics, json=body)
    if r.status_code in (401, 403) or r.status_code >= 400:
        raise Stop(f'{path}: HTTP {r.status_code} {r.text[:200]!r}')
    if not r.headers.get('Content-Type', '').startswith('application/json'):
        raise Stop(f'{path}: not JSON ({r.headers.get("Content-Type")}) {r.text[:200]!r}')
    return r, r.json()


def eligibility(info):
    detail, kogl = info.get('asmtDetail') or {}, info.get('koglDetail') or {}
    if detail.get('infoRlsCd') != PUBLIC:
        return f"not_public:{detail.get('infoRlsCd')}"
    if kogl.get('koglTypeCd') not in KOGL:
        return f"kogl:{kogl.get('koglTypeCd') or 'none'}"
    if detail.get('koglCmrcUtztnYn') != 'Y' or detail.get('koglWrgsChgYn') != 'Y':
        return 'kogl_flags_restrict'
    if not any(f.get('fileTypeCd') == REPORT_FILE for f in info.get('asmtFileList') or []):
        return 'no_report_file'
    return 'ok'


def fetch(args):
    s3, metrics = s3_client(), Metrics()
    client = make_client(args.interval)
    done = {k.rsplit('/', 1)[1][:-5] for k in list_keys(s3, f'{STATE}/asmt/')}
    acquired = sum(1 for k in list_keys(s3, f'{STATE}/asmt/') if read_json_s3(s3, k, {}).get('status') == 'acquired')
    pages_key = f'{STATE}/pages.json'
    pages = read_json_s3(s3, pages_key, {'done': [], 'total': None})
    body = {'asmtNm': '', 'startDate': args.start, 'endDate': args.end, 'pageSize': 10}
    if pages['total'] is None:
        _, first = post_json(client, 'entire/list-organtheme', {**body, 'currentPage': 1}, metrics)
        pages['total'] = first['resultData']['organthemeRschListCnt']
    order = list(range(1, (pages['total'] + 9) // 10 + 1))
    random.Random(args.seed).shuffle(order)
    order = [p for p in order if p not in pages['done']][:args.max_pages - len(pages['done'])]
    counts = Counter()
    for page in order:
        if acquired >= args.max:
            break
        _, data = post_json(client, 'entire/list-organtheme', {**body, 'currentPage': page}, metrics)
        for item in data['resultData']['organthemeRschList']:
            if acquired >= args.max:
                break
            asmt = item['asmtId']
            if asmt in done:
                continue
            state = {'asmtId': asmt, 'list_page': page, 'list_item': item, 'page_url': PAGE.format(asmt),
                     'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
            if item.get('infoRlsCd') != '공개':
                state['status'] = f"skipped:list_not_public:{item.get('infoRlsCd')}"
            else:
                r, info = post_json(client, 'entire/info', {'mainYn': 'N', 'asmtId': asmt}, metrics)
                info = info.get('resultData')
                if not isinstance(info, dict):
                    raise Stop(f'entire/info {asmt}: {str(info)[:200]}')
                state['info_sha256'], state['info_key'] = put_raw_keep(s3, f'corpus/raw/{SOURCE}/{asmt}/info.json', r.content)
                why = eligibility(info)
                state['eligibility'] = why
                state['kogl'] = info.get('koglDetail')
                if why != 'ok':
                    state['status'] = f'skipped:{why}'
                else:
                    state['files'] = []
                    for f in [f for f in info['asmtFileList'] if f.get('fileTypeCd') == REPORT_FILE]:
                        req = {k: f[k] for k in ('asmtId', 'fileTypeCd', 'fileSn', 'fileWkky', 'pdfTrsfYn')}
                        d = client.request('POST', f'{API}/progress/download-file', metrics, json=req)
                        rec = {'request': req, 'file_name': f.get('fileNm'), 'http': d.status_code,
                               'content_type': d.headers.get('Content-Type'), 'bytes': len(d.content)}
                        if d.status_code in (401, 403):
                            raise Stop(f'download {asmt}: HTTP {d.status_code}')
                        if d.status_code == 200 and d.content and not d.headers.get('Content-Type', '').startswith(('application/json', 'text/html')):
                            ext = (f.get('fileNm') or '').rsplit('.', 1)[-1].lower() if '.' in (f.get('fileNm') or '') else 'bin'
                            if d.content[:4] == b'%PDF':
                                ext = 'pdf'
                            rec['raw_sha256'], rec['raw_key'] = put_raw_keep(
                                s3, f"corpus/raw/{SOURCE}/{asmt}/{f['fileTypeCd']}-{f['fileSn']}-{f['fileWkky']}.{ext}", d.content)
                        else:
                            rec['error'] = d.text[:200] if d.headers.get('Content-Type', '').startswith(('application/json', 'text/')) else 'empty_or_bad'
                        state['files'].append(rec)
                    ok = [x for x in state['files'] if x.get('raw_key')]
                    state['status'] = 'acquired' if ok else 'failed:download'
                    acquired += bool(ok)
            counts[state['status']] += 1
            s3.put_object(Bucket=BUCKET, Key=f'{STATE}/asmt/{asmt}.json', Body=json.dumps(state, ensure_ascii=False).encode())
            done.add(asmt)
        pages['done'].append(page)
        s3.put_object(Bucket=BUCKET, Key=pages_key, Body=json.dumps(pages).encode())
        print(json.dumps({'page': page, 'acquired': acquired, 'counts': dict(counts), 'metrics': metrics.snapshot()},
                         ensure_ascii=False), flush=True)


def extract(raw, name):
    """Dispatch on the bytes, not the name: the site can serve a PDF conversion or an HWP binary under any extension."""
    kind = ('pdf' if raw[:4] == b'%PDF' else 'hwpx' if raw[:2] == b'PK' else 'hwp' if raw[:8] == bytes.fromhex('d0cf11e0a1b11ae1')
            else name.rsplit('.', 1)[-1])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f'f.{kind}'
        path.write_bytes(raw)
        if kind == 'pdf':
            out = subprocess.run(['pdftotext', '-enc', 'UTF-8', str(path), '-'], capture_output=True, timeout=600)
            return out.stdout.decode('utf-8', 'replace'), 'pdftotext'
        if kind == 'hwp':
            out = subprocess.run(['hwp5txt', str(path)], capture_output=True, timeout=600)
            return out.stdout.decode('utf-8', 'replace'), 'hwp5txt'
        if kind == 'hwpx':
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                parts = sorted(n for n in z.namelist() if re.match(r'Contents/section\d+\.xml$', n))
                text = '\n'.join(html.unescape(re.sub(r'<[^>]+>', '', p)) for n in parts
                                 for p in re.findall(r'<hp:p\b.*?</hp:p>', z.read(n).decode('utf-8', 'replace'), re.S))
            return text, 'hwpx-xml'
    return '', f'unsupported:{kind}'


def normalize(text):
    text = unicodedata.normalize('NFC', text.replace('\f', '\n'))
    lines = [' '.join(line.split()) for line in text.split('\n')]
    out, blank = [], 0
    for line in lines:
        blank = blank + 1 if not line else 0
        if blank <= 1:
            out.append(line)
    return '\n'.join(out).strip()


def build(args):
    from build_press_clean import dedup_key, eval_exclusions
    from trid.korean_quality import PUBLIC_DOCS, assess, redact, redacted_in_url, script_ratios
    s3 = s3_client()
    held = json.loads((ROOT / args.hold).read_text()) if args.hold else {}
    eval_ids, eval_keys, eval_used = eval_exclusions()
    built_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    rows, seen, reasons = [], {}, Counter()
    for key in sorted(list_keys(s3, f'{STATE}/asmt/')):
        st = read_json_s3(s3, key, {})
        if st.get('status') != 'acquired':
            continue
        asmt = st['asmtId']
        info = json.loads(s3.get_object(Bucket=BUCKET, Key=st['info_key'])['Body'].read())['resultData']
        detail, kogl, report = info['asmtDetail'], info['koglDetail'], (info.get('reportList') or [{}])[0]
        texts, extractors, raws = [], [], []
        for f in [x for x in st['files'] if x.get('raw_key')]:
            raw = s3.get_object(Bucket=BUCKET, Key=f['raw_key'])['Body'].read()
            text, how = extract(raw, f['raw_key'])
            texts.append(normalize(text))
            extractors.append(how)
            raws.append({'key': f['raw_key'], 'sha256': f['raw_sha256'], 'file_name': f['file_name'],
                         'extracted_sha256': hashlib.sha256(text.encode()).hexdigest(), 'extracted_chars': len(text)})
        text = '\n\n'.join(t for t in texts if t)
        hangul = script_ratios(text)[0] if text else 0.0
        text, pii = redact(text)
        ok, quality = assess(text, t=PUBLIC_DOCS) if text else (False, 'empty')
        group = f'{SOURCE}:{asmt}'
        title = (detail.get('asmtNm') or '').strip()
        flags = (['pii_redacted'] if pii else []) + ([] if ok else [f'quality:{quality}'])
        hold = []
        if len(text) < 1000 or hangul < 0.3:
            hold.append('extraction_failed')
        if redacted_in_url(text):
            hold.append('pii_redaction_inside_url')
        if dedup_key(text) in eval_keys or asmt in eval_ids:
            hold.append('dev_eval_source')
        if group in held:
            hold.append(held[group].split(' ', 1)[0])
        k = dedup_key(text)
        if 'extraction_failed' not in hold:  # empty/garbled texts are failures, not copies of each other
            if k in seen:
                reasons['exact_duplicate'] += 1
                print(asmt, 'exact duplicate of', seen[k], flush=True)
                continue
            seen[k] = group
        flags += [f'quarantine:{h}' for h in hold]
        for h in hold or ['ok']:
            reasons[h] += 1
        url = PAGE.format(asmt)
        rows.append({'id': group, 'group': group, 'kind': 'policy_research_report', 'title': title, 'text': text,
                     'source': url, 'license': KOGL[kogl['koglTypeCd']],
                     'attribution': f"[{detail.get('instNm')}]이(가) 보유한 [{title}]은(는) 공공누리 {kogl.get('cdNm')} 조건에 "
                                    f"따라 이용할 수 있습니다. 출처: 정책연구관리시스템(PRISM), {url}",
                     'date': None,  # only the publication year is known (meta.publication_year)
                     'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': st['checked_at'],
                     'extractor': '+'.join(extractors), 'clean_version': args.version, 'flags': flags,
                     'split': split_of(group),
                     'provenance': {'institution': detail.get('instNm'), 'doc_type': '정책연구보고서',
                                    'domain': ' > '.join(x for x in (detail.get('hghrkFwkClsfSysNm'), detail.get('clsfSysNm')) if x) or None,
                                    'published_date': None, 'site': 'www.prism.go.kr'},
                     'meta': {'asmtId': asmt, 'publication_year': report.get('pblcnYr'), 'raw_files': raws,
                              'info_key': st['info_key'], 'info_sha256': st['info_sha256'],
                              'kogl': {k: kogl.get(k) for k in ('koglTypeCd', 'cdNm', 'koglAiYn')} | {
                                  k: detail.get(k) for k in ('koglRlsYn', 'koglCmrcUtztnYn', 'koglWrgsChgYn')},
                              'info_release': detail.get('infoRlsCd'), 'research_period': [detail.get('rschBgngYmd'), detail.get('rschEndYmd')],
                              'quality': quality, 'hangul_ratio_before_redaction': round(hangul, 3), 'pii_redactions': dict(pii)}})
        rows[-1]['provenance'] = {k: v for k, v in rows[-1]['provenance'].items() if v}
        print(asmt, len(text), quality, hold or 'ok', flush=True)
    shards = []
    for name, part in (('docs', [r for r in rows if not any(f.startswith('quarantine:') for f in r['flags'])]),
                       ('quarantine', [r for r in rows if any(f.startswith('quarantine:') for f in r['flags'])])):
        body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part).encode(), mtime=0)
        key = f'corpus/clean/{SOURCE}/{args.version}/{name}-00000.jsonl.gz'
        shards.append({'part': name, 'key': key, 'sha256': put_verified(s3, key, body), 'bytes': len(body), 'rows': len(part),
                       'characters': sum(len(r['text']) for r in part)})
    manifest = {'source': SOURCE, 'clean_version': args.version, 'built_at': built_at,
                'code_sha256': CODE_SHA256,
                'korean_quality_sha256': hashlib.sha256((ROOT / 'trid/korean_quality.py').read_bytes()).hexdigest(),
                'reasons': dict(reasons), 'hold_input': args.hold, 'eval_files': eval_used,
                'rows': shards[0]['rows'], 'quarantine_rows': shards[1]['rows'], 'shards': shards,
                'splits': dict(Counter(r['split'] for r in rows if not any(f.startswith('quarantine:') for f in r['flags']))),
                'note': 'pilot; research input candidate only; quality:<reason> flags are not quarantine (A decides)'}
    body = json.dumps(manifest, ensure_ascii=False, indent=1).encode()
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{args.version}.manifest.json', Body=body)
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{args.version}.done', Body=hashlib.sha256(body).hexdigest().encode())
    print(json.dumps({k: v for k, v in manifest.items() if k != 'shards'}, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['fetch', 'build'])
    p.add_argument('--max', type=int, default=100, help='fetch: stop after this many acquired reports')
    p.add_argument('--max-pages', type=int, default=80, help='fetch: list pages to look at in total')
    p.add_argument('--start', default='2023.10.30')
    p.add_argument('--end', default='2026.10.30')
    p.add_argument('--seed', default='20261030')
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--version', default='clean-v1')
    p.add_argument('--hold', help='build: JSON {row id: reason} to quarantine (e.g. reserved-overlap partials)')
    args = p.parse_args()
    fetch(args) if args.action == 'fetch' else build(args)


if __name__ == '__main__':
    main()

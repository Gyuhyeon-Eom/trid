"""data.go.kr file datasets (manuals, 편람, official FAQ / 질의회신) with an explicit licence, via the normal UI path.

For each dataset id: read the public detail page (/data/<pk>/fileData.do), parse 이용허락범위, and download only when it
is '제한 없음' or 공공누리 제1유형 (출처표시). Other licences (제2유형 상업적 이용금지, 제3유형 변경금지, 제4유형) are recorded
as rights_hold and not downloaded. The download repeats what the page's download button does, no login:
  POST /tcs/dss/selectFileDataDownload.do -> POST /cmm/cmm/check-limit.json -> GET /cmm/cmm/fileDownload.do
If the site asks for a captcha (download limit), the run stops; it is not bypassed. One request in flight, start >=
previous end + --interval (log logs/requests-data.go.kr.jsonl). Raw files are written once to
data/public-data/raw/<pk>/<sha16>.<ext>; data/public-data/files-state.jsonl records page metadata and licence evidence.

  uv run -q --no-project --python 3.12 --with requests python scripts/collect_datago_files.py 15043134 15038493 ...
"""
import argparse
import datetime
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
OUT = ROOT / 'data/public-data'
BASE = 'https://www.data.go.kr'
ALLOWED = ('제한 없음', '출처표시 (제 1유형)')
FIELDS = ('파일데이터명', '분류체계', '제공기관', '관리부서명', '보유근거', '수집방법', '업데이트 주기', '차기 등록 예정일',
          '매체유형', '확장자', '등록일', '수정일', '이용허락범위', '키워드', '설명')


def page_fields(page):
    """Label/value pairs of the detail table, as text."""
    text = re.sub(r'[ \t\r\f\v]+', ' ', html.unescape(re.sub(r'<[^>]+>', '\n', page)))
    lines = [x.strip() for x in text.split('\n') if x.strip()]
    out = {}
    for i, line in enumerate(lines[:-1]):
        if line in FIELDS and line not in out:
            out[line] = '' if lines[i + 1] in FIELDS else lines[i + 1]
    return out


def main():
    import collect_assembly_minutes as am
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pks', nargs='+')
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--state', default='files-state.jsonl',
                   help='state file under data/public-data owned by this process (one file per parallel slice)')
    args = p.parse_args()
    state_path = OUT / args.state
    done = set()
    for path in OUT.glob('files-state*.jsonl'):  # a dataset fetched by any slice is not fetched again
        done |= {json.loads(x)['pk'] for x in path.read_text().split('\n') if x and json.loads(x)['status'] == 'stored'}
    client = am.Client(am.HostLimiter(args.interval, host='data.go.kr'), ROOT / 'logs/requests-data.go.kr.jsonl')
    metrics = am.Metrics()
    with open(state_path, 'a') as state:
        for pk in args.pks:
            if pk in done:
                continue
            url = f'{BASE}/data/{pk}/fileData.do'
            page = client.request('GET', url, metrics).text
            f = page_fields(page)
            licence = f.get('이용허락범위', '')
            rec = {'pk': pk, 'page': url, 'fields': f, 'licence': licence, 'checked_at':
                   datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'code_sha256': CODE_SHA256}
            m = re.search(r"fn_fileDataDown\('(\d+)',\s*'([^']*)',\s*'([^']*)',\s*'(\d+)',\s*'(\d+)'\)", page)
            if not any(a in licence for a in ALLOWED):
                rec['status'] = 'rights_hold'
            elif not m:
                rec['status'] = 'no_download_button'
            else:
                meta = client.request('POST', f'{BASE}/tcs/dss/selectFileDataDownload.do', metrics, data={
                    'publicDataDetailPk': m.group(2), 'publicDataPk': m.group(1), 'atchFileId': m.group(3),
                    'fileDetailSn': m.group(4), 'publicDataTyCode': 'PR0051'}).json()
                if not meta.get('status'):
                    rec['status'] = f"download_refused: {meta.get('error')}"
                else:
                    info = meta['dataSetFileDetailInfo']
                    name = info['publicDataSj'] if info.get('publicDataTyDetailCode') == 'DATY05' else info['dataNm']
                    lim = client.request('POST', f'{BASE}/cmm/cmm/check-limit.json', metrics,
                                         data={'atchFileId': meta['atchFileId'], 'fileDetailSn': meta['fileDetailSn']})
                    if lim.status_code == 403 or lim.json().get('needCaptcha'):
                        state.write(json.dumps({**rec, 'status': 'captcha_required'}, ensure_ascii=False) + '\n')
                        raise SystemExit(f'stopped at {pk}: download limit asks for a captcha (not bypassed)')
                    r = client.request('GET', f'{BASE}/cmm/cmm/fileDownload.do', metrics, params={
                        'atchFileId': meta['atchFileId'], 'fileDetailSn': meta['fileDetailSn'], 'dataNm': name})
                    body = r.content
                    disp = r.headers.get('Content-Disposition') or ''
                    fname = unquote(re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disp).group(1)) if 'filename' in disp else name
                    try:
                        fname = fname.encode('latin-1').decode('utf-8')  # header bytes are UTF-8 read as Latin-1
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else 'bin'
                    digest = hashlib.sha256(body).hexdigest()
                    dest = OUT / 'raw' / pk / f'{digest[:16]}.{ext}'
                    if r.status_code == 200 and body and not body.lstrip()[:15].lower().startswith((b'<!doctype', b'<html')):
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(body)
                        rec.update({'status': 'stored', 'raw': str(dest.relative_to(ROOT)), 'sha256': digest, 'bytes': len(body),
                                    'file_name': fname, 'content_type': r.headers.get('Content-Type')})
                    else:
                        rec.update({'status': f'not_a_file (HTTP {r.status_code})', 'bytes': len(body)})
            state.write(json.dumps(rec, ensure_ascii=False) + '\n')
            state.flush()
            print(pk, rec['status'], licence[:40], f.get('파일데이터명', '')[:40])
    print(json.dumps(metrics.snapshot(), ensure_ascii=False))


if __name__ == '__main__':
    main()

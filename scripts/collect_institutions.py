"""Public institutions (ALIO, www.alio.go.kr) inventory for the corpus contract (docs/CORPUS_CONTRACT.md).

  agencies     every institution ALIO lists (/organ/findOrganApbaList.json): name, ALIO id, type (공기업/준정부기관/
               기타공공기관), supervising ministry, homepage, parent -> corpus/state/institutions/agencies.json and the
               repo copy data/corpus-inventory/alio-agencies.json.
  rules-index  the 내부규정 수시공시 catalogue (/occasional/findRuleList.json): one entry per published rule with
               institution, rule category (insdRuleDivis), title and the page's 재·개정일자 (ruleStDa) ->
               corpus/state/institutions/rules-index.json (resumable by page). Metadata only.
  reserved     benchmark-reserved rule families (supervisor reservation JSONL, read-only): detail JSON, every revision
               and attachment file, extracted text -> corpus/reserved/<reservation>/ only, never corpus/clean.
               Per-family state corpus/reserved/<reservation>/state/<seq>.json (resume skips finished families);
               acquisition status JSONL, A's check_reserved_overlap.py `sign` input and a summary are rebuilt from
               the states after every batch.

Licence evidence: ALIO 저작권정책 (https://www.alio.go.kr/notice/copyright.do) - public data free to use including
commercial use under 공공데이터법; ALIO-owned works under 저작권법 제24조의2; third-party rights and 비공개 정보 excluded.
Each institution's own regulation files remain that institution's public works; their per-file reuse basis is recorded
when text is collected. robots.txt: "Allow: /". Requests share one host-wide start interval (default 1 s), with
back-off on 429/5xx/Retry-After/latency (the HostLimiter of collect_assembly_minutes).
"""
import argparse
import datetime
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from collect_assembly_minutes import (BUCKET, Client, HostLimiter, Metrics, list_keys, put_raw_keep,  # noqa: E402
                                      put_verified, read_json_s3, s3_client)

SOURCE = 'institutions'
BASE = 'https://www.alio.go.kr'


def agency_row(item):
    keep = ('apbaId', 'apbaNa', 'parnApbaId', 'parnApbaNa', 'typeNa', 'jidtNa', 'apbaType', 'jidtDptm', 'homepage',
            'addrCd', 'fdate')
    row = {k: item.get(k) for k in keep}
    row['homepage'] = (row['homepage'] or '').strip() or None
    return row


def rule_row(item):
    return {'seq': item.get('seq'), 'institution_id': item.get('apbaId'), 'institution': item.get('pname'),
            'institution_type_code': item.get('apbaType'), 'ministry_code': item.get('jidtDptm'),
            'title': item.get('title'), 'category': item.get('insdRuleDivis'),
            'revision_date': item.get('ruleStDa'),  # shown on the detail page as 재·개정일자, not 시행일
            'disclosed_date': item.get('idate'), 'base_year': item.get('critYyyy')}


def agencies(client, s3, metrics):
    rows, page = [], 1
    while True:
        r = client.request('POST', f'{BASE}/organ/findOrganApbaList.json', metrics,
                           json={'apbaNa': '', 'pageNo': page}).json()
        organ = r['data']['organList']
        rows += [agency_row(x) for x in organ['result']]
        if not organ['page'].get('nextPage_is') and page >= organ['page'].get('totalPage', page):
            break
        page += 1
    body = json.dumps({'source': f'{BASE}/guide/publicAgencyList.do', 'count': len(rows),
                       'by_type': dict(Counter(r['typeNa'] for r in rows)), 'agencies': rows,
                       'metrics': metrics.snapshot()}, ensure_ascii=False, indent=1)
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/agencies.json', Body=body.encode())
    out = ROOT / 'data/corpus-inventory/alio-agencies.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + '\n')
    print(json.dumps({'agencies': len(rows), 'by_type': dict(Counter(r['typeNa'] for r in rows))}, ensure_ascii=False))


def rules_index(client, s3, metrics, max_pages=None, workers=1):
    """Pages are fetched in batches of 50; state advances only after a whole batch, so a stop loses <50 pages.
    Workers overlap server latency; request starts still share the one host interval."""
    key = f'corpus/state/{SOURCE}/rules-index.json'
    state = read_json_s3(s3, key, {'next_page': 1, 'total_pages': None, 'rules': {}})
    fetch = lambda page: client.request('GET', f'{BASE}/occasional/findRuleList.json', metrics,
                                        params={'pageNo': page}).json()
    if state['total_pages'] is None:
        first = fetch(1)
        state['total_pages'], state['total_count'] = first['data']['page']['totalPage'], first['data']['totalCnt']
    last = min(state['total_pages'], max_pages or state['total_pages'])
    with ThreadPoolExecutor(workers) as pool:
        while state['next_page'] <= last:
            batch = range(state['next_page'], min(last, state['next_page'] + 49) + 1)
            for r in pool.map(fetch, batch):
                state['total_count'] = r['data']['totalCnt']
                for item in r['data']['result']:
                    state['rules'][str(item['seq'])] = rule_row(item)
            state['next_page'] = batch[-1] + 1
            state['metrics'] = metrics.snapshot()
            s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(state, ensure_ascii=False).encode())
            print(json.dumps({'page': batch[-1], 'total_pages': state['total_pages'], 'rules': len(state['rules']),
                              'workers': workers, 'interval_s': round(client.limiter.interval, 2),
                              'metrics': metrics.snapshot()}), flush=True)


FILE_URL = BASE + '/download/rulefiledown.json?fileNo={}'
DATE_IN_NAME = re.compile(r'\d{4}\s*(?:년도?|[.\-])\s*\d{1,2}\s*(?:월|[.\-])\s*\d{1,2}\s*일?')
LICENCE_EVIDENCE = {'site_policy': 'https://www.alio.go.kr/notice/copyright.do',
                    'site_policy_text': ('공공데이터법에 따라 알리오 제공 공공데이터는 영리 포함 자유이용; ALIO 보유 저작물은 '
                                         '저작권법 제24조의2; 제3자 권리·비공개정보는 제공 대상 제외'),
                    'original_licence': 'unconfirmed (per-file 공공누리 type not shown on ALIO)'}


def files_of(b_files):
    """The site's own split of bFiles ("fileNo|name,fileNo|name"); a name containing ',' is re-joined."""
    out = []
    for part in (b_files or '').split(','):
        if '|' in part and part.split('|', 1)[0].strip().isdigit():
            no, name = part.split('|', 1)
            out.append([no.strip(), name.strip()])
        elif out:
            out[-1][1] += ',' + part
    return [tuple(x) for x in out]


def kind_of(data, name):
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if data[:4] == b'%PDF':
        return 'pdf'
    if data[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return 'hwp' if ext == 'hwp' else f'ole:{ext}'
    if data[:2] == b'PK':  # HWPX is a zip; some institutions upload HWPX bytes under a .hwp name
        return 'hwpx' if ext in ('hwpx', 'hwp') else f'zip:{ext}'
    return f'unknown:{ext}'


def extract(kind, data):
    from trid.service.ingest import hwpx_text, normalize, tool_text
    if kind == 'hwpx':
        return normalize(hwpx_text(data))
    if kind in ('hwp', 'pdf'):
        return normalize(tool_text(kind, data))
    raise ValueError(f'no extractor for {kind}')


def acquire_family(client, s3, metrics, rid, fam):
    seq = fam['doc_family_id'].rsplit(':', 1)[1]
    prefix = f'corpus/reserved/{rid}'
    r = client.request('GET', f'{BASE}/occasional/findRuleDtl.json', metrics, params={'seq': seq})
    t = time.monotonic()
    detail_sha, detail_key = put_raw_keep(s3, f'{prefix}/raw/{seq}/detail.json', r.content)  # raw is never replaced
    metrics.add('s3_s', time.monotonic() - t)
    state = {'doc_family_id': fam['doc_family_id'], 'seq': seq, 'reservation': rid,
             'reserved_business_group': fam.get('business_group'), 'business_group_review': 'pending',
             'page_url': f'{BASE}/occasional/ruleDtl.do?seq={seq}', 'detail_http': r.status_code,
             'detail_key': detail_key, 'detail_sha256': detail_sha,
             'collected_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
             'licence_evidence': LICENCE_EVIDENCE, 'licence_status': 'unconfirmed', 'files': []}
    body = r.json() if r.status_code == 200 and r.content[:1] == b'{' else {}
    d = body.get('data') if body.get('status') == 'success' else None
    if not d:
        state['status'] = 'detail_unavailable'
        return state
    state.update({'institution': d.get('pname'), 'institution_id': d.get('instNo'), 'title': d.get('title'),
                  'category': d.get('insdRuleDivis'),
                  'dates': {'alio_revision_date': d.get('retryRvsnYmd'),  # page label 재·개정일자
                            'alio_bdate': d.get('bdate'), 'alio_idate': d.get('idate'),  # unlabelled on the page
                            'effective_date': None, 'effective_date_note': '시행일 is only in the text; see effective_clauses'}})
    for index, (no, name) in enumerate(files_of(d.get('bFiles'))):
        m = DATE_IN_NAME.search(name)
        rec = {'file_no': no, 'file_name': name, 'list_index': index, 'url': FILE_URL.format(no),
               'filename_date_text': m.group(0) if m else None}
        state['files'].append(rec)
        f = client.request('GET', FILE_URL.format(no), metrics)
        rec['http'] = f.status_code
        if f.status_code != 200 or not f.content:
            rec['download_status'] = f'http_{f.status_code}' if f.status_code != 200 else 'empty_body'
            continue
        data = f.content
        rec.update({'download_status': 'ok', 'bytes': len(data), 'content_type': f.headers.get('Content-Type'),
                    'kind': kind_of(data, name)})
        t = time.monotonic()
        rec['raw_sha256'], rec['raw_key'] = put_raw_keep(s3, f'{prefix}/raw/{seq}/{no}.{rec["kind"].split(":")[-1] or "bin"}', data)
        metrics.add('s3_s', time.monotonic() - t)
        if rec['kind'] == 'zip:zip':
            rec['members'] = zip_members(s3, metrics, prefix, seq, no, data)
            states = Counter(x.get('extract_status') for x in rec['members'])
            rec['extract_status'] = ('empty_archive' if not rec['members'] else 'ok' if states == {'ok': len(rec['members'])}
                                     else 'partial' if states.get('ok') else 'failed')
        else:
            rec.update(text_record(s3, metrics, prefix, seq, no, rec['kind'], data))
    ok = [x for x in state['files'] if x.get('extract_status') == 'ok']
    state['status'] = ('no_files' if not state['files'] else 'text_acquired' if len(ok) == len(state['files'])
                       else 'partial' if ok else 'extraction_failed'
                       if all(x.get('download_status') == 'ok' for x in state['files']) else 'download_failed')
    return state


EXTRACTOR = {'hwp': 'rhwp export-doclang', 'pdf': 'pdftotext', 'hwpx': 'hwpx section XML'}


def text_record(s3, metrics, prefix, seq, stem, kind, data):
    """Extract one document; text stored at <prefix>/text/<seq>/<stem>.txt. Failures are recorded, never promoted."""
    rec, t = {}, time.monotonic()
    try:
        text = unicodedata.normalize('NFC', extract(kind, data))
        rec['extract_status'] = 'ok' if text.strip() else 'empty_text'
    except Exception as exc:
        text, rec['extract_status'], rec['extract_error'] = '', 'failed', repr(exc)[:300]
    metrics.add('parse_s', time.monotonic() - t)
    if text.strip():
        t = time.monotonic()
        rec['text_sha256'] = put_verified(s3, f'{prefix}/text/{seq}/{stem}.txt', text.encode())
        metrics.add('s3_s', time.monotonic() - t)
        rec.update({'text_key': f'{prefix}/text/{seq}/{stem}.txt', 'chars': len(text), 'extractor': EXTRACTOR[kind],
                    'effective_clauses': [line.strip() for line in text.splitlines() if '시행한다' in line][:8],
                    'mentions_kogl': '공공누리' in text})
    return rec


def zip_members(s3, metrics, prefix, seq, no, data):
    """Archived revisions/attachments (e.g. "…(개정 전).zip"): one record per file member; the zip itself is the raw."""
    import io
    import zipfile
    members = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        return [{'member': None, 'extract_status': 'failed', 'extract_error': repr(exc)[:300]}]
    for i, info in enumerate(m for m in archive.infolist() if not m.is_dir()):
        name = info.filename
        if not info.flag_bits & 0x800:  # no UTF-8 flag: Korean archives are usually CP949
            try:
                name = name.encode('cp437').decode('cp949')
            except UnicodeError:
                pass
        body = archive.read(info) if info.file_size <= 200 * 1024 * 1024 else b''
        m = DATE_IN_NAME.search(name)
        rec = {'member': name, 'member_index': i, 'bytes': info.file_size, 'kind': kind_of(body, name),
               'member_sha256': hashlib.sha256(body).hexdigest(), 'filename_date_text': m.group(0) if m else None}
        rec.update(text_record(s3, metrics, prefix, seq, f'{no}-m{i:03d}', rec['kind'], body))
        members.append(rec)
    return members


def round_robin(families):
    """Interleave business groups so a partial run covers every group (supervisor order kept inside a group)."""
    groups = {}
    for fam in families:
        groups.setdefault(fam.get('business_group'), []).append(fam)
    queues, out = list(groups.values()), []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def publish(s3, rid, states, reservation_count, metrics=None):
    rows = [states[k] for k in sorted(states, key=lambda k: int(k))]
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (ROOT / f'reports/coordination/{rid}-acquisition.jsonl').write_text(text)
    s3.put_object(Bucket=BUCKET, Key=f'corpus/reserved/{rid}/acquisition.jsonl', Body=text.encode())
    files = [f for r in rows for f in r['files']]
    summary = {'reservation': rid, 'reserved_families': reservation_count, 'families_attempted': len(rows),
               'families_pending': reservation_count - len(rows),
               'family_status': dict(Counter(r['status'] for r in rows)),
               'family_status_by_group': dict(Counter(f"{r['reserved_business_group']} | {r['status']}" for r in rows)),
               'licence_status': dict(Counter(r['licence_status'] for r in rows)),
               'files': len(files), 'file_kinds': dict(Counter(f.get('kind') for f in files)),
               'file_download_status': dict(Counter(f.get('download_status') for f in files)),
               'file_extract_status': dict(Counter(f.get('extract_status') for f in files)),
               'archive_members': sum(len(f.get('members', [])) for f in files),
               'archive_member_extract_status': dict(Counter(m.get('extract_status') for f in files for m in f.get('members', []))),
               'text_chars': sum(f.get('chars', 0) + sum(m.get('chars', 0) for m in f.get('members', [])) for f in files),
               'files_mentioning_kogl': sum(bool(f.get('mentions_kogl')) for f in files),
               'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
    if metrics:
        summary['metrics'] = metrics.snapshot()
    s3.put_object(Bucket=BUCKET, Key=f'corpus/reserved/{rid}/summary.json',
                  Body=json.dumps(summary, ensure_ascii=False, indent=1).encode())
    (ROOT / f'reports/coordination/{rid}-acquisition-summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    return summary


def load_states(s3, rid):
    states = {}
    for key in list_keys(s3, f'corpus/reserved/{rid}/state/'):
        st = read_json_s3(s3, key, None)
        states[st['seq']] = st
    return states


def reserved(client, s3, metrics, reservation, workers, limit=None, retry=()):
    """retry: family statuses to fetch again (e.g. partial after an extractor fix); their old state is replaced."""
    rid = reservation.name.removesuffix('.jsonl')
    families = [json.loads(line) for line in reservation.read_text().splitlines() if line.strip()]
    families = [f for f in families if f['doc_family_id'].startswith('institutions:alio-rule:')]
    states = load_states(s3, rid)
    pending = lambda f: (st := states.get(f['doc_family_id'].rsplit(':', 1)[1])) is None or st['status'] in retry
    todo = [f for f in round_robin(families) if pending(f)][:limit]
    print(json.dumps({'reserved': len(families), 'done': len(states), 'todo': len(todo)}), flush=True)
    started = time.monotonic()

    def one(fam):
        try:
            return acquire_family(client, s3, metrics, rid, fam)
        except Exception as exc:  # leave the family pending; a re-run retries it
            print(f"pending {fam['doc_family_id']}: {exc!r}", flush=True)
            return None

    with ThreadPoolExecutor(workers) as pool:
        for begin in range(0, len(todo), 20):
            for st in pool.map(one, todo[begin:begin + 20]):
                if st:
                    s3.put_object(Bucket=BUCKET, Key=f"corpus/reserved/{rid}/state/{st['seq']}.json",
                                  Body=json.dumps(st, ensure_ascii=False).encode())
                    states[st['seq']] = st
            summary = publish(s3, rid, states, len(families), metrics)
            done = begin + len(todo[begin:begin + 20])
            rate = done / max(1e-9, time.monotonic() - started)
            print(json.dumps({'done': len(states), 'of': len(families), 'families_per_min': round(rate * 60, 2),
                              'eta_min': round((len(todo) - done) / max(rate, 1e-9) / 60, 1),
                              'status': summary['family_status'], 'interval_s': round(client.limiter.interval, 2),
                              'metrics': metrics.snapshot()}, ensure_ascii=False), flush=True)


def sign_input(s3, rid):
    """A's check_reserved_overlap.py `sign` input: one row per extracted file {family_id, text, doc_id, source_url,
    group}. Written to S3 and data/ (gitignored); reserved text never goes under corpus/clean."""
    states = load_states(s3, rid)
    rows = []
    for st in states.values():
        for f in st['files']:
            for doc, doc_id in [(f, f"{st['doc_family_id']}:file:{f['file_no']}")] + [
                    (m, f"{st['doc_family_id']}:file:{f['file_no']}:member:{m['member_index']}") for m in f.get('members', [])]:
                if doc.get('text_key'):
                    text = s3.get_object(Bucket=BUCKET, Key=doc['text_key'])['Body'].read().decode()
                    rows.append({'family_id': st['doc_family_id'], 'text': text, 'doc_id': doc_id,
                                 'source_url': f['url'], 'group': st['doc_family_id']})
    body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(), mtime=0)
    digest = put_verified(s3, f'corpus/reserved/{rid}/sign-input.jsonl.gz', body)
    local = ROOT / f'data/{rid}/sign-input.jsonl.gz'
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(body)
    print(json.dumps({'rows': len(rows), 'families': len({r['family_id'] for r in rows}), 'sha256': digest,
                      's3': f's3://{BUCKET}/corpus/reserved/{rid}/sign-input.jsonl.gz', 'local': str(local)}))


def homepage_host(homepage):
    """'www.sonosa.or.kr', 'http://x.go.kr/main' -> 'sonosa.or.kr', 'x.go.kr' (lower case, no www., no path)."""
    from urllib.parse import urlsplit
    host = (urlsplit(homepage if '//' in homepage else '//' + homepage).hostname or '').lower().strip('.')
    return host[4:] if host.startswith('www.') else host


def reserved_hosts(s3, reservation):
    """Hosts whose pages may carry reserved texts: alio.go.kr plus each reserved institution's ALIO homepage."""
    raw = reservation.read_bytes()
    inst = {json.loads(line)['institution_id'] for line in raw.decode().splitlines() if line.strip()}
    agencies = read_json_s3(s3, f'corpus/state/{SOURCE}/agencies.json', None)['agencies']
    by_host = {'alio.go.kr': 'ALIO'}
    for a in agencies:
        host = homepage_host(a.get('homepage') or '')
        if a['apbaId'] in inst and host:
            by_host.setdefault(host, a['apbaNa'])
    with_home = {a['apbaId'] for a in agencies if homepage_host(a.get('homepage') or '')}
    out = {'reservation': str(reservation.relative_to(ROOT)), 'reservation_sha256': hashlib.sha256(raw).hexdigest(),
           'agencies': f's3://{BUCKET}/corpus/state/{SOURCE}/agencies.json', 'institutions': len(inst),
           'institutions_without_homepage': sorted(inst - with_home), 'hosts': sorted(by_host),
           'institution_of_host': by_host,
           'use': 'provisional quarantine of web rows from these hosts until a text-overlap check against reserved texts'}
    path = ROOT / f"reports/coordination/{reservation.name.removesuffix('.jsonl')}-hosts.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({'institutions': len(inst), 'without_homepage': len(out['institutions_without_homepage']),
                      'hosts': len(out['hosts'])}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['agencies', 'rules-index', 'reserved', 'reserved-sign-input', 'reserved-hosts'])
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--max-pages', type=int)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--reservation', type=Path, default=ROOT / 'reports/coordination/benchmark-reserve-001.jsonl')
    p.add_argument('--limit', type=int)
    p.add_argument('--retry-status', nargs='*', default=[], help='re-fetch families whose state has these statuses')
    args = p.parse_args()
    s3, metrics = s3_client(), Metrics()
    client = Client(HostLimiter(args.interval, host='www.alio.go.kr'), ROOT / 'logs/requests-www.alio.go.kr.jsonl')
    if args.action == 'agencies':
        agencies(client, s3, metrics)
    elif args.action == 'rules-index':
        rules_index(client, s3, metrics, args.max_pages, args.workers)
    elif args.action == 'reserved':
        reserved(client, s3, metrics, args.reservation, args.workers, args.limit, tuple(args.retry_status))
    elif args.action == 'reserved-hosts':
        reserved_hosts(s3, args.reservation)
    else:
        sign_input(s3, args.reservation.name.removesuffix('.jsonl'))


if __name__ == '__main__':
    main()

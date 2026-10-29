"""National Assembly minutes (국회회의록, record.assembly.go.kr) -> corpus contract clean-v1 (docs/CORPUS_CONTRACT.md).

  discover  walk the public meeting tree of each 대수 and meeting class (본회의, 상임위원회, 특별위원회,
            예산결산특별위원회, 국정감사, 국정조사) through the site's own list endpoints (mnts/async/*.do, the same
            POSTs its list.js issues). The index and the finished (대수, class) pairs are saved to
            corpus/state/assembly-minutes/<index> every few list requests, so an interrupted walk resumes.
  fetch     fetch each indexed meeting's text viewer (viewer/minutes/xml.do?id=N&type=view) with a small worker pool
            that shares ONE per-host request-start limiter; keep the original HTML gzip-compressed under
            corpus/raw/assembly-minutes/; extract text, redact PII (trid.korean_quality.redact) and write clean-v1
            shards with S3-verified SHA-256 to corpus/clean/assembly-minutes/clean-v1/. Each shard has a manifest
            under corpus/state/assembly-minutes/clean-v1/ listing its meeting ids, hashes and timing metrics;
            re-running skips those ids.
  aggregate verify every shard checksum, then write stats.json, provenance.json, ATTRIBUTION.txt and the _DONE marker.

Licence: the minutes site footer carries 공공누리 제1유형 (출처표시); the Assembly copyright policy
(https://open.assembly.go.kr/portal/policy/copyRightPage.do) declares Assembly-authored works free to use under
저작권법 제24조의2. robots.txt is absent (404) and no request policy is published; absence is NOT read as permission
for unlimited speed. Default request-start interval is 1 s for the whole host, slowed automatically on 429,
Retry-After, 5xx or rising latency. Never spread requests over other IPs or accounts.
"""
import argparse
import base64
import datetime
import gzip
import hashlib
import html
import json
import re
import statistics
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKET = 'trid-training-148761639313'
SOURCE = 'assembly-minutes'
VERSION = 'clean-v1'
BASE = 'https://record.assembly.go.kr/assembly'
UA = 'Mozilla/5.0 (compatible; trid-collector/0.1; +public-sector research corpus)'
LICENSE = 'KOGL-Type1'
CLASSES = {'1': 'assembly_plenary', '2': 'assembly_committee', '3': 'assembly_special_committee',
           '4': 'assembly_budget_committee', '5': 'assembly_audit', '6': 'assembly_investigation'}
TERMS = [str(n) for n in range(1, 23)] + ['10.5']
MINUTE_LINK = re.compile(r'viewer/minutes/xml\.do\?id=(\d+)&(?:amp;)?type=view')
HEADER = re.compile(r'제(\d+(?:\.\d+)?)대국회\s*제(\d+)회\s*\(([^)]*)\)\s*제(\d+)차\s*(.+?)\s*\((\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\)')
UI_TAIL = {'top', 'down', '발언 저장', '발언 인쇄'}
SPLITS = (('test', 50), ('valid', 100))  # per-10,000 bucket bounds; dev splits, not an independent final test


def payload(kind, d, term):
    """(endpoint, form) for a list node, mirroring resources/mnts/total/list.js."""
    g = lambda k: d.get(k, '')
    if kind in ('sess', 'sess2', 'mem'):
        return '/mnts/async/sess.do', {'th_sch': g('th'), 'class_id_sch': g('class'), 'sess_sch': g('sess'),
                                      'chk': g('chk1'), 'cmit_chk_cmit': g('chk2'), 'cmit_chk_sub': g('chk3'),
                                      'mnts_id_sch': g('id'), 'mnts_year_sch': g('year')}
    if kind == 'cmit':
        url = '/mnts/async/sessCmit2.do' if (g('th') == '5' and g('year') and g('class') in ('2', '4')) else '/mnts/async/sessCmit.do'
        return url, {'council_cd_sch': g('cl'), 'th_sch': g('th'), 'sess_sch': g('sess'), 'class_id_sch': g('class'),
                     'cmit_id_sch': g('cmit'), 'cmit_chk': g('chk1'), 'cmit_chk_cmit': g('chk2'),
                     'cmit_chk_sub': g('chk3'), 'cmit_chk_etc': g('chk4'), 'mnts_year_sch': g('year')}
    if kind == 'cmit2':
        return '/mnts/async/sessCmit2.do', {'th_sch': g('th'), 'class_id_sch': g('class'), 'sess_sch': g('sess'),
                                           'cmit_cd_sch': g('cmit'), 'chk': g('chk')}
    if kind == 'sub':
        return '/mnts/async/cmit.do', {'council_cd_sch': g('cl'), 'th_sch': g('th'), 'class_id_sch': g('class'),
                                      'sess_sch': g('sess'), 'cmit_cd_sch': g('cmitcd'), 'cmit_chk': g('chk1'),
                                      'cmit_chk_cmit': g('chk2'), 'cmit_chk_sub': g('chk3'), 'cmit_chk_etc': g('chk4'),
                                      'mnts_id_sch': g('id'), 'conf_year': g('dt')}
    if kind == 'ord_num':
        return '/mnts/async/ord.do', {'council_cd_sch': g('cl'), 'mnts_id_sch': g('id'), 'class_id_sch': g('class'),
                                     'th_sch': g('th')}
    raise ValueError(kind)


class NodeParser(HTMLParser):
    """Collects expandable list nodes (class tit/btn_tit + loader class) with their data-* attributes."""
    KINDS = ('sess', 'sess2', 'mem', 'cmit', 'cmit2', 'sub', 'ord_num')

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get('class') or '').split())
        if not classes & {'tit', 'btn_tit'}:
            return
        for kind in self.KINDS:
            if kind in classes:
                data = {k[5:]: (v or '') for k, v in attrs.items() if k.startswith('data-')}
                self.nodes.append((kind, data))
                return


def nodes_of(page):
    parser = NodeParser()
    parser.feed(page)
    return parser.nodes


def minute_ids(page):
    return sorted({int(i) for i in MINUTE_LINK.findall(page)})


def extract(page):
    """Viewer HTML -> (header metadata dict, minutes text). Text starts at the document <h1>…회의록 heading."""
    starts = [m.start() for m in re.finditer(r'<h1>[^<]*회의록', page)]
    head_text = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', page[:starts[-1] if starts else 0])))
    body = page[starts[-1]:] if starts else page
    body = re.sub(r'(?is)<(script|style|noscript|button)\b.*?</\1>', ' ', body)
    body = re.sub(r'(?i)<br\s*/?>|</(p|div|li|tr|h\d|dd|dt|table)>', '\n', body)
    lines = [' '.join(line.split()) for line in html.unescape(re.sub(r'<[^>]+>', ' ', body)).splitlines()]
    lines = [line for line in lines if line]
    while lines and (lines[-1] in UI_TAIL or lines[-1].startswith('Scroll :')):
        lines.pop()
    # The viewer repeats each speaker as an interactive "<name> 선택" control right before the speaker line.
    lines = [line for i, line in enumerate(lines)
             if not (line.endswith(' 선택') and i + 1 < len(lines) and lines[i + 1].startswith(line[:-3]))]
    meta = {}
    m = HEADER.search(head_text)
    if m:
        term, session, session_kind, order, meeting, y, mo, d = m.groups()
        meta = {'term': term, 'session': int(session), 'session_kind': session_kind, 'order': int(order),
                'meeting': meeting, 'date': f'{int(y):04d}-{int(mo):02d}-{int(d):02d}'}
    return meta, unicodedata.normalize('NFC', '\n'.join(lines))


def split_of(group):
    bucket = int(hashlib.sha256(f'split:{group}'.encode()).hexdigest()[:8], 16) % 10000
    return next((name for name, upper in SPLITS if bucket < upper), 'train')


def row_of(meeting, page, collected_at):
    """-> (clean-v1 row, pii Counter)."""
    from trid.korean_quality import redact
    meta, text = extract(page)
    text, pii = redact(text)
    title = (f"제{meta['term']}대국회 제{meta['session']}회 제{meta['order']}차 {meta['meeting']}"
             if meta else meeting.get('label', ''))
    url = f"{BASE}/viewer/minutes/xml.do?id={meeting['id']}&type=view"
    group = f"{SOURCE}:{meeting['id']}"
    return {'id': group, 'group': group, 'kind': CLASSES.get(str(meeting.get('class')), 'assembly_minutes'),
            'title': title, 'text': text, 'source': url, 'license': LICENSE,
            'attribution': f'출처: 국회사무처 국회회의록, {url}', 'date': meta.get('date'),
            'sha256': hashlib.sha256(text.encode()).hexdigest(), 'collected_at': collected_at, 'extractor': 'html:viewer',
            'clean_version': VERSION, 'flags': (['pii_redacted'] if pii else []) + ([] if meta else ['header_unparsed']),
            'split': split_of(group),
            'provenance': {'institution': '국회', 'institution_type': '국회', 'doc_type': '회의록',
                           'committee': meta.get('meeting'), 'published_date': meta.get('date')},
            'meta': {**{k: v for k, v in meta.items() if k != 'date'}, 'minutes_id': meeting['id'],
                     'class_id': meeting.get('class'), 'list_term': meeting.get('th'), 'pii_redactions': dict(pii)}}, pii


class HostLimiter:
    """One request-start interval for the whole host, shared by every worker thread. Slows down on 429/5xx,
    Retry-After or time-to-first-byte above max(2x running median, median + 1 s); drifts back toward the base
    interval when healthy. Callers pass time-to-first-byte, not total transfer time: a large file download is not a
    sign of server distress (2026-10-30: total time pushed the ALIO interval from 1 s to the 30 s cap)."""

    def __init__(self, interval, max_interval=30.0):
        self.base = self.interval = interval
        self.max_interval = max_interval
        self.next_start, self.lock, self.latencies = 0.0, threading.Lock(), []

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        wait = start - now
        if wait > 0:
            time.sleep(wait)
        return wait

    def observe(self, status, latency, retry_after=None):
        with self.lock:
            self.latencies = (self.latencies + [latency])[-50:]
            median = statistics.median(self.latencies)
            if status == 429 or status >= 500 or retry_after:
                self.interval = min(self.max_interval, max(self.interval * 2, float(retry_after or 0)))
                self.next_start = time.monotonic() + self.interval
            elif len(self.latencies) >= 10 and latency > max(2 * median, median + 1.0):
                self.interval = min(self.max_interval, self.interval * 1.5)
            else:
                self.interval = max(self.base, self.interval * 0.95)


class Client:
    def __init__(self, limiter):
        import requests
        self.session, self.limiter = requests.Session(), limiter
        self.session.headers.update({'User-Agent': UA})

    def request(self, method, url, metrics, **kw):
        for attempt in range(6):
            waited = self.limiter.acquire()
            started = time.monotonic()
            try:
                r = self.session.request(method, url, timeout=180, **kw)
            except Exception as exc:  # network error: count as a 5xx-like failure and back off
                self.limiter.observe(599, time.monotonic() - started)
                metrics.record(waited, time.monotonic() - started, 0, 599)
                print(f'retry after {exc!r}', flush=True)
                continue
            latency = time.monotonic() - started
            retry_after = r.headers.get('Retry-After')
            retry_after = float(retry_after) if retry_after and retry_after.isdigit() else None
            self.limiter.observe(r.status_code, r.elapsed.total_seconds(), retry_after)  # elapsed = time to headers
            metrics.record(waited, latency, len(r.content), r.status_code)
            if r.status_code < 500 and r.status_code != 429:
                return r
        raise RuntimeError(f'request failed after retries: {url}')


class Metrics:
    """Thread-safe counters separating limiter wait, server response, payload, parsing and S3 time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {'requests': 0, 'wait_s': 0.0, 'response_s': 0.0, 'bytes': 0, 'parse_s': 0.0, 's3_s': 0.0,
                     'status': {}}

    def record(self, wait, response, size, status):
        with self.lock:
            d = self.data
            d['requests'] += 1
            d['wait_s'] += wait
            d['response_s'] += response
            d['bytes'] += size
            d['status'][str(status)] = d['status'].get(str(status), 0) + 1

    def add(self, field, seconds):
        with self.lock:
            self.data[field] += seconds

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data))


def s3_client():
    import boto3
    return boto3.client('s3')


def read_json_s3(s3, key, default):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)['Body'].read())
    except s3.exceptions.NoSuchKey:
        return default


def list_keys(s3, prefix):
    keys = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [item['Key'] for item in page.get('Contents', [])]
    return keys


def put_verified(s3, key, body):
    digest = hashlib.sha256(body).digest()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ChecksumSHA256=base64.b64encode(digest).decode())
    return digest.hex()


def discover(client, s3, terms, classes, index, metrics):
    """Resumable walk; state = {'done': ['<term>/<class>', ...], 'meetings': {id: {...}}}."""
    key = f'corpus/state/{SOURCE}/{index}'
    state = read_json_s3(s3, key, {'done': [], 'meetings': {}})
    save = lambda: s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(state).encode())
    for term in terms:
        for cls in classes:
            if f'{term}/{cls}' in state['done']:
                continue
            list_url = f'{BASE}/mnts/total/{term}.do?' + urlencode({'class_id_sch': cls, 'cmit_chk': 'all'})
            page = client.request('GET', list_url, metrics).text
            queue, seen_requests, requests_made = nodes_of(page), set(), 0
            for mid in minute_ids(page):
                state['meetings'].setdefault(str(mid), {'id': mid, 'th': term, 'class': cls})
            while queue:
                kind, data = queue.pop()
                data.setdefault('th', term)
                data.setdefault('class', cls)
                endpoint, form = payload(kind, data, term)
                request_key = (endpoint, tuple(sorted(form.items())))
                if request_key in seen_requests:
                    continue
                seen_requests.add(request_key)
                sub = client.request('POST', BASE + endpoint, metrics, data=form,
                                     headers={'X-Requested-With': 'XMLHttpRequest', 'Referer': list_url}).text
                requests_made += 1
                for mid in minute_ids(sub):
                    state['meetings'].setdefault(str(mid), {'id': mid, 'th': term, 'class': cls, 'sess': data.get('sess', '')})
                queue.extend(nodes_of(sub))
                if requests_made % 25 == 0:
                    save()
            state['done'].append(f'{term}/{cls}')
            save()
            print(json.dumps({'term': term, 'class': cls, 'meetings_total': len(state['meetings']),
                              'list_requests': requests_made, 'metrics': metrics.snapshot()}), flush=True)


def done_ids(s3):
    ids = set()
    for key in list_keys(s3, f'corpus/state/{SOURCE}/{VERSION}/shard-'):
        ids.update(read_json_s3(s3, key, {'ids': []})['ids'])
    return ids


def stratified(meetings, seed):
    """Seeded order that alternates meeting classes (plenary, committee, audit ...) so a limited fetch is spread."""
    import random
    by_class = {}
    for m in sorted(meetings, key=lambda m: m['id']):
        by_class.setdefault(str(m.get('class')), []).append(m)
    queues = []
    for cls in sorted(by_class):
        q = by_class[cls]
        random.Random(f'{seed}:{cls}').shuffle(q)
        queues.append(q)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


def fetch(client, s3, index, per_shard, workers, limit, metrics, seed=None):
    meetings = list(read_json_s3(s3, f'corpus/state/{SOURCE}/{index}', {'meetings': {}})['meetings'].values())
    meetings = stratified(meetings, seed) if seed is not None else sorted(meetings, key=lambda m: m['id'])
    done = done_ids(s3)
    todo = [m for m in meetings if m['id'] not in done][:limit]
    print(f'{len(meetings)} meetings indexed, {len(done)} done, {len(todo)} to fetch '
          f'({dict(Counter(str(m.get("class")) for m in todo))} by class)', flush=True)
    shard = 1 + max([-1] + [int(k.rsplit('-', 1)[1].split('.')[0]) for k in list_keys(s3, f'corpus/state/{SOURCE}/{VERSION}/shard-')])
    started = time.monotonic()

    def one(meeting):
        r = client.request('GET', f"{BASE}/viewer/minutes/xml.do?id={meeting['id']}&type=view", metrics)
        if r.status_code != 200 or '회의록' not in r.text:
            return meeting, None, None, f'http_{r.status_code}'
        t = time.monotonic()
        raw_key = f"corpus/raw/{SOURCE}/{meeting['id']}.html.gz"
        raw_sha = put_verified(s3, raw_key, gzip.compress(r.content, mtime=0))
        metrics.add('s3_s', time.monotonic() - t)
        t = time.monotonic()
        row, pii = row_of(meeting, r.text, datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
        row['meta'].update({'raw_key': raw_key, 'raw_gzip_sha256': raw_sha,
                            'raw_html_sha256': hashlib.sha256(r.content).hexdigest()})
        metrics.add('parse_s', time.monotonic() - t)
        return meeting, row, pii, None

    for begin in range(0, len(todo), per_shard):
        chunk = todo[begin:begin + per_shard]
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(one, chunk))
        seen_text = {}
        rows = []
        for _, row, _, _ in results:
            if row and row['text']:
                if row['sha256'] in seen_text:
                    row['flags'].append(f"duplicate_text_of:{seen_text[row['sha256']]}")
                seen_text.setdefault(row['sha256'], row['id'])
                rows.append(row)
        empty = [str(m['id']) for m, row, _, _ in results if row is not None and not row['text']]
        failures = {str(m['id']): err for m, _, _, err in results if err}
        ids = [m['id'] for m, row, _, _ in results if row is not None]
        body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(), mtime=0)
        t = time.monotonic()
        key = f'corpus/clean/{SOURCE}/{VERSION}/docs-{shard:05d}.jsonl.gz'
        digest = put_verified(s3, key, body)
        pii = {}
        for _, _, p, _ in results:
            for label, n in (p or {}).items():
                pii[label] = pii.get(label, 0) + n
        manifest = {'key': key, 'sha256': digest, 'bytes': len(body), 'ids': ids, 'failures': failures,
                    'empty_text_ids': empty,
                    'duplicate_text_rows': sum(any(f.startswith('duplicate_text_of:') for f in r['flags']) for r in rows),
                    'rows': len(rows), 'characters': sum(len(r['text']) for r in rows),
                    'splits': {s: sum(r['split'] == s for r in rows) for s in ('train', 'valid', 'test')},
                    'chars_by_kind': {k: sum(len(r['text']) for r in rows if r['kind'] == k) for k in {r['kind'] for r in rows}},
                    'pii_redactions': pii, 'metrics': metrics.snapshot(), 'elapsed_s': round(time.monotonic() - started, 1),
                    'workers': workers, 'interval_s': client.limiter.interval}
        s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{VERSION}/shard-{shard:05d}.json',
                      Body=json.dumps(manifest, ensure_ascii=False).encode())
        metrics.add('s3_s', time.monotonic() - t)
        done_n = begin + len(chunk)
        rate = done_n / max(1e-9, time.monotonic() - started)
        print(json.dumps({'shard': shard, 'rows': len(rows), 'failures': len(failures), 'done': done_n, 'todo': len(todo),
                          'meetings_per_s': round(rate, 3), 'eta_h': round((len(todo) - done_n) / max(rate, 1e-9) / 3600, 2),
                          'interval_s': round(client.limiter.interval, 2), 'metrics': metrics.snapshot()}), flush=True)
        shard += 1


def aggregate(s3, index):
    manifests = [read_json_s3(s3, k, {}) for k in list_keys(s3, f'corpus/state/{SOURCE}/{VERSION}/shard-')]
    for m in manifests:
        head = s3.head_object(Bucket=BUCKET, Key=m['key'], ChecksumMode='ENABLED')
        if base64.b64decode(head.get('ChecksumSHA256', '')).hex() != m['sha256'] or head['ContentLength'] != m['bytes']:
            raise RuntimeError(f"shard verification failed: {m['key']}")
    meetings = read_json_s3(s3, f'corpus/state/{SOURCE}/{index}', {'meetings': {}, 'done': []})
    chars = sum(m['characters'] for m in manifests)
    sums = lambda field: {k: sum(m[field].get(k, 0) for m in manifests) for k in {k for m in manifests for k in m[field]}}
    stats = {'source': SOURCE, 'clean_version': VERSION, 'meetings_indexed': len(meetings['meetings']),
             'index_pairs_done': len(meetings['done']), 'meetings_fetched': sum(len(m['ids']) for m in manifests),
             'rows': sum(m['rows'] for m in manifests), 'characters': chars, 'estimated_tokens': round(chars / 1.6),
             'estimate_note': 'chars / 1.6, not a tokenizer count', 'chars_by_kind': sums('chars_by_kind'),
             'splits': sums('splits'), 'pii_redactions': sums('pii_redactions'), 'shards': len(manifests)}
    provenance = {'source': SOURCE, 'clean_version': VERSION, 'site': 'https://record.assembly.go.kr/assembly/',
                  'license': LICENSE, 'license_evidence': ['https://record.assembly.go.kr/assembly/ footer: 공공누리 제1유형(출처표시)',
                                                           'https://open.assembly.go.kr/portal/policy/copyRightPage.do'],
                  'collector': 'scripts/collect_assembly_minutes.py', 'terms': TERMS, 'classes': CLASSES,
                  'request_policy': 'one host-wide request-start limiter (default 1 s), adaptive slow-down on 429/5xx/Retry-After/latency; no IP or account spreading',
                  'extraction': 'viewer HTML from the document <h1> heading; tags stripped; viewer controls removed; ko-pii redaction'}
    prefix = f'corpus/clean/{SOURCE}/{VERSION}'
    for name, body in (('stats.json', json.dumps(stats, ensure_ascii=False, indent=2)),
                       ('provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2)),
                       ('ATTRIBUTION.txt', '출처: 국회사무처 국회회의록 (https://record.assembly.go.kr), 공공누리 제1유형(출처표시). '
                                           '각 행의 source URL이 원문 회의록입니다.\n')):
        s3.put_object(Bucket=BUCKET, Key=f'{prefix}/{name}', Body=body.encode())
    if stats['meetings_fetched'] >= stats['meetings_indexed'] and stats['index_pairs_done'] == len(TERMS) * len(CLASSES):
        s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{VERSION}/_DONE', Body=json.dumps(stats).encode())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['discover', 'fetch', 'aggregate'])
    p.add_argument('--index', default='index.json', help='state file name for the resumable meeting index')
    p.add_argument('--terms', nargs='*', default=TERMS)
    p.add_argument('--classes', nargs='*', default=list(CLASSES))
    p.add_argument('--interval', type=float, default=1.0, help='host-wide minimum seconds between request starts')
    p.add_argument('--workers', type=int, default=4, help='concurrent fetches sharing the host limiter')
    p.add_argument('--per-shard', type=int, default=200)
    p.add_argument('--limit', type=int, help='fetch at most this many meetings (pilot)')
    p.add_argument('--seed', help='fetch order: seeded, alternating meeting classes (default: ascending id)')
    args = p.parse_args()
    s3, metrics = s3_client(), Metrics()
    client = Client(HostLimiter(args.interval))
    if args.action == 'discover':
        discover(client, s3, args.terms, args.classes, args.index, metrics)
    elif args.action == 'fetch':
        fetch(client, s3, args.index, args.per_shard, args.workers, args.limit, metrics, args.seed)
    else:
        aggregate(s3, args.index)


if __name__ == '__main__':
    main()

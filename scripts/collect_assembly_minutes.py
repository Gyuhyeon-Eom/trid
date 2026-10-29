"""National Assembly minutes (국회회의록, record.assembly.go.kr) -> trid corpus contract shards in S3.

  discover  walk the public meeting tree of every 대수 and meeting class (본회의, 상임위원회, 특별위원회,
            예산결산특별위원회, 국정감사, 국정조사) through the site's own list endpoints (mnts/async/*.do, the same
            POSTs its list.js issues) and store the meeting index in corpus/state/assembly-minutes/meetings.jsonl.
  fetch     download each meeting's text viewer page (viewer/minutes/xml.do?id=N&type=view), keep the original
            HTML gzip-compressed under corpus/raw/assembly-minutes/, extract the minutes text and write gzip JSONL
            shards to corpus/clean/assembly-minutes/. A per-shard manifest in corpus/state/ lists the ids it holds, so
            re-running skips done meetings and never duplicates rows.
  aggregate write stats.json, provenance.json and ATTRIBUTION.txt.

Licence: the minutes site carries 공공누리 제1유형 (출처표시) and the Assembly's copyright policy
(https://open.assembly.go.kr/portal/policy/copyRightPage.do) declares Assembly-authored works free to use under
저작권법 제24조의2. One request per second; the site has no robots.txt (404).
"""
import argparse
import datetime
import gzip
import hashlib
import html
import io
import json
import re
import time
import unicodedata
from html.parser import HTMLParser
from urllib.parse import urlencode

BUCKET = 'trid-training-148761639313'
SOURCE = 'assembly-minutes'
BASE = 'https://record.assembly.go.kr/assembly'
UA = 'Mozilla/5.0 (compatible; trid-collector/0.1; +public-sector research corpus)'
LICENSE = 'KOGL-Type1'
CLASSES = {'1': 'assembly_plenary', '2': 'assembly_committee', '3': 'assembly_special_committee',
           '4': 'assembly_budget_committee', '5': 'assembly_audit', '6': 'assembly_investigation'}
TERMS = [str(n) for n in range(1, 23)] + ['10.5']
MINUTE_LINK = re.compile(r'viewer/minutes/xml\.do\?id=(\d+)&(?:amp;)?type=view')
HEADER = re.compile(r'제(\d+(?:\.\d+)?)대국회\s*제(\d+)회\s*\(([^)]*)\)\s*제(\d+)차\s*(.+?)\s*\((\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\)')
UI_TAIL = {'top', 'down', '발언 저장', '발언 인쇄'}


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


def row_of(meeting, page, collected_at):
    meta, text = extract(page)
    title = (f"제{meta['term']}대국회 제{meta['session']}회 제{meta['order']}차 {meta['meeting']}"
             if meta else meeting.get('label', ''))
    url = f"{BASE}/viewer/minutes/xml.do?id={meeting['id']}&type=view"
    return {'id': f"{SOURCE}:{meeting['id']}", 'group': f"{SOURCE}:{meeting['id']}",
            'kind': CLASSES.get(str(meeting.get('class')), 'assembly_minutes'), 'title': title, 'text': text,
            'source': url, 'license': LICENSE, 'attribution': f'출처: 국회사무처 국회회의록, {url}',
            'date': meta.get('date'), 'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'collected_at': collected_at, 'extractor': 'html:viewer',
            'meta': {**{k: v for k, v in meta.items() if k != 'date'}, 'minutes_id': meeting['id'],
                     'class_id': meeting.get('class'), 'list_term': meeting.get('th')}}


class Client:
    def __init__(self, delay=1.0):
        import requests
        self.session, self.delay, self.last = requests.Session(), delay, 0.0
        self.session.headers.update({'User-Agent': UA})

    def _wait(self):
        pause = self.delay - (time.time() - self.last)
        if pause > 0:
            time.sleep(pause)
        self.last = time.time()

    def get(self, url, **kw):
        return self._retry(lambda: self.session.get(url, timeout=120, **kw))

    def post(self, url, data, referer):
        return self._retry(lambda: self.session.post(url, data=data, timeout=120, headers={
            'X-Requested-With': 'XMLHttpRequest', 'Referer': referer}))

    def _retry(self, call):
        for attempt in range(6):
            self._wait()
            try:
                r = call()
                if r.status_code < 500 and r.status_code != 429:
                    return r
            except Exception as exc:  # network errors: back off and retry
                print(f'retry after {exc!r}', flush=True)
            time.sleep(min(300, 10 * 2 ** attempt))
        raise RuntimeError('request failed after retries')


def s3_client():
    import boto3
    return boto3.client('s3')


def read_jsonl_s3(s3, key):
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)['Body'].read().decode()
    except s3.exceptions.NoSuchKey:
        return []
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def discover(client, s3, terms=TERMS, classes=tuple(CLASSES), state_key='meetings.jsonl'):
    seen_requests, found = set(), {}
    for term in terms:
        for cls in classes:
            list_url = f'{BASE}/mnts/total/{term}.do?' + urlencode({'class_id_sch': cls, 'cmit_chk': 'all'})
            page = client.get(list_url).text
            queue = [(kind, data) for kind, data in nodes_of(page)]
            for mid in minute_ids(page):
                found.setdefault(mid, {'id': mid, 'th': term, 'class': cls})
            while queue:
                kind, data = queue.pop()
                data.setdefault('th', term)
                data.setdefault('class', cls)
                endpoint, form = payload(kind, data, term)
                key = (endpoint, tuple(sorted(form.items())))
                if key in seen_requests:
                    continue
                seen_requests.add(key)
                sub = client.post(BASE + endpoint, form, list_url).text
                for mid in minute_ids(sub):
                    found.setdefault(mid, {'id': mid, 'th': term, 'class': cls, 'sess': data.get('sess', '')})
                queue.extend(nodes_of(sub))
            print(f'term {term} class {cls}: {len(found)} meetings so far, {len(seen_requests)} list requests', flush=True)
            s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/{state_key}',
                          Body=''.join(json.dumps(m) + '\n' for m in sorted(found.values(), key=lambda m: m['id'])).encode())
    return list(found.values())


def done_ids(s3):
    ids = set()
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=f'corpus/state/{SOURCE}/shard-'):
        for item in page.get('Contents', []):
            ids.update(json.loads(s3.get_object(Bucket=BUCKET, Key=item['Key'])['Body'].read())['ids'])
    return ids


def fetch(client, s3, per_shard, index='meetings.jsonl', limit=None):
    meetings = read_jsonl_s3(s3, f'corpus/state/{SOURCE}/{index}')
    done = done_ids(s3)
    todo = [m for m in meetings if m['id'] not in done][:limit]
    print(f'{len(meetings)} meetings indexed, {len(done)} done, {len(todo)} to fetch', flush=True)
    shard = 1 + max([-1] + [int(k.split('-')[-1].split('.')[0]) for k in list_keys(s3, f'corpus/state/{SOURCE}/shard-')])
    rows, ids = [], []
    for n, meeting in enumerate(todo, 1):
        r = client.get(f"{BASE}/viewer/minutes/xml.do?id={meeting['id']}&type=view")
        if r.status_code != 200 or '회의록' not in r.text:
            print(f"skip {meeting['id']}: HTTP {r.status_code}", flush=True)
            continue
        s3.put_object(Bucket=BUCKET, Key=f"corpus/raw/{SOURCE}/{meeting['id']}.html.gz",
                      Body=gzip.compress(r.content, mtime=0))
        row = row_of(meeting, r.text, datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
        if row['text']:
            rows.append(row)
        ids.append(meeting['id'])
        if len(ids) >= per_shard or n == len(todo):
            write_shard(s3, shard, rows, ids)
            print(f'shard {shard}: {len(rows)} rows ({n}/{len(todo)})', flush=True)
            shard, rows, ids = shard + 1, [], []


def list_keys(s3, prefix):
    keys = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [item['Key'] for item in page.get('Contents', [])]
    return keys


def write_shard(s3, shard, rows, ids):
    body = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(), mtime=0)
    s3.put_object(Bucket=BUCKET, Key=f'corpus/clean/{SOURCE}/docs-{shard:05d}.jsonl.gz', Body=body)
    s3.put_object(Bucket=BUCKET, Key=f'corpus/state/{SOURCE}/shard-{shard:05d}.json', Body=json.dumps({
        'ids': ids, 'rows': len(rows), 'characters': sum(len(r['text']) for r in rows),
        'chars_by_kind': {k: sum(len(r['text']) for r in rows if r['kind'] == k) for k in {r['kind'] for r in rows}},
        'years': {y: sum(1 for r in rows if (r['date'] or '????')[:4] == y) for y in {(r['date'] or '????')[:4] for r in rows}}}).encode())


def aggregate(s3):
    manifests = [json.loads(s3.get_object(Bucket=BUCKET, Key=k)['Body'].read())
                 for k in list_keys(s3, f'corpus/state/{SOURCE}/shard-')]
    meetings = read_jsonl_s3(s3, f'corpus/state/{SOURCE}/meetings.jsonl')
    chars = sum(m['characters'] for m in manifests)
    by_kind = {}
    for m in manifests:
        for k, v in m['chars_by_kind'].items():
            by_kind[k] = by_kind.get(k, 0) + v
    stats = {'source': SOURCE, 'meetings_indexed': len(meetings), 'meetings_fetched': sum(len(m['ids']) for m in manifests),
             'documents': sum(m['rows'] for m in manifests), 'characters': chars,
             'estimated_tokens': round(chars / 1.6), 'estimate_note': 'chars / 1.6, not a tokenizer count',
             'chars_by_kind': by_kind, 'shards': len(manifests)}
    provenance = {'source': SOURCE, 'site': 'https://record.assembly.go.kr/assembly/', 'license': LICENSE,
                  'license_evidence': ['https://record.assembly.go.kr/assembly/ (footer: 공공누리 제1유형 출처표시)',
                                       'https://open.assembly.go.kr/portal/policy/copyRightPage.do'],
                  'collector': 'scripts/collect_assembly_minutes.py', 'terms': TERMS, 'classes': CLASSES,
                  'extraction': 'viewer HTML from the document <h1> heading; tags stripped, block tags as newlines, '
                                'viewer controls removed; speaker labels kept', 'politeness': '1 request/second'}
    for name, body in (('stats.json', json.dumps(stats, ensure_ascii=False, indent=2)),
                       ('provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2)),
                       ('ATTRIBUTION.txt', '출처: 국회사무처 국회회의록 (https://record.assembly.go.kr), 공공누리 제1유형(출처표시). '
                                           '각 행의 source URL이 원문 회의록입니다.\n')):
        s3.put_object(Bucket=BUCKET, Key=f'corpus/clean/{SOURCE}/{name}', Body=body.encode())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['discover', 'fetch', 'aggregate', 'all'])
    p.add_argument('--per-shard', type=int, default=300)
    p.add_argument('--delay', type=float, default=1.0)
    p.add_argument('--terms', nargs='*', default=TERMS)
    p.add_argument('--classes', nargs='*', default=list(CLASSES))
    p.add_argument('--limit', type=int, help='fetch at most this many meetings (pilot)')
    p.add_argument('--index', default='meetings.jsonl', help='state file name for the meeting index')
    args = p.parse_args()
    s3 = s3_client()
    if args.action in ('discover', 'all'):
        discover(Client(args.delay), s3, args.terms, args.classes, args.index)
    if args.action in ('fetch', 'all'):
        fetch(Client(args.delay), s3, args.per_shard, args.index, args.limit)
    if args.action in ('aggregate', 'all'):
        aggregate(s3)


if __name__ == '__main__':
    main()

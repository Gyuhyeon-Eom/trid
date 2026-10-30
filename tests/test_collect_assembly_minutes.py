"""Assembly minutes: list-tree parsing must reproduce the site's own requests; text must start at the minutes body."""
import threading
import time
import unittest
from scripts.collect_assembly_minutes import extract, minute_ids, nodes_of, payload, row_of

LIST = '''<ul class="tree_list"><li><div class="tit_wrap"><div class="txt">
<a href="javascript:void(0);" class="tit sess" data-id="" data-sess="430" data-th="22" data-class="1"
   data-chk1="" data-chk2="" data-chk3="" data-chk4="" data-year=>제430회</a>
<button type="button" class="btn_tit cmit" data-th="22" data-class="2" data-sess="" data-cl="NAM" data-cmit="9700005">법사위</button>
<a class="other" data-sess="1">x</a></div></div></li></ul>
<a href="/assembly/viewer/minutes/xml.do?id=55958&type=view">보기</a>
<a href="/assembly/viewer/minutes/xml.do?id=55982&amp;type=view">보기</a>
<a href="/assembly/viewer/minutes/xml.do?id=55958&type=summary">요약</a>'''

VIEWER = '''<html><head><title>국회회의록 상세보기 팝업창</title><script>var a = "<h1>가짜회의록</h1>";</script></head><body>
<div class="nav">국회회의록 NATIONAL ASSEMBLY MINUTES 제22대국회 제430회 (임시회) 제1차 국회본회의 (2025.12.11.)</div>
<div><h1>국회본회의회의록</h1></div><div>제1호</div><div>일시</div><div>2025년12월11일(목) 오후 2시</div>
<div><span class="name">의장 우원식 선택</span></div><div>의장 우원식 (서울 노원구갑)</div>
<p>성원이 되었으므로 회의를 개의하겠습니다.</p><button>발언 저장</button>
<div>top</div><div>Scroll : 0 %</div><div>down</div></body></html>'''


class AssemblyMinutesTests(unittest.TestCase):
    def test_list_nodes_map_to_the_sites_requests(self):
        nodes = nodes_of(LIST)
        self.assertEqual([k for k, _ in nodes], ['sess', 'cmit'])
        endpoint, form = payload(*nodes[0], '22')
        self.assertEqual(endpoint, '/mnts/async/sess.do')
        self.assertEqual((form['th_sch'], form['class_id_sch'], form['sess_sch']), ('22', '1', '430'))
        endpoint, form = payload(*nodes[1], '22')
        self.assertEqual(endpoint, '/mnts/async/sessCmit.do')
        self.assertEqual((form['council_cd_sch'], form['cmit_id_sch']), ('NAM', '9700005'))
        self.assertEqual(minute_ids(LIST), [55958, 55982])

    def test_text_starts_at_minutes_heading_without_viewer_controls(self):
        meta, text = extract(VIEWER)
        self.assertEqual(meta['date'], '2025-12-11')
        self.assertEqual((meta['term'], meta['session'], meta['order'], meta['meeting']), ('22', 430, 1, '국회본회의'))
        lines = text.splitlines()
        self.assertEqual(lines[0], '국회본회의회의록')
        self.assertIn('의장 우원식 (서울 노원구갑)', lines)
        self.assertNotIn('의장 우원식 선택', lines)
        for ui in ('top', 'down', '발언 저장', '가짜회의록'):
            self.assertNotIn(ui, text)
        self.assertFalse(any(line.startswith('Scroll') for line in lines))

    def test_row_schema(self):
        row, pii = row_of({'id': 55958, 'class': '1', 'th': '22'}, VIEWER.replace('개의하겠습니다.', '개의하겠습니다. 문의 02-123-4567'),
                          '2026-10-30T00:00:00+00:00')
        self.assertNotIn('02-123-4567', row['text'])
        self.assertEqual(dict(pii), {'PHONE': 1})
        self.assertEqual(row['provenance']['institution'], '국회')
        self.assertIn(row['split'], ('train', 'valid', 'test'))
        self.assertEqual(row['id'], 'assembly-minutes:55958')
        self.assertEqual(row['kind'], 'assembly_plenary')
        self.assertEqual(row['license'], 'KOGL-Type1')
        self.assertEqual(row['title'], '제22대국회 제430회 제1차 국회본회의')
        self.assertTrue(row['source'].endswith('xml.do?id=55958&type=view'))
        self.assertFalse(any(f.startswith('quarantine:') for f in row['flags']))

    def test_private_persons_and_missing_metadata_quarantine_the_meeting(self):
        petition = VIEWER.replace('개의하겠습니다.', '개의하겠습니다. 청원(홍길동 외 52,585인 국민동의로 제출)')
        row, _ = row_of({'id': 1, 'class': '2'}, petition, 't')
        self.assertIn('quarantine:private_person_names', row['flags'])
        witness = VIEWER.replace('<p>성원이', '<div>증인 김철수</div><p>성원이')
        self.assertIn('quarantine:private_person_names', row_of({'id': 2, 'class': '5'}, witness, 't')[0]['flags'])
        headerless = VIEWER.replace('제22대국회 제430회 (임시회) 제1차 국회본회의 (2025.12.11.)', '개회식')
        row, _ = row_of({'id': 3, 'class': '1'}, headerless, 't')
        self.assertIn('quarantine:metadata_incomplete', row['flags'])
        official = {'title': '제418회 국회(정기회) 개회식', 'date': '2024-09-02', 'source': 'list:/mnts/async/sess.do?sess=418'}
        row, _ = row_of({'id': 3, 'class': '1'}, headerless, 't', official)
        self.assertEqual((row['title'], row['date'], row['meta']['metadata_source']), (official['title'], '2024-09-02', official['source']))
        self.assertFalse(any(f.startswith('quarantine:') for f in row['flags']))

    def test_audit_minutes_header_is_parsed_from_the_document(self):
        audit = VIEWER.replace('제22대국회 제430회 (임시회) 제1차 국회본회의 (2025.12.11.)', '제22대국회 2024년도 국정감사 문화체육관광위원회 (2024.10.24.)')
        row, _ = row_of({'id': 52032, 'class': '5'}, audit, 't')
        self.assertEqual((row['title'], row['date'], row['kind']), ('제22대국회 2024년도 국정감사 문화체육관광위원회', '2024-10-24', 'assembly_audit'))
        self.assertNotIn('quarantine:metadata_incomplete', row['flags'])
        ceremony = VIEWER.replace('제22대국회 제430회 (임시회) 제1차 국회본회의 (2025.12.11.)', '제22대국회 제429회 (정기회) 개회식 국회본회의 (2025.09.01.)')
        row, _ = row_of({'id': 55273, 'class': '1'}, ceremony, 't')
        self.assertEqual((row['title'], row['date']), ('제22대국회 제429회 개회식 국회본회의', '2025-09-01'))

    def test_joint_meeting_heading_without_minutes_word_starts_the_text(self):
        joint = VIEWER.replace('<h1>국회본회의회의록</h1></div><div>제1호</div>',
                               '<h1>교육<div class="tit_s"><span>보건복지위원회와 </span><span>연석회의</span></div>위원회</h1>'
                               '<p class="tit_sm">(의학교육소위원회)</p></div></div><p class="num">제1호</p>')
        meta, text = extract(joint)
        self.assertEqual(meta['date'], '2025-12-11')
        self.assertTrue(text.startswith('교육'), text[:40])
        for ui in ('NATIONAL ASSEMBLY MINUTES', '가짜회의록'):
            self.assertNotIn(ui, text)

    def test_redaction_inside_url_and_listed_overlap_quarantine(self):
        url = VIEWER.replace('개의하겠습니다.', '개의하겠습니다. https://example.go.kr/view?no=02-123-4567')
        row, _ = row_of({'id': 4, 'class': '1'}, url, 't')
        self.assertIn('quarantine:pii_redaction_inside_url', row['flags'])
        row, _ = row_of({'id': 4, 'class': '1'}, VIEWER, 't', listed='reserved_overlap_partial')
        self.assertIn('quarantine:reserved_overlap_partial', row['flags'])

    def test_discover_resumes_the_walk_without_repeating_finished_list_requests(self):
        import json as _json
        from scripts import collect_assembly_minutes as am

        class S3:
            def __init__(self):
                self.objects = {}

            def put_object(self, Bucket, Key, Body):
                self.objects[Key] = Body

            def get_object(self, Bucket, Key):
                import io
                return {'Body': io.BytesIO(self.objects[Key])}

        class Missing(Exception):
            pass
        S3.exceptions = type('E', (), {'NoSuchKey': Missing})

        def node(sess):
            return (f'<a class="tit sess" data-id="" data-sess="{sess}" data-th="22" data-class="1" data-chk1="" '
                    f'data-chk2="" data-chk3="" data-chk4="" data-year="">x</a>')
        root = ''.join(node(s) for s in range(400, 430))

        class Client:
            def __init__(self, fail_after):
                self.calls, self.fail_after = [], fail_after

            def request(self, method, url, metrics, data=None, headers=None):
                if self.fail_after is not None and len(self.calls) >= self.fail_after:
                    raise RuntimeError('stopped')
                self.calls.append(data and data['sess_sch'])
                text = root if method == 'GET' else f'<a href="/assembly/viewer/minutes/xml.do?id={data["sess_sch"]}&type=view">v</a>'
                return type('R', (), {'text': text})()

        s3 = S3()
        s3.objects['corpus/state/assembly-minutes/idx.json'] = _json.dumps({'done': [], 'meetings': {}}).encode()
        first = Client(fail_after=28)  # 1 list page + 27 session requests, checkpoint after 25
        with self.assertRaises(RuntimeError):
            am.discover(first, s3, ['22'], ['1'], 'idx.json', am.Metrics())
        second = Client(fail_after=None)
        am.discover(second, s3, ['22'], ['1'], 'idx.json', am.Metrics())
        state = _json.loads(s3.objects['corpus/state/assembly-minutes/idx.json'])
        self.assertEqual(state['done'], ['22/1'])
        self.assertEqual(len(state['meetings']), 30)
        self.assertNotIn(None, second.calls)  # the root list page was not requested again
        self.assertEqual(len(second.calls), 30 - 25)  # only sessions after the last checkpoint

    def test_requests_reach_the_server_spaced_across_processes_threads_late_wakeups_and_stalls(self):
        import multiprocessing
        import tempfile
        with _Server() as server, tempfile.TemporaryDirectory() as d:
            with multiprocessing.get_context('spawn').Pool(2) as pool:
                logged = pool.starmap(_burst, [(d, server.url, 6, 3, False), (d, server.url, 6, 1, True)])
            arrivals = sorted(server.arrivals)
        starts = sorted(t for batch in logged for t in batch)
        self.assertEqual(len(arrivals), 12)
        # Server-side arrival times: no epsilon. A request is sent after its start and before its end, and the next
        # start waits for that end + interval, so even late wake-ups and pre-send stalls cannot close the gap.
        self.assertGreaterEqual(min(b - a for a, b in zip(arrivals, arrivals[1:])), INTERVAL, arrivals)
        self.assertGreaterEqual(min(b - a for a, b in zip(starts, starts[1:])), INTERVAL, starts)

    def test_retry_after_pushes_the_next_request_out(self):
        import tempfile
        from scripts import collect_assembly_minutes as am
        with _Server(first_status=429) as server, tempfile.TemporaryDirectory() as d:
            client = am.Client(am.HostLimiter(INTERVAL, host='retry-after', shared_dir=d))
            self.assertEqual(client.request('GET', server.url, am.Metrics()).status_code, 200)  # retried after the 429
            arrivals = sorted(server.arrivals)
        self.assertEqual(len(arrivals), 2)
        self.assertGreaterEqual(arrivals[1] - arrivals[0], 1.0)  # Retry-After: 1

    def test_new_clean_version_never_downloads_or_replaces_stored_raw(self):
        import gzip
        import json as _json
        from unittest import mock
        from scripts import collect_assembly_minutes as am
        s3 = _FakeS3()
        meetings = {str(i): {'id': i, 'class': '2', 'th': '22'} for i in (1, 2, 3)}
        s3.objects['corpus/state/assembly-minutes/idx.json'] = _json.dumps({'done': [], 'meetings': meetings}).encode()
        stored = gzip.compress(b'<h1>stored minutes</h1>', mtime=0)
        am.put_new(s3, 'corpus/raw/assembly-minutes/1.html.gz', stored)  # collected under an older clean version
        s3.objects['corpus/state/assembly-minutes/clean-v1/shard-00000.json'] = _json.dumps(
            {'ids': [1], 'failures': {'2': 'http_400'}}).encode()
        client = _FakeClient()
        with mock.patch.object(am, 'VERSION', 'clean-v99'):
            am.fetch(client, s3, 'idx.json', 10, 1, None, am.Metrics())
        self.assertEqual(client.ids, [3])  # 1 has raw (any version), 2 got HTTP 400 before: no request for either
        self.assertEqual(s3.objects['corpus/raw/assembly-minutes/1.html.gz'], stored)
        self.assertIn('corpus/raw/assembly-minutes/3.html.gz', s3.objects)
        # A direct write of different bytes to a stored raw key is refused and leaves the bytes alone.
        with self.assertRaises(am.RawExists):
            am.put_new(s3, 'corpus/raw/assembly-minutes/1.html.gz', b'other')
        self.assertEqual(s3.objects['corpus/raw/assembly-minutes/1.html.gz'], stored)
        self.assertEqual(am.put_new(s3, 'corpus/raw/assembly-minutes/1.html.gz', stored), __import__('hashlib').sha256(stored).hexdigest())
        sha, key = am.put_raw_keep(s3, 'corpus/raw/x/9.html.gz', b'a')
        sha2, key2 = am.put_raw_keep(s3, 'corpus/raw/x/9.html.gz', b'b')  # deliberate re-capture: both kept
        self.assertEqual((key, s3.objects[key]), ('corpus/raw/x/9.html.gz', b'a'))
        self.assertEqual((key2, s3.objects[key2]), (f'corpus/raw/x/9.{sha2[:16]}.html.gz', b'b'))


INTERVAL = 0.15


class _Server:
    """Local HTTP server that records when each request arrives (handler entry) and answers after a random delay."""

    def __init__(self, first_status=200):
        self.arrivals, self.first_status = [], first_status

    def __enter__(self):
        import http.server
        import random
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.arrivals.append(time.time())
                time.sleep(random.uniform(0, 0.05))
                status = outer.first_status if len(outer.arrivals) == 1 else 200
                self.send_response(status)
                if status == 429:
                    self.send_header('Retry-After', '1')
                self.send_header('Content-Length', '2')
                self.end_headers()
                self.wfile.write(b'ok')

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _burst(shared_dir, url, n, threads, misbehave):
    """One process: n requests over `threads` threads. misbehave: every sleep overshoots by up to 120 ms and every
    request stalls up to 200 ms between its start and the actual send."""
    import random
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from scripts import collect_assembly_minutes as am
    if misbehave:
        real_sleep = time.sleep
        am.time.sleep = lambda s: real_sleep(s + random.uniform(0, 0.12))
    log = Path(tempfile.mkdtemp()) / 'requests.jsonl'
    client = am.Client(am.HostLimiter(INTERVAL, host='local-proof', shared_dir=shared_dir), log)
    if misbehave:
        send = client.session.request

        def stalled(*a, **kw):
            time.sleep(random.uniform(0, 0.2))
            return send(*a, **kw)
        client.session.request = stalled
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda _: client.request('GET', url, am.Metrics()), range(n)))
    import json as _json
    return [_json.loads(line)['t'] for line in log.read_text().split('\n') if line]


class _FakeS3:
    """Enough of S3 for fetch: conditional put (If-None-Match), checksums, listing."""

    class _Missing(Exception):
        pass
    exceptions = type('E', (), {'NoSuchKey': _Missing})

    def __init__(self):
        self.objects, self.checksums = {}, {}

    def put_object(self, Bucket, Key, Body, ChecksumSHA256=None, IfNoneMatch=None):
        if IfNoneMatch == '*' and Key in self.objects:
            err = Exception('PreconditionFailed')
            err.response = {'Error': {'Code': 'PreconditionFailed'}}
            raise err
        self.objects[Key], self.checksums[Key] = Body, ChecksumSHA256

    def get_object(self, Bucket, Key):
        import io
        if Key not in self.objects:
            raise self._Missing(Key)
        return {'Body': io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key, ChecksumMode=None):
        return {'ChecksumSHA256': self.checksums[Key], 'ContentLength': len(self.objects[Key])}

    def get_paginator(self, name):
        objects = self.objects
        return type('P', (), {'paginate': lambda self, Bucket, Prefix: [
            {'Contents': [{'Key': k} for k in sorted(objects) if k.startswith(Prefix)]}]})()


class _FakeClient:
    limiter = type('L', (), {'interval': 1.0})()

    def __init__(self):
        self.ids = []

    def request(self, method, url, metrics, **kw):
        self.ids.append(int(url.split('id=')[1].split('&')[0]))
        return type('R', (), {'status_code': 200, 'text': VIEWER, 'content': VIEWER.encode()})()


if __name__ == '__main__':
    unittest.main()

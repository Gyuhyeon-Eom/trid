"""Assembly minutes: list-tree parsing must reproduce the site's own requests; text must start at the minutes body."""
import threading
import time
import unittest
from scripts.collect_assembly_minutes import HostLimiter, extract, minute_ids, nodes_of, payload, row_of

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
        self.assertEqual((row['clean_version'], row['provenance']['institution']), ('clean-v4', '국회'))
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

    def test_host_limiter_spaces_starts_across_threads_and_backs_off(self):
        limiter, waits = HostLimiter(0.05), []
        started = time.monotonic()
        threads = [threading.Thread(target=lambda: waits.append(limiter.acquire())) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Four starts share one host interval: the last one cannot begin before 3 intervals have passed.
        self.assertGreaterEqual(time.monotonic() - started, 0.15 - 0.01)
        limiter.observe(429, 0.1, retry_after=0.2)
        self.assertGreaterEqual(limiter.interval, 0.2)
        for _ in range(200):
            limiter.observe(200, 0.1)
        self.assertAlmostEqual(limiter.interval, 0.05)

    def test_shared_limiter_spaces_reservations_across_processes(self):
        import multiprocessing
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with multiprocessing.get_context('spawn').Pool(3) as pool:
                starts = sorted(s for batch in pool.starmap(_reserve_n, [(d, 4)] * 3) for s in batch)
        self.assertEqual(len(starts), 12)
        self.assertTrue(all(b - a >= 0.1 - 1e-6 for a, b in zip(starts, starts[1:])), starts)  # epoch-float rounding


def _reserve_n(shared_dir, n):
    limiter = HostLimiter(0.1, host='proof.example', shared_dir=shared_dir)
    return [limiter._reserve()[0] for _ in range(n)]


if __name__ == '__main__':
    unittest.main()

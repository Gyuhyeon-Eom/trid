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
        self.assertEqual((row['clean_version'], row['provenance']['institution']), ('clean-v1', '국회'))
        self.assertIn(row['split'], ('train', 'valid', 'test'))
        self.assertEqual(row['id'], 'assembly-minutes:55958')
        self.assertEqual(row['kind'], 'assembly_plenary')
        self.assertEqual(row['license'], 'KOGL-Type1')
        self.assertEqual(row['title'], '제22대국회 제430회 제1차 국회본회의')
        self.assertTrue(row['source'].endswith('xml.do?id=55958&type=view'))

    def test_host_limiter_spaces_starts_across_threads_and_backs_off(self):
        limiter, starts = HostLimiter(0.05), []
        threads = [threading.Thread(target=lambda: (limiter.acquire(), starts.append(time.monotonic()))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        starts.sort()
        self.assertTrue(all(b - a >= 0.045 for a, b in zip(starts, starts[1:])))
        limiter.observe(429, 0.1, retry_after=0.2)
        self.assertGreaterEqual(limiter.interval, 0.2)
        for _ in range(200):
            limiter.observe(200, 0.1)
        self.assertAlmostEqual(limiter.interval, 0.05)


if __name__ == '__main__':
    unittest.main()

"""임원 업무추진비 집행내역: facts only for sections both readers agree on and whose rows add up (B-028)."""
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts import build_institution_rule_snapshot as b

KORDOC = """# ‘26년 3월 원장 업무추진비 집행내역
<table>
<tr><th>사용일자</th><th>내 역</th><th>사용처</th><th>대 상</th><th>인원(명)</th><th>금액(원)</th></tr>
<tr><td>2026-03-11</td><td>유관기관 업무 간담회</td><td>가게<br>선릉점</td><td>유관기관 관계자</td><td>7</td><td>100,000</td></tr>
<tr><td>2026-03-12</td><td>유관기관 업무 간담회</td><td>식당</td><td>학계 관계자</td><td>3</td><td>50,000</td></tr>
<tr><td colspan="3"></td><td colspan="2">합 계</td><td>150,000</td></tr>
</table>
# ‘26년 3월 감사 업무추진비 집행내역
<table>
<tr><td>2026-03-20</td><td>유관기관 업무 간담회</td><td>식당</td><td>학계 관계자</td><td>3</td><td>50,000</td></tr>
<tr><td colspan="3"></td><td colspan="2">합 계</td><td>60,000</td></tr>
</table>
"""
LAYOUT = """  ‘26년 3월 원장 업무추진비 집행내역
2026-03-11   유관기관 업무 간담회   가게   유관기관 관계자   7   100,000
                                    선릉점
2026-03-12   유관기관 업무 간담회   식당   학계 관계자   3   50,000
                         합 계    150,000
  ‘26년 3월 감사 업무추진비 집행내역
2026-03-20   유관기관 업무 간담회   식당   학계 관계자   3   50,000
                         합 계    60,000
"""


class SpendCheckTests(unittest.TestCase):
    def sections(self, layout=LAYOUT):
        with mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=layout)):
            return b.spend_sections(KORDOC, 'x.pdf')

    def test_only_sections_that_add_up_and_agree_become_facts(self):
        s = self.sections()
        self.assertEqual([(x['position'], x['verified']) for x in s], [('원장', True), ('감사', False)])
        self.assertIn('rows_do_not_add_up_to_total', s[1]['problems'])
        facts = b.spend_facts(s, {'apbaId': 'C0150', 'apbaNa': '한국부동산원'}, 'u', 'h')
        self.assertEqual([(f['value'], f['population'], f['period']['to']) for f in facts],
                         [(150000, '임원(원장) 1명 분', '2026-03-31')])

    def test_reader_disagreement_withholds_the_fact(self):
        s = self.sections(LAYOUT.replace('7   100,000', '8   100,000'))
        self.assertEqual(s[0]['problems'], ['rows_differ_between_readers'])
        self.assertEqual(b.spend_facts(s, None, 'u', 'h'), [])

    def test_model_text_leaves_out_the_venue(self):
        text = b.spend_view(self.sections(), 't')
        self.assertNotIn('가게', text)
        self.assertNotIn('선릉점', text)
        self.assertIn('| 2026-03-11 | 유관기관 업무 간담회 | 유관기관 관계자 | 7 | 100,000 |', text)


if __name__ == '__main__':
    unittest.main()

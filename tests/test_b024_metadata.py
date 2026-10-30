"""Who a rule applies to is only stated from evidence, and long tables are cut by whole first-column units."""
import unittest

from scripts.build_api_snapshot import applies_to, statute_applies_to
from scripts.build_structural_context_candidates import units


class AppliesToTests(unittest.TestCase):
    def test_title_naming_the_issuer_scopes_the_rule_to_that_agency(self):
        got = applies_to('농림축산검역본부 공무국외출장관리규정', {'id': '1543061', 'name': '농림축산검역본부'})
        self.assertEqual((got['scope'], got['ids']), ('institution', ['1543061']))

    def test_rule_for_other_people_stays_unknown(self):
        # 병무청's travel-expense rule is for 병역의무자, not 병무청 staff and not every public servant
        got = applies_to('병역의무자 여비지급 규정', {'id': '1300000', 'name': '병무청'})
        self.assertEqual(got['scope'], 'unknown')

    def test_local_government_rule(self):
        got = applies_to('위법한 공무국외출장에 대한 지방자치단체 처리기준', {'id': '1741000', 'name': '행정안전부'})
        self.assertEqual(got['scope'], 'local_government')

    def test_statute_scope_comes_from_its_purpose_article_only(self):
        self.assertEqual(statute_applies_to('제1조(목적) 이 영은 국가공무원이 공무로 여행을 하는 경우에 지급하는 여비')['scope'],
                         'central_government')
        self.assertEqual(statute_applies_to('제1조(목적) 이 영은 국가공무원과 지방공무원의 …')['scope'], 'unknown')


class UnitTests(unittest.TestCase):
    def test_rowspan_rows_stay_in_their_unit(self):
        body = ('<table><tr><th>단위업무</th><th>세부</th></tr>'
                '<tr><td rowspan="2">1. 계획</td><td>가. 중요</td></tr><tr><td>1) 일반</td></tr>'
                '<tr><td>2. 협의</td><td>가. 협의</td></tr>'
                '<tr><td rowspan="3">3. 예산</td><td>가.</td></tr><tr><td>나.</td></tr><tr><td>다.</td></tr>')
        texts = [body[s:e] for s, e in units(body)]
        self.assertEqual(len(texts), 3)
        self.assertIn('1) 일반', texts[0])  # a sub-row numbered '1)' does not open a new unit
        self.assertEqual(texts[2].count('<tr>'), 3)


if __name__ == '__main__':
    unittest.main()

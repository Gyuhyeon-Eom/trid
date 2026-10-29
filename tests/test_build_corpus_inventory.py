"""Business-group proposals must follow the documented precedence and never hide unmatched documents."""
import unittest
from scripts.build_corpus_inventory import business_group


class BusinessGroupTests(unittest.TestCase):
    def test_title_keywords_take_precedence_in_declared_order(self):
        self.assertEqual(business_group('계약사무규정')[0], '계약·조달')
        self.assertEqual(business_group('여비규정', '국가보훈부')[0], '회계·예산')
        self.assertEqual(business_group('연구윤리 지침')[0], '교육·연구')
        self.assertEqual(business_group('임대주택 공급 규정')[0], '주거·부동산')

    def test_ministry_and_committee_fallbacks_are_recorded(self):
        self.assertEqual(business_group('직원 행동강령', '국가보훈부'), ('보훈·복지', 'ministry:국가보훈부'))
        self.assertEqual(business_group(None, committee='보건복지위원회'), ('보훈·복지', 'committee:보건복지'))

    def test_unmatched_stays_unclassified(self):
        self.assertEqual(business_group('정보공개 처리 지침', '외교부'), ('미분류', 'none'))


if __name__ == '__main__':
    unittest.main()

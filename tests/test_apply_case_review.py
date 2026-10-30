"""Reviewed case targets must still cite real passages and take every number from what they cite."""
import unittest

from scripts.apply_case_review import contract_problems, relabel

CONTEXT = ['자료 정보: 문서 x · 기준 2025-04-07 자 수집본\n제2호 정원의 1/5이상', '제14조 ① 비상근무 중에는 당직근무를 중지한다.']


class ContractTests(unittest.TestCase):
    def test_correct_citations_pass(self):
        self.assertEqual(contract_problems('당직근무를 중지합니다 [D2]. 정원의 1/5 이상입니다 [D1].', CONTEXT, '질문'), [])

    def test_number_cited_to_the_wrong_passage_is_rejected(self):
        # D2's '①' stands for 1, so only the 5 has no source there
        self.assertEqual(contract_problems('정원의 1/5 이상입니다 [D2].', CONTEXT, '질문'), ['number_not_in_cited_passage:5'])

    def test_citation_to_a_missing_passage_is_rejected(self):
        self.assertEqual(contract_problems('중지합니다 [D3].', CONTEXT, '질문'), ['citation_out_of_range:[3]'])

    def test_target_without_any_citation_is_rejected(self):
        self.assertEqual(contract_problems('중지합니다.', CONTEXT, '질문'), ['no_citation'])


class RelabelTests(unittest.TestCase):
    def test_statute_title_is_not_called_an_institution(self):
        row = {'doc': 'law-go-kr:law-byl:1', 'institution': '물품관리법 시행규칙',
               'context': ['자료 정보: 문서 law-go-kr:law-byl:1 · 기관 물품관리법 시행규칙 · 기준 2026-01-02 자 수집본\n본문']}
        context, changed = relabel(row)
        self.assertTrue(changed)
        self.assertIn('· 법령 물품관리법 시행규칙', context[0])

    def test_institution_rows_keep_their_label(self):
        row = {'doc': 'law-go-kr:admrul-file:1', 'institution': '중앙소방학교', 'context': ['자료 정보: · 기관 중앙소방학교\n본문']}
        self.assertEqual(relabel(row), (row['context'], False))


if __name__ == '__main__':
    unittest.main()

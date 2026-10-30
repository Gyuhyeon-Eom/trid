"""Internal-work rule candidates: title topics must not come from agency names or civil-law look-alikes."""
import unittest

from scripts.build_code_basis_links import labels
from scripts.build_internal_rules import topics_of


class InternalRuleTopicTests(unittest.TestCase):
    def test_internal_work_titles_get_their_topic(self):
        self.assertEqual(topics_of('국립중앙과학관 인사관리규정'), ['인사'])
        self.assertEqual(topics_of('남해어업관리단 위임전결규정'), ['결재·문서'])
        self.assertEqual(topics_of('(계약예규) 일괄입찰 등에 의한 낙찰자 결정기준'), ['계약·구매'])
        self.assertIn('계약·구매', topics_of('민원인 - 「국가를 당사자로 하는 계약에 관한 법률」 제7조 관련', 'expc'))

    def test_agency_names_and_look_alikes_do_not_make_a_topic(self):
        self.assertEqual(topics_of('기획예산처 공익신고 처리 및 신고자 보호 등에 관한 규정'), [])
        self.assertEqual(topics_of('조달청 공무원행동강령'), [])
        for title in ('요양비의 보험급여 기준 및 방법', '공익근무요원 복무관리지침', '하천보수원 관리규정',
                      '화학물질의 분류·표시 및 물질안전보건자료에 관한 기준', '수입금지식물 중 생과실의 수입금지 제외기준'):
            self.assertEqual(topics_of(title), [], title)
        self.assertEqual(topics_of('민원인 - 임대사업자가 새로운 임차인과 임대차계약을 체결하는 경우', 'expc'), [])

    def test_value_labels_are_read_from_the_printed_allowed_values(self):
        self.assertEqual(labels('10 : 파면, 20 : 해임 ... '), ['파면', '해임'])
        self.assertEqual(labels('1: 일반경쟁, 2:지명경쟁'), ['일반경쟁', '지명경쟁'])
        self.assertEqual(labels('001 : 본인, 002 : 처 …'), ['본인'])  # one-letter label dropped


if __name__ == '__main__':
    unittest.main()

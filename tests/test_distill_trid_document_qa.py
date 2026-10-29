"""Distilled document answers must come from the document, cite it at the end, and not echo instructions."""
import unittest
from scripts.distill_trid_document_qa import screen_answer, screen_question

DOC = {'title': '전파산업 제품화 지원', 'text': '총제작비의 70%까지 최대 4천만 원 한도 내에서 지원한다.'}


class DocumentQAScreenTests(unittest.TestCase):
    def test_answer_quantities_must_be_in_the_document(self):
        self.assertEqual(screen_answer(DOC, '총제작비의 70%까지, 최대 4천만 원입니다. [D1]')[1], None)
        self.assertEqual(screen_answer(DOC, '총제작비의 80%까지 지원합니다. [D1]')[1], 'unsupported_quantity')

    def test_answer_shape(self):
        self.assertEqual(screen_answer(DOC, '70%까지입니다.')[1], 'no_citation')
        self.assertEqual(screen_answer(DOC, '[D1] 70%까지입니다.')[1], 'citation_first')
        self.assertEqual(screen_answer(DOC, '70%까지입니다. [D1]\n질문: 한도는?')[1], 'continued_question')

    def test_instruction_echo_questions_dropped(self):
        self.assertIsNone(screen_question('지원 비율과 최대 한도는 얼마인가요?'))
        self.assertIsNone(screen_question('지원 대상(중소기업)은 누구인가요?'))
        self.assertEqual(screen_question('지원 비율은? (지원 비율, 최대 한도가 모두 포함되어야 함'), 'instruction_echo')
        self.assertEqual(screen_question('참석한 장관은 누구인가요? (기후부 장관, 중기부 장관 중 선택'), 'instruction_echo')


if __name__ == '__main__':
    unittest.main()

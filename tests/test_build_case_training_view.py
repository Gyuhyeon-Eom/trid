"""Case training view: targets cite the passages they use, numbers must come from the evidence, transforms are reversible."""
import unittest

from scripts.build_case_training_view import PREFIX, cite, nums, provenance, strip


def row(target, context, question='질문'):
    return {'target': target, 'context': context, 'question': question}


class CiteTests(unittest.TestCase):
    def test_sentence_gets_the_passage_that_holds_its_number(self):
        out, problems = cite(row('총지출은 656.9입니다. 단위는 조원입니다.',
                                 ['연도,구분,본예산\n2024,총지출,656.9', 'unit=조원 (설명 문구)']))
        self.assertEqual(problems, [])
        self.assertTrue(out.startswith('총지출은 656.9입니다 [D1].'))

    def test_number_absent_from_evidence_and_question_holds_the_row(self):
        _, problems = cite(row('기준 금액은 5,000만원입니다.', ['장비 교체는 1억원 초과이면 본부장 전결']))
        self.assertIn('ungrounded_number:5000', problems)

    def test_number_taken_from_the_question_needs_no_source(self):
        out, problems = cite(row('1,500만 원은 1천만원 이상이므로 청장 전결입니다.',
                                 ['추정가격 1천만원 이상 사업은 청장 전결'], question='1,500만 원 사업은 누가 전결하나요?'))
        self.assertEqual(problems, [])
        self.assertIn('[D1]', out)

    def test_row_without_any_supported_sentence_is_held(self):
        _, problems = cite(row('관련 내용이 없습니다.', ['전혀 다른 조문 본문입니다 그리고 길게 이어집니다']))
        self.assertEqual(problems, ['no_supported_sentence'])


class NumberTests(unittest.TestCase):
    def test_csv_fields_are_not_joined_into_one_number(self):
        self.assertEqual(nums('100,D02,13010602,Y'), {'100', '02', '13010602'})

    def test_thousands_separator_and_circled_paragraph_numbers(self):
        self.assertEqual(nums('2,345,678원 ②'), {'2345678', '2'})


class ContractTests(unittest.TestCase):
    def test_markers_and_prefix_strip_back_to_the_original_target(self):
        self.assertEqual(strip(PREFIX + '금액은 근거에 없습니다 [D1][D2]. 확인해 주세요.'), '금액은 근거에 없습니다. 확인해 주세요.')

    def test_provenance_names_the_snapshot_date_without_calling_it_current(self):
        line = provenance({'doc': 'law-go-kr:admrul-file:1', 'institution': '가기관',
                           'time_label': 'snapshot 2023-02-13; current value unverified'})
        self.assertIn('2023-02-13 자 수집본(이후 개정 여부 미확인)', line)
        self.assertNotIn('현행입니다', line)


if __name__ == '__main__':
    unittest.main()

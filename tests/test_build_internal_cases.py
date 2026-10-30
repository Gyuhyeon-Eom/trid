"""Footnote handling in scripts/build_internal_cases.py (C-015).

Rows imitate real 법령해석례 text: footnote bodies sit inline as 'N)N) body\n' right after the cited word, and B's
v3-norm2 'footnote' edits give their v3 offsets and a '[각주N]' replacement. The premise and definition sentences
are taken from law-go-kr:expc:329517 and expc:333611.
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('bic', Path(__file__).resolve().parents[1] / 'scripts/build_internal_cases.py')
bic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bic)

PREMISE = '1)1) 「공유재산 및 물품 관리법」 제5조제2항에 따른 행정재산임을 전제함.\n'
DEFINITION = '2)2) 「식품위생법」 제2조제12호에 따른 집단급식소를 말함\n'
CITATION = '3)3) 대법원 2009. 4. 23. 선고 2006다81035 판결례 참조\n'


def row(question, answer, reasons):
    text = f'제목\n【질의요지】\n{question}\n【회답】\n{answer}\n【이유】\n{reasons}'
    return dict(id='t', title='제목', text=text, date='2020-01-01', meta=dict(date_type='해석일자', raw_fields={}))


def edits_for(text, *bodies):
    out = []
    for body in bodies:
        start = text.index(body)
        n = body.split(')', 1)[0]
        out.append(dict(rule='footnote', src_start=start, src_end=start + len(body), replacement=f'[각주{n}]'))
    return out


class FootnoteSections(unittest.TestCase):
    def build(self, question, answer, reasons, *bodies):
        r = row(question, answer, reasons)
        return bic.official_case(r, edits_for(r['text'], *bodies))

    def test_question_premise_and_definition_stay_with_the_question(self):
        case, why = self.build(f'영구시설물{PREMISE}을 설치한 경우 집단급식소{DEFINITION}에 기부할 수 있는지?',
                               '기부할 수 있습니다.', f'문언상 명확합니다{CITATION}. 따라서 기부할 수 있습니다.',
                               PREMISE, DEFINITION, CITATION)
        self.assertIsNone(why)
        self.assertIn('영구시설물[각주1]을 설치한 경우', case['question'])
        self.assertTrue(case['question'].endswith(
            '\n[각주1] 「공유재산 및 물품 관리법」 제5조제2항에 따른 행정재산임을 전제함.'
            '\n[각주2] 「식품위생법」 제2조제12호에 따른 집단급식소를 말함'))
        # the reasons' citation goes with the reasons only
        self.assertNotIn('2006다81035', case['question'])
        self.assertTrue(case['evidence']['reasons'].endswith('\n[각주3] 대법원 2009. 4. 23. 선고 2006다81035 판결례 참조'))
        self.assertEqual(sorted(case['footnotes']), ['question', 'reasons'])

    def test_without_edits_footnote_text_is_not_turned_into_a_case(self):
        r = row(f'영구시설물{PREMISE}을 설치한 경우 기부할 수 있는지?', '기부할 수 있습니다.', '따라서 기부할 수 있습니다.')
        case, why = bic.official_case(r)
        self.assertIsNone(case)
        self.assertIn('footnote', why)

    def test_repeated_number_in_one_section_is_rejected(self):
        first, second = '1)1) 법제처 해석례 참조\n', '1)1) 구 「지방공무원수당규정」 참조\n'
        r = row('수당을 지급할 수 있는지?', '지급할 수 없습니다.', f'가{first}이고 나{second}입니다. 따라서 지급할 수 없습니다.')
        case, why = bic.official_case(r, edits_for(r['text'], first, second))
        self.assertIsNone(case)
        self.assertIn('one to one', why)

    def test_body_attached_in_two_sections_is_rejected(self):
        a, b = '1)1) 같은 정의를 말함\n', '2)2) 같은 정의를 말함\n'
        r = row(f'공무원{a}이 할 수 있는지?', '할 수 있습니다.', f'공무원{b}은 할 수 있습니다.')
        case, why = bic.official_case(r, edits_for(r['text'], a, b))
        self.assertIsNone(case)
        self.assertIn('two sections', why)

    def test_footnote_crossing_a_section_boundary_is_rejected(self):
        r = row(f'기부할 수 있는지?{PREMISE}', '기부할 수 있습니다.', '따라서 기부할 수 있습니다.')
        e = edits_for(r['text'], PREMISE)[0]
        e['src_end'] = r['text'].index('【회답】') + 2
        case, why = bic.official_case(r, [e])
        self.assertIsNone(case)
        self.assertIn('boundary', why)


if __name__ == '__main__':
    unittest.main()

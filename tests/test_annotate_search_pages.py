"""Value checks of the search corpus: HWPX reference text and the classes given to numbers missing from PDF pages."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.annotate_search_pages import MARKUP, number_class, numbers, hwpx_text


def cls(n, text, near, near_raw='', mapped=None, whole=''):
    return number_class(n, text, near, near_raw, near if mapped is None else mapped, whole)


class HwpxTextTests(unittest.TestCase):
    def test_runs_with_nested_elements_attributes_and_equation_scripts_are_read(self):
        section = ('<hs:sec><hp:p><hp:run><hp:t>인터넷질의답변<hp:tab/>(2006.11.28.)</hp:t></hp:run>'
                   '<hp:run><hp:t xml:space="preserve">지급액 15,000원</hp:t></hp:run>'
                   '<hp:equation><hp:script>{8,000} over {15,000} TIMES 100=53.3%</hp:script></hp:equation></hp:p></hs:sec>')
        path = Path(tempfile.mkdtemp()) / 'a.hwpx'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('Contents/section0.xml', section)
        ref = numbers(hwpx_text(path))
        extracted = numbers('인터넷질의답변(2006.11.28.) 지급액 15,000원 $\\frac{8000}{15000} \\times 100=53.3%$')
        self.assertEqual(extracted - ref, {}, 'every extracted number has a source in the XML')

    def test_image_file_names_are_markup_not_values(self):
        self.assertEqual(numbers(MARKUP.sub(' ', '영상회의 ![image](images/out/image_052.bmp) 개설')), {})


class NumberClassTests(unittest.TestCase):
    def test_digits_of_separate_cells_joined_by_extraction_are_held(self):
        self.assertEqual(cls('12345', '등급 12345', '찬성        1      2      3       4      5'), 'merged_tokens')
        self.assertEqual(cls('0100', '1 2019-01-0100 0.01', '    1     2019-01-01 00     0.01'), 'merged_tokens')

    def test_letter_spaced_token_in_layout_text_is_the_same_value(self):
        self.assertEqual(cls('0.1346', 'v 0.1346 km/s', 'v = V ÷ 3600       0 . 1 3 4 6 v'), 'letter_spaced')

    def test_raw_text_cannot_prove_letter_spacing(self):
        # -raw collapses column gaps to one space, so '1 2 3' there may be three cells
        self.assertEqual(cls('123', '등급 123', near='', near_raw='등급 1 2 3'), 'merged_tokens')

    def test_space_after_a_separator_is_spacing_not_a_different_value(self):
        self.assertEqual(cls('2008.8', '일부개정 2008.8.11 국방부령', '[일부개정 2008.\n8.11 국방부령'), 'separator_spacing')

    def test_short_numbers_are_not_matched_across_cells(self):
        self.assertEqual(cls('61', '61 ~ 80', '6   1:1.0  45.0', whole='표 61 행'), 'elsewhere_in_pdf_short_number')

    def test_digits_beside_a_redaction_token_are_privacy_residue(self):
        self.assertEqual(cls('4039', '출입국심사과 [전화]4039', '02-2110-4037,4039'), 'redaction_residue')

    def test_page_without_text_layer_is_not_checkable(self):
        self.assertEqual(cls('2021', '공고 2021/5/12', '21', mapped=' '), 'image_only_page')


if __name__ == '__main__':
    unittest.main()

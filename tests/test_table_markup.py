import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from collect_admrul_attachments import strip_table_markup  # noqa: E402


class StripTableMarkupTest(unittest.TestCase):
    def test_statute_angle_brackets_survive(self):
        # statute text writes amendment and deletion dates in angle brackets; they carry dates and must not be stripped
        text = '■ 공무원보수규정 [별표 1] <개정 2026. 9. 29.>\n[별표 5] 삭제 <2007.11.22>'
        self.assertEqual(strip_table_markup(text), text)

    def test_table_markup_removed_cell_text_kept(self):
        out = strip_table_markup('<table><tr><td rowspan="2">대표<br>신청인</td><th>성명</th></tr></table>')
        self.assertEqual(out.split(), ['대표', '신청인', '성명'])


if __name__ == '__main__':
    unittest.main()

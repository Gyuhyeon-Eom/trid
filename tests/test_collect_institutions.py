"""ALIO reserved acquisition: every revision file must be listed, typed by content, and groups interleaved."""
import unittest
from scripts.collect_institutions import files_of, kind_of, round_robin


class AlioRulesTests(unittest.TestCase):
    def test_bfiles_split_like_the_site_and_keeps_commas_in_names(self):
        self.assertEqual(files_of('69727|취업 규칙(2018년도 10월 18일 개정).hwp,78319|규정 1,2호 서식.pdf'),
                         [('69727', '취업 규칙(2018년도 10월 18일 개정).hwp'), ('78319', '규정 1,2호 서식.pdf')])
        self.assertEqual(files_of(None), [])

    def test_kind_comes_from_bytes_not_only_the_name(self):
        self.assertEqual(kind_of(b'%PDF-1.7', 'x.hwp'), 'pdf')
        self.assertEqual(kind_of(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest', 'x.hwp'), 'hwp')
        self.assertEqual(kind_of(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest', 'x.doc'), 'ole:doc')
        self.assertEqual(kind_of(b'PK\x03\x04', 'x.hwpx'), 'hwpx')
        self.assertEqual(kind_of(b'<html>error', 'x.hwp'), 'unknown:hwp')
        self.assertEqual(kind_of(b'PK\x03\x04', 'x.hwp'), 'hwpx')
        self.assertEqual(kind_of(b'PK\x03\x04', 'x.zip'), 'zip:zip')

    def test_round_robin_covers_every_group_before_repeating(self):
        fams = [{'business_group': g, 'i': i} for g, n in (('A', 3), ('B', 1), ('C', 2)) for i in range(n)]
        self.assertEqual([f['business_group'] for f in round_robin(fams)], ['A', 'B', 'C', 'A', 'C', 'A'])


if __name__ == '__main__':
    unittest.main()

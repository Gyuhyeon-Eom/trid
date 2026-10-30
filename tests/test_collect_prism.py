"""PRISM pilot: only public reports under 공공누리 1유형/자유이용 with commercial use and modification allowed are taken."""
import unittest

from scripts.collect_prism import eligibility


def info(rls='B0030001', kogl='D0590002', cmrc='Y', chg='Y', files=('D0150004',)):
    return {'asmtDetail': {'infoRlsCd': rls, 'koglCmrcUtztnYn': cmrc, 'koglWrgsChgYn': chg},
            'koglDetail': {'koglTypeCd': kogl} if kogl else None,
            'asmtFileList': [{'fileTypeCd': f} for f in files]}


class PrismRightsTests(unittest.TestCase):
    def test_only_open_commercial_reports_with_a_report_file_are_taken(self):
        self.assertEqual(eligibility(info()), 'ok')
        self.assertEqual(eligibility(info(kogl='D0590006')), 'ok')
        for kogl in ('D0590003', 'D0590004', 'D0590005', None):
            self.assertTrue(eligibility(info(kogl=kogl)).startswith('kogl:'), kogl)
        self.assertEqual(eligibility(info(rls='B0030003')), 'not_public:B0030003')  # 부분공개
        self.assertEqual(eligibility(info(cmrc='N')), 'kogl_flags_restrict')
        self.assertEqual(eligibility(info(chg='N')), 'kogl_flags_restrict')
        self.assertEqual(eligibility(info(files=('D0150018',))), 'no_report_file')  # 활용결과 보고서 only


if __name__ == '__main__':
    unittest.main()

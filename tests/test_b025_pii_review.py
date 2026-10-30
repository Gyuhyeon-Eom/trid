"""A PII hint in a statute is waved through only when it is the '+N' term of a written formula."""
import unittest

from scripts.build_api_snapshot import pii_review


class PiiReviewTests(unittest.TestCase):
    def test_formula_plus_terms_are_explained(self):
        hits = pii_review('결원 5인을 초과하는 매 1인당 3배수+20인 / 1억원 이하 ×20/100+300만원')
        self.assertEqual([h['explained'] for h in hits], ['arithmetic_formula_plus_term'] * 2)

    def test_phone_like_numbers_stay_unexplained(self):
        hits = pii_review('문의 전화 +82 2-2100-1234, 담당자 010-1234-5678')
        self.assertTrue(hits)
        self.assertTrue(all(h['explained'] is None for h in hits))


if __name__ == '__main__':
    unittest.main()

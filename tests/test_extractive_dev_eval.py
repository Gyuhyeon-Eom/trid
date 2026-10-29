"""Risk/coverage must retain failed and abstaining rows in the denominator."""
import unittest
from scripts.evaluate_extractive_dev import risk_coverage


class DevelopmentRiskTests(unittest.TestCase):
    def test_ties_are_not_cherry_picked_and_failures_remain_in_denominator(self):
        rows = [
            dict(presence=0, presence_probs=[.9, .05, .05], strict=True, exception=None),
            dict(presence=0, presence_probs=[.9, .05, .05], strict=False, exception=None),
            dict(presence=0, presence_probs=[.7, .2, .1], strict=True, exception=None),
            dict(presence=1, presence_probs=[.01, .98, .01], strict=True, exception=None),
            dict(exception='input too long', strict=False),
        ]
        points = risk_coverage(rows)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[1]['answered'], 2)
        self.assertEqual(points[1]['correct'], 1)
        self.assertEqual(points[1]['coverage'], .4)
        self.assertEqual(points[1]['risk'], .5)
        self.assertEqual(points[2]['answered'], 3)
        self.assertEqual(points[2]['coverage'], .6)
        self.assertAlmostEqual(points[2]['risk'], 1/3)

    def test_no_answers_has_undefined_risk_not_perfect_accuracy(self):
        for rows in ([], [dict(presence=1, strict=True, exception=None)]):
            with self.subTest(rows=rows):
                points = risk_coverage(rows)
                self.assertEqual(len(points), 1)
                self.assertEqual(points[0]['answered'], 0)
                self.assertEqual(points[0]['coverage'], 0)
                self.assertIsNone(points[0]['risk'])

    def test_error_row_with_partial_output_cannot_be_accepted(self):
        points = risk_coverage([dict(presence=0, presence_probs=[1., 0., 0.],
                                    strict=True, exception='failed output verification')])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['answered'], 0)


if __name__ == '__main__':
    unittest.main()

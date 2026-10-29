import unittest
from scripts.evaluate_institution_probe import summarize, groups_of


def result(exact, legacy, category, groups, exception=None):
    return dict(exact=exact, legacy=legacy, category=category, groups=groups, exception=exception)


class SummarizeTest(unittest.TestCase):
    def test_groups_by_task_and_evidence_count(self):
        self.assertEqual(groups_of(dict(task='institution_value', evidence=[{}])),
                         ('all', 'institution_value', 'single_evidence'))
        self.assertEqual(groups_of(dict(task='institution_missing', evidence=[{}, {}])),
                         ('all', 'institution_missing', 'multi_evidence'))

    def test_counts_exact_separately_from_legacy(self):
        rows = [result(True, True, 'correct', ('all', 'institution_value', 'single_evidence')),
                result(False, True, 'wrong_value_right_source', ('all', 'institution_value', 'multi_evidence')),
                result(False, False, 'false_presence', ('all', 'institution_missing', 'single_evidence')),
                result(False, False, 'wrong_abstention', ('all', 'institution_value', 'single_evidence')),
                result(False, False, 'execution', ('all', 'institution_value', 'single_evidence'), exception='too long')]
        summary = summarize(rows)
        self.assertEqual(summary['all']['total'], 5)
        self.assertEqual(summary['all']['exact'], 1)
        self.assertEqual(summary['all']['legacy_strict'], 2)
        self.assertEqual(summary['all']['false_presence'], 1)
        self.assertEqual(summary['all']['wrong_abstention'], 1)
        self.assertEqual(summary['all']['exceptions'], 1)
        self.assertEqual(summary['institution_missing']['exact_rate'], 0.)
        self.assertEqual(summary['multi_evidence']['categories'], {'wrong_value_right_source': 1})
        self.assertIsNone(summarize([])['all']['exact_rate'])


if __name__ == '__main__':
    unittest.main()

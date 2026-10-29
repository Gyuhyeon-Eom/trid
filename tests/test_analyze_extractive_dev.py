"""Analysis of a dev report must not shrink denominators or misfile abstention errors."""
import unittest
from trid.prompts import rag_prompt
from scripts.analyze_extractive_dev import expected_rows, analyze, categorize, evaluate_gates, protocol_mismatches


def raw_row(id, task, answer, passages=(('D1', '마감일은 3월 2일이다.'), ('D2', '접수는 4월 5일까지다.'))):
    return dict(id=id, task=task, answer=answer, group='rag', history=None,
                prompt=rag_prompt(list(passages), '마감일은?'))


RAW = [raw_row('p1', 'rag_value', '3월 2일 [D1]'),
       raw_row('m1', 'rag_missing', '확인 불가'),
       raw_row('c1', 'rag_conflict', '근거 충돌'),
       raw_row('k1', 'rag_combine', '3월 2일 [D1], 4월 5일 [D2]')]


def result(id, generated, exception=None, reason=None):
    row = next(r for r in RAW if r['id'] == id)
    expected = {'rag_value': 0, 'rag_missing': 1, 'rag_conflict': 2, 'rag_combine': 0}[row['task']]
    presence = None if generated is None else ({'확인 불가': 1, '근거 충돌': 2}.get(generated, 0))
    return dict(id=id, task=row['task'], answer=row['answer'], expected_presence=expected, generated=generated,
                strict=generated == row['answer'].strip(), exception=exception, presence=presence, reason=reason,
                seconds=0.01)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.expected = expected_rows(RAW)

    def test_missing_ids_reject_instead_of_inflating_accuracy(self):
        with self.assertRaises(ValueError):
            analyze([result('p1', '3월 2일 [D1]')], self.expected)

    def test_duplicate_and_unexpected_ids_reject(self):
        full = [result('p1', '3월 2일 [D1]'), result('m1', '확인 불가'), result('c1', '근거 충돌'), result('k1', '3월 2일 [D1]')]
        with self.assertRaises(ValueError):
            analyze(full + [result('p1', '3월 2일 [D1]')], self.expected)
        with self.assertRaises(ValueError):
            analyze(full + [dict(result('p1', '3월 2일 [D1]'), id='ghost')], self.expected)

    def test_inconsistent_strict_flag_rejects(self):
        full = [result('p1', '3월 2일 [D1]'), result('m1', '확인 불가'), result('c1', '근거 충돌'), result('k1', '3월 2일 [D1]')]
        full[0]['strict'] = False
        with self.assertRaises(ValueError):
            analyze(full, self.expected)

    def test_conflict_answered_missing_is_kind_swap_not_false_presence_and_not_strict(self):
        summary, rows = analyze([result('p1', '3월 2일 [D1]'), result('m1', '확인 불가'),
                                 result('c1', '확인 불가'), result('k1', '3월 2일 [D1]')], self.expected)
        conflict = summary['conflict']
        self.assertEqual((conflict['total'], conflict['strict'], conflict['false_presence']), (1, 0, 0))
        self.assertEqual(conflict['categories'], {'abstention_kind_swap': 1})
        self.assertEqual(next(r for r in rows if r['id'] == 'c1')['detail'], 'conflict->missing')

    def test_unsupported_combine_stays_in_totals_and_out_of_reachable_present(self):
        summary, rows = analyze([result('p1', '3월 2일 [D1]'), result('m1', '확인 불가'),
                                 result('c1', '근거 충돌'), result('k1', '3월 2일 [D1]')], self.expected)
        self.assertEqual(summary['all']['total'], 4)
        self.assertEqual(summary['all']['strict'], 3)
        self.assertEqual(summary['combine']['categories'], {'unsupported_combine': 1})
        self.assertEqual(summary['reachable_present']['total'], 1)
        self.assertEqual(summary['multi_passage_present']['total'], 1)

    def test_error_categories(self):
        rows = self.expected
        self.assertEqual(categorize(result('p1', '3월 2일 [D2]'), rows['p1']), ('wrong_source_right_value', None))
        self.assertEqual(categorize(result('p1', '3월 [D1]'), rows['p1']), ('wrong_value_right_source', 'truncated'))
        self.assertEqual(categorize(result('p1', '마감일은 3월 2일 [D1]'), rows['p1']), ('wrong_value_right_source', 'overlong'))
        self.assertEqual(categorize(result('p1', '4월 5일 [D2]'), rows['p1']), ('wrong_both', None))
        self.assertEqual(categorize(result('p1', '확인 불가', reason='no_valid_span'), rows['p1']),
                         ('wrong_abstention', 'no_valid_span:missing'))
        self.assertEqual(categorize(result('p1', '근거 충돌'), rows['p1']), ('wrong_abstention', 'head:conflict'))
        self.assertEqual(categorize(result('m1', '3월 2일 [D1]'), rows['m1']), ('false_presence', 'missing'))
        self.assertEqual(categorize(result('p1', None, exception='too long'), rows['p1']), ('execution', None))
        self.assertEqual(categorize(result('p1', '3월 2일'), rows['p1']), ('format', 'unparsable'))

    def test_all_abstention_data_without_reachable_group_is_reported_not_crashed(self):
        raw = [r for r in RAW if r['task'] in ('rag_missing', 'rag_conflict')]
        expected = expected_rows(raw)
        summary, _ = analyze([result('m1', '확인 불가'), result('c1', '근거 충돌')], expected)
        self.assertNotIn('reachable_present', summary)
        protocol = dict(run='runs/x', evaluation=dict(data='data/x/valid.jsonl', rag_rows=2, expected_present_rows=0,
                                                       missing_rows=1, conflict_rows=1,
                                                       unsupported_combine_rows_included_in_present=0),
                        structural_limits=dict(reachable_present_rows=297),
                        development_gates_registered_before_candidate_results=dict(
                            strict_accuracy_at_least=.8, klue_strict_accuracy_at_least=.7,
                            multi_passage_reachable_present_strict_accuracy_at_least=.7,
                            reachable_present_wrong_abstention_rate_at_most=.15, execution_exceptions=0))
        reasons = protocol_mismatches(dict(run='runs/x', data='data/x/valid.jsonl'), protocol, expected, summary)
        self.assertEqual(reasons, ['reachable_present_rows: protocol 297 vs data 0'])
        gates = evaluate_gates(summary, protocol, complete=True)
        self.assertTrue(gates['strict_accuracy_at_least']['passed'])
        self.assertFalse(gates['klue_strict_accuracy_at_least']['passed'])
        self.assertFalse(gates['multi_passage_reachable_present_strict_accuracy_at_least']['passed'])
        self.assertFalse(gates['reachable_present_wrong_abstention_rate_at_most']['passed'])


if __name__ == '__main__':
    unittest.main()

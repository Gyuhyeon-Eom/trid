import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.compare_reader_reports import compare, digest
from trid.prompts import rag_prompt


class PairedReportsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / 'valid.jsonl'
        rows = [dict(id=str(i), task='rag_value', group='rag', answer='10원 [D1]',
                     prompt=rag_prompt([('D1', '요금은 10원이다.')], '요금은?')) for i in range(4)]
        self.data.write_text('\n'.join(json.dumps(r) for r in rows))
        def report(bits):
            return dict(run='fixture', model_sha256='fixture', data_sha256=digest(self.data), evaluated_rows=4,
                        results=[dict(id=str(i), task='rag_value', answer='10원 [D1]', expected_presence=0,
                                      generated='10원 [D1]' if ok else '확인 불가', strict=ok, presence=0 if ok else 1)
                                 for i, ok in enumerate(bits)])
        self.before = report([True, False, False, True])
        self.after = report([False, True, False, True])

    def test_pairs_by_id_not_order_and_counts_regressions(self):
        self.after['results'].reverse()
        result = compare(self.before, self.after, self.data)
        self.assertEqual(result['paired_counts'], dict(both_correct=1, improved=1, regressed=1, both_wrong=1))
        self.assertEqual(result['net_accuracy_change_pp'], 0)
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 1)
        self.assertEqual(result['transitions']['correct -> wrong_abstention:head:missing'], 1)

    def test_rejects_duplicate_ids_or_changed_gold(self):
        for change in ('duplicate', 'gold'):
            altered = copy.deepcopy(self.after)
            if change == 'duplicate':
                altered['results'][0]['id'] = '1'
            else:
                altered['results'][0]['answer'] = '20원 [D1]'
            with self.assertRaises(ValueError):
                compare(self.before, altered, self.data)

    def test_rejects_different_evaluation_data(self):
        self.after['data_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            compare(self.before, self.after, self.data)

import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_journal import Journal

HEADER = {'model_sha256': 'abc', 'prompts_sha256': 'def', 'max_tokens': 256}


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'run.json'

    def test_resume_keeps_finished_items_and_skips_them(self):
        first = Journal(self.output, HEADER)
        first.append({'id': 'a', 'generated': 'x'})
        first.append({'id': 'b', 'generated': 'y'})
        again = Journal(self.output, HEADER)
        self.assertTrue(again.done('a') and again.done('b') and not again.done('c'))
        again.append({'id': 'c', 'generated': 'z'})
        rows, counts = again.finalize(['a', 'b', 'c'])
        self.assertEqual([r['generated'] for r in rows], ['x', 'y', 'z'])
        self.assertEqual(counts['items'], 3)

    def test_torn_last_line_from_crash_is_dropped_and_item_rerun(self):
        Journal(self.output, HEADER).append({'id': 'a', 'generated': 'x'})
        with Path(f'{self.output}.partial.jsonl').open('a') as f:
            f.write('{"id": "b", "gener')
        again = Journal(self.output, HEADER)
        self.assertFalse(again.done('b'))
        again.append({'id': 'b', 'generated': 'y'})
        self.assertEqual([r['id'] for r in again.finalize(['a', 'b'])[0]], ['a', 'b'])

    def test_different_run_settings_are_refused(self):
        Journal(self.output, HEADER).append({'id': 'a'})
        with self.assertRaisesRegex(ValueError, 'different run'):
            Journal(self.output, {**HEADER, 'max_tokens': 64})

    def test_incomplete_run_cannot_be_finalized(self):
        journal = Journal(self.output, HEADER)
        journal.append({'id': 'a'})
        with self.assertRaisesRegex(ValueError, '1 missing'):
            journal.finalize(['a', 'b'])

    def test_duplicate_lines_count_once_and_failures_are_counted(self):
        path = Path(f'{self.output}.partial.jsonl')
        path.write_text('\n'.join(json.dumps(x) for x in [HEADER, {'id': 'a', 'exception': 'OSError: x'},
                                                                 {'id': 'a', 'exception': None},
                                                                 {'id': 'b', 'truncated_output': True}]) + '\n')
        rows, counts = Journal(self.output, HEADER).finalize(['a', 'b'])
        self.assertEqual(counts['duplicates_dropped_on_resume'], 1)
        self.assertEqual((counts['exceptions'], counts['truncated_output']), (1, 1))
        self.assertEqual(rows[0]['exception'], 'OSError: x')


if __name__ == '__main__':
    unittest.main()

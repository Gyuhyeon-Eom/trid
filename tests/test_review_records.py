import json
import tempfile
import unittest
from pathlib import Path

from scripts.review_records import aggregate, export, read_jsonl, validate_verdicts

PROMPTS = {'results': [{'id': 'q1', 'prompt': '[D1] 지원액은 30만원이다.\n질문: 지원액은?'},
                       {'id': 'q2', 'prompt': '[D1] 마감은 5월 1일이다.\n질문: 마감은?'}]}
CHECKLIST = {'items': [{'id': 'q1', 'question': '지원액은?'}, {'id': 'q2', 'question': '마감은?'}]}


def trid_output(rows):
    return {'run': 'runs/x', 'weights_sha256': 'w1', 'results': rows}


def external_output(rows):
    return {'model_id': 'org/m', 'revision': 'r', 'model_sha256': 'w2', 'max_output_tokens': 1024, 'results': rows}


class ReviewRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / 'prompts.json').write_text(json.dumps(PROMPTS, ensure_ascii=False))
        (self.tmp / 'checklist.json').write_text(json.dumps(CHECKLIST, ensure_ascii=False))
        self.a = self.write('a.json', trid_output([
            {'id': 'q1', 'answer': '30만원 [D1]', 'stopped_on_eos': True},
            {'id': 'q2', 'answer': None, 'exception': 'ValueError: 문맥 초과'}]))
        self.b = self.write('b.json', external_output([
            {'id': 'q1', 'generated': '지원액은 40만원입니다 [D2]', 'finish_reason': 'eos'},
            {'id': 'q2', 'generated': '마감은', 'finish_reason': 'limit'}]))

    def write(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data, ensure_ascii=False))
        return path

    def run_export(self, out='exp', seed=1, outputs=None):
        return export(outputs or [('model-a', self.a), ('model-b', self.b)], self.tmp / 'prompts.json',
                      self.tmp / 'checklist.json', {}, self.tmp / out, seed)

    def verdict(self, mapping, code, **kw):
        return {**dict(packet_code=code, packet_sha256=mapping['packet_sha256'], reviewer_id='ai:x', reviewer_type='ai',
                       conclusion='정답'), **kw}

    def test_each_response_appears_once_and_packet_hides_models(self):
        mapping = self.run_export()
        packet = read_jsonl(self.tmp / 'exp/packet.jsonl')
        self.assertEqual(len(packet), 4)
        self.assertEqual(len({p['packet_code'] for p in packet}), 4)
        self.assertEqual(sorted((v['model'], v['item_id']) for v in mapping['codes'].values()),
                         [('model-a', 'q1'), ('model-a', 'q2'), ('model-b', 'q1'), ('model-b', 'q2')])
        text = (self.tmp / 'exp/packet.jsonl').read_text()
        self.assertNotIn('model-a', text)
        self.assertNotIn('w1', text)
        flagged = next(p for p in packet if p['response'] == '지원액은 40만원입니다 [D2]')['auxiliary']
        self.assertEqual((flagged['cited_ids_in_prompt'], flagged['numbers_not_in_prompt']), (False, ['40']))

    def test_output_missing_an_item_is_rejected(self):
        short = self.write('c.json', external_output([{'id': 'q1', 'generated': 'x', 'finish_reason': 'eos'}]))
        with self.assertRaisesRegex(ValueError, '1 missing'):
            self.run_export(outputs=[('model-c', short)])

    def test_ai_reviewer_cannot_be_recorded_as_human_and_unregistered_is_refused(self):
        mapping = self.run_export()
        code = next(iter(mapping['codes']))
        registry = {'ai:x': 'ai', 'human:kim': 'human'}
        with self.assertRaisesRegex(ValueError, 'registered as ai, not human'):
            validate_verdicts([self.verdict(mapping, code, reviewer_type='human')], mapping, registry)
        with self.assertRaisesRegex(ValueError, 'not registered'):
            validate_verdicts([{**self.verdict(mapping, code), 'reviewer_id': 'ai:other'}], mapping, registry)

    def test_verdicts_from_another_export_and_duplicates_are_refused(self):
        first, second = self.run_export('one', 1), self.run_export('two', 2)
        code = next(iter(second['codes']))
        with self.assertRaisesRegex(ValueError, 'another export'):
            validate_verdicts([self.verdict(second, code)], first, {'ai:x': 'ai'})
        code = next(iter(first['codes']))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_verdicts([self.verdict(first, code), self.verdict(first, code)], first, {'ai:x': 'ai'})

    def test_unreviewed_and_uncertain_stay_unscored_and_failures_stay_in_denominator(self):
        mapping = self.run_export()
        records = read_jsonl(self.tmp / 'exp/records.jsonl')
        code = {(v['model'], v['item_id']): c for c, v in mapping['codes'].items()}
        verdicts = [self.verdict(mapping, code[('model-a', 'q1')]),
                    self.verdict(mapping, code[('model-b', 'q1')], conclusion='불확실')]
        validate_verdicts(verdicts, mapping, {'ai:x': 'ai'})
        out = aggregate(records, verdicts, mapping)
        self.assertEqual(out['model-a']['execution'], {'eos': 1, 'exception': 1})
        self.assertEqual(out['model-a']['items'], 2)
        ai_a, ai_b = out['model-a']['reviews']['ai'], out['model-b']['reviews']['ai']
        self.assertEqual((ai_a['reviewed'], ai_a['unreviewed'], ai_a['correct_rate_over_reviewed']), (1, 1, 1.0))
        self.assertEqual((ai_b['reviewed'], ai_b['uncertain'], ai_b['unreviewed'], ai_b['correct_rate_over_reviewed']), (0, 1, 1, None))
        self.assertEqual(out['model-a']['reviews']['human']['unreviewed'], 2)


if __name__ == '__main__':
    unittest.main()

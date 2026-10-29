"""Behavioral regressions for trid.span_boundaries, the one cut rule behind trid.span_scoring.candidates and
trid.extractive (masks, label placement, best_span)."""
import tempfile
import unittest
from pathlib import Path
import torch
from trid.text import train_tokenizer
from trid.prompts import rag_prompt
from trid.span_boundaries import start_ok, end_ok, BOUNDARY_POLICY
from trid.span_scoring import candidates
from trid.extractive import encode, best_span, decode, span_issue, read, ExtractiveReader, MAX_SPAN
from trid.model import Config


def spans_of(text):
    return {span for _, span, _ in candidates([('D1', text)]) if span is not None}


class CutRuleTests(unittest.TestCase):
    def test_policy_identifier(self):
        self.assertEqual(BOUNDARY_POLICY, 'span_boundary_v2')

    def test_ascii_alphanumerics_are_indivisible_at_both_ends(self):
        for text in ('1675', 'ab12', 'SRTPlay'):
            for i in range(1, len(text)):
                self.assertFalse(end_ok(text, i - 1), (text, i))
                self.assertFalse(start_ok(text, i), (text, i))
        self.assertTrue(end_ok('1675-3561', 3))  # hyphen policy: '1675' ends before '-'
        self.assertTrue(start_ok('1675-3561', 5))  # and '3561' may start after it
        self.assertFalse(start_ok('A3,', 1))

    def test_internal_comma_and_period_protect_both_cuts(self):
        for text, cuts in (('1,000', (1, 2)), ('30.4', (2, 3)), ('foo.bar', (3, 4)), ('1,000,000', (1, 2, 5, 6))):
            for cut in cuts:
                self.assertFalse(end_ok(text, cut - 1), (text, cut))
                self.assertFalse(start_ok(text, cut), (text, cut))
        self.assertEqual(spans_of('가격 1,000원'), {'가', '가격', '가격 1,000', '가격 1,000원', '1,000', '1,000원'})
        self.assertEqual(spans_of('30.4%다'), {'30.4', '30.4%', '30.4%다', '다'})  # '다' starts after the delimiter '%'
        self.assertIn('www.huvis.com', spans_of('홈페이지(www.huvis.com)를'))
        self.assertNotIn('huvis.com', spans_of('홈페이지(www.huvis.com)를'))

    def test_separating_comma_and_terminal_period_are_delimiters(self):
        text = 'KJB 0.0636663, KNB 0.097'
        self.assertTrue(end_ok(text, text.index(',') - 1))
        self.assertIn('0.0636663', spans_of(text))
        text = '경쟁률이 7.31 대 1, 7.94 대 1을 기록했다.'
        self.assertIn('7.31 대 1', spans_of(text))
        text = '90대 노인 A, 50대 간병인 B,'
        self.assertIn('90대 노인 A', spans_of(text))
        text = '말했다.고금 의제매입세액공제란'
        self.assertTrue(end_ok(text, text.index('.') - 1))
        self.assertTrue(start_ok(text, text.index('고금')))
        self.assertIn('고금 의제매입세액공제', spans_of(text))
        self.assertIn('말했다', spans_of(text))

    def test_starts_after_punctuation_delimiters_but_never_inside_words(self):
        text = '7월(1911건)보다 ‘모찌(2,800원)’ 4만~20만원 서울-고양·부산 《책》'
        found = spans_of(text)
        for span in ('1911건', '(1911건)', '모찌', '‘모찌', '2,800원', '20만원', '고양', '부산', '책', '《책》'):
            self.assertIn(span, found)
        for span in ('11건', '만원', '0만원', '양', '석동', '800원', ',800원'):
            self.assertNotIn(span, found)
        hangul = '김석동 전 금융위원장은'
        self.assertFalse(start_ok(hangul, 1)); self.assertFalse(start_ok(hangul, 2))
        self.assertFalse(start_ok('3일 전', 1))  # digit + Hangul is still one word
        self.assertFalse(start_ok('A3', 1))
        self.assertTrue(end_ok('3일 전까지', 3))  # ending inside a Hangul word stays allowed (particles)
        self.assertIn('3일 전', spans_of('3일 전까지'))
        self.assertIn('102%', spans_of('102%를'))

    def test_candidates_keep_word_and_length_limits_and_bracket_rule(self):
        words = ' '.join(f'w{i}' for i in range(12))
        self.assertIn(' '.join(f'w{i}' for i in range(8)), spans_of(words))
        self.assertNotIn(' '.join(f'w{i}' for i in range(9)), spans_of(words))
        self.assertTrue(all(len(span) <= 64 for span in spans_of('가' * 100)))
        self.assertTrue(all('[' not in span and ']' not in span for span in spans_of('참고 [별표 1] 기준')))
        self.assertIn('별표 1', spans_of('참고 [별표 1] 기준'))
        self.assertTrue(all(not span[0].isspace() and not span[-1].isspace() for span in spans_of('가 나  다')))
        out = candidates([('D1', '가 나')])
        self.assertEqual([a for a, s, _ in out if s is None], ['확인 불가', '근거 충돌'])
        self.assertEqual(out[0], ('가 [D1]', '가', 'D1'))


class SharedWithExtractiveTests(unittest.TestCase):
    PASSAGES = [('D1', '지난 7월(1911건)보다 117% 증가했다. KJB 0.0636663, KNB 0.097'),
                ('D2', '경쟁률이 7.31 대 1, 7.94 대 1을 기록했다.말했다.고금 1,000원 전화 1675-3561.')]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.prompt = rag_prompt(self.PASSAGES, '7월은?')
        self.tok = train_tokenizer([{'prompt': self.prompt, 'answer': '1911건 [D1]'}], Path(self.tmp.name) / 'tok.json', 300)

    def test_encode_masks_agree_with_candidates_on_every_passage(self):
        row = encode(self.tok, self.prompt)
        for k, (sid, lo, hi) in enumerate(row['passages']):
            text = self.prompt[lo:hi]
            enumerated = {span for _, span, _ in candidates([(sid, text)], max_words=10 ** 6) if span is not None}
            decodable = set()
            for start in range(lo, hi):
                for end in range(start + 1, min(hi, start + MAX_SPAN) + 1):
                    if not self.prompt[start].isspace() and not self.prompt[end - 1].isspace() and span_issue(row, start, end) is None:
                        decodable.add(self.prompt[start:end])
            self.assertEqual(decodable, enumerated, sid)

    def test_labels_formerly_unreachable_are_placed_and_decodable(self):
        for answer in ('1911건 [D1]', '0.0636663 [D1]', '7.31 대 1 [D2]', '고금 [D2]', '1,000원 [D2]', '1675 [D2]', '3561 [D2]'):
            row = encode(self.tok, self.prompt, answer)
            self.assertTrue(row['label_decodable'], answer)
            self.assertEqual(self.prompt[row['start']:row['end'] + 1], answer.rsplit(' [', 1)[0])
            n = len(row['mask'])
            scores = torch.full((2, n), -50.); scores[0, row['start']] = 100; scores[1, row['end']] = 100
            self.assertEqual(decode((torch.tensor([10., -10., -10.]), scores), row)['answer'], answer)
        for answer, reason in (('11건 [D1]', 'start_inside_word'), ('1 [D1]', 'end_splits_run'), ('1,0 [D2]', 'end_splits_run'),
                               ('000원 [D2]', 'start_inside_word'), ('0636663 [D1]', 'start_inside_word'), ('7.3 [D2]', 'end_splits_run')):
            row = encode(self.tok, self.prompt, answer)
            self.assertEqual((row['label_decodable'], row['label_reason']), (False, reason), answer)

    def test_best_span_never_cuts_a_number_even_with_extreme_scores(self):
        row = encode(self.tok, self.prompt)
        n = len(row['mask'])
        for value, cut in (('1,000', 1), ('1,000', 2), ('0.0636663', 1), ('0.0636663', 2), ('1911건', 1), ('1675-3561', 2)):
            at = self.prompt.index(value)
            scores = torch.full((2, n), -50.); scores[0, at] = 100; scores[1, at + cut - 1] = 100
            start, end, _ = best_span(scores, row)
            self.assertNotEqual((start, end), (at, at + cut), (value, cut))
            scores = torch.full((2, n), -50.); scores[0, at + cut] = 100; scores[1, at + len(value) - 1] = 100
            start, end, _ = best_span(scores, row)
            self.assertNotEqual((start, end), (at + cut, at + len(value)), (value, cut))

    def test_read_reports_prompt_offsets_and_policy(self):
        cfg = Config(vocab_size=self.tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=512)
        model = ExtractiveReader(cfg).eval()
        result = read(model, self.tok, self.prompt, 'cpu')
        self.assertEqual(result['boundary_policy'], BOUNDARY_POLICY)
        if result['span'] is not None:
            start, end = result['span']
            self.assertEqual(self.prompt[start:end], result['span_text'])
        row = encode(self.tok, self.prompt)
        at = self.prompt.index('1911건')
        scores = torch.full((2, len(row['mask'])), -50.); scores[0, at] = 100; scores[1, at + 4] = 100
        out = decode((torch.tensor([10., -10., -10.]), scores), row)
        self.assertEqual((out['span'], out['span_text'], out['source'], out['answer']), ([at, at + 5], '1911건', 'D1', '1911건 [D1]'))


if __name__ == '__main__':
    unittest.main()

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('study_tokenizer', SCRIPTS / 'study_tokenizer.py')
study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)

LINES = ['2026년 3월 15일 1,500,000원 02-123-4567', '숙박비 상한은 서울특별시 10만 원, 광역시 8만 원이다.',
         '제1조(목적) 이 법은 「교통약자의 이동편의 증진법」 제25조에 따른다.\n□ 조사 결과\n○ 저상버스 38.7%',
         'ASCII text, émoji 🙂 and 漢字 mixed; tabs\tand  double  spaces']


def train(pre, vocab=600):
    tok, trainer = study.build_tokenizer(pre, vocab)
    tok.train_from_iterator(LINES * 3, trainer)
    return tok


class NumberSplitTests(unittest.TestCase):
    def test_numbers_split_into_at_most_three_digits(self):
        for pre in ('l3', 'ds'):
            pieces = [p for p, _ in study.PRE_TOKENIZERS[pre]().pre_tokenize_str(LINES[0])]
            digits = [p.lstrip('Ġ') for p in pieces if p.lstrip('Ġ').isdigit()]
            self.assertEqual(digits, ['202', '6', '3', '15', '1', '500', '000', '02', '123', '456', '7'], pre)
            self.assertTrue(all(len(d) <= 3 for d in digits))

    def test_digit_split_pattern_keeps_punctuation_off_the_next_word(self):
        inverse = {v: k for k, v in study.bytes_to_unicode().items()}
        text = '기준율의 102%를 적용한다.\n4. 마감'
        pieces = {pre: [bytes(inverse[c] for c in p).decode('utf-8') for p, _ in study.PRE_TOKENIZERS[pre]().pre_tokenize_str(text)]
                  for pre in ('l3', 'ds')}
        self.assertIn('%를', pieces['l3'])  # Llama-3 pattern glues one symbol to the following word
        self.assertIn('.\n', pieces['l3'])
        self.assertEqual(pieces['ds'], ['기준율의', ' 102', '%', '를', ' 적용한다', '.', '\n', '4', '.', ' 마감'])

    def test_gpt2_regex_keeps_whole_numbers(self):
        pieces = [p.lstrip('Ġ') for p, _ in study.PRE_TOKENIZERS['bl']().pre_tokenize_str(LINES[0])]
        self.assertIn('2026', pieces)
        self.assertIn('500', pieces)
        self.assertIn('4567', pieces)

    def test_trained_tokenizers_round_trip(self):
        for pre in study.PRE_TOKENIZERS:
            tok = train(pre)
            with tempfile.TemporaryDirectory() as d:
                tok.save(str(Path(d) / 'tokenizer.json'))
                tok = Tokenizer.from_file(str(Path(d) / 'tokenizer.json'))
            for line in LINES + ['\n\n  줄바꿈 뒤 공백\n', '미등록 글자 ㉿ ✓ 𝒳']:
                self.assertEqual(tok.decode(tok.encode(line).ids), line, (pre, line))


class BoundaryTests(unittest.TestCase):
    def test_boundaries_are_character_positions_of_token_ends(self):
        tok = train('l3')
        text = LINES[1]
        bounds = study.token_boundaries(tok, text)
        self.assertIn(0, bounds)
        self.assertIn(len(text), bounds)
        self.assertTrue(all(0 <= b <= len(text) for b in bounds))
        # Decoding the prefix up to any boundary reproduces the text prefix exactly.
        ids = tok.encode(text).ids
        prefix = ''
        for i in range(len(ids)):
            prefix = tok.decode(ids[:i + 1])
            self.assertIn(len(prefix), bounds)

    def test_span_alignment_allows_one_leading_space(self):
        text = '상한은 8만 원이다'
        bounds = {0, 3, 6, 9, len(text)}  # tokens: '상한은', ' 8만', ' 원이', '다' (illustrative)
        start = text.index('8만 원')
        self.assertEqual(study.span_alignment(bounds, text, start, start + len('8만 원')), (True, False))
        self.assertEqual(study.span_alignment(bounds, text, start, start + len('8만')), (True, True))

    def test_partial_char_tokens_are_invalid_utf8_fragments(self):
        tok = train('bl')
        partial = study.partial_char_token_ids(tok)
        self.assertTrue(partial)  # a 600-entry vocab cannot hold every syllable, so byte fragments exist
        for tid in partial:
            self.assertNotIn(tok.id_to_token(tid), study.SPECIAL)
        self.assertNotIn(tok.token_to_id('A'), partial)


if __name__ == '__main__':
    unittest.main()

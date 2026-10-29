import unittest
from types import SimpleNamespace

from scripts.build_task_pilot import measure

SPECIAL = ['<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>', '<|eos|>']


class CharTokenizer:
    """One token per character; specials have their own ids."""

    def encode(self, text):
        return SimpleNamespace(ids=[len(SPECIAL) + ord(c) for c in text])

    def token_to_id(self, token):
        return SPECIAL.index(token)


def row(prompt_chars, answer_chars):
    return {'prompt': 'p' * prompt_chars, 'answer': 'a' * answer_chars}


class MeasureTests(unittest.TestCase):
    # prompt tokens = bos + user + text + assistant = chars + 3; example = prompt + answer + eos.
    def test_example_that_fits_keeps_whole_answer_and_eos_label(self):
        m = measure(CharTokenizer(), row(10, 6), context=20)  # 13 + 7 = 20 tokens
        self.assertTrue(m['fits_training'])
        self.assertEqual((m['input_tokens'], m['label_tokens'], m['example_tokens']), (19, 7, 20))
        self.assertTrue(m['labels_end_with_eos'])
        self.assertEqual(m['generation_room'], 7)
        self.assertTrue(m['gold_fits_generation'])

    def test_context_plus_one_trains_but_gold_generation_does_not_fit(self):
        # The final EOS is only a label, so the encoder accepts context + 1 tokens, while the runtime
        # needs room for the answer and its EOS after the prompt.
        m = measure(CharTokenizer(), row(10, 7), context=20)  # 13 + 8 = 21 tokens
        self.assertTrue(m['fits_training'])
        self.assertEqual((m['input_tokens'], m['label_tokens']), (20, 8))
        self.assertEqual((m['generation_room'], m['answer_tokens_with_eos']), (7, 8))
        self.assertFalse(m['gold_fits_generation'])

    def test_longer_example_is_rejected_not_truncated(self):
        m = measure(CharTokenizer(), row(10, 8), context=20)  # 22 tokens
        self.assertFalse(m['fits_training'])
        self.assertIsNone(m['input_tokens'])
        self.assertIsNone(m['label_tokens'])
        self.assertEqual(m['example_tokens'], 22)


if __name__ == '__main__':
    unittest.main()

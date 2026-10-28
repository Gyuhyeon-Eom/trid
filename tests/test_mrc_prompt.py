import json
from pathlib import Path
import unittest
from trid.prompts import korean_mrc_prompt


class MrcPromptTests(unittest.TestCase):
    def test_matches_frozen_reading_training_format(self):
        data = Path(__file__).resolve().parents[1] / 'data/korean-mrc-v1/train.jsonl'
        if not data.exists():
            self.skipTest('Local frozen curriculum not installed')
        with data.open() as source:
            for line in source:
                row = json.loads(line)
                header, content = row['prompt'].split('\n자료: ', 1)
                document, question = content.rsplit('\n질문: ', 1)
                self.assertEqual(korean_mrc_prompt(document, question), row['prompt'])
                self.assertFalse(row.get('history'))
                self.assertTrue(row['task'].startswith('korean_mrc_'))

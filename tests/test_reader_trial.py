import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_reader_trial import completed_training, report_ready, digest


class ReaderTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.run, self.initial, self.data = [root / n for n in ('run', 'initial', 'data')]
        for p in (self.run, self.initial, self.data):
            p.mkdir()
        for p in (self.run / 'reader.safetensors', self.run / 'tokenizer.json', self.initial / 'model.safetensors',
                  self.data / 'train.jsonl', self.data / 'valid.jsonl'):
            p.write_bytes(b'fingerprinted fixture')
        self.meta = dict(status='complete', completed_steps=2000, steps_requested=2000,
                         reader_head='v2', reader_attention='bidirectional', batch_size=16, micro_batch_size=16,
                         learning_rate=0.0001, seed=271029, eval_every=100, context_length=1024,
                         data=str(self.data), initialization=str(self.initial),
                         initial_weights_sha256=digest(self.initial / 'model.safetensors'),
                         tokenizer_sha256=digest(self.run / 'tokenizer.json'),
                         data_sha256={s: digest(self.data / (s+'.jsonl')) for s in ('train', 'valid')})
        self.write_metadata()

    def write_metadata(self):
        (self.run / 'training.json').write_text(json.dumps(self.meta))

    def test_completed_matching_model_can_be_reused_without_mutation(self):
        before = (self.run / 'training.json').read_bytes()
        completed_training(self.run, self.initial, self.data, 'v2')
        self.assertEqual((self.run / 'training.json').read_bytes(), before)
        self.meta['status'] = 'training'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'status differs'):
            completed_training(self.run, self.initial, self.data, 'v2')

    def test_changed_recipe_or_dataset_refuses_reuse(self):
        with self.assertRaisesRegex(ValueError, 'reader_head differs'):
            completed_training(self.run, self.initial, self.data, 'v1')
        (self.data / 'train.jsonl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Training data changed'):
            completed_training(self.run, self.initial, self.data, 'v2')

    def test_report_requires_matching_model_and_complete_rows(self):
        path = self.run / 'report.json'
        report = dict(run=str(self.run), model_sha256='weights', data_sha256=digest(self.data / 'valid.jsonl'),
                      summary={'all': {'total': 2}}, results=[{}, {}])
        path.write_text(json.dumps(report))
        self.assertTrue(report_ready(path, self.run, 'weights', self.data / 'valid.jsonl', 2))
        with self.assertRaisesRegex(ValueError, 'Stale report'):
            report_ready(path, self.run, 'other-weights', self.data / 'valid.jsonl', 2)
        report['results'].pop()
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'Incomplete report'):
            report_ready(path, self.run, 'weights', self.data / 'valid.jsonl', 2)

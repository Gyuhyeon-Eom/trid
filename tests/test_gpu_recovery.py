import json
from pathlib import Path
import tempfile
import unittest
from scripts.recover_gpu_experiments import plan, UNITS


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'trid'
        self.cpt = self.root / 'runs/trid-korean-medium-cpt-v1'
        self.cpt.mkdir(parents=True)
        (self.root / 'reports').mkdir()
        (self.cpt / 'training.json').write_text(json.dumps(dict(status='training', batch_size=32, accumulate=2)))
        (self.cpt / 'latest.json').write_text(json.dumps(dict(checkpoint='step-00000050')))
        checkpoint = self.cpt / 'checkpoints/step-00000050'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'state.json').write_text(json.dumps(dict(step=50)))
        # File presence is the planner's contract; trainer validates tensor contents on load.
        for name in ('model.safetensors', 'optimizer.safetensors'):
            (checkpoint / name).touch()
        for name in ('reader-ablation-8.sh', 'reader-removals.sh'):
            (self.root.parent / name).touch()

    def test_active_jobs_never_duplicate(self):
        self.assertEqual(plan(self.root, set(UNITS)), [])

    def test_checkpoint_step_mismatch_refuses_all_launches(self):
        (self.cpt / 'checkpoints/step-00000050/state.json').write_text('{"step": 49}')
        with self.assertRaisesRegex(ValueError, 'step mismatch'):
            plan(self.root, set())

    def test_partial_reader_requires_review_not_overwrite(self):
        (self.root / 'runs/reader-factorial-01-base-v1-v1-v2').mkdir()
        with self.assertRaisesRegex(ValueError, 'manual review'):
            plan(self.root, set())

    def test_missing_optimizer_refuses_cpt_resume(self):
        (self.cpt / 'checkpoints/step-00000050/optimizer.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'):
            plan(self.root, set())

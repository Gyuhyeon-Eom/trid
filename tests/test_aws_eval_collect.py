import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / 'scripts/aws_eval_collect.py'
SET = [{'id': 'S-1', 'turns': ['안녕하세요', '<NEW_CHAT>', '제 이름이 뭐였죠?']},
       {'id': 'S-2', 'turns': ['점심 추천해 주세요', '매운 건 빼고요']}]


def turns_only(rows):
    return ''.join(json.dumps({'id': r['id'], 'turns': r['turns']}, ensure_ascii=False) + '\n' for r in rows).encode()


class CollectCheckpointsTests(unittest.TestCase):
    """Checkpoints as the AWS-EVAL-006 runner lays them out: P and Q share one checkpoints/ prefix per run."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', str(cls.tmp / 'key.pem'),
                        '-out', str(cls.tmp / 'cert.pem'), '-days', '1', '-subj', '/CN=test'], capture_output=True, check=True)
        (cls.tmp / 'set.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in SET))
        cls.input_sha = hashlib.sha256(turns_only(SET)).hexdigest()

    def checkpoint(self, d, name, alias, sid, answer='답', **status):
        rows = [{'dialogue': sid, 'turn': i, 'question': q, 'output': f'{answer}-{alias}-{i}', 'stop_type': 'eos'}
                for i, q in enumerate(next(r for r in SET if r['id'] == sid)['turns']) if q != '<NEW_CHAT>']
        st = {'package': alias, 'shared_settings_sha256': 's', 'backend': 'cuda', 'render_parity_token_ids_equal': True,
              'effective_settings_sha256': 'e', 'effective_settings_constant': True, 'run_id': 'run-1',
              'input_sha256': self.input_sha, **status}
        body = json.dumps({'alias': alias, 'dialogue': sid, 'rows': rows, 'status': st}, ensure_ascii=False).encode()
        out = subprocess.run(['openssl', 'cms', '-encrypt', '-binary', '-aes256', '-outform', 'DER', str(self.tmp / 'cert.pem')],
                             input=body, capture_output=True, check=True).stdout
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(out)

    def collect(self, d, alias='P', run_ids=('run-1',)):
        out = Path(tempfile.mkdtemp()) / 'run'
        args = [sys.executable, str(COLLECT), '--checkpoints', str(d), '--key', str(self.tmp / 'key.pem'),
                '--set', str(self.tmp / 'set.jsonl'), '--alias', alias, '--out', str(out)]
        for r in run_ids:
            args += ['--run-id', r]
        r = subprocess.run(args, capture_output=True, text=True)
        return r.returncode, json.loads(r.stdout.splitlines()[-1]), out

    def full_run(self, d):
        for n, sid in enumerate(('S-1', 'S-2'), 1):
            self.checkpoint(d, f'P-{n:03d}.cms', 'P', sid)
            self.checkpoint(d, f'Q-{n:03d}.cms', 'Q', sid)

    def test_shared_prefix_collects_only_the_requested_alias(self):
        d = Path(tempfile.mkdtemp())
        self.full_run(d)
        code, report, out = self.collect(d, 'Q')
        self.assertEqual((code, report['status'], report['turns']), (0, 'complete', 4))
        outputs = [json.loads(l)['output'] for l in (out / 'outputs.jsonl').read_text().splitlines()]
        self.assertTrue(all('-Q-' in o for o in outputs))

    def test_checkpoint_from_another_run_is_refused(self):
        d = Path(tempfile.mkdtemp())
        self.checkpoint(d, 'P-001.cms', 'P', 'S-1')
        self.checkpoint(d, 'P-002.cms', 'P', 'S-2', run_id='run-2')  # complementary half of a different run
        code, report, _ = self.collect(d)
        self.assertEqual(code, 2)
        self.assertIn('run id', report['problem'])

    def test_checkpoint_of_another_delivered_set_is_refused(self):
        d = Path(tempfile.mkdtemp())
        self.checkpoint(d, 'P-001.cms', 'P', 'S-1', input_sha256=hashlib.sha256(b'smoke').hexdigest())
        code, report, _ = self.collect(d)
        self.assertEqual(code, 2)
        self.assertIn('delivered input', report['problem'])

    def test_drifted_effective_settings_are_refused(self):
        d = Path(tempfile.mkdtemp())
        self.checkpoint(d, 'P-001.cms', 'P', 'S-1', effective_settings_constant=False)
        code, report, _ = self.collect(d)
        self.assertEqual(code, 2)
        self.assertIn('not constant', report['problem'])

    def test_file_name_alias_must_match_content(self):
        d = Path(tempfile.mkdtemp())
        self.checkpoint(d, 'Q-001.cms', 'P', 'S-1')
        code, report, _ = self.collect(d, 'Q')
        self.assertEqual(code, 2)
        self.assertIn('differs from the file name', report['problem'])


if __name__ == '__main__':
    unittest.main()

"""AWS-EVAL-006 end-to-end contract check on local files (CPU only, no AWS, no model): NOT a GPU or AWS success.

Runs the real pieces on the public 20 set: the instance-side delivery key (same openssl command as bootstrap.sh),
C's encrypted chunked delivery (c_deliver_set: CMS, base64, CHUNK, the same shell command strings, executed by bash),
instance-side decrypt and sha check, the unmodified scripts/chat_blind_runner.py main() for P and Q writing encrypted
checkpoints into one shared directory, then scripts/aws_eval_collect.py for each alias and scripts/chat_blind_pack.py.
Only the llama-server generation is replaced by fixed public answers (CHAT-CAPABILITY-005/006 public runs), the memory
guard is held at level 1 and a throwaway package directory stands in for the sealed P/Q packages (never opened).

Expectations (exit 1 if any fails): delivery leaves no plaintext or plaintext base64 in command parameters and decrypts
to the same sha; no leftover temporary checkpoint files; both aliases collect complete (63 turns each) from the shared
directory and pack; a second run's checkpoints mixed in are refused for their run id; a run whose runner reports
changed effective settings is refused for that reason. By default a throwaway key pair stands in for C's checkpoint key, so anyone can run it.

  .venv/bin/python scripts/aws_eval_e2e_check.py [--run-id-env TRID_RUN_ID] [--c-cert PEM --c-key PEM]
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/aws_eval')]
PUB = ROOT / 'reports/coordination/CHAT-EVAL-004/public-20.jsonl'
RAW = {'P': ROOT / 'reports/coordination/CHAT-CAPABILITY-005/public-20-run/turns.jsonl',
       'Q': ROOT / 'reports/coordination/CHAT-CAPABILITY-006/public-20-run/turns.jsonl'}


def sh(b):
    return hashlib.sha256(b).hexdigest()


def deliver(payload, inbox, chunk):
    inbox.mkdir(parents=True)
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:3072', '-nodes', '-keyout', str(inbox / 'key.pem'),
                    '-out', str(inbox / 'input-cert.pem'), '-days', '2', '-subj', '/CN=e2e set delivery'], capture_output=True, check=True)
    cipher = subprocess.run(['openssl', 'cms', '-encrypt', '-binary', '-aes256', '-outform', 'DER', str(inbox / 'input-cert.pem')],
                            input=payload, capture_output=True, check=True).stdout
    enc = base64.b64encode(cipher).decode()
    cmds = [f"printf '%s' '{enc[i:i + chunk]}' >> /run/trid-input/set.cms.b64" for i in range(0, len(enc), chunk)]
    cmds.append('touch /run/trid-input/set.cms.b64.ready')
    first_q = json.loads(payload.decode().splitlines()[0])['turns'][0]
    leak = any(first_q in c or base64.b64encode(payload[:48]).decode() in c for c in cmds)
    for c in cmds:
        subprocess.run(['bash', '-c', c.replace('/run/trid-input', str(inbox))], check=True)
    # stdin form: GNU and BSD base64 both accept it
    subprocess.run(['bash', '-c', f'base64 -d < "{inbox}/set.cms.b64" > "{inbox}/set.cms" && openssl cms -decrypt -binary '
                    f'-inform DER -in "{inbox}/set.cms" -inkey "{inbox}/key.pem" -out "{inbox}/set.jsonl"'], check=True)
    return {'commands': len(cmds), 'cipher_bytes': len(cipher), 'plaintext_in_params': leak,
            'decrypted_sha_matches': sh((inbox / 'set.jsonl').read_bytes()) == sh(payload)}, inbox / 'set.jsonl'


def run_alias(runner, base, alias, dialogues, out, ckdir, cert, packages, drift_after=None):
    raw = {(r['dialogue'], int(r['turn'])): r for r in (json.loads(l) for l in RAW[alias].read_text().splitlines())}

    def fixed_run(model_file, sampling, system, tk, hf, dlg, port, ctx, mt, seed, log, write_turn, manifest, end_dialogue=None):
        manifest['render_parity_with_official'] = {'token_ids_equal': True}
        effective, constant = {'fixed_public_answers': True}, True
        for n, (name, turns) in enumerate(dlg.items(), 1):
            for i, q in enumerate(turns):
                if q in base.NEW_CONVERSATION:
                    continue
                r = raw[(name, i)]
                write_turn({'dialogue': name, 'turn': i, 'question': q, 'output': r['output'], 'stop_type': r['stop_type']})
            if drift_after and n > drift_after:
                constant = False
            if end_dialogue:
                end_dialogue(name, effective, constant)
        manifest.update(status='complete', effective_settings_constant=constant)

    runner.run, runner.PACKAGES = fixed_run, packages
    runner.memory_state = lambda: {'pressure_level': 1, 'free_pct': 50}
    sys.argv = ['chat_blind_runner.py', '--package', alias, '--dialogues', str(dialogues), '--out', str(out),
                '--checkpoint-dir', str(ckdir), '--checkpoint-cert', str(cert)]
    runner.main()


def collect(ckdir, key, alias, run_id, out):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/aws_eval_collect.py'), '--checkpoints', str(ckdir), '--key', str(key),
                        '--set', str(PUB), '--alias', alias, '--run-id', run_id, '--out', str(out)], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout.splitlines()[-1]) if r.stdout.strip() else {'stderr': r.stderr[-300:]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id-env', default='TRID_RUN_ID', help='environment variable the runner reads its run id from')
    p.add_argument('--c-cert', type=Path)
    p.add_argument('--c-key', type=Path)
    p.add_argument('--keep', action='store_true', help='keep the work directory')
    a = p.parse_args()
    os.environ.update(TRID_RUNNER_REMOTE='1', TRID_BACKEND='llama.cpp-8ca4060-cuda-e2e-fixture')
    import base_dialogue_runner as base
    import c_deliver_set
    import chat_blind_runner as runner
    w = Path(tempfile.mkdtemp(prefix='aws-eval-e2e-'))
    cert, key = a.c_cert, a.c_key
    if not cert:
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', str(w / 'c-key.pem'),
                        '-out', str(w / 'c-cert.pem'), '-days', '1', '-subj', '/CN=e2e C stand-in'], capture_output=True, check=True)
        cert, key = w / 'c-cert.pem', w / 'c-key.pem'
    packages = w / 'packages'
    for al in 'PQ':
        (packages / al).mkdir(parents=True)
        (packages / al / 'model.gguf').write_bytes(b'e2e-' + al.encode())
        (packages / al / 'package.json').write_text(json.dumps({'expected_sha256': sh(b'e2e-' + al.encode()), 'sampling': {}}))
    rows = [json.loads(l) for l in PUB.read_text().splitlines()]
    payload = ''.join(json.dumps({'id': r['id'], 'turns': r['turns']}, ensure_ascii=False) + '\n' for r in rows).encode()
    checks, report = {}, {'note': 'fixed public answers instead of model generation; not a GPU or AWS success'}
    report['delivery'], delivered = deliver(payload, w / 'inbox', c_deliver_set.CHUNK)
    checks['delivery'] = report['delivery']['decrypted_sha_matches'] and not report['delivery']['plaintext_in_params']

    shared = w / 'run-1/checkpoints'
    os.environ[a.run_id_env] = 'e2e-run-1'
    for al in 'PQ':
        run_alias(runner, base, al, delivered, w / f'run-1/out/{al}', shared, cert, packages)
    files = sorted(f.name for f in shared.iterdir())
    report['checkpoints'] = {'cms': sum(f.endswith('.cms') for f in files), 'other_files': [f for f in files if not f.endswith('.cms')]}
    checks['no_leftover_temporary_files'] = not report['checkpoints']['other_files']
    for al in 'PQ':
        code, out = collect(shared, key, al, 'e2e-run-1', w / f'collected/{al}')
        report[f'collect_{al}'] = out
        checks[f'collect_{al}_complete'] = code == 0 and out.get('status') == 'complete' and out.get('turns') == 63
    if checks['collect_P_complete'] and checks['collect_Q_complete']:
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/chat_blind_pack.py'), 'pack', '--set', str(PUB),
                            '--run', f'P={w}/collected/P', '--run', f'Q={w}/collected/Q', '--packet', str(w / 'packet.jsonl'),
                            '--key', str(w / 'pack-key.json')], capture_output=True, text=True)
        checks['pack'] = r.returncode == 0
    else:
        checks['pack'] = False

    os.environ[a.run_id_env] = 'e2e-run-2'
    first5 = w / 'first5.jsonl'
    first5.write_bytes(b''.join(delivered.read_bytes().splitlines(keepends=True)[:5]))
    run_alias(runner, base, 'P', first5, w / 'run-2/out/P', w / 'run-2/checkpoints', cert, packages)
    mixed = w / 'mixed'
    shutil.copytree(shared, mixed)
    for f in (w / 'run-2/checkpoints').iterdir():
        shutil.copy(f, mixed / f.name)  # run-2 P-001..005 replace run-1's: a complete-looking mix of two runs
    code, out = collect(mixed, key, 'P', 'e2e-run-1', w / 'collected/mixed')
    report['mixed_runs'] = out
    checks['mixed_runs_refused'] = code == 2 and "'e2e-run-2'" in out.get('problem', '')  # refused for the other run, not for something else

    os.environ[a.run_id_env] = 'e2e-run-3'
    run_alias(runner, base, 'P', delivered, w / 'run-3/out/P', w / 'run-3/checkpoints', cert, packages, drift_after=10)
    code, out = collect(w / 'run-3/checkpoints', key, 'P', 'e2e-run-3', w / 'collected/drift')
    report['changed_settings'] = out
    checks['changed_settings_refused'] = code == 2 and 'not constant' in out.get('problem', '')

    report['checks'] = checks
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not a.keep:
        shutil.rmtree(w)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()

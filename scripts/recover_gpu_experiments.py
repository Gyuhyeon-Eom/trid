"""Plan or explicitly execute recovery after an EC2 reboot. Never reboot or delete data."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

UNITS = ('trid-cpt', 'trid-cpt-resumed', 'trid-reader-factorial-01', 'trid-reader-removals-01')


def plan(root, active):
    commands = []
    cpt = root / 'runs/trid-korean-medium-cpt-v1'
    meta = json.loads((cpt / 'training.json').read_text())
    if meta['status'] != 'complete' and not any(u in active for u in UNITS[:2]):
        name = json.loads((cpt / 'latest.json').read_text())['checkpoint']
        if not re.fullmatch(r'step-\d+', name):
            raise ValueError('Invalid checkpoint path')
        checkpoint = cpt / 'checkpoints' / name
        for file in ('model.safetensors', 'optimizer.safetensors', 'state.json'):
            if not (checkpoint / file).is_file():
                raise ValueError(f'Incomplete checkpoint: {checkpoint / file}')
        state = json.loads((checkpoint / 'state.json').read_text())
        if state['step'] != int(name[5:]):
            raise ValueError('Checkpoint step mismatch')
        commands.append(('trid-cpt-resumed', 'medium-cpt-gpu.log', [str(root.parent / 'trid-venv/bin/python'), '-u',
            'scripts/train_medium_cpt_gpu.py', '--init-run', 'runs/trid-korean-medium-base-v1',
            '--output', 'runs/trid-korean-medium-cpt-v1', '--batch-size', str(meta['batch_size']),
            '--accumulate', str(meta['accumulate']), '--stop-after', '500']))
    for unit, prefix, expected, runner in (
        ('trid-reader-factorial-01', 'reader-factorial-01',
         [f'reader-factorial-01-{b}-{h}-{d}' for b in ('base-v1', 'cpt-v1') for h in ('v1', 'v2') for d in ('v2', 'v4')],
         root.parent / 'reader-ablation-8.sh'),
        ('trid-reader-removals-01', 'reader-removals-01',
         [f'reader-removal-01-cpt-v1-v2-without-{d}' for d in ('tydi', 'synthetic-hard', 'klue-hard')],
         root.parent / 'reader-removals.sh')):
        if unit in active:
            continue
        remaining = [n for n in expected if not (root / 'reports' / (n + '.archived')).is_file()]
        if not remaining:
            continue
        for name in remaining:
            attempt = root / 'runs' / name
            if attempt.exists():
                metadata = attempt / 'training.json'
                trained = json.loads(metadata.read_text()) if metadata.is_file() else {}
                if trained.get('status') != 'complete' or not (attempt / 'reader.safetensors').is_file():
                    raise ValueError(f'Reader attempt needs manual review; no optimizer resume available, preserve it: {name}')
                # The trial runner validates recipe/data/model fingerprints before
                # reusing a completed model and resuming its missing evaluations.
        if not runner.is_file():
            raise ValueError(f'Missing previously deployed runner: {runner}')
        commands.append((unit, prefix + '.log', ['/bin/bash', str(runner)]))
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/ubuntu/trid'))
    p.add_argument('--execute', action='store_true', help='without this flag, print a read-only recovery plan')
    a = p.parse_args()
    root = a.root.resolve()
    with (root / 'reports/recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = {u for u in UNITS if subprocess.run(['systemctl', 'is-active', '--quiet', u]).returncode == 0}
        commands = plan(root, active)
        # A manually started trainer is not managed by this recovery protocol.
        processes = subprocess.check_output(['ps', '-eo', 'args='], text=True)
        if not any(u in active for u in UNITS[:2]) and any('python' in line.split()[0] and 'scripts/train_medium_cpt_gpu.py' in line for line in processes.splitlines() if line.split()):
            raise ValueError('Unmanaged CPT process exists; refusing duplicate launch')
        print(json.dumps({'active': sorted(active), 'planned_units': [c[0] for c in commands], 'execute': a.execute}))
        if a.execute:
            if os.geteuid() != 0:
                raise PermissionError('Run with sudo for systemd recovery')
            for unit, log, argv in commands:
                subprocess.run(['systemctl', 'reset-failed', unit], check=False, capture_output=True)
                subprocess.run(['systemd-run', '--unit=' + unit, '--uid=ubuntu', '--gid=ubuntu',
                    '--working-directory=' + str(root), '-E', 'PYTHONPATH=' + str(root),
                    '-p', 'StandardOutput=append:' + str(root / 'reports' / log),
                    '-p', 'StandardError=append:' + str(root / 'reports' / log), *argv], check=True)


if __name__ == '__main__':
    main()

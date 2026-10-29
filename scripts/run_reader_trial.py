"""Run one fixed reader comparison; resume evaluation only for a matching completed model.

Invoke with Python 3.12 (CPU evaluation environment). CUDA training uses --train-python
so an existing experiment's training runtime is not silently changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

BUCKET = 's3://trid-training-148761639313'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def completed_training(run, initial, data, head):
    meta = json.loads((run / 'training.json').read_text())
    expected = dict(status='complete', completed_steps=2000, steps_requested=2000,
                    reader_head=head, reader_attention='bidirectional', batch_size=16,
                    micro_batch_size=16, learning_rate=0.0001, seed=271029, eval_every=100,
                    context_length=1024)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'{run}: {key} differs; preserve and review existing attempt')
    if Path(meta['data']).resolve() != data.resolve() or Path(meta['initialization']).resolve() != initial.resolve():
        raise ValueError('Training input paths differ')
    if meta['initial_weights_sha256'] != digest(initial / 'model.safetensors'):
        raise ValueError('Initial model changed')
    if meta['tokenizer_sha256'] != digest(run / 'tokenizer.json'):
        raise ValueError('Tokenizer changed')
    if meta['data_sha256'] != {split: digest(data / (split + '.jsonl')) for split in ('train', 'valid')}:
        raise ValueError('Training data changed')
    if not (run / 'reader.safetensors').is_file():
        raise ValueError('Completed model weights missing')


def report_ready(path, run, weights_hash, data, count):
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    if (Path(report['run']).resolve() != run.resolve() or report['model_sha256'] != weights_hash
            or report['data_sha256'] != digest(data)):
        raise ValueError(f'Stale report: {path}')
    if report['summary']['all']['total'] != count or len(report['results']) != count:
        raise ValueError(f'Incomplete report: {path}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--init-run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--head', choices=('v1', 'v2'), required=True)
    p.add_argument('--train-python', default='/home/ubuntu/trid-venv/bin/python')
    a = p.parse_args()
    if sys.version_info[:2] != (3, 12):
        p.error('Use the Python 3.12 evaluation environment')
    if Path(a.name).name != a.name or a.name in ('.', '..'):
        p.error('name must be one path component')
    run = Path('runs') / a.name
    reports = Path('reports')
    marker = reports / (a.name + '.archived')
    if marker.exists():
        print(f'Already archived: {a.name}', flush=True)
        return
    if not run.exists():
        if shutil.disk_usage('.').free < 1500 * 1024**2:
            raise RuntimeError('Need 1.5 GiB free before training')
        with (reports / (a.name + '.log')).open('x') as log:
            subprocess.run([a.train_python, '-u', 'scripts/train_extractive.py', '--init-run', str(a.init_run),
                '--data', str(a.data), '--output', str(run), '--attention', 'bidirectional', '--head', a.head,
                '--steps', '2000', '--batch-size', '16', '--micro-batch-size', '16', '--lr', '0.0001',
                '--seed', '271029', '--device', 'cuda', '--context', '1024', '--eval-every', '100', '--threads', '4'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
    completed_training(run, a.init_run, a.data, a.head)
    print(f'Completed training validated; evaluating {a.name}', flush=True)
    weights_hash = digest(run / 'reader.safetensors')
    dev = reports / (a.name + '-dev.json')
    probe = reports / (a.name + '-probe.json')
    for output, script, data, count in (
        (dev, 'evaluate_extractive_dev.py', Path('data/korean-rag-curriculum-v2/valid.jsonl'), 480),
        (probe, 'evaluate_institution_probe.py', Path('data/institution-qa-frozen-01/test.jsonl'), 40)):
        if not report_ready(output, run, weights_hash, data, count):
            subprocess.run([sys.executable, 'scripts/' + script, '--run', str(run), '--output', str(output), '--threads', '4'], check=True)
            report_ready(output, run, weights_hash, data, count)
    analysis = reports / (a.name + '-dev-analysis.json')
    if analysis.exists():
        if json.loads(analysis.read_text())['report_sha256'] != digest(dev):
            raise ValueError('Stale dev analysis')
    else:
        subprocess.run([sys.executable, 'scripts/analyze_extractive_dev.py', '--report', str(dev), '--output', str(analysis)], check=True)
    subprocess.run(['aws', 's3', 'sync', str(run), BUCKET + '/' + str(run), '--only-show-errors'], check=True)
    subprocess.run(['aws', 's3', 'sync', 'reports', BUCKET + '/reports', '--exclude', '*', '--include', a.name + '*', '--only-show-errors'], check=True)
    files = list(run.glob('*.safetensors'))
    hashes = {}
    for path in files:
        local = digest(path)
        with subprocess.Popen(['aws', 's3', 'cp', BUCKET + '/' + str(path), '-', '--only-show-errors'], stdout=subprocess.PIPE) as proc:
            remote = hashlib.file_digest(proc.stdout, 'sha256').hexdigest()
            if proc.wait() or remote != local:
                raise ValueError(f'Remote weight verification failed: {path}')
        hashes[path.name] = local
    # All remote weights are verified before deleting any. Marker first: an interruption
    # during cleanup can leave extra local files, but never forces retraining or loses weights.
    with marker.open('x') as stream:
        json.dump({'s3': BUCKET + '/' + str(run), 'sha256': hashes}, stream, indent=2)
    for path in files:
        path.unlink()
    print(f'Archived and verified: {a.name}', flush=True)


if __name__ == '__main__':
    main()

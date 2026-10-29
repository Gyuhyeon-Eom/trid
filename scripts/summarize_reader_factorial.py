"""Summarize completed paired dev/probe reports; never treat missing runs as zero."""
import argparse
import itertools
import json
from pathlib import Path


def collect(reports, prefix):
    rows, pending = [], []
    fingerprints = None
    for backbone, head, data in itertools.product(('base-v1', 'cpt-v1'), ('v1', 'v2'), ('v2', 'v4')):
        name = f'{prefix}-{backbone}-{head}-{data}'
        paths = [reports / f'{name}-{kind}.json' for kind in ('dev', 'probe')]
        if not all(p.exists() for p in paths):
            pending.append(name)
            continue
        dev, probe = [json.loads(p.read_text()) for p in paths]
        if dev['model_sha256'] != probe['model_sha256']:
            raise ValueError(f'Model mismatch: {name}')
        if any(Path(r['run']).name != name for r in (dev, probe)):
            raise ValueError(f'Run mismatch: {name}')
        current = (dev['data_sha256'], probe['data_sha256'], probe['source_docs_sha256'])
        if fingerprints is not None and current != fingerprints:
            raise ValueError(f'Evaluation data mismatch: {name}')
        fingerprints = current
        d, p = dev['summary']['all'], probe['summary']['all']
        if d['total'] != 480 or p['total'] != 40:
            raise ValueError(f'Incomplete evaluation: {name}')
        if len(dev['results']) != 480 or len(probe['results']) != 40:
            raise ValueError(f'Result count mismatch: {name}')
        if sum(bool(r['strict']) for r in dev['results']) != d['strict']:
            raise ValueError(f'Dev summary mismatch: {name}')
        if sum(bool(r['exact']) for r in probe['results']) != p['exact']:
            raise ValueError(f'Probe summary mismatch: {name}')
        rows.append({'name': name, 'backbone': backbone, 'head': head, 'data': data,
                     'dev_correct': d['strict'], 'dev_total': d['total'],
                     'probe_correct': p['exact'], 'probe_total': p['total'],
                     'model_sha256': dev['model_sha256']})
    return {'complete': len(rows) == 8, 'results': rows, 'pending': pending,
            'evaluation_hashes': fingerprints,
            'note': 'Fixed-step comparison, not equal epochs. Probe is reused diagnostic data; no automatic model selection.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, default=Path('reports'))
    p.add_argument('--prefix', default='reader-factorial-01')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = collect(a.reports, a.prefix)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output:
        with a.output.open('x') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()

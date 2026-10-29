"""Remove one v4 training component without regenerating any retained examples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

COMPONENTS = {
    'tydi': {'rag_tydi'},
    'synthetic-hard': {'rag_missing_hard', 'rag_value_hard'},
    'klue-hard': {'rag_klue_hard', 'rag_klue_missing_hard'},
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(source, output, component):
    provenance = json.loads((source / 'provenance.json').read_text())
    for split, expected in provenance['hashes'].items():
        if digest(source / f'{split}.jsonl') != expected:
            raise ValueError(f'Source hash mismatch: {split}')
    removed = Counter()
    retained = []
    for line in (source / 'train.jsonl').read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        if row['task'] in COMPONENTS[component]:
            removed[row['task']] += 1
        else:
            retained.append(line)
    if not removed:
        raise ValueError(f'No rows for component {component}')
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('train.jsonl', 'provenance.json'):
            shutil.copyfile(path, output / path.name)
    (output / 'train.jsonl').write_bytes(b''.join(retained))
    provenance['ablation'] = {'source': str(source.resolve()),
        'source_hashes': dict(provenance['hashes']), 'removed_component': component,
        'removed_tasks': dict(removed), 'retained_rows_byte_identical': True}
    provenance['rows']['train'] = len(retained)
    provenance['hashes']['train'] = digest(output / 'train.jsonl')
    # Parent generator counts describe the source, not this subset.
    for key in ('extra_rows', 'extra_tasks', 'extra_hard_modes', 'extra_passage_counts',
                'rag_train_before', 'rag_train_after', 'counts', 'extractive_trainer_diagnostic'):
        provenance.pop(key, None)
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    return {'output': str(output), 'train_rows': len(retained), 'removed': dict(removed),
            'valid_sha256': digest(output / 'valid.jsonl')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/korean-rag-curriculum-v4'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remove', choices=COMPONENTS, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output, a.remove), ensure_ascii=False))


if __name__ == '__main__':
    main()

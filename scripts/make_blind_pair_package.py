"""Blind paired package of two models' pilot_dev answers for C scoring (C-019 section 8).

Per item the two answers are shown as `answer_1` / `answer_2` in a random order (cryptographic randomness); model
names are removed. The item-to-model mapping is written only to --mapping (kept by A, outside the shared package); the
package carries the mapping's sha256 as a commitment so the mapping cannot be changed after scoring.

  python scripts/make_blind_pair_package.py --a <pilot-dev report> --b <pilot-dev report> --pilot <pilot.jsonl> \
      --package <shared json> --mapping <A-only json>
"""
import argparse
import hashlib
import json
import secrets
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    a = p.parse_args()
    if a.package.exists() or a.mapping.exists():
        p.error('Use new output files')
    reports = {name: json.loads(path.read_text()) for name, path in (('A', a.a), ('B', a.b))}
    answers = {name: {r['id']: r for r in rep['results']} for name, rep in reports.items()}
    if answers['A'].keys() != answers['B'].keys():
        raise ValueError('The two reports cover different items')
    rows = {r['id']: r for r in (json.loads(line) for line in a.pilot.read_text().splitlines() if line.strip())}
    items, mapping = [], {'models': {name: {'label': rep['label'], 'model_sha256': rep['model_sha256']}
                                     for name, rep in reports.items()}, 'items': {}}
    order = sorted(answers['A'])
    for i in range(len(order) - 1, 0, -1):  # shuffle item order too
        j = secrets.randbelow(i + 1)
        order[i], order[j] = order[j], order[i]
    for n, item_id in enumerate(order, 1):
        first = 'A' if secrets.randbelow(2) == 0 else 'B'
        second = 'B' if first == 'A' else 'A'
        row = rows[item_id]
        blind_id = f'pair-{n:02d}'
        items.append({'pair': blind_id, 'case_id': item_id, 'task_type': row['task_type'], 'behavior': row['behavior'],
                      'time_label': row.get('time_label'), 'question': row['question'], 'context': row['context'],
                      'answer_1': answers[first][item_id]['generated'], 'answer_2': answers[second][item_id]['generated']})
        mapping['items'][blind_id] = {'answer_1': first, 'answer_2': second}
    mapping_text = json.dumps(mapping, ensure_ascii=False, indent=1)
    a.mapping.parent.mkdir(parents=True, exist_ok=True)
    a.mapping.write_text(mapping_text)
    package = {
        'protocol': 'C-019 section 8 blind comparison. Score each answer from the source context only: '
                    '(1) conclusion correct, (2) conditions/exceptions kept, (3) unsupported claims (invented numbers or '
                    'articles), (4) appropriateness of hold / ask-back / lookup-needed, (5) time labelling; 0/1 each with '
                    'a note. Answers are unlabelled and their order is random per pair. Do not put these outputs or '
                    'notes into any training data.',
        'pilot_sha256': hashlib.sha256(a.pilot.read_bytes()).hexdigest(),
        'mapping_sha256_commitment': hashlib.sha256(mapping_text.encode()).hexdigest(),
        'generation': 'llama.cpp 8ca4060, Q8_0, greedy, thinking off, context 8192, 512 output tokens, CASE_SYSTEM',
        'pairs': items,
    }
    a.package.parent.mkdir(parents=True, exist_ok=True)
    a.package.write_text(json.dumps(package, ensure_ascii=False, indent=1))
    print(f'{len(items)} pairs -> {a.package}; mapping (A only) -> {a.mapping}')


if __name__ == '__main__':
    main()

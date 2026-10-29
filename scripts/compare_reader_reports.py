"""Paired dev comparison using the existing strict-answer error taxonomy."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from scripts.analyze_extractive_dev import digest, expected_rows, validate_results, categorize


def compare(before, after, data):
    fingerprint = digest(data)
    if any(r['data_sha256'] != fingerprint for r in (before, after)):
        raise ValueError('Evaluation data hash mismatch')
    expected = expected_rows([json.loads(line) for line in data.read_text().splitlines()])
    if not expected:
        raise ValueError('Empty evaluation')
    for report in (before, after):
        validate_results(report['results'], expected)
        if report['evaluated_rows'] != len(expected):
            raise ValueError('Evaluation denominator mismatch')
    previous = {r['id']: r for r in before['results']}
    counts, transitions, details = Counter(), Counter(), []
    for new in after['results']:
        old = previous[new['id']]
        a, b = bool(old['strict']), bool(new['strict'])
        outcome = 'both_correct' if a and b else 'regressed' if a else 'improved' if b else 'both_wrong'
        counts[outcome] += 1
        left = categorize(old, expected[new['id']])
        right = categorize(new, expected[new['id']])
        label = lambda value: ':'.join(str(v) for v in value if v is not None)
        transitions[f'{label(left)} -> {label(right)}'] += 1
        if old.get('generated') != new.get('generated') or left != right:
            details.append({'id': new['id'], 'task': new['task'], 'outcome': outcome,
                            'before_category': left, 'after_category': right,
                            'gold': new['answer'], 'before': old.get('generated'), 'after': new.get('generated')})
    improved, regressed = counts['improved'], counts['regressed']
    discordant = improved + regressed
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2**discordant)
    return {'before_run': before['run'], 'after_run': after['run'], 'data_sha256': fingerprint,
            'before_model_sha256': before['model_sha256'], 'after_model_sha256': after['model_sha256'],
            'total': len(expected), 'paired_counts': {k: counts[k] for k in ('both_correct', 'improved', 'regressed', 'both_wrong')},
            'net_accuracy_change_pp': 100 * (improved-regressed)/len(expected),
            'mcnemar_exact_two_sided_p': p, 'transitions': dict(sorted(transitions.items())), 'changed_rows': details,
            'limitations': 'Development-set diagnostic, no deployment claim. Raw paired p-value is unadjusted for multiple comparisons; report all pairs, do not use as an automatic selection gate.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/korean-rag-curriculum-v2/valid.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(json.loads(a.before.read_text()), json.loads(a.after.read_text()), a.data)
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('total', 'paired_counts', 'net_accuracy_change_pp', 'mcnemar_exact_two_sided_p')}))


if __name__ == '__main__':
    main()

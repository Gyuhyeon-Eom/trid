"""Offline analysis of one scripts/evaluate_extractive_dev.py report against the preregistered reader gates.

No inference: the report's raw `generated` strings are re-parsed with the service answer grammar
(trid.extractive.parse_answer) and the development rows are re-read from the report's own data path, so every
result row is checked against its gold row (id, task, gold answer, strict consistency) before any number is
computed. Missing, duplicated or unexpected result ids reject the report instead of shrinking a denominator;
unsupported combination rows stay in every total. Gates come from the protocol file, never from this script,
and a report whose run or data differs from the protocol is analysed as a historical baseline only
(`run_matches_protocol` false, `protocol_candidate_accepted` false) whatever its numbers.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.extractive import parse_answer
from trid.prompts import rag_passages

PROJECT = Path(__file__).resolve().parents[1]
KIND_NAMES = ('present', 'missing', 'conflict')
KLUE_TASKS = ('rag_klue', 'rag_klue_missing')
GATE_KEYS = {  # protocol key -> (metric path, comparison)
    'strict_accuracy_at_least': (('all', 'strict_rate'), '>='),
    'false_presence_rate_on_each_missing_and_conflict_at_most': ((('missing', 'false_presence_rate'),
                                                                 ('conflict', 'false_presence_rate')), '<='),
    'conflict_strict_accuracy_at_least': (('conflict', 'strict_rate'), '>='),
    'klue_strict_accuracy_at_least': (('klue', 'strict_rate'), '>='),
    'multi_passage_reachable_present_strict_accuracy_at_least': (('multi_passage_present', 'strict_rate'), '>='),
    'reachable_present_wrong_abstention_rate_at_most': (('reachable_present', 'wrong_abstention_rate'), '<='),
    'execution_exceptions': (('all', 'exceptions'), '=='),
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def resolve(path):
    """Report/protocol paths are stored as written on the command line, i.e. relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def expected_rows(raw):
    """Same selection as evaluate_extractive_dev: RAG groups without history, keyed by id, with derived facts."""
    rows = {}
    for row in raw:
        if row.get('group') not in ('rag', 'rag_real') or row.get('history'):
            continue
        if row['id'] in rows:
            raise ValueError(f"duplicate development id {row['id']}")
        parsed = parse_answer(row['answer'])
        rows[row['id']] = dict(task=row['task'], answer=row['answer'], parsed=parsed,
                               expected_presence=parsed[0] if parsed is not None else (0 if row['task'] == 'rag_combine' else None),
                               reachable_present=parsed is not None and parsed[0] == 0,
                               passages=len(rag_passages(row['prompt'])))
    return rows


def validate_results(results, expected):
    """Raise ValueError on any id/label/consistency problem; never drop or repair a row."""
    ids = [r.get('id') for r in results]
    counts = Counter(ids)
    duplicates = sorted(i for i, n in counts.items() if n > 1)
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(i for i in counts if i not in expected)
    if duplicates or missing or unexpected:
        raise ValueError(f'result ids differ from development rows: missing={missing[:5]} ({len(missing)}), '
                         f'duplicate={duplicates[:5]} ({len(duplicates)}), unexpected={unexpected[:5]} ({len(unexpected)})')
    for r in results:
        row = expected[r['id']]
        if r.get('task') != row['task'] or r.get('answer') != row['answer']:
            raise ValueError(f"result {r['id']} task/gold differs from development data")
        if r.get('expected_presence') != row['expected_presence']:
            raise ValueError(f"result {r['id']} expected_presence differs from parse_answer(gold)")
        if r.get('exception') is not None:
            if r.get('strict') or r.get('generated') is not None:
                raise ValueError(f"result {r['id']} has an exception but also an answer or strict credit")
            continue
        generated = r.get('generated')
        if not isinstance(generated, str):
            raise ValueError(f"result {r['id']} has neither exception nor generated text")
        if bool(r.get('strict')) != (generated == row['answer'].strip()):
            raise ValueError(f"result {r['id']} strict flag inconsistent with generated == gold.strip()")
        parsed = parse_answer(generated)
        if parsed is not None and r.get('presence') is not None and parsed[0] != r['presence']:
            raise ValueError(f"result {r['id']} presence head field inconsistent with emitted answer kind")


def categorize(result, row):
    """One error category per row, in the fixed priority order of the module docstring; 'correct' when strict."""
    if result.get('strict'):
        return 'correct', None
    if result.get('exception') is not None:
        return 'execution', None
    if row['parsed'] is None:
        return 'unsupported_combine' if row['task'] == 'rag_combine' else 'unsupported_label', None
    generated = parse_answer(result['generated'])
    if generated is None:
        return 'format', 'unparsable'
    expected_kind, gold_value, gold_sid = row['parsed']
    kind, value, sid = generated
    if expected_kind in (1, 2) and kind == 0:
        return 'false_presence', KIND_NAMES[expected_kind]
    if expected_kind == 0 and kind in (1, 2):
        return 'wrong_abstention', ('no_valid_span' if result.get('reason') == 'no_valid_span' else 'head') + ':' + KIND_NAMES[kind]
    if expected_kind in (1, 2) and kind in (1, 2):
        return 'abstention_kind_swap', f'{KIND_NAMES[expected_kind]}->{KIND_NAMES[kind]}'
    if value == gold_value and sid != gold_sid:
        return 'wrong_source_right_value', None
    if value != gold_value and sid == gold_sid:
        return 'wrong_value_right_source', 'overlong' if gold_value in value else 'truncated' if value in gold_value else 'disjoint'
    if value != gold_value and sid != gold_sid:
        return 'wrong_both', None
    return 'format', 'whitespace_or_spacing'  # same value and source, raw string still differs from gold.strip()


def percentile(values, fraction):
    """Nearest-rank percentile; None without values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def rate(count, total):
    return count / total if total else None


def summarize_group(members):
    """members: [(result, row, category, detail)] of one group."""
    total = len(members)
    strict = sum(bool(r.get('strict')) for r, *_ in members)
    exceptions = sum(r.get('exception') is not None for r, *_ in members)
    expected_abstain = sum(row['expected_presence'] in (1, 2) for _, row, *_ in members)
    reachable = sum(row['reachable_present'] for _, row, *_ in members)
    false_presence = sum(c == 'false_presence' for *_, c, _ in members)
    wrong_abstention = sum(c == 'wrong_abstention' for *_, c, _ in members)
    categories = Counter(c for *_, c, _ in members)
    subcategories = Counter(f'{c}:{d}' for *_, c, d in members if d is not None)
    return dict(total=total, strict=strict, strict_rate=rate(strict, total), exceptions=exceptions,
                expected_abstain_rows=expected_abstain, false_presence=false_presence,
                false_presence_rate=rate(false_presence, expected_abstain),
                reachable_present_rows=reachable, wrong_abstention=wrong_abstention,
                wrong_abstention_rate=rate(wrong_abstention, reachable),
                categories=dict(sorted(categories.items())), subcategories=dict(sorted(subcategories.items())))


def group_names(result, row):
    names = ['all', f"task:{row['task']}"]
    if row['reachable_present']:
        names.append('reachable_present')
        if row['passages'] >= 2:
            names.append('multi_passage_present')
    if row['expected_presence'] == 1:
        names.append('missing')
    if row['expected_presence'] == 2:
        names.append('conflict')
    if row['task'] == 'rag_combine':
        names.append('combine')
    if row['task'] in KLUE_TASKS:
        names.append('klue')
    return names


def analyze(results, expected):
    """Validated results + expected rows -> (summary by group, per-row categories)."""
    validate_results(results, expected)
    groups, rows_out = {'all': []}, []  # 'all' always present, even for an empty report
    for result in sorted(results, key=lambda r: r['id']):
        row = expected[result['id']]
        category, detail = categorize(result, row)
        for name in group_names(result, row):
            groups.setdefault(name, []).append((result, row, category, detail))
        rows_out.append(dict(id=result['id'], task=row['task'],
                             expected='unknown_label' if row['expected_presence'] is None else KIND_NAMES[row['expected_presence']],
                             strict=bool(result.get('strict')), category=category, detail=detail,
                             generated=result.get('generated'), gold=row['answer'].strip(),
                             exception=result.get('exception'), reason=result.get('reason')))
    return {name: summarize_group(members) for name, members in sorted(groups.items())}, rows_out


def compare(value, threshold, op):
    if value is None:
        return False
    return {'>=': value >= threshold, '<=': value <= threshold, '==': value == threshold}[op]


def evaluate_gates(summary, protocol, complete):
    registered = protocol['development_gates_registered_before_candidate_results']
    gates = {}
    for key, threshold in registered.items():
        if key not in GATE_KEYS:
            raise ValueError(f'protocol gate {key} unknown to this analysis; extend GATE_KEYS deliberately')
        paths, op = GATE_KEYS[key]
        paths = paths if isinstance(paths[0], tuple) else (paths,)
        observed = {f'{g}.{m}': summary.get(g, {}).get(m) for g, m in paths}
        gates[key] = dict(threshold=threshold, comparison=op, observed=observed,
                          passed=all(compare(v, threshold, op) for v in observed.values()))
    gates['complete_rows'] = dict(threshold=protocol['evaluation']['rag_rows'], comparison='==',
                                  observed={'all.total': summary['all']['total']}, passed=complete)
    return gates


def protocol_mismatches(report, protocol, expected, summary):
    """Reasons the report is not the protocol's candidate run; empty when it is."""
    reasons = []
    if resolve(report['run']) != resolve(protocol['run']):
        reasons.append(f"run {report['run']} is not protocol run {protocol['run']}")
    if resolve(report['data']) != resolve(protocol['evaluation']['data']):
        reasons.append(f"data {report['data']} is not protocol data {protocol['evaluation']['data']}")
    evaluation = protocol['evaluation']
    counts = dict(rag_rows=len(expected),
                  expected_present_rows=sum(r['expected_presence'] == 0 for r in expected.values()),
                  missing_rows=sum(r['expected_presence'] == 1 for r in expected.values()),
                  conflict_rows=sum(r['expected_presence'] == 2 for r in expected.values()),
                  unsupported_combine_rows_included_in_present=sum(r['task'] == 'rag_combine' for r in expected.values()))
    for key, observed in counts.items():
        if evaluation.get(key) != observed:
            reasons.append(f'{key}: protocol {evaluation.get(key)} vs data {observed}')
    limits = protocol.get('structural_limits', {})
    reachable = summary.get('reachable_present', {}).get('total', 0)  # absent group: zero rows, still compared
    if limits.get('reachable_present_rows') != reachable:
        reasons.append(f"reachable_present_rows: protocol {limits.get('reachable_present_rows')} vs data {reachable}")
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--report', type=Path, required=True, help='evaluate_extractive_dev.py output')
    p.add_argument('--protocol', type=Path, default=Path('reports/reader-dev-01-protocol.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=None,
                   help="override the report's recorded valid.jsonl path (same hash still required)")
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path')
    report = json.loads(args.report.read_text())
    protocol = json.loads(args.protocol.read_text())
    if 'results' not in report or 'data_sha256' not in report or 'model_sha256' not in report:
        raise ValueError('Not an evaluate_extractive_dev report')
    data_path = resolve(args.data if args.data is not None else report['data'])
    data_hash = digest(data_path)
    provenance = json.loads((data_path.parent / 'provenance.json').read_text())
    if data_hash != report['data_sha256'] or data_hash != provenance['hashes']['valid']:
        raise ValueError('Development data hash differs from the report and/or provenance')
    raw = [json.loads(line) for line in data_path.read_text().splitlines()]
    expected = expected_rows(raw)
    if report.get('evaluated_rows') != len(expected) or len(report['results']) != len(expected):
        raise ValueError(f"row count mismatch: report evaluated {report.get('evaluated_rows')}, "
                         f"results {len(report['results'])}, development rows {len(expected)}")
    summary, rows = analyze(report['results'], expected)
    complete = len(expected) == protocol['evaluation']['rag_rows'] == summary['all']['total']
    gates = evaluate_gates(summary, protocol, complete)
    mismatches = protocol_mismatches(report, protocol, expected, summary)
    all_pass = all(g['passed'] for g in gates.values())
    seconds = [r['seconds'] for r in report['results'] if isinstance(r.get('seconds'), (int, float))]
    out = dict(protocol=__doc__, report=str(args.report), report_sha256=digest(args.report),
               protocol_path=str(args.protocol), protocol_sha256=digest(args.protocol),
               run=report['run'], model_sha256=report['model_sha256'], data=str(data_path), data_sha256=data_hash,
               run_matches_protocol=not mismatches, protocol_mismatches=mismatches,
               all_gates_pass=all_pass, protocol_candidate_accepted=all_pass and not mismatches,
               acceptance_note=('Gates are the protocol\'s internal engineering gates on the training validation split; '
                                'passing them is not deployment acceptance.' if not mismatches else
                                'Historical/other run analysed for reference only; cannot be the protocol candidate.'),
               strict_definition=protocol['evaluation']['strict'],
               false_presence_definition='emitted answer parses as `<value> [Dn]` on a row whose gold is an abstention',
               wrong_abstention_definition='emitted answer is an abstention on a reachable single-span present row',
               timing=dict(device='cpu', threads=report.get('threads'), rows_timed=len(seconds),
                           p50_seconds=percentile(seconds, .5), p95_seconds=percentile(seconds, .95),
                           note='Per-row CPU latency from the evaluation report; not a serving measurement.'),
               gates=gates, summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(run_matches_protocol=not mismatches, all_gates_pass=all_pass,
                          gates={k: g['passed'] for k, g in gates.items()},
                          all=summary['all'], timing=out['timing']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

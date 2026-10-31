"""CHAT-EVAL-004 next-stage gate aggregator (deterministic; thresholds sealed in reports/coordination/CHAT-EVAL-004/gate.json).

This is a gate for entering the next comparison stage, not an operational pass. Input: a judgment JSONL with one row per
user turn: {"id": scenario id, "turn": 1-based index among user turns (NEW_CHAT markers not counted),
"verdict": "pass"|"partial"|"fail", "severe": [codes], "note": ...}. Every user turn of every scenario in the given
split must be judged exactly once; otherwise the aggregation is refused (exit 3).

Counting (sealed): a turn counts as fully passed only with verdict "pass" and no severe code; "partial" is not in the
numerator. A scenario is fully passed when all its user turns are fully passed. Any severe code anywhere (G1-G4 or the
item's own severe failure) fails the gate regardless of the other numbers. Verdict "pass" needs, for the split:
fully passed turns >= min_turns and fully passed scenarios >= min_scenarios and severe == 0.

  .venv/bin/python scripts/chat_eval_gate.py --gate reports/coordination/CHAT-EVAL-004/gate.json \
      --set reports/coordination/CHAT-EVAL-004/public-20.jsonl --split public --judgments <file>
"""
import argparse
import json
import sys
from pathlib import Path

NEW = '<NEW_CHAT>'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--set', type=Path, required=True)
    p.add_argument('--split', choices=('public', 'private'), required=True)
    p.add_argument('--judgments', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads(a.gate.read_text())
    th = gate['thresholds'][a.split]
    scen = {r['id']: r for r in (json.loads(l) for l in a.set.read_text().splitlines() if l.strip()) if r['split'] == a.split}
    need = {(sid, k) for sid, r in scen.items() for k in range(1, len([t for t in r['turns'] if t != NEW]) + 1)}
    rows = [json.loads(l) for l in a.judgments.read_text().splitlines() if l.strip()]
    got = {}
    problems = []
    for r in rows:
        key = (r['id'], int(r['turn']))
        if key not in need:
            problems.append(f'unexpected judgment {key}')
        elif key in got:
            problems.append(f'duplicate judgment {key}')
        elif r['verdict'] not in ('pass', 'partial', 'fail'):
            problems.append(f'bad verdict {key}: {r["verdict"]}')
        else:
            got[key] = r
    missing = sorted(need - set(got))
    if missing:
        problems.append(f'missing judgments: {len(missing)} (first {missing[:3]})')
    if problems:
        print(json.dumps({'verdict': 'refused', 'problems': problems}, ensure_ascii=False, indent=1))
        sys.exit(3)
    full = {k: (v['verdict'] == 'pass' and not v.get('severe')) for k, v in got.items()}
    severe = [{'id': k[0], 'turn': k[1], 'severe': v['severe']} for k, v in sorted(got.items()) if v.get('severe')]
    turns_full = sum(full.values())
    scen_full = sum(1 for sid in scen if all(full[(sid, k)] for (s, k) in need if s == sid))
    counts = {v: sum(1 for x in got.values() if x['verdict'] == v) for v in ('pass', 'partial', 'fail')}
    ok = turns_full >= th['min_turns'] and scen_full >= th['min_scenarios'] and not severe
    report = {'split': a.split, 'verdict': 'pass' if ok else 'fail', 'turns': len(need), 'turns_fully_passed': turns_full,
              'min_turns': th['min_turns'], 'scenarios': len(scen), 'scenarios_fully_passed': scen_full,
              'min_scenarios': th['min_scenarios'], 'severe_count': len(severe), 'severe': severe, 'verdict_counts': counts,
              'note': 'next-stage gate only; not an operational pass'}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

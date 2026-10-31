"""Preflight gate for CHAT-EVAL-004 (everyday multi-turn chat evaluation). Exit status 2 refuses the run.

Checks (no model is called):
  set      every given scenario file matches the sha256 in the manifest; row counts match
  schema   each scenario has id, category, split, turns (>= 3 user turns), expect aligned with turns, track, severe,
           boundary, external; categories and per-category split counts match the manifest
  unique   unique ids and turn sequences; no user utterance shared with earlier development dialogues
           (BASELINE-COMPARE-002 raw, and any extra JSON given with --old) except short courtesy phrases
  run      with --run-config: model artifact sha256, template, sampling, max_tokens, mode and the scenario set sha256 are
           recorded; mode "service" also needs a passing scripts/eval_artifact_preflight.py manifest (EVAL-PREFLIGHT-001)

  .venv/bin/python scripts/chat_eval_preflight.py --manifest reports/coordination/CHAT-EVAL-004/manifest.json \
      --set reports/coordination/CHAT-EVAL-004/public-20.jsonl [--set <private-80>] [--run-config run.json]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = '<NEW_CHAT>'
FIELDS = ('id', 'category', 'split', 'turns', 'expect', 'track', 'severe', 'boundary', 'external')
COURTESY_MAX = 8  # characters; "고마워요." and similar short phrases may recur
OLD_DEFAULT = ROOT / 'reports/coordination/BASELINE-COMPARE-002/comparison.json'
RUN_FIELDS = ('model_sha256', 'template', 'sampling', 'max_tokens', 'mode', 'scenario_set_sha256')


def norm(t):
    return ''.join(t.split())


def old_utterances(paths):
    out = set()
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        stack = [d]
        while stack:  # collect every 'q' / 'turns' string in any nested structure
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == 'q' and isinstance(v, str):
                        out.add(norm(v))
                    elif k == 'turns' and isinstance(v, list):
                        out.update(norm(t) for t in v if isinstance(t, str) and t != NEW)
                    else:
                        stack.append(v)
            elif isinstance(x, list):
                stack.extend(x)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--set', type=Path, action='append', required=True)
    p.add_argument('--old', type=Path, action='append', default=[], help='extra earlier dialogue JSON to check overlap')
    p.add_argument('--run-config', type=Path)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    problems, rows = [], []
    by_sha = {f['sha256']: name for name, f in m['files'].items()}
    seen_splits = set()
    for path in a.set:
        text = path.read_text()
        sha = hashlib.sha256(text.encode()).hexdigest()
        name = by_sha.get(sha)
        if not name:
            problems.append(f'set: {path} sha256 {sha[:12]} not in manifest')
            continue
        seen_splits.add(name)
        part = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(part) != m['files'][name]['rows']:
            problems.append(f'set: {path} rows {len(part)} != manifest {m["files"][name]["rows"]}')
        rows += part
    cats = m['categories']
    for r in rows:
        miss = [f for f in FIELDS if f not in r]
        if miss:
            problems.append(f'schema: {r.get("id")} missing {miss}')
            continue
        user = [t for t in r['turns'] if t != NEW]
        if len(user) < 3:
            problems.append(f'schema: {r["id"]} has {len(user)} user turns (< 3)')
        if len(r['expect']) != len(r['turns']):
            problems.append(f'schema: {r["id"]} expect/turns length mismatch')
        if r['category'] not in cats:
            problems.append(f'schema: {r["id"]} unknown category {r["category"]}')
        if not r['severe'] and not m.get('global_severe'):
            problems.append(f'schema: {r["id"]} has no severe-failure definition')
        if any(not str(e).strip() for e in r['expect']):
            problems.append(f'schema: {r["id"]} has an empty expectation')
    for name in seen_splits:
        split = 'public' if name == 'public' else 'private'
        for c, n in cats.items():
            got = sum(1 for r in rows if r['category'] == c and r['split'] == split)
            if got != n[split]:
                problems.append(f'schema: category {c} {split} {got} != manifest {n[split]}')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append('unique: duplicate ids')
    seqs = [tuple(norm(t) for t in r['turns']) for r in rows]
    if len(seqs) != len(set(seqs)):
        problems.append('unique: duplicate turn sequences')
    old = old_utterances([OLD_DEFAULT] + a.old)
    for r in rows:
        for t in r['turns']:
            if t != NEW and len(norm(t)) > COURTESY_MAX and norm(t) in old:
                problems.append(f'unique: {r["id"]} reuses an earlier development utterance: {t[:30]}')
    run = None
    if a.run_config:
        run = json.loads(a.run_config.read_text())
        miss = [f for f in RUN_FIELDS if not run.get(f)]
        if miss:
            problems.append(f'run: missing {miss}')
        if run.get('scenario_set_sha256') and run['scenario_set_sha256'] not in by_sha:
            problems.append('run: scenario_set_sha256 not in manifest')
        if run.get('mode') == 'service':
            pf = run.get('artifact_preflight')
            if not pf or not Path(pf).exists() or json.loads(Path(pf).read_text()).get('verdict') != 'pass':
                problems.append('run: service mode needs a passing eval_artifact_preflight manifest (artifact_preflight)')
    report = {'verdict': 'fail' if problems else 'pass', 'sets': sorted(seen_splits), 'scenarios': len(rows),
              'user_turns': sum(len([t for t in r['turns'] if t != NEW]) for r in rows),
              'old_utterances_checked': len(old), 'problems': problems}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(2 if problems else 0)


if __name__ == '__main__':
    main()

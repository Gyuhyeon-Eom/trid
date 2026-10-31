"""Collect AWS-EVAL-006 per-dialogue checkpoints into one blind run directory (outputs.jsonl + status.json).

Checkpoint contract (one file "<alias>-<NNN>.cms" per finished dialogue, written on the remote host and stored in S3 only
as CMS/DER ciphertext for C's certificate reports/coordination/AWS-EVAL-006/c-checkpoint-cert.pem; P and Q share one
checkpoints/ prefix per run):
  {"alias": "P"|"Q", "dialogue": <scenario id>, "rows": [{"dialogue","turn","question","output","stop_type"}...],
   "status": {"package", "shared_settings_sha256", "backend", "render_parity_token_ids_equal": true,
              "effective_settings_sha256": <sha of the server's effective generation settings>,
              "effective_settings_constant": true, "run_id": <bootstrap RUN_ID>,
              "input_sha256": <sha256 of the delivered dialogue file (id + turns only)>}}
Nothing identifying the model is read here beyond what the checkpoint carries; sealed remote files are not touched.

Only checkpoints of --alias are collected; a file of the other alias is skipped when its name and content agree.
Refuses (exit 2) on: a file name outside "<P|Q>-<NNN>.cms", a name whose alias differs from the content, a file that does
not decrypt with C's key, a run id not given with --run-id (several only for an approved resume), an input sha256 other
than the id+turns derivative of --set (so ciphertext of another set or the smoke run cannot slip in), a duplicate
dialogue, a dialogue missing any user turn or with a question / turn index differing from the set, parity or constant
effective settings not true, or any difference between checkpoints in package, shared settings, backend or effective
settings. Writes status "complete" only when every dialogue of the set is present; otherwise "incomplete" with the
missing count (never fills anything in).

  .venv/bin/python scripts/aws_eval_collect.py --checkpoints <dir of *.cms> --key <C private key> \
      --set <set.jsonl> --alias P --run-id <run> --out <run dir>
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

NEW = '<NEW_CHAT>'
NAME = re.compile(r'^([PQ])-\d{3}\.cms$')


def fail(msg):
    print(json.dumps({'verdict': 'refused', 'problem': msg}, ensure_ascii=False))
    sys.exit(2)


def decrypt(path, key):
    r = subprocess.run(['openssl', 'cms', '-decrypt', '-binary', '-inform', 'DER', '-in', str(path), '-inkey', str(key)],
                       capture_output=True)
    if r.returncode != 0:
        fail(f'{path.name}: does not decrypt with the C key')
    return json.loads(r.stdout)


def turns_only(scenarios):
    """The bytes delivered to the instance: one {"id", "turns"} line per scenario, in set order."""
    return ''.join(json.dumps({'id': s['id'], 'turns': s['turns']}, ensure_ascii=False) + '\n' for s in scenarios).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--set', type=Path, required=True)
    p.add_argument('--alias', choices=('P', 'Q'), required=True)
    p.add_argument('--run-id', action='append', required=True, help='run id of the checkpoints (repeat only for an approved resume)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        fail(f'{a.out} is not empty; refusing to overwrite')
    set_bytes = a.set.read_bytes()
    scenarios = [json.loads(l) for l in set_bytes.decode().splitlines() if l.strip()]
    by_id = {s['id']: s for s in scenarios}
    input_sha = hashlib.sha256(turns_only(scenarios)).hexdigest()
    got, identity, sources, runs = {}, None, {}, {}
    for f in sorted(a.checkpoints.glob('*.cms')):
        m = NAME.match(f.name)
        if not m:
            fail(f'{f.name}: not a checkpoint file name')
        c = decrypt(f, a.key)
        if c.get('alias') != m.group(1):
            fail(f'{f.name}: content alias {c.get("alias")!r} differs from the file name')
        if c['alias'] != a.alias:
            continue
        sid = c.get('dialogue')
        if sid not in by_id:
            fail(f'{f.name}: dialogue outside the set')
        if sid in got:
            fail(f'{f.name}: duplicate dialogue (also in {sources[sid]})')
        st = c.get('status') or {}
        if st.get('run_id') not in a.run_id:
            fail(f'{f.name}: run id {st.get("run_id")!r} not among --run-id')
        if st.get('input_sha256') != input_sha:
            fail(f'{f.name}: delivered input {st.get("input_sha256")!r} is not the id+turns derivative of --set ({input_sha})')
        if st.get('render_parity_token_ids_equal') is not True:
            fail(f'{f.name}: official-render parity not confirmed')
        if st.get('effective_settings_constant') is not True:
            fail(f'{f.name}: effective generation settings not constant')
        ident = tuple(st.get(k) for k in ('package', 'shared_settings_sha256', 'backend', 'effective_settings_sha256'))
        if None in ident:
            fail(f'{f.name}: status lacks package/shared settings/backend/effective settings')
        if identity is None:
            identity = ident
        elif ident != identity:
            fail(f'{f.name}: package, shared settings, backend or effective settings differ from other checkpoints')
        expected = [(i, q) for i, q in enumerate(by_id[sid]['turns']) if q != NEW]
        rows = c.get('rows') or []
        if [(int(r['turn']), r['question']) for r in rows] != expected or any(r.get('dialogue') != sid for r in rows):
            fail(f'{f.name}: turns or questions differ from the set')
        got[sid], sources[sid] = rows, f.name
        runs[st['run_id']] = runs.get(st['run_id'], 0) + 1
    a.out.mkdir(parents=True, exist_ok=True)
    stops, n = {}, 0
    with open(a.out / 'outputs.jsonl', 'w') as o:
        for s in scenarios:
            for r in got.get(s['id'], []):
                o.write(json.dumps({k: r[k] for k in ('dialogue', 'turn', 'question', 'output', 'stop_type')}
                                   | {'truncated_at_limit': r['stop_type'] == 'limit'}, ensure_ascii=False) + '\n')
                stops[r['stop_type']] = stops.get(r['stop_type'], 0) + 1
                n += 1
    missing = [s['id'] for s in scenarios if s['id'] not in got]
    status = {'package': a.alias, 'status': 'complete' if not missing else f'incomplete: {len(missing)} dialogues missing',
              'turns': n, 'stop_types': stops, 'dialogues': len(got), 'dialogues_sha256': hashlib.sha256(set_bytes).hexdigest(),
              'input_sha256': input_sha, 'run_ids': runs,
              'shared_settings_sha256': identity[1] if identity else None, 'backend': identity[2] if identity else None,
              'render_parity_token_ids_equal': True if got else None, 'effective_settings_constant': True if got else None,
              'outputs_sha256': hashlib.sha256((a.out / 'outputs.jsonl').read_bytes()).hexdigest(),
              'checkpoints': len(got)}
    (a.out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1))
    print(json.dumps({k: status[k] for k in ('package', 'status', 'turns', 'dialogues', 'stop_types')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

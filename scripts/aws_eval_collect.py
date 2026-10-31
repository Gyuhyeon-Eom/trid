"""Collect AWS-EVAL-006 per-dialogue checkpoints into one blind run directory (outputs.jsonl + status.json).

Checkpoint contract (one file per finished dialogue, written on the remote host and stored in S3 only as CMS/DER
ciphertext for C's certificate reports/coordination/AWS-EVAL-006/c-checkpoint-cert.pem):
  {"alias": "P"|"Q", "dialogue": <scenario id>, "rows": [{"dialogue","turn","question","output","stop_type"}...],
   "status": {"package", "shared_settings_sha256", "backend", "render_parity_token_ids_equal": true,
              "effective_settings_sha256": <sha of the server's effective generation settings>}}
Nothing identifying the model is read here beyond what the checkpoint carries; sealed remote files are not touched.

Refuses (exit 2) on: a file that does not decrypt with C's key, another alias, a duplicate dialogue, a dialogue missing
any user turn or with a question / turn index differing from the set, parity not true, or any difference between
checkpoints in package, shared settings, backend or effective settings. Writes status "complete" only when every
dialogue of the set is present; otherwise "incomplete" with the missing count (never fills anything in).

  .venv/bin/python scripts/aws_eval_collect.py --checkpoints <dir of *.cms> --key <C private key> \
      --set <set.jsonl> --alias P --out <run dir>
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

NEW = '<NEW_CHAT>'


def fail(msg):
    print(json.dumps({'verdict': 'refused', 'problem': msg}, ensure_ascii=False))
    sys.exit(2)


def decrypt(path, key):
    r = subprocess.run(['openssl', 'cms', '-decrypt', '-binary', '-inform', 'DER', '-in', str(path), '-inkey', str(key)],
                       capture_output=True)
    if r.returncode != 0:
        fail(f'{path.name}: does not decrypt with the C key')
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--set', type=Path, required=True)
    p.add_argument('--alias', choices=('P', 'Q'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        fail(f'{a.out} is not empty; refusing to overwrite')
    set_bytes = a.set.read_bytes()
    scenarios = [json.loads(l) for l in set_bytes.decode().splitlines() if l.strip()]
    by_id = {s['id']: s for s in scenarios}
    got, identity, sources = {}, None, {}
    for f in sorted(a.checkpoints.glob('*.cms')):
        c = decrypt(f, a.key)
        if c.get('alias') != a.alias:
            fail(f'{f.name}: alias {c.get("alias")!r}')
        sid = c.get('dialogue')
        if sid not in by_id:
            fail(f'{f.name}: dialogue outside the set')
        if sid in got:
            fail(f'{f.name}: duplicate dialogue (also in {sources[sid]})')
        st = c.get('status') or {}
        if st.get('render_parity_token_ids_equal') is not True:
            fail(f'{f.name}: official-render parity not confirmed')
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
              'shared_settings_sha256': identity[1] if identity else None, 'backend': identity[2] if identity else None,
              'render_parity_token_ids_equal': True if got else None, 'effective_settings_constant': True if got else None,
              'outputs_sha256': hashlib.sha256((a.out / 'outputs.jsonl').read_bytes()).hexdigest(),
              'checkpoints': len(got)}
    (a.out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1))
    print(json.dumps({k: status[k] for k in ('package', 'status', 'turns', 'dialogues', 'stop_types')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

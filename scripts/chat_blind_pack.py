"""Blind packing and unblinding for a two-run CHAT-EVAL-004 comparison (CHAT-BLIND-005).

pack     reads two raw runs of scripts/base_dialogue_runner.py over the same scenario set and writes, per scenario, the
         two model transcripts (user turns and the model's own answers, "<NEW_CHAT>" kept as a marker, stop type kept)
         in a random order labelled "1" and "2". Nothing about the model, alias, file, template or timing is written to
         the packet. The order key goes to a separate file that the judge does not open until the scores are sealed.
         Refuses unless both runs are complete, used the given set (sha256) and contain every user turn exactly once.
unblind  reads the sealed judgments ({"id", "label", "turn", "verdict", "severe", ...}, turn = 1-based user-turn
         index), checks the seal sha256, and writes one judgments file per run alias in the format of
         scripts/chat_eval_gate.py.

  .venv/bin/python scripts/chat_blind_pack.py pack --set <set.jsonl> --run P=<dir> --run Q=<dir> \
      --packet <out.jsonl> --key <key.json>
  .venv/bin/python scripts/chat_blind_pack.py unblind --key <key.json> --judgments <sealed.jsonl> \
      --sealed-sha256 <sha> --out-dir <dir>
"""
import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

NEW = '<NEW_CHAT>'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(run_dir, scenarios, set_sha):
    # scripts/chat_blind_runner.py output (outputs.jsonl + status.json; sealed/ is never read here), else the plain
    # scripts/base_dialogue_runner.py output (turns.jsonl + manifest.json)
    blind = (run_dir / 'outputs.jsonl').exists()
    status = json.loads((run_dir / ('status.json' if blind else 'manifest.json')).read_text())
    if status.get('status') != 'complete':
        sys.exit(f'{run_dir}: status {status.get("status")!r}, not complete')
    if status.get('dialogues_sha256') != set_sha:
        sys.exit(f'{run_dir}: run used set {status.get("dialogues_sha256")}, not {set_sha}')
    rows = [json.loads(l) for l in (run_dir / ('outputs.jsonl' if blind else 'turns.jsonl')).read_text().splitlines() if l.strip()]
    got = {}
    for r in rows:
        key = (r['dialogue'], int(r['turn']))
        if key in got:
            sys.exit(f'{run_dir}: duplicate turn {key}')
        got[key] = r
    out = {}
    for sid, sc in scenarios.items():
        transcript, user_index = [], 0
        for index, q in enumerate(sc['turns']):
            if q == NEW:
                transcript.append({'marker': NEW})
                continue
            r = got.pop((sid, index), None)
            if r is None or r['question'] != q:
                sys.exit(f'{run_dir}: missing or mismatched turn {sid}#{index}')
            user_index += 1
            transcript.append({'turn': user_index, 'user': q, 'answer': r['output'], 'stop_type': r['stop_type']})
        out[sid] = transcript
    if got:
        sys.exit(f'{run_dir}: {len(got)} turns outside the set')
    return out


def pack(a):
    scenarios = {r['id']: r for r in (json.loads(l) for l in a.set.read_text().splitlines() if l.strip())}
    set_sha = sha256(a.set)
    runs = dict(x.split('=', 1) for x in a.run)
    if len(runs) != 2:
        sys.exit('exactly two --run ALIAS=DIR')
    if a.key.exists() or a.packet.exists():
        sys.exit('key or packet exists; refusing to overwrite')
    transcripts = {alias: load_run(Path(d), scenarios, set_sha) for alias, d in runs.items()}
    aliases = sorted(runs)
    rng = secrets.SystemRandom()
    key, lines = {'set_sha256': set_sha, 'runs': runs, 'order': {}}, []
    for sid in rng.sample(sorted(scenarios), len(scenarios)):
        order = rng.sample(aliases, 2)
        key['order'][sid] = {'1': order[0], '2': order[1]}
        lines.append(json.dumps({'id': sid, 'category': scenarios[sid]['category'], 'turns_spec': scenarios[sid]['turns'],
                                 'expect': scenarios[sid]['expect'], 'track': scenarios[sid]['track'],
                                 'severe': scenarios[sid]['severe'], 'boundary': scenarios[sid]['boundary'],
                                 'external': scenarios[sid]['external'],
                                 'transcripts': {'1': transcripts[order[0]][sid], '2': transcripts[order[1]][sid]}},
                                ensure_ascii=False))
    a.packet.write_text('\n'.join(lines) + '\n')
    a.key.write_text(json.dumps(key, ensure_ascii=False, indent=1))
    print(json.dumps({'scenarios': len(lines), 'packet_sha256': sha256(a.packet), 'set_sha256': set_sha}))


def unblind(a):
    if sha256(a.judgments) != a.sealed_sha256:
        sys.exit(f'judgments sha256 {sha256(a.judgments)} differs from the sealed {a.sealed_sha256}')
    key = json.loads(a.key.read_text())
    per = {}
    for line in a.judgments.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        alias = key['order'][r['id']][str(r['label'])]
        row = {k: v for k, v in r.items() if k != 'label'}
        per.setdefault(alias, []).append(row)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for alias, rows in sorted(per.items()):
        (a.out_dir / f'judgments-{alias}.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in rows) + '\n')
        print(alias, len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('pack')
    k.add_argument('--set', type=Path, required=True)
    k.add_argument('--run', action='append', required=True, help='ALIAS=run directory (twice)')
    k.add_argument('--packet', type=Path, required=True)
    k.add_argument('--key', type=Path, required=True)
    u = sub.add_parser('unblind')
    u.add_argument('--key', type=Path, required=True)
    u.add_argument('--judgments', type=Path, required=True)
    u.add_argument('--sealed-sha256', required=True)
    u.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    pack(a) if a.cmd == 'pack' else unblind(a)


if __name__ == '__main__':
    main()

"""Run one blind chat package (P or Q) over a dialogue set, keeping everything that identifies the model sealed.

The package directory (runs/chat-blind-005/packages/<P|Q>/: model.gguf and tokenizer/ aliases, package.json with the
model's own sampling and expected sha256) is read by this program only; the judge does not open it. Settings shared by
both packages (system contract, output limit, context, seed, non-thinking template argument) come from the
non-identifying shared-settings file. Outputs for judging go to <out>/outputs.jsonl (dialogue, turn, question,
output, stop type); prompts, token ids, chat template, server settings and log go to <out>/sealed/, which stays
unopened until the scoring is sealed. One model at a time on an isolated port; refuses to run on a memory-pressure
warning or when the live demo is not running; never touches the live service. No retries, filters or special cases.

    .venv/bin/python scripts/chat_blind_runner.py --package P --dialogues <set.jsonl> --out <dir>
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_dialogue_runner import ROOT, live_state, load_dialogues, memory_state, run, sha256  # noqa: E402

PACKAGES = ROOT / 'runs/chat-blind-005/packages'
SHARED = ROOT / 'reports/coordination/CHAT-BLIND-005/shared-settings.json'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--package', choices=('P', 'Q'), required=True)
    p.add_argument('--dialogues', type=Path, required=True, help='CHAT-EVAL-004 style JSONL (id, turns)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f'{a.out} is not empty; refusing to overwrite')
    shared = json.loads(SHARED.read_text())
    package_dir = PACKAGES / a.package
    package = json.loads((package_dir / 'package.json').read_text())
    system = (ROOT / shared['system_prompt_file']).read_text().strip()
    if hashlib.sha256(system.encode()).hexdigest() != shared['system_prompt_sha256']:
        sys.exit('system contract file changed; refusing to run')
    dialogues = load_dialogues(a.dialogues)
    mem, live = memory_state(), live_state()
    if mem['pressure_level'] != 1 or live['status'] != 'running':
        sys.exit(f'not starting: memory {mem}, live service {live}')
    if sha256(package_dir / 'model.gguf') != package['expected_sha256']:
        sys.exit(f'package {a.package}: model file does not match its sealed sha256; refusing to run')

    sealed = a.out / 'sealed'
    sealed.mkdir(parents=True, exist_ok=True)
    manifest = {'package': a.package, 'model_sha256': package['expected_sha256'], 'sampling_request': package['sampling'],
                'template_kwargs': shared['template_kwargs'], 'system_prompt': system, 'max_tokens': shared['max_tokens'],
                'context': shared['context'], 'seed': shared['seed'], 'dialogues_file': str(a.dialogues),
                'dialogues_sha256': sha256(a.dialogues), 'started': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'memory_before': mem, 'live_before': live, 'guard': [], 'status': 'running'}
    stops = {}
    with open(a.out / 'outputs.jsonl', 'w') as outputs, open(sealed / 'turns.jsonl', 'w') as full:
        def write_turn(record):
            stops[record['stop_type']] = stops.get(record['stop_type'], 0) + 1
            outputs.write(json.dumps({k: record[k] for k in ('dialogue', 'turn', 'question', 'output', 'stop_type')}
                                     | {'truncated_at_limit': record['stop_type'] == 'limit'}, ensure_ascii=False) + '\n')
            full.write(json.dumps(record, ensure_ascii=False) + '\n')
            outputs.flush()
            full.flush()
        try:
            run(package_dir / 'model.gguf', package['sampling'], system, shared['template_kwargs'], package_dir / 'tokenizer',
                dialogues, shared['port'], shared['context'], shared['max_tokens'], shared['seed'],
                sealed / 'llama-server.log', write_turn, manifest)
        finally:
            (sealed / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            parity = manifest.get('render_parity_with_official') or {}
            status = {'package': a.package, 'status': manifest['status'], 'turns': sum(stops.values()), 'stop_types': stops,
                      'dialogues_sha256': manifest['dialogues_sha256'], 'shared_settings_sha256': sha256(SHARED),
                      'render_parity_token_ids_equal': parity.get('token_ids_equal'),
                      'effective_settings_constant': manifest.get('effective_settings_constant'),
                      'started': manifest['started'], 'finished': manifest.get('finished'),
                      'memory_after': manifest.get('memory_after'), 'live_after': manifest.get('live_after'),
                      'outputs_sha256': sha256(a.out / 'outputs.jsonl'),
                      'sealed_sha256': {f.name: sha256(f) for f in sorted(sealed.iterdir())}}
            (a.out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=1))
    print(json.dumps({k: status[k] for k in ('package', 'status', 'turns', 'stop_types', 'render_parity_token_ids_equal',
                                             'effective_settings_constant', 'memory_after', 'live_after')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

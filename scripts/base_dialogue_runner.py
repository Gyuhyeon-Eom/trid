"""Run fixed development dialogues against one local GGUF model with a full provenance manifest.

One model at a time on an isolated llama-server port (the live demo is never touched): each turn records the exact
rendered prompt string, its token ids, the request's sampling, the server's effective generation settings, the output
text and token ids, the stop type and timings; the manifest records the model file sha256, its declared source and
revision, the embedded chat template (sha and text), optional parity of that template's rendering with the official
Hugging Face tokenizer, the history policy and the machine's memory state. Each dialogue keeps its own history (the
model's own answers); "__CLEAR__" starts a new conversation. A memory-pressure warning or a stopped live service ends
the run (the partial output stays). No retries, no output filtering, no per-question handling.

    .venv/bin/python scripts/base_dialogue_runner.py --model-file runs/x.gguf --model-id x --source URL@REV \
        --sampling '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0}' --out reports/coordination/RUN-ID
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / 'runs/tools/llama.cpp-8ca4060-metal/llama-server'
DIALOGUES = ROOT / 'reports/coordination/BASE-RESUME-004/dev-dialogues-13.json'
SYSTEM_DEFAULT = ('당신은 TRID입니다. 공공기관 직원과 이야기하는 한국어 AI 비서입니다. 사용자의 말에 공감하고, 앞선 대화를 이어서 '
                  '자연스럽고 구체적으로 답하세요. 대화 도중에 인사나 자기소개를 반복하지 마세요. 한국어로 답하세요.')


def request(base, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def memory_state():
    level = int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True).stdout)
    out = subprocess.run(['memory_pressure'], capture_output=True, text=True).stdout
    free = next((int(l.rsplit(' ', 1)[1].rstrip('%')) for l in out.splitlines() if 'free percentage' in l), None)
    return {'pressure_level': level, 'free_pct': free}


def live_state():
    path = ROOT / '.shared-demo/state.json'
    if not path.exists():
        return {'status': None}
    s = json.loads(path.read_text())
    return {k: s.get(k) for k in ('status', 'backend_pgid', 'backend_started', 'connection_status')}


def start(model_file, port, context, log_path):
    log = open(log_path, 'wb')
    args = [str(SERVER), '-m', str(model_file), '--host', '127.0.0.1', '--port', str(port), '-c', str(context),
            '-np', '1', '-t', '4', '-tb', '4', '--jinja']
    proc = subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True)
    base = f'http://127.0.0.1:{port}'
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'llama-server exited; see {log_path}')
        try:
            props = request(base, '/props')
            if Path(props['model_path']).resolve() == Path(model_file).resolve():
                return proc, base, props, args
        except OSError:
            pass
        time.sleep(.5)
    raise RuntimeError('llama-server readiness timeout')


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(5)
    except ProcessLookupError:
        pass


def official_render(hf_dir, messages, template_kwargs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-file', type=Path, required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--source', required=True, help='official source and pinned revision, recorded as given')
    p.add_argument('--expected-sha256', help='abort unless the model file has this sha256')
    p.add_argument('--sampling', required=True, help='JSON of the official recommended sampling for this model')
    p.add_argument('--sampling-source', default='', help='where the sampling values come from')
    p.add_argument('--system', default=SYSTEM_DEFAULT, help="system prompt; '' for none")
    p.add_argument('--template-kwargs', default='{}', help='JSON chat_template_kwargs (e.g. {"enable_thinking": false})')
    p.add_argument('--hf-dir', help='local official tokenizer dir for a rendering parity check (no download)')
    p.add_argument('--dialogues', type=Path, default=DIALOGUES)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=51990)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--max-tokens', type=int, default=384)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()

    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f'{a.out} is not empty; refusing to overwrite')
    a.out.mkdir(parents=True, exist_ok=True)
    sampling = json.loads(a.sampling)
    template_kwargs = json.loads(a.template_kwargs)
    dialogues = json.loads(a.dialogues.read_text())['dialogues']
    mem, live = memory_state(), live_state()
    if mem['pressure_level'] != 1 or live['status'] != 'running':
        sys.exit(f'not starting: memory {mem}, live service {live}')
    file_sha = sha256(a.model_file)
    if a.expected_sha256 and file_sha != a.expected_sha256:
        sys.exit(f'sha256 mismatch: {file_sha} != {a.expected_sha256}')
    manifest = {
        'model_id': a.model_id, 'model_file': str(a.model_file), 'model_sha256': file_sha, 'source': a.source,
        'sampling_request': sampling, 'sampling_source': a.sampling_source, 'max_tokens': a.max_tokens, 'seed': a.seed,
        'system_prompt': a.system or None, 'template_kwargs': template_kwargs,
        'history_policy': 'each dialogue keeps its own history of the model\'s own answers; "__CLEAR__" starts a new one',
        'dialogues_file': str(a.dialogues), 'dialogues_sha256': sha256(a.dialogues),
        'started': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'memory_before': mem, 'live_before': live,
        'guard': [], 'status': 'running'}
    proc, base, props, args = start(a.model_file, a.port, a.context, a.out / 'llama-server.log')
    turns_file = open(a.out / 'turns.jsonl', 'w')
    try:
        template = props.get('chat_template') or ''
        manifest.update(server_args=args, build=props.get('build_info'), chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                        chat_template=template, server_default_params=props.get('default_generation_settings', {}).get('params'),
                        eos_token=props.get('eos_token'), bos_token=props.get('bos_token'))
        sample = ([{'role': 'system', 'content': a.system}] if a.system else []) + [
            {'role': 'user', 'content': '안녕하세요'}, {'role': 'assistant', 'content': '안녕하세요.'}, {'role': 'user', 'content': '고마워요'}]
        served = request(base, '/apply-template', {'messages': sample, **({'chat_template_kwargs': template_kwargs} if template_kwargs else {})})['prompt']
        manifest['render_sample'] = served
        if a.hf_dir:
            official = official_render(a.hf_dir, sample, template_kwargs)
            manifest['render_parity_with_official'] = {'hf_dir': a.hf_dir, 'equal': served == official, 'official': official}
        else:
            # no local official tokenizer: parity is unconfirmed, never assumed
            manifest['render_parity_with_official'] = {'hf_dir': None, 'equal': None, 'note': 'unconfirmed: no local official tokenizer'}
        effective, drift = None, []
        for name, turns in dialogues.items():
            mem, live = memory_state(), live_state()
            manifest['guard'].append({'t': time.strftime('%H:%M:%S'), 'dialogue': name, **mem, 'live': live.get('status')})
            if mem['pressure_level'] != 1 or live.get('status') != 'running':
                manifest['status'] = f'stopped before {name}: memory {mem}, live {live}'
                break
            history = []
            for index, question in enumerate(turns):
                if question == '__CLEAR__':
                    history = []
                    continue
                messages = ([{'role': 'system', 'content': a.system}] if a.system else []) + history + [{'role': 'user', 'content': question}]
                prompt = request(base, '/apply-template', {'messages': messages, **({'chat_template_kwargs': template_kwargs} if template_kwargs else {})})['prompt']
                tokens = request(base, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
                started = time.monotonic()
                c = request(base, '/completion', dict(prompt=tokens, n_predict=a.max_tokens, seed=a.seed, cache_prompt=False,
                                                      return_tokens=True, **sampling))
                wall = time.monotonic() - started
                settings = {k: v for k, v in (c.get('generation_settings') or {}).items() if k not in ('n_predict', 'seed')}
                if effective is None:
                    effective = settings
                elif settings != effective:
                    drift.append({'dialogue': name, 'turn': index,
                                  'changed': {k: [effective.get(k), settings.get(k)] for k in set(effective) | set(settings)
                                              if effective.get(k) != settings.get(k)}})
                text = c['content'].strip()
                turns_file.write(json.dumps({'dialogue': name, 'turn': index, 'question': question, 'prompt': prompt,
                                             'prompt_token_ids': tokens, 'output': text, 'output_token_ids': c.get('tokens'),
                                             'stop_type': c.get('stop_type'), 'stopping_word': c.get('stopping_word'),
                                             'timings': c.get('timings'), 'wall_seconds': round(wall, 3)}, ensure_ascii=False) + '\n')
                turns_file.flush()
                history += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': text}]
        else:
            manifest['status'] = 'complete'
        manifest['effective_settings_constant'] = not drift
        manifest['effective_settings_drift'] = drift
        manifest['effective_generation_settings'] = effective
    finally:
        turns_file.close()
        stop(proc)
        manifest.update(finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'), memory_after=memory_state(), live_after=live_state())
        (a.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('model_id', 'model_sha256', 'status', 'memory_after', 'live_after')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

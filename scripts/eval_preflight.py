"""Pre-run length check for a prompt set against each model's real tokenizer, chat template and context.

Counts every prompt; nothing is dropped. Per model and prompt: rendered input tokens, context, planned output
cap, room left (context - input), and status:
  fits              input + planned cap <= context
  output_limited    input fits but room < planned cap (the model can still answer, with less room)
  input_exceeds     input alone does not fit the context (the item cannot be run unchanged)
When a reference answer is present, `gold_fits` says whether room >= reference tokens + 1 (EOS) on that tokenizer.

Token counts come from the model itself: trid's tokenizer and prompt framing for trid, and for llama.cpp models
the server's own /apply-template (thinking disabled) and /tokenize. Different tokenizers mean equal token counts
are not equal text lengths.

  python scripts/eval_preflight.py --prompts <file> [--prompts ...] --trid-run runs/<run> \
      --server name=http://127.0.0.1:PORT --output reports/<name>.json
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def status(input_tokens, context, cap):
    if input_tokens >= context:
        return 'input_exceeds'
    return 'fits' if input_tokens + cap <= context else 'output_limited'


def classify(items, count_input, count_answer, context, cap):
    rows = []
    for item in items:
        n = count_input(item['prompt'])
        row = dict(id=item['id'], set=item['set'], input_tokens=n, context=context, planned_cap=cap,
                   room=context - n, status=status(n, context, cap))
        if item.get('reference'):
            need = count_answer(item['reference']) + 1
            row.update(reference_tokens_with_eos=need, gold_fits=context - n >= need)
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for name in sorted({r['set'] for r in rows}) + ['all']:
        part = [r for r in rows if name in ('all', r['set'])]
        inputs = sorted(r['input_tokens'] for r in part)
        gold = [r for r in part if 'gold_fits' in r]
        out[name] = dict(total=len(part), **Counter(r['status'] for r in part),
                         input_tokens_p50=inputs[len(inputs) // 2], input_tokens_max=inputs[-1],
                         min_room=min(r['room'] for r in part),
                         with_reference=len(gold), gold_does_not_fit=sum(not r['gold_fits'] for r in gold))
    return out


def post(url, path, payload):
    req = urllib.request.Request(url + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=60) as r:
        return json.load(r)


def server_counter(url):
    def count_input(prompt):
        rendered = post(url, '/apply-template', {'messages': [{'role': 'user', 'content': prompt}],
                                                 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
        return len(post(url, '/tokenize', {'content': rendered, 'add_special': True, 'parse_special': True})['tokens'])

    def count_answer(text):
        return len(post(url, '/tokenize', {'content': text, 'add_special': False})['tokens'])
    return count_input, count_answer


def load_prompts(paths):
    items = []
    for path in paths:
        for row in json.loads(Path(path).read_text())['results']:
            items.append(dict(row, set=Path(path).stem))
    ids = [i['id'] for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError('Prompt ids must be unique across sets')
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prompts', type=Path, action='append', required=True)
    p.add_argument('--trid-run', type=Path)
    p.add_argument('--trid-cap', type=int, default=1024)
    p.add_argument('--server', action='append', default=[], help='name=http://127.0.0.1:PORT (llama.cpp)')
    p.add_argument('--server-cap', type=int, default=1024)
    p.add_argument('--server-context', type=int, default=8192, help='planned serving context for llama.cpp models')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    items = load_prompts(a.prompts)
    report = dict(protocol=__doc__, prompts=[dict(path=str(x), sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in a.prompts],
                  models={})
    if a.trid_run:
        from tokenizers import Tokenizer
        from trid.text import prompt_ids
        tok = Tokenizer.from_file(str(a.trid_run / 'tokenizer.json'))
        context = json.loads((a.trid_run / 'config.json').read_text())['context_length']
        rows = classify(items, lambda t: len(prompt_ids(tok, t)), lambda t: len(tok.encode(t).ids), context, a.trid_cap)
        report['models']['trid'] = dict(run=str(a.trid_run), context=context, context_source='config.json (trained context)',
                                        planned_cap=a.trid_cap, summary=summarize(rows), rows=rows)
    for spec in a.server:
        name, url = spec.split('=', 1)
        meta = get(url, '/v1/models')['data'][0]['meta']
        trained = meta['n_ctx_train']
        context = min(a.server_context, trained)
        count_input, count_answer = server_counter(url)
        rows = classify(items, count_input, count_answer, context, a.server_cap)
        props = get(url, '/props')
        report['models'][name] = dict(model_path=props['model_path'], n_ctx_train=trained, context=context,
                                      context_source=f'planned serving context min({a.server_context}, n_ctx_train)',
                                      planned_cap=a.server_cap, summary=summarize(rows), rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({m: v['summary']['all'] for m, v in report['models'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

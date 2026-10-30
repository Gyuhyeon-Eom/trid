"""Diagnostic generation on the C pilot_dev split (never trained on) for a llama.cpp chat model.

Same rendering as training (trid.prompts.rag_prompt over the case context as D1..Dn, CASE_SYSTEM from
scripts/train_open_weight_lora.py, native template, Qwen3 no-thinking), greedy, 512 output tokens. Stored raw answers
are the primary artifact for review. Two deterministic number checks are reported as diagnostics only, not a quality
score: `target_numbers_covered` (numeric tokens of the target that appear in the answer) and `unsupported_numbers`
(numeric tokens of the answer that appear neither in the context nor in the question).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external_chat import complete, digest, request  # noqa: E402
from trid.case_prompts import CASE_SYSTEM  # noqa: E402
from trid.prompts import rag_prompt  # noqa: E402

NUMBER = re.compile(r'\d[\d,.]*')
MAX_TOKENS = 512


def numbers(text):
    return {n.rstrip('.,').replace(',', '') for n in NUMBER.findall(text)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', required=True)
    p.add_argument('--model-file', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--pilot', type=Path, required=True, help='C pilot.jsonl; only split=pilot_dev rows are used')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output file')
    base = a.url.rstrip('/')
    props = request(base, '/props')
    if Path(props['model_path']).resolve() != a.model_file.resolve():
        raise ValueError('Server model does not match --model-file')
    context = props['default_generation_settings']['n_ctx']
    rows = [json.loads(line) for line in a.pilot.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get('split') == 'pilot_dev']
    results = []
    for row in rows:
        prompt = rag_prompt([(f'D{i}', t) for i, t in enumerate(row['context'], 1)], row['question'])
        messages = [{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': prompt}]
        started = time.perf_counter()
        r = complete(base, messages, 'qwen3', MAX_TOKENS, context)
        answer = r.get('generated') or ''
        target_numbers, answer_numbers = numbers(row['target']), numbers(answer)
        support = numbers('\n'.join(row['context']) + '\n' + row['question'])
        results.append(dict(id=row['id'], task_type=row['task_type'], behavior=row['behavior'], target=row['target'],
                            generated=r.get('generated'), skipped_context=r.get('skipped_context', False),
                            prompt_tokens=r.get('prompt_tokens'), output_tokens=r.get('output_tokens'),
                            truncated_output=r.get('truncated_output'), seconds=round(time.perf_counter() - started, 2),
                            target_numbers_covered=f'{len(target_numbers & answer_numbers)}/{len(target_numbers)}',
                            unsupported_numbers=sorted(answer_numbers - support)))
        print(row['id'], results[-1]['target_numbers_covered'], results[-1]['unsupported_numbers'][:5], flush=True)
    report = dict(protocol=__doc__, label=a.label, model_sha256=digest(a.model_file), pilot=str(a.pilot),
                  pilot_sha256=hashlib.sha256(a.pilot.read_bytes()).hexdigest(), case_system_prompt=CASE_SYSTEM,
                  context=context, max_output_tokens=MAX_TOKENS, results=results)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()

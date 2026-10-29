"""Compare a local llama.cpp chat model on the unchanged RAG310 questions.

Comparison-only: no model enters the trid product. Uses the native chat template,
greedy decoding, 64 output tokens and disabled thinking. Two preregistered prompt
conditions are reported separately; neither is selected using evaluation scores.
Legacy scoring is preserved; strict_product additionally requires full output grammar.
This measures supplied-evidence QA, not retrieval, security or general Qwen parity.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rag_eval import score

SYSTEM = ('주어진 근거만 사용하세요. 답은 값 [Dn] 형식으로 쓰고, 여러 값은 쉼표로 구분하세요. '
          '근거가 없으면 확인 불가, 근거가 충돌하면 근거 충돌만 출력하세요. 설명은 덧붙이지 마세요.')
GRAMMAR = re.compile(r'([^\[\]]+?)\s*\[(D[1-9])\](?:\s*,\s*([^\[\]]+?)\s*\[(D[1-9])\])?')
MAX_TOKENS = 64


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def format_ok(text):
    text = text.strip()
    if text in ('확인 불가', '근거 충돌'):
        return True
    m = GRAMMAR.fullmatch(text)
    return bool(m and m[1].strip() and (m[3] is None or m[3].strip()))


def product_score(row, generated):
    result = score(row, generated)
    result.update(format_ok=format_ok(generated))
    result['strict_product'] = result['strict'] and result['format_ok']
    return result


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-radius, center+radius]


def summarize(rows, results):
    counts = defaultdict(Counter)
    known = defaultdict(Counter)
    for row, result in zip(rows, results, strict=True):
        for key in ('all', row['origin'], row['task'], 'answerable' if row['gold'] else 'unanswerable'):
            c = counts[key]
            c['total'] += 1
            for field in ('strict', 'strict_product', 'value', 'lenient_value', 'source', 'abstained',
                          'skipped_context', 'truncated_output', 'context_truncated'):
                c[field] += bool(result.get(field, False))
                known[key][field] += field in result
            c['errors'] += result.get('exception') is not None
            known[key]['errors'] += 'exception' in result
            c['format_error'] += result.get('generated') is not None and not result['format_ok']
    summary = {}
    for key, value in sorted(counts.items()):
        entry = dict(value)
        for field in ('skipped_context', 'truncated_output', 'context_truncated', 'errors'):
            if known[key][field] != value['total']:
                entry[field] = None
        entry['strict_product_wilson95'] = wilson(value['strict_product'], value['total'])
        summary[key] = entry
    return summary


def request(base, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(base+endpoint, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18473')
    parser.add_argument('--model-file', required=True, type=Path)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data', type=Path, default=Path('data/korean-rag-eval-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1') or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        parser.error('Use a literal loopback HTTP llama.cpp server; remote document upload is not supported')
    if args.output.exists():
        parser.error('Use a new output file')
    base = args.url.rstrip('/')
    test = args.data/'test.jsonl'
    test_hash = digest(test)
    provenance = json.loads((args.data/'provenance.json').read_text())
    if test_hash != provenance['hashes']['test']:
        raise ValueError('Evaluation data changed')
    rows = [json.loads(line) for line in test.read_text().splitlines()]
    props = request(base, '/props')
    if Path(props['model_path']).resolve() != args.model_file.resolve():
        raise ValueError('Server model does not match --model-file')
    model_hash = digest(args.model_file)
    context = props['default_generation_settings']['n_ctx']
    report = dict(protocol=__doc__, model_id=args.model_id, revision=args.revision,
                  model_sha256=model_hash, model_bytes=args.model_file.stat().st_size,
                  test_sha256=test_hash, runtime=props, max_output_tokens=MAX_TOKENS,
                  temperature=0, thinking=False, evaluated_at_unix=time.time(), conditions={}, comparisons=[])
    # Both conditions are fixed before any evaluation item is generated.
    for condition, system in (('prompt_only', None), ('grammar_instructed', SYSTEM)):
        results = []
        for i, row in enumerate(rows):
            result = dict(id=row['id'], task=row['task'], origin=row['origin'], answer=row['answer'],
                          generated=None, prompt_tokens=None, output_tokens=None, finish_reason=None,
                          exception=None, skipped_context=False, context_truncated=False,
                          truncated_output=False, format_ok=False)
            started = time.perf_counter()
            try:
                messages = ([] if system is None else [{'role': 'system', 'content': system}])
                messages.append({'role': 'user', 'content': row['prompt']})
                prompt = request(base, '/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
                result['rendered_prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
                # Qwen3's native no-thinking generation prefix must actually be rendered.
                if not prompt.endswith('<think>\n\n</think>\n\n'):
                    raise ValueError('Native Qwen3 no-thinking prefix was not rendered')
                tokens = request(base, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
                result['prompt_tokens'] = len(tokens)
                if len(tokens) + MAX_TOKENS > context:
                    result['skipped_context'] = True
                else:
                    completion = request(base, '/completion', dict(prompt=tokens, n_predict=MAX_TOKENS,
                        temperature=0, seed=42, samplers=['temperature'], repeat_penalty=1,
                        cache_prompt=False, return_tokens=True))
                    result.update(generated=completion['content'].strip(),
                        output_tokens=len(completion['tokens']), finish_reason=completion['stop_type'],
                        truncated_output=completion['stop_type']=='limit',
                        context_truncated=completion['truncated'], timings=completion['timings'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result['exception'] = f'{type(exc).__name__}: {exc}'
            if i == 0 and result['exception']:
                raise RuntimeError(f'{condition}: first inference failed: {result["exception"]}')
            result['seconds'] = time.perf_counter()-started
            result.update(product_score(row, result['generated'] or ''))
            if result['exception'] or result['skipped_context'] or result['context_truncated']:
                for metric in ('strict', 'strict_product', 'value', 'lenient_value', 'source', 'abstained'):
                    result[metric] = False
            results.append(result)
            if (i+1) % 25 == 0:
                print(f'{condition}: {i+1}/{len(rows)}', flush=True)
        report['conditions'][condition] = dict(system_prompt=system,
            system_prompt_sha256=hashlib.sha256((system or '').encode()).hexdigest(),
            summary=summarize(rows, results), results=results)
    if digest(args.model_file) != model_hash or digest(test) != test_hash:
        raise ValueError('Model or evaluation data changed during evaluation')
    # Rescore the exact stored raw trid answers, with no model-specific text cleanup.
    for path in args.compare_report:
        previous = json.loads(path.read_text())
        if previous['test_sha256'] != test_hash or [r['id'] for r in previous['results']] != [r['id'] for r in rows]:
            raise ValueError(f'Comparison must contain the same complete ordered dataset: {path}')
        results = []
        for row, old in zip(rows, previous['results'], strict=True):
            result = {**old, **product_score(row, old['generated'] or '')}
            if old.get('skipped_context'):
                for metric in ('strict', 'strict_product', 'value', 'lenient_value', 'source', 'abstained'):
                    result[metric] = False
            results.append(result)
        report['comparisons'].append(dict(path=str(path), report_sha256=digest(path),
            run=previous['run'], model_sha256=previous['model_sha256'], decoding=previous['decoding'],
            accounting_limit='Original trid report lacks finish-reason, output-token and latency fields; not assumed zero.',
            summary=summarize(rows, results)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({name: item['summary']['all'] for name, item in report['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()

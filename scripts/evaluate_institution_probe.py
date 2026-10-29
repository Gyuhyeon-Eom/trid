"""Evaluate an extractive reader on the frozen institution probe (data/institution-qa-frozen-01).

Primary metric per row is `exact`: the raw generated answer is a member of the row's declared
`accepted_answers` (full strings such as '225대 [D1]' or '확인 불가'); no normalization. The legacy
scripts/evaluate_rag_eval.score strict value (NFKC, whitespace-stripped) is reported as `legacy_strict`
and is secondary only. Forty cases: a probe, not a certification.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_extractive_dev import categorize  # noqa: E402
from evaluate_rag_eval import score  # noqa: E402
from trid.extractive import load_reader, encode, decode, batch, WEIGHTS, parse_answer  # noqa: E402

GROUPS = ('all', 'institution_value', 'institution_missing', 'single_evidence', 'multi_evidence')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def groups_of(row):
    return ('all', row['task'], 'multi_evidence' if len(row['evidence']) > 1 else 'single_evidence')


def evaluate_row(model, tok, device, pad, row):
    """One row -> result dict; exact / legacy are False on an exception, never None."""
    result = dict(id=row['id'], task=row['task'], question=row['question'], evidence_count=len(row['evidence']),
                  generated=None, accepted_answers=row['accepted_answers'], exact=False, legacy=False,
                  presence=None, presence_probs=None, reason=None, exception=None)
    started = time.perf_counter()
    try:
        encoded = encode(tok, row['prompt'], context=model.backbone.cfg.context_length)
        kind, scores = model(*batch([encoded], pad, device))
        output = decode((kind[0], scores[0]), encoded)
        generated = output['answer']
        result.update(generated=generated, exact=generated in row['accepted_answers'],
                      legacy=bool(score(row, generated)['strict']), presence=output['presence'],
                      presence_probs=output['presence_probs'], reason=output['reason'],
                      input_tokens=len(encoded['ids']))
    except ValueError as exc:
        result['exception'] = str(exc)
    result['seconds'] = time.perf_counter() - started
    # analyze_extractive_dev.categorize keys on `strict` and a parsed canonical gold; exact is our strict.
    category, detail = categorize(dict(result, strict=result['exact']),
                                  dict(task=row['task'], parsed=parse_answer(row['answer'])))
    result.update(category=category, detail=detail)
    return result


def summarize(results):
    summary = {name: Counter(total=0, exact=0, legacy_strict=0, false_presence=0, wrong_abstention=0, exceptions=0)
               for name in GROUPS}
    categories = defaultdict(Counter)
    for result in results:
        for name in result['groups']:
            c = summary[name]
            c['total'] += 1
            c['exact'] += result['exact']
            c['legacy_strict'] += result['legacy']
            c['false_presence'] += result['category'] == 'false_presence'
            c['wrong_abstention'] += result['category'] == 'wrong_abstention'
            c['exceptions'] += result['exception'] is not None
            categories[name][result['category']] += 1
    out = {}
    for name, c in summary.items():
        out[name] = dict(c, exact_rate=c['exact'] / c['total'] if c['total'] else None,
                         categories=dict(sorted(categories[name].items())))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/institution-qa-frozen-01'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.output.exists() or args.threads < 1:
        p.error('Use a new output path and positive thread count')
    evaluated_at = time.time()
    meta = json.loads((args.run / 'training.json').read_text())
    if meta.get('status') != 'complete':
        raise ValueError('Use a completed reader, not an in-progress checkpoint')
    data_path = args.data / 'test.jsonl'
    data_hash = digest(data_path)
    provenance = json.loads((args.data / 'provenance.json').read_text())
    if data_hash != provenance['hashes']['test']:
        raise ValueError('Probe data changed')
    source_docs_hash = digest(args.data / 'source-docs.jsonl')
    if source_docs_hash != provenance['hashes']['source_docs']:
        raise ValueError('Probe source documents changed')
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    weights_hash = digest(args.run / WEIGHTS)
    torch.set_num_threads(args.threads)
    model, tok, device = load_reader(args.run, args.device)
    pad = tok.token_to_id('<|pad|>')
    print(f'Loaded completed reader ({model.attention}); evaluating {len(rows)} probe rows on {device}', flush=True)
    started_all = time.perf_counter()
    results = []
    with torch.inference_mode():
        for row in rows:
            result = evaluate_row(model, tok, device, pad, row)
            result['groups'] = groups_of(row)
            results.append(result)
    if digest(args.run / WEIGHTS) != weights_hash or digest(data_path) != data_hash:
        raise ValueError('Model or probe data changed during inference')
    report = dict(protocol=__doc__, run=str(args.run.resolve()), model_sha256=weights_hash,
                  reader_attention=model.attention, data=str(args.data), data_sha256=data_hash,
                  source_docs_sha256=source_docs_hash, device=str(device), threads=args.threads,
                  evaluated_at_unix=evaluated_at, total_seconds=time.perf_counter() - started_all,
                  exact_definition='raw generated string is a member of accepted_answers; no normalization',
                  legacy_definition='scripts/evaluate_rag_eval.score strict (NFKC, whitespace-insensitive); secondary',
                  summary=summarize(results), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

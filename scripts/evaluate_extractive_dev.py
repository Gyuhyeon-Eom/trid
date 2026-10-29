"""Development-only extractive evaluation on curriculum valid.jsonl, never locked RAG310.

Keep every RAG validation row, including multi-value/unreachable labels, in the
strict denominator. This is development evidence, not an unseen test or calibrated
confidence guarantee. Confidence curves accept only argmax-present predictions
whose raw P(present) meets the threshold, and use all evaluated rows as coverage
denominator, including failures and unsupported multi-value answers.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.extractive import load_reader, encode, decode, batch, WEIGHTS, parse_answer
from trid.span_boundaries import BOUNDARY_POLICY


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def risk_coverage(results):
    """Accept tied confidence groups together after the present decision; zero-answer risk is undefined."""
    groups = defaultdict(list)
    for result in results:
        if result.get('presence') == 0 and result.get('exception') is None:
            groups[result['presence_probs'][0]].append(result['strict'])
    points = [dict(threshold=None, answered=0, correct=0, coverage=0., risk=None)]
    answered = correct = 0
    for probability in sorted(groups, reverse=True):
        group = groups[probability]
        answered += len(group)
        correct += sum(group)
        points.append(dict(threshold=probability, answered=answered, correct=correct,
                           coverage=answered/len(results), risk=1-correct/answered))
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/korean-rag-curriculum-v2'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.output.exists() or args.threads < 1:
        p.error('Use a new output path and positive thread count')
    evaluated_at = time.time()
    started_all = time.perf_counter()
    meta = json.loads((args.run/'training.json').read_text())
    if meta.get('status') != 'complete':
        raise ValueError('Use a completed reader, not an in-progress checkpoint')
    data_path = args.data/'valid.jsonl'
    data_hash = digest(data_path)
    provenance = json.loads((args.data/'provenance.json').read_text())
    if data_hash != provenance['hashes']['valid']:
        raise ValueError('Development data changed')
    raw = [json.loads(line) for line in data_path.read_text().splitlines()]
    rows = [r for r in raw if r.get('group') in ('rag', 'rag_real') and not r.get('history')]
    weights_hash = digest(args.run/WEIGHTS)
    project = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve()] + [
        Path(module.__file__).resolve() for name, module in list(sys.modules.items())
        if (name == 'trid' or name.startswith('trid.')) and getattr(module, '__file__', None)]
    source_hashes = {str(path.relative_to(project)): digest(path) for path in source_paths}
    torch.set_num_threads(args.threads)
    model, tok, device = load_reader(args.run, 'cpu')
    pad = tok.token_to_id('<|pad|>')
    results = []
    print(f'Loaded completed reader; evaluating all {len(rows)} RAG development rows on CPU', flush=True)
    with torch.inference_mode():
        for row in rows:
            parsed = parse_answer(row['answer'])
            truth_kind = parsed[0] if parsed is not None else (0 if row['task'] == 'rag_combine' else None)
            result = dict(id=row['id'], task=row['task'], answer=row['answer'], expected_presence=truth_kind,
                          generated=None, strict=False, exception=None, canonical_label_issue=None)
            started = time.perf_counter()
            try:
                label = encode(tok, row['prompt'], row['answer'], context=model.backbone.cfg.context_length)
                encoded = label if label is not None else encode(tok, row['prompt'], context=model.backbone.cfg.context_length)
                result['canonical_label_issue'] = ('unmapped_or_multi_value' if label is None else
                    label.get('label_reason') if label.get('kind') == 0 else None)
                kind, scores = model(*batch([encoded], pad, device))
                output = decode((kind[0], scores[0]), encoded)
                result.update(generated=output['answer'], presence=output['presence'],
                              presence_probs=output['presence_probs'], reason=output['reason'],
                              input_tokens=len(encoded['ids']), strict=output['answer'] == row['answer'].strip())
            except ValueError as exc:
                result['exception'] = str(exc)
            result['seconds'] = time.perf_counter()-started
            results.append(result)
    if digest(args.run/WEIGHTS) != weights_hash or digest(data_path) != data_hash:
        raise ValueError('Model or development data changed during inference')
    summary = defaultdict(Counter)
    for result in results:
        kind_name = 'unknown_label' if result['expected_presence'] is None else ('present', 'missing', 'conflict')[result['expected_presence']]
        for group in ('all', result['task'], kind_name):
            c = summary[group]
            c['total'] += 1
            c['strict'] += result['strict']
            c['exceptions'] += result['exception'] is not None
            c['canonical_label_unreachable'] += result['canonical_label_issue'] is not None
            c['predicted_present'] += result.get('presence') == 0
            c['wrong_abstention'] += result['expected_presence'] == 0 and result.get('presence') in (1, 2)
            c['false_presence'] += result['expected_presence'] in (1, 2) and result.get('presence') == 0
            c['presence_correct'] += result['expected_presence'] is not None and result.get('presence') == result['expected_presence']
            c['abstained_no_valid_span'] += result.get('reason') == 'no_valid_span'
            c['abstained_by_head'] += result.get('presence') in (1, 2) and result.get('reason') != 'no_valid_span'
    report = dict(protocol=__doc__, run=str(args.run.resolve()), model_sha256=weights_hash,
                  evaluated_at_unix=evaluated_at, wall_seconds_including_load=time.perf_counter()-started_all,
                  python_version=sys.version, torch_version=torch.__version__,
                  config_sha256=digest(args.run/'config.json'), training_json_sha256=digest(args.run/'training.json'),
                  tokenizer_sha256=digest(args.run/'tokenizer.json'), loaded_source_sha256=source_hashes,
                  inference_boundary_policy=BOUNDARY_POLICY, trained_boundary_policy=meta.get('boundary_policy'),
                  data=str(data_path.resolve()), data_sha256=data_hash, raw_rows=len(raw), evaluated_rows=len(rows),
                  excluded_non_rag_or_history=len(raw)-len(rows), threads=args.threads,
                  development_only=True, historical_training={key: meta.get(key) for key in
                      ('stage', 'status', 'completed_steps', 'best_step', 'best_validation_loss',
                       'checkpoint_selection', 'counts', 'train_limit', 'validation_limit')},
                  summary={key: dict(value) for key, value in sorted(summary.items())},
                  risk_coverage=risk_coverage(results), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

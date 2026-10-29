"""Train the extractive head (trid.extractive) on the RAG rows of a curriculum, from a base run.

Uses only rows with group in {rag, rag_real}; combine (two-pair) rows, rows whose value cannot be located in
the cited passage and present rows whose label no admissible decode span can reproduce are skipped and counted.
The presence CE is weighted per kind by the inverse training frequency (mean 1), so the rarer abstentions are
not drowned by present rows. Checkpoint selection: highest internal-validation macro strict accuracy (mean of
the present / missing / conflict strict answer accuracies, i.e. the emitted answer string equals the gold
answer), ties broken by lower validation loss; the initial checkpoint takes part. Held-out evaluation
(scripts/evaluate_extractive.py) is never used for selection.
"""
import argparse
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from trid.corpus import sha256, write_json
from trid.extractive import (ExtractiveReader, encode, loss_terms, decode, batch, MAX_SPAN, STAGE, WEIGHTS, PRESENCE,
                             ATTENTION, HEADS)
from trid.span_boundaries import BOUNDARY_POLICY
from trid.runtime import load_run, sync

GROUPS = ('rag', 'rag_real')
KINDS = ('present', 'missing', 'conflict')
SELECTION = ('Highest internal-validation macro_strict_accuracy (unweighted mean of present_strict_accuracy, '
             'missing_strict_accuracy and conflict_strict_accuracy over the kinds the validation set contains; '
             'strict = presence + exact span text + source id), ties broken by lower valid_loss; initial checkpoint '
             'included. strict_accuracy, presence_accuracy and span_exact_on_present are diagnostics only, never '
             'the selection or final QA metric.')


def load_rows(tok, data, split, context, limit, seed):
    """(rows, skipped, counts): skipped counts every excluded line by reason; counts['usable'] is the per-kind count
    before `limit`, counts['sampled'] after (equal when no limit)."""
    rows, skipped = [], Counter()
    lines = (data / f'{split}.jsonl').read_text().splitlines()
    for line in lines:
        r = json.loads(line)
        if r.get('group') not in GROUPS or r.get('history'):
            skipped['other_group'] += 1
            continue
        if r['task'] == 'rag_combine':
            skipped['combine'] += 1
            continue
        try:
            row = encode(tok, r['prompt'], r['answer'], context)
        except ValueError:
            skipped['context'] += 1
            continue
        if row is None:
            skipped['unmapped_answer'] += 1
            continue
        if row['kind'] == 0 and not row['label_decodable']:
            skipped['label_' + (row.get('label_reason') or 'outside_decode_rules')] += 1
            continue
        row['task'] = r['task']
        rows.append(row)
    usable = Counter(KINDS[r['kind']] for r in rows)
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    if not rows:
        raise ValueError(f'No usable rows in {split}')
    counts = {'usable': dict(usable), 'sampled': dict(Counter(KINDS[r['kind']] for r in rows))}
    return rows, dict(skipped), counts


def presence_weights(rows):
    """Per-kind presence CE weights (PRESENCE order) from the training rows: inverse frequency, normalised to mean
    1 over the three kinds; a kind without rows gets the weight of a single row."""
    counts = Counter(r['kind'] for r in rows)
    raw = [1 / max(counts[k], 1) for k in range(len(KINDS))]
    return [w * len(raw) / sum(raw) for w in raw]


def expected_answer(row):
    """Gold answer string in the RAG answer grammar, rebuilt from the encoded row."""
    if row['kind'] != 0:
        return PRESENCE[row['kind']]
    sid = row['passages'][row['char_passage'][row['start']]][0]
    return f"{row['prompt'][row['start']:row['end'] + 1]} [{sid}]"


def strict_correct(out, row):
    """The emitted answer equals the gold answer: presence, span text and source id all right. A right span with
    a wrong presence fails; an equal text at another occurrence inside the cited passage passes."""
    return out['answer'] == expected_answer(row)


NORMALIZATION = ('valid_loss and each train step: weighted presence CE summed over all rows divided by the row count '
                 '(N) + summed CE(start) + CE(end) over present rows divided by the present count (P); the weight '
                 'of a row is presence_class_weights[kind] (training inverse frequency, mean 1); N and P are '
                 'counted over the whole validation set / the whole effective batch, so the value and the gradient '
                 'do not depend on the micro-batch partition.')


@torch.inference_mode()
def validate(model, rows, pad, device, batch_size, weights=None):
    """One forward per batch: the same outputs feed loss_terms() and decode(). valid_loss normalises over the
    whole set (NORMALIZATION, with the training `weights`), so it is independent of `batch_size`. Returns strict
    accuracy over all rows, per-kind strict accuracy with row counts, their macro mean (the selection metric),
    per-kind weighted presence CE, and the diagnostics presence_accuracy / span_exact_on_present."""
    was_training = model.training
    model.eval()
    presence_total = span_total = 0.; ok = Counter(); n = Counter(); presence_loss = Counter(); presence_ok = 0; span_ok = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        outputs = model(*batch(chunk, pad, device))
        presence, span, _ = loss_terms(model, chunk, pad, device, outputs=outputs, weights=weights)
        per_row = presence.tolist(); presence_total += sum(per_row); span_total += float(span)
        kind, scores = outputs
        for i, row in enumerate(chunk):
            out = decode((kind[i], scores[i]), row)
            name = KINDS[row['kind']]
            n[name] += 1; ok[name] += strict_correct(out, row); presence_loss[name] += per_row[i]
            presence_ok += out['presence'] == row['kind']
            if row['kind'] == 0:
                span_ok += out['span'] == [row['start'], row['end'] + 1]
    if was_training:
        model.train()
    size = len(rows)
    total = presence_total / size + (span_total / n['present'] if n['present'] else 0.)
    result = {'valid_loss': total, 'strict_accuracy': sum(ok.values()) / size, 'rows': size}
    for name in KINDS:
        result[f'{name}_rows'] = n[name]
        result[f'{name}_strict_accuracy'] = ok[name] / n[name] if n[name] else None
        result[f'{name}_presence_loss'] = presence_loss[name] / n[name] if n[name] else None
    result['macro_strict_accuracy'] = macro_strict_accuracy(result)
    result.update(presence_accuracy=presence_ok / size, span_exact_on_present=span_ok / n['present'] if n['present'] else None)
    return result


def macro_strict_accuracy(score):
    """Unweighted mean of the per-kind strict accuracies over the kinds with rows (None for none)."""
    values = [score[f'{name}_strict_accuracy'] for name in KINDS if score.get(f'{name}_strict_accuracy') is not None]
    return sum(values) / len(values) if values else None


def train_step(model, draws, pad, device, micro_batch_size, weights=None):
    """Accumulates the gradient of loss(draws, weights=weights) over independently padded micro-batches of
    `micro_batch_size` rows (NORMALIZATION: one N and one P for the whole effective batch) and returns the loss
    value as a float. The caller zeroes gradients before and clips / steps once after."""
    present = sum(r['kind'] == 0 for r in draws)
    total = 0.
    for start in range(0, len(draws), micro_batch_size):
        presence, span, _ = loss_terms(model, draws[start:start + micro_batch_size], pad, device, weights=weights)
        partial = presence.sum() / len(draws)
        if present:
            partial = partial + span / present
        if not torch.isfinite(partial):
            raise RuntimeError('Nonfinite loss')
        partial.backward()
        total += float(partial.detach())
    return total


def better(score, best):
    """Selection order: higher macro_strict_accuracy, then lower valid_loss. `best` None accepts."""
    if best is None:
        return True
    return (score['macro_strict_accuracy'], -score['valid_loss']) > (best['macro_strict_accuracy'], -best['valid_loss'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-run', required=True)
    p.add_argument('--data', default='data/korean-rag-curriculum-v2')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-limit', type=int, default=0, help='random subsample of usable training rows (smoke tests)')
    p.add_argument('--validation-limit', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--micro-batch-size', type=int, default=4,
                   help='rows per forward/backward; gradients accumulate to --batch-size with unchanged draws and '
                        'normalisation (memory bound); capped at --batch-size')
    p.add_argument('--attention', choices=ATTENTION, default='causal',
                   help='backbone attention inside each prompt: causal (passage states never see the question) or '
                        'bidirectional (full attention, padding keys masked); recorded as reader_attention')
    p.add_argument('--head', choices=HEADS, default='v2',
                   help='head layout (trid.extractive.ExtractiveReader): v1 linear per-character key and presence from '
                        'the last hidden state only; v2 neighbour-aware MLP key and presence that sees the span '
                        'evidence; recorded as reader_head')
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.micro_batch_size, a.eval_every, a.context) < 1 or not math.isfinite(a.lr) or a.lr <= 0:
        p.error('Invalid training options')
    micro = min(a.micro_batch_size, a.batch_size)
    root, data = Path(a.output), Path(a.data)
    if root.exists():
        p.error('Use a new output directory')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    base, tok, device = load_run(a.init_run, a.device)
    if a.context > base.cfg.context_length:
        p.error('--context exceeds the base context length')
    cfg = replace(base.cfg, context_length=a.context)
    model = ExtractiveReader(cfg, a.attention, a.head)
    model.backbone.load_state_dict(base.state_dict()); del base
    model.to(device)
    rows, skipped, counts = {}, {}, {}
    for split, limit in (('train', a.train_limit), ('valid', a.validation_limit)):
        rows[split], skipped[split], counts[split] = load_rows(tok, data, split, a.context, limit, a.seed)
    weights = presence_weights(rows['train'])
    root.mkdir(parents=True)
    shutil.copy2(Path(a.init_run) / 'tokenizer.json', root / 'tokenizer.json')
    write_json(root / 'config.json', asdict(cfg))
    provenance_path = data / 'provenance.json'
    meta = {'stage': STAGE, 'status': 'training', 'initialization': a.init_run, 'reader_attention': a.attention,
            'reader_head': a.head,
            'initial_weights_sha256': sha256(Path(a.init_run) / 'model.safetensors'),
            'tokenizer_sha256': sha256(root / 'tokenizer.json'), 'parameters': sum(p.numel() for p in model.parameters()),
            'steps_requested': a.steps, 'batch_size': a.batch_size, 'micro_batch_size': micro, 'learning_rate': a.lr,
            'seed': a.seed, 'eval_every': a.eval_every, 'device': device, 'context_length': a.context,
            'groups': list(GROUPS), 'counts': counts, 'skipped_rows': skipped,
            'train_limit': a.train_limit or None, 'validation_limit': a.validation_limit or None,
            'data': str(data), 'data_sha256': {s: sha256(data / f'{s}.jsonl') for s in rows},
            'data_provenance': json.loads(provenance_path.read_text()) if provenance_path.exists() else None,
            'input_format': 'trid.prompts.rag_prompt (unchanged RAG prompt)',
            'objective': 'Weighted CE over 3-way presence (답 있음/확인 불가/근거 충돌) + CE(start) + CE(end) over passage characters on present rows; all backbone weights trainable; no LM objective',
            'loss_normalization': NORMALIZATION,
            'presence_class_weights': dict(zip(KINDS, weights)),
            'presence_class_weights_rule': 'inverse frequency of the sampled training rows per kind, normalised to mean 1 over the three kinds; applied to training and validation presence CE alike',
            'span_unit': 'Unicode characters of the prompt; source id from the passage containing the span',
            'max_span_characters': MAX_SPAN, 'boundary_policy': BOUNDARY_POLICY, 'sampling': 'Uniform rows, original class ratio',
            'checkpoint_selection': SELECTION, 'selection_metric': 'macro_strict_accuracy', 'selection_tiebreak': 'valid_loss',
            'weights_file': WEIGHTS}
    write_json(root / 'training.json', meta)
    pad = tok.token_to_id('<|pad|>')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(.9, .95), weight_decay=.1)
    rng = random.Random(a.seed)

    def save(name):
        temp = root / 'checkpoint.tmp.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp))
        temp.replace(root / name)

    sync(device); started = time.perf_counter()
    initial = validate(model, rows['valid'], pad, device, micro, weights); best = initial; best_step = 0
    save(WEIGHTS)
    metrics = [{'step': 0, **initial}]
    write_json(root / 'metrics.json', metrics)
    print(f'{meta["parameters"]:,} parameters | {device} | {counts} | skipped {skipped} | initial {initial}', flush=True)
    for step in range(1, a.steps + 1):
        warmup = min(50, max(1, a.steps // 10))
        ratio = step / warmup if step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, a.steps - warmup)))
        for group in opt.param_groups:
            group['lr'] = a.lr * ratio
        opt.zero_grad(set_to_none=True)
        error = train_step(model, rng.choices(rows['train'], k=a.batch_size), pad, device, micro, weights)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            score = validate(model, rows['valid'], pad, device, micro, weights); sync(device)
            if better(score, best):
                best, best_step = score, step
                save(WEIGHTS)
            save('last-' + WEIGHTS)
            entry = {'step': step, 'train_loss': error, **score,
                     'elapsed_seconds': round(time.perf_counter() - started, 2)}
            metrics.append(entry); write_json(root / 'metrics.json', metrics)
            meta.update(completed_steps=step, best_step=best_step, best_macro_strict_accuracy=best['macro_strict_accuracy'],
                        best_strict_accuracy=best['strict_accuracy'], best_validation_loss=best['valid_loss'])
            write_json(root / 'training.json', meta)
            print(json.dumps(entry), flush=True)
    meta.update(status='complete', initial_validation=initial, best_step=best_step, best_validation=best,
                best_macro_strict_accuracy=best['macro_strict_accuracy'], best_strict_accuracy=best['strict_accuracy'],
                best_validation_loss=best['valid_loss'],
                training_seconds=round(time.perf_counter() - started, 2),
                weight_bytes=(root / WEIGHTS).stat().st_size, optimizer_resume_supported=False)
    write_json(root / 'training.json', meta)


if __name__ == '__main__':
    main()

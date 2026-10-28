"""Train the extractive head (trid.extractive) on the RAG rows of a curriculum, from a base run.

Uses only rows with group in {rag, rag_real}; combine (two-pair) rows and rows whose value cannot be located in
the cited passage are skipped and counted. Checkpoint selection by internal validation loss only.
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
from safetensors.torch import save_file
from trid.corpus import sha256, write_json
from trid.extractive import ExtractiveReader, encode, loss, decode, batch, MAX_SPAN, STAGE, WEIGHTS
from trid.runtime import load_run, sync

GROUPS = ('rag', 'rag_real')


def load_rows(tok, data, split, context, limit, seed):
    rows, skipped, counts = [], Counter(), Counter()
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
        row['task'] = r['task']
        rows.append(row)
        counts[('present', 'missing', 'conflict')[row['kind']]] += 1
        if row['kind'] == 0 and not row['label_decodable']:
            counts['present_label_outside_decode_rules'] += 1
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
        counts = Counter(('present', 'missing', 'conflict')[r['kind']] for r in rows)
    if not rows:
        raise ValueError(f'No usable rows in {split}')
    return rows, dict(skipped), dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-run', required=True)
    p.add_argument('--data', default='data/korean-rag-curriculum-v2')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-limit', type=int, default=0, help='random subsample of usable training rows (smoke tests)')
    p.add_argument('--validation-limit', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.eval_every, a.context) < 1 or not math.isfinite(a.lr) or a.lr <= 0:
        p.error('Invalid training options')
    root, data = Path(a.output), Path(a.data)
    if root.exists():
        p.error('Use a new output directory')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    base, tok, device = load_run(a.init_run, a.device)
    if a.context > base.cfg.context_length:
        p.error('--context exceeds the base context length')
    cfg = replace(base.cfg, context_length=a.context)
    model = ExtractiveReader(cfg)
    model.backbone.load_state_dict(base.state_dict()); del base
    model.to(device)
    rows, skipped, counts = {}, {}, {}
    for split, limit in (('train', a.train_limit), ('valid', a.validation_limit)):
        rows[split], skipped[split], counts[split] = load_rows(tok, data, split, a.context, limit, a.seed)
    root.mkdir(parents=True)
    shutil.copy2(Path(a.init_run) / 'tokenizer.json', root / 'tokenizer.json')
    write_json(root / 'config.json', asdict(cfg))
    provenance_path = data / 'provenance.json'
    meta = {'stage': STAGE, 'status': 'training', 'initialization': a.init_run,
            'initial_weights_sha256': sha256(Path(a.init_run) / 'model.safetensors'),
            'tokenizer_sha256': sha256(root / 'tokenizer.json'), 'parameters': sum(p.numel() for p in model.parameters()),
            'steps_requested': a.steps, 'batch_size': a.batch_size, 'learning_rate': a.lr, 'seed': a.seed,
            'eval_every': a.eval_every, 'device': device, 'context_length': a.context,
            'groups': list(GROUPS), 'counts': counts, 'skipped_rows': skipped,
            'train_limit': a.train_limit or None, 'validation_limit': a.validation_limit or None,
            'data': str(data), 'data_sha256': {s: sha256(data / f'{s}.jsonl') for s in rows},
            'data_provenance': json.loads(provenance_path.read_text()) if provenance_path.exists() else None,
            'input_format': 'trid.prompts.rag_prompt (unchanged RAG prompt)',
            'objective': 'CE over 3-way presence (답 있음/확인 불가/근거 충돌) + CE(start) + CE(end) over passage characters on present rows; all backbone weights trainable; no LM objective',
            'span_unit': 'Unicode characters of the prompt; source id from the passage containing the span',
            'max_span_characters': MAX_SPAN, 'sampling': 'Uniform rows, original class ratio',
            'checkpoint_selection': 'Minimum internal-validation loss, including initial checkpoint',
            'weights_file': WEIGHTS}
    write_json(root / 'training.json', meta)
    pad = tok.token_to_id('<|pad|>')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(.9, .95), weight_decay=.1)
    rng = random.Random(a.seed)

    @torch.inference_mode()
    def validate():
        model.eval(); total = 0.; presence_ok = 0; span_ok = 0; present = 0
        for start in range(0, len(rows['valid']), a.batch_size):
            chunk = rows['valid'][start:start + a.batch_size]
            total += float(loss(model, chunk, pad, device)) * len(chunk)
            kind, scores = model(*batch(chunk, pad, device))
            for i, row in enumerate(chunk):
                out = decode((kind[i], scores[i]), row)
                presence_ok += out['presence'] == row['kind']
                if row['kind'] == 0:
                    present += 1
                    span_ok += out['span'] == [row['start'], row['end'] + 1]
        model.train()
        n = len(rows['valid'])
        return {'valid_loss': total / n, 'presence_accuracy': presence_ok / n,
                'span_exact_on_present': span_ok / max(1, present), 'present_rows': present}

    def save(name):
        temp = root / 'checkpoint.tmp.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp))
        temp.replace(root / name)

    sync(device); started = time.perf_counter()
    initial = validate(); best = initial['valid_loss']; best_step = 0
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
        error = loss(model, rng.choices(rows['train'], k=a.batch_size), pad, device)
        if not torch.isfinite(error):
            raise RuntimeError('Nonfinite loss')
        error.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            score = validate(); sync(device)
            if score['valid_loss'] < best:
                best, best_step = score['valid_loss'], step
                save(WEIGHTS)
            save('last-' + WEIGHTS)
            entry = {'step': step, 'train_loss': float(error.detach()), **score,
                     'elapsed_seconds': round(time.perf_counter() - started, 2)}
            metrics.append(entry); write_json(root / 'metrics.json', metrics)
            meta.update(completed_steps=step, best_step=best_step, best_validation_loss=best)
            write_json(root / 'training.json', meta)
            print(json.dumps(entry), flush=True)
    meta.update(status='complete', initial_validation=initial, best_step=best_step, best_validation_loss=best,
                training_seconds=round(time.perf_counter() - started, 2),
                weight_bytes=(root / WEIGHTS).stat().st_size, optimizer_resume_supported=False)
    write_json(root / 'training.json', meta)


if __name__ == '__main__':
    main()

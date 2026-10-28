"""Train only source positions and answer presence, without dialogue or generation loss."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import shutil
import time
import torch
from safetensors.torch import save_file
from trid.corpus import sha256, write_json
from trid.korean_reader import KoreanReader, encode, source_parts, loss, MAX_SPAN
from trid.runtime import load_run, sync


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/korean-mrc-v1')
    p.add_argument('--init-run', default='runs/trid-korean-base-v1')
    p.add_argument('--output', default='runs/trid-korean-span-v1')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.eval_every) < 1 or not math.isfinite(a.lr) or a.lr <= 0:
        p.error('Invalid training options')
    root, data = Path(a.output), Path(a.data)
    if root.exists():
        p.error('Use a new output directory')
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    base, tok, device = load_run(a.init_run)
    cfg = replace(base.cfg, context_length=512)
    model = KoreanReader(cfg)
    model.backbone.load_state_dict(base.state_dict()); del base
    model.to(device)
    rows, contexts, counts = {}, {}, {}
    for split in ('train', 'valid'):
        rows[split], contexts[split] = [], set()
        counts[split] = {'answerable': 0, 'unanswerable': 0}
        for line in (data/f'{split}.jsonl').read_text().splitlines():
            r = json.loads(line)
            if not r['task'].startswith('korean_mrc_') or r.get('history'):
                raise ValueError('Only reading examples without conversation history are allowed')
            document, question = source_parts(r)
            row = encode(tok, document, question, cfg.context_length,
                         r['answer_variants'] or [r['answer']], r['is_impossible'])
            rows[split].append(row)
            contexts[split].add(r['context_sha256'])
            counts[split]['unanswerable' if r['is_impossible'] else 'answerable'] += 1
        if not rows[split]:
            raise ValueError('Empty data split')
    if contexts['train'] & contexts['valid']:
        raise ValueError('Training/validation document overlap')
    root.mkdir(parents=True)
    shutil.copy2(Path(a.init_run)/'tokenizer.json', root/'tokenizer.json')
    write_json(root/'config.json', asdict(cfg))
    meta = {'stage': 'korean_span_reader_v1', 'status': 'training', 'initialization': a.init_run,
            'initial_weights_sha256': sha256(Path(a.init_run)/'model.safetensors'),
            'tokenizer_sha256': sha256(root/'tokenizer.json'), 'parameters': sum(p.numel() for p in model.parameters()),
            'steps_requested': a.steps, 'batch_size': a.batch_size, 'learning_rate': a.lr, 'seed': a.seed,
            'eval_every': a.eval_every, 'device': device, 'counts': counts, 'skipped_rows': 0,
            'data_sha256': {s: sha256(data/f'{s}.jsonl') for s in rows},
            'input_format': '질문: {question}\n자료: {full_document}',
            'objective': 'Per-example 2-class answer-presence CE plus half negative log marginal probability of valid source start/end pairs; all backbone weights trainable; no LM objective',
            'span_unit': 'Unicode characters; document-relative half-open offsets at inference',
            'max_span_characters': MAX_SPAN, 'sampling': 'Uniform rows, original class ratio',
            'checkpoint_selection': 'Minimum full internal-validation loss, including initial checkpoint',
            'data_provenance': json.loads((data/'provenance.json').read_text())}
    write_json(root/'training.json', meta)
    pad = tok.token_to_id('<|pad|>')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(.9, .95), weight_decay=.1)
    rng = random.Random(a.seed)

    @torch.inference_mode()
    def validate():
        model.eval(); total = 0.
        for start in range(0, len(rows['valid']), a.batch_size):
            chunk = rows['valid'][start:start+a.batch_size]
            total += float(loss(model, chunk, pad, device)) * len(chunk)
        model.train()
        return total/len(rows['valid'])

    def save(name):
        temp = root/'checkpoint.tmp.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp))
        temp.replace(root/name)

    sync(device); started = time.perf_counter()
    best = initial = validate(); best_step = 0
    save('reader.safetensors')
    metrics = [{'step': 0, 'valid_loss': initial}]
    print(f'{meta["parameters"]:,} parameters | {device} | {counts} | initial {initial:.4f}', flush=True)
    for step in range(1, a.steps+1):
        warmup = min(50, max(1, a.steps//10))
        ratio = step/warmup if step <= warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1, a.steps-warmup)))
        for group in opt.param_groups:
            group['lr'] = a.lr*ratio
        opt.zero_grad(set_to_none=True)
        error = loss(model, rng.choices(rows['train'], k=a.batch_size), pad, device)
        if not torch.isfinite(error):
            raise RuntimeError('Nonfinite loss')
        error.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            score = validate(); sync(device)
            if score < best:
                best, best_step = score, step
                save('reader.safetensors')
            save('last-reader.safetensors')
            entry = {'step': step, 'train_loss': float(error.detach()), 'valid_loss': score,
                     'elapsed_seconds': round(time.perf_counter()-started, 2)}
            metrics.append(entry); write_json(root/'metrics.json', metrics)
            meta.update(completed_steps=step, best_step=best_step, best_validation_loss=best)
            write_json(root/'training.json', meta)
            print(json.dumps(entry), flush=True)
    meta.update(status='complete', initial_validation_loss=initial,
                training_seconds=round(time.perf_counter()-started, 2),
                weight_bytes=(root/'reader.safetensors').stat().st_size,
                optimizer_resume_supported=False)
    write_json(root/'training.json', meta)


if __name__ == '__main__':
    main()

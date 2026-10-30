"""Measure SFT step time for trid.training's objective under execution variants (no weights are saved).

Same model init, data (mix-04), sampling (seed 271030, batch 8) and example-normalized answer loss as
trid.training.train; only the execution changes:
  fp32-micro2   current recipe: fp32, 4 micro-batches of 2 (gradient accumulation)
  fp32-micro8   fp32, one pass of 8 (same expected gradient; padding to the batch maximum)
  bf16-micro8   bf16 autocast, one pass of 8
  bf16-micro8-compile  bf16 autocast, one pass of 8, torch.compile(dynamic=True), sequence padded to a multiple of 64
Each variant runs --warmup steps (not timed) then --steps timed optimizer steps; validation (64 rows) is timed once.

  python scripts/bench_sft_speed.py --init <run dir> --data data/public-dialogue-mix-04 --output <json>
"""
import argparse, json, random, time
from pathlib import Path

import torch

from trid.runtime import load_run, sync
from trid.text import encode_example, read_rows
from trid.training import batch, loss_terms, validation_loss


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    a = p.parse_args()
    results = {}
    rows = read_rows(a.data / 'train.jsonl')
    for variant in ('fp32-micro2', 'fp32-micro8', 'bf16-micro8', 'bf16-micro8-compile'):
        model, tok, device = load_run(a.init, 'cuda')
        model.cfg.context_length = 1024
        examples = [encode_example(tok, r, 1024) for r in rows]
        weights = [float(r.get('weight', 1)) for r in rows]
        valid = [encode_example(tok, r, 1024) for r in random.Random(271031).sample(read_rows(a.data / 'valid.jsonl'), 64)]
        pad = tok.token_to_id('<|pad|>')
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.1, betas=(.9, .95))
        micro = 2 if variant == 'fp32-micro2' else 8
        bf16 = variant.startswith('bf16')
        fwd = torch.compile(model, dynamic=True) if variant.endswith('compile') else model
        rng = random.Random(271030)
        times = []
        for i in range(a.warmup + a.steps):
            selected = [examples[j] for j in rng.choices(range(len(examples)), weights=weights, k=8)]
            sync(device); t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for s in range(0, 8, micro):
                x, y = batch(selected[s:s + micro], pad, device)
                if variant.endswith('compile'):
                    extra = -x.shape[1] % 64
                    x = torch.nn.functional.pad(x, (0, extra), value=pad)
                    y = torch.nn.functional.pad(y, (0, extra), value=-100)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    answer_sum, _, _, _ = loss_terms(fwd, x, y, None, 'example')
                (answer_sum / 8).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sync(device)
            if i >= a.warmup:
                times.append(time.perf_counter() - t)
        sync(device); t = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            validation_loss(model, valid, pad, device, micro, loss_normalization='example')
        sync(device)
        results[variant] = {'seconds_per_step_mean': sum(times) / len(times), 'seconds_per_step_median': sorted(times)[len(times) // 2],
                            'validation_64_rows_seconds': time.perf_counter() - t, 'timed_steps': len(times),
                            'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30}
        print(variant, json.dumps(results[variant]), flush=True)
        del model, opt, fwd
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    results['gpu'] = torch.cuda.get_device_name()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()

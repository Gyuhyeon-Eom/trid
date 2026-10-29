"""Continued pretraining on a rented GPU (AWS g6e L40S / g5 A10G): scripts/train_medium_cpt_v1.py with every
fixed value exposed as a flag. Same resumable session loop: runs `--stop-after` steps per pretrain() call until
training.json is complete, re-reads the stop flag between sessions, and Ctrl-C keeps the last complete checkpoint.

Tokens per optimizer step = batch_size x accumulate x context (1,024). The default 16 x 4 = 65,536 tokens/step
matches the Mac job's 2 x 32 x 1,024; the batch/accumulate split only changes peak memory (activations scale with
batch_size), not the expected update: pretrain() draws one batch of batch_size x accumulate rows per step and
divides each micro loss by accumulate, so the gradient is identical in expectation for any split with the same
product. Raise batch_size (lower accumulate) until the GPU is nearly full; 16 x 4 fits a 24 GB A10G for the 91M
model with bf16 autocast, 48 GB L40S can take 32 x 2 or 64 x 1.

Example (L40S):
  python scripts/train_medium_cpt_gpu.py --data data/mix-v3-basetok-tokens --init-run runs/trid-korean-medium-base-v1 \
      --output runs/trid-korean-medium-cpt-gpu --steps 15259 --device cuda --precision bf16 --compile
Resume is automatic: rerun the same command (only --stop-after/--device/--precision/--compile are per-session;
schedule, optimizer, sampling and checkpoint cadence come from training.json, and runs created before those keys
existed keep their old behaviour). Validation runs every --eval-every steps; checkpoints and best-weight writes
only every --checkpoint-every (a multiple of eval). --sampling epoch walks every non-overlapping 1,024-token window
of train.bin once per epoch in a seeded permutation (cursor checkpointed); --source-weights (epoch only, needs
data/sources.json from trid.corpus.write_sources) repeats or thins windows by the document kind at their start,
e.g. --source-weights press_release=2 ordin=0.5. Never exports or activates a model.
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='data/mix-v3-basetok-tokens')
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument('--init-run', help='completed local pretraining run to continue from')
    start.add_argument('--config', help='model config JSON for a fresh random initialization')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=15259)
    p.add_argument('--stop-after', type=int, default=5000, help='optimizer steps per session before checkpoint/rescan')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=271029)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=1000, help='multiple of --eval-every; best weights written only here')
    p.add_argument('--sampling', choices=['replacement', 'epoch'], default='epoch')
    p.add_argument('--source-weights', nargs='*', metavar='KIND=W|JSON',
                   help='epoch sampling only: per-kind window multipliers as kind=weight pairs or one JSON object')
    p.add_argument('--decay-norms', action=argparse.BooleanOptionalAction, default=False,
                   help='apply weight decay to RMSNorm gains / 1-D params too (old behaviour)')
    p.add_argument('--validation-batches', type=int, default=8)
    p.add_argument('--schedule', choices=['cosine', 'wsd'], default='wsd')
    p.add_argument('--decay-fraction', type=float, default=0.2)
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stop-flag', default='reports/medium-cpt-gpu-stop.flag',
                   help='create this file to end after the current session, keeping the checkpoint')
    p.add_argument('--max-sessions', type=int, default=0,
                   help='end after this many pretrain() sessions in this process, keeping the checkpoint (0: until complete)')
    a = p.parse_args(); os.chdir(ROOT)
    weights = None
    if a.source_weights:
        if len(a.source_weights) == 1 and a.source_weights[0].lstrip().startswith('{'):
            weights = json.loads(a.source_weights[0])
        else:
            weights = {k: float(v) for k, v in (item.split('=', 1) for item in a.source_weights)}
    out, stop = Path(a.output), Path(a.stop_flag)
    from trid.pretraining import pretrain
    sessions = 0
    while True:
        meta = json.loads((out / 'training.json').read_text()) if (out / 'training.json').exists() else {}
        if meta.get('status') == 'complete':
            print('complete'); return
        if stop.exists():
            print('stop flag present; leaving checkpoint'); return
        if a.max_sessions and sessions >= a.max_sessions:
            print(f'max sessions ({a.max_sessions}) reached; rerun the same command to resume'); return
        sessions += 1
        session = dict(stop_after=a.stop_after, device=a.device, precision=a.precision, compile=a.compile)
        if meta:
            pretrain(a.data, str(out), resume=True, **session)
        else:
            pretrain(a.data, str(out), init_run=a.init_run, config_path=a.config, steps=a.steps,
                     batch_size=a.batch_size, accumulate=a.accumulate, lr=a.lr, seed=a.seed,
                     eval_every=a.eval_every, validation_batches=a.validation_batches, checkpoint_every=a.checkpoint_every,
                     sampling=a.sampling, source_weights=weights, decay_norms=a.decay_norms,
                     schedule=a.schedule, decay_fraction=a.decay_fraction, **session)
        if json.loads((out / 'training.json').read_text()).get('status') == 'interrupted':
            print('interrupted; rerun the same command to resume'); return


if __name__ == '__main__':
    main()

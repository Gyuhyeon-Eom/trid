"""Continued pretraining of the medium base on the law/briefing/wiki mix (data/mix-v3-basetok-tokens, base tokenizer).

1,000,013,824 tokens = 15,259 steps x (batch 2 x accumulate 32 x context 1,024), WSD schedule (20% decay), lr 3e-4,
bf16 autocast + torch.compile. Runs in resumable sessions of `--stop-after` steps until training.json is complete;
Ctrl-C keeps the last checkpoint. Never touches runs/trid-korean-medium-base-v1. No activation or export.
Pilot (20 steps, 2026-10-29): valid loss 3.173 -> 3.012, 8,081 tokens/s.
"""
import argparse, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = 'data/mix-v3-basetok-tokens'
BASE = 'runs/trid-korean-medium-base-v1'
OUT = ROOT / 'runs/trid-korean-medium-cpt-v1'
STOP = ROOT / 'reports/medium-cpt-stop.flag'
STEPS = 15259


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stop-after', type=int, default=250)
    a = p.parse_args(); os.chdir(ROOT)
    from trid.pretraining import pretrain
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    try:
        while True:
            meta = json.loads((OUT / 'training.json').read_text()) if (OUT / 'training.json').exists() else {}
            if meta.get('status') == 'complete':
                print('complete'); return
            if STOP.exists():
                print('stop flag present; leaving checkpoint'); return
            if meta:
                pretrain(DATA, str(OUT), resume=True, stop_after=a.stop_after, device='mps', precision='bf16', compile=True)
            else:
                pretrain(DATA, str(OUT), init_run=BASE, steps=STEPS, stop_after=a.stop_after, batch_size=2, accumulate=32,
                         lr=3e-4, device='mps', seed=271029, eval_every=50, validation_batches=8,
                         precision='bf16', compile=True, schedule='wsd', decay_fraction=0.2)
    finally:
        if awake.poll() is None:
            awake.terminate()


if __name__ == '__main__':
    main()

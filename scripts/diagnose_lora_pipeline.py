"""Separate "the training pipeline is broken" from "the training data teaches the wrong thing" (MODEL-REPAIR-001).

Uses scripts/train_open_weight_lora.py's own load_rows / encode / batch_loss / LoRA settings, so the template, the
answer-only loss mask and the EOS are exactly the training contract.

  zero-update  An untrained LoRA (B = 0) and its merge must not change the base model: max |logit difference| over
               the encoded rows, before and after merge_and_unload. A non-zero difference means the wrap/merge path
               itself alters the model.
  overfit      Train the same LoRA on a few rows for --steps full-batch steps, then greedy-generate each row's answer
               from its prompt. A working pipeline drives the answer loss near zero and reproduces the targets
               verbatim, ending with EOS. Failure here means masking/EOS/optimisation is broken, independent of data.

Same isolated environment as the trainer (see its docstring); development diagnostic, no checkpoint is saved.

  uv run --no-project --python 3.12 --with 'torch==2.9.*' --with 'transformers>=4.51,<5' --with 'peft==0.17.*' \\
      --with safetensors --with numpy --with tokenizers python scripts/diagnose_lora_pipeline.py \\
      --model runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc --manifest <manifest.json> --mode overfit \\
      --ids id1,id2,... --out reports/<name>.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_open_weight_lora import TARGETS, batch_loss, encode, load_rows, sha256


@torch.no_grad()
def logits_of(model, items, device):
    return [model(input_ids=torch.tensor([x['ids']], device=device)).logits.float().cpu() for x in items]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--mode', choices=['zero-update', 'overfit'], required=True)
    p.add_argument('--ids', help='comma-separated row ids (default: first --rows train rows)')
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=6144)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    started = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    manifest = json.loads(a.manifest.read_text())
    rows = [r for e in manifest['train'] for r in load_rows(e, a.manifest.resolve().parent)]
    if a.ids:
        wanted = a.ids.split(',')
        rows = [r for r in rows if r['id'] in wanted]
    rows = list({r['id']: r for r in rows}.values())[:a.rows if not a.ids else None]
    items = [x for x in (encode(tok, r, 'qwen3', a.max_tokens) for r in rows) if x is not None]
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
    config = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=TARGETS, task_type='CAUSAL_LM')
    report = dict(mode=a.mode, model=str(a.model), manifest=str(a.manifest), manifest_sha256=sha256(a.manifest),
                  rows=[r['id'] for r in rows], answer_tokens=[sum(t != -100 for t in x['labels']) for x in items],
                  device=a.device, dtype=str(dtype), versions=dict(torch=torch.__version__))
    if a.mode == 'zero-update':
        model.eval()
        base = logits_of(model, items, a.device)
        wrapped = get_peft_model(model, config)
        wrapped.eval()
        lora = logits_of(wrapped, items, a.device)
        merged = wrapped.merge_and_unload()
        merged.eval()
        after = logits_of(merged, items, a.device)
        report.update(max_abs_diff_lora=max(float((x - y).abs().max()) for x, y in zip(base, lora)),
                      max_abs_diff_merged=max(float((x - y).abs().max()) for x, y in zip(base, after)))
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        params = [q for q in model.parameters() if q.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        history = []
        model.train()
        for step in range(1, a.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for x in items:  # full batch, micro batch 1: the trainer's example-normalised answer loss
                loss = batch_loss(model, [x], a.device, pad_id) / len(items)
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            history.append({'step': step, 'loss': round(total, 5), 'elapsed_seconds': round(time.time() - started, 1)})
            print(json.dumps(history[-1]), flush=True)
        model.eval()
        generations = []
        with torch.no_grad():
            for r, x in zip(rows, items):
                prompt = [t for t, l in zip(x['ids'], x['labels']) if l == -100]
                target = [t for t in x['labels'] if t != -100]
                out = model.generate(input_ids=torch.tensor([prompt], device=a.device), max_new_tokens=len(target) + 32,
                                     do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=pad_id)[0][len(prompt):].tolist()
                generations.append({'id': r['id'], 'exact': out == target, 'stopped_at_eos': bool(out) and out[-1] == tok.eos_token_id,
                                    'generated_tokens': len(out), 'target_tokens': len(target),
                                    'first_mismatch': next((i for i, (g, t) in enumerate(zip(out, target)) if g != t), None)})
        report.update(steps=a.steps, lr=a.lr, history=history, generations=generations)
    report['elapsed_seconds'] = round(time.time() - started, 1)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ('history',)}, ensure_ascii=False)[:2000])


if __name__ == '__main__':
    main()

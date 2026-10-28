from dataclasses import asdict, replace
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from .model import Config, Trid
from .runtime import device_for, sync
from .text import read_rows, train_tokenizer, encode_example


def batch(examples, pad, device):
    length = max(len(x) for x, _ in examples)
    x = torch.full((len(examples), length), pad, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (tokens, labels) in enumerate(examples):
        x[i, :len(tokens)] = torch.tensor(tokens)
        y[i, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)


def loss_for(model, x, y, intents=None, intent_weight=0.0, loss_normalization="token"):
    if loss_normalization not in {"token", "example"}:
        raise ValueError("loss_normalization must be token or example")
    logits, _, hidden = model(x, return_hidden=True)
    if loss_normalization == "token":
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
    else:
        token_losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                       ignore_index=-100, reduction="none").reshape_as(y)
        counts = (y != -100).sum(dim=1)
        if (counts == 0).any():
            raise ValueError("Every example must contain at least one answer target")
        loss = (token_losses.sum(dim=1) / counts).mean()
    if intents is not None and intent_weight and (intents != -100).any():
        # Classify the question at its final prompt position, before any answer token.
        position = (y != -100).to(torch.int32).argmax(dim=1)
        question = hidden[torch.arange(len(x),device=x.device), position]
        loss = loss + intent_weight * F.cross_entropy(model.intent_head(question),intents,ignore_index=-100)
    return loss


@torch.inference_mode()
def validation_loss(model, examples, pad, device, batch_size, intents=None, intent_weight=0.0, loss_normalization="token"):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        x, y = batch(examples[start:start + batch_size], pad, device)
        n = len(x) if loss_normalization == "example" else int((y != -100).sum().item())
        labels=None if intents is None else torch.tensor(intents[start:start+batch_size],device=device)
        total += loss_for(model, x, y, labels, intent_weight, loss_normalization).item() * n
        count += n
    model.train()
    return total / count


def train(data_dir, output, steps=300, batch_size=24, lr=0.0005, device="auto", seed=42, init_run=None,
          eval_every=100, validation_limit=256, intent_weight=0.0, context_length=None, loss_normalization="token"):
    if loss_normalization not in {"token", "example"}:
        raise ValueError("loss_normalization must be token or example")
    if steps < 1 or batch_size < 1 or lr <= 0 or eval_every < 1 or validation_limit < 0 or intent_weight < 0:
        raise ValueError("steps, batch-size, lr은 양수여야 합니다.")
    if context_length is not None and context_length < 1:
        raise ValueError("문맥 길이는 양수여야 합니다.")
    root, data = Path(output), Path(data_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("출력 폴더에 기존 파일이 있습니다. 새 경로를 사용하세요.")
    training = read_rows(data / "train.jsonl")
    valid = read_rows(data / "valid.jsonl")
    def example_key(r):
        return json.dumps([r.get("history", []), r["prompt"]], ensure_ascii=False, sort_keys=True)
    if {example_key(r) for r in training} & {example_key(r) for r in valid}:
        raise ValueError("train과 valid에 중복 질문이 있습니다.")
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = device_for(device)
    if init_run:
        from .runtime import load_run
        import shutil
        model, tok, _ = load_run(init_run, device)
        cfg = model.cfg
        shutil.copy2(Path(init_run) / "tokenizer.json", root / "tokenizer.json")
    else:
        tok = train_tokenizer(training, root / "tokenizer.json")
        cfg = Config(vocab_size=tok.get_vocab_size())
        model = Trid(cfg).to(device)
    inherited_context = cfg.context_length if init_run else None
    if context_length is not None:
        # RoPE has no learned position table; keep every weight and token ID.
        # This changes the permitted training length, not demonstrated capability.
        cfg = replace(cfg, context_length=context_length)
        model.cfg = cfg
    intent_map={}
    if intent_weight:
        topics=sorted({r.get('task','public_daily') for r in training}-{'public_daily'})
        intent_map={name:i for i,name in enumerate(topics)}
        cfg=replace(cfg,num_intents=len(intent_map))
        previous={k:v for k,v in model.state_dict().items() if not k.startswith('intent_head.')}
        updated=Trid(cfg).to(device)
        missing,unexpected=updated.load_state_dict(previous,strict=False)
        if unexpected or set(missing)!={'intent_head.weight'}:
            raise ValueError('Unexpected weights when attaching training-only intent head')
        model=updated
    (root / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    train_examples = [encode_example(tok, r, cfg.context_length) for r in training]
    valid_used=valid
    if validation_limit and len(valid_used)>validation_limit:
        valid_used=random.Random(seed+1).sample(valid_used,validation_limit)
    valid_examples = [encode_example(tok, r, cfg.context_length) for r in valid_used]
    train_intents=[intent_map.get(r.get('task'),-100) for r in training] if intent_map else None
    valid_intents=[intent_map.get(r.get('task'),-100) for r in valid_used] if intent_map else None
    weights = [float(r.get("weight", 1)) for r in training]
    if not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError("학습 가중치는 유한한 양수여야 합니다.")
    pad = tok.token_to_id("<|pad|>")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.1, betas=(.9, .95))
    metadata = {"version": "0.1.1", "stage": "sft", "initialization": f"continued from local run: {init_run}" if init_run else "random; no pretrained weights",
                "tokenizer": "inherited unchanged from local init run" if init_run else "byte-level BPE trained on train.jsonl only",
                "objective": "supervised next-token prediction, answer-only loss",
                "device": device, "parameters": model.parameter_count(), "config": asdict(cfg),
                "inherited_context_length": inherited_context,
                "seed": seed, "steps_requested": steps, "batch_size": batch_size, "learning_rate": lr,
                "data_sha256": {s: hashlib.sha256((data / f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "valid")},
                "train_rows": len(training), "valid_rows": len(valid),
                "validation_rows_used": len(valid_examples), "eval_every": eval_every,
                "intent_auxiliary_weight": intent_weight,"intent_map":intent_map,
                "loss_normalization": loss_normalization,
                "validation_metric": f"{loss_normalization}-mean answer CE" + (" plus question-intent auxiliary CE" if intent_weight else ""),
                "status": "training", "pretrained_model": str(Path(init_run).resolve()) if init_run else None}
    if init_run:
        from .corpus import sha256
        metadata["initial_weights_sha256"] = sha256(Path(init_run) / "model.safetensors")
    if (data / "provenance.json").exists():
        metadata["data_provenance"] = json.loads((data / "provenance.json").read_text())
    (root / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"trid: {metadata['parameters']:,} parameters | {device} | {metadata['initialization']}", flush=True)
    sync(device)
    started, best, best_step = time.perf_counter(), float("inf"), 0
    initial = validation_loss(model, valid_examples, pad, device, batch_size, valid_intents,intent_weight,loss_normalization)
    # A worsening fine-tune must not replace a better initial checkpoint.
    best = initial
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(root / "model.safetensors"))
    print(f"step 0 | validation loss {initial:.4f}", flush=True)
    metrics = [{"step": 0, "valid_loss": initial}]
    for step in range(1, steps + 1):
        warmup = min(20, max(1, steps // 10))
        ratio = step / warmup if step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = lr * ratio
        indices=rng.choices(range(len(train_examples)),weights=weights,k=batch_size)
        selected=[train_examples[i] for i in indices]
        x, y = batch(selected, pad, device)
        optimizer.zero_grad(set_to_none=True)
        labels=None if train_intents is None else torch.tensor([train_intents[i] for i in indices],device=device)
        loss = loss_for(model, x, y, labels,intent_weight,loss_normalization)
        if not torch.isfinite(loss):
            raise RuntimeError("손실이 유한하지 않습니다. 학습을 중단합니다.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            score = validation_loss(model, valid_examples, pad, device, batch_size,valid_intents,intent_weight,loss_normalization)
            sync(device)
            entry = {"step": step, "train_loss": loss.item(), "valid_loss": score,
                     "elapsed_seconds": round(time.perf_counter() - started, 2)}
            metrics.append(entry)
            print(f"step {step}/{steps} | train {loss.item():.4f} | valid {score:.4f} | {entry['elapsed_seconds']:.1f}s", flush=True)
            if score < best:
                best, best_step = score, step
                tmp=root/'best.tmp.safetensors'
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},str(tmp))
                tmp.replace(root/'model.safetensors')
            last_tmp=root/'last.tmp.safetensors'
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(last_tmp))
            last_tmp.replace(root/'last-model.safetensors')
            (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sync(device)
    metadata.update(status="complete", best_validation_loss=best, initial_validation_loss=initial,
                    best_step=best_step, completed_steps=steps,
                    training_seconds=round(time.perf_counter() - started, 2),
                    weight_bytes=(root / "model.safetensors").stat().st_size,
                    note="Checkpoint selected by validation loss. Optimizer state is not saved; rerun training into a new directory.")
    (root / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved: {root / 'model.safetensors'}", flush=True)

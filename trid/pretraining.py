"""Continuous text pretraining, safe optimizer checkpoints, and exact-step resume."""
import contextlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .corpus import sha256, write_json
from .model import Config, Trid
from .runtime import device_for, sync
from .training import loss_for

PRECISIONS, SCHEDULES = ("fp32", "bf16"), ("cosine", "wsd")


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def lr_ratio(step, steps, schedule="cosine", decay_fraction=.1):
    """Fraction of peak lr at 1-based `step`. cosine: warmup → cosine to 10%.
    wsd: warmup → stable → linear decay to 0 over the last `decay_fraction` of steps
    (MiniCPM, arXiv 2404.06395 §4.2: a 10% decay stage is enough; SmolLM2 uses 20%)."""
    warmup = min(100, max(1, steps // 10))
    if step <= warmup:
        return step / warmup
    if schedule == "cosine":
        return .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    decay_steps = max(1, round(decay_fraction * steps))
    return min(1., (steps - step) / decay_steps)


def train_loss(model, x, y, z_loss=0.):
    if not z_loss:
        return loss_for(model, x, y)
    # PaLM (arXiv 2204.02311 §5): z_loss * log²Z keeps the softmax normalizer near 0.
    logits, _ = model(x)
    flat = logits.reshape(-1, logits.size(-1))
    return F.cross_entropy(flat, y.reshape(-1)) + z_loss * torch.logsumexp(flat.float(), -1).square().mean()


def sample_batch(tokens, length, size, generator, device):
    if len(tokens) <= length:
        raise ValueError("문맥 길이보다 많은 토큰이 필요합니다.")
    starts = torch.randint(0, len(tokens) - length, (size,), generator=generator).tolist()
    values = np.stack([tokens[i:i + length + 1] for i in starts]).astype(np.int64)
    values = torch.from_numpy(values).to(device)
    return values[:, :-1], values[:, 1:]


def save_checkpoint(root, model, optimizer, generator, state):
    name = f"step-{state['step']:08d}"
    folder = root / "checkpoints" / name
    pending = folder.with_name(name + ".tmp")
    pending.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(pending / "model.safetensors"))
    tensors = {"batch_rng": generator.get_state()}
    for i, p in enumerate(model.parameters()):
        for key, value in optimizer.state[p].items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("Optimizer state must contain tensors only.")
            tensors[f"p{i}.{key}"] = value.detach().cpu().contiguous()
    save_file(tensors, str(pending / "optimizer.safetensors"))
    write_json(pending / "state.json", state)
    pending.rename(folder)
    atomic_json(root / "latest.json", {"checkpoint": name})
    # Keep two recoverable training snapshots; best inference weights are separate.
    checkpoints = sorted(p for p in (root / "checkpoints").glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
    for old in checkpoints[:-2]:
        shutil.rmtree(old)


def restore_checkpoint(root, model, optimizer, generator, device):
    name = json.loads((root / "latest.json").read_text())["checkpoint"]
    if not isinstance(name, str) or not name.startswith("step-") or not name[5:].isdigit():
        raise ValueError("잘못된 체크포인트 경로입니다.")
    folder = root / "checkpoints" / name
    model.load_state_dict(load_file(str(folder / "model.safetensors")))
    tensors = load_file(str(folder / "optimizer.safetensors"))
    generator.set_state(tensors.pop("batch_rng"))
    for i, p in enumerate(model.parameters()):
        for key in ["step", "exp_avg", "exp_avg_sq"]:
            value = tensors.get(f"p{i}.{key}")
            if value is not None:
                optimizer.state[p][key] = value if key == "step" else value.to(device)
    return json.loads((folder / "state.json").read_text())


@torch.inference_mode()
def validate(model, tokens, batch_size, device, batches=8):
    model.eval()
    generator = torch.Generator().manual_seed(20261026)
    values = []
    for _ in range(batches):
        x, y = sample_batch(tokens, model.cfg.context_length, batch_size, generator, device)
        values.append(loss_for(model, x, y).item())
    model.train()
    return sum(values) / len(values)


def pretrain(data, output, config_path=None, steps=2000, stop_after=200,
             batch_size=8, lr=.0003, device="auto", seed=42, resume=False,
             eval_every=50, validation_batches=8, init_run=None, accumulate=1,
             precision="fp32", compile=False, schedule="cosine", decay_fraction=.1, z_loss=0.):
    if resume and init_run:
        raise ValueError("--resume과 --init-run을 함께 사용할 수 없습니다.")
    if min(steps, stop_after, batch_size, eval_every, validation_batches, accumulate) < 1 or lr <= 0 or z_loss < 0:
        raise ValueError("학습 설정은 양수여야 합니다.")
    if precision not in PRECISIONS or schedule not in SCHEDULES or not 0 < decay_fraction <= 1:
        raise ValueError("precision은 fp32|bf16, schedule은 cosine|wsd, decay_fraction은 (0, 1]이어야 합니다.")
    root, data = Path(output), Path(data)
    device = device_for(device)
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["format"] != "uint16-little-endian":
        raise ValueError("지원하지 않는 토큰 데이터 형식입니다.")
    for split in ["train", "valid"]:
        if sha256(data / f"{split}.bin") != manifest["sha256"][split]:
            raise ValueError(f"{split} 토큰 파일이 변경되었습니다.")
    if sha256(data / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("토크나이저가 변경되었습니다.")
    train_tokens = np.memmap(data / "train.bin", dtype="<u2", mode="r")
    valid_tokens = np.memmap(data / "valid.bin", dtype="<u2", mode="r")
    fingerprint = sha256(data / "manifest.json")
    if resume:
        metadata = json.loads((root / "training.json").read_text())
        if metadata.get("stage") != "pretrain":
            raise ValueError("사전학습 체크포인트가 아닙니다.")
        if metadata["data_manifest_sha256"] != fingerprint:
            raise ValueError("재개 시 데이터나 토크나이저를 변경할 수 없습니다.")
        if sha256(root / "tokenizer.json") != manifest["tokenizer_sha256"]:
            raise ValueError("저장된 실행의 토크나이저가 변경되었습니다.")
        cfg = Config(**json.loads((root / "config.json").read_text()))
        if config_path:
            requested = json.loads(Path(config_path).read_text())
            requested["vocab_size"] = manifest["vocab_size"]
            if asdict(Config(**requested)) != asdict(cfg):
                raise ValueError("재개 시 모델 구조를 변경할 수 없습니다.")
        # Resume the original schedule and optimizer hyperparameters; precision/compile are per-session.
        steps, batch_size, lr, seed = (metadata[k] for k in ["planned_steps", "batch_size", "learning_rate", "seed"])
        accumulate, schedule = metadata.get("accumulate", 1), metadata.get("schedule", "cosine")
        decay_fraction, z_loss = metadata.get("decay_fraction", .1), metadata.get("z_loss", 0.)
        eval_every, validation_batches = metadata["eval_every"], metadata["validation_batches"]
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("새 출력 경로를 사용하거나 --resume으로 이어서 학습하세요.")
        parent_metadata = None
        if init_run:
            parent = Path(init_run)
            parent_metadata = json.loads((parent / "training.json").read_text())
            if parent_metadata.get("stage") != "pretrain":
                raise ValueError("--init-run에는 자체 사전학습 실행을 지정하세요.")
            if parent_metadata.get("status") not in {"complete", "checkpointed", "interrupted"}:
                raise ValueError("원본 실행의 학습을 먼저 종료하고 저장된 가중치에서 이어 학습하세요.")
            if sha256(parent / "tokenizer.json") != manifest["tokenizer_sha256"]:
                raise ValueError("이어 학습할 모델과 데이터의 토크나이저가 다릅니다.")
            cfg = Config(**json.loads((parent / "config.json").read_text()))
            if cfg.vocab_size != manifest["vocab_size"]:
                raise ValueError("모델과 데이터의 어휘 수가 다릅니다.")
            if config_path and asdict(Config(**json.loads(Path(config_path).read_text()))) != asdict(cfg):
                raise ValueError("이어 학습에서는 모델 구조를 변경할 수 없습니다.")
        else:
            if not config_path:
                raise ValueError("새 학습에는 --config가 필요합니다.")
            values = json.loads(Path(config_path).read_text())
            values["vocab_size"] = manifest["vocab_size"]
            cfg = Config(**values)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "config.json", asdict(cfg))
        shutil.copy2(data / "tokenizer.json", root / "tokenizer.json")
        shutil.copy2(data / "manifest.json", root / "data-manifest.json")
        metadata = {"version": "0.1.1", "stage": "pretrain", "initialization": "random; no pretrained weights",
                    "objective": "continuous next-token prediction, all tokens", "planned_steps": steps,
                    "batch_size": batch_size, "accumulate": accumulate, "learning_rate": lr, "seed": seed,
                    "schedule": schedule, "decay_fraction": decay_fraction, "z_loss": z_loss,
                    "eval_every": eval_every, "validation_batches": validation_batches,
                    "data_manifest_sha256": fingerprint, "device": device,
                    "status": "training", "note": "Base text-completion model, not dialogue-trained. Test split not used for training or selection."}
        if parent_metadata is not None:
            # The exported best weights may precede the parent's final update.
            inherited_tokens = (parent_metadata.get("inherited_tokens_seen", 0) + parent_metadata["best_step"]
                                * parent_metadata["batch_size"] * parent_metadata.get("accumulate", 1) * cfg.context_length)
            metadata.update(initialization="continued from local pretraining best weights; fresh optimizer and schedule",
                            init_run=str(parent.resolve()), initial_weights_sha256=sha256(parent / "model.safetensors"),
                            parent_training_sha256=sha256(parent / "training.json"),
                            parent_best_step=parent_metadata["best_step"], inherited_tokens_seen=inherited_tokens)
    if min(len(train_tokens), len(valid_tokens)) <= cfg.context_length:
        raise ValueError("학습/검증 토큰이 문맥 길이보다 짧습니다.")
    if max(int(train_tokens.max()), int(valid_tokens.max())) >= cfg.vocab_size:
        raise ValueError("토큰 ID가 모델 어휘 범위를 초과합니다.")
    step_tokens = batch_size * accumulate * cfg.context_length
    torch.manual_seed(seed)
    model = Trid(cfg).to(device).train()
    if init_run:
        model.load_state_dict(load_file(str(Path(init_run) / "model.safetensors")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.1)
    generator = torch.Generator().manual_seed(seed)
    metrics_path = root / "metrics.json"
    if resume:
        state = restore_checkpoint(root, model, optimizer, generator, device)
        metrics = [m for m in json.loads(metrics_path.read_text()) if m["step"] <= state["step"]]
    else:
        baseline = validate(model, valid_tokens, batch_size, device, validation_batches)
        state = {"step": 0, "best_loss": baseline, "best_step": 0, "training_seconds": 0.0}
        metrics = [{"step": 0, "valid_loss": baseline, "perplexity": math.exp(min(30, baseline))}]
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(root / "model.safetensors"))
        save_checkpoint(root, model, optimizer, generator, state)
        atomic_json(metrics_path, metrics)
    if state["step"] >= steps:
        print("계획한 학습 단계를 이미 완료했습니다.", flush=True)
        return
    metadata.update(status="training", parameters=model.parameter_count(), device=device,
                    precision=precision, compile=bool(compile))
    atomic_json(root / "training.json", metadata)
    print(f"pretrain: {model.parameter_count():,} parameters | {device} | {precision}{' compile' if compile else ''} | "
          f"{state['step']}/{steps} steps | {step_tokens:,} tokens/step", flush=True)
    # Checkpoints and validation always use the eager module; compile wraps it only for the training forward.
    trainer = torch.compile(model) if compile else model
    autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16) if precision == "bf16"
                else contextlib.nullcontext())
    stop = min(steps, state["step"] + stop_after)
    sync(device)
    started = time.perf_counter()
    session_start_step = state["step"]
    prior_seconds = state["training_seconds"]
    step, interrupted = state["step"], False
    try:
        for step in range(state["step"] + 1, stop + 1):
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_ratio(step, steps, schedule, decay_fraction)
            # One draw per optimizer step keeps the batch RNG stream identical across accumulate settings.
            x, y = sample_batch(train_tokens, cfg.context_length, batch_size * accumulate, generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss = 0.
            for xi, yi in zip(x.chunk(accumulate), y.chunk(accumulate)):
                with autocast:
                    micro = train_loss(trainer, xi, yi, z_loss) / accumulate
                if not torch.isfinite(micro):
                    raise RuntimeError("학습 손실이 유한하지 않습니다.")
                micro.backward()
                loss = loss + micro.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step % eval_every == 0 or step == stop:
                score = validate(model, valid_tokens, batch_size, device, validation_batches)
                sync(device)
                elapsed = time.perf_counter() - started
                state.update(step=step, training_seconds=prior_seconds + elapsed)
                if score < state["best_loss"]:
                    state.update(best_loss=score, best_step=step)
                    temp = root / "best.tmp.safetensors"
                    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp))
                    temp.replace(root / "model.safetensors")
                entry = {"step": step, "train_loss": loss.item(), "valid_loss": score,
                         "perplexity": math.exp(min(30, score)),
                         "tokens_seen": step * step_tokens,
                         "lineage_tokens_seen": metadata.get("inherited_tokens_seen", 0) + step * step_tokens,
                         "training_seconds": round(state["training_seconds"], 3),
                         "session_tokens_per_second": round((step - session_start_step) * step_tokens / elapsed, 1)}
                metrics.append(entry)
                atomic_json(metrics_path, metrics)
                save_checkpoint(root, model, optimizer, generator, state)
                print(f"step {step}/{steps} | train {loss.item():.4f} | valid {score:.4f} | {entry['session_tokens_per_second']:,.0f} tokens/s", flush=True)
    except KeyboardInterrupt:
        # Do not serialize a partly-applied update; continue from last complete checkpoint.
        interrupted = True
        print("중단했습니다. 마지막 완전한 체크포인트부터 --resume으로 재개할 수 있습니다.", flush=True)
    latest_name = json.loads((root / "latest.json").read_text())["checkpoint"]
    state = json.loads((root / "checkpoints" / latest_name / "state.json").read_text())
    metadata.update(status="complete" if state["step"] >= steps else "interrupted" if interrupted else "checkpointed",
                    completed_steps=state["step"], best_step=state["best_step"], best_validation_loss=state["best_loss"],
                    training_seconds=round(state["training_seconds"], 3),
                    tokens_seen=state["step"] * step_tokens,
                    lineage_tokens_seen=metadata.get("inherited_tokens_seen", 0) + state["step"] * step_tokens,
                    weight_bytes=(root / "model.safetensors").stat().st_size)
    atomic_json(root / "training.json", metadata)
    print(f"체크포인트 저장: {state['step']}/{steps}. 백그라운드 학습은 실행되지 않습니다.", flush=True)

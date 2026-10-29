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

PRECISIONS, SCHEDULES, SAMPLINGS = ("fp32", "bf16"), ("cosine", "wsd"), ("replacement", "epoch")


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
    """Dense token-mean CE over every position. Pretraining supervises all tokens, so the
    sparse supervised-position path in trid.training would only copy the hidden states."""
    logits, _ = model(x)
    flat = logits.reshape(-1, logits.size(-1))
    loss = F.cross_entropy(flat, y.reshape(-1))
    if z_loss:
        # PaLM (arXiv 2204.02311 §5): z_loss * log²Z keeps the softmax normalizer near 0.
        loss = loss + z_loss * torch.logsumexp(flat.float(), -1).square().mean()
    return loss


def window_batch(tokens, length, starts, device):
    values = np.stack([tokens[i:i + length + 1] for i in starts]).astype(np.int64)
    values = torch.from_numpy(values).to(device)
    return values[:, :-1], values[:, 1:]


def sample_batch(tokens, length, size, generator, device):
    """Windows drawn i.i.d. with replacement (sampling='replacement'; also validation)."""
    if len(tokens) <= length:
        raise ValueError("문맥 길이보다 많은 토큰이 필요합니다.")
    starts = torch.randint(0, len(tokens) - length, (size,), generator=generator).tolist()
    return window_batch(tokens, length, starts, device)


def window_kinds(tokens, length, sources):
    """Kind index of each non-overlapping window, by the document covering its first token.
    `sources` is one split of data/sources.json (trid.corpus.write_sources)."""
    offsets, kinds = np.asarray(sources["offsets"], dtype=np.int64), np.asarray(sources["kind"], dtype=np.int64)
    if len(offsets) != len(kinds) or len(offsets) == 0 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0) or offsets[-1] >= len(tokens):
        raise ValueError("sources.json의 문서 경계가 토큰 파일과 맞지 않습니다.")
    starts = np.arange((len(tokens) - 1) // length, dtype=np.int64) * length
    return kinds[np.searchsorted(offsets, starts, side="right") - 1]


class EpochSampler:
    """sampling='epoch': every non-overlapping context window of train.bin once per epoch, in a
    permutation seeded by seed + epoch. `weights` (per kind index, from `source_weights`) repeat
    windows of a kind floor(w) times plus one more with probability frac(w), so w=2 doubles a kind
    and w=.5 keeps a seeded half of it. Position = (epoch, cursor into that epoch's order)."""

    def __init__(self, tokens, length, seed, kinds=None, weights=None):
        if len(tokens) <= length:
            raise ValueError("문맥 길이보다 많은 토큰이 필요합니다.")
        self.tokens, self.length, self.seed = tokens, length, seed
        self.count = (len(tokens) - 1) // length
        self.kinds = None if weights is None else np.asarray(kinds, dtype=np.int64)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.epoch, self.cursor, self._order = 0, 0, None

    def order(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        if self.weights is None:
            return rng.permutation(self.count)
        w = self.weights[self.kinds]
        repeats = np.floor(w).astype(np.int64) + (rng.random(self.count) < (w - np.floor(w)))
        ids = np.repeat(np.arange(self.count), repeats)
        if len(ids) == 0:
            raise ValueError("source_weights로 남는 학습 구간이 없습니다.")
        return ids[rng.permutation(len(ids))]

    def position(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def seek(self, position):
        self.epoch, self.cursor, self._order = int(position["epoch"]), int(position["cursor"]), None

    def draw(self, size, device):
        starts = []
        while len(starts) < size:
            if self._order is None:
                self._order = self.order(self.epoch)
            take = self._order[self.cursor:self.cursor + size - len(starts)]
            starts.extend((take * self.length).tolist())
            self.cursor += len(take)
            if self.cursor >= len(self._order):
                self.epoch, self.cursor, self._order = self.epoch + 1, 0, None
        return window_batch(self.tokens, self.length, starts, device)


def param_groups(model, decay_norms):
    """decay_norms=False: weight decay only on matrices (embedding included); RMSNorm gains and
    other 1-D parameters are left undecayed. True reproduces the single decayed group."""
    if decay_norms:
        return [{"params": list(model.parameters()), "weight_decay": .1}]
    params = list(model.parameters())
    return [{"params": [p for p in params if p.ndim > 1], "weight_decay": .1},
            {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.}]


def save_weights(model, path):
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))


def save_checkpoint(root, model, optimizer, generator, state):
    name = f"step-{state['step']:08d}"
    folder = root / "checkpoints" / name
    pending = folder.with_name(name + ".tmp")
    pending.mkdir(parents=True, exist_ok=False)
    save_weights(model, pending / "model.safetensors")
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
def validate(model, tokens, batch_size, device, batches=8, autocast=None):
    model.eval()
    generator = torch.Generator().manual_seed(20261026)
    values = []
    for _ in range(batches):
        x, y = sample_batch(tokens, model.cfg.context_length, batch_size, generator, device)
        with autocast or contextlib.nullcontext():
            values.append(train_loss(model, x, y).item())
    model.train()
    return sum(values) / len(values)


def pretrain(data, output, config_path=None, steps=2000, stop_after=200,
             batch_size=8, lr=.0003, device="auto", seed=42, resume=False,
             eval_every=50, validation_batches=8, init_run=None, accumulate=1,
             precision="fp32", compile=False, schedule="cosine", decay_fraction=.1, z_loss=0.,
             checkpoint_every=None, sampling="replacement", source_weights=None, decay_norms=False):
    """New-run options are recorded in training.json and restored on resume; only stop_after,
    device, precision and compile are per-session. Runs created before checkpoint_every /
    sampling / source_weights / decay_norms existed resume with the old behaviour
    (checkpoint at every eval, sampling with replacement, every parameter decayed)."""
    if resume and init_run:
        raise ValueError("--resume과 --init-run을 함께 사용할 수 없습니다.")
    if checkpoint_every is None:
        checkpoint_every = eval_every
    if min(steps, stop_after, batch_size, eval_every, checkpoint_every, validation_batches, accumulate) < 1 or lr <= 0 or z_loss < 0:
        raise ValueError("학습 설정은 양수여야 합니다.")
    if checkpoint_every % eval_every:
        raise ValueError("checkpoint_every는 eval_every의 배수여야 합니다.")
    if precision not in PRECISIONS or schedule not in SCHEDULES or not 0 < decay_fraction <= 1:
        raise ValueError("precision은 fp32|bf16, schedule은 cosine|wsd, decay_fraction은 (0, 1]이어야 합니다.")
    if sampling not in SAMPLINGS:
        raise ValueError("sampling은 replacement|epoch이어야 합니다.")
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
        # Keys absent from runs created before these options existed: the old behaviour.
        checkpoint_every, sampling = metadata.get("checkpoint_every", eval_every), metadata.get("sampling", "replacement")
        source_weights, decay_norms = metadata.get("source_weights"), metadata.get("decay_norms", True)
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
    sampler = kinds = weights = None
    if source_weights is not None:
        if sampling != "epoch":
            raise ValueError("source_weights는 sampling='epoch'에서만 쓸 수 있습니다.")
        if not (data / "sources.json").exists():
            raise ValueError("source_weights에는 data/sources.json이 필요합니다 (trid.corpus.write_sources).")
        sources = json.loads((data / "sources.json").read_text())
        names = sources["kinds"]
        if not isinstance(source_weights, dict) or not source_weights:
            raise ValueError("source_weights는 {kind: weight} 사전이어야 합니다.")
        unknown = sorted(set(source_weights) - set(names))
        if unknown or any(not isinstance(w, (int, float)) or w < 0 for w in source_weights.values()):
            raise ValueError(f"source_weights의 kind는 {names} 중 하나이고 가중치는 0 이상이어야 합니다. 알 수 없음: {unknown}")
        source_weights = {k: float(v) for k, v in source_weights.items()}
        weights = [source_weights.get(k, 1.) for k in names]
        kinds = window_kinds(train_tokens, cfg.context_length, sources["splits"]["train"])
    if sampling == "epoch":
        sampler = EpochSampler(train_tokens, cfg.context_length, seed, kinds, weights)
    step_tokens = batch_size * accumulate * cfg.context_length
    torch.manual_seed(seed)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = Trid(cfg).to(device).train()
    if init_run:
        model.load_state_dict(load_file(str(Path(init_run) / "model.safetensors")))
    optimizer = torch.optim.AdamW(param_groups(model, decay_norms), lr=lr, betas=(.9, .95), fused=device == "cuda")
    generator = torch.Generator().manual_seed(seed)
    autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16) if precision == "bf16"
                else contextlib.nullcontext())
    metrics_path = root / "metrics.json"
    if resume:
        state = restore_checkpoint(root, model, optimizer, generator, device)
        metrics = [m for m in json.loads(metrics_path.read_text()) if m["step"] <= state["step"]]
        if sampler is not None:
            if "sampler" not in state:
                raise ValueError("epoch 샘플링 실행인데 체크포인트에 sampler 위치가 없습니다.")
            sampler.seek(state["sampler"])
    else:
        baseline = validate(model, valid_tokens, batch_size, device, validation_batches, autocast)
        state = {"step": 0, "best_loss": baseline, "best_step": 0, "training_seconds": 0.0}
        if sampler is not None:
            state["sampler"] = sampler.position()
        metrics = [{"step": 0, "valid_loss": baseline, "perplexity": math.exp(min(30, baseline))}]
        save_weights(model, root / "model.safetensors")
        save_checkpoint(root, model, optimizer, generator, state)
        atomic_json(metrics_path, metrics)
    if state["step"] >= steps:
        print("계획한 학습 단계를 이미 완료했습니다.", flush=True)
        return
    metadata.update(status="training", parameters=model.parameter_count(), device=device,
                    precision=precision, compile=bool(compile), checkpoint_every=checkpoint_every,
                    sampling=sampling, source_weights=source_weights, decay_norms=decay_norms)
    if sampler is not None:
        metadata["epoch_windows"] = int(len(sampler.order(0)))
    atomic_json(root / "training.json", metadata)
    print(f"pretrain: {model.parameter_count():,} parameters | {device} | {precision}{' compile' if compile else ''} | "
          f"{state['step']}/{steps} steps | {step_tokens:,} tokens/step | {sampling}", flush=True)
    # Checkpoints and validation always use the eager module; compile wraps it only for the training forward.
    trainer = torch.compile(model) if compile else model
    stop = min(steps, state["step"] + stop_after)
    sync(device)
    started = time.perf_counter()
    session_start_step = state["step"]
    prior_seconds = state["training_seconds"]
    step, interrupted, grad_peak = state["step"], False, None
    try:
        for step in range(state["step"] + 1, stop + 1):
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_ratio(step, steps, schedule, decay_fraction)
            # One draw per optimizer step keeps the batch RNG stream identical across accumulate settings.
            if sampler is None:
                x, y = sample_batch(train_tokens, cfg.context_length, batch_size * accumulate, generator, device)
            else:
                x, y = sampler.draw(batch_size * accumulate, device)
            optimizer.zero_grad(set_to_none=True)
            loss = 0.
            for xi, yi in zip(x.chunk(accumulate), y.chunk(accumulate)):
                with autocast:
                    micro = train_loss(trainer, xi, yi, z_loss) / accumulate
                micro.backward()
                loss = loss + micro.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # One host sync per step: a non-finite loss or gradient norm raises before the update is applied.
            if not torch.isfinite(torch.stack([loss.float(), grad_norm.float()])).all():
                raise RuntimeError(f"학습 손실 또는 기울기 노름이 유한하지 않습니다: loss={loss.item()}, grad_norm={grad_norm.item()}")
            grad_peak = grad_norm.detach() if grad_peak is None else torch.maximum(grad_peak, grad_norm.detach())
            optimizer.step()
            evaluating = step % eval_every == 0 or step == stop
            checkpointing = step % checkpoint_every == 0 or step == stop
            if not evaluating:
                continue
            score = validate(model, valid_tokens, batch_size, device, validation_batches, autocast)
            sync(device)
            elapsed = time.perf_counter() - started
            state.update(step=step, training_seconds=prior_seconds + elapsed)
            # Best weights are written only alongside a checkpoint (or at session end), so model.safetensors
            # always corresponds to a recorded best_step and eval-only steps cost no weight I/O.
            if score < state["best_loss"] and checkpointing:
                state.update(best_loss=score, best_step=step)
                temp = root / "best.tmp.safetensors"
                save_weights(model, temp)
                temp.replace(root / "model.safetensors")
            entry = {"step": step, "train_loss": loss.item(), "valid_loss": score,
                     "perplexity": math.exp(min(30, score)),
                     "tokens_seen": step * step_tokens,
                     "lineage_tokens_seen": metadata.get("inherited_tokens_seen", 0) + step * step_tokens,
                     "training_seconds": round(state["training_seconds"], 3),
                     "session_tokens_per_second": round((step - session_start_step) * step_tokens / elapsed, 1),
                     "grad_norm": round(grad_norm.item(), 4), "grad_norm_max_since_eval": round(grad_peak.item(), 4)}
            grad_peak = None
            if str(device).startswith("cuda"):
                entry["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if sampler is not None:
                entry["epoch"] = sampler.epoch
            metrics.append(entry)
            atomic_json(metrics_path, metrics)
            if checkpointing:
                if sampler is not None:
                    state["sampler"] = sampler.position()
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

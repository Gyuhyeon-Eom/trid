import contextlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from trid.corpus import sha256, write_json, prepare, split_for
from trid.pretraining import EpochSampler, lr_ratio, param_groups, pretrain, sample_batch, train_loss, validate, window_kinds
from trid.text import train_tokenizer
from trid.model import Config, Trid
from trid.training import train
from dataclasses import asdict

class PretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_data(self, root):
        data = root / "data"
        data.mkdir()
        tok = train_tokenizer([{"prompt": "한국어 문장 학습", "answer": "업무 기록을 정리합니다."}], data / "tokenizer.json", 300)
        for split in ["train", "valid", "test"]:
            np.asarray(list(range(5, 30)) * 8, dtype="<u2").tofile(data / f"{split}.bin")
        write_json(data / "manifest.json", {"format": "uint16-little-endian", "vocab_size": tok.get_vocab_size(),
                    "tokenizer_sha256": sha256(data / "tokenizer.json"),
                    "sha256": {s: sha256(data / f"{s}.bin") for s in ["train", "valid", "test"]}})
        config = root / "config.json"
        write_json(config, {"vocab_size": 300, "dim": 32, "layers": 2, "heads": 4,
                            "hidden_dim": 64, "context_length": 8})
        return data, config

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=4, batch_size=2, device="cpu", eval_every=2, validation_batches=1, accumulate=2)
            pretrain(data, root / "full", stop_after=4, **options)
            pretrain(data, root / "resumed", stop_after=2, **options)
            # Resume restores accumulate from training.json even if the caller omits it.
            pretrain(data, root / "resumed", stop_after=2, resume=True, **{**options, "accumulate": 1})
            def last(run):
                name = json.loads((run / "latest.json").read_text())["checkpoint"]
                return load_file(str(run / "checkpoints" / name / "model.safetensors"))
            a, b = last(root / "full"), last(root / "resumed")
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            metadata = json.loads((root / "resumed" / "training.json").read_text())
            self.assertEqual(metadata["completed_steps"], 4)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual((metadata["accumulate"], metadata["tokens_seen"]), (2, 4 * 2 * 2 * 8))
            self.assertEqual(json.loads((root / "resumed" / "metrics.json").read_text())[-1]["tokens_seen"], 128)
            self.assertEqual(len(list((root / "resumed" / "checkpoints").glob("step-*"))), 2)
            with (data / "train.bin").open("ab") as f:
                f.write(b"\x01\x00")
            with self.assertRaisesRegex(ValueError, "변경"):
                pretrain(data, root / "resumed", resume=True, **options)

    def test_epoch_sampler_covers_every_window_once_and_resumes_at_cursor(self):
        tokens = np.arange(41, dtype="<u2")  # 5 non-overlapping windows of 8 (+1 target token)
        sampler = EpochSampler(tokens, 8, seed=7)
        self.assertEqual(sampler.count, 5)
        starts = []
        for _ in range(3):
            x, y = sampler.draw(2, "cpu")
            torch.testing.assert_close(y, x + 1)
            starts.extend(x[:, 0].tolist())
        self.assertEqual(sorted(starts[:5]), [0, 8, 16, 24, 32])
        self.assertEqual(sampler.position(), {"epoch": 1, "cursor": 1})
        self.assertNotEqual(sampler.order(0).tolist(), sampler.order(1).tolist())
        self.assertEqual(sampler.order(1).tolist(), EpochSampler(tokens, 8, seed=7).order(1).tolist())
        position = sampler.position()
        expected = [sampler.draw(3, "cpu")[0][:, 0].tolist() for _ in range(2)]
        other = EpochSampler(tokens, 8, seed=7)
        other.seek(position)
        self.assertEqual([other.draw(3, "cpu")[0][:, 0].tolist() for _ in range(2)], expected)
        self.assertEqual(sorted(sum(expected, [])[:4] + starts[5:]), [0, 8, 16, 24, 32])  # rest of epoch 1 before epoch 2 starts
        # Documents at 0 and 20: windows 0-2 start in kind 0, windows 3-4 in kind 1.
        kinds = window_kinds(tokens, 8, {"offsets": [0, 20], "kind": [0, 1]})
        self.assertEqual(kinds.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(sorted(EpochSampler(tokens, 8, 7, kinds, [2., 0.]).order(0).tolist()), [0, 0, 1, 1, 2, 2])
        self.assertEqual(sorted(EpochSampler(tokens, 8, 7, kinds, [1., 1.]).order(0).tolist()), [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "남는"):
            EpochSampler(tokens, 8, 7, kinds, [0., 0.]).order(0)
        with self.assertRaisesRegex(ValueError, "경계"):
            window_kinds(tokens, 8, {"offsets": [3, 20], "kind": [0, 1]})

    def test_optimizer_param_groups_exclude_norms_from_decay(self):
        model = Trid(Config(vocab_size=64, dim=32, layers=2, heads=4, hidden_dim=64, context_length=8))
        names = {id(p): n for n, p in model.named_parameters()}
        groups = param_groups(model, decay_norms=False)
        decayed = {names[id(p)] for p in groups[0]["params"]}
        plain = {names[id(p)] for p in groups[1]["params"]}
        self.assertEqual((groups[0]["weight_decay"], groups[1]["weight_decay"]), (.1, 0.))
        self.assertIn("embedding.weight", decayed)
        self.assertIn("norm.weight", plain)
        self.assertTrue(all(n.endswith("norm.weight") for n in plain), plain)
        self.assertTrue(all(p.ndim <= 1 for p in groups[1]["params"]) and all(p.ndim > 1 for p in groups[0]["params"]))
        self.assertEqual(decayed | plain, set(names.values()))
        self.assertFalse(decayed & plain)
        old = param_groups(model, decay_norms=True)
        self.assertEqual((len(old), len(old[0]["params"]), old[0]["weight_decay"]), (1, len(names), .1))

    def test_checkpoint_cadence_separate_from_eval(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=5, stop_after=5, batch_size=2, device="cpu", eval_every=1,
                           checkpoint_every=2, validation_batches=1)
            pretrain(data, root / "run", **options)
            metrics = json.loads((root / "run/metrics.json").read_text())
            self.assertEqual([m["step"] for m in metrics], [0, 1, 2, 3, 4, 5])
            # Checkpoints at 2 and 4 (never 1 or 3) plus the final step; the last two are kept.
            self.assertEqual(sorted(p.name for p in (root / "run/checkpoints").glob("step-*")), ["step-00000004", "step-00000005"])
            meta = json.loads((root / "run/training.json").read_text())
            self.assertIn(meta["best_step"], {0, 2, 4, 5})
            self.assertEqual([meta[k] for k in ["checkpoint_every", "eval_every", "sampling", "source_weights", "decay_norms"]],
                             [2, 1, "replacement", None, False])
            if meta["best_step"] >= 4:
                best = load_file(str(root / "run/model.safetensors"))
                saved = load_file(str(root / f"run/checkpoints/step-{meta['best_step']:08d}/model.safetensors"))
                for key in best:
                    torch.testing.assert_close(best[key], saved[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "배수"):
                pretrain(data, root / "bad", **{**options, "eval_every": 2, "checkpoint_every": 3})

    def test_epoch_sampling_resumes_at_cursor(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=4, batch_size=2, device="cpu", eval_every=2, validation_batches=1, sampling="epoch")
            pretrain(data, root / "full", stop_after=4, **options)
            pretrain(data, root / "resumed", stop_after=2, **options)
            pretrain(data, root / "resumed", stop_after=2, resume=True, **{**options, "sampling": "replacement"})
            def last(run):
                name = json.loads((run / "latest.json").read_text())["checkpoint"]
                return run / "checkpoints" / name
            a, b = load_file(str(last(root / "full") / "model.safetensors")), load_file(str(last(root / "resumed") / "model.safetensors"))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            state = json.loads((last(root / "resumed") / "state.json").read_text())
            self.assertEqual(state["sampler"], {"epoch": 0, "cursor": 8})  # 4 steps x 2 windows of 24
            meta = json.loads((root / "resumed/training.json").read_text())
            self.assertEqual((meta["sampling"], meta["epoch_windows"], meta["status"]), ("epoch", 24, "complete"))
            self.assertEqual(json.loads((root / "resumed/metrics.json").read_text())[-1]["epoch"], 0)
            replacement = root / "replacement"
            pretrain(data, replacement, stop_after=4, **{**options, "sampling": "replacement"})
            c = load_file(str(last(replacement) / "model.safetensors"))
            self.assertFalse(all(torch.equal(a[k], c[k]) for k in a))
            state.pop("sampler")
            write_json(last(root / "resumed") / "state.json", state)
            with self.assertRaisesRegex(ValueError, "sampler"):
                pretrain(data, root / "resumed", stop_after=1, resume=True, **options)

    def test_run_without_new_keys_resumes_with_old_behaviour(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=4, batch_size=2, device="cpu", eval_every=2, validation_batches=1)
            pretrain(data, root / "full", stop_after=4, decay_norms=True, **options)
            pretrain(data, root / "old", stop_after=2, decay_norms=True, **options)
            meta = json.loads((root / "old/training.json").read_text())
            for key in ["checkpoint_every", "sampling", "source_weights", "decay_norms", "epoch_windows"]:
                meta.pop(key, None)
            write_json(root / "old/training.json", meta)
            # The caller's new defaults (decay_norms=False, epoch sampling) must not override the recorded run.
            pretrain(data, root / "old", stop_after=2, resume=True, sampling="epoch", checkpoint_every=4, **options)
            def last(run):
                name = json.loads((run / "latest.json").read_text())["checkpoint"]
                return load_file(str(run / "checkpoints" / name / "model.safetensors"))
            a, b = last(root / "full"), last(root / "old")
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            meta = json.loads((root / "old/training.json").read_text())
            self.assertEqual([meta[k] for k in ["status", "checkpoint_every", "sampling", "source_weights", "decay_norms"]],
                             ["complete", 2, "replacement", None, True])
            self.assertEqual(len(list((root / "old/checkpoints").glob("step-*"))), 2)

    def test_prepare_writes_sources_and_weighted_epoch_sampling(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            base, cfg = self.make_data(root)
            source = root / "source"
            source.mkdir()
            rows = {"train": [("law", "제1조 이 법은 국민의 권리를 보호한다. " * 3), ("press", "정부는 오늘 새 정책을 발표했다. " * 3),
                              ("press", "장관은 현장을 방문해 점검했다. " * 3), ("law", "제2조 시행일은 공포 후 6개월이다. " * 3)],
                    "valid": [("press", "검증용 보도자료 문장입니다. " * 4)], "test": [("law", "시험용 법령 문장입니다. " * 4)]}
            for split, docs in rows.items():
                (source / f"{split}.jsonl").write_text("".join(json.dumps({"kind": k, "text": t}, ensure_ascii=False) + "\n" for k, t in docs))
            prepare(source, root / "tokens", tokenizer_path=base / "tokenizer.json")
            sources = json.loads((root / "tokens/sources.json").read_text())
            self.assertEqual(sources["kinds"], ["law", "press"])
            train = sources["splits"]["train"]
            self.assertEqual(train["kind"], [0, 1, 1, 0])
            tokens = np.memmap(root / "tokens/train.bin", dtype="<u2", mode="r")
            from tokenizers import Tokenizer
            bos = Tokenizer.from_file(str(base / "tokenizer.json")).token_to_id("<|bos|>")
            self.assertEqual(train["offsets"], np.flatnonzero(tokens == bos).tolist())
            self.assertEqual(train["offsets"][0], 0)
            kinds = window_kinds(tokens, 8, train)
            press_windows = int((kinds == 1).sum())
            self.assertGreater(press_windows, 0)
            options = dict(config_path=cfg, steps=2, stop_after=2, batch_size=2, device="cpu", eval_every=1, validation_batches=1)
            pretrain(root / "tokens", root / "weighted", sampling="epoch", source_weights={"press": 2, "law": 0}, **options)
            meta = json.loads((root / "weighted/training.json").read_text())
            self.assertEqual((meta["source_weights"], meta["epoch_windows"], meta["status"]), ({"press": 2., "law": 0.}, 2 * press_windows, "complete"))
            pretrain(root / "tokens", root / "weighted", resume=True, stop_after=1)
            with self.assertRaisesRegex(ValueError, "kind"):
                pretrain(root / "tokens", root / "bad-kind", sampling="epoch", source_weights={"wiki": 1}, **options)
            with self.assertRaisesRegex(ValueError, "epoch"):
                pretrain(root / "tokens", root / "bad-mode", sampling="replacement", source_weights={"press": 2}, **options)
            with self.assertRaisesRegex(ValueError, "sources.json"):
                pretrain(base, root / "no-sidecar", sampling="epoch", source_weights={"press": 2}, **options)

    def test_gpu_script_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cfg = self.make_data(root)
            script = Path(__file__).resolve().parents[1] / "scripts/train_medium_cpt_gpu.py"
            command = [sys.executable, str(script), "--data", str(data), "--config", str(cfg), "--output", str(root / "out"),
                       "--steps", "2", "--stop-after", "2", "--batch-size", "2", "--accumulate", "1", "--eval-every", "1",
                       "--checkpoint-every", "1", "--validation-batches", "1", "--device", "cpu", "--precision", "fp32",
                       "--no-compile", "--sampling", "epoch", "--stop-flag", str(root / "stop.flag")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            meta = json.loads((root / "out/training.json").read_text())
            self.assertEqual([meta[k] for k in ["status", "completed_steps", "sampling", "checkpoint_every", "decay_norms", "precision", "compile"]],
                             ["complete", 2, "epoch", 1, False, "fp32", False])
            self.assertIn("complete", result.stdout)

    def test_accumulation_matches_larger_batch(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=4, stop_after=4, device="cpu", eval_every=1, validation_batches=1)
            pretrain(data, root / "acc", batch_size=2, accumulate=2, **options)
            pretrain(data, root / "big", batch_size=4, accumulate=1, **options)
            acc = json.loads((root / "acc/metrics.json").read_text())[1:]
            big = json.loads((root / "big/metrics.json").read_text())[1:]
            for a, b in zip(acc, big):
                self.assertAlmostEqual(a["train_loss"], b["train_loss"], delta=1e-5)
                self.assertEqual(a["tokens_seen"], b["tokens_seen"])
            a = load_file(str(root / "acc/checkpoints/step-00000004/model.safetensors"))
            b = load_file(str(root / "big/checkpoints/step-00000004/model.safetensors"))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=1e-5)

    def test_wsd_schedule_and_optional_losses(self):
        self.assertEqual(lr_ratio(10, 100, "wsd"), 1.)
        self.assertEqual(lr_ratio(50, 100, "wsd"), 1.)
        self.assertEqual(lr_ratio(90, 100, "wsd"), 1.)
        self.assertAlmostEqual(lr_ratio(95, 100, "wsd"), .5)
        self.assertEqual(lr_ratio(100, 100, "wsd"), 0.)
        self.assertEqual(lr_ratio(80, 100, "wsd", decay_fraction=.2), 1.)
        self.assertAlmostEqual(lr_ratio(90, 100, "wsd", decay_fraction=.2), .5)
        self.assertEqual(lr_ratio(5, 100), .5)
        self.assertAlmostEqual(lr_ratio(100, 100), .1)
        self.assertAlmostEqual(lr_ratio(55, 100), .1 + .9 * .5 * (1 + math.cos(math.pi * .5)))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=3, stop_after=3, batch_size=2, device="cpu", eval_every=3, validation_batches=1,
                           schedule="wsd", decay_fraction=.34, z_loss=1e-4, precision="bf16")
            pretrain(data, root / "run", **options)
            metadata = json.loads((root / "run/training.json").read_text())
            self.assertEqual({k: metadata[k] for k in ["schedule", "decay_fraction", "z_loss", "precision", "compile", "accumulate"]},
                             {"schedule": "wsd", "decay_fraction": .34, "z_loss": 1e-4, "precision": "bf16", "compile": False, "accumulate": 1})
            self.assertTrue(all(not k.startswith("_orig_mod") for k in load_file(str(root / "run/model.safetensors"))))
            with self.assertRaisesRegex(ValueError, "schedule"):
                pretrain(data, root / "bad", **{**options, "schedule": "linear"})

    def test_all_tokens_are_next_token_targets(self):
        values = np.arange(80, dtype="<u2")
        x, y = sample_batch(values, 8, 2, torch.Generator().manual_seed(1), "cpu")
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        self.assertTrue(torch.all(y == x + 1))
        self.assertFalse(torch.any(y == -100))

    def test_dense_train_loss_matches_token_mean_ce_and_gradients(self):
        torch.manual_seed(3)
        model = Trid(Config(vocab_size=64, dim=32, layers=2, heads=4, hidden_dim=64, context_length=8))
        values = np.arange(60, dtype="<u2")
        x, y = sample_batch(values, 8, 3, torch.Generator().manual_seed(2), "cpu")
        logits, _ = model(x)
        expected = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        model.zero_grad(set_to_none=True)
        expected.backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters()}
        actual = train_loss(model, x, y)
        torch.testing.assert_close(actual, expected)
        model.zero_grad(set_to_none=True)
        actual.backward()
        for name, p in model.named_parameters():
            self.assertGreater(float(grads[name].abs().sum()), 0.0, name)
            torch.testing.assert_close(p.grad, grads[name], msg=name)
        with_z = train_loss(model, x, y, z_loss=1e-2)
        self.assertGreater(with_z.item(), expected.item())
        self.assertAlmostEqual(validate(model, values, 3, "cpu", batches=1),
                               train_loss(model, *sample_batch(values, 8, 3, torch.Generator().manual_seed(20261026), "cpu")).item(),
                               places=6)

    def test_continuation_preserves_parent_and_resumes_new_schedule(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            parent = root / "parent"
            opts = dict(steps=4, batch_size=2, device="cpu", eval_every=2, validation_batches=1)
            pretrain(data, parent, config_path=cfg, stop_after=4, **opts)
            original = {p.name: p.read_bytes() for p in parent.iterdir() if p.is_file()}
            pretrain(data, root / "fork", init_run=parent, stop_after=2, **opts)
            initial = json.loads((root / "fork/metrics.json").read_text())[0]
            parent_meta = json.loads((parent / "training.json").read_text())
            self.assertAlmostEqual(initial["valid_loss"], parent_meta["best_validation_loss"], places=6)
            pretrain(data, root / "fork", resume=True, stop_after=2, **opts)
            pretrain(data, root / "full-fork", init_run=parent, stop_after=4, **opts)
            def last(run):
                name = json.loads((run / "latest.json").read_text())["checkpoint"]
                return load_file(str(run / "checkpoints" / name / "model.safetensors"))
            a, b = last(root / "fork"), last(root / "full-fork")
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            result = json.loads((root / "fork/training.json").read_text())
            self.assertEqual(result["initial_weights_sha256"], sha256(parent / "model.safetensors"))
            self.assertEqual(result["lineage_tokens_seen"], parent_meta["best_step"] * 16 + 64)
            self.assertEqual(original, {p.name: p.read_bytes() for p in parent.iterdir() if p.is_file()})
            with self.assertRaisesRegex(ValueError, "함께"):
                pretrain(data, root / "bad", init_run=parent, resume=True, **opts)
            write_json(parent / "training.json", {**parent_meta, "status": "training"})
            with self.assertRaisesRegex(ValueError, "먼저 종료"):
                pretrain(data, root / "live-parent", init_run=parent, **opts)
            write_json(parent / "training.json", parent_meta)
            (data / "tokenizer.json").write_text((data / "tokenizer.json").read_text() + "\n")
            manifest = json.loads((data / "manifest.json").read_text())
            manifest["tokenizer_sha256"] = sha256(data / "tokenizer.json")
            write_json(data / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "토크나이저가 다릅니다"):
                pretrain(data, root / "wrong-tokens", init_run=parent, **opts)

    def test_corpus_reuses_tokenizer_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            base, _ = self.make_data(root)
            source = root / "source"
            source.mkdir()
            for split, text in [("train", "한국어 문장을 학습합니다."), ("valid", "금액은 이만 원입니다."), ("test", "별도의 문서입니다.")]:
                (source / f"{split}.jsonl").write_text(json.dumps({"text": text}))
            prepare(source, root / "tokens", tokenizer_path=base / "tokenizer.json")
            self.assertEqual((base / "tokenizer.json").read_bytes(), (root / "tokens/tokenizer.json").read_bytes())

    def test_document_split_and_duplicate_rejection(self):
        text = "문서 단위로 나눕니다."
        self.assertEqual(split_for(text), split_for(text))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for split in ["train", "valid", "test"]:
                (source / f"{split}.jsonl").write_text(json.dumps({"text": text}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "동일 문서"):
                prepare(source, root / "tokens", 300)

    def test_sft_keeps_initial_weights_if_validation_worsens_and_extends_context(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            base, data = root / "base", root / "sft-data"
            base.mkdir(); data.mkdir()
            tok = train_tokenizer([{"prompt": "한국어", "answer": "문장"}], base / "tokenizer.json", 300)
            cfg = Config(vocab_size=tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=8)
            model = Trid(cfg)
            write_json(base / "config.json", asdict(cfg))
            save_file(model.state_dict(), str(base / "model.safetensors"))
            for split, text in [("train", "새 한국어 질문을 읽고 대답합니다."), ("valid", "별도의 한국어 문장을 확인합니다.")]:
                (data / f"{split}.jsonl").write_text(json.dumps({"prompt": text, "answer": "문장"}))
            output = root / "extended"
            with patch("trid.training.validation_loss", side_effect=[1.0, 2.0]):
                train(data, output, steps=1, batch_size=1, device="cpu", init_run=base, context_length=128)
            actual = load_file(str(output / "model.safetensors"))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, actual[key], rtol=0, atol=0)
            metadata = json.loads((output / "training.json").read_text())
            self.assertEqual(metadata["best_step"], 0)
            self.assertEqual(metadata["config"]["context_length"], 128)
            self.assertEqual(metadata["inherited_context_length"], 8)
            self.assertEqual(json.loads((base / "config.json").read_text())["context_length"], 8)

    def test_dialogue_training_inherits_own_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            base, data = root / "base", root / "dialogue"
            base.mkdir(); data.mkdir()
            tok = train_tokenizer([{"prompt": "질문 확인", "answer": "네"}], base / "tokenizer.json", 300)
            cfg = Config(vocab_size=tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=128)
            model = Trid(cfg)
            write_json(base / "config.json", asdict(cfg))
            save_file(model.state_dict(), str(base / "model.safetensors"))
            for split, prompt in [("train", "질문"), ("valid", "확인")]:
                (data / f"{split}.jsonl").write_text(json.dumps({"prompt": prompt, "answer": "네"}), encoding="utf-8")
            output = root / "sft"
            train(data, output, steps=1, batch_size=1, lr=1e-12, device="cpu", init_run=base)
            self.assertEqual((base / "tokenizer.json").read_bytes(), (output / "tokenizer.json").read_bytes())
            actual = load_file(str(output / "model.safetensors"))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, actual[key], atol=1e-9, rtol=1e-6)
            metadata = json.loads((output / "training.json").read_text())
            self.assertEqual(metadata["initial_weights_sha256"], sha256(base / "model.safetensors"))

    def test_dialogue_training_split_into_sessions_matches_one_run(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            base, data = root / "base", root / "dialogue"
            base.mkdir(); data.mkdir()
            rows = [{"prompt": f"질문 {i}", "answer": f"답 {i}"} for i in range(6)]
            tok = train_tokenizer(rows, base / "tokenizer.json", 300)
            cfg = Config(vocab_size=tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=64)
            torch.manual_seed(0)
            write_json(base / "config.json", asdict(cfg))
            save_file(Trid(cfg).state_dict(), str(base / "model.safetensors"))
            (data / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:5]), encoding="utf-8")
            (data / "valid.jsonl").write_text(json.dumps(rows[5], ensure_ascii=False), encoding="utf-8")
            args = dict(steps=7, batch_size=2, micro_batch_size=1, lr=1e-2, device="cpu", init_run=base, eval_every=2, seed=7,
                        save_steps=(2, 7), checkpoint_every=3, resume=True)
            full, cut = root / "full", root / "cut"
            self.assertEqual(train(data, full, **args)["status"], "complete")
            # Session 1 ends after step 4 with a recovery checkpoint; session 2 restores it and finishes.
            self.assertEqual(train(data, cut, session_steps=4, **args)["status"], "checkpointed")
            self.assertEqual(sorted(p.name for p in (cut / "checkpoints").iterdir()), ["step-00000003", "step-00000004"])
            meta = train(data, cut, **args)
            self.assertEqual((meta["status"], meta["resumed_from_step"], meta["completed_steps"]), ("complete", 4, 7))
            for name in ("last-model.safetensors", "model.safetensors", "milestones/step-00000002/model.safetensors",
                         "milestones/step-00000007/model.safetensors"):
                self.assertEqual(sha256(full / name), sha256(cut / name), name)
            strip = lambda path: [{k: v for k, v in m.items() if k != "elapsed_seconds"} for m in json.loads(path.read_text())]
            self.assertEqual(strip(full / "metrics.json"), strip(cut / "metrics.json"))


if __name__ == "__main__":
    unittest.main()

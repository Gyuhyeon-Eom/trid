import contextlib
import importlib.util
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from trid.corpus import sha256, write_json, prepare, split_for
from trid.pretraining import lr_ratio, pretrain, sample_batch
from trid.text import train_tokenizer
from trid.model import Config, Trid
from trid.training import train
from dataclasses import asdict

# Snapshot of trid/pretraining.py taken before the accumulate/precision/compile/schedule/z_loss options.
BASELINE = Path(os.environ.get("TRID_PRETRAIN_BASELINE", "/tmp/pretraining_before.py"))


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

    @unittest.skipUnless(BASELINE.exists(), f"{BASELINE} 없음: 옵션 추가 전 pretraining.py 스냅샷이 필요합니다")
    def test_defaults_match_pre_option_code_bitwise(self):
        spec = importlib.util.spec_from_file_location("trid._pretraining_before", BASELINE)
        before = importlib.util.module_from_spec(spec)
        before.__package__ = "trid"
        spec.loader.exec_module(before)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, cfg = self.make_data(root)
            options = dict(config_path=cfg, steps=3, stop_after=3, batch_size=2, device="cpu", eval_every=1, validation_batches=1)
            before.pretrain(data, root / "old", **options)
            pretrain(data, root / "new", **options)
            old = json.loads((root / "old/metrics.json").read_text())
            new = json.loads((root / "new/metrics.json").read_text())
            self.assertEqual(len(old), 4)
            for a, b in zip(old, new):
                for key in ["step", "train_loss", "valid_loss", "perplexity", "tokens_seen"]:
                    self.assertEqual(a.get(key), b.get(key), key)
            a, b = load_file(str(root / "old/model.safetensors")), load_file(str(root / "new/model.safetensors"))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(json.loads((root / "new/training.json").read_text())["schedule"], "cosine")

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


if __name__ == "__main__":
    unittest.main()

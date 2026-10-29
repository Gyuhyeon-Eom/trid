import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from trid.model import Config, Trid
from trid.text import train_tokenizer
from trid.training import batch, loss_for, loss_terms, train, validation_loss

# Unequal answer lengths (4, 1, 2, 2, 2 targets) and one example without an intent label.
EXAMPLES = [([1, 2, 3, 4, 5, 6], [-100, -100, 4, 5, 6, 7]),
            ([7, 8], [-100, 9]),
            ([3, 9, 10, 11, 12, 13, 14], [-100, -100, -100, -100, -100, 13, 14]),
            ([5, 6, 7, 8, 9], [-100, -100, -100, 9, 10]),
            ([2, 3, 4], [-100, 4, 5])]
INTENTS = [0, 1, -100, 2, 1]
PAD = 0


def tiny(num_intents=0, seed=0):
    torch.manual_seed(seed)
    return Trid(Config(vocab_size=64, dim=16, layers=2, heads=4, hidden_dim=32,
                       context_length=16, num_intents=num_intents))


def dense_objective(model, x, y, intents=None, intent_weight=0.0, loss_normalization="token"):
    """The original single-pass objective over full (batch, length, vocab) logits."""
    logits, _, hidden = model(x, return_hidden=True)
    flat = logits.reshape(-1, logits.size(-1))
    if loss_normalization == "token":
        loss = F.cross_entropy(flat, y.reshape(-1), ignore_index=-100)
    else:
        token_losses = F.cross_entropy(flat, y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
        loss = (token_losses.sum(dim=1) / (y != -100).sum(dim=1)).mean()
    if intents is not None and intent_weight:
        position = (y != -100).to(torch.int32).argmax(dim=1)
        question = hidden[torch.arange(len(x)), position]
        loss = loss + intent_weight * F.cross_entropy(model.intent_head(question), intents, ignore_index=-100)
    return loss


def gradients(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grads


def assert_same_gradients(case, expected, actual):
    case.assertEqual(set(expected), set(actual))
    for name in expected:
        if expected[name] is None:
            # No intent labels: the intent head is untouched by both objectives.
            case.assertEqual(name, "intent_head.weight")
            case.assertIsNone(actual[name])
            continue
        case.assertGreater(float(expected[name].abs().sum()), 0.0, name)
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)


class LossNormalizationTests(unittest.TestCase):
    def test_sparse_projection_matches_dense_objective_and_every_gradient(self):
        x, y = batch(EXAMPLES, PAD, "cpu")
        intents = torch.tensor(INTENTS)
        for mode in ("token", "example"):
            for labels, weight in ((None, 0.0), (intents, 0.5)):
                model = tiny(num_intents=3)
                expected = dense_objective(model, x, y, labels, weight, mode)
                actual = loss_for(model, x, y, labels, weight, mode)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                assert_same_gradients(self, gradients(model, expected), gradients(model, actual))

    def test_example_normalization_is_not_token_weighted(self):
        model = tiny()
        x, y = batch(EXAMPLES, PAD, "cpu")
        singles = [loss_for(model, *batch([e], PAD, "cpu")) for e in EXAMPLES]
        counts = [sum(label != -100 for label in labels) for _, labels in EXAMPLES]
        example_mean = sum(singles) / len(singles)
        token_mean = sum(c * s for c, s in zip(counts, singles)) / sum(counts)
        torch.testing.assert_close(loss_for(model, x, y, loss_normalization="example"), example_mean)
        torch.testing.assert_close(loss_for(model, x, y, loss_normalization="token"), token_mean)
        self.assertFalse(torch.isclose(example_mean, token_mean))

    def test_intent_term_is_mean_over_labelled_questions_only(self):
        model = tiny(num_intents=3)
        x, y = batch(EXAMPLES, PAD, "cpu")
        intents = torch.tensor(INTENTS)
        _, _, intent_sum, intent_count = loss_terms(model, x, y, intents)
        self.assertEqual(int(intent_count), 4)
        answer = loss_for(model, x, y)
        with_intent = loss_for(model, x, y, intents, 0.5)
        torch.testing.assert_close(with_intent - answer, 0.5 * intent_sum / 4)
        self.assertIsNone(loss_terms(model, x, y, torch.full((5,), -100))[2])
        self.assertIsNone(loss_terms(model, x, y)[2])

    def test_micro_batch_partitions_sum_to_single_pass_gradient(self):
        intents = torch.tensor(INTENTS)
        x, y = batch(EXAMPLES, PAD, "cpu")
        for mode in ("token", "example"):
            model = tiny(num_intents=3)
            expected = gradients(model, dense_objective(model, x, y, intents, 0.5, mode))
            answer_total = len(EXAMPLES) if mode == "example" else sum(
                sum(1 for label in labels if label != -100) for _, labels in EXAMPLES)
            intent_total = sum(1 for label in INTENTS if label != -100)
            model.zero_grad(set_to_none=True)
            total = 0.0
            # Partition of 2 over 5 examples: the last micro-batch is partial and padded shorter.
            for start in range(0, len(EXAMPLES), 2):
                xs, ys = batch(EXAMPLES[start:start + 2], PAD, "cpu")
                answer_sum, _, intent_sum, _ = loss_terms(model, xs, ys, intents[start:start + 2], mode)
                partial = answer_sum / answer_total
                if intent_sum is not None:
                    partial = partial + 0.5 * intent_sum / intent_total
                partial.backward()
                total += partial.item()
            actual = {name: p.grad.clone() for name, p in model.named_parameters()}
            self.assertAlmostEqual(total, dense_objective(model, x, y, intents, 0.5, mode).item(), places=5)
            assert_same_gradients(self, expected, actual)

    def test_validation_invariant_to_partition_size_with_intents(self):
        model = tiny(num_intents=3)
        x, y = batch(EXAMPLES, PAD, "cpu")
        intents = torch.tensor(INTENTS)
        for mode in ("token", "example"):
            with torch.no_grad():
                reference = dense_objective(model, x, y, intents, 0.5, mode).item()
            for size in (1, 2, 3, 5):
                self.assertAlmostEqual(validation_loss(model, EXAMPLES, PAD, "cpu", size, INTENTS, 0.5, mode),
                                       reference, places=5)
            self.assertAlmostEqual(validation_loss(model, EXAMPLES, PAD, "cpu", 2, loss_normalization=mode),
                                   dense_objective(model, x, y, loss_normalization=mode).item(), places=5)

    def test_missing_answer_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, "answer target"):
            loss_for(tiny(), torch.tensor([[1]]), torch.tensor([[-100]]), loss_normalization="example")


class MicroBatchTrainingTests(unittest.TestCase):
    ROWS = [("train", "출장비 한도는 얼마입니까", "일일 사만 원입니다", "travel"),
            ("train", "연차는 며칠인가요", "십오 일", "leave"),
            ("train", "회의실 예약 방법", "포털에서 신청합니다", "travel"),
            ("train", "점심시간은", "한 시간", "public_daily"),
            ("train", "보안 교육 주기", "연 일 회", "leave"),
            ("valid", "휴가 신청 절차", "결재 후 등록", "leave"),
            ("valid", "주차 등록은 어디서", "총무팀", "travel"),
            ("valid", "비밀번호 변경 주기", "구십 일", "public_daily")]

    def run_training(self, root, name, **options):
        base, data = root / "base", root / "data"
        if not base.exists():
            base.mkdir(); data.mkdir()
            rows = [{"prompt": p, "answer": a} for _, p, a, _ in self.ROWS]
            tok = train_tokenizer(rows, base / "tokenizer.json", 300)
            cfg = Config(vocab_size=tok.get_vocab_size(), dim=16, layers=1, heads=4, hidden_dim=32, context_length=128)
            model = Trid(cfg)
            (base / "config.json").write_text(json.dumps(asdict(cfg)))
            save_file(model.state_dict(), str(base / "model.safetensors"))
            for split in ("train", "valid"):
                lines = [json.dumps({"prompt": p, "answer": a, "task": t}, ensure_ascii=False)
                         for s, p, a, t in self.ROWS if s == split]
                (data / f"{split}.jsonl").write_text("\n".join(lines), encoding="utf-8")
        output = root / name
        with contextlib.redirect_stdout(io.StringIO()):
            train(data, output, steps=3, batch_size=5, lr=1e-3, device="cpu", seed=7, init_run=base,
                  eval_every=3, validation_limit=0, **options)
        return (load_file(str(output / "last-model.safetensors")),
                json.loads((output / "metrics.json").read_text()),
                json.loads((output / "training.json").read_text()))

    def test_partial_micro_batches_reproduce_single_pass_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = dict(intent_weight=0.5, loss_normalization="example")
            # micro_batch_size=2 over batch_size=5 ends with a partial micro-batch of 1.
            whole, whole_metrics, whole_meta = self.run_training(root, "whole", micro_batch_size=5, **options)
            split, split_metrics, split_meta = self.run_training(root, "split", micro_batch_size=2, **options)
            initial = load_file(str(root / "base/model.safetensors"))
            self.assertEqual(whole_meta["micro_batch_size"], 5)
            self.assertEqual(split_meta["micro_batch_size"], 2)
            self.assertEqual(split_meta["intent_map"], {"leave": 0, "travel": 1})
            self.assertEqual(set(whole), set(split))
            moved = False
            for key in whole:
                torch.testing.assert_close(split[key], whole[key], rtol=1e-5, atol=1e-6, msg=key)
                if key in initial and not torch.equal(whole[key], initial[key]):
                    moved = True
            self.assertTrue(moved)
            self.assertIn("intent_head.weight", whole)
            for a, b in zip(whole_metrics, split_metrics):
                self.assertEqual(a["step"], b["step"])
                self.assertAlmostEqual(a["valid_loss"], b["valid_loss"], places=5)
                if "train_loss" in a:
                    self.assertAlmostEqual(a["train_loss"], b["train_loss"], places=5)

    def test_micro_batch_size_is_capped_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, meta = self.run_training(root, "capped", micro_batch_size=99)
            self.assertEqual(meta["micro_batch_size"], 5)
            with self.assertRaisesRegex(ValueError, "micro-batch-size"):
                self.run_training(root, "invalid", micro_batch_size=0)

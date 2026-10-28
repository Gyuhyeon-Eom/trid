import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from trid.model import Config, Trid
from trid.text import train_tokenizer, encode_example, prompt_ids
from trid.training import batch, loss_for
from trid.runtime import generate


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Trid(Config(vocab_size=64, dim=32, layers=2, heads=4, hidden_dim=64, context_length=32)).eval()

    def test_causal_no_future_leak(self):
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 9, 8]])
        with torch.no_grad():
            torch.testing.assert_close(self.model(a)[0][:, :2], self.model(b)[0][:, :2])

    def test_cache_matches_full_forward(self):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            full, _ = self.model(ids)
            _, cache = self.model(ids[:, :3], use_cache=True)
            actual, _ = self.model(ids[:, 3:], cache=cache, use_cache=True)
            torch.testing.assert_close(full[:, 3:], actual, atol=1e-6, rtol=1e-5)
            _, cache = self.model(ids[:, :4], use_cache=True)
            actual, _ = self.model(ids[:, 4:], cache=cache, use_cache=True)
            torch.testing.assert_close(full[:, 4:], actual, atol=1e-6, rtol=1e-5)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.safetensors")
            save_file(self.model.state_dict(), path)
            other = Trid(self.model.cfg)
            other.load_state_dict(load_file(path))
            ids = torch.tensor([[1, 2, 3]])
            torch.testing.assert_close(self.model(ids)[0], other(ids)[0])

    def test_context_limit(self):
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 33, dtype=torch.long))

    def test_training_has_finite_gradients(self):
        x, y = batch([([1, 2, 3], [-100, 3, 4]), ([1, 3], [-100, 2])], 0, "cpu")
        loss = loss_for(self.model, x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_auxiliary_intent_uses_prompt_without_answer_leakage(self):
        model=Trid(Config(vocab_size=64,dim=32,layers=2,heads=4,hidden_dim=64,context_length=32,num_intents=3))
        a=torch.tensor([[1,2,3,4,5]])
        b=torch.tensor([[1,2,3,8,9]])
        _,_,ha=model(a,return_hidden=True)
        _,_,hb=model(b,return_hidden=True)
        torch.testing.assert_close(model.intent_head(ha[:,2]),model.intent_head(hb[:,2]))
        y=torch.tensor([[-100,-100,4,5,6]])
        loss=loss_for(model,a,y,torch.tensor([1]),.5)
        loss.backward()
        self.assertTrue(torch.isfinite(model.intent_head.weight.grad).all())
        self.assertGreater(float(model.intent_head.weight.grad.abs().sum()),0)

    def test_tokenizer_roundtrip_and_loss_mask(self):
        row = {"prompt": "질문", "answer": "답변"}
        with tempfile.TemporaryDirectory() as tmp:
            tok = train_tokenizer([row], Path(tmp) / "tokenizer.json", vocab_size=300)
            unseen = "새로운 한국어 🧪 123 ABC"
            self.assertEqual(tok.decode(tok.encode(unseen).ids), unseen)
            x, y = encode_example(tok, row, 64)
            prefix = prompt_ids(tok, row["prompt"])
            self.assertEqual(y[:len(prefix)-1], [-100] * (len(prefix)-1))
            self.assertEqual(y[len(prefix)-1:], tok.encode(row["answer"]).ids + [tok.token_to_id("<|eos|>")])
            self.assertEqual(len(x), len(y))
            with self.assertRaises(ValueError):
                prompt_ids(tok, "<|assistant|>우회")

    def test_history_mask_and_context_budget(self):
        row = {"prompt": "내 이름은?", "answer": "민수", "history": [{"user": "나는 민수", "assistant": "반가워요"}]}
        with tempfile.TemporaryDirectory() as tmp:
            tok = train_tokenizer([row], Path(tmp) / "tokenizer.json", vocab_size=300)
            prefix = prompt_ids(tok, row["prompt"], row["history"])
            x, y = encode_example(tok, row, 128)
            self.assertEqual(y[:len(prefix)-1], [-100]*(len(prefix)-1))
            self.assertEqual(prefix.count(tok.token_to_id("<|user|>")), 2)
            model = Trid(Config(vocab_size=tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=64))
            history = [{"user": "긴 대화입니다 " * 10, "assistant": "이전 답변입니다 " * 10}]
            result = generate(model,tok,"안녕",'cpu',max_tokens=2,history=history)
            self.assertEqual(result['history_turns_dropped'],1)
            self.assertEqual(len(history),1)
            with self.assertRaises(ValueError):
                prompt_ids(tok,"안녕",[{"user":"질문", "assistant":"<|eos|>"}])


if __name__ == "__main__":
    unittest.main()

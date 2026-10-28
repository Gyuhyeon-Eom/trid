import unittest
from dataclasses import asdict

import torch

from trid.model import Attention, Config, Trid


class AttentionOptionsTests(unittest.TestCase):
    def config(self, **kwargs):
        return Config(vocab_size=64, dim=32, heads=4, layers=2,
                      hidden_dim=64, context_length=32, **kwargs)

    def test_old_configuration_and_state_keys(self):
        cfg = self.config()
        legacy = asdict(cfg)
        legacy.pop('kv_heads')
        legacy.pop('qk_norm')
        a = Trid(Config(**legacy))
        b = Trid(self.config(kv_heads=4, qk_norm=False))
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertFalse(any('q_norm' in k or 'k_norm' in k for k in a.state_dict()))
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(a(ids)[0], b(ids)[0], atol=0, rtol=0)

    def test_invalid_options(self):
        for value in (0, -1, 3, 8, 2.0, True):
            with self.subTest(kv_heads=value), self.assertRaises(ValueError):
                self.config(kv_heads=value)
        with self.assertRaises(ValueError):
            self.config(qk_norm='false')

    def test_cache_causality_and_gradients(self):
        devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
        for device in devices:
            for kv_heads in (1, 2, 4):
                for qk_norm in (False, True):
                    with self.subTest(device=device, kv_heads=kv_heads, qk_norm=qk_norm):
                        model = Trid(self.config(kv_heads=kv_heads, qk_norm=qk_norm)).to(device)
                        ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
                        with torch.no_grad():
                            full, _ = model(ids)
                            changed, _ = model(torch.tensor([[1, 2, 9, 8, 7]], device=device))
                            torch.testing.assert_close(full[:, :2], changed[:, :2], atol=2e-6, rtol=1e-5)
                            for prefix in (3, 4):
                                _, cache = model(ids[:, :prefix], use_cache=True)
                                actual, cache = model(ids[:, prefix:], cache=cache, use_cache=True)
                                torch.testing.assert_close(full[:, prefix:], actual, atol=2e-6, rtol=1e-5)
                                self.assertEqual(cache[0][0].shape, (1, kv_heads, 5, 8))
                        logits, _ = model(ids)
                        torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids.flatten()).backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in model.parameters()))

    def test_grouped_attention_matches_explicit_shared_heads(self):
        # A full-head reference with repeated K/V weights must produce the same
        # results and input gradients as grouped attention, including QK norm.
        for normalize in (False, True):
            small = Attention(self.config(kv_heads=2, qk_norm=normalize))
            reference = Attention(self.config(kv_heads=4, qk_norm=normalize))
            with torch.no_grad():
                q, k, v = small.qkv.weight.split((32, 16, 16))
                repeated = [w.reshape(2, 8, 32).repeat_interleave(2, dim=0).reshape(32, 32)
                            for w in (k, v)]
                reference.qkv.weight.copy_(torch.cat([q, *repeated]))
                reference.output.load_state_dict(small.output.state_dict())
                reference.q_norm.load_state_dict(small.q_norm.state_dict())
                reference.k_norm.load_state_dict(small.k_norm.state_dict())
            x = torch.randn(2, 7, 32, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            a, _ = small(x)
            b, _ = reference(y)
            torch.testing.assert_close(a, b)
            a.square().sum().backward()
            b.square().sum().backward()
            torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)

    def test_experimental_checkpoint_roundtrip_and_size(self):
        cfg = self.config(kv_heads=2, qk_norm=True)
        model = Trid(cfg)
        restored = Trid(Config(**asdict(cfg)))
        restored.load_state_dict(model.state_dict(), strict=True)
        ids = torch.tensor([[2, 4, 6]])
        torch.testing.assert_close(model(ids)[0], restored(ids)[0], atol=0, rtol=0)
        self.assertLess(model.parameter_count(), Trid(self.config()).parameter_count())
        with self.assertRaises(RuntimeError):
            model.load_state_dict(Trid(self.config()).state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()

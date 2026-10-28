import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import export_gguf  # noqa: E402
from trid.model import Config, Trid, rotary  # noqa: E402
from trid.text import train_tokenizer  # noqa: E402

gguf = export_gguf.gguf

ROWS = [{"prompt": "광역시 숙박비 상한은 얼마인가요?", "answer": "8만 원 [D3]"},
        {"prompt": "일비는 1일 얼마인가요?", "answer": "2만 원 [D3]"},
        {"prompt": "식비 기준을 알려 주세요.", "answer": "2만 5천 원 [D3]"}] * 4


def field_value(reader, key):
    field = reader.fields[key]
    if field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    return field.parts[field.data[0]].tolist()[0]


class ExportGgufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tok_path = Path(cls.tmp.name) / "tokenizer.json"
        cls.tok = train_tokenizer(ROWS, cls.tok_path, vocab_size=320)
        cls.vocab_size = cls.tok.get_vocab_size()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_run(self, name, **overrides):
        cfg = Config(vocab_size=self.vocab_size, dim=32, heads=4, layers=2, hidden_dim=64,
                     context_length=32, **overrides)
        torch.manual_seed(0)
        model = Trid(cfg)
        root = Path(self.tmp.name) / name
        root.mkdir()
        (root / "config.json").write_text(json.dumps(asdict(cfg)))
        (root / "tokenizer.json").write_bytes(self.tok_path.read_bytes())
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(root / "model.safetensors"))
        return cfg, model, root

    def test_llama_round_trip(self):
        cfg, model, root = self.make_run("llama")
        out = root / "model.gguf"
        info = export_gguf.export(root, out, "f32")
        self.assertEqual(info["arch"], "llama")
        reader = gguf.GGUFReader(str(out))
        self.assertEqual(field_value(reader, "general.architecture"), "llama")
        self.assertEqual(field_value(reader, "llama.embedding_length"), 32)
        self.assertEqual(field_value(reader, "llama.block_count"), 2)
        self.assertEqual(field_value(reader, "llama.attention.head_count"), 4)
        self.assertEqual(field_value(reader, "llama.attention.head_count_kv"), 4)
        self.assertEqual(field_value(reader, "llama.rope.dimension_count"), 8)
        self.assertEqual(field_value(reader, "llama.context_length"), 32)
        self.assertEqual(field_value(reader, "tokenizer.ggml.model"), "gpt2")
        self.assertEqual(field_value(reader, "tokenizer.ggml.pre"), "gpt-2")
        self.assertEqual(field_value(reader, "tokenizer.ggml.bos_token_id"), self.tok.token_to_id("<|bos|>"))
        self.assertEqual(field_value(reader, "tokenizer.ggml.eos_token_id"), self.tok.token_to_id("<|eos|>"))
        self.assertEqual(len(reader.fields["tokenizer.ggml.tokens"].data), self.vocab_size)
        tensors = {t.name: t for t in reader.tensors}
        expected = {"token_embd.weight": (self.vocab_size, 32), "output_norm.weight": (32,)}
        for i in range(2):
            expected.update({f"blk.{i}.attn_norm.weight": (32,), f"blk.{i}.attn_q.weight": (32, 32),
                             f"blk.{i}.attn_k.weight": (32, 32), f"blk.{i}.attn_v.weight": (32, 32),
                             f"blk.{i}.attn_output.weight": (32, 32), f"blk.{i}.ffn_norm.weight": (32,),
                             f"blk.{i}.ffn_gate.weight": (64, 32), f"blk.{i}.ffn_up.weight": (64, 32),
                             f"blk.{i}.ffn_down.weight": (32, 64)})
        self.assertEqual(set(tensors), set(expected))
        for name, shape in expected.items():
            self.assertEqual(tuple(reversed(tensors[name].shape.tolist())), shape, name)
            self.assertEqual(tensors[name].tensor_type, gguf.GGMLQuantizationType.F32)
        # 값 자체도 그대로 들어간다 (qkv 분할 포함).
        qkv = model.blocks[0].attn.qkv.weight.detach().numpy()
        np.testing.assert_array_equal(np.asarray(tensors["blk.0.attn_k.weight"].data).reshape(32, 32), qkv[32:64])
        np.testing.assert_array_equal(np.asarray(tensors["token_embd.weight"].data).reshape(-1, 32),
                                      model.embedding.weight.detach().numpy())
        self.assertEqual(export_gguf.describe(out)["tensor_count"], len(expected))

    def test_gqa_and_q8_0(self):
        cfg, model, root = self.make_run("gqa", kv_heads=2)
        out = root / "model.q8_0.gguf"
        export_gguf.export(root, out, "q8_0")
        reader = gguf.GGUFReader(str(out))
        self.assertEqual(field_value(reader, "llama.attention.head_count_kv"), 2)
        tensors = {t.name: t for t in reader.tensors}
        self.assertEqual(tuple(reversed(tensors["blk.0.attn_k.weight"].shape.tolist())), (16, 32))
        self.assertEqual(tuple(reversed(tensors["blk.0.attn_q.weight"].shape.tolist())), (32, 32))
        self.assertEqual(tensors["blk.0.attn_k.weight"].tensor_type, gguf.GGMLQuantizationType.Q8_0)
        self.assertEqual(tensors["blk.0.attn_norm.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
        back = gguf.quants.dequantize(np.asarray(tensors["blk.0.ffn_up.weight"].data), gguf.GGMLQuantizationType.Q8_0)
        np.testing.assert_allclose(back.reshape(64, 32), model.blocks[0].up.weight.detach().numpy(), atol=2e-3)

    def test_qk_norm_uses_qwen3_with_neox_permutation(self):
        cfg, model, root = self.make_run("qk", qk_norm=True, kv_heads=2)
        out = root / "model.gguf"
        self.assertEqual(export_gguf.export(root, out, "f32")["arch"], "qwen3")
        reader = gguf.GGUFReader(str(out))
        self.assertEqual(field_value(reader, "qwen3.attention.head_count_kv"), 2)
        tensors = {t.name: t for t in reader.tensors}
        self.assertIn("blk.1.attn_q_norm.weight", tensors)
        self.assertEqual(tuple(reversed(tensors["blk.1.attn_k_norm.weight"].shape.tolist())), (8,))
        self.assertNotIn("output.weight", tensors)
        # NEOX(앞·뒤 절반 쌍) RoPE를 치환된 q에 적용한 결과가 trid 인접 쌍 RoPE와 같아야 한다.
        head_dim, perm = 8, export_gguf.neox_permutation(8)
        x = torch.randn(1, 1, 5, head_dim)
        trid_rope = rotary(x, 0, cfg.rope_theta)
        xp = x[..., perm]
        half = head_dim // 2
        inv = cfg.rope_theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.arange(5, dtype=torch.float32)[:, None] * inv[None, :]
        cos, sin = angles.cos(), angles.sin()
        a, b = xp[..., :half], xp[..., half:]
        neox_rope = torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)
        torch.testing.assert_close(neox_rope, trid_rope[..., perm])
        # 저장된 q 행과 q_norm은 같은 치환을 받는다.
        q = model.blocks[0].attn.qkv.weight.detach().numpy()[:32]
        stored = np.asarray(tensors["blk.0.attn_q.weight"].data).reshape(32, 32)
        np.testing.assert_array_equal(stored, export_gguf.permute_heads(q, 8, perm))
        np.testing.assert_array_equal(np.asarray(tensors["blk.0.attn_q_norm.weight"].data),
                                      model.blocks[0].attn.q_norm.weight.detach().numpy()[perm])

    def test_unsupported_configs(self):
        _, _, root = self.make_run("intent", num_intents=3)
        with self.assertRaisesRegex(ValueError, "num_intents"):
            export_gguf.export(root, root / "x.gguf", "f32")
        _, _, root = self.make_run("tok")
        tok = json.loads((root / "tokenizer.json").read_text())
        tok["normalizer"] = {"type": "NFKC"}
        (root / "tokenizer.json").write_text(json.dumps(tok))
        with self.assertRaisesRegex(ValueError, "ByteLevel"):
            export_gguf.export(root, root / "x.gguf", "f32")
        with self.assertRaises(ValueError):
            export_gguf.export(root, root / "x.gguf", "q4_0")

    def test_check_prints_metadata(self):
        _, _, root = self.make_run("check")
        out = root / "model.gguf"
        export_gguf.export(root, out, "f16")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            export_gguf.main(["--check", str(out)])
        text = buf.getvalue()
        self.assertIn("general.architecture = llama", text)
        self.assertIn("blk.0.ffn_down.weight", text)
        self.assertIn("F16", text)


if __name__ == "__main__":
    unittest.main()

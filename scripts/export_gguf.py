"""trid 체크포인트를 llama.cpp용 GGUF로 내보낸다 (reports/oss-model-size-and-inference.md §1.4).

    python scripts/export_gguf.py --run runs/<run> --output trid.gguf [--quant f32|f16|q8_0]
    python scripts/export_gguf.py --check trid.gguf

llama.cpp 체크아웃은 필요 없고 `gguf` 패키지(llama.cpp gguf-py, MIT)만 쓴다.

아키텍처 선택
- qk_norm=false → `llama`. trid의 RoPE는 인접 쌍(짝·홀) 회전이라 ggml NORM 모드와 같고, 묶인 임베딩은
  `output.weight`를 생략하면 llama.cpp가 `token_embd.weight`를 재사용한다. 가중치 치환 없음.
- qk_norm=true → `qwen3`. head 차원 RMSNorm → RoPE 순서가 trid와 같고 묶인 임베딩도 허용된다. 단 qwen3는
  NEOX RoPE(앞 절반·뒤 절반 쌍)라 q·k 투영 행과 q_norm·k_norm 가중치를 head 안에서
  (0,2,4,…,1,3,5,…) 순서로 치환해 넣는다. RMSNorm은 순서에 불변이므로 결과는 동일하다.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - 안내 메시지만
    raise SystemExit("gguf 패키지가 필요합니다: uv sync --extra gguf") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trid.model import Config  # noqa: E402
from trid.text import SPECIAL  # noqa: E402

QUANT_TYPES = {
    "f32": (gguf.GGMLQuantizationType.F32, gguf.LlamaFileType.ALL_F32),
    "f16": (gguf.GGMLQuantizationType.F16, gguf.LlamaFileType.MOSTLY_F16),
    "q8_0": (gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0),
}
RMS_EPS = 1e-5  # trid/model.py RMSNorm


def load_run(run):
    root = Path(run)
    cfg = Config(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    from safetensors.numpy import load_file
    weights = load_file(str(root / "model.safetensors"))
    tok = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    return cfg, weights, tok


def check_supported(cfg, tok):
    if cfg.num_intents:
        raise ValueError("num_intents>0 체크포인트는 GGUF로 내보낼 수 없습니다 (intent_head는 llama.cpp에 없음).")
    model = tok.get("model", {})
    pre = tok.get("pre_tokenizer") or {}
    if model.get("type") != "BPE" or pre.get("type") != "ByteLevel" or not pre.get("use_regex", True) \
            or tok.get("normalizer") is not None:
        raise ValueError("GPT-2 정규식 ByteLevel BPE(정규화 없음) 토크나이저만 지원합니다.")
    if cfg.heads * (cfg.dim // cfg.heads) != cfg.dim:
        raise ValueError("dim이 heads로 나누어떨어지지 않습니다.")
    if (cfg.dim // cfg.heads) % 2:
        raise ValueError("head 차원은 짝수여야 합니다 (RoPE).")


def neox_permutation(head_dim):
    """trid 인접 쌍 (2i, 2i+1) → NEOX 쌍 (i, i+head_dim/2). 행 인덱스 치환."""
    return np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])


def permute_heads(weight, head_dim, perm):
    """[heads*head_dim, ...] 행렬의 행을 head 단위로 치환."""
    shaped = weight.reshape(-1, head_dim, *weight.shape[1:])
    return shaped[:, perm].reshape(weight.shape)


def tokenizer_entries(cfg, tok):
    vocab = tok["model"]["vocab"]
    if len(vocab) != cfg.vocab_size:
        raise ValueError(f"토크나이저 어휘 {len(vocab)}개가 config vocab_size {cfg.vocab_size}와 다릅니다.")
    id_to_token = {i: t for t, i in vocab.items()}
    special = {a["content"] for a in tok.get("added_tokens", [])}
    tokens, types = [], []
    for i in range(cfg.vocab_size):
        if i not in id_to_token:
            raise ValueError(f"토큰 id {i}가 토크나이저에 없습니다.")
        tokens.append(id_to_token[i])
        types.append(gguf.TokenType.CONTROL if id_to_token[i] in special else gguf.TokenType.NORMAL)
    merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]]
    missing = [name for name in SPECIAL if name not in vocab]
    if missing:
        raise ValueError(f"특수 토큰 {missing}이 토크나이저에 없습니다.")
    return tokens, types, merges, {name: vocab[name] for name in SPECIAL}


def tensor_plan(cfg, weights):
    """(gguf 이름, numpy 배열) 목록. qwen3 경로는 q·k·q_norm·k_norm 행 치환."""
    dim, head_dim = cfg.dim, cfg.dim // cfg.heads
    kv_heads = cfg.heads if cfg.kv_heads is None else cfg.kv_heads
    kv_dim = kv_heads * head_dim
    names = gguf.TENSOR_NAMES
    T = gguf.MODEL_TENSOR
    perm = neox_permutation(head_dim) if cfg.qk_norm else None

    def w(key):
        return np.ascontiguousarray(weights[key].astype(np.float32))

    plan = [(names[T.TOKEN_EMBD] + ".weight", w("embedding.weight")),
            (names[T.OUTPUT_NORM] + ".weight", w("norm.weight"))]
    for i in range(cfg.layers):
        qkv = w(f"blocks.{i}.attn.qkv.weight")
        q, k, v = qkv[:dim], qkv[dim:dim + kv_dim], qkv[dim + kv_dim:]
        if perm is not None:
            q, k = permute_heads(q, head_dim, perm), permute_heads(k, head_dim, perm)
        layer = [(T.ATTN_NORM, w(f"blocks.{i}.attn_norm.weight")),
                 (T.ATTN_Q, np.ascontiguousarray(q)), (T.ATTN_K, np.ascontiguousarray(k)),
                 (T.ATTN_V, np.ascontiguousarray(v)),
                 (T.ATTN_OUT, w(f"blocks.{i}.attn.output.weight"))]
        if perm is not None:
            layer += [(T.ATTN_Q_NORM, w(f"blocks.{i}.attn.q_norm.weight")[perm]),
                      (T.ATTN_K_NORM, w(f"blocks.{i}.attn.k_norm.weight")[perm])]
        layer += [(T.FFN_NORM, w(f"blocks.{i}.ffn_norm.weight")),
                  (T.FFN_GATE, w(f"blocks.{i}.gate.weight")), (T.FFN_UP, w(f"blocks.{i}.up.weight")),
                  (T.FFN_DOWN, w(f"blocks.{i}.down.weight"))]
        plan += [(names[t].format(bid=i) + ".weight", a) for t, a in layer]
    return plan


def quantize(name, data, qtype):
    """convert_hf_to_gguf와 같은 규칙: 1차원(norm) 텐서와 블록 크기에 안 맞는 텐서는 F32 유지."""
    if qtype == gguf.GGMLQuantizationType.F32 or data.ndim == 1:
        return data, gguf.GGMLQuantizationType.F32
    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    if data.shape[-1] % block_size:
        return data, gguf.GGMLQuantizationType.F32
    return gguf.quants.quantize(data, qtype), qtype


def export(run, output, quant="f32", name=None):
    if quant not in QUANT_TYPES:
        raise ValueError(f"--quant는 {sorted(QUANT_TYPES)} 중 하나여야 합니다.")
    cfg, weights, tok = load_run(run)
    check_supported(cfg, tok)
    qtype, file_type = QUANT_TYPES[quant]
    arch = gguf.MODEL_ARCH.QWEN3 if cfg.qk_norm else gguf.MODEL_ARCH.LLAMA
    head_dim = cfg.dim // cfg.heads
    kv_heads = cfg.heads if cfg.kv_heads is None else cfg.kv_heads

    writer = gguf.GGUFWriter(str(output), gguf.MODEL_ARCH_NAMES[arch])
    writer.add_name(name or Path(run).resolve().name)
    writer.add_context_length(cfg.context_length)
    writer.add_embedding_length(cfg.dim)
    writer.add_block_count(cfg.layers)
    writer.add_feed_forward_length(cfg.hidden_dim)
    writer.add_head_count(cfg.heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    writer.add_vocab_size(cfg.vocab_size)
    writer.add_file_type(file_type)

    tokens, types, merges, ids = tokenizer_entries(cfg, tok)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(ids["<|bos|>"])
    writer.add_eos_token_id(ids["<|eos|>"])
    writer.add_pad_token_id(ids["<|pad|>"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    for tensor_name, data in tensor_plan(cfg, weights):
        quantized, dtype = quantize(tensor_name, data, qtype)
        writer.add_tensor(tensor_name, quantized, raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"output": str(output), "arch": gguf.MODEL_ARCH_NAMES[arch], "quant": quant,
            "bytes": Path(output).stat().st_size, "sha256": sha256(output)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path):
    reader = gguf.GGUFReader(str(path))
    meta = {}
    for key, field in reader.fields.items():
        if key.startswith("tokenizer.ggml.") and key.split(".")[-1] in ("tokens", "token_type", "merges", "scores"):
            meta[key] = f"[{len(field.data)} entries]"
        elif len(field.types) == 1 and field.types[0] == gguf.GGUFValueType.STRING:
            meta[key] = bytes(field.parts[field.data[0]]).decode("utf-8")
        elif field.types and field.types[0] != gguf.GGUFValueType.ARRAY:
            meta[key] = field.parts[field.data[0]].tolist()[0]
        else:
            meta[key] = f"[{len(field.data)} entries]"
    tensors = [{"name": t.name, "shape": [int(d) for d in reversed(t.shape.tolist())],
                "type": t.tensor_type.name, "bytes": int(t.n_bytes)} for t in reader.tensors]
    return {"metadata": meta, "tensors": tensors,
            "tensor_count": len(tensors), "file_bytes": Path(path).stat().st_size}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="config.json · model.safetensors · tokenizer.json이 있는 디렉터리")
    p.add_argument("--output", help="쓸 .gguf 경로")
    p.add_argument("--quant", default="f32", choices=sorted(QUANT_TYPES))
    p.add_argument("--name", help="general.name (기본: run 디렉터리 이름)")
    p.add_argument("--check", metavar="GGUF", help="메타데이터·텐서 목록을 출력하고 끝낸다")
    args = p.parse_args(argv)
    if args.check:
        info = describe(args.check)
        for k, v in info["metadata"].items():
            print(f"{k} = {v}")
        print(f"\n{info['tensor_count']} tensors, {info['file_bytes'] / 2**20:.1f} MiB")
        for t in info["tensors"]:
            print(f"  {t['name']:28s} {t['type']:5s} {t['shape']}")
        return
    if not args.run or not args.output:
        p.error("--run과 --output이 필요합니다 (또는 --check).")
    result = export(args.run, args.output, args.quant, args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

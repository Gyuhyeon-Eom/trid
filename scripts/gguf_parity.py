"""GGUF 내보내기 동일성 측정: llama.cpp(CPU) greedy 출력을 trid.runtime.generate와 비교한다.

    python scripts/gguf_parity.py --run /tmp/trid-base-snap --llama-bin /tmp/llama.cpp/build/bin \
        --output reports/gguf-export-parity.json

f32는 토큰 id 열이 정확히 같아야 하고 q8_0은 일치율을 보고한다. 파일 크기, 최대 RSS, tok/s도 llama.cpp 쪽에서 기록한다.
llama.cpp는 CPU 전용 빌드(`cmake -DGGML_METAL=OFF -DLLAMA_USE_PREBUILT_UI=OFF`)의 `llama-server`를 쓴다.
`/completion`에 `temperature=0, return_tokens=true`로 요청하면 생성 토큰 id가 그대로 돌아와 재토큰화 없이 비교할 수 있다.
"""
import argparse
from dataclasses import asdict
import json
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import export_gguf  # noqa: E402
from trid.model import Config, Trid  # noqa: E402
from trid.runtime import generate, load_run  # noqa: E402
from trid.text import read_rows  # noqa: E402

MAX_TOKENS = 32
THREADS = 4
PROMPTS = [
    ("plain", "대한민국의 수도는"),
    ("law", "제22조(과태료) ① 다음 각 호의 어느 하나에 해당하는 자에게는"),
]


def torch_greedy_ids(model, tok, prompt, max_tokens):
    """runtime.generate와 같은 규칙(금지 토큰, eos 정지)의 greedy 루프. 토큰 id 비교용."""
    ids = [tok.token_to_id("<|bos|>")] + tok.encode(prompt).ids
    eos = tok.token_to_id("<|eos|>")
    forbidden = [tok.token_to_id(t) for t in ["<|pad|>", "<|bos|>", "<|user|>", "<|assistant|>"]]
    x, cache, out = torch.tensor([ids]), None, []
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits, cache = model(x, cache=cache, use_cache=True)
            scores = logits[0, -1].clone()
            scores[forbidden] = -float("inf")
            token = int(scores.argmax())
            out.append(token)
            if token == eos:
                break
            x = torch.tensor([[token]])
    return ids, out


class LlamaServer:
    def __init__(self, binary, gguf_path, context_length):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        cmd = ["nice", "-n", "10", str(binary), "-m", str(gguf_path), "--port", str(self.port), "--host", "127.0.0.1",
               "-t", str(THREADS), "-c", str(context_length), "-ngl", "0", "--no-warmup", "--log-disable"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    if json.load(r).get("status") == "ok":
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server가 종료되었습니다.")
            time.sleep(0.1)
        raise RuntimeError("llama-server가 준비되지 않았습니다.")

    def complete(self, prompt, max_tokens, n_probs=0):
        body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "return_tokens": True,
                           "cache_prompt": False, "samplers": ["temperature"], "n_probs": n_probs}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def rss_bytes(self):
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True, text=True).stdout
        return int(out.strip()) * 1024

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=30)


TOP_N = 5


TIE_GAP = 1e-3  # logprob; llama.cpp가 확률을 주지 않는 위치(불완전 UTF-8 조각)에서 동률 판정에 쓰는 PyTorch 1·2위 차


def teacher_forced_check(model, server, prompt_ids, continuation):
    """prompt+continuation[:k] 접두마다 다음 토큰을 PyTorch와 llama.cpp에서 비교한다.
    무작위 작은 모델은 greedy 동률이 흔해 토큰 열 대신 분포를 본다: llama-server가 top-5 logprob를 주면 절대차를
    재고, argmax가 다르면 PyTorch의 1·2위 차가 그 오차(없으면 TIE_GAP) 이내인 '동률 뒤집힘'인지 따로 센다.
    (llama-server는 생성 토큰이 불완전한 UTF-8 조각이면 completion_probabilities를 주지 않는다.)"""
    max_diff, mismatches, tie_flips, with_probs = 0.0, 0, 0, 0
    for k in range(len(continuation)):
        prefix = prompt_ids + continuation[:k]
        with torch.inference_mode():
            logprobs = model(torch.tensor([prefix]))[0][0, -1].log_softmax(-1)
        res = server.complete(prefix, 1, n_probs=TOP_N)
        llama_token = res["tokens"][0]
        probs = res.get("completion_probabilities")
        tolerance = TIE_GAP
        if probs:
            with_probs += 1
            diff = max(abs(float(logprobs[t["id"]]) - t["logprob"]) for t in probs[0]["top_logprobs"])
            max_diff = max(max_diff, diff)
            tolerance = max(2 * diff, 1e-4)
        torch_top2 = logprobs.topk(2)
        if int(torch_top2.indices[0]) != llama_token:
            gap = float(torch_top2.values[0] - torch_top2.values[1])
            if gap <= tolerance and llama_token == int(torch_top2.indices[1]):
                tie_flips += 1
            else:
                mismatches += 1
    return {"positions": len(continuation), "positions_with_probs": with_probs,
            "max_abs_logprob_diff": round(max_diff, 6), "argmax_mismatches": mismatches, "tie_flips": tie_flips}


def agreement(reference, candidate):
    n = min(len(reference), len(candidate))
    prefix = 0
    while prefix < n and reference[prefix] == candidate[prefix]:
        prefix += 1
    positional = sum(a == b for a, b in zip(reference, candidate))
    return {"prefix_match": prefix, "positional_match": positional, "reference_len": len(reference),
            "candidate_len": len(candidate), "exact": list(reference) == list(candidate)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--llama-bin", required=True, help="llama-server가 있는 디렉터리")
    p.add_argument("--output", required=True)
    p.add_argument("--eval", default=str(ROOT / "data/korean-rag-eval-v1/test.jsonl"))
    p.add_argument("--eval-id", default="A-001")
    p.add_argument("--workdir", default=None, help="GGUF를 둘 디렉터리 (기본: 임시)")
    args = p.parse_args(argv)

    row = next(r for r in read_rows(args.eval) if r["id"] == args.eval_id)
    prompts = PROMPTS + [(f"rag:{args.eval_id}", row["prompt"])]
    torch.set_num_threads(THREADS)
    model, tok, device = load_run(args.run, "cpu")

    reference = {}
    for name, prompt in prompts:
        ids, out = torch_greedy_ids(model, tok, prompt, MAX_TOKENS)
        gen = generate(model, tok, prompt, device, max_tokens=MAX_TOKENS, raw_text=True)
        decoded = tok.decode(out, skip_special_tokens=True).strip()
        if gen["answer"] != decoded or gen["generated_tokens_including_eos"] != len(out):
            raise RuntimeError(f"{name}: greedy 루프와 runtime.generate가 다릅니다.")
        reference[name] = {"prompt_tokens": len(ids), "ids": out, "text": decoded,
                           "torch_tokens_per_second_including_prefill": gen["tokens_per_second_including_prefill"]}

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="trid-gguf-"))
    workdir.mkdir(parents=True, exist_ok=True)
    binary = Path(args.llama_bin) / "llama-server"
    formats = {}
    for quant in ("f32", "q8_0"):
        path = workdir / f"trid-{quant}.gguf"
        start = time.perf_counter()
        info = export_gguf.export(args.run, path, quant)
        info["export_seconds"] = round(time.perf_counter() - start, 2)
        info["mib"] = round(info["bytes"] / 2**20, 1)
        server = LlamaServer(binary, path, model.cfg.context_length)
        try:
            results = {}
            for name, prompt in prompts:
                res = server.complete(prompt, MAX_TOKENS)
                ref = reference[name]
                timings = res["timings"]
                results[name] = {"text": res["content"].strip(), "ids": res["tokens"],
                                 "prompt_tokens": timings["prompt_n"],
                                 "prompt_tokens_match": timings["prompt_n"] == ref["prompt_tokens"],
                                 "prompt_tok_s": round(timings["prompt_per_second"], 1),
                                 "eval_tok_s": round(timings["predicted_per_second"], 1),
                                 "text_exact": res["content"].strip() == ref["text"],
                                 "tokens": agreement(ref["ids"], res["tokens"])}
            info["max_rss_bytes"] = server.rss_bytes()
        finally:
            server.close()
        info["prompts"] = results
        info["all_text_exact"] = all(r["text_exact"] for r in results.values())
        info["all_tokens_exact"] = all(r["tokens"]["exact"] for r in results.values())
        info["all_prompt_tokens_match"] = all(r["prompt_tokens_match"] for r in results.values())
        total_ref = sum(len(reference[n]["ids"]) for n, _ in prompts)
        info["token_agreement"] = round(sum(r["tokens"]["positional_match"] for r in results.values()) / total_ref, 4)
        info["mean_eval_tok_s"] = round(sum(r["eval_tok_s"] for r in results.values()) / len(results), 1)
        info["mean_prompt_tok_s"] = round(sum(r["prompt_tok_s"] for r in results.values()) / len(results), 1)
        info["max_rss_mib"] = round(info["max_rss_bytes"] / 2**20, 1)
        formats[quant] = info

    # qk_norm(+GQA) 구성은 실제 체크포인트가 없어 같은 토크나이저를 쓰는 무작위 작은 모델로 qwen3 경로를 검증한다.
    tiny_root = workdir / "tiny-qk-norm"
    tiny_root.mkdir(exist_ok=True)
    tiny_cfg = Config(vocab_size=model.cfg.vocab_size, dim=64, heads=4, kv_heads=2, layers=2, hidden_dim=128,
                      context_length=model.cfg.context_length, qk_norm=True)
    torch.manual_seed(0)
    tiny = Trid(tiny_cfg).eval()
    (tiny_root / "config.json").write_text(json.dumps(asdict(tiny_cfg)))
    (tiny_root / "tokenizer.json").write_bytes((Path(args.run) / "tokenizer.json").read_bytes())
    save_file({k: v.contiguous() for k, v in tiny.state_dict().items()}, str(tiny_root / "model.safetensors"))
    tiny_path = tiny_root / "tiny.gguf"
    tiny_info = export_gguf.export(tiny_root, tiny_path, "f32")
    server = LlamaServer(binary, tiny_path, tiny_cfg.context_length)
    try:
        tiny_results = {}
        for name, prompt in prompts:
            ids, out = torch_greedy_ids(tiny, tok, prompt, 16)
            tiny_results[name] = teacher_forced_check(tiny, server, ids, out)
    finally:
        server.close()
    qwen3_check = {"config": asdict(tiny_cfg), "arch": tiny_info["arch"], "prompts": tiny_results,
                   "max_abs_logprob_diff": max(r["max_abs_logprob_diff"] for r in tiny_results.values()),
                   "argmax_mismatches": sum(r["argmax_mismatches"] for r in tiny_results.values()),
                   "tie_flips": sum(r["tie_flips"] for r in tiny_results.values())}
    # 가중치를 키우지 않은 무작위 모델에서 llama 아키텍처도 같은 절차로 1e-4 수준이므로 1e-3을 통과 기준으로 둔다.
    qwen3_check["pass"] = qwen3_check["argmax_mismatches"] == 0 and qwen3_check["max_abs_logprob_diff"] < 1e-3

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "run": str(Path(args.run).resolve()), "max_tokens": MAX_TOKENS,
        "llama_cpp": {"binary": str(binary), "commit": git_commit(binary),
                      "build": "GGML_METAL=OFF (CPU only, Accelerate BLAS)", "threads": THREADS,
                      "request": {"temperature": 0, "samplers": ["temperature"], "return_tokens": True, "cache_prompt": False}},
        "machine": {"platform": platform.platform(), "machine": platform.machine(), "torch": torch.__version__},
        "reference_torch_fp32": reference,
        "formats": formats,
        "qk_norm_qwen3_tiny_check": qwen3_check,
        "f32_exact": formats["f32"]["all_tokens_exact"] and formats["f32"]["all_text_exact"],
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"tokens_exact": v["all_tokens_exact"], "token_agreement": v["token_agreement"],
                          "mib": v["mib"], "eval_tok_s": v["mean_eval_tok_s"], "prompt_tok_s": v["mean_prompt_tok_s"],
                          "rss_mib": v["max_rss_mib"]} for k, v in formats.items()}, ensure_ascii=False, indent=2))
    print("qk_norm→qwen3 tiny check:", {k: qwen3_check[k] for k in ("pass", "max_abs_logprob_diff", "argmax_mismatches", "tie_flips")})
    if not report["f32_exact"]:
        raise SystemExit("f32 GGUF 출력이 PyTorch와 다릅니다.")


def git_commit(binary):
    try:
        return subprocess.run(["git", "-C", str(Path(binary).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


if __name__ == "__main__":
    main()

#!/bin/bash
# PILOT-SFT-001 evaluation, sequential (parallel llama.cpp runs on one Mac contend badly): pilot_dev diagnostics for the
# original and the tuned model (-c 8192, same settings), then the fixed suite for each (-c 4096, baseline settings).
set -euo pipefail
cd "$(dirname "$0")/.."
S=runs/tools/llama.cpp-8ca4060/llama-server
BASE=runs/open-weight-base/Qwen3-1.7B-rev70d244cc-localconv-Q8_0.gguf
TUNED=runs/pilot-sft-001-a-qwen3-1p7b/pilot-sft-001-a-qwen3-1p7b.q8_0.gguf
PILOT=reports/coordination/C-017/pilot/pilot.jsonl
OUT=reports/open-weight-pilot
mkdir -p "$OUT" .eval-logs
dev() {  # gguf label
  "$S" -m "$1" --host 127.0.0.1 --port 51916 -c 8192 -np 1 -t 6 -tb 6 --jinja > ".eval-logs/$2-8192.log" 2>&1 &
  local sp=$!
  until curl -sf http://127.0.0.1:51916/health >/dev/null; do kill -0 "$sp" 2>/dev/null || { echo "server died"; exit 1; }; sleep 1; done
  .venv/bin/python scripts/evaluate_pilot_dev.py --url http://127.0.0.1:51916 --model-file "$1" --label "$2" \
    --pilot "$PILOT" --output "$OUT/pilot-dev-$2.json" > ".eval-logs/$2-dev.log" 2>&1
  kill "$sp"; wait "$sp" 2>/dev/null || true
}
[ -f "$OUT/pilot-dev-qwen3-1.7b-base-localconv.json" ] || dev "$BASE" qwen3-1.7b-base-localconv
[ -f "$OUT/pilot-dev-qwen3-1.7b-pilot-sft-001.json" ] || dev "$TUNED" qwen3-1.7b-pilot-sft-001
echo "pilot_dev done $(date)"
[ -f "$OUT/counterfactual-qwen3-1.7b-base-localconv.json" ] || \
  scripts/eval_open_weight_suite.sh "$BASE" Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e qwen3-1.7b-base-localconv 51914 "$OUT"
[ -f "$OUT/counterfactual-qwen3-1.7b-pilot-sft-001.json" ] || \
  scripts/eval_open_weight_suite.sh "$TUNED" pilot-sft-001-a-qwen3-1p7b 82f09136206a7be7-merged qwen3-1.7b-pilot-sft-001 51915 "$OUT"
echo "all done $(date)"

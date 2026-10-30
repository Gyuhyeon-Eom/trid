#!/bin/bash
# Fixed development evaluation of one GGUF chat model with the baseline runtime (llama.cpp 8ca4060, -c 4096, greedy,
# no thinking): RAG310 (3 conditions), institution probe (3 conditions), public-dialogue prompts, counterfactual.
# Optional C pilot_dev diagnostics (-c 8192, both models alike) when PILOT_DEV is set.
#   eval_open_weight_suite.sh <gguf> <model-id> <revision> <label> <port> [outdir]
set -euo pipefail
GGUF=$1; MODEL_ID=$2; REV=$3; LABEL=$4; PORT=$5; OUT=${6:-reports/open-weight-pilot}
cd "$(dirname "$0")/.."
SERVER=runs/tools/llama.cpp-8ca4060/llama-server
mkdir -p "$OUT" .eval-logs
serve() {
  "$SERVER" -m "$GGUF" --host 127.0.0.1 --port "$PORT" -c "$1" -np 1 -t 4 -tb 4 --jinja > ".eval-logs/$LABEL-$1.log" 2>&1 &
  SP=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do kill -0 "$SP" 2>/dev/null || { echo "server died"; exit 1; }; sleep 1; done
}
E=(.venv/bin/python scripts/evaluate_external_chat.py --url "http://127.0.0.1:$PORT" --model-file "$GGUF"
   --model-id "$MODEL_ID" --revision "$REV" --family qwen3)
serve 4096
trap 'kill $SP 2>/dev/null' EXIT
"${E[@]}" --data data/korean-rag-eval-v1 --output "$OUT/rag310-$LABEL.json" > ".eval-logs/$LABEL-rag.log"
"${E[@]}" --data data/institution-qa-frozen-01 --output "$OUT/probe-$LABEL.json" > ".eval-logs/$LABEL-probe.log"
"${E[@]}" --freeform-prompts reports/public-dialogue-01-after.json --output "$OUT/public-dialogue-$LABEL.json"
"${E[@]}" --freeform-prompts reports/open-weight-baseline/counterfactual-prompts.json --output "$OUT/counterfactual-$LABEL.json"
kill "$SP"; wait "$SP" 2>/dev/null || true
if [ -n "${PILOT_DEV:-}" ]; then
  serve 8192
  .venv/bin/python scripts/evaluate_pilot_dev.py --url "http://127.0.0.1:$PORT" --model-file "$GGUF" --label "$LABEL" \
    --pilot "$PILOT_DEV" --output "$OUT/pilot-dev-$LABEL.json"
fi
rm -f "$OUT"/*.partial.jsonl
echo "$LABEL done"

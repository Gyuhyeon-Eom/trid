#!/bin/bash
# SageMaker entrypoint: span-scoring diagnostic (scripts/evaluate_rag_eval.py --decoding span) on an existing SFT run,
# all RAG310 items, both span normalizations, with wall time and per-item candidate counts. A diagnostic only: the
# evidence/free strict metrics stay the primary development metrics.
#   SFT_S3  s3 prefix of the SFT run directory (config.json, tokenizer.json, model.safetensors)
#   RUN     output name
set -euo pipefail
: "${RUN:?}" "${SFT_S3:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work
pip install -q tokenizers safetensors
mkdir -p data /opt/sft
ln -s /opt/ml/input/data/data/korean-rag-eval-v1 data/korean-rag-eval-v1
OUT=/opt/ml/checkpoints/$RUN/reports
mkdir -p "$OUT"
python - "$SFT_S3" <<'PY'
import sys, boto3
bucket, _, key = sys.argv[1].removeprefix('s3://').partition('/')
s3 = boto3.client('s3')
for name in ('config.json', 'tokenizer.json', 'model.safetensors'):
    s3.download_file(bucket, key + name, f'/opt/sft/{name}')
    print('downloaded', key + name)
PY
sha256sum /opt/sft/model.safetensors | tee "$OUT/sft-weights.sha256"
nvidia-smi -L
python - "$OUT/span-candidates.json" <<'PY'
import json, sys
from statistics import median
from trid.prompts import rag_passages
from trid.span_scoring import candidates
rows = [json.loads(l) for l in open('data/korean-rag-eval-v1/test.jsonl')]
counts = {r['id']: len(candidates(rag_passages(r['prompt']))) for r in rows}
v = sorted(counts.values())
json.dump({'per_item': counts, 'total': sum(v), 'median': median(v), 'max': v[-1]}, open(sys.argv[1], 'w'))
print('candidates total', sum(v), 'median', median(v), 'max', v[-1])
PY
for norm in sum mean; do
  report="$OUT/rag310-$RUN-span-$norm.json"
  [ -f "$report" ] && continue
  start=$(date +%s.%N)
  python scripts/evaluate_rag_eval.py --run /opt/sft --decoding span --span-norm "$norm" --device cuda --output "$report"
  end=$(date +%s.%N)
  python -c "import json; json.dump({'span_norm': '$norm', 'wall_seconds': $end - $start, 'device': 'cuda', 'items': 310}, open('$OUT/timing-span-$norm.json', 'w'))"
done
echo done

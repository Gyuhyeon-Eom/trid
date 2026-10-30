#!/bin/bash
# SageMaker entrypoint: the fixed development evaluation (RAG310 evidence + free, KLUE dev question sensitivity) of one
# saved weight file, without training. Used for SFT campaign milestones and the validation-selected weights.
#   RUN_S3      s3 prefix of the SFT run directory (config.json, tokenizer.json)
#   WEIGHTS_S3  s3 key prefix of the directory holding model.safetensors (+ state.json)
#   LABEL       output label, e.g. step-00036000 or best-valid
set -euo pipefail
: "${RUN:?}" "${RUN_S3:?}" "${WEIGHTS_S3:?}" "${LABEL:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work
pip install -q tokenizers safetensors
mkdir -p data /opt/w
for d in korean-rag-eval-v1 klue-mrc-source; do ln -s "/opt/ml/input/data/data/$d" "data/$d"; done
OUT=/opt/ml/checkpoints/$RUN/$LABEL
mkdir -p "$OUT"
python - "$RUN_S3" "$WEIGHTS_S3" <<'PY'
import sys, boto3
s3 = boto3.client('s3')
for prefix, names in ((sys.argv[1], ('config.json', 'tokenizer.json')), (sys.argv[2], ('model.safetensors',))):
    bucket, _, key = prefix.removeprefix('s3://').partition('/')
    for name in names:
        s3.download_file(bucket, key + name, f'/opt/w/{name}')
PY
sha256sum /opt/w/model.safetensors | sed 's|/opt/w/||' | tee "$OUT/weights.sha256"
for decoding in evidence free; do
  report="$OUT/rag310-$decoding.json"
  [ -f "$report" ] || python scripts/evaluate_rag_eval.py --run /opt/w --decoding "$decoding" --device cuda --output "$report"
done
[ -f "$OUT/question-sensitivity.json" ] || python scripts/evaluate_question_sensitivity.py --run /opt/w --device cuda \
  --output "$OUT/question-sensitivity.json"
echo done

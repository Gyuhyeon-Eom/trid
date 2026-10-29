#!/bin/bash
# SageMaker entrypoint: SFT a pretrained trid checkpoint on public-dialogue-mix-04, then run the development
# evaluations (RAG310 evidence + free, KLUE dev question sensitivity). Launched by scripts/launch_sm_pretrain.py
# --entry sm_sft_eval.sh. Channels: code -> /opt/ml/input/data/code, data -> /opt/ml/input/data/data (holds
# public-dialogue-mix-04/, korean-rag-eval-v1/, klue-mrc-source/dev.json). Outputs live under the checkpoint path.
#   INIT_S3   s3 prefix of the pretraining run directory (holds config.json, tokenizer.json, checkpoints/)
#   INIT_STEP pretraining step whose checkpoints/step-XXXXXXXX/model.safetensors is used (the fixed final step)
#   RUN       output run name; SFT_STEPS default 1200
set -euo pipefail
: "${RUN:?}" "${INIT_S3:?}" "${INIT_STEP:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work
pip install -q tokenizers safetensors
mkdir -p data
for d in public-dialogue-mix-04 korean-rag-eval-v1 klue-mrc-source; do ln -s "/opt/ml/input/data/data/$d" "data/$d"; done
OUT=/opt/ml/checkpoints/$RUN
mkdir -p "$OUT/reports" /opt/init
python - "$INIT_S3" "$INIT_STEP" <<'PY'
import sys, boto3
prefix, step = sys.argv[1], int(sys.argv[2])
bucket, _, key = prefix.removeprefix('s3://').partition('/')
s3 = boto3.client('s3')
for src, dst in ((f'{key}config.json', 'config.json'), (f'{key}tokenizer.json', 'tokenizer.json'),
                 (f'{key}checkpoints/step-{step:08d}/model.safetensors', 'model.safetensors'),
                 (f'{key}checkpoints/step-{step:08d}/state.json', 'pretrain-state.json')):
    s3.download_file(bucket, src, f'/opt/init/{dst}')
    print('downloaded', src)
PY
python -c "import json; s=json.load(open('/opt/init/pretrain-state.json')); assert s['step']==int('$INIT_STEP'), s; print('init state', s)"
sha256sum /opt/init/model.safetensors | tee "$OUT/reports/init-weights.sha256"
if [ ! -f "$OUT/sft/training.json" ] || ! grep -q '"status": "complete"' "$OUT/sft/training.json"; then
  rm -rf "$OUT/sft"
  python scripts/train_public_dialogue.py train --data data/public-dialogue-mix-04 --run "$OUT/sft" --init-run /opt/init \
    --steps "${SFT_STEPS:-1200}" --device cuda
fi
for decoding in evidence free; do
  report="$OUT/reports/rag310-$RUN-$decoding.json"
  [ -f "$report" ] || python scripts/evaluate_rag_eval.py --run "$OUT/sft" --decoding "$decoding" --device cuda --output "$report"
done
report="$OUT/reports/question-sensitivity-$RUN.json"
[ -f "$report" ] || python scripts/evaluate_question_sensitivity.py --run "$OUT/sft" --device cuda --output "$report"
sha256sum "$OUT/sft/model.safetensors" | tee "$OUT/reports/sft-weights.sha256"
echo done

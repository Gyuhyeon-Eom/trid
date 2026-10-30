#!/bin/bash
# SageMaker entrypoint: long SFT with saved, resumable checkpoints, then the same development evaluation at every
# saved step and at the validation-loss-selected weights. Training arguments equal scripts/train_public_dialogue.py
# `train` (mix-04, batch 8, micro 2, lr 1e-4, seed 271030, eval every 100 on 64 valid rows, context 1024, example
# normalization); only `steps` and `save_steps` are set here. Run directory lives under the checkpoint path, so a new
# job with the same checkpoint S3 URI resumes from the latest saved step.
#   INIT_S3, INIT_STEP  pretraining run prefix and its fixed final step
#   RUN, SFT_STEPS, SAVE_STEPS (comma separated)
set -euo pipefail
: "${RUN:?}" "${INIT_S3:?}" "${INIT_STEP:?}" "${SFT_STEPS:?}" "${SAVE_STEPS:?}"
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
PY
python -c "import json; s=json.load(open('/opt/init/pretrain-state.json')); assert s['step']==int('$INIT_STEP'), s; print('init state', s)"
sha256sum /opt/init/model.safetensors | tee "$OUT/reports/init-weights.sha256"
if [ -d "$OUT/sft/checkpoints" ]; then echo "resuming; saved:"; ls "$OUT/sft/checkpoints"; fi
python - <<PY
import torch
from trid.training import train
torch.set_num_threads(2)
train('data/public-dialogue-mix-04', '$OUT/sft', init_run='/opt/init', steps=$SFT_STEPS, batch_size=8,
      micro_batch_size=2, lr=0.0001, device='cuda', seed=271030, eval_every=100, validation_limit=64,
      context_length=1024, loss_normalization='example', save_steps=[int(s) for s in '$SAVE_STEPS'.split(',')], resume=True)
PY
evaluate() {  # $1 label, $2 weights file
  local dir=/opt/eval/$1
  mkdir -p "$dir"
  cp "$OUT/sft/config.json" "$OUT/sft/tokenizer.json" "$dir/"
  cp "$2" "$dir/model.safetensors"
  sha256sum "$dir/model.safetensors" | sed "s|$dir/||" > "$OUT/reports/weights-$1.sha256"
  for decoding in evidence free; do
    local report="$OUT/reports/rag310-$RUN-$1-$decoding.json"
    [ -f "$report" ] || python scripts/evaluate_rag_eval.py --run "$dir" --decoding "$decoding" --device cuda --output "$report"
  done
  local report="$OUT/reports/question-sensitivity-$RUN-$1.json"
  [ -f "$report" ] || python scripts/evaluate_question_sensitivity.py --run "$dir" --device cuda --output "$report"
  rm -rf "$dir"
}
for s in ${SAVE_STEPS//,/ }; do
  evaluate "step-$(printf %06d "$s")" "$OUT/sft/checkpoints/step-$(printf %06d "$s")/model.safetensors"
done
evaluate best-valid "$OUT/sft/model.safetensors"
python -c "import json; t=json.load(open('$OUT/sft/training.json')); print({k: t.get(k) for k in ('status','best_step','best_validation_loss','completed_steps','resumed_from_step')})"
echo done

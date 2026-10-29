#!/bin/bash
# SageMaker entrypoint for trid from-scratch pretraining (scripts/launch_sm_pretrain.py creates the job).
# Channels: code -> /opt/ml/input/data/code (trid/, scripts/, configs/), data -> /opt/ml/input/data/data (token folder).
# The run directory lives under the CheckpointConfig local path, so SageMaker syncs it to S3 while training and
# restores it at job start: rerunning a job with the same checkpoint S3 URI resumes from latest.json.
set -euo pipefail
: "${RUN:?}" "${CONFIG:?}" "${STEPS:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work  # scripts/*.py import the trid package from the copied code channel
pip install -q tokenizers safetensors
CKPT=/opt/ml/checkpoints
mkdir -p "$CKPT/monitor"
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 \
  > "$CKPT/monitor/${TRAINING_JOB_NAME:-job}-gpu.csv" &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT
if [ -f "$CKPT/$RUN/training.json" ]; then
  echo "restored run directory from checkpoint S3:"; ls "$CKPT/$RUN"; cat "$CKPT/$RUN/latest.json"
fi
python scripts/train_medium_cpt_gpu.py --config "configs/$CONFIG" --data /opt/ml/input/data/data \
  --output "$CKPT/$RUN" --steps "$STEPS" --stop-after "${STOP_AFTER:-$STEPS}" --max-sessions "${MAX_SESSIONS:-0}" \
  --batch-size "${BATCH:-16}" --accumulate "${ACCUM:-4}" --lr "${LR:-3e-4}" --seed "${SEED:-291101}" \
  --eval-every "${EVAL_EVERY:-51}" --checkpoint-every "${CHECKPOINT_EVERY:-153}" --validation-batches "${VALID_BATCHES:-8}" \
  --sampling "${SAMPLING:-epoch}" --schedule "${SCHEDULE:-wsd}" --decay-fraction "${DECAY_FRACTION:-0.2}" \
  --device cuda --precision bf16 --compile --stop-flag "$CKPT/$RUN.stop"
python - "$CKPT/$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
meta = json.loads((run / 'training.json').read_text())
print(json.dumps({k: meta.get(k) for k in ('status', 'completed_steps', 'tokens_seen', 'parameters', 'best_step',
                                            'best_validation_loss', 'training_seconds')}))
PY

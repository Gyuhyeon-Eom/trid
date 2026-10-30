#!/bin/bash
# SageMaker entrypoint: one segment of a long SFT campaign (trid.training.train with a fixed full-length schedule).
# Training arguments equal scripts/train_public_dialogue.py `train` (mix-04, batch 8, micro 2, lr 1e-4, seed 271030,
# validation every 100 steps on 64 rows, context 1024, example normalization). The run directory lives under the
# checkpoint path: SageMaker restores it from S3 at job start, train(resume=True) continues from the newest recovery
# checkpoint (weights, optimizer, sampling RNG, best-so-far, metrics) and the segment ends after SEGMENT_SECONDS
# (or SESSION_STEPS) with a fresh recovery checkpoint. scripts/drive_sft_campaign.py chains the segments.
#   RUN, INIT_S3, INIT_STEP, SFT_STEPS, SAVE_STEPS (comma list), CHECKPOINT_EVERY, SEGMENT_SECONDS | SESSION_STEPS
set -euo pipefail
: "${RUN:?}" "${INIT_S3:?}" "${INIT_STEP:?}" "${SFT_STEPS:?}" "${SAVE_STEPS:?}" "${CHECKPOINT_EVERY:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work
pip install -q tokenizers safetensors
mkdir -p data
ln -s /opt/ml/input/data/data/public-dialogue-mix-04 data/public-dialogue-mix-04
OUT=/opt/ml/checkpoints/$RUN
mkdir -p "$OUT" /opt/init
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
python -c "import json; s=json.load(open('/opt/init/pretrain-state.json')); assert s['step']==int('$INIT_STEP'), s"
[ -d "$OUT/sft/checkpoints" ] && { echo "restored recovery checkpoints:"; ls "$OUT/sft/checkpoints"; }
nvidia-smi -L
python - <<PY
import json, torch
from trid.training import train
torch.set_num_threads(2)
meta = train('data/public-dialogue-mix-04', '$OUT/sft', init_run='/opt/init', steps=$SFT_STEPS, batch_size=8,
             micro_batch_size=2, lr=0.0001, device='cuda', seed=271030, eval_every=100, validation_limit=64,
             context_length=1024, loss_normalization='example', save_steps=[int(s) for s in '$SAVE_STEPS'.split(',')],
             checkpoint_every=$CHECKPOINT_EVERY, keep_recovery=2, resume=True,
             session_steps=${SESSION_STEPS:-None}, max_seconds=${SEGMENT_SECONDS:-None})
print('segment end', json.dumps({k: meta.get(k) for k in ('status', 'completed_steps', 'resumed_from_step', 'best_step',
                                                         'best_validation_loss', 'training_seconds')}))
PY

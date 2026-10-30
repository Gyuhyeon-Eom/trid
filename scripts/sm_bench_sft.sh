#!/bin/bash
# SageMaker entrypoint: scripts/bench_sft_speed.py on each A-003 final pretraining checkpoint (no weights saved).
#   ARMS  space-separated list of pretraining run names, e.g. "a003-1b-390m-mha a003-1b-91m-mha"
set -euo pipefail
: "${RUN:?}" "${ARMS:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work
pip install -q tokenizers safetensors
mkdir -p data
ln -s /opt/ml/input/data/data/public-dialogue-mix-04 data/public-dialogue-mix-04
OUT=/opt/ml/checkpoints/$RUN
mkdir -p "$OUT"
nvidia-smi -L
for arm in $ARMS; do
  mkdir -p "/opt/init/$arm"
  python - "$arm" <<'PY'
import sys, boto3
arm = sys.argv[1]
key = f'corpus/runs/pretrain/{arm}/{arm}/'
s3 = boto3.client('s3')
for src, dst in (('config.json', 'config.json'), ('tokenizer.json', 'tokenizer.json'),
                 ('checkpoints/step-00015259/model.safetensors', 'model.safetensors')):
    s3.download_file('trid-training-148761639313', key + src, f'/opt/init/{arm}/{dst}')
PY
  python scripts/bench_sft_speed.py --init "/opt/init/$arm" --data data/public-dialogue-mix-04 --output "$OUT/bench-$arm.json"
done
echo done

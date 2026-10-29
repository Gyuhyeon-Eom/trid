#!/bin/bash
set -euo pipefail
cd /home/ubuntu/trid
source /home/ubuntu/trid-venv/bin/activate
export PYTHONPATH=/home/ubuntu/trid
EVAL_PYTHON=/home/ubuntu/trid-eval-venv/bin/python
"$EVAL_PYTHON" -c 'import sys,torch,tokenizers,safetensors,numpy; assert sys.version_info[:2] == (3,12)'
B=s3://trid-training-148761639313
# Wait without changing the running trainer or its source.
while systemctl is-active --quiet trid-cpt || systemctl is-active --quiet trid-cpt-resumed; do sleep 60; done
python - <<'PY'
import json
m=json.load(open('runs/trid-korean-medium-cpt-v1/training.json'))
assert m['status']=='complete', m
PY
aws s3 sync runs/trid-korean-medium-cpt-v1 "$B/runs/trid-korean-medium-cpt-v1" --exclude 'checkpoints/*' --only-show-errors
for data in korean-rag-curriculum-v2 korean-rag-curriculum-v4 institution-qa-frozen-01; do
 aws s3 sync "$B/data/$data" "data/$data" --only-show-errors
done
mkdir -p reports
for backbone in cpt-v1 base-v1; do
 for head in v2 v1; do
  for data in v4 v2; do
   name=reader-factorial-01-$backbone-$head-$data
   "$EVAL_PYTHON" scripts/run_reader_trial.py --name "$name" --init-run "runs/trid-korean-medium-$backbone" --data "data/korean-rag-curriculum-$data" --head "$head"
  done
 done
done
echo 'EIGHT COMBINATIONS COMPLETE'

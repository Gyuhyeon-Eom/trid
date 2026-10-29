#!/bin/bash
set -euo pipefail
cd /home/ubuntu/trid
source /home/ubuntu/trid-venv/bin/activate
export PYTHONPATH=/home/ubuntu/trid
EVAL_PYTHON=/home/ubuntu/trid-eval-venv/bin/python
"$EVAL_PYTHON" -c 'import sys,torch,tokenizers,safetensors,numpy; assert sys.version_info[:2] == (3,12)'
B=s3://trid-training-148761639313
# Wait without changing the running trainer or its source.
while systemctl is-active --quiet trid-cpt || systemctl is-active --quiet trid-cpt-resumed || systemctl is-active --quiet trid-reader-factorial-01; do sleep 60; done
for b in base-v1 cpt-v1; do for h in v1 v2; do for d in v2 v4; do test -f "reports/reader-factorial-01-$b-$h-$d.archived"; done; done; done
python - <<'PY'
import json
m=json.load(open('runs/trid-korean-medium-cpt-v1/training.json'))
assert m['status']=='complete', m
PY
aws s3 sync runs/trid-korean-medium-cpt-v1 "$B/runs/trid-korean-medium-cpt-v1" --exclude 'checkpoints/*' --only-show-errors
for data in reader-ablation-without-tydi reader-ablation-without-synthetic-hard reader-ablation-without-klue-hard; do
 aws s3 sync "$B/data/$data" "data/$data" --only-show-errors
done
mkdir -p reports
for backbone in cpt-v1; do
 for head in v2; do
  for data in tydi synthetic-hard klue-hard; do
   name=reader-removal-01-$backbone-$head-without-$data
   "$EVAL_PYTHON" scripts/run_reader_trial.py --name "$name" --init-run "runs/trid-korean-medium-$backbone" --data "data/reader-ablation-without-$data" --head "$head"
  done
 done
done
echo 'THREE REMOVAL EXPERIMENTS COMPLETE'

#!/bin/bash
set -euo pipefail
cd /home/ubuntu/trid
source /home/ubuntu/trid-venv/bin/activate
export PYTHONPATH=/home/ubuntu/trid
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
   export name B
   if [ -f "reports/$name.archived" ]; then continue; fi
   # Never overwrite an incomplete prior attempt.
   if [ -e "runs/$name" ]; then echo "Existing attempt requires review: $name"; exit 1; fi
   python - <<'PY'
import shutil
assert shutil.disk_usage('.').free > 1500*1024**2, 'Need 1.5 GiB free before reader training'
PY
   python -u scripts/train_extractive.py --init-run "runs/trid-korean-medium-$backbone" --data "data/reader-ablation-without-$data" --output "runs/$name" --attention bidirectional --head "$head" --steps 2000 --batch-size 16 --micro-batch-size 16 --lr 0.0001 --seed 271029 --device cuda --context 1024 --eval-every 100 --threads 4 > "reports/$name.log" 2>&1
   python scripts/evaluate_extractive_dev.py --run "runs/$name" --output "reports/$name-dev.json" --threads 4
   python scripts/analyze_extractive_dev.py --report "reports/$name-dev.json" --output "reports/$name-dev-analysis.json"
   python scripts/evaluate_institution_probe.py --run "runs/$name" --output "reports/$name-probe.json" --threads 4
   aws s3 sync "runs/$name" "$B/runs/$name" --only-show-errors
   aws s3 sync reports "$B/reports" --exclude '*' --include "$name*" --only-show-errors
   # Verify exact remote bytes before reclaiming only this new run's weights.
   python - <<'PY'
import os,hashlib,subprocess,pathlib
root=pathlib.Path('runs')/os.environ['name']
for p in root.glob('*.safetensors'):
 local=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
 proc=subprocess.Popen(['aws','s3','cp',os.environ['B']+'/'+str(p),'-','--only-show-errors'],stdout=subprocess.PIPE)
 remote=hashlib.file_digest(proc.stdout,'sha256').hexdigest()
 assert proc.wait()==0 and local==remote, str(p)
 p.unlink()
pathlib.Path('reports',os.environ['name']+'.archived').write_text('Weights verified SHA256 in S3; local weights reclaimed.\n')
PY
  done
 done
done
echo 'THREE REMOVAL EXPERIMENTS COMPLETE'

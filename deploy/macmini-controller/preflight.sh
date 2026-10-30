#!/bin/bash
# Preflight for the trid SFT controller on the Mac mini (monitor-only policy). Prints the identity ARN and PASS/FAIL
# for exactly the calls the monitor makes; never prints keys. Exit 0 only when every check passes.
set -uo pipefail
cd ~/trid-controller/repo
export PATH="$PWD/.venv/bin:/usr/bin:/bin" AWS_PROFILE="${AWS_PROFILE:-trid-controller}" AWS_DEFAULT_REGION=ap-northeast-2
ok=0
check() { if "${@:2}" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; ok=1; fi; }
command -v aws >/dev/null || { echo "FAIL aws CLI missing in $PWD/.venv"; exit 1; }
echo "identity: $(aws sts get-caller-identity --query Arn --output text 2>&1)"
job=$(python3 -c "import json; c=json.load(open('$HOME/trid-controller/state/campaign.json')); print(next(iter(c['arms'].values()))['job'])")
check "sagemaker:ListTrainingJobs" aws sagemaker list-training-jobs --max-results 1
check "sagemaker:DescribeTrainingJob $job" aws sagemaker describe-training-job --training-job-name "$job"
check "s3:ListBucket corpus/runs/s1/" aws s3api list-objects-v2 --bucket trid-training-148761639313 --prefix corpus/runs/s1/ --max-items 1
echo '{"preflight": true}' > /tmp/trid-preflight.json
check "s3:PutObject controller-status.json" aws s3 cp --only-show-errors /tmp/trid-preflight.json s3://trid-training-148761639313/corpus/runs/s1/controller-status.json
check "s3:GetObject corpus/runs/s1/" aws s3 cp --only-show-errors s3://trid-training-148761639313/corpus/runs/s1/controller-status.json -
[ -f ~/trid-controller/state/campaign.json ] && echo "PASS campaign file" || { echo "FAIL campaign file"; ok=1; }
exit $ok

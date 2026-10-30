#!/bin/bash
# Preflight for the trid SFT controller on the Mac mini (monitor-only policy). Uses AWS CLI v2 (`aws login` sessions,
# /opt/homebrew/bin/aws), not the repo .venv CLI v1. Prints the identity ARN and PASS/FAIL for exactly the calls the
# monitor makes; never prints credentials or tokens. Exit 0 only when every check passes.
set -uo pipefail
cd ~/trid-controller/repo
AWS="${TRID_AWS_CLI:-/opt/homebrew/bin/aws}"
export AWS_PROFILE="${AWS_PROFILE:-trid-controller}" AWS_DEFAULT_REGION=ap-northeast-2
ok=0
check() { if "${@:2}" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; ok=1; fi; }
version=$("$AWS" --version 2>&1) || { echo "FAIL aws CLI missing at $AWS"; exit 1; }
case "$version" in aws-cli/2.*) echo "PASS CLI v2 ($AWS)";; *) echo "FAIL CLI is not v2: ${version%% *}"; ok=1;; esac
echo "identity: $("$AWS" sts get-caller-identity --query Arn --output text 2>&1 | head -1)"
job=$(/usr/bin/python3 -c "import json; c=json.load(open('$HOME/trid-controller/state/campaign.json')); print(next(iter(c['arms'].values()))['job'])")
check "sagemaker:ListTrainingJobs" "$AWS" sagemaker list-training-jobs --max-results 1
check "sagemaker:DescribeTrainingJob $job" "$AWS" sagemaker describe-training-job --training-job-name "$job"
check "s3:ListBucket corpus/runs/s1/" "$AWS" s3api list-objects-v2 --bucket trid-training-148761639313 --prefix corpus/runs/s1/ --max-items 1
echo '{"preflight": true}' > /tmp/trid-preflight.json
check "s3:PutObject controller-status.json" "$AWS" s3 cp --only-show-errors /tmp/trid-preflight.json s3://trid-training-148761639313/corpus/runs/s1/controller-status.json
check "s3:GetObject corpus/runs/s1/" "$AWS" s3 cp --only-show-errors s3://trid-training-148761639313/corpus/runs/s1/controller-status.json -
[ -f ~/trid-controller/state/campaign.json ] && echo "PASS campaign file" || { echo "FAIL campaign file"; ok=1; }
exit $ok

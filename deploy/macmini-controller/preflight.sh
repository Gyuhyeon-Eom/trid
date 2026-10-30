#!/bin/bash
# Preflight for the trid SFT controller on the Mac mini. Prints identity ARN and permission checks only; never prints
# keys. Exit 0 only when every check passes.
set -uo pipefail
cd "$(dirname "$0")/../.." 2>/dev/null || cd ~/trid-controller/repo
export PATH="$PWD/.venv/bin:/usr/bin:/bin" AWS_PROFILE="${AWS_PROFILE:-trid-controller}" AWS_DEFAULT_REGION=ap-northeast-2
ok=0
check() { if "${@:2}" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; ok=1; fi; }
command -v aws >/dev/null || { echo "FAIL aws CLI missing in $PWD/.venv"; exit 1; }
echo "identity: $(aws sts get-caller-identity --query Arn --output text 2>&1)"
check "list training jobs" aws sagemaker list-training-jobs --max-results 1
check "list corpus/ in S3" aws s3 ls s3://trid-training-148761639313/corpus/runs/ --page-size 1
check "read S1 status object" aws s3 ls s3://trid-training-148761639313/corpus/runs/s1/
[ -f ~/trid-controller/state/campaign.json ] && echo "PASS campaign file" || { echo "FAIL campaign file"; ok=1; }
exit $ok

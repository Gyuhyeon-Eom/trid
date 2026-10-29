set -eu
# Run one LLM stage (draft|verify) of scripts/build_task_candidates.py on this instance, with S3 checkpoints.
# usage: stage.sh <work-name> <draft|verify>
B=s3://trid-training-148761639313/c008
S=/opt/dlami/nvme/c008
NAME=$1; STAGE=$2
W=$S/work/$NAME
mkdir -p $S/code/scripts $W && cd $S/code
aws s3 cp $B/code/build_task_candidates.py scripts/ --only-show-errors
aws s3 cp $B/code/build_teacher_qa.py scripts/ --only-show-errors
sha256sum scripts/*.py > $W/$STAGE.code-sha256
aws s3 sync $B/work/$NAME $W --delete --only-show-errors   # S3 is the source of truth; stale local files are removed
if [ "$STAGE" = draft ]; then URL=http://127.0.0.1:18473; else URL=http://127.0.0.1:18474; fi
( while true; do sleep 60; aws s3 sync $W $B/work/$NAME --only-show-errors; done ) &
SYNC=$!
python3 -u scripts/build_task_candidates.py $STAGE --work $W --url $URL --workers 16 > $W/$STAGE.log 2>&1 || echo "exit $?" >> $W/$STAGE.log
kill $SYNC
date -u > $W/$STAGE.done
aws s3 sync $W $B/work/$NAME --only-show-errors

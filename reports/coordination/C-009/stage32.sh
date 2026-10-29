set -eu
# C-009: run one model stage (draft|verify) of scripts/build_task_candidates.py --profile strong on this instance, with
# S3 checkpoints every 60 s. S3 is the source of truth for the work directory. usage: stage32.sh <work-name> <draft|verify> [workers]
B=s3://trid-training-148761639313/c009
S=/opt/dlami/nvme/c009
NAME=$1; STAGE=$2; WORKERS=${3:-6}
W=$S/work/$NAME
mkdir -p $S/code/scripts $W && cd $S/code
for f in build_task_candidates.py build_task_candidates_strong.py build_teacher_qa.py; do
  aws s3 cp $B/code/$f scripts/ --only-show-errors
done
sha256sum scripts/*.py > $W/$STAGE.code-sha256
aws s3 sync $B/work/$NAME $W --delete --only-show-errors
if [ "$STAGE" = draft ]; then URL=http://127.0.0.1:18475; else URL=http://127.0.0.1:18476; fi
( while true; do sleep 60; aws s3 sync $W $B/work/$NAME --only-show-errors; done ) &
SYNC=$!
python3 -u scripts/build_task_candidates.py $STAGE --profile strong --work $W --url $URL --workers $WORKERS > $W/$STAGE.log 2>&1 || echo "exit $?" >> $W/$STAGE.log
kill $SYNC
date -u > $W/$STAGE.done
aws s3 sync $W $B/work/$NAME --only-show-errors

#!/bin/bash
# SageMaker entrypoint: one segment of a long SFT campaign (trid.training.train with a fixed full-length schedule).
# Training arguments equal scripts/train_public_dialogue.py `train` (mix-04, batch 8, micro 2, lr 1e-4, seed 271030,
# validation every 100 steps on 64 rows, context 1024, example normalization). The run directory lives under the
# checkpoint path: SageMaker restores it from S3 at job start, train(resume=True) continues from the newest recovery
# checkpoint (weights, optimizer, sampling RNG, best-so-far, metrics) and the segment ends after SEGMENT_SECONDS
# (or SESSION_STEPS) with a fresh recovery checkpoint. scripts/drive_sft_campaign.py chains segments of long runs.
# Optional in-job evaluation (no driver needed): EVAL_STEPS (comma list of saved milestone steps) and EVAL_BEST=1 run
# the fixed development evaluation (RAG310 evidence + free, KLUE dev question sensitivity) on this instance after
# training, write reports under $RUN/eval/<label>/, then a DONE.json summary; the job then ends and the instance stops.
#   RUN, INIT_S3, INIT_STEP, SFT_STEPS, SAVE_STEPS (comma list), CHECKPOINT_EVERY, SEGMENT_SECONDS | SESSION_STEPS,
#   EVAL_STEPS, EVAL_BEST
set -euo pipefail
: "${RUN:?}" "${INIT_S3:?}" "${INIT_STEP:?}" "${SFT_STEPS:?}" "${SAVE_STEPS:?}" "${CHECKPOINT_EVERY:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
export PYTHONPATH=/opt/work
pip install -q tokenizers safetensors
mkdir -p data
for d in public-dialogue-mix-04 korean-rag-eval-v1 klue-mrc-source; do
  [ -e "/opt/ml/input/data/data/$d" ] && ln -s "/opt/ml/input/data/data/$d" "data/$d"
done
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
evaluate() {  # $1 label, $2 weights file
  local dir=/opt/eval/$1 out="$OUT/eval/$1"
  [ -f "$out/question-sensitivity.json" ] && return 0
  mkdir -p "$dir" "$out"
  cp "$OUT/sft/config.json" "$OUT/sft/tokenizer.json" "$dir/"
  cp "$2" "$dir/model.safetensors"
  sha256sum "$dir/model.safetensors" | sed "s|$dir/||" > "$out/weights.sha256"
  for decoding in evidence free; do
    [ -f "$out/rag310-$decoding.json" ] || python scripts/evaluate_rag_eval.py --run "$dir" --decoding "$decoding" --device cuda \
      --output "$out/rag310-$decoding.json"
  done
  python scripts/evaluate_question_sensitivity.py --run "$dir" --device cuda --output "$out/question-sensitivity.json"
  rm -rf "$dir"
}
EVAL_STEPS=${EVAL_STEPS:-}
for s in ${EVAL_STEPS//,/ }; do
  m="$OUT/sft/milestones/step-$(printf %08d "$s")/model.safetensors"
  if [ -f "$m" ]; then evaluate "step-$(printf %08d "$s")" "$m"; else echo "milestone $s not reached; not evaluated"; fi
done
if [ "${EVAL_BEST:-0}" = 1 ] && grep -q '"status": "complete"' "$OUT/sft/training.json"; then
  evaluate best-valid "$OUT/sft/model.safetensors"
fi
python - "$OUT" <<'PY'
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
t = json.loads((out / 'sft/training.json').read_text())
evals = sorted(p.name for p in (out / 'eval').glob('*') if (p / 'question-sensitivity.json').exists()) if (out / 'eval').exists() else []
(out / 'DONE.json').write_text(json.dumps({'status': t.get('status'), 'completed_steps': t.get('completed_steps'),
    'best_step': t.get('best_step'), 'best_validation_loss': t.get('best_validation_loss'), 'evaluated': evals,
    'finished_unix': time.time()}, indent=1))
print('DONE', (out / 'DONE.json').read_text())
PY

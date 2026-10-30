#!/bin/bash
# SageMaker entrypoint: one self-contained LoRA pilot (PILOT-SFT-001). Downloads the pinned base model, trains with
# scripts/train_open_weight_lora.py on the manifest in the data channel, merges, and leaves adapter + merged model +
# training.json + DONE.json in /opt/ml/checkpoints/$RUN (synced to the checkpoint S3 URI). No retries, no resume:
# a failure ends the job; MaxRuntime is the hard stop. Evaluation runs afterwards with the baseline's llama.cpp
# runtime (GGUF conversion is done there with the same converter for base and tuned models).
#   MODEL_ID, REV   base model and pinned Hugging Face revision
#   MANIFEST        manifest file name inside the data channel (paths in it are relative to that directory)
#   TRAIN_ARGS      fixed preregistered hyperparameters, e.g. "--batch-size 8 --micro-batch-size 1 --max-tokens 4096"
set -euo pipefail
: "${RUN:?}" "${MODEL_ID:?}" "${REV:?}" "${MANIFEST:?}"
cp -r /opt/ml/input/data/code /opt/work
cd /opt/work
pip install -q 'transformers>=4.51,<5' 'peft==0.17.*' tokenizers safetensors 'huggingface_hub<1'
python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir='/opt/base')" "$MODEL_ID" "$REV"
OUT=/opt/ml/checkpoints/$RUN
[ ! -e "$OUT/DONE.json" ] || { echo "already complete"; exit 0; }
rm -rf "$OUT/train"
mkdir -p "$OUT"
# shellcheck disable=SC2086
python scripts/train_open_weight_lora.py --model /opt/base --model-id "$MODEL_ID" --model-revision "$REV" \
  --manifest "/opt/ml/input/data/data/$MANIFEST" --out "$OUT/train" --device cuda --merge ${TRAIN_ARGS:-} \
  2>&1 | tee "$OUT/train.log"
python - "$OUT" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
report = json.loads((out / 'train/training.json').read_text())
merged = out / 'train/merged/model.safetensors'
h = hashlib.sha256()
with merged.open('rb') as f:
    for block in iter(lambda: f.read(1 << 24), b''):
        h.update(block)
(out / 'DONE.json').write_text(json.dumps({'status': report['status'], 'steps': report['steps'],
    'manifest_sha256': report['manifest_sha256'], 'merged_sha256': h.hexdigest(), 'finished_unix': time.time()}, indent=1))
PY
echo done

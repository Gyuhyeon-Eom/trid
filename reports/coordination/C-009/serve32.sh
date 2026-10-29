set -eu
# C-009 model serving (owner: trid session C 01a17c28). usage: serve32.sh gen|verify
# gen    : Qwen3-32B Q8_0 (generator)         port 18475
# verify : gpt-oss-120b MXFP4 (independent verifier) port 18476, MoE experts partly on CPU when VRAM is short
B=s3://trid-training-148761639313
S=/opt/dlami/nvme/c009
PHASE=$1
R=$B/c009/serve/$(date -u +%Y%m%dT%H%MZ)-$PHASE
mkdir -p $S/models && cd $S
if [ ! -x llama.cpp/build/bin/llama-server ]; then
  aws s3 cp $B/teacher-tools/llama-server-cuda-8ca4060.tgz bin.tgz --only-show-errors && tar -xzf bin.tgz && rm bin.tgz
fi
export LD_LIBRARY_PATH=$S/llama.cpp/build/bin   # cached build carries the old instance's RUNPATH
fetch() {   # repo rev file sha256: S3 copy first, else Hugging Face at the pinned revision; always verified
  [ -f models/$3 ] || aws s3 cp $B/teacher-models/c009/$3 models/$3 --only-show-errors || \
    curl -sSL --retry 5 -o models/$3 "https://huggingface.co/$1/resolve/$2/$3"
  echo "$4  models/$3" | sha256sum -c
  aws s3 ls $B/teacher-models/c009/$3 >/dev/null 2>&1 || aws s3 cp models/$3 $B/teacher-models/c009/$3 --only-show-errors
}
L=llama.cpp/build/bin/llama-server
echo "$R" > run-prefix-$PHASE
if [ "$PHASE" = gen ]; then
  fetch Qwen/Qwen3-32B-GGUF 938a7432affaec9157f883a87164e2646ae17555 Qwen3-32B-Q8_0.gguf 2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169
  $L -m models/Qwen3-32B-Q8_0.gguf --host 127.0.0.1 --port 18475 -c ${CTX:-24576} -np ${NP:-6} -ngl 99 -fa on --jinja > $PHASE.log 2>&1 &
else
  fetch ggml-org/gpt-oss-120b-GGUF 7200b74bea78b0a2964ef9f3897eefa401e890ff gpt-oss-120b-MXFP4.gguf 582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d
  $L -m models/gpt-oss-120b-MXFP4.gguf --host 127.0.0.1 --port 18476 -c ${CTX:-32768} -np ${NP:-8} -ngl 99 --n-cpu-moe ${CPU_MOE:-0} -fa on --jinja > $PHASE.log 2>&1 &
fi
( while true; do sleep 120; aws s3 cp $PHASE.log $R/$PHASE.log --only-show-errors; done ) &
wait

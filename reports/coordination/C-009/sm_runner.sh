set -eu
# C-009 SageMaker job (ml.g6e.12xlarge, 4x L40S): both teacher models GPU-resident, stage commands taken from an S3 queue.
# Not a training job in the ML sense: no weights are updated; SageMaker's job API is only the compute container.
# GPUs 0-1: Qwen3-32B Q8_0 generator (:18475)   GPUs 2-3: gpt-oss-120b MXFP4 verifier (:18476)
# queue: s3://.../c009/sm/queue/<seq>-<work>-<draft|verify>   done markers: c009/sm/done/<same name>
# exits after IDLE_MIN minutes with an empty queue, or when c009/sm/queue/stop exists.
B=s3://trid-training-148761639313/c009
T=s3://trid-training-148761639313/teacher-models/c009
R=/opt/ml/c009
JOB=${TRAINING_JOB_NAME:-local}
L=$B/sm/logs/$JOB
IDLE_MIN=${IDLE_MIN:-20}
mkdir -p $R/models $R/work && cd $R
log() { echo "$(date -u +%FT%TZ) $*" | tee -a runner.log; }
command -v aws >/dev/null || pip install -q awscli
command -v cmake >/dev/null || pip install -q cmake
( while true; do sleep 120; for f in runner.log gen.log verify.log; do [ -f $f ] && aws s3 cp $f $L/$f --only-show-errors || true; done; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> gpu.log; aws s3 cp gpu.log $L/gpu.log --only-show-errors || true; done ) &
log "start $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l)"
BIN=llama-server-cu$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')-8ca4060.tgz
if aws s3 cp $B/sm/$BIN bin.tgz --only-show-errors 2>/dev/null; then
  tar -xzf bin.tgz
else
  aws s3 cp $B/sm/llama.cpp-8ca4060.tar.gz src.tgz --only-show-errors
  echo "99899baa53532380cf234d4c2bd5d3295e5e1f864aa06c7813b5c696d94219c6  src.tgz" | sha256sum -c
  tar -xzf src.tgz && cd llama.cpp-8ca4060
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > $R/build.log 2>&1
  cmake --build build --target llama-server -j $(nproc) >> $R/build.log 2>&1
  mkdir -p $R/bin && cp build/bin/llama-server build/bin/*.so* $R/bin/ 2>/dev/null || cp build/bin/llama-server $R/bin/
  cd $R && tar -czf bin.tgz bin && aws s3 cp bin.tgz $B/sm/$BIN --only-show-errors
fi
export LD_LIBRARY_PATH=$R/bin:${LD_LIBRARY_PATH:-}
log "llama-server $(bin/llama-server --version 2>&1 | grep -m1 version)"
# SERVERS=verify loads only the verifier (e.g. a verify-only job): no generator download, no generator server.
SERVERS=${SERVERS:-both}
MODELS="gpt-oss-120b-MXFP4.gguf:582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d"
[ "$SERVERS" = both ] && MODELS="Qwen3-32B-Q8_0.gguf:2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169 $MODELS"
for spec in $MODELS; do
  f=${spec%%:*}; h=${spec##*:}
  aws s3 cp $T/$f models/$f --only-show-errors
  echo "$h  models/$f" | sha256sum -c
done
log "models verified ($SERVERS)"
# slot context = ctx / np (8192 with the defaults: thinking-mode drafts and reasoning judges need more than 4096)
GEN_NP=${GEN_NP:-8}; VER_NP=${VER_NP:-8}
PORTS=18476
if [ "$SERVERS" = both ]; then
  CUDA_VISIBLE_DEVICES=0,1 bin/llama-server -m models/Qwen3-32B-Q8_0.gguf --host 127.0.0.1 --port 18475 -c 65536 -np $GEN_NP -ngl 99 -fa on --jinja > gen.log 2>&1 &
  PORTS="18475 18476"
fi
CUDA_VISIBLE_DEVICES=2,3 bin/llama-server -m models/gpt-oss-120b-MXFP4.gguf --host 127.0.0.1 --port 18476 -c 65536 -np $VER_NP -ngl 99 -fa on --jinja > verify.log 2>&1 &
for port in $PORTS; do
  for i in $(seq 1 120); do curl -s 127.0.0.1:$port/health | grep -q ok && break; sleep 10; done
  curl -s 127.0.0.1:$port/health | grep -q ok || { log "server $port not healthy"; exit 1; }
done
log "servers healthy: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
mkdir -p code/scripts run
run_item() {   # $1 queue entry name; draft and verify run on separate GPU pairs, so one of each may run at once
  local q=$1 NAME STAGE W URL SYNC
  NAME=$(echo $q | cut -d- -f2-); STAGE=${NAME##*-}; NAME=${NAME%-*}
  W=$R/work/$NAME
  mkdir -p $W code/$STAGE/scripts
  for f in build_task_candidates.py build_task_candidates_strong.py build_teacher_qa.py; do aws s3 cp $B/code/$f code/$STAGE/scripts/ --only-show-errors; done
  aws s3 sync $B/work/$NAME $W --delete --only-show-errors
  sha256sum code/$STAGE/scripts/*.py > $W/$STAGE.code-sha256
  if [ "$STAGE" = draft ]; then URL=http://127.0.0.1:18475; N=$GEN_NP; else URL=http://127.0.0.1:18476; N=$VER_NP; fi
  log "run $NAME $STAGE"
  ( while true; do sleep 60; aws s3 sync $W $B/work/$NAME --only-show-errors; done ) & SYNC=$!
  python3 -u code/$STAGE/scripts/build_task_candidates.py $STAGE --profile strong --work $W --url $URL --workers $N > $W/$STAGE.log 2>&1 || echo "exit $?" >> $W/$STAGE.log
  kill $SYNC
  date -u > $W/$STAGE.done
  aws s3 sync $W $B/work/$NAME --only-show-errors
  date -u | aws s3 cp - $B/sm/done/$q --only-show-errors
  log "done $NAME $STAGE"
  rm -f run/$STAGE
}
idle=0
while true; do
  if aws s3 ls $B/sm/queue/stop >/dev/null 2>&1; then   # finish running stage items only (not the log loop), then exit
    log "stop marker"
    while [ -n "$(ls run | grep -v '^started-')" ]; do sleep 10; done
    break
  fi
  started=0
  for q in $(aws s3 ls $B/sm/queue/ | awk '{print $NF}' | sort); do
    STAGE=${q##*-}
    [ -e run/$STAGE ] && continue
    [ -e run/started-$q ] && continue
    aws s3 ls $B/sm/done/$q >/dev/null 2>&1 && continue
    touch run/$STAGE run/started-$q
    run_item $q &
    started=1
  done
  if [ $started = 0 ] && [ -z "$(ls run | grep -v '^started-')" ]; then
    idle=$((idle + 1)); [ $idle -ge $((IDLE_MIN * 2)) ] && { log "idle ${IDLE_MIN} min, exiting"; break; }
  else
    idle=0
  fi
  sleep 30
done
for f in runner.log gen.log verify.log gpu.log build.log; do [ -f $f ] && aws s3 cp $f $L/$f --only-show-errors || true; done

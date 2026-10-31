# AWS-EVAL-006 instance bootstrap (cloud-init user data body). scripts/aws_eval/aws_eval_006.py prepends "#!/bin/bash"
# and the run variables (RUN_ID, MODE, MAX_MINUTES, BUCKET, PREFIX, CODE_KEY, CODE_SHA, LLAMA_COMMIT, CUDA_ARCH,
# INPUT_SHA, WAIT_INPUT_MINUTES, CERT_SHA). The instance does the whole job by itself and then powers off; it was
# launched with InstanceInitiatedShutdownBehavior=terminate, so every power-off ends it:
#   - first command: an OS timer (shutdown -h +MAX_MINUTES) — the job's own hard limit;
#   - on success or any failure: the EXIT trap uploads the log and phase, then powers off;
#   - independently of this OS, an EventBridge Scheduler one-time schedule terminates the instance after the limit.
# Nothing listens on the network (no inbound rules); access is SSM only. Results go to the private bucket prefix.
# "smoke" runs the first two turns of the first public scenario; "full" waits for C to deliver the private set to this
# host over SSM (never through S3). Both write one checkpoint per finished dialogue, encrypted to C's certificate, and
# sync them to S3 every 30 s; at the end outputs.jsonl and every sealed/ file are encrypted the same way and the
# plaintext (and the delivered set) is shredded before upload. Only status.json, phase, GPU/CUDA/binary facts and
# this log stay readable to the operator.
set -uo pipefail
exec > >(tee -a /var/log/trid-eval.log) 2>&1
umask 077
export AWS_DEFAULT_REGION=ap-northeast-2 HOME=/root DEBIAN_FRONTEND=noninteractive
S3="s3://$BUCKET/$PREFIX/runs/$RUN_ID"
WORK=/opt/trid
shutdown -h +"$MAX_MINUTES" "trid AWS-EVAL-006 max runtime"

phase() {
  printf '{"run_id": "%s", "mode": "%s", "phase": "%s", "utc": "%s", "detail": "%s"}\n' \
    "$RUN_ID" "$MODE" "$1" "$(date -u +%FT%TZ)" "$(echo "${2:-}" | tr -d '"\\' | tr '\n' ' ')" > /tmp/phase.json
  aws s3 cp --only-show-errors --sse AES256 /tmp/phase.json "$S3/phase.json" || true
  echo "PHASE $1 ${2:-}"
}
finish() {
  rc=$?
  if [ $rc -eq 0 ]; then phase done "exit 0"; else phase failed "exit $rc"; fi
  aws s3 cp --only-show-errors --sse AES256 /var/log/trid-eval.log "$S3/bootstrap.log" || true
  sync
  shutdown -h now
}
trap finish EXIT
fail() { echo "FAIL: $*"; exit 1; }
fetch() {  # url path sha256
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 --retry-delay 5 -o "$2" "$1" || fail "download $2"
  echo "$3  $2" | sha256sum -c --quiet - || fail "sha256 mismatch $2"
}

phase start "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
mkdir -p "$WORK/out"
aws s3 cp --only-show-errors "s3://$BUCKET/$CODE_KEY" /tmp/code.tar.gz || fail "code download"
echo "$CODE_SHA  /tmp/code.tar.gz" | sha256sum -c --quiet - || fail "code sha256"
tar -xzf /tmp/code.tar.gz -C "$WORK" || fail "code extract"

phase build "llama.cpp $LLAMA_COMMIT CUDA arch $CUDA_ARCH"
command -v cmake >/dev/null || apt-get -y -q install cmake >/dev/null || fail "cmake install"
export PATH=/usr/local/cuda/bin:$PATH
git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp /opt/llama.cpp || fail "llama.cpp clone"
git -C /opt/llama.cpp checkout -q "$LLAMA_COMMIT" || fail "llama.cpp checkout"
[ "$(git -C /opt/llama.cpp rev-parse HEAD)" = "$LLAMA_COMMIT" ] || fail "llama.cpp commit mismatch"
cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake.log 2>&1 || { tail -40 /tmp/cmake.log; fail "cmake configure"; }
cmake --build /opt/llama.cpp/build --target llama-server -j "$(nproc)" > /tmp/build.log 2>&1 || { tail -60 /tmp/build.log; fail "build"; }
SERVER=/opt/llama.cpp/build/bin/llama-server
"$SERVER" --version > "$WORK/out/llama-server-version.txt" 2>&1 || true
{ nvcc --version; nvidia-smi; } > "$WORK/out/cuda.txt" 2>&1 || true

phase setup "python tokenizer env and pinned assets"
python3 -m venv /opt/venv && /opt/venv/bin/pip -q install transformers==5.21.0 tokenizers==0.23.3 jinja2==3.1.6 \
  > /tmp/pip.log 2>&1 || { tail -30 /tmp/pip.log; fail "pip"; }
while IFS=$'\t' read -r url path sha; do
  [ -z "$url" ] || fetch "$url" "$WORK/$path" "$sha"
done < "$WORK/assets.tsv"
for pk in P Q; do
  sha=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['expected_sha256'])" \
        "$WORK/runs/chat-blind-005/packages/$pk/package.json")
  [ -f "$WORK/models/$sha.gguf" ] && [ -d "$WORK/tokenizers/$sha" ] || fail "package $pk assets missing"
  ln -s "$WORK/models/$sha.gguf" "$WORK/runs/chat-blind-005/packages/$pk/model.gguf"
  ln -s "$WORK/tokenizers/$sha" "$WORK/runs/chat-blind-005/packages/$pk/tokenizer"
done

echo "$CERT_SHA  $WORK/c-cert.pem" | sha256sum -c --quiet - || fail "C certificate sha256"
if [ "$MODE" = smoke ]; then
  DIALOGUES=$WORK/smoke.jsonl  # first two turns of the first public CHAT-EVAL-004 scenario
  python3 -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); print(json.dumps({'id': r['id'], 'turns': r['turns'][:2]}, ensure_ascii=False))" \
    "$WORK/reports/coordination/CHAT-EVAL-004/public-20.jsonl" > "$DIALOGUES" || fail "smoke set"
else
  # C sends the private set straight to this host over SSM (never through S3); wait for it at the set boundary.
  DIALOGUES=$WORK/input/private-80.jsonl
  mkdir -p "$WORK/input" && chmod 700 "$WORK/input"
  phase awaiting-input "C delivers the set to $DIALOGUES (sha256 $INPUT_SHA) within $WAIT_INPUT_MINUTES min"
  for _ in $(seq 1 $((WAIT_INPUT_MINUTES * 6))); do
    [ -f "$DIALOGUES.ready" ] && break
    sleep 10
  done
  [ -f "$DIALOGUES.ready" ] || fail "no input within $WAIT_INPUT_MINUTES min"
  echo "$INPUT_SHA  $DIALOGUES" | sha256sum -c --quiet - || fail "input sha256"
fi

BACKEND="llama.cpp-8ca4060-cuda-sm$CUDA_ARCH-$(sha256sum "$SERVER" | cut -c1-12)"
sha256sum "$SERVER" > "$WORK/out/llama-server.sha256"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$WORK/out/gpu.csv" &
GPU_SAMPLER=$!
mkdir -p "$WORK/checkpoints"
( while sleep 30; do aws s3 sync --only-show-errors --sse AES256 --exclude '*' --include '*.cms' \
    "$WORK/checkpoints" "$S3/checkpoints/"; done ) &
SYNCER=$!
cd "$WORK"
for pk in P Q; do
  phase "run-$pk"
  TRID_RUNNER_REMOTE=1 TRID_LLAMA_SERVER=$SERVER TRID_LLAMA_SERVER_ARGS="-ngl 999" TRID_BACKEND=$BACKEND \
    /opt/venv/bin/python scripts/chat_blind_runner.py --package "$pk" --dialogues "$DIALOGUES" --out "$WORK/out/$pk" \
    --checkpoint-dir "$WORK/checkpoints" --checkpoint-cert "$WORK/c-cert.pem" || fail "runner $pk"
done
kill "$GPU_SAMPLER" "$SYNCER" 2>/dev/null
aws s3 sync --only-show-errors --sse AES256 --exclude '*' --include '*.cms' "$WORK/checkpoints" "$S3/checkpoints/" \
  || fail "checkpoint upload"

phase encrypt "outputs and every sealed file to C certificate"
for pk in P Q; do
  for f in "$WORK/out/$pk/outputs.jsonl" "$WORK/out/$pk"/sealed/*; do
    [ -f "$f" ] || continue
    openssl cms -encrypt -binary -aes256 -in "$f" -outform DER -out "$f.cms" "$WORK/c-cert.pem" || fail "encrypt $f"
    shred -u "$f"
  done
done
[ "$MODE" = smoke ] || shred -u "$DIALOGUES"
phase upload
aws s3 cp --only-show-errors --recursive --sse AES256 "$WORK/out" "$S3/results/" || fail "upload"

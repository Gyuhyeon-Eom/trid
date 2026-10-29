set -eu
# C-008 model serving on trid-c-taskgen-01 (owner: trid session C 01a17c28). Serves only; the pipeline runs elsewhere
# through SSM port forwarding. Logs are synced to S3 every 2 minutes.
B=s3://trid-training-148761639313
S=/opt/dlami/nvme/c008
R=$B/c008/serve/$(date -u +%Y%m%dT%H%MZ)
mkdir -p $S/models && cd $S
aws s3 cp $B/teacher-tools/llama-server-cuda-8ca4060.tgz bin.tgz --only-show-errors && tar -xzf bin.tgz && rm bin.tgz
aws s3 cp $B/teacher-models/kanana-1.5-2.1b-instruct-Q8_0.gguf models/kanana.gguf --only-show-errors
aws s3 cp $B/teacher-models/Qwen3-1.7B-Q8_0.gguf models/qwen.gguf --only-show-errors
echo "244c7d717763e885490f48c1dd01a0f7d5f247194c152baf8a8c0158d340adba  models/kanana.gguf" | sha256sum -c
echo "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a  models/qwen.gguf" | sha256sum -c
echo "$R" > run-prefix
export LD_LIBRARY_PATH=$S/llama.cpp/build/bin   # cached build has RUNPATH of the old instance's path
L=llama.cpp/build/bin/llama-server
$L -m models/kanana.gguf --host 127.0.0.1 --port 18473 -c 65536 -np 16 -ngl 99 --jinja > kanana.log 2>&1 &
$L -m models/qwen.gguf --host 127.0.0.1 --port 18474 -c 65536 -np 16 -ngl 99 --jinja > qwen.log 2>&1 &
( while true; do sleep 120; aws s3 cp kanana.log $R/kanana.log --only-show-errors; aws s3 cp qwen.log $R/qwen.log --only-show-errors; done ) &
wait

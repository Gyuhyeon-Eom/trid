#!/bin/bash
# AWS-EVAL-006 contract check on PUBLIC fixtures, CPU only, NO MODEL (a fake llama-server answers with placeholders).
# Exercises the exact code of the real run between delivery and collection:
#   C side   c_deliver_set.smoke_set() bytes → CMS to an instance key → base64 → 24,000-char SendCommand chunks
#   instance bootstrap's reassembly → decrypt → sha check against bootstrap's smoke expectation
#   runner   chat_blind_runner.py P and Q (TRID_RUNNER_REMOTE=1, TRID_RUN_ID, checkpoints CMS-encrypted, atomic rename)
#   C        scripts/aws_eval_collect.py with C's contract (run id, input sha, constant settings) → complete
# and three refusals: a wrong run id, a checkpoint of another run mixed in, a smoke checkpoint against another set.
# The C key used here is a throwaway generated for this check — never C's real key. Exit 0 only if every check holds.
#
#   bash scripts/aws_eval/fixture_e2e.sh            (from the repo root; port 51990 must be free)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cd "$ROOT"
RUN1="fixture-$(date -u +%Y%m%dT%H%M%SZ)-$$-a"
RUN2="fixture-$(date -u +%Y%m%dT%H%M%SZ)-$$-b"
key() { openssl req -x509 -newkey rsa:3072 -nodes -keyout "$T/$1-key.pem" -out "$T/$1-cert.pem" -days 1 -subj "/CN=fixture $1" 2>/dev/null; }
key c; key instance
check() { local desc=$1; shift; if "$@"; then echo "PASS $desc"; else echo "FAIL $desc"; exit 1; fi; }

# C side: the smoke set bytes, encrypted to the instance certificate, split exactly as c_deliver_set.py does
"$PY" - "$T" <<'EOF'
import base64, json, subprocess, sys
sys.path.insert(0, 'scripts/aws_eval')
from c_deliver_set import CHUNK, smoke_set
t = sys.argv[1]
payload = smoke_set()
open(f'{t}/smoke.jsonl', 'wb').write(payload)
cipher = subprocess.run(['openssl', 'cms', '-encrypt', '-binary', '-aes256', '-outform', 'DER', f'{t}/instance-cert.pem'],
                        input=payload, capture_output=True, check=True).stdout
enc = base64.b64encode(cipher).decode()
cmds = [f"printf '%s' '{enc[i:i + CHUNK]}' >> {t}/inbox.cms.b64" for i in range(0, len(enc), CHUNK)]
assert not any(payload.decode()[:20] in c for c in cmds), 'plaintext in a command'
open(f'{t}/commands.sh', 'w').write('\n'.join(cmds) + f'\ntouch {t}/inbox.cms.b64.ready\n')
EOF
bash "$T/commands.sh"
# instance side, as in bootstrap.sh (expected sha computed the bootstrap way)
EXPECTED=$(python3 -c "import json,sys,hashlib; r=json.loads(open(sys.argv[1]).readline()); print(hashlib.sha256((json.dumps({'id': r['id'], 'turns': r['turns'][:2]}, ensure_ascii=False) + chr(10)).encode()).hexdigest())" \
  reports/coordination/CHAT-EVAL-004/public-20.jsonl)
"$PY" -c "import base64,sys; sys.stdout.buffer.write(base64.b64decode(open(sys.argv[1]).read()))" "$T/inbox.cms.b64" > "$T/inbox.cms"
openssl cms -decrypt -binary -inform DER -in "$T/inbox.cms" -inkey "$T/instance-key.pem" -out "$T/delivered.jsonl"
check "delivery: decrypted set sha = bootstrap expectation" [ "$(shasum -a 256 "$T/delivered.jsonl" | cut -d' ' -f1)" = "$EXPECTED" ]

run() {  # alias run_id out checkpoints
  TRID_RUNNER_REMOTE=1 TRID_LLAMA_SERVER="$ROOT/scripts/aws_eval/fake_llama_server.sh" TRID_BACKEND=fixture-no-model \
  TRID_RUN_ID="$2" TRID_CODE_SHA=fixture "$PY" scripts/chat_blind_runner.py --package "$1" --dialogues "$T/delivered.jsonl" \
    --out "$3" --checkpoint-dir "$4" --checkpoint-cert "$T/c-cert.pem" > "$3.stdout"
}
run P "$RUN1" "$T/run1-P" "$T/ck1"; run Q "$RUN1" "$T/run1-Q" "$T/ck1"
check "runner: exactly P-001.cms and Q-001.cms, no partial files" [ "$(ls "$T/ck1" | tr '\n' ' ')" = "P-001.cms Q-001.cms " ]

collect() {  # alias out run-id checkpoints set
  "$PY" scripts/aws_eval_collect.py --checkpoints "$4" --key "$T/c-key.pem" --set "$5" --alias "$1" --run-id "$3" --out "$2"
}
for A in P Q; do
  out=$(collect "$A" "$T/col1-$A" "$RUN1" "$T/ck1" "$T/smoke.jsonl")
  check "collector $A complete: $out" grep -q '"status": "complete"' <<< "$out"
done
# refusals
out=$(collect P "$T/col-wrong" "$RUN2" "$T/ck1" "$T/smoke.jsonl" || true)
check "refused: wrong run id" grep -q 'not among --run-id' <<< "$out"
run P "$RUN2" "$T/run2-P" "$T/ck2"
mkdir -p "$T/mixed" && cp "$T/ck1/Q-001.cms" "$T/mixed/" && cp "$T/ck2/P-001.cms" "$T/mixed/"
out=$(collect P "$T/col-mixed" "$RUN1" "$T/mixed" "$T/smoke.jsonl" || true)
check "refused: checkpoint of another run mixed in" grep -q 'not among --run-id' <<< "$out"
head -n 1 reports/coordination/CHAT-EVAL-004/public-20.jsonl > "$T/other-set.jsonl"
out=$(collect P "$T/col-other" "$RUN1" "$T/ck1" "$T/other-set.jsonl" || true)
check "refused: checkpoint of the smoke set against another set" grep -q 'is not the id+turns derivative' <<< "$out"
echo "ALL FIXTURE CHECKS PASSED (no model was run)"

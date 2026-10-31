#!/bin/bash
# Fixture stand-in for llama-server (no model): see fake_llama_server.py. Used only by scripts/aws_eval/fixture_e2e.sh.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/aws_eval/fake_llama_server.py" "$@"

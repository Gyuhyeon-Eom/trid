#!/bin/sh
# Rerun a cache-resuming collector until it exits 0; waits 5 min after a failure (outage, IP change).
cd "$(dirname "$0")/.." || exit 1
until .venv/bin/python -u "$@"; do echo "collector exited $?; retrying in 300 s"; sleep 300; done
echo "collector finished"

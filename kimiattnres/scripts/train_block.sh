#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
exec "$ROOT/.venv/bin/python" -m kimiattnres.train_sequential --config kimiattnres/configs/qwen3_1.7b_block.yaml "$@"

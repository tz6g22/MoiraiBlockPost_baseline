#!/usr/bin/env bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=kimi-full-smoke
#SBATCH --output=kimiattnres/outputs/qwen3_1.7b/full_smoke_%j.out
#SBATCH --error=kimiattnres/outputs/qwen3_1.7b/full_smoke_%j.err

set -Eeuo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PYTHON="$ROOT/.venv/bin/python"
CONFIG="kimiattnres/configs/qwen3_1.7b_full.yaml"
OUT="$ROOT/kimiattnres/outputs/qwen3_1.7b/full_fsdp_smoke_${SLURM_JOB_ID:-manual}"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m kimiattnres.train_sequential --config "$CONFIG" --output-dir "$OUT" --steps-per-task 3 --stop-after-task math
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m kimiattnres.train_sequential --config "$CONFIG" --output-dir "$OUT" --resume "$OUT/after_math" --steps-per-task 3

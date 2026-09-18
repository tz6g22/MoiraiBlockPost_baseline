#!/usr/bin/env bash
# Two-GPU Qwen3-1.7B Kimi formal training launcher.
# Usage: sbatch --job-name=kimi-block-formal formal_2gpu.sbatch block

#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=outputs/formal_retrain/slurm/%x_%j.out
#SBATCH --error=outputs/formal_retrain/slurm/%x_%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:?expected block or full}"
case "$MODE" in
  block)
    CONFIG="${KIMI_CONFIG:-kimiattnres/configs/qwen3_1.7b_block.yaml}"
    ;;
  full)
    CONFIG="${KIMI_CONFIG:-kimiattnres/configs/qwen3_1.7b_full.yaml}"
    ;;
  *)
    echo "expected block or full, got: $MODE" >&2
    exit 2
    ;;
esac

exec "$ROOT/.venv/bin/python" -u -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m kimiattnres.train_sequential \
  --config "$CONFIG"

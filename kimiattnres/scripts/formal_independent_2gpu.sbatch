#!/usr/bin/env bash
# One fresh two-GPU Native Kimi task run. Math and Multi-hop are submitted as
# separate jobs and never share a Python process, model, or optimizer state.
# Usage: sbatch --job-name=kimi-block-math formal_independent_2gpu.sbatch block math [output_dir]

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
TASK="${2:?expected math or multihop}"
case "$MODE:$TASK" in
  block:math) CONFIG="kimiattnres/configs/qwen3_1.7b_block.yaml"; DEFAULT_OUT="outputs/formal_retrain/kimi_block_independent/math" ;;
  block:multihop) CONFIG="kimiattnres/configs/qwen3_1.7b_block.yaml"; DEFAULT_OUT="outputs/formal_retrain/kimi_block_independent/multihop" ;;
  full:math) CONFIG="kimiattnres/configs/qwen3_1.7b_full.yaml"; DEFAULT_OUT="outputs/formal_retrain/kimi_full_independent/math" ;;
  full:multihop) CONFIG="kimiattnres/configs/qwen3_1.7b_full.yaml"; DEFAULT_OUT="outputs/formal_retrain/kimi_full_independent/multihop" ;;
  *) echo "expected mode block/full and task math/multihop" >&2; exit 2 ;;
esac
OUT="${3:-$DEFAULT_OUT}"

exec "$ROOT/.venv/bin/python" -u -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m kimiattnres.train_sequential \
  --config "$CONFIG" \
  --independent-task "$TASK" \
  --output-dir "$OUT"

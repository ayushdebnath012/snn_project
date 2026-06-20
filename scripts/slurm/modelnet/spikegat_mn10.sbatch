#!/usr/bin/env bash
#SBATCH --job-name=spikegat-mn10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${SPIKEGAT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$ROOT"
source "${SPIKEGAT_VENV:-$ROOT/.venv}/bin/activate"

: "${MODELNET10_DIR:?Set MODELNET10_DIR to the ModelNet10 root before sbatch}"
OUTPUT_ROOT="${SPIKEGAT_OUTPUT_ROOT:-$ROOT/outputs}"
export SPIKEGAT_CKPT_DIR="${SPIKEGAT_CKPT_DIR:-$OUTPUT_ROOT/spikegat_mn10}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$SPIKEGAT_CKPT_DIR"

srun python -u experiments/modelnet/train_spikegat_modelnet10.py

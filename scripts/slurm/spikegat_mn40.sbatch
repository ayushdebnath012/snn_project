#!/usr/bin/env bash
#SBATCH --job-name=spikegat-mn40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${ASP_SNN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
source "${ASP_SNN_VENV:-$ROOT/.venv}/bin/activate"

: "${MODELNET40_DIR:?Set MODELNET40_DIR to the ModelNet40 root before sbatch}"
OUTPUT_ROOT="${ASP_SNN_OUTPUT_ROOT:-$ROOT/outputs}"
export SPIKEGAT_CKPT_DIR="${SPIKEGAT_CKPT_DIR:-$OUTPUT_ROOT/spikegat_mn40}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export ASP_SNN_AUTO_INSTALL=0
mkdir -p "$SPIKEGAT_CKPT_DIR"

srun python -u experiments/kaggle/spikegat/modelnet40.py

#!/usr/bin/env bash
#SBATCH --job-name=asp-s3dis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${ASP_SNN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
source "${ASP_SNN_VENV:-$ROOT/.venv}/bin/activate"
: "${S3DIS_DIR:?Set S3DIS_DIR before sbatch}"
OUT="${ASP_SNN_OUTPUT_ROOT:-$ROOT/outputs}/s3dis"
mkdir -p "$OUT/checkpoints" "$OUT/logs"

srun python -u -m tasks.train_s3dis \
  --config configs/s3dis_seg.yaml \
  --set data_dir="$S3DIS_DIR" ckpt_dir="$OUT/checkpoints" \
        log_dir="$OUT/logs" num_workers="${NUM_WORKERS:-8}"

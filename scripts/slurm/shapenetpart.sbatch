#!/usr/bin/env bash
#SBATCH --job-name=asp-shapenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${ASP_SNN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
source "${ASP_SNN_VENV:-$ROOT/.venv}/bin/activate"
: "${SHAPENETPART_DIR:?Set SHAPENETPART_DIR before sbatch}"
OUT="${ASP_SNN_OUTPUT_ROOT:-$ROOT/outputs}/shapenetpart"
mkdir -p "$OUT/checkpoints" "$OUT/logs"

srun python -u -m tasks.train_shapenetpart \
  --config configs/shapenet_seg.yaml \
  --set data_dir="$SHAPENETPART_DIR" ckpt_dir="$OUT/checkpoints" \
        log_dir="$OUT/logs" num_workers="${NUM_WORKERS:-8}"

#!/usr/bin/env bash
#SBATCH --job-name=asp-modelnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${ASP_SNN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
source "${ASP_SNN_VENV:-$ROOT/.venv}/bin/activate"

OUTPUT_ROOT="${ASP_SNN_OUTPUT_ROOT:-$ROOT/outputs}"
mkdir -p "$OUTPUT_ROOT/modelnet_a100" "$OUTPUT_ROOT/data"

srun python -u experiments/cluster/train_modelnet_a100.py \
  --datasets "${DATASETS:-ModelNet10,ModelNet40}" \
  --epochs "${EPOCHS:-300}" \
  --batch "${BATCH_SIZE:-64}" \
  --ckpt_dir "$OUTPUT_ROOT/modelnet_a100" \
  --data_dir "${MODELNET_DATA_ROOT:-$OUTPUT_ROOT/data}"

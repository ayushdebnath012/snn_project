#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPIKEGAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

mapfile -t JOBS < <(find scripts/slurm -mindepth 2 -type f -name '*.sbatch' | sort)
if ((${#JOBS[@]} == 0)); then
  echo "No dataset jobs found under scripts/slurm/<dataset>/" >&2
  exit 1
fi

for job in "${JOBS[@]}"; do
  echo "Submitting $job"
  sbatch --export=ALL "$job"
done

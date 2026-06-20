#!/usr/bin/env bash
set -euo pipefail

: "${MODELNET10_DIR:?Set MODELNET10_DIR}"
: "${MODELNET40_DIR:?Set MODELNET40_DIR}"

sbatch --export=ALL scripts/slurm/spikegat_mn10.sbatch
sbatch --export=ALL scripts/slurm/spikegat_mn40.sbatch

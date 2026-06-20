#!/usr/bin/env bash
set -euo pipefail

: "${MODELNET10_DIR:?Set MODELNET10_DIR}"
: "${MODELNET40_DIR:?Set MODELNET40_DIR}"
: "${SCANOBJECTNN_DIR:?Set SCANOBJECTNN_DIR}"
: "${SHAPENETPART_DIR:?Set SHAPENETPART_DIR}"
: "${S3DIS_DIR:?Set S3DIS_DIR}"

sbatch --export=ALL scripts/slurm/spikegat_mn10.sbatch
sbatch --export=ALL scripts/slurm/spikegat_mn40.sbatch
sbatch --export=ALL scripts/slurm/scanobjectnn.sbatch
sbatch --export=ALL scripts/slurm/shapenetpart.sbatch
sbatch --export=ALL scripts/slurm/s3dis.sbatch

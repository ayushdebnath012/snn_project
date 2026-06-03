#!/bin/bash
# scripts/prepare_shapenet_kaggle.sh
# Download ShapeNetPart from Kaggle and convert to HDF5 format.
#
# Prerequisites:
#   1. Kaggle CLI installed:  pip install kaggle
#   2. Kaggle API key at ~/.kaggle/kaggle.json
#      (https://www.kaggle.com/docs/api)
#
# Usage: bash scripts/prepare_shapenet_kaggle.sh

set -e

DATA_DIR="data"
RAW_DIR="${DATA_DIR}/shapenetcore_partanno_segmentation_benchmark_v0_normal"
OUT_DIR="${DATA_DIR}/shapenet_part_seg_hdf5_data"

mkdir -p "${DATA_DIR}"

echo "========================================"
echo "  ShapeNetPart from Kaggle"
echo "========================================"

# Check Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "[ERROR] kaggle CLI not found"
    echo "  Install with: pip install kaggle"
    echo "  Set up API key at ~/.kaggle/kaggle.json"
    exit 1
fi

# Skip if already converted
if [ -f "${OUT_DIR}/all_object_categories.txt" ]; then
    echo "[Skip] HDF5 already present at ${OUT_DIR}"
    exit 0
fi

# Download from Kaggle (try multiple known datasets)
DOWNLOADED=0
for ds in "mitkir/shapenet" "rkrispin/shapenet-part"; do
    echo "[Try] kaggle datasets download -d ${ds} -p ${DATA_DIR}"
    if kaggle datasets download -d "${ds}" -p "${DATA_DIR}" --force 2>/dev/null; then
        DOWNLOADED=1
        break
    fi
done

if [ ${DOWNLOADED} -eq 0 ]; then
    echo "[ERROR] Could not download from Kaggle"
    echo "  Manually download a ShapeNetPart dataset from kaggle.com"
    echo "  and place the raw extracted folder at:"
    echo "    ${RAW_DIR}"
    exit 1
fi

# Extract any zip in data/
echo "[Extract] unzipping downloaded archive(s) ..."
for zf in ${DATA_DIR}/*.zip; do
    [ -f "$zf" ] || continue
    unzip -q -o "${zf}" -d "${DATA_DIR}"
done

# Find the raw directory if it has a different name
if [ ! -d "${RAW_DIR}" ]; then
    for cand in ${DATA_DIR}/shapenetcore*; do
        if [ -d "$cand" ]; then
            RAW_DIR="$cand"
            echo "[Found] raw dir: ${RAW_DIR}"
            break
        fi
    done
fi

if [ ! -d "${RAW_DIR}" ]; then
    echo "[ERROR] Raw directory not found after extraction"
    echo "  Expected: ${RAW_DIR}"
    ls -la "${DATA_DIR}/"
    exit 1
fi

# Convert raw to HDF5
echo "[Convert] raw -> HDF5 ..."
python datasets/convert_shapenet_raw.py \
    --raw_dir "${RAW_DIR}" \
    --out_dir "${OUT_DIR}"

# Verify
N_TRAIN=$(ls ${OUT_DIR}/train*.h5 2>/dev/null | wc -l)
N_TEST=$(ls ${OUT_DIR}/test*.h5 2>/dev/null | wc -l)
echo ""
echo "========================================"
echo "  Done!"
echo "  ${N_TRAIN} train H5 + ${N_TEST} test H5 in ${OUT_DIR}"
echo "========================================"

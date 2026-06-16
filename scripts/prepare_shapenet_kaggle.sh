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
if [ -f "${OUT_DIR}/all_object_categories.txt" ] \
   && ls "${OUT_DIR}"/train*.h5 >/dev/null 2>&1 \
   && ls "${OUT_DIR}"/val*.h5 >/dev/null 2>&1 \
   && ls "${OUT_DIR}"/test*.h5 >/dev/null 2>&1; then
    echo "[Skip] HDF5 already present at ${OUT_DIR}"
    exit 0
fi

# Download from Kaggle. The converter requires this raw ShapeNetPart mirror
# because it includes train_test_split/*.json.
echo "[Try] kaggle datasets download -d mitkir/shapenet -p ${DATA_DIR}"
if ! kaggle datasets download -d "mitkir/shapenet" -p "${DATA_DIR}" --force 2>/dev/null; then
    echo "[ERROR] Could not download from Kaggle"
    echo "  Manually download https://www.kaggle.com/datasets/mitkir/shapenet"
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
    for cand in "${DATA_DIR}"/shapenetcore*; do
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

if [ ! -f "${RAW_DIR}/synsetoffset2category.txt" ] \
   || [ ! -f "${RAW_DIR}/train_test_split/shuffled_train_file_list.json" ] \
   || [ ! -f "${RAW_DIR}/train_test_split/shuffled_val_file_list.json" ] \
   || [ ! -f "${RAW_DIR}/train_test_split/shuffled_test_file_list.json" ]; then
    echo "[ERROR] Raw directory is incomplete: ${RAW_DIR}"
    echo "  Expected synsetoffset2category.txt and train_test_split/*.json."
    echo "  This usually means the wrong Kaggle dataset/archive was downloaded."
    echo "  Use: kaggle datasets download -d mitkir/shapenet -p ${DATA_DIR} --force"
    exit 1
fi

# Convert raw to HDF5
echo "[Convert] raw -> HDF5 ..."
python datasets/convert_shapenet_raw.py \
    --raw_dir "${RAW_DIR}" \
    --out_dir "${OUT_DIR}"

# Verify
N_TRAIN=$(ls ${OUT_DIR}/train*.h5 2>/dev/null | wc -l)
N_VAL=$(ls ${OUT_DIR}/val*.h5 2>/dev/null | wc -l)
N_TEST=$(ls ${OUT_DIR}/test*.h5 2>/dev/null | wc -l)
if [ "${N_TRAIN}" -eq 0 ] || [ "${N_VAL}" -eq 0 ] || [ "${N_TEST}" -eq 0 ]; then
    echo "[ERROR] Conversion finished but a train/val/test split is missing."
    exit 1
fi
echo ""
echo "========================================"
echo "  Done!"
echo "  ${N_TRAIN} train + ${N_VAL} val + ${N_TEST} test H5 in ${OUT_DIR}"
echo "========================================"

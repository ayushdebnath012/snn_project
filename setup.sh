#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "PyTorch is not installed. Install the wheel matching this cluster's CUDA driver,"
  echo "then rerun setup.sh. See docs/CLUSTER.md."
  exit 2
fi

python -m pip install -r requirements.txt
python tools/validate_repo.py

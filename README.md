# SpikeGAT-SNN

Extensible research repository for SpikeGAT-based spiking point-cloud learning.
The currently maintained baselines target ModelNet10 and ModelNet40; additional
datasets and tasks can be added as independent modules without changing those
baselines. ASP is not used in this codebase.

## Available experiments

| Dataset | Full training entrypoint | Paper target |
|---|---|---:|
| ModelNet10 | `experiments/modelnet/train_spikegat_modelnet10.py` | 94.93% single-pass OA |
| ModelNet40 | `experiments/modelnet/train_spikegat_modelnet40.py` | 92.38% single-pass OA |

The target values are comparison thresholds, not claimed results. Completed
runs write measured `single_pass_oa` and supplementary `scale_tta_oa` to
`final_metrics.json`.

## Repository layout

```text
experiments/
  modelnet/                 # current MN10/MN40 full runners
  <dataset>/                # future dataset-specific runners
scripts/slurm/
  modelnet/                 # current cluster jobs
  <dataset>/                # future dataset-specific jobs
docs/
  CLUSTER.md
  EXPERIMENTS.md
CONTRIBUTING.md             # extension contract for new datasets
tools/
  validate_repo.py
```

Reusable code may be introduced in top-level `datasets/`, `models/`,
`training/`, `tasks/`, or `configs/` packages as the project grows. Datasets,
checkpoints, papers, reports, and generated results themselves remain excluded
from Git.

## Environment

```bash
git clone https://github.com/ayushdebnath012/SNN.git
cd SNN

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tools/validate_repo.py
```

Choose the PyTorch wheel that matches the CUDA driver on your machine.

## ModelNet baseline data

Raw datasets are not committed because the official archives are large
(approximately 451 MiB for ModelNet10 and 1.90 GiB for ModelNet40). Download
and extract either dataset from its official Princeton source with:

```bash
python tools/download_modelnet.py --dataset 10
python tools/download_modelnet.py --dataset 40
```

By default this creates `data/ModelNet10` and `data/ModelNet40`; `data/` is
ignored by Git. Use `--data-root /shared/datasets` on a cluster, `--check` to
validate an existing copy, or `--list` to inspect sources without downloading.

Each ModelNet root must contain class directories, each with `train/` and
`test/` folders containing `.off`, `.txt`, or `.npy` point-cloud files:

```text
ModelNet40/
  airplane/train/*.off
  airplane/test/*.off
  ...
```

Training jobs never download datasets or install packages at runtime.

## Full ModelNet runs

```bash
MODELNET10_DIR=$PWD/data/ModelNet10 \
SPIKEGAT_CKPT_DIR=/checkpoints/spikegat_mn10 \
python -u experiments/modelnet/train_spikegat_modelnet10.py

MODELNET40_DIR=$PWD/data/ModelNet40 \
SPIKEGAT_CKPT_DIR=/checkpoints/spikegat_mn40 \
python -u experiments/modelnet/train_spikegat_modelnet40.py
```

Optional environment overrides:

```bash
export EPOCHS=180
export TEACHER_EPOCHS=150
export BATCH_SIZE=32
export NUM_WORKERS=4
```

Rerunning the same command resumes from the latest checkpoint.

## Cluster execution

See [docs/CLUSTER.md](docs/CLUSTER.md). The supplied ModelNet jobs request one
GPU each:

```bash
export MODELNET10_DIR=/shared/datasets/ModelNet10
export MODELNET40_DIR=/shared/datasets/ModelNet40
export SPIKEGAT_OUTPUT_ROOT=$SCRATCH/spikegat

sbatch scripts/slurm/modelnet/spikegat_mn10.sbatch
sbatch scripts/slurm/modelnet/spikegat_mn40.sbatch
```

## Adding a dataset

Follow [CONTRIBUTING.md](CONTRIBUTING.md). New datasets should have a dedicated
experiment folder, documented data contract and metrics, a reproducible full
runner, and a matching cluster job where applicable. The repository validator
accepts additional datasets and Python packages while preserving the working
ModelNet baselines.

## Metric integrity

- Use the benchmark's standard primary metric and state it explicitly.
- For ModelNet, single-pass overall accuracy is the primary comparison metric.
- Label test-time augmentation results separately from single-pass results.
- A smoke test validates execution and gradients; it does not establish
  benchmark accuracy.

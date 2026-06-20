# SpikeGAT-SNN for ModelNet

Focused full-training repository for Max-First SpikeGAT classification on
ModelNet10 and ModelNet40. ASP is not used in this codebase.

## Scope

Only two experiments are supported:

| Dataset | Full training entrypoint | Paper target |
|---|---|---:|
| ModelNet10 | `experiments/full/train_spikegat_modelnet10.py` | 94.93% single-pass OA |
| ModelNet40 | `experiments/full/train_spikegat_modelnet40.py` | 92.38% single-pass OA |

The target values are comparison thresholds, not claimed results. Completed
runs write measured `single_pass_oa` and supplementary `scale_tta_oa` to
`final_metrics.json`.

## Repository layout

```text
experiments/full/
  train_spikegat_modelnet10.py
  train_spikegat_modelnet40.py
scripts/slurm/
  spikegat_mn10.sbatch
  spikegat_mn40.sbatch
  submit_all.sh
docs/
  CLUSTER.md
  EXPERIMENTS.md
tools/
  validate_repo.py
```

Datasets, checkpoints, papers, reports, and generated results are deliberately
excluded.

## Environment

```bash
git clone https://github.com/ayushdebnath012/ASP-SNN.git
cd ASP-SNN

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tools/validate_repo.py
```

Choose the PyTorch wheel that matches the CUDA driver on your machine.

## Dataset layout

Each dataset root must contain class directories, each with `train/` and
`test/` folders containing `.off`, `.txt`, or `.npy` point-cloud files:

```text
ModelNet40/
  airplane/train/*.off
  airplane/test/*.off
  ...
```

Training jobs never download datasets or install packages at runtime.

## Full runs

```bash
MODELNET10_DIR=/datasets/ModelNet10 \
SPIKEGAT_CKPT_DIR=/checkpoints/spikegat_mn10 \
python -u experiments/full/train_spikegat_modelnet10.py

MODELNET40_DIR=/datasets/ModelNet40 \
SPIKEGAT_CKPT_DIR=/checkpoints/spikegat_mn40 \
python -u experiments/full/train_spikegat_modelnet40.py
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

See [docs/CLUSTER.md](docs/CLUSTER.md). The supplied jobs request one GPU each:

```bash
export MODELNET10_DIR=/shared/datasets/ModelNet10
export MODELNET40_DIR=/shared/datasets/ModelNet40
export SPIKEGAT_OUTPUT_ROOT=$SCRATCH/spikegat

sbatch scripts/slurm/spikegat_mn10.sbatch
sbatch scripts/slurm/spikegat_mn40.sbatch
```

## Metric integrity

- Single-pass overall accuracy is the primary comparison metric.
- Scale-TTA accuracy is reported separately.
- A smoke test validates execution and gradients; it does not establish
  benchmark accuracy.

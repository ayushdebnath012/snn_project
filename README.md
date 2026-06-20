# ASP-SNN

Clean training repository for active and graph-based spiking neural networks on
3D point clouds. It contains reusable ASP-SNN components, dataset-specific
training tasks, self-contained Kaggle runners, and SLURM launchers.

The repository intentionally excludes datasets, checkpoints, generated plots,
paper PDFs, and reviewer supplementary material.

## Supported training surfaces

| Folder | Purpose |
|---|---|
| `experiments/kaggle/spikegat/` | Updated Max-First SpikeGAT ModelNet10/40 runs |
| `experiments/kaggle/asp/` | Self-contained ASP ModelNet10/40 runs |
| `experiments/kaggle/dgcnn/` | Spiking DGCNN comparison runs |
| `experiments/cluster/` | A100/H100-oriented ModelNet training |
| `experiments/standalone/` | Download-and-run ScanObjectNN, ShapeNetPart, and S3DIS jobs |
| `tasks/` | Config-driven reusable training and evaluation entrypoints |
| `models/`, `training/`, `datasets/`, `data/` | Shared implementation |
| `scripts/slurm/` | One-GPU cluster jobs with resumable output paths |

The updated ModelNet targets used for comparison are 94.93% single-pass OA on
ModelNet10 and 92.38% single-pass OA on ModelNet40. These are targets, not
claimed results; use `final_metrics.json` from a completed run as evidence.

## Quick start

```bash
git clone https://github.com/ayushdebnath012/ASP-SNN.git
cd ASP-SNN

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch build appropriate for the cluster CUDA driver first.
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python tools/validate_repo.py --imports
```

## Run the updated SpikeGAT training

Kaggle T4:

```bash
python experiments/kaggle/spikegat/modelnet40.py
python experiments/kaggle/spikegat/modelnet10.py
```

SLURM/HPC:

```bash
export MODELNET40_DIR=/path/to/ModelNet40
export SPIKEGAT_CKPT_DIR=$SCRATCH/asp-snn/spikegat_mn40
sbatch scripts/slurm/spikegat_mn40.sbatch
```

The ModelNet40 T4 profile uses cached teacher logits and mixed-precision KNN.
Rerunning the same command resumes from `teacher_latest.pt` or
`spikegat_mn40_latest.pt` in the checkpoint directory.

## Reusable task entrypoints

```bash
python -m tasks.train_scanobjectnn --config configs/scanobj_cls.yaml
python -m tasks.train_shapenetpart --config configs/shapenet_seg.yaml
python -m tasks.train_s3dis --config configs/s3dis_seg.yaml

python -m tasks.eval_scanobjectnn --ckpt checkpoints/scanobj_best.pt --n_votes 1
python -m tasks.eval_shapenetpart --ckpt checkpoints/shapenet_best.pt --per_cat
```

Configuration values can be overridden without editing YAML:

```bash
python -m tasks.train_scanobjectnn \
  --config configs/scanobj_cls.yaml \
  --set data_dir=/datasets/ScanObjectNN/main_split batch_size=16 epochs=300
```

## Documentation

- [Cluster and SLURM guide](docs/CLUSTER.md)
- [Kaggle guide](docs/KAGGLE.md)
- [Experiment and metric guide](docs/EXPERIMENTS.md)

## Result integrity

- Paper comparisons use single-pass overall accuracy unless explicitly stated.
- Scale-TTA/voting results are reported separately.
- Checkpoints and generated results belong under `outputs/`, `checkpoints/`, or
  cluster scratch storage and are ignored by Git.
- A successful smoke test proves code-path integrity, not benchmark accuracy.

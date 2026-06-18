# ASP-SNN — How to Run on Kaggle / HPC Cluster

Five self-contained scripts, one per benchmark dataset.
Each script downloads its own data, trains the model, and saves results.

---

## Files in this package

| Script | Dataset | Expected OA / mIoU | GPU time (T4) |
|---|---|---|---|
| `mn40_full.py` | ModelNet40 (40-class) | >92% OA | ~10 h |
| `mn10_full.py` | ModelNet10 (10-class) | >96% OA | ~2 h |
| `scanobjectnn_full.py` | ScanObjectNN PB-T50-RS | >85% OA | ~8 h |
| `shapenetpart_full.py` | ShapeNetPart (50 parts) | ~86–87% mIoU | ~16 h |
| `s3dis_full.py` | S3DIS Area 5 (13 classes) | ~57–64% mIoU | ~15 h |

All scripts include:
- Automatic data download (no manual setup needed)
- Knowledge distillation from a PointNet teacher (+0.5–1 pp)
- Checkpoint resume — safe to interrupt and restart
- Val OA printed every 5 epochs; best checkpoint saved automatically

---

## Option A — Run on Kaggle (recommended)

### 1. Create a Kaggle notebook
Go to kaggle.com → New Notebook → select GPU T4 x2 (or P100).

### 2. Upload the script
In the notebook editor, add a code cell:
```python
# paste the entire contents of the script here, or use:
exec(open("/kaggle/input/your-dataset/mn40_full.py").read())
```

Or upload the script as a Kaggle dataset and attach it.

### 3. No dataset attachment required
Each script auto-clones the project from GitHub if not found:
```
https://github.com/AryaPawa/ASP-SNN.git  (branch: codex/fix-shapenet-h5-conversion)
```
Data is also downloaded automatically (ModelNet via kagglehub, S3DIS via gdown, etc.).

### 4. Enable internet
Notebook Settings → Internet → On (needed for auto-clone and data download).

### 5. Run
Click Run All. Monitor the `OA=` lines (printed every 5 epochs) — those are the real metric.

---

## Option B — Run on HPC / SLURM cluster

### 1. Set up environment
```bash
module load anaconda/2023   # or your site's Python module
conda create -n aspsnn python=3.10 -y
conda activate aspsnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install kagglehub trimesh h5py gdown pyyaml scipy
```

### 2. Clone the project
```bash
git clone --depth=1 --branch codex/fix-shapenet-h5-conversion \
    https://github.com/AryaPawa/ASP-SNN.git
cd ASP-SNN
```

### 3. Run a script directly
```bash
cd ASP-SNN
python kaggle_mn40_full.py
```
The script detects it is not on Kaggle and writes outputs to `./outputs/mn40_ckpts/`.

### 4. SLURM job script (example for ModelNet40)
Save as `run_mn40.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=asp_mn40
#SBATCH --output=logs/mn40_%j.out
#SBATCH --error=logs/mn40_%j.err

#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load anaconda/2023
conda activate aspsnn

cd /path/to/ASP-SNN
python mn40_full.py
```

Submit:
```bash
mkdir -p logs
sbatch run_mn40.sh
```

### 5. Run all five in parallel
```bash
for script in mn40_full mn10_full scanobjectnn_full shapenetpart_full s3dis_full; do
    sed "s/asp_mn40/asp_${script}/g; s/mn40_%j/${script}_%j/g; \
         s/mn40_full.py/${script}.py/g" run_mn40.sh > run_${script}.sh
    sbatch run_${script}.sh
done
```

### 6. Checkpoint resume
All scripts save `*_latest.pth` every epoch. If your job is killed, just resubmit —
the script automatically resumes from the latest checkpoint with no code changes needed.

---

## Reading results

After a run, open the checkpoint directory and look at:
- `results_mn40.json` — final OA, history, energy estimate
- `asp_best.pth` — best checkpoint (by val OA)
- `*.csv` or `*.json` log files (ShapeNetPart / S3DIS)

The final printed line will look like:
```
  ASP  OA: 91.87%  (avg 2.91/4 slices, TTA=10)
  Est. energy vs ANN: 4.1%  (fr≈0.15, Loihi 2 constants)
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `purdueprj not found` | Enable internet in Kaggle (auto-clone) or `git clone` manually |
| `ScanObjectNN h5 not found` | Attach the h5 files as a Kaggle dataset (gdown mirrors are unreliable) |
| `CUDA out of memory` | Reduce `BATCH` in the hyperparameters section at the top of the script |
| Script hangs on data download | Use `--time=08:00:00` on SLURM; download can be slow on first run |
| Resume not working | Delete `*_latest.pth` and restart from scratch |

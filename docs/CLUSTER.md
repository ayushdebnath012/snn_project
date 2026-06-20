# ModelNet10/40 cluster guide

## Prepare once on the login node

```bash
git clone https://github.com/ayushdebnath012/ASP-SNN.git
cd ASP-SNN

module load cuda/12.1  # replace with your site's CUDA module
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tools/validate_repo.py
```

Do not train on the login node.

## Configure shared paths

```bash
export MODELNET10_DIR=/shared/datasets/ModelNet10
export MODELNET40_DIR=/shared/datasets/ModelNet40
export SPIKEGAT_OUTPUT_ROOT=$SCRATCH/spikegat
export SPIKEGAT_VENV=$PWD/.venv
```

Dataset roots must contain class-level `train/` and `test/` directories. Keep
checkpoints on persistent scratch/shared storage rather than node-local `/tmp`.

## Submit

Edit the partition, wall time, memory, and module assumptions in the two SLURM
files to match your site, then run:

```bash
sbatch --export=ALL scripts/slurm/spikegat_mn10.sbatch
sbatch --export=ALL scripts/slurm/spikegat_mn40.sbatch
```

Or submit both:

```bash
bash scripts/slurm/submit_all.sh
```

## Interactive GPU run

```bash
source .venv/bin/activate
export MODELNET40_DIR=/shared/datasets/ModelNet40
export SPIKEGAT_CKPT_DIR=$SCRATCH/spikegat/mn40
export BATCH_SIZE=32 NUM_WORKERS=4
python -u experiments/full/train_spikegat_modelnet40.py
```

## Resume and monitoring

Rerun or resubmit the same job to resume. Do not change architecture or epoch
schedule while resuming an existing optimizer/scheduler checkpoint.

```bash
squeue -u "$USER"
tail -f slurm-spikegat-mn40-<job-id>.out
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The jobs are single-GPU. Allocating additional GPUs will not accelerate them.

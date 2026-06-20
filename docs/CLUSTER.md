# Cluster and SLURM guide

## 1. Prepare the environment

Use a login node for cloning and dependency installation. Do not run training
on the login node.

```bash
git clone https://github.com/ayushdebnath012/ASP-SNN.git
cd ASP-SNN
module load cuda/12.1  # use the module available at your site

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tools/validate_repo.py --imports
```

Alternatively, use `conda env create -f environment.yml`.

## 2. Place datasets on shared storage

ModelNet directories must contain class folders with `train/` and `test/`
subdirectories. Set paths through environment variables rather than editing
source files:

```bash
export MODELNET10_DIR=/shared/datasets/ModelNet10
export MODELNET40_DIR=/shared/datasets/ModelNet40
export SCANOBJECTNN_DIR=/shared/datasets/ScanObjectNN/main_split
export SHAPENETPART_DIR=/shared/datasets/shapenet_part_seg_hdf5_data
export S3DIS_DIR=/shared/datasets/s3dis
```

Use `$SCRATCH` or another persistent high-throughput filesystem for checkpoints.
Node-local `/tmp` is appropriate only for copied dataset caches, not the sole
checkpoint location.

## 3. Submit SpikeGAT

Edit partition, account, time, and module names in the `.sbatch` files once.

```bash
export MODELNET40_DIR=/shared/datasets/ModelNet40
export ASP_SNN_OUTPUT_ROOT=$SCRATCH/asp-snn
sbatch scripts/slurm/spikegat_mn40.sbatch

export MODELNET10_DIR=/shared/datasets/ModelNet10
sbatch scripts/slurm/spikegat_mn10.sbatch
```

Direct interactive run:

```bash
source .venv/bin/activate
export MODELNET40_DIR=/shared/datasets/ModelNet40
export SPIKEGAT_CKPT_DIR=$SCRATCH/asp-snn/spikegat_mn40
export BATCH_SIZE=32 NUM_WORKERS=4
python -u experiments/kaggle/spikegat/modelnet40.py
```

Rerunning the command resumes automatically from the latest complete
checkpoint. Do not change epoch counts or architecture halfway through a run.

## 4. Submit the reusable ASP tasks

```bash
sbatch scripts/slurm/scanobjectnn.sbatch
sbatch scripts/slurm/shapenetpart.sbatch
sbatch scripts/slurm/s3dis.sbatch
```

For A100/H100 ModelNet training:

```bash
sbatch scripts/slurm/modelnet_a100.sbatch
```

## 5. Monitor and recover

```bash
squeue -u "$USER"
tail -f slurm-<job-name>-<job-id>.out
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

If a job reaches its wall-time limit, resubmit the same script. Checkpoint
directories are deterministic and the runners restore model, optimizer,
scheduler, scaler, epoch, and best metric where supported.

## 6. Multi-GPU note

The supplied SLURM jobs request one GPU. Do not request multiple GPUs unless the
selected entrypoint explicitly supports DDP/DataParallel; extra allocated GPUs
would otherwise remain idle.

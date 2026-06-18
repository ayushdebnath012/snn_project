"""
kaggle_s3dis_full.py — Full ASP-SNN run on S3DIS Area 5 scene segmentation
(13 semantic classes, train on Areas 1,2,3,4,6 — test on Area 5).

HOW TO USE ON KAGGLE
--------------------
1. Upload this project as a Kaggle dataset and attach it to the notebook.
2. The S3DIS preprocessed .npy room files are downloaded automatically.
   Source: OpenPoints / HuggingFace public preprocessed tarball via gdown.
   If gdown fails, attach your own Kaggle dataset with the .npy files:
     Expected layout: Area_1/room_name.npy ... Area_6/room_name.npy
     Each .npy: [N, 7]  x,y,z,r,g,b,label (RGB 0-255, label 0-12)
3. Run the script. Training takes ~4-6h on T4 (100 epochs, batch=64).

Metrics (standard S3DIS Area 5 protocol):
  mIoU  — mean IoU over 13 classes
  OA    — overall accuracy
  mAcc  — mean per-class accuracy

Training targets:
  mIoU:  55-62%  (ASP-SNN, T=6, 100 epochs)
  Reference: PointNet 47.6%, PointNet++ 54.5%, PointTransformer 70.4%

Full run: 100 epochs, AdamW + cosine LR, AMP (bf16 on Ampere+),
          Gumbel annealing, S3DIS-specific augmentation.
Expected runtime on T4: ~6h   on V100/A100: ~3h
"""

# ── 0. Install dependencies ────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pip("h5py", "gdown", "pyyaml")

import json, math, time, tarfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.is_bf16_supported():
        print("bfloat16 AMP supported")

# ── 1. Locate project root ─────────────────────────────────────────────────────
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs"
)
os.makedirs(WORK, exist_ok=True)

def _find_proj_root(sentinel="train_s3dis.py"):
    """Walk /kaggle/input (any depth) looking for the sentinel train script."""
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if sentinel in files:
                return root
    # Fallback: script running from inside the extracted project directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    if os.path.isfile(os.path.join(script_dir, sentinel)):
        return script_dir
    # Final fallback: clone from GitHub (no dataset attachment needed)
    clone_dir = "/kaggle/working/ASP-SNN"
    if not os.path.isdir(clone_dir):
        print("Project not in /kaggle/input — cloning from GitHub ...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "--branch", "codex/fix-shapenet-h5-conversion",
            "https://github.com/AryaPawa/ASP-SNN.git", clone_dir,
        ], check=True)
    if os.path.isfile(os.path.join(clone_dir, sentinel)):
        return clone_dir
    return None

PROJ = None
if ON_KAGGLE:
    PROJ = _find_proj_root("train_s3dis.py")
    if PROJ is None:
        raise RuntimeError(
            "Project not found in /kaggle/input/.\n"
            "Attach the Kaggle dataset containing this project.\n"
            "Searched recursively for train_s3dis.py."
        )
else:
    try:
        PROJ = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        PROJ = os.getcwd()

print(f"Project root: {PROJ}")
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
os.chdir(PROJ)

# ── 2. Download / locate S3DIS preprocessed .npy rooms ────────────────────────
DATA_ROOT = os.path.join(WORK, "data")
S3DIS_DIR = os.path.join(DATA_ROOT, "s3dis")
os.makedirs(S3DIS_DIR, exist_ok=True)

def _s3dis_ready(d):
    """Check that at least Area 1 and Area 5 exist with .npy files."""
    import glob
    for area in [1, 5]:
        area_dir = os.path.join(d, f"Area_{area}")
        flat_pat = os.path.join(d, f"Area_{area}_*.npy")
        flat_raw = os.path.join(d, "raw", f"Area_{area}_*.npy")
        if os.path.isdir(area_dir) and glob.glob(os.path.join(area_dir, "*.npy")):
            continue
        if glob.glob(flat_pat) or glob.glob(flat_raw):
            continue
        return False
    return True

def download_s3dis():
    if _s3dis_ready(S3DIS_DIR):
        print(f"S3DIS already present at {S3DIS_DIR}")
        return True

    # Check Kaggle input datasets
    if ON_KAGGLE:
        for ds in sorted(os.listdir("/kaggle/input")):
            cand = f"/kaggle/input/{ds}"
            if _s3dis_ready(cand):
                import shutil
                print(f"S3DIS found in Kaggle input: {cand}")
                if not _s3dis_ready(S3DIS_DIR):
                    shutil.copytree(cand, S3DIS_DIR, dirs_exist_ok=True)
                return True
            # Check one level deeper (e.g., /kaggle/input/s3dis/s3dis/)
            for sub in os.listdir(cand):
                subcand = os.path.join(cand, sub)
                if os.path.isdir(subcand) and _s3dis_ready(subcand):
                    import shutil
                    print(f"S3DIS found at: {subcand}")
                    shutil.copytree(subcand, S3DIS_DIR, dirs_exist_ok=True)
                    return True

    # Try OpenPoints preprocessed S3DIS (flat .npy layout, all areas)
    # Google Drive: s3disfull.tar — preprocessed by OpenPoints team
    # File ID: 1MX3ZjFSQRbmOOnRoZTu_mxDl9hPXlEZv  (if still available)
    OPENPOINTS_DRIVE_IDS = [
        "1MX3ZjFSQRbmOOnRoZTu_mxDl9hPXlEZv",   # primary: s3disfull.tar
        "1H9Ep76l8KkUpwILY-13owL-ACLY-afLY",    # alternate mirror
    ]
    tar_path = os.path.join(DATA_ROOT, "s3dis.tar")
    import gdown
    for gid in OPENPOINTS_DRIVE_IDS:
        try:
            print(f"Trying gdown id={gid} ...")
            gdown.download(id=gid, output=tar_path, quiet=False)
            if os.path.isfile(tar_path) and os.path.getsize(tar_path) > 10_000_000:
                print("Extracting S3DIS tar ...")
                with tarfile.open(tar_path, "r:*") as tf:
                    tf.extractall(DATA_ROOT)
                os.remove(tar_path)
                # Find where it was extracted
                for sub in os.listdir(DATA_ROOT):
                    subcand = os.path.join(DATA_ROOT, sub)
                    if os.path.isdir(subcand) and _s3dis_ready(subcand):
                        if subcand != S3DIS_DIR:
                            import shutil
                            shutil.copytree(subcand, S3DIS_DIR, dirs_exist_ok=True)
                        print(f"S3DIS extracted → {S3DIS_DIR}")
                        return True
                # Try the raw/ subfolder layout
                raw_sub = os.path.join(DATA_ROOT, "s3dis", "raw")
                if os.path.isdir(raw_sub):
                    import shutil
                    shutil.copytree(os.path.join(DATA_ROOT, "s3dis"), S3DIS_DIR, dirs_exist_ok=True)
                    if _s3dis_ready(S3DIS_DIR):
                        return True
        except Exception as e:
            print(f"  gdown id={gid} failed: {e}")
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)

    # Also try datasets/download.py which has its own gdown logic
    try:
        print("Trying datasets/download.py --s3dis ...")
        result = subprocess.run([
            sys.executable, "datasets/download.py", "--s3dis",
        ], cwd=PROJ, capture_output=False)
        # download.py puts data in <PROJ>/data/s3dis
        proj_s3dis = os.path.join(PROJ, "data", "s3dis")
        if _s3dis_ready(proj_s3dis):
            import shutil
            shutil.copytree(proj_s3dis, S3DIS_DIR, dirs_exist_ok=True)
            return True
    except Exception as e:
        print(f"datasets/download.py failed: {e}")

    return False

if not download_s3dis():
    print(
        "\n[WARNING] Could not download S3DIS automatically.\n"
        "Please attach a Kaggle dataset containing the preprocessed S3DIS .npy files.\n"
        "Expected layout in the dataset:\n"
        "  Area_1/room_name.npy  ...  Area_6/room_name.npy\n"
        "Each file: [N, 7] float32 — x, y, z, r, g, b, label (label in 0-12)\n"
        "OR flat layout: Area_1_room_name.npy in a single folder."
    )
    # Try to proceed anyway — the train script will give a clear error
    S3DIS_DIR = os.path.join(PROJ, "data", "s3dis")

print(f"S3DIS data dir: {S3DIS_DIR}")

# ── 3. Build config ────────────────────────────────────────────────────────────
# Detect GPU memory to scale batch size
def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9

gpu_gb   = gpu_mem_gb()
batch_sz = 64 if gpu_gb >= 15 else 32 if gpu_gb >= 8 else 16
print(f"GPU mem: {gpu_gb:.1f} GB → batch size: {batch_sz}")

class Cfg:
    # Dataset
    dataset        = "s3dis"
    data_dir       = S3DIS_DIR
    num_points     = 4096
    num_classes    = 13
    use_category   = False
    test_area      = 5
    block_size     = 1.0
    use_rgb        = True
    use_height     = True
    input_channels = 7      # xyz(3) + rgb(3) + height(1)
    # Slicing
    num_slices       = 16
    points_per_slice = 256
    geo_dim          = 8
    # Model
    feat_dim       = 512
    hidden_dim     = 512
    slice_pool     = "meanmax"
    slice_token_dropout = 0.05
    context_ensemble = 2
    k_edge         = 20
    transformer_heads = 4
    transformer_ffn_dim = 1024
    num_lif_layers = 3
    lif_leak       = 0.9
    lif_threshold  = 1.0
    d_ssp          = 128
    T              = 6
    point_feat_dim = 64
    seg_head_dropout = 0.1
    in_channels    = 7      # matches input_channels
    point_in_channels = 7
    # Training
    epochs         = 500     # 500 epochs for full convergence
    batch_size     = batch_sz
    lr             = 5e-4   # was 2e-3 — lower LR reduces ±5% mIoU oscillation
    weight_decay   = 0.01
    grad_clip      = 0.5   # was 1.0 — tighter clipping for S3DIS stability
    warmup_epochs  = 20    # was 5 — longer warmup before hitting full LR
    use_amp        = True
    use_class_weights = True
    # Gumbel
    tau_start      = 1.0
    tau_end        = 0.1
    tau_decay      = 0.985   # was 0.95 — tau hit min at ep~43/500; now hits min ~ep155/250
    tau_anneal_epochs = 350  # scaled up with 500-epoch run
    # Augmentation
    aug_rotate_z   = True
    aug_scale_lo   = 0.9
    aug_scale_hi   = 1.1
    aug_translate  = 0.1
    aug_jitter_sigma = 0.005
    aug_jitter_clip  = 0.02
    aug_color_drop   = 0.2
    aug_color_jitter = 0.05
    aug_anisotropic_scale = False
    aug_tilt       = 0.0
    aug_elastic    = True
    aug_elastic_strength  = 0.015
    aug_elastic_clip      = 0.03
    aug_elastic_bandwidth = 0.35
    aug_elastic_anchors   = 12
    aug_elastic_z_scale   = 0.2
    # Logging
    eval_interval  = 5
    log_dir        = os.path.join(WORK, "logs")
    ckpt_dir       = os.path.join(WORK, "s3dis_ckpts")
    seed           = 42
    num_workers    = 4 if not ON_KAGGLE else 2
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)}

cfg = Cfg()
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

# ── 4. Launch training via train_s3dis.py ─────────────────────────────────────
print(f"\n{'='*70}")
print("Launching train_s3dis.py (full-featured training script)")
print(f"  Protocol: train Areas 1,2,3,4,6 — test Area {cfg.test_area}")
print(f"  Epochs: {cfg.epochs}  Batch: {cfg.batch_size}  T: {cfg.T}  AMP: {cfg.use_amp}")
print(f"  Data:   {cfg.data_dir}")
print(f"  Output: {cfg.ckpt_dir}")
print(f"{'='*70}\n")

cmd = [
    sys.executable, "train_s3dis.py",
    "--config", "configs/s3dis_seg.yaml",
    "--set",
    f"data_dir={cfg.data_dir}",
    f"log_dir={cfg.log_dir}",
    f"ckpt_dir={cfg.ckpt_dir}",
    f"epochs={cfg.epochs}",   # 250
    f"batch_size={cfg.batch_size}",
    f"num_workers={cfg.num_workers}",
    f"seed={cfg.seed}",
    f"T={cfg.T}",
    f"use_amp={cfg.use_amp}",
    f"eval_interval={cfg.eval_interval}",
    f"test_area={cfg.test_area}",
    "kd_teacher_epochs=50",
    "kd_temp=4.0",
    "kd_lam=0.3",
    f"lr={cfg.lr}",
    f"grad_clip={cfg.grad_clip}",
    f"warmup_epochs={cfg.warmup_epochs}",
    f"tau_decay={cfg.tau_decay}",
    f"tau_anneal_epochs={cfg.tau_anneal_epochs}",
]

result = subprocess.run(cmd, cwd=PROJ)

if result.returncode != 0:
    print(f"\n[WARNING] train_s3dis.py exited with code {result.returncode}")
    print("Check logs above for details.")
else:
    print(f"\nTraining complete.")

# ── 5. Print final results ─────────────────────────────────────────────────────
best_path = os.path.join(cfg.ckpt_dir, "s3dis_best.pt")
if not os.path.isfile(best_path):
    # Also check the project's default ckpt dir
    best_path = os.path.join(PROJ, "checkpoints", "s3dis_best.pt")

if os.path.isfile(best_path):
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"\n{'='*70}")
    print("FINAL RESULTS — ASP-SNN on S3DIS (Area 5)")
    print(f"{'='*70}")
    for key, label in [
        ("best_metric",    "Best val mIoU"),
        ("test_miou",      "Test mIoU"),
        ("test_macc",      "Test mAcc"),
        ("test_oa",        "Test OA"),
    ]:
        if key in ckpt:
            print(f"  {label:20s}: {ckpt[key]*100:.2f}%")
    print(f"\n  Reference targets (Area 5):")
    print(f"    PointNet:        47.6% mIoU")
    print(f"    PointNet++:      54.5% mIoU")
    print(f"    PointTransformer: 70.4% mIoU")
    print(f"\n  Checkpoint: {best_path}")
else:
    # Try to find any checkpoint
    import glob
    ckpts = sorted(glob.glob(os.path.join(cfg.ckpt_dir, "*.pt")))
    if ckpts:
        print(f"\nCheckpoints found: {ckpts}")
    else:
        print(f"\nNo checkpoint found at {best_path}")
        print("Training may have been interrupted.")

print(f"\nLog dir:   {cfg.log_dir}")
print(f"Ckpt dir:  {cfg.ckpt_dir}")

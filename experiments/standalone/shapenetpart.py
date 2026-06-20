"""
kaggle_shapenetpart_full.py — Full ASP-SNN run on ShapeNetPart
(16 categories, 50 part labels, 14007 train / 2874 test shapes).

HOW TO USE ON KAGGLE
--------------------
1. Upload this project as a Kaggle dataset and attach it to the notebook.
2. The ShapeNetPart HDF5 data is downloaded automatically from:
     - Stanford direct URL (primary)
     - Google Drive mirror via gdown (fallback)
     - Raw Kaggle dataset 'mitkir/shapenet' + conversion (second fallback)
3. Run the script. No other configuration needed.

Metrics reported (official ShapeNetPart protocol):
  - Instance mIoU (IoU averaged per shape, then averaged over all shapes)
  - Class mIoU    (IoU averaged per class, then averaged over 16 classes)

Training targets:
  Instance mIoU: 85–86%  (ASP-SNN with LIF temporal backend, T=8, 250 epochs)
  Reference:     ~85.1%  (PointNet baseline), ~86.4%  (PointNet++)

Full run: 250 epochs, AdamW + cosine LR + warmup, balanced category sampling,
          AMP (bf16 on Ampere+), Gumbel annealing.
Expected runtime on T4: ~8h   on V100/A100: ~4h
"""

# ── 0. Install dependencies ────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pip("h5py", "gdown", "pyyaml", "kagglehub")

import json, math, time, warnings, zipfile, urllib.request
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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs", "standalone_shapenetpart"
)
os.makedirs(WORK, exist_ok=True)

def _find_proj_root(sentinel="config.py"):
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
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    if os.path.isfile(os.path.join(repo_root, sentinel)):
        return repo_root
    # Final fallback: clone from GitHub (no dataset attachment needed)
    clone_dir = "/kaggle/working/ASP-SNN"
    if not os.path.isdir(clone_dir):
        print("Project not in /kaggle/input — cloning from GitHub ...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "--branch", "main",
            "https://github.com/ayushdebnath012/ASP-SNN.git", clone_dir,
        ], check=True)
    if os.path.isfile(os.path.join(clone_dir, sentinel)):
        return clone_dir
    return None

PROJ = None
if ON_KAGGLE:
    PROJ = _find_proj_root("config.py")
    if PROJ is None:
        raise RuntimeError(
            "Project not found in /kaggle/input/.\n"
            "Attach the Kaggle dataset containing this project.\n"
            "Searched recursively for config.py."
        )
else:
    try:
        PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        PROJ = os.getcwd()

print(f"Project root: {PROJ}")
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
os.chdir(PROJ)

# ── 2. Download ShapeNetPart HDF5 ─────────────────────────────────────────────
DATA_ROOT = os.path.join(WORK, "data")
HDF5_DIR  = os.path.join(DATA_ROOT, "shapenet_part_seg_hdf5_data")
os.makedirs(DATA_ROOT, exist_ok=True)

STANFORD_URL   = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
GDRIVE_FILE_ID = "1tEnSGAdgfp-NPVS5y_ALD8eF18bzwhM_"   # public PointNeXt mirror

def _hdf5_ready(d):
    import glob
    return (os.path.exists(os.path.join(d, "all_object_categories.txt"))
            and glob.glob(os.path.join(d, "train*.h5"))
            and glob.glob(os.path.join(d, "test*.h5")))

def _extract_zip(zip_path, dst):
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)
    os.remove(zip_path)

def download_shapenet_hdf5():
    if _hdf5_ready(HDF5_DIR):
        print(f"ShapeNetPart HDF5 already present at {HDF5_DIR}")
        return True

    # Check Kaggle input datasets first
    if ON_KAGGLE:
        for ds in sorted(os.listdir("/kaggle/input")):
            cand = f"/kaggle/input/{ds}"
            import glob as _glob
            if (os.path.isfile(os.path.join(cand, "all_object_categories.txt"))
                    and _glob.glob(os.path.join(cand, "train*.h5"))):
                # Symlink or copy
                import shutil
                if not os.path.exists(HDF5_DIR):
                    shutil.copytree(cand, HDF5_DIR)
                print(f"ShapeNetPart HDF5 found in Kaggle input: {cand}")
                return True

    zip_path = os.path.join(DATA_ROOT, "shapenet_part_seg_hdf5_data.zip")

    # Method 1: Stanford direct
    print(f"Downloading ShapeNetPart from Stanford ...")
    try:
        urllib.request.urlretrieve(STANFORD_URL, zip_path)
        _extract_zip(zip_path, DATA_ROOT)
        if _hdf5_ready(HDF5_DIR):
            print("Done (Stanford)")
            return True
    except Exception as e:
        print(f"Stanford failed: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # Method 2: gdown Google Drive mirror
    print("Trying Google Drive mirror via gdown ...")
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False)
        if os.path.isfile(zip_path) and os.path.getsize(zip_path) > 1_000_000:
            _extract_zip(zip_path, DATA_ROOT)
            if _hdf5_ready(HDF5_DIR):
                print("Done (gdown)")
                return True
    except Exception as e:
        print(f"gdown failed: {e}")

    # Method 3: Kaggle raw mirror + conversion
    print("Trying Kaggle raw mirror 'mitkir/shapenet' + conversion ...")
    try:
        import kagglehub
        raw_path = kagglehub.dataset_download("mitkir/shapenet")
        # Find synsetoffset2category.txt
        for root_dir, dirs, files in os.walk(raw_path):
            if "synsetoffset2category.txt" in files:
                conv_script = os.path.join(PROJ, "datasets", "convert_shapenet_raw.py")
                subprocess.run([
                    sys.executable, conv_script,
                    "--raw_dir", root_dir,
                    "--out_dir", HDF5_DIR,
                ], check=True, cwd=PROJ)
                if _hdf5_ready(HDF5_DIR):
                    print("Done (Kaggle raw → converted)")
                    return True
                break
    except Exception as e:
        print(f"Kaggle raw conversion failed: {e}")

    return False

if not download_shapenet_hdf5():
    raise RuntimeError(
        "Could not download ShapeNetPart HDF5 data.\n"
        "Please attach it as a Kaggle input dataset, or download manually from:\n"
        f"  {STANFORD_URL}"
    )

print(f"ShapeNetPart HDF5 dir: {HDF5_DIR}")

# ── 3. Build config ────────────────────────────────────────────────────────────
class Cfg:
    # Dataset
    dataset          = "shapenetpart"
    data_dir         = HDF5_DIR
    num_points       = 2048
    num_classes      = 50
    num_categories   = 16
    use_category     = True
    in_channels      = 6
    point_in_channels = 6
    use_normals      = True
    val_fraction     = 0.1
    # Slicing
    num_slices       = 16
    points_per_slice = 128
    geo_dim          = 8
    # Model
    feat_dim         = 512
    hidden_dim       = 512
    slice_pool       = "meanmax"
    slice_token_dropout = 0.05
    context_ensemble = 3
    k_edge           = 20
    transformer_heads = 4
    transformer_ffn_dim = 1024
    num_lif_layers   = 3
    lif_leak         = 0.9
    lif_threshold    = 1.0
    d_ssp            = 128
    T                = 8
    temporal_backend = "lif"
    point_feat_dim   = 128
    seg_head_dropout = 0.1
    # Training
    epochs           = 500     # 500 epochs for full convergence
    batch_size       = 32
    lr               = 3e-4
    encoder_lr_scale = 1.0
    weight_decay     = 0.01
    grad_clip        = 1.0
    warmup_epochs    = 10
    use_amp          = True
    balanced_sampling = True
    balanced_sampling_power = 0.5
    early_stopping_patience = 50
    grad_accum_steps = 1
    use_compile      = False
    prefetch_factor  = 2
    num_workers      = 4
    # Gumbel
    tau_start        = 1.0
    tau_end          = 0.1
    tau_anneal_epochs = 200
    # Augmentation
    aug_rotate_so3   = True    # full SO3 rotation (was False)
    aug_rotate_z     = False   # SO3 supersedes z-only rotation
    aug_scale_lo     = 0.8
    aug_scale_hi     = 1.25
    aug_translate    = 0.1
    aug_jitter_sigma = 0.01
    aug_jitter_clip  = 0.05
    aug_point_dropout = 0.1
    aug_slice_dropout = 0.05
    aug_anisotropic_scale = True
    aug_tilt         = 0.05
    aug_elastic      = False
    # Logging
    eval_interval    = 5
    test_t_values    = [6, 8, 12, 16]
    log_dir          = os.path.join(WORK, "logs")
    ckpt_dir         = os.path.join(WORK, "shapenet_ckpts")
    seed             = 42
    checkpoint_interval = 5
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)}

cfg = Cfg()
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# ── 4. Launch training via subprocess (uses train_shapenet.py directly) ────────
# This is the cleanest path: reuse the full-featured train script
# with overrides passed as --set key=value flags.

print(f"\n{'='*70}")
print("Launching train_shapenet.py (full-featured training script)")
print(f"  Epochs: {cfg.epochs}  T: {cfg.T}  Batch: {cfg.batch_size}  AMP: {cfg.use_amp}")
print(f"  Data:   {cfg.data_dir}")
print(f"  Output: {cfg.ckpt_dir}")
print(f"{'='*70}\n")

cmd = [
    sys.executable, "train_shapenet.py",
    "--config", "configs/shapenet_seg.yaml",
    "--set",
    f"data_dir={cfg.data_dir}",
    f"log_dir={cfg.log_dir}",
    f"ckpt_dir={cfg.ckpt_dir}",
    f"epochs={cfg.epochs}",   # 500
    f"batch_size={cfg.batch_size}",
    f"num_workers={cfg.num_workers}",
    f"seed={cfg.seed}",
    f"T={cfg.T}",
    f"use_amp={cfg.use_amp}",
    f"eval_interval={cfg.eval_interval}",
    f"checkpoint_interval={cfg.checkpoint_interval}",
    f"early_stopping_patience={cfg.early_stopping_patience}",
    "kd_teacher_epochs=50",
    "kd_temp=4.0",
    "kd_lam=0.3",
    "aug_rotate_so3=True",
    "aug_rotate_z=False",
]

result = subprocess.run(cmd, cwd=PROJ)

if result.returncode != 0:
    print(f"\n[WARNING] train_shapenet.py exited with code {result.returncode}")
    print("Check logs above for details.")
else:
    print(f"\nTraining complete.")

# ── 5. Print final results ─────────────────────────────────────────────────────
best_path = os.path.join(cfg.ckpt_dir, "shapenet_best.pt")
if os.path.isfile(best_path):
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"\n{'='*70}")
    print("FINAL RESULTS — ASP-SNN on ShapeNetPart")
    print(f"{'='*70}")
    print(f"  Best val inst mIoU:  {ckpt.get('best_metric', 0)*100:.2f}%")
    if "test_inst_iou" in ckpt:
        print(f"  Test inst mIoU:      {ckpt['test_inst_iou']*100:.2f}%")
    if "test_cls_iou" in ckpt:
        print(f"  Test cls  mIoU:      {ckpt['test_cls_iou']*100:.2f}%")
    if "test_results" in ckpt:
        print(f"\n  Per-T test results:")
        for t_val, tr in ckpt["test_results"].items():
            print(f"    T={t_val:2s}: Inst={tr['inst_miou']*100:.2f}%  "
                  f"Cls={tr['cls_miou']*100:.2f}%  "
                  f"coverage={tr.get('slice_coverage',0)*100:.1f}%")
    print(f"\n  Reference targets:")
    print(f"    PointNet:   85.1% inst mIoU")
    print(f"    PointNet++: 86.4% inst mIoU")
    print(f"\n  Checkpoint: {best_path}")
else:
    print(f"\nNo checkpoint found at {best_path} — training may have been interrupted.")

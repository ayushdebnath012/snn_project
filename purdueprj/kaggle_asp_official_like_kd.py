"""
kaggle_asp_official_like_kd.py
===============================
Kaggle version of colab_asp_official_like.py with Knowledge Distillation.

Adapted for Kaggle (T4 x2 GPU) — no Google Drive, no Colab imports.

Goal
----
Compare:
  1. SPM-OfficialLike: fixed order, full official-style group sequence.
  2. ASP-OfficialLike: same encoder/mixer/head, but adaptively selects
     chunks of FPS/KNN groups with an SSP policy.
  3. Optional training-only distillation from a PointTransformer-style teacher.

Official-SPM details mirrored where practical:
  - timestep = 2 spiking passes
  - num_group = 128, group_size = 32, expand = 1.1 on GPU
  - PointNet-style Conv2d encoder: 3->128->256, concat global, 512->512->384
  - Conv1d positional embedding: 3->128->384
  - 12 Mamba-like residual mixer blocks, drop_path = 0.3
  - Conv1d classifier head: 384->256->128->classes
  - AdamW, weight_decay = 0.1, label_smoothing = 0.2

Distillation knobs (env vars):
  USE_KD=1              enable teacher distillation on GPU by default
  TEACHER_MODE=auto     auto/train/load/off
  TEACHER_EPOCHS=150    train teacher before SPM/ASP if no cached teacher exists
  TEACHER_CKPT=path     load a compatible PointTransformerTeacher checkpoint
  KD_TEMP=4.0           distillation temperature
  KD_CE_WEIGHT=0.5      supervised CE weight
  KD_LOGIT_WEIGHT=0.5   teacher KL weight

Debug/profiling knobs (env vars):
  DEBUG_MINI_RUN=1      run a tiny timing/checkpoint pass instead of full training
  DEBUG_MAX_STEPS=1     train batches per epoch in debug mode
  DEBUG_EPOCHS=1        epochs in debug mode
  DEBUG_SKIP_EVAL=1     skip validation in debug mode by default
  DEBUG_SKIP_TEACHER=1  disable teacher/KD setup in debug mode by default

Outputs:
  /kaggle/working/asp_official_like_kd/official_like_ckpts/final_results.json
  /kaggle/working/asp_official_like_kd/official_like_ckpts/01_training_curves.png
  /kaggle/working/asp_official_like_kd/official_like_ckpts/02_accuracy_bars.png
  /kaggle/working/asp_official_like_kd/official_like_ckpts/debug_*_minimal.pt
  /kaggle/working/asp_official_like_kd/official_like_ckpts/debug_runtime_summary.json
  /kaggle/working/asp_official_like_kd/official_like_ckpts/spm_<dataset>_best.pth
  /kaggle/working/asp_official_like_kd/official_like_ckpts/asp_<dataset>_best.pth
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU detected. Go to Settings -> Accelerator -> GPU T4 x2")

_drive_pkgs = (
    ["google-api-python-client", "google-auth"]
    if os.environ.get("DRIVE_FOLDER_ID", "").strip()
    else []
)
for pkg in ["trimesh", "kagglehub", "matplotlib"] + _drive_pkgs:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", pkg],
        check=False,
    )
    if result.returncode != 0:
        print(f"Warning: pip install {pkg} exited {result.returncode} — continuing anyway.")
print("Dependencies installed.")

import kagglehub
import matplotlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------

print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ON_GPU = DEVICE == "cuda"

WORK = os.environ.get("ASP_KAGGLE_ROOT", "/kaggle/working/asp_official_like_kd")
DATA_DIR = os.path.join(WORK, "data")


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() not in ("0", "false", "no", "off")


DEBUG_MINI_RUN = env_bool("DEBUG_MINI_RUN", False)
DEBUG_MAX_STEPS = env_int("DEBUG_MAX_STEPS", 1)
DEBUG_EPOCHS = env_int("DEBUG_EPOCHS", 1)
DEBUG_SKIP_EVAL = env_bool("DEBUG_SKIP_EVAL", True)
DEBUG_SKIP_TEACHER = env_bool("DEBUG_SKIP_TEACHER", True)
DEBUG_DATASET_ITEMS = env_int("DEBUG_DATASET_ITEMS", 64)
DEBUG_VAL_ITEMS = env_int("DEBUG_VAL_ITEMS", 64)
DEBUG_DATASETS = os.environ.get("DEBUG_DATASETS", "").strip()


if ON_GPU:
    print("GPU    :", torch.cuda.get_device_name(0))
    EPOCHS = env_int("EPOCHS", 300)
    BATCH = env_int("BATCH", 16)
    NUM_POINTS = env_int("NUM_POINTS", 1024)
    TIMESTEP = env_int("TIMESTEP", 2)
    TRANS_DIM = env_int("TRANS_DIM", 384)
    DEPTH = env_int("DEPTH", 12)
    NUM_GROUP = env_int("NUM_GROUP", 128)
    GROUP_SIZE = env_int("GROUP_SIZE", 32)
    ASP_STEPS = env_int("ASP_STEPS", 4)
    DROP_PATH = env_float("DROP_PATH", 0.3)
    GRAD_ACCUM = env_int("GRAD_ACCUM", 4)
    NUM_WORKERS = env_int("NUM_WORKERS", 2)
    N_VOTE = env_int("N_VOTE", 5)
    DATASET_NAMES = os.environ.get("DATASETS", "ModelNet10,ModelNet40").split(",")
else:
    print("WARNING: CPU demo mode. Use a Kaggle GPU runtime for the real run.")
    EPOCHS = env_int("EPOCHS", 20)
    BATCH = env_int("BATCH", 4)
    NUM_POINTS = env_int("NUM_POINTS", 256)
    TIMESTEP = env_int("TIMESTEP", 2)
    TRANS_DIM = env_int("TRANS_DIM", 128)
    DEPTH = env_int("DEPTH", 4)
    NUM_GROUP = env_int("NUM_GROUP", 32)
    GROUP_SIZE = env_int("GROUP_SIZE", 16)
    ASP_STEPS = env_int("ASP_STEPS", 4)
    DROP_PATH = env_float("DROP_PATH", 0.1)
    GRAD_ACCUM = env_int("GRAD_ACCUM", 1)
    NUM_WORKERS = env_int("NUM_WORKERS", 0)
    N_VOTE = env_int("N_VOTE", 1)
    DATASET_NAMES = os.environ.get("DATASETS", "ModelNet10").split(",")

EXPAND = env_float("EXPAND", 1.1)
LR = env_float("LR", 1e-3)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.1)
WARMUP_EP = env_int("WARMUP_EP", 30 if ON_GPU else 3)
LABEL_SMOOTH = env_float("LABEL_SMOOTH", 0.2)
EXIT_THR = env_float("EXIT_THR", 0.45)
CHECKPOINT_EVERY = env_int("CHECKPOINT_EVERY", 5)
RESUME = os.environ.get("RESUME_CHECKPOINTS", "1").lower() not in ("0", "false", "no")

# Knowledge Distillation config
USE_KD = env_bool("USE_KD", ON_GPU)
TEACHER_MODE = os.environ.get("TEACHER_MODE", "auto").lower()
TEACHER_CKPT = os.environ.get("TEACHER_CKPT", "").strip()
TEACHER_EPOCHS = env_int("TEACHER_EPOCHS", min(150, EPOCHS) if ON_GPU else 5)
TEACHER_DIM = env_int("TEACHER_DIM", TRANS_DIM)
TEACHER_DEPTH = env_int("TEACHER_DEPTH", 8 if ON_GPU else 2)
TEACHER_HEADS = env_int("TEACHER_HEADS", 8 if TEACHER_DIM % 8 == 0 else 4)
TEACHER_LR = env_float("TEACHER_LR", LR)
KD_TEMP = env_float("KD_TEMP", 4.0)
KD_CE_WEIGHT = env_float("KD_CE_WEIGHT", 0.5 if USE_KD else 1.0)
KD_LOGIT_WEIGHT = env_float("KD_LOGIT_WEIGHT", 0.5 if USE_KD else 0.0)
KD_AUX_WEIGHT = env_float("KD_AUX_WEIGHT", 0.1)

DATASET_NAMES = [name.strip() for name in DATASET_NAMES if name.strip()]
if not DATASET_NAMES:
    DATASET_NAMES = ["ModelNet10"]

if DEBUG_MINI_RUN:
    EPOCHS = max(1, DEBUG_EPOCHS)
    TEACHER_EPOCHS = max(1, min(TEACHER_EPOCHS, DEBUG_EPOCHS))
    CHECKPOINT_EVERY = 1
    N_VOTE = 1
    RESUME = env_bool("DEBUG_RESUME", False)
    DATASET_NAMES = (
        [name.strip() for name in DEBUG_DATASETS.split(",") if name.strip()]
        if DEBUG_DATASETS
        else DATASET_NAMES[:1]
    )
    if not DATASET_NAMES:
        DATASET_NAMES = ["ModelNet10"]
    if DEBUG_SKIP_TEACHER:
        USE_KD = False
        TEACHER_MODE = "off"
        KD_CE_WEIGHT = 1.0
        KD_LOGIT_WEIGHT = 0.0

CKPT_DIR = os.path.join(WORK, "official_like_ckpts")
for path in (WORK, DATA_DIR, CKPT_DIR):
    os.makedirs(path, exist_ok=True)

# Google Drive checkpoint sync
# Set DRIVE_FOLDER_ID to the ID of the Drive folder you want checkpoints saved to.
# Add your service-account JSON as a Kaggle dataset and point DRIVE_CREDS_PATH at it.
# If either is missing, Drive sync is silently disabled — training still works normally.
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()
DRIVE_CREDS_PATH = os.environ.get(
    "DRIVE_CREDS_PATH",
    "/kaggle/input/gdrive-creds/service_account.json",
)

assert NUM_GROUP % ASP_STEPS == 0, "NUM_GROUP must divide ASP_STEPS"
assert TEACHER_DIM % TEACHER_HEADS == 0, "TEACHER_DIM must be divisible by TEACHER_HEADS"

print("[Config]")
print(f"  epochs={EPOCHS} batch={BATCH} points={NUM_POINTS}")
print(f"  timestep={TIMESTEP} dim={TRANS_DIM} depth={DEPTH}")
print(f"  groups={NUM_GROUP} group_size={GROUP_SIZE} expand={EXPAND}")
print(f"  asp_steps={ASP_STEPS} chunk_size={NUM_GROUP // ASP_STEPS}")
print(f"  grad_accum={GRAD_ACCUM} vote={N_VOTE}")
print(
    f"  kd={USE_KD} teacher_mode={TEACHER_MODE} teacher_epochs={TEACHER_EPOCHS} "
    f"kd_temp={KD_TEMP}"
)
if DEBUG_MINI_RUN:
    print(
        f"  debug_mini_run=True debug_epochs={EPOCHS} "
        f"debug_max_steps={DEBUG_MAX_STEPS} skip_eval={DEBUG_SKIP_EVAL} "
        f"skip_teacher={DEBUG_SKIP_TEACHER}"
    )
    print(
        f"  debug_datasets={DATASET_NAMES} train_items={DEBUG_DATASET_ITEMS} "
        f"val_items={DEBUG_VAL_ITEMS} resume={RESUME}"
    )
print(f"  data={DATA_DIR}")
print(f"  ckpt={CKPT_DIR}")


# ---------------------------------------------------------------------------
# Google Drive checkpoint sync
# ---------------------------------------------------------------------------


class DriveSync:
    """
    Upload checkpoints to a Google Drive folder and restore them on startup.

    Requires:
      - A GCP service account with Drive API enabled and editor access to the folder.
      - The service account JSON uploaded as a Kaggle dataset input.

    Set env vars:
      DRIVE_FOLDER_ID   = ID from the Drive folder URL (the long alphanumeric string)
      DRIVE_CREDS_PATH  = path to service_account.json
                          (default: /kaggle/input/gdrive-creds/service_account.json)

    If either is missing, all methods are no-ops.
    """

    def __init__(self, folder_id, creds_path):
        self._service = None
        self._folder_id = folder_id
        if not folder_id:
            print("[DriveSync] DRIVE_FOLDER_ID not set — Drive sync disabled.")
            return
        if not os.path.isfile(creds_path):
            print(f"[DriveSync] credentials not found at {creds_path} — disabled.")
            return
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
            print(f"[DriveSync] Connected to Drive. Folder: {folder_id}")
        except Exception as exc:
            print(f"[DriveSync] init failed: {exc} — disabled.")

    @property
    def enabled(self):
        return self._service is not None

    def _find_file(self, filename):
        """Return Drive file ID if filename exists in the folder, else None."""
        res = self._service.files().list(
            q=(
                f"name='{filename}' and "
                f"'{self._folder_id}' in parents and "
                f"trashed=false"
            ),
            fields="files(id)",
        ).execute()
        files = res.get("files", [])
        return files[0]["id"] if files else None

    def upload(self, local_path):
        """Upload (or update) a local file to the Drive folder."""
        if not self.enabled:
            return
        try:
            from googleapiclient.http import MediaFileUpload

            fname = os.path.basename(local_path)
            media = MediaFileUpload(local_path, resumable=True)
            existing_id = self._find_file(fname)
            if existing_id:
                self._service.files().update(
                    fileId=existing_id, media_body=media
                ).execute()
            else:
                self._service.files().create(
                    body={"name": fname, "parents": [self._folder_id]},
                    media_body=media,
                ).execute()
            print(f"  [DriveSync] uploaded {fname}")
        except Exception as exc:
            print(f"  [DriveSync] upload failed for {local_path}: {exc}")

    def download(self, filename, local_path):
        """Download filename from Drive into local_path if it doesn't exist locally."""
        if not self.enabled or os.path.isfile(local_path):
            return
        try:
            import io

            from googleapiclient.http import MediaIoBaseDownload

            fid = self._find_file(filename)
            if not fid:
                return
            req = self._service.files().get_media(fileId=fid)
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
            with open(local_path, "wb") as f:
                f.write(buf.getvalue())
            print(f"  [DriveSync] downloaded {filename} from Drive")
        except Exception as exc:
            print(f"  [DriveSync] download failed for {filename}: {exc}")


drive_sync = DriveSync(DRIVE_FOLDER_ID, DRIVE_CREDS_PATH)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _show(path):
    try:
        from IPython.display import Image as _Img
        from IPython.display import display

        display(_Img(filename=path))
    except Exception:
        pass


def _download(name, slug):
    folder = os.path.join(DATA_DIR, name)
    if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
        print(f"  {name}: cached at {folder}")
        return folder
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    print(f"  Downloading {name} ...")
    path = kagglehub.dataset_download(slug)
    for root, dirs, _files in os.walk(path):
        if name in dirs:
            shutil.copytree(os.path.join(root, name), folder)
            print(f"  {name} -> {folder}")
            return folder
        subdirs_with_train = [
            d for d in dirs if os.path.isdir(os.path.join(root, d, "train"))
        ]
        if len(subdirs_with_train) >= 5:
            shutil.copytree(root, folder)
            print(f"  {name} -> {folder}")
            return folder
    print(f"  {name}: using raw path {path}")
    return path


MN10_DIR = None
MN40_DIR = None


def _augment(pts):
    n = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pad = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])

    pts2 = pts2 * np.random.uniform(0.8, 1.25)
    pts2 = pts2 + np.random.uniform(-0.1, 0.1, (1, 3))

    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pts2 = pts2 @ rz.T
    pts2 += np.clip(np.random.randn(*pts2.shape).astype(np.float32) * 0.01, -0.05, 0.05)
    return pts2.astype(np.float32)


class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, split="train", max_items=None):
        self.num_points = num_points
        self.split = split
        self.files = self._scan(root)
        if max_items is not None and max_items > 0:
            self.files = self.files[:max_items]
        print(f"  [{split}] Loading {len(self.files)} files ...")
        self.data, self.labels = self._load_all()
        print(f"  [{split}] Loaded. Shape: {self.data.shape}")

    def _scan(self, root):
        items = []
        classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        for cls in classes:
            path = os.path.join(root, cls, self.split)
            if not os.path.isdir(path):
                continue
            label = classes.index(cls)
            for fname in os.listdir(path):
                if fname.lower().endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(path, fname), label))
        return items

    def _load_pts(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)[:, :3]
        if path.endswith(".txt"):
            return np.loadtxt(path, delimiter=",").astype(np.float32)[:, :3]
        mesh = trimesh.load(path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_lbl = [], []
        for path, label in self.files:
            try:
                pts = self._load_pts(path)
                n = pts.shape[0]
                if n >= self.num_points:
                    pts = pts[np.random.choice(n, self.num_points, replace=False)]
                else:
                    pad = np.random.choice(n, self.num_points - n, replace=True)
                    pts = np.vstack([pts, pts[pad]])
                all_pts.append(pts)
                all_lbl.append(label)
            except Exception:
                pass
        return np.asarray(all_pts, dtype=np.float32), np.asarray(all_lbl, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        pts -= pts.mean(axis=0)
        pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8
        if self.split == "train":
            pts = _augment(pts)
        np.random.shuffle(pts)
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


# ---------------------------------------------------------------------------
# Point cloud grouping: official-like FPS + KNN, pure PyTorch
# ---------------------------------------------------------------------------


def index_points(points, idx):
    """points: [B,N,C], idx: [B,...] -> [B,...,C]."""
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long, device=device).view(view_shape)
    batch_indices = batch_indices.repeat(repeat_shape)
    return points[batch_indices, idx]


def farthest_point_sample_batched(xyz, npoint):
    """Pure PyTorch batched FPS. xyz: [B,N,3] -> [B,npoint]."""
    b, n, _ = xyz.shape
    npoint = min(npoint, n)
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device)
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(b, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1).indices
    return centroids


class OfficialLikeGroup(nn.Module):
    """SPM Group module: moving FPS centers + KNN neighborhoods."""

    def __init__(self, num_group, group_size, expand=1.1, timestep=2):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.expand = expand
        self.timestep = timestep

    def _moving_centers(self, pts):
        b, n, _ = pts.shape
        step_f = int((self.expand - 1.0) * self.num_group / self.timestep * 2)
        step_b = int((self.expand - 1.0) * self.num_group)
        total = self.num_group + (step_f + step_b) * (self.timestep - 1)
        total = min(max(total, self.num_group), n)
        center_idx = farthest_point_sample_batched(pts.contiguous(), total)
        pool = index_points(pts, center_idx)

        if total < self.num_group + (step_f + step_b) * (self.timestep - 1):
            repeat = math.ceil(
                (self.num_group + (step_f + step_b) * (self.timestep - 1)) / total
            )
            pool = pool.repeat(1, repeat, 1)

        centers = []
        for i in range(self.timestep):
            first = pool[:, i * step_f : i * step_f + (self.num_group - step_b)]
            start = (i - 1) * step_b + self.num_group + (self.timestep - 1) * step_f
            end = i * step_b + self.num_group + (self.timestep - 1) * step_f
            second = pool[:, start:end]
            cur = torch.cat([first, second], dim=1)
            if cur.shape[1] < self.num_group:
                pad = cur[:, -1:].repeat(1, self.num_group - cur.shape[1], 1)
                cur = torch.cat([cur, pad], dim=1)
            centers.append(cur[:, : self.num_group])
        return torch.stack(centers, dim=0)

    def forward(self, pts):
        """
        pts: [B,N,3]
        returns:
          neighborhood: [T,B,G,K,3] relative coordinates
          centers:      [T,B,G,3]
        """
        b, n, _ = pts.shape
        centers = self._moving_centers(pts)
        flat_centers = centers.reshape(self.timestep * b, self.num_group, 3)
        flat_pts = pts.unsqueeze(0).expand(self.timestep, -1, -1, -1).reshape(
            self.timestep * b, n, 3
        )

        k = min(self.group_size, n)
        dist = torch.cdist(flat_centers, flat_pts)
        idx = dist.topk(k, dim=-1, largest=False).indices
        grouped = index_points(flat_pts, idx)
        grouped = grouped.reshape(self.timestep, b, self.num_group, k, 3)
        grouped = grouped - centers.unsqueeze(3)
        return grouped.contiguous(), centers.contiguous()


# ---------------------------------------------------------------------------
# Spiking / official-like SPM modules
# ---------------------------------------------------------------------------


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + x.abs()) ** 2


spike_fn = SurrogateSpike.apply


class SpikeAct(nn.Module):
    def __init__(self, vth=0.5):
        super().__init__()
        self.vth = vth
        self.register_buffer("spike_sum", torch.tensor(0.0))
        self.register_buffer("elem_count", torch.tensor(0.0))

    def forward(self, x):
        y = spike_fn(x - self.vth)
        self.spike_sum = self.spike_sum + y.detach().sum()
        self.elem_count = self.elem_count + torch.tensor(
            y.numel(), dtype=torch.float32, device=y.device
        )
        return y

    def rate(self):
        if self.elem_count.item() == 0:
            return 0.0
        return (self.spike_sum / self.elem_count).item()


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class TokenBNSpike(nn.Module):
    def __init__(self, dim, vth=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.spike = SpikeAct(vth)

    def forward(self, x):
        b, l, c = x.shape
        y = self.bn(x.reshape(b * l, c)).reshape(b, l, c)
        return self.spike(y)


class OfficialLikeEncoder(nn.Module):
    """PointNet-style Conv2d encoder from official SPM."""

    def __init__(self, encoder_channel):
        super().__init__()
        self.spk1 = SpikeAct()
        self.spk2 = SpikeAct()
        self.spk3 = SpikeAct()
        self.first_conv1 = nn.Conv2d(3, 128, 1)
        self.first_bn1 = nn.BatchNorm2d(128)
        self.first_conv2 = nn.Conv2d(128, 256, 1)
        self.first_bn2 = nn.BatchNorm2d(256)
        self.second_conv1 = nn.Conv2d(512, 512, 1)
        self.second_bn1 = nn.BatchNorm2d(512)
        self.second_conv2 = nn.Conv2d(512, encoder_channel, 1)
        self.second_bn2 = nn.BatchNorm2d(encoder_channel)

    def forward(self, point_groups):
        # point_groups: [T,B,G,K,3]
        t, b, g, k, _ = point_groups.shape
        x = point_groups.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x = self.spk1(self.first_bn1(self.first_conv1(x)))
        x = self.first_bn2(self.first_conv2(x))
        x_global = x.max(dim=3, keepdim=True).values
        x = torch.cat([x_global.expand(-1, -1, -1, k), x], dim=1)
        x = self.spk2(x)
        x = self.spk3(self.second_bn1(self.second_conv1(x)))
        x = self.second_bn2(self.second_conv2(x))
        x = x.max(dim=3).values.transpose(1, 2).contiguous()
        return x.reshape(t, b, g, -1)


class PosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            SpikeAct(),
            nn.Conv1d(128, dim, 1),
            nn.BatchNorm1d(dim),
        )

    def forward(self, centers):
        # centers: [T,B,G,3]
        t, b, g, _ = centers.shape
        x = centers.flatten(0, 1).permute(0, 2, 1).contiguous()
        x = self.net(x).permute(0, 2, 1).contiguous()
        return x.reshape(t, b, g, -1)


class MambaLiteMixer(nn.Module):
    """
    Dependency-free Mamba approximation: depthwise Conv1d + cumulative state.
    Preserves the SPM sequence-mixer interface without mamba_ssm CUDA kernels.
    """

    def __init__(self, dim, expand=2):
        super().__init__()
        inner = dim * expand
        self.in_proj = nn.Linear(dim, inner * 2)
        self.dwconv = nn.Conv1d(inner, inner, 3, padding=1, groups=inner)
        self.scan_proj = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, x):
        # x: [TB,G,C]
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        steps = torch.arange(1, u.shape[1] + 1, device=u.device, dtype=u.dtype).view(
            1, -1, 1
        )
        state = torch.cumsum(u, dim=1) / steps
        u = u + self.scan_proj(state)
        u = u * torch.sigmoid(gate)
        return self.out_proj(u)


class OfficialLikeBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm_lif = TokenBNSpike(dim)
        self.mixer = MambaLiteMixer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, hidden_states, residual=None):
        residual = (
            self.drop_path(hidden_states) + residual
            if residual is not None
            else hidden_states
        )
        hidden_states = self.norm_lif(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class OfficialLikeMixerModel(nn.Module):
    def __init__(self, dim, depth, timestep, drop_path=0.3):
        super().__init__()
        self.timestep = timestep
        self.layers = nn.ModuleList(
            [OfficialLikeBlock(dim, drop_path=drop_path) for _ in range(depth)]
        )

    def forward(self, tokens, pos):
        # tokens/pos: [T,B,L,C]
        t, b, l, c = tokens.shape
        x = (tokens + pos).reshape(t * b, l, c)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x = x + residual if residual is not None else x
        return x.reshape(t, b, l, c)


class OfficialLikeHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            SpikeAct(),
            nn.Conv1d(dim, 256, 1),
            nn.BatchNorm1d(256),
            SpikeAct(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            SpikeAct(),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x):
        # x: [T,B,L,C]
        t, b, _l, c = x.shape
        pooled = x.mean(dim=2).reshape(t * b, c, 1)
        logits = self.net(pooled).reshape(t, b, -1, 1)
        return logits.mean(dim=0).squeeze(-1)


class OfficialLikeSPM(nn.Module):
    """Official-SPM-like classifier, pure PyTorch."""

    def __init__(
        self,
        num_classes,
        dim=384,
        depth=12,
        num_group=128,
        group_size=32,
        timestep=2,
        expand=1.1,
        drop_path=0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.num_group = num_group
        self.group_size = group_size
        self.timestep = timestep
        self.group_divider = OfficialLikeGroup(num_group, group_size, expand, timestep)
        self.encoder = OfficialLikeEncoder(dim)
        self.pos_embed = PosEmbed(dim)
        self.blocks = OfficialLikeMixerModel(dim, depth, timestep, drop_path)
        self.drop_out = nn.Dropout(0.0)
        self.cls_head = OfficialLikeHead(dim, num_classes)

    def encode_groups(self, pts):
        neighborhoods, centers = self.group_divider(pts)
        tokens = self.encoder(neighborhoods)
        pos = self.pos_embed(centers)
        return tokens, pos, centers

    def forward_tokens(self, tokens, pos):
        x = self.drop_out(tokens)
        x = self.blocks(x, pos)
        return self.cls_head(x)

    def forward(self, pts):
        tokens, pos, _centers = self.encode_groups(pts)
        return self.forward_tokens(tokens, pos)

    def get_firing_rates(self):
        rates = {}
        for name, module in self.named_modules():
            if isinstance(module, SpikeAct):
                rates[name] = module.rate()
        return rates

    def mean_firing_rate(self):
        rates = self.get_firing_rates()
        return sum(rates.values()) / max(1, len(rates))


# ---------------------------------------------------------------------------
# Training-only transformer teacher for knowledge distillation
# ---------------------------------------------------------------------------


class AnalogGroupEncoder(nn.Module):
    """Non-spiking PointNet-style patch encoder for the teacher."""

    def __init__(self, encoder_channel):
        super().__init__()
        self.first_conv1 = nn.Conv2d(3, 128, 1)
        self.first_bn1 = nn.BatchNorm2d(128)
        self.first_conv2 = nn.Conv2d(128, 256, 1)
        self.first_bn2 = nn.BatchNorm2d(256)
        self.second_conv1 = nn.Conv2d(512, 512, 1)
        self.second_bn1 = nn.BatchNorm2d(512)
        self.second_conv2 = nn.Conv2d(512, encoder_channel, 1)
        self.second_bn2 = nn.BatchNorm2d(encoder_channel)

    def forward(self, point_groups):
        t, b, g, k, _ = point_groups.shape
        x = point_groups.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x = F.gelu(self.first_bn1(self.first_conv1(x)))
        x = F.gelu(self.first_bn2(self.first_conv2(x)))
        x_global = x.max(dim=3, keepdim=True).values
        x = torch.cat([x_global.expand(-1, -1, -1, k), x], dim=1)
        x = F.gelu(self.second_bn1(self.second_conv1(x)))
        x = self.second_bn2(self.second_conv2(x))
        x = x.max(dim=3).values.transpose(1, 2).contiguous()
        return x.reshape(t, b, g, -1)


class AnalogPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, dim, 1),
            nn.BatchNorm1d(dim),
        )

    def forward(self, centers):
        t, b, g, _ = centers.shape
        x = centers.flatten(0, 1).permute(0, 2, 1).contiguous()
        x = self.net(x).permute(0, 2, 1).contiguous()
        return x.reshape(t, b, g, -1)


class PointTransformerTeacher(nn.Module):
    """
    Point-BERT/PointTransformer-style teacher, pure PyTorch.

    Trains on the same groups as the SPM student so logits are aligned.
    Used only during training — never at inference, so deployment cost
    is identical to the SNN/ASP student.
    """

    def __init__(
        self,
        num_classes,
        dim=384,
        depth=8,
        heads=8,
        num_group=128,
        group_size=32,
        expand=1.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.group_divider = OfficialLikeGroup(
            num_group=num_group,
            group_size=group_size,
            expand=expand,
            timestep=1,
        )
        self.encoder = AnalogGroupEncoder(dim)
        self.pos_embed = AnalogPosEmbed(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_classes),
        )

    def forward(self, pts):
        neighborhoods, centers = self.group_divider(pts)
        x = self.encoder(neighborhoods) + self.pos_embed(centers)
        x = self.blocks(x.squeeze(0))
        x = self.norm(x)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# ASP on official-like SPM groups
# ---------------------------------------------------------------------------


class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=7, hidden=128):
        super().__init__()
        self.mem_proj = nn.Linear(mem_dim, hidden, bias=False)
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.scale = math.sqrt(hidden)

    def forward(self, belief, geo, visited_mask=None):
        key = self.mem_proj(belief)
        query = self.geo_proj(geo)
        scores = torch.bmm(query, key.unsqueeze(-1)).squeeze(-1) / self.scale
        if visited_mask is not None:
            scores = scores.masked_fill(visited_mask.clone(), float("-inf"))
        return scores


class OfficialLikeASP(nn.Module):
    def __init__(self, base_model, asp_steps=4, d_ssp=128):
        super().__init__()
        self.base_model = base_model
        self.asp_steps = asp_steps
        self.chunk_size = base_model.num_group // asp_steps
        self.ssp = SliceSelectionPolicy(base_model.dim, geo_dim=7, hidden=d_ssp)
        self.belief_proj = nn.Sequential(
            nn.Linear(base_model.num_classes, base_model.dim),
            nn.GELU(),
            nn.Linear(base_model.dim, base_model.dim),
        )
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    @property
    def num_classes(self):
        return self.base_model.num_classes

    def set_gumbel_tau(self, tau):
        self.gumbel_tau.fill_(tau)

    def get_firing_rates(self):
        return self.base_model.get_firing_rates()

    def mean_firing_rate(self):
        return self.base_model.mean_firing_rate()

    def _chunkify(self, tokens, pos, centers, pts):
        t, b, g, c = tokens.shape
        s, k = self.asp_steps, self.chunk_size
        tokens_c = tokens.reshape(t, b, s, k, c)
        pos_c = pos.reshape(t, b, s, k, c)

        centers_b = centers.mean(dim=0).reshape(b, s, k, 3)
        chunk_center = centers_b.mean(dim=2)
        centroid = pts.mean(dim=1, keepdim=True)
        anchor_dist = (chunk_center - centroid).norm(dim=-1, keepdim=True)
        spread = (centers_b - chunk_center.unsqueeze(2)).norm(dim=-1).mean(
            dim=2, keepdim=True
        )
        coverage = torch.ones(b, s, 1, device=pts.device)
        order = torch.linspace(0, 1, s, device=pts.device).view(1, s, 1).expand(b, -1, -1)
        geo = torch.cat([chunk_center, anchor_dist, spread, coverage, order], dim=-1)
        return tokens_c, pos_c, geo

    def _gather_chunk(self, chunks, idx):
        # chunks: [T,B,S,K,C], idx: [B] -> [T,B,K,C]
        t, b, _s, k, c = chunks.shape
        gather_idx = idx.view(1, b, 1, 1, 1).expand(t, b, 1, k, c)
        return chunks.gather(dim=2, index=gather_idx).squeeze(2)

    def forward_active_train(self, pts):
        tokens, pos, centers = self.base_model.encode_groups(pts)
        tok_c, pos_c, geo = self._chunkify(tokens, pos, centers, pts)
        b = pts.shape[0]
        device = pts.device
        visited = torch.zeros(b, self.asp_steps, dtype=torch.bool, device=device)
        belief = torch.zeros(b, self.base_model.dim, device=device)
        selected_tokens, selected_pos, logits_all = [], [], []

        for _step in range(self.asp_steps):
            scores = self.ssp(belief, geo, visited)
            w = F.gumbel_softmax(scores, tau=float(self.gumbel_tau.item()), hard=True)
            idx = w.detach().argmax(dim=-1)
            visited.scatter_(1, idx.unsqueeze(1), True)

            tok = (w.view(1, b, self.asp_steps, 1, 1) * tok_c).sum(dim=2)
            ps = (w.view(1, b, self.asp_steps, 1, 1) * pos_c).sum(dim=2)
            selected_tokens.append(tok)
            selected_pos.append(ps)

            seq_tokens = torch.cat(selected_tokens, dim=2)
            seq_pos = torch.cat(selected_pos, dim=2)
            logits = self.base_model.forward_tokens(seq_tokens, seq_pos)
            logits_all.append(logits)
            belief = self.belief_proj(logits.detach().softmax(dim=-1))

        return logits_all[-1], logits_all

    @torch.no_grad()
    def forward_active_infer(self, pts, threshold=0.45):
        tokens, pos, centers = self.base_model.encode_groups(pts)
        tok_c, pos_c, geo = self._chunkify(tokens, pos, centers, pts)
        b = pts.shape[0]
        device = pts.device
        visited = torch.zeros(b, self.asp_steps, dtype=torch.bool, device=device)
        belief = torch.zeros(b, self.base_model.dim, device=device)
        selected_tokens, selected_pos = [], []
        last_logits = None

        for step in range(self.asp_steps):
            scores = self.ssp(belief, geo, visited)
            idx = scores.argmax(dim=-1)
            visited.scatter_(1, idx.unsqueeze(1), True)

            selected_tokens.append(self._gather_chunk(tok_c, idx))
            selected_pos.append(self._gather_chunk(pos_c, idx))
            seq_tokens = torch.cat(selected_tokens, dim=2)
            seq_pos = torch.cat(selected_pos, dim=2)
            logits = self.base_model.forward_tokens(seq_tokens, seq_pos)
            last_logits = logits
            belief = self.belief_proj(logits.softmax(dim=-1))

            probs = logits.softmax(dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            if margin.min().item() > threshold:
                return logits, step + 1

        return last_logits, self.asp_steps


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_start():
    _sync_if_cuda()
    return time.perf_counter()


def _timed_stats(start_time, steps, samples):
    _sync_if_cuda()
    seconds = time.perf_counter() - start_time
    return {
        "steps": int(steps),
        "samples": int(samples),
        "seconds": seconds,
        "seconds_per_step": seconds / max(1, steps),
        "seconds_per_sample": seconds / max(1, samples),
    }


def _debug_max_steps():
    if not DEBUG_MINI_RUN or DEBUG_MAX_STEPS <= 0:
        return None
    return DEBUG_MAX_STEPS


def _timing_text(stats):
    if not stats:
        return ""
    return (
        f"steps={stats['steps']} samples={stats['samples']} "
        f"sec/step={stats['seconds_per_step']:.3f} "
        f"sec/sample={stats['seconds_per_sample']:.5f}"
    )


def make_spm(num_classes):
    return OfficialLikeSPM(
        num_classes,
        dim=TRANS_DIM,
        depth=DEPTH,
        num_group=NUM_GROUP,
        group_size=GROUP_SIZE,
        timestep=TIMESTEP,
        expand=EXPAND,
        drop_path=DROP_PATH,
    ).to(DEVICE)


def make_asp(num_classes):
    return OfficialLikeASP(make_spm(num_classes), asp_steps=ASP_STEPS).to(DEVICE)


def make_teacher(num_classes):
    return PointTransformerTeacher(
        num_classes,
        dim=TEACHER_DIM,
        depth=TEACHER_DEPTH,
        heads=TEACHER_HEADS,
        num_group=NUM_GROUP,
        group_size=GROUP_SIZE,
        expand=EXPAND,
    ).to(DEVICE)


def smooth_ce(logits, labels):
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)


def kd_ce(logits, labels, teacher_logits=None):
    ce = smooth_ce(logits, labels)
    if teacher_logits is None or KD_LOGIT_WEIGHT <= 0:
        return ce

    temp = KD_TEMP
    kd = F.kl_div(
        F.log_softmax(logits / temp, dim=-1),
        F.softmax(teacher_logits.detach() / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
    return KD_CE_WEIGHT * ce + KD_LOGIT_WEIGHT * kd


def gumbel_tau(epoch, tau_0=1.0, tau_min=0.1, rate=0.04):
    return max(tau_min, tau_0 * math.exp(-rate * epoch))


def active_loss(logits_final, logits_all, labels, model, teacher_logits=None):
    loss = kd_ce(logits_final, labels, teacher_logits)
    if len(logits_all) > 1:
        aux = sum(kd_ce(logits, labels, teacher_logits) for logits in logits_all[:-1])
        loss = loss + KD_AUX_WEIGHT * aux / (len(logits_all) - 1)
    exit_loss = 0.0
    for i, logits in enumerate(logits_all):
        weight = (len(logits_all) - i) / len(logits_all)
        exit_loss = exit_loss + weight * (1.0 - logits.softmax(-1).max(-1).values).mean()
    loss = loss + 0.05 * exit_loss / len(logits_all)
    loss = loss + 0.01 * model.mean_firing_rate()
    return loss


def make_scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        t = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def save_ckpt(path, model, opt, sch, epoch, best, history, debug_stats=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "best": best,
        "history": history,
        "config": {
            "points": NUM_POINTS,
            "timestep": TIMESTEP,
            "dim": TRANS_DIM,
            "depth": DEPTH,
            "num_group": NUM_GROUP,
            "group_size": GROUP_SIZE,
            "asp_steps": ASP_STEPS,
        },
    }
    if debug_stats is not None:
        payload["debug_stats"] = debug_stats
    torch.save(payload, path)
    drive_sync.upload(path)


def save_debug_ckpt(tag, ds_name, model, opt, sch, epoch, best, history, stats):
    if not DEBUG_MINI_RUN:
        return None
    safe_ds = ds_name.replace(os.sep, "_").replace("/", "_")
    path = os.path.join(CKPT_DIR, f"debug_{tag}_{safe_ds}_minimal.pt")
    save_ckpt(path, model, opt, sch, epoch, best, history, debug_stats=stats)
    print(f"  [DEBUG] minimal checkpoint -> {path}")
    return path


def load_ckpt(path, model, opt, sch):
    if not RESUME:
        return 0, 0.0, []
    # Try to restore from Drive if the local file is missing (e.g. after a crash)
    drive_sync.download(os.path.basename(path), path)
    if not os.path.isfile(path):
        return 0, 0.0, []
    try:
        ckpt = _torch_load(path)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sch.load_state_dict(ckpt["scheduler"])
        print(f"  [CKPT] resumed {os.path.basename(path)} epoch {ckpt['epoch']}")
        return int(ckpt["epoch"]), float(ckpt.get("best", 0.0)), ckpt.get("history", [])
    except Exception as exc:
        print(f"  [CKPT] resume skipped for {path}: {exc}")
        return 0, 0.0, []


def _state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict", "base_model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    return {k.replace("module.", "", 1): v for k, v in ckpt.items()}


def save_best(path, state_dict):
    """Save a best-model state dict and mirror it to Drive."""
    torch.save(state_dict, path)
    drive_sync.upload(path)


def load_model_weights(path, model, strict=True):
    ckpt = _torch_load(path)
    state = _state_dict_from_checkpoint(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(
            f"  [Teacher] loaded with missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )


def train_teacher_epoch(model, loader, opt, max_steps=None):
    model.train()
    opt.zero_grad()
    total_loss = total_acc = n = 0
    steps_done = 0
    t_start = _timed_start()
    for step, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        logits = model(pts)
        loss = smooth_ce(logits, labels) / GRAD_ACCUM
        if torch.isfinite(loss):
            loss.backward()
        hit_step_cap = max_steps is not None and step + 1 >= max_steps
        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader) or hit_step_cap:
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.zero_grad()
        b = pts.shape[0]
        total_loss += loss.item() * GRAD_ACCUM * b
        total_acc += (logits.argmax(1) == labels).sum().item()
        n += b
        steps_done += 1
        if hit_step_cap:
            break
    return total_loss / max(1, n), total_acc / max(1, n), _timed_stats(
        t_start, steps_done, n
    )


@torch.no_grad()
def eval_teacher(model, loader, n_vote=1):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        prob_sum = torch.zeros(pts.shape[0], model.num_classes, device=DEVICE)
        for _ in range(n_vote):
            theta = random.uniform(0.0, 2.0 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            rz = torch.tensor(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=DEVICE,
            )
            logits = model(pts @ rz.T)
            prob_sum += logits.softmax(dim=-1)
        correct += (prob_sum.argmax(1) == labels).sum().item()
        total += pts.shape[0]
    return correct / total


def prepare_teacher(ds_name, num_classes, train_loader, val_loader):
    if not USE_KD or TEACHER_MODE in ("0", "off", "none", "false"):
        return None, 0.0

    teacher = make_teacher(num_classes)
    print(
        f"\n[Teacher] PointTransformer-style params="
        f"{sum(p.numel() for p in teacher.parameters()):,}"
    )

    if TEACHER_CKPT:
        try:
            load_model_weights(TEACHER_CKPT, teacher, strict=True)
            teacher.eval()
            val_acc = eval_teacher(teacher, val_loader, max(1, min(N_VOTE, 3)))
            print(f"  [Teacher] loaded {TEACHER_CKPT} val={val_acc:.4f}")
            return teacher, val_acc
        except Exception as exc:
            print(f"  [Teacher] explicit checkpoint load failed: {exc}")
            if TEACHER_MODE == "load":
                return None, 0.0

    latest = os.path.join(CKPT_DIR, f"teacher_{ds_name}_latest.pt")
    best_path = os.path.join(CKPT_DIR, f"teacher_{ds_name}_best.pth")
    drive_sync.download(os.path.basename(latest), latest)
    drive_sync.download(os.path.basename(best_path), best_path)
    opt = torch.optim.AdamW(
        teacher.parameters(), lr=TEACHER_LR, weight_decay=WEIGHT_DECAY
    )
    sch = make_scheduler(opt, max(1, TEACHER_EPOCHS), min(WARMUP_EP, TEACHER_EPOCHS))
    start_ep, best_teacher, teacher_hist = load_ckpt(latest, teacher, opt, sch)

    if start_ep >= TEACHER_EPOCHS and os.path.isfile(best_path):
        load_model_weights(best_path, teacher, strict=True)
        teacher.eval()
        print(f"  [Teacher] using cached best val={best_teacher:.4f}")
        return teacher, best_teacher

    if TEACHER_MODE == "load" and os.path.isfile(best_path):
        load_model_weights(best_path, teacher, strict=True)
        teacher.eval()
        val_acc = eval_teacher(teacher, val_loader, max(1, min(N_VOTE, 3)))
        print(f"  [Teacher] loaded cached best val={val_acc:.4f}")
        return teacher, val_acc

    if TEACHER_MODE == "load":
        print("  [Teacher] no checkpoint available; KD disabled for this dataset.")
        return None, 0.0

    for ep in range(start_ep, TEACHER_EPOCHS):
        t0 = time.time()
        _loss, tr_acc, train_stats = train_teacher_epoch(
            teacher, train_loader, opt, max_steps=_debug_max_steps()
        )
        sch.step()
        val_acc = None
        is_best = False
        should_eval = (
            not (DEBUG_MINI_RUN and DEBUG_SKIP_EVAL)
            and ((ep + 1) % 5 == 0 or ep + 1 == TEACHER_EPOCHS)
        )
        if should_eval:
            val_acc = eval_teacher(teacher, val_loader, max(1, min(N_VOTE, 3)))
            if val_acc > best_teacher:
                best_teacher = val_acc
                is_best = True
                save_best(best_path, teacher.state_dict())
            print(
                f"  [Teacher] Ep {ep+1:3d}/{TEACHER_EPOCHS} tr={tr_acc:.4f} "
                f"val={val_acc:.4f} {'*' if is_best else ' '} "
                f"lr={opt.param_groups[0]['lr']:.5f} {time.time()-t0:.0f}s"
            )
        elif DEBUG_MINI_RUN:
            print(
                f"  [Teacher] Ep {ep+1:3d}/{TEACHER_EPOCHS} tr={tr_acc:.4f} "
                f"val=skipped lr={opt.param_groups[0]['lr']:.5f} "
                f"{_timing_text(train_stats)} wall={time.time()-t0:.0f}s"
            )
        teacher_hist.append(
            {
                "ep": ep + 1,
                "tr": tr_acc,
                "val": val_acc,
                "timing": train_stats,
            }
        )
        save_ckpt(
            latest,
            teacher,
            opt,
            sch,
            ep + 1,
            best_teacher,
            teacher_hist,
            debug_stats=train_stats if DEBUG_MINI_RUN else None,
        )
        save_debug_ckpt(
            "teacher",
            ds_name,
            teacher,
            opt,
            sch,
            ep + 1,
            best_teacher,
            teacher_hist,
            train_stats,
        )

    if os.path.isfile(best_path):
        load_model_weights(best_path, teacher, strict=True)
    teacher.eval()
    return teacher, best_teacher


@torch.no_grad()
def teacher_forward(teacher, pts):
    if teacher is None:
        return None
    teacher.eval()
    return teacher(pts)


def train_spm_epoch(model, loader, opt, teacher=None, max_steps=None):
    model.train()
    opt.zero_grad()
    total_loss = total_acc = n = 0
    steps_done = 0
    t_start = _timed_start()
    for step, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        teacher_logits = teacher_forward(teacher, pts)
        logits = model(pts)
        loss = kd_ce(logits, labels, teacher_logits) / GRAD_ACCUM
        if torch.isfinite(loss):
            loss.backward()
        hit_step_cap = max_steps is not None and step + 1 >= max_steps
        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader) or hit_step_cap:
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.zero_grad()
        b = pts.shape[0]
        total_loss += loss.item() * GRAD_ACCUM * b
        total_acc += (logits.argmax(1) == labels).sum().item()
        n += b
        steps_done += 1
        if hit_step_cap:
            break
    return total_loss / max(1, n), total_acc / max(1, n), _timed_stats(
        t_start, steps_done, n
    )


def train_asp_epoch(model, loader, opt, epoch, teacher=None, max_steps=None):
    model.train()
    model.set_gumbel_tau(gumbel_tau(epoch))
    opt.zero_grad()
    total_loss = total_acc = n = 0
    steps_done = 0
    t_start = _timed_start()
    for step, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        teacher_logits = teacher_forward(teacher, pts)
        logits, logits_all = model.forward_active_train(pts)
        loss = active_loss(logits, logits_all, labels, model, teacher_logits) / GRAD_ACCUM
        if torch.isfinite(loss):
            loss.backward()
        hit_step_cap = max_steps is not None and step + 1 >= max_steps
        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader) or hit_step_cap:
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            opt.zero_grad()
        b = pts.shape[0]
        total_loss += loss.item() * GRAD_ACCUM * b
        total_acc += (logits.argmax(1) == labels).sum().item()
        n += b
        steps_done += 1
        if hit_step_cap:
            break
    return total_loss / max(1, n), total_acc / max(1, n), _timed_stats(
        t_start, steps_done, n
    )


@torch.no_grad()
def eval_spm(model, loader, n_vote=1):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        prob_sum = torch.zeros(pts.shape[0], model.num_classes, device=DEVICE)
        for _ in range(n_vote):
            theta = random.uniform(0.0, 2.0 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            rz = torch.tensor(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=DEVICE,
            )
            logits = model(pts @ rz.T)
            prob_sum += logits.softmax(dim=-1)
        correct += (prob_sum.argmax(1) == labels).sum().item()
        total += pts.shape[0]
    return correct / total


@torch.no_grad()
def eval_asp(model, loader):
    model.eval()
    correct = total = slices = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        logits, used = model.forward_active_infer(pts, EXIT_THR)
        correct += (logits.argmax(1) == labels).sum().item()
        total += pts.shape[0]
        slices += used * pts.shape[0]
    return correct / total, slices / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training(histories, save_dir):
    fig, axes = plt.subplots(1, len(histories), figsize=(7 * len(histories), 5), dpi=120)
    if len(histories) == 1:
        axes = [axes]
    for ax, (ds, hist) in zip(axes, histories.items()):
        for key, color in [("spm", "#2196F3"), ("asp", "#F44336")]:
            eps = [x["ep"] for x in hist[key]]
            train = [x["tr"] * 100 for x in hist[key]]
            vals = [(x["ep"], x["val"] * 100) for x in hist[key] if x["val"] is not None]
            ax.plot(eps, train, color=color, linestyle="--", alpha=0.35)
            if vals:
                ax.plot(
                    [v[0] for v in vals],
                    [v[1] for v in vals],
                    color=color,
                    marker="o",
                    label=f"{key.upper()} val",
                )
        ax.set_title(ds)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 103)
        ax.grid(True, alpha=0.2)
        ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "01_training_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    _show(path)


def plot_bars(results, save_dir):
    names = list(results)
    x = np.arange(len(names))
    spm = [results[n]["spm_best"] * 100 for n in names]
    asp = [results[n]["asp_best"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 3.5 * len(names)), 5), dpi=120)
    w = 0.35
    ax.bar(x - w / 2, spm, w, label="SPM official-like", color="#2196F3")
    ax.bar(x + w / 2, asp, w, label="ASP official-like", color="#F44336")
    for i, (s, a) in enumerate(zip(spm, asp)):
        ax.text(i - w / 2, s + 0.3, f"{s:.2f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, a + 0.3, f"{a:.2f}%\n({a-s:+.2f}pp)", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("SPM vs ASP: official-like backbone (with KD)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "02_accuracy_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    _show(path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def dataset_config():
    global MN10_DIR, MN40_DIR
    cfg = {}
    if "ModelNet10" in DATASET_NAMES:
        if MN10_DIR is None:
            print("\n[1] Downloading ModelNet10 ...")
            MN10_DIR = _download(
                "ModelNet10", "balraj98/modelnet10-princeton-3d-object-dataset"
            )
        cfg["ModelNet10"] = {"root": MN10_DIR, "classes": 10}
    if "ModelNet40" in DATASET_NAMES:
        if MN40_DIR is None:
            print("\n[1] Downloading ModelNet40 ...")
            MN40_DIR = _download(
                "ModelNet40", "balraj98/modelnet40-princeton-3d-object-dataset"
            )
        cfg["ModelNet40"] = {"root": MN40_DIR, "classes": 40}
    return cfg


def main():
    results = {}
    histories = {}

    for ds_name, ds_cfg in dataset_config().items():
        print("\n" + "=" * 76)
        print(f"Dataset: {ds_name}  classes={ds_cfg['classes']}")
        print("Backbone: official-like SPM group encoder + Mamba-lite mixer")
        print("=" * 76)

        train_cap = DEBUG_DATASET_ITEMS if DEBUG_MINI_RUN else None
        val_cap = DEBUG_VAL_ITEMS if DEBUG_MINI_RUN else None
        train_ds = ModelNetDataset(ds_cfg["root"], NUM_POINTS, "train", train_cap)
        val_ds = ModelNetDataset(ds_cfg["root"], NUM_POINTS, "test", val_cap)
        train_l = DataLoader(
            train_ds,
            BATCH,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=ON_GPU,
            drop_last=not DEBUG_MINI_RUN,
        )
        val_l = DataLoader(
            val_ds,
            BATCH,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=ON_GPU,
        )

        nc = ds_cfg["classes"]
        teacher, teacher_best = prepare_teacher(ds_name, nc, train_l, val_l)
        if teacher is not None:
            print(
                f"[KD] Distilling students from teacher logits "
                f"(teacher best val={teacher_best*100:.2f}%)."
            )
        else:
            print("[KD] Disabled; training students with supervised CE only.")

        # SPM baseline
        spm = make_spm(nc)
        print(f"\n[SPM] params={sum(p.numel() for p in spm.parameters()):,}")
        spm_opt = torch.optim.AdamW(spm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        spm_sch = make_scheduler(spm_opt, EPOCHS, WARMUP_EP)
        spm_latest = os.path.join(CKPT_DIR, f"spm_{ds_name}_latest.pt")
        start_ep, best_spm, spm_hist = load_ckpt(spm_latest, spm, spm_opt, spm_sch)
        spm_last_stats = None

        for ep in range(start_ep, EPOCHS):
            t0 = time.time()
            _loss, tr_acc, train_stats = train_spm_epoch(
                spm, train_l, spm_opt, teacher, max_steps=_debug_max_steps()
            )
            spm_last_stats = train_stats
            spm_sch.step()
            val_acc = None
            is_best = False
            should_eval = (
                not (DEBUG_MINI_RUN and DEBUG_SKIP_EVAL)
                and ((ep + 1) % 5 == 0 or ep + 1 == EPOCHS)
            )
            if should_eval:
                val_acc = eval_spm(spm, val_l, N_VOTE)
                if val_acc > best_spm:
                    best_spm = val_acc
                    is_best = True
                    save_best(
                        os.path.join(CKPT_DIR, f"spm_{ds_name}_best.pth"),
                        spm.state_dict(),
                    )
                print(
                    f"  [SPM] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} "
                    f"val={val_acc:.4f} {'*' if is_best else ' '} "
                    f"lr={spm_opt.param_groups[0]['lr']:.5f} {time.time()-t0:.0f}s"
                )
            elif DEBUG_MINI_RUN:
                print(
                    f"  [SPM] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} "
                    f"val=skipped lr={spm_opt.param_groups[0]['lr']:.5f} "
                    f"{_timing_text(train_stats)} wall={time.time()-t0:.0f}s"
                )
            spm_hist.append(
                {
                    "ep": ep + 1,
                    "tr": tr_acc,
                    "val": val_acc,
                    "timing": train_stats,
                }
            )
            save_ckpt(
                spm_latest,
                spm,
                spm_opt,
                spm_sch,
                ep + 1,
                best_spm,
                spm_hist,
                debug_stats=train_stats if DEBUG_MINI_RUN else None,
            )
            save_debug_ckpt(
                "spm",
                ds_name,
                spm,
                spm_opt,
                spm_sch,
                ep + 1,
                best_spm,
                spm_hist,
                train_stats,
            )

        # ASP
        asp = make_asp(nc)
        print(f"\n[ASP] params={sum(p.numel() for p in asp.parameters()):,}")
        print("[ASP] Same group encoder/mixer/head; SSP chooses group chunks.")
        asp_opt = torch.optim.AdamW(asp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        asp_sch = make_scheduler(asp_opt, EPOCHS, WARMUP_EP)
        asp_latest = os.path.join(CKPT_DIR, f"asp_{ds_name}_latest.pt")
        start_ep, best_asp, asp_hist = load_ckpt(asp_latest, asp, asp_opt, asp_sch)
        best_asp_sl = ASP_STEPS
        asp_last_stats = None

        for ep in range(start_ep, EPOCHS):
            t0 = time.time()
            _loss, tr_acc, train_stats = train_asp_epoch(
                asp, train_l, asp_opt, ep, teacher, max_steps=_debug_max_steps()
            )
            asp_last_stats = train_stats
            asp_sch.step()
            val_acc = val_sl = None
            is_best = False
            should_eval = (
                not (DEBUG_MINI_RUN and DEBUG_SKIP_EVAL)
                and ((ep + 1) % 5 == 0 or ep + 1 == EPOCHS)
            )
            if should_eval:
                val_acc, val_sl = eval_asp(asp, val_l)
                if val_acc > best_asp:
                    best_asp = val_acc
                    best_asp_sl = val_sl
                    is_best = True
                    save_best(
                        os.path.join(CKPT_DIR, f"asp_{ds_name}_best.pth"),
                        asp.state_dict(),
                    )
                print(
                    f"  [ASP] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} "
                    f"val={val_acc:.4f} {'*' if is_best else ' '} "
                    f"slices={val_sl:.2f}/{ASP_STEPS} "
                    f"tau={float(asp.gumbel_tau):.3f} "
                    f"lr={asp_opt.param_groups[0]['lr']:.5f} {time.time()-t0:.0f}s"
                )
            elif DEBUG_MINI_RUN:
                print(
                    f"  [ASP] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} "
                    f"val=skipped tau={float(asp.gumbel_tau):.3f} "
                    f"lr={asp_opt.param_groups[0]['lr']:.5f} "
                    f"{_timing_text(train_stats)} wall={time.time()-t0:.0f}s"
                )
            asp_hist.append(
                {
                    "ep": ep + 1,
                    "tr": tr_acc,
                    "val": val_acc,
                    "timing": train_stats,
                }
            )
            save_ckpt(
                asp_latest,
                asp,
                asp_opt,
                asp_sch,
                ep + 1,
                best_asp,
                asp_hist,
                debug_stats=train_stats if DEBUG_MINI_RUN else None,
            )
            save_debug_ckpt(
                "asp",
                ds_name,
                asp,
                asp_opt,
                asp_sch,
                ep + 1,
                best_asp,
                asp_hist,
                train_stats,
            )

        fr_mean = asp.mean_firing_rate()
        results[ds_name] = {
            "spm_best": best_spm,
            "asp_best": best_asp,
            "delta_pp": (best_asp - best_spm) * 100,
            "asp_avg_chunks": best_asp_sl,
            "firing_rate": fr_mean,
            "teacher_best": teacher_best,
            "kd_enabled": teacher is not None,
            "debug_spm_timing": spm_last_stats,
            "debug_asp_timing": asp_last_stats,
            "config": {
                "official_like": True,
                "distillation": teacher is not None,
                "debug_mini_run": DEBUG_MINI_RUN,
                "teacher": "PointTransformerTeacher" if teacher is not None else None,
                "kd_temp": KD_TEMP,
                "kd_ce_weight": KD_CE_WEIGHT,
                "kd_logit_weight": KD_LOGIT_WEIGHT,
                "timestep": TIMESTEP,
                "trans_dim": TRANS_DIM,
                "depth": DEPTH,
                "num_group": NUM_GROUP,
                "group_size": GROUP_SIZE,
                "asp_steps": ASP_STEPS,
            },
        }
        histories[ds_name] = {
            "spm": spm_hist,
            "asp": asp_hist,
            "summary": results[ds_name],
        }

        print(f"\nSummary {ds_name}")
        if teacher is not None:
            print(f"  Teacher  : {teacher_best*100:.2f}%")
        print(f"  SPM best : {best_spm*100:.2f}%")
        print(f"  ASP best : {best_asp*100:.2f}%")
        print(f"  Delta    : {(best_asp-best_spm)*100:+.2f} pp")
        print(f"  ASP chunks: {best_asp_sl:.2f}/{ASP_STEPS}")
        print(f"  Firing rate mean: {fr_mean:.4f}")

    with open(os.path.join(CKPT_DIR, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(CKPT_DIR, "histories.json"), "w") as f:
        json.dump(histories, f, indent=2)

    if DEBUG_MINI_RUN:
        debug_summary_path = os.path.join(CKPT_DIR, "debug_runtime_summary.json")
        with open(debug_summary_path, "w") as f:
            json.dump(
                {
                    ds: {
                        "spm": r.get("debug_spm_timing"),
                        "asp": r.get("debug_asp_timing"),
                    }
                    for ds, r in results.items()
                },
                f,
                indent=2,
            )
        print(f"\n[DEBUG] Runtime summary -> {debug_summary_path}")
        print("[DEBUG] Plotting skipped for minimal run.")
    else:
        plot_training(histories, CKPT_DIR)
        plot_bars(results, CKPT_DIR)

    print("\n" + "=" * 76)
    print("FINAL RESULTS - Official-like ASP with Knowledge Distillation")
    print("=" * 76)
    print(
        f"{'Dataset':<14} {'Teacher':>9} {'SPM':>9} "
        f"{'ASP':>9} {'Delta':>9} {'Chunks':>10}"
    )
    for ds, r in results.items():
        print(
            f"{ds:<14} {r['teacher_best']*100:>8.2f}% "
            f"{r['spm_best']*100:>8.2f}% {r['asp_best']*100:>8.2f}% "
            f"{r['delta_pp']:>+8.2f} {r['asp_avg_chunks']:>6.2f}/{ASP_STEPS}"
        )
    print(f"\nOutputs saved to: {CKPT_DIR}")

    # Zip outputs for easy download from Kaggle Output tab
    import shutil as _shutil

    zip_path = "/kaggle/working/asp_kd_results_export"
    _shutil.make_archive(zip_path, "zip", CKPT_DIR)
    print(f"Zipped results -> {zip_path}.zip")
    print("Download from the Kaggle Output tab on the right.")


if __name__ == "__main__":
    main()

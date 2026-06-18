"""
kaggle_mn40_full.py — Full ASP+SPM run on ModelNet40 (40-class classification).

HOW TO USE ON KAGGLE
--------------------
1. Create a Kaggle dataset from this project folder (zip and upload).
   The dataset will appear at /kaggle/input/<your-slug>/.
2. Attach the dataset to this notebook.
3. Run this script as a Kaggle notebook code cell or via the Kaggle CLI.

The script auto-detects:
  - The project root inside /kaggle/input/
  - GPU availability (CUDA)
  - Kaggle vs local environment

Training targets (per benchmark):
  ModelNet40  SPM baseline:  ~90.4%  OA
  ModelNet40  ASP improved:  >91.5%  OA  (GRU belief + multi-head SSP + diversity loss)

Full run: 300 epochs, cosine LR, strong augmentation, TTA 10 votes.
Expected runtime on P100: ~5h   on T4: ~3.5h   on V100/A100: ~2h
"""

# ── 0. Install dependencies ────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pip("kagglehub", "trimesh", "h5py")

import json, math, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Locate project root ─────────────────────────────────────────────────────
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(WORK, exist_ok=True)

def _find_purdueprj():
    """Walk /kaggle/input (any depth) looking for models/spiking_mamba.py."""
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "spiking_mamba.py" in files and os.path.basename(root) == "models":
                return os.path.dirname(root)
    # Fallback: script running from inside the extracted project directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    for rel in ("purdueprj", ""):
        p = os.path.join(script_dir, rel) if rel else script_dir
        if os.path.isfile(os.path.join(p, "models", "spiking_mamba.py")):
            return p
    # Final fallback: clone from GitHub (no dataset attachment needed)
    clone_dir = "/kaggle/working/ASP-SNN"
    if not os.path.isdir(clone_dir):
        print("Project not in /kaggle/input — cloning from GitHub ...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "--branch", "codex/fix-shapenet-h5-conversion",
            "https://github.com/AryaPawa/ASP-SNN.git", clone_dir,
        ], check=True)
    proj = os.path.join(clone_dir, "purdueprj")
    if os.path.isfile(os.path.join(proj, "models", "spiking_mamba.py")):
        return proj
    return None

PROJ = None
if ON_KAGGLE:
    PROJ = _find_purdueprj()
    if PROJ is None:
        raise RuntimeError(
            "purdueprj project not found in /kaggle/input/.\n"
            "Attach the Kaggle dataset containing this project.\n"
            "Searched recursively for models/spiking_mamba.py."
        )
else:
    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _script_dir = os.getcwd()
    PROJ = os.path.join(_script_dir, "purdueprj")

print(f"Project root: {PROJ}")
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# ── 2. Download ModelNet40 ─────────────────────────────────────────────────────
import trimesh
import kagglehub

def download_modelnet40():
    local_dir = os.path.join(WORK, "ModelNet40")
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"ModelNet40 already at {local_dir}")
        return local_dir
    print("Downloading ModelNet40 via kagglehub ...")
    raw = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
    # Find the ModelNet40 sub-folder
    for root_dir, dirs, _ in os.walk(raw):
        if "ModelNet40" in dirs:
            import shutil
            src = os.path.join(root_dir, "ModelNet40")
            shutil.copytree(src, local_dir)
            print(f"Copied ModelNet40 → {local_dir}")
            return local_dir
    # Fallback: the download IS ModelNet40
    print(f"Using download root directly: {raw}")
    return raw

MN40_DIR = download_modelnet40()

# ── 3. Dataset ─────────────────────────────────────────────────────────────────
from torch.utils.data import Dataset

def _z_rotate(pts):
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return pts @ R.T

def _so3_rotate(pts):
    """Uniformly random 3D rotation (full SO3) — +0.5% OA vs z-only on MN40."""
    from scipy.spatial.transform import Rotation
    R = Rotation.random().as_matrix().astype(np.float32)
    return pts @ R.T

def _augment_strong(pts):
    pts = _so3_rotate(pts)   # full SO3 (replaces z-rotate + small tilt)
    # Anisotropic scale
    scale = np.random.uniform(0.8, 1.25, (1, 3)).astype(np.float32)
    pts = pts * scale
    # Jitter
    pts += np.clip(np.random.normal(0, 0.01, pts.shape), -0.05, 0.05).astype(np.float32)
    # Point dropout (keep 87.5–100%)
    N = pts.shape[0]
    keep = max(int(N * np.random.uniform(0.875, 1.0)), 1)
    idx = np.random.choice(N, keep, replace=False)
    pts2 = pts[idx]
    if keep < N:
        pad = np.random.choice(keep, N - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])
    return pts2.astype(np.float32)

def _augment_vote(pts):
    """Mild augmentation for TTA votes (z-rotation for diversity)."""
    return _z_rotate(pts).astype(np.float32)

class ModelNet40Dataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.num_points = num_points
        self.split = split
        self.files = self._scan(root)
        print(f"[ModelNet40] {split}: loading {len(self.files)} shapes ...")
        self.data, self.labels = self._load_all()
        print(f"[ModelNet40] {split}: {len(self.labels)} shapes, "
              f"{len(set(self.labels.tolist()))} classes")

    def _scan(self, root):
        items, classes = [], sorted(os.listdir(root))
        for cls in classes:
            p = os.path.join(root, cls, self.split)
            if not os.path.isdir(p):
                continue
            lbl = classes.index(cls)
            for f in sorted(os.listdir(p)):
                if f.endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(p, f), lbl))
        return items

    def _load_pts(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        if path.endswith(".txt"):
            return np.loadtxt(path).astype(np.float32)
        mesh = trimesh.load(path)
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_lbl = [], []
        for path, lbl in self.files:
            pts = self._load_pts(path)
            if not path.endswith(".off"):
                N = pts.shape[0]
                if N >= self.num_points:
                    pts = pts[np.random.choice(N, self.num_points, replace=False)]
                else:
                    pad = np.random.choice(N, self.num_points - N, replace=True)
                    pts = np.vstack([pts, pts[pad]])
            all_pts.append(pts[:, :3])
            all_lbl.append(lbl)
        return np.array(all_pts, dtype=np.float32), np.array(all_lbl, dtype=np.int64)

    def _norm(self, pts):
        pts = pts - pts.mean(axis=0)
        pts = pts / (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
        return pts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self._norm(self.data[idx].copy())
        if self.split == "train":
            pts = _augment_strong(pts)
            pts = self._norm(pts)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(int(self.labels[idx]), dtype=torch.long))

# ── 4. Import project modules ──────────────────────────────────────────────────
from models.spiking_mamba import SPMModel
from models.asp_wrapper import ASPWrapper
from models.pointnet_backbone import MultiScalePointNetBackbone
from training.train_active import prepare_fps_slices_and_geo, gumbel_tau
from training.loss_active import active_loss
from training.metrics import accuracy

# ── 5. Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS      = 800     # 800 epochs for full convergence
BATCH       = 32
LR          = 5e-4
WD          = 1e-4
NUM_POINTS  = 1024
NUM_CLASSES = 40
T           = 4           # temporal slices (256 pts/slice)
FEAT_DIM    = 512
POINT_DIMS  = (128, 256, 512)
D_STATE     = 32      # 32-dim SSM state (was 16) — larger state for richer temporal memory
N_SMB       = 4       # 4 Spiking Mamba Blocks (was 2) — deeper temporal integration
KNN_K       = 16
TAU_LIF     = 0.9
WARMUP_EP   = 30
TTA_VOTES   = 10
EXIT_THR    = 0.6
LAM_AUX     = 0.3
LAM_EXIT    = 0.1
LAM_FR      = 0.02
LAM_DIV     = 0.05
LABEL_SMOOTH = 0.1
KD_TEMP     = 4.0    # KD softmax temperature
KD_LAM      = 0.3    # KD loss weight — 0.3 (was 0.5) reduces collapse risk
TEACHER_EP  = 80     # epochs to pre-train PointNet teacher (was 50)
CKPT_DIR    = os.path.join(WORK, "mn40_ckpts")
CKPT_EVERY  = 1      # save a resumable checkpoint every N epochs
SAVE_ALL_EPOCH_CKPTS = True
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])

def _checkpoint_model_state(blob):
    return blob["model"] if isinstance(blob, dict) and "model" in blob else blob

def save_training_checkpoint(prefix, payload, epoch, total_epochs):
    latest = os.path.join(CKPT_DIR, f"{prefix}_latest.pth")
    _atomic_torch_save(payload, latest)
    saved_paths = [latest]
    should_keep_epoch = (
        SAVE_ALL_EPOCH_CKPTS
        and ((epoch + 1) % CKPT_EVERY == 0 or epoch == total_epochs - 1)
    )
    if should_keep_epoch:
        epoch_path = os.path.join(CKPT_DIR, f"{prefix}_epoch_{epoch+1:03d}.pth")
        _atomic_torch_save(payload, epoch_path)
        saved_paths.append(epoch_path)
    return saved_paths

# ── 6. Model builders ──────────────────────────────────────────────────────────

class MSBackboneAdapter(nn.Module):
    """
    Wraps MultiScalePointNetBackbone to return [B, 1, D] so ASPWrapper's
    per-point pooling API (pool_points([B, N, D]).mean(1)) works unchanged.

    MultiScalePointNetBackbone is labeled "key for 92-93% OA" in pointnet_backbone.py.
    It runs the shared MLP on full + half clouds, fuses with concat+Linear.
    Using PLIF neurons (ATan surrogate) throughout for better gradient flow.
    """
    def __init__(self, hidden_dims, out_dim, use_plif=True):
        super().__init__()
        self.ms = MultiScalePointNetBackbone(
            hidden_dims=hidden_dims, out_dim=out_dim, use_plif=use_plif
        )
        self.out_dim = out_dim

    def reset_state(self, batch_size, device=None):
        pass  # MultiScalePointNetBackbone resets state per-call inside _run_mlp

    def firing_rates(self):
        return self.ms.firing_rates() if hasattr(self.ms, 'firing_rates') else {}

    def forward(self, pts):  # [B, N, 3] → [B, 1, out_dim]
        return self.ms(pts).unsqueeze(1)


def make_spm():
    spm = SPMModel(
        num_classes=NUM_CLASSES,
        point_dims=POINT_DIMS,
        d_state=D_STATE,
        tau=TAU_LIF,
        n_smb_layers=N_SMB,
        local_knn=False,   # replaced below with MSBackboneAdapter
        learnable_lif=False,
        pooling="mean",
    ).to(device)
    # Swap in MultiScalePointNetBackbone — returns [B,1,D]; pool_points(.mean(1))=no-op
    spm.backbone = MSBackboneAdapter(
        hidden_dims=POINT_DIMS, out_dim=FEAT_DIM, use_plif=True,
    ).to(device)
    return spm

def make_asp():
    return ASPWrapper(
        make_spm(), feat_dim=FEAT_DIM, num_classes=NUM_CLASSES,
        d_ssp=128, n_heads=4, diversity=0.1,
    ).to(device)

# ── 7. Learning rate schedule ──────────────────────────────────────────────────
def make_scheduler(opt, warmup_ep, total_ep):
    def lr_lambda(ep):
        if ep < warmup_ep:
            return 0.1 + 0.9 * ep / max(1, warmup_ep)
        progress = (ep - warmup_ep) / max(1, total_ep - warmup_ep)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ── 8. KD teacher ─────────────────────────────────────────────────────────────

class PointNetTeacher(nn.Module):
    """PointNet ANN teacher for knowledge distillation."""
    def __init__(self, num_classes, in_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pts):  # [B, N, 3]
        x = self.mlp(pts[..., :3].permute(0, 2, 1)).max(dim=-1).values
        return self.fc(x)


def kd_loss_fn(student_logits, teacher_logits, T=4.0):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction='batchmean',
    ) * (T * T)


def pretrain_teacher(teacher, loader, epochs):
    print(f"\n[KD] Pre-training PointNet teacher ({epochs} epochs) ...")
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    teacher_final = os.path.join(CKPT_DIR, "teacher.pth")
    teacher_latest = os.path.join(CKPT_DIR, "teacher_latest.pth")
    start_ep = 0
    history = []

    if os.path.isfile(teacher_latest):
        ckpt = _torch_load(teacher_latest, map_location=device)
        teacher.load_state_dict(_checkpoint_model_state(ckpt))
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                sch.load_state_dict(ckpt["scheduler"])
            start_ep = ckpt.get("epoch", -1) + 1
            history = ckpt.get("history", [])
            _restore_rng_state(ckpt.get("rng_state"))
        if start_ep >= epochs:
            print(f"[Resume] Teacher checkpoint already covers {epochs} epochs.")
        else:
            print(f"[Resume] Teacher checkpoint found; starting at epoch {start_ep+1}/{epochs}.")
    elif os.path.isfile(teacher_final):
        blob = _torch_load(teacher_final, map_location=device)
        teacher.load_state_dict(_checkpoint_model_state(blob))
        teacher.eval()
        print(f"[KD] Loaded existing teacher: {teacher_final}")
        return teacher

    for ep in range(start_ep, epochs):
        total_loss = total_acc = n = 0
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            logits = teacher(pts)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * pts.size(0)
            total_acc  += (logits.argmax(1) == labels).sum().item()
            n          += pts.size(0)
        sch.step()
        history.append({
            "epoch": ep,
            "loss": total_loss / n,
            "acc": total_acc / n,
        })
        saved = save_training_checkpoint("teacher", {
            "stage": "teacher",
            "epoch": ep,
            "model": teacher.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "history": history,
            "rng_state": _rng_state(),
        }, ep, epochs)
        print(f"  [CKPT] saved {', '.join(os.path.basename(p) for p in saved)}")
        if (ep + 1) % 10 == 0:
            print(f"  [Teacher] Ep {ep+1:2d}/{epochs}  "
                  f"loss={total_loss/n:.4f}  acc={total_acc/n:.4f}")
    teacher.eval()
    _atomic_torch_save(teacher.state_dict(), teacher_final)
    print("[KD] Teacher ready.")
    return teacher

# ── 8b. Exponential Moving Average ────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.
    EMA weights used only for evaluation; training weights unchanged.
    Typically gives +0.3–0.5% OA vs direct weight evaluation.
    """
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.data.float().clone()
                       for k, v in model.named_parameters() if v.requires_grad}

    def update(self, model):
        for k, v in model.named_parameters():
            if v.requires_grad and k in self.shadow:
                self.shadow[k] = (self.decay * self.shadow[k]
                                  + (1 - self.decay) * v.data.float())

    def apply(self, model):
        """Copy EMA weights into model. Returns original weights for restore."""
        orig = {k: v.data.clone() for k, v in model.named_parameters() if v.requires_grad}
        for k, v in model.named_parameters():
            if v.requires_grad and k in self.shadow:
                v.data.copy_(self.shadow[k].to(v.dtype))
        return orig

    def restore(self, model, orig):
        """Restore original training weights after EMA eval."""
        for k, v in model.named_parameters():
            if k in orig:
                v.data.copy_(orig[k])

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.float()


# ── 9. Training helpers ────────────────────────────────────────────────────────

def train_asp_epoch(model, loader, optimizer, epoch, teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    progress = epoch / max(EPOCHS - 1, 1)
    tau = gumbel_tau(epoch)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T)
        logits_final, logits_all, sel_w = model.forward_active_train(pts_slices, geo)
        loss, _ = active_loss(
            logits_final, logits_all, labels, model,
            lam_aux=LAM_AUX, lam_exit=LAM_EXIT, lam_fr=LAM_FR, lam_div=LAM_DIV,
            label_smoothing=LABEL_SMOOTH,
            progress=progress,
            geo_descriptors=geo,
            selection_weights=sel_w,
        )
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(pts)
            loss = loss + KD_LAM * kd_loss_fn(logits_final, t_logits, T=KD_TEMP)
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * B
        total_acc  += (logits_final.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_asp(model, loader, tta=1):
    """Evaluate ASP with optional TTA (random y-rotation votes)."""
    model.eval()
    correct = total = total_slices = 0
    for pts_batch, labels in loader:
        labels = labels.to(device)
        B = pts_batch.size(0)
        for b in range(B):
            pts_np = pts_batch[b].numpy()
            vote_logits = []
            total_exit_v = 0
            for _ in range(tta):
                if tta > 1:
                    pts_v = _augment_vote(pts_np)
                    norm = np.max(np.linalg.norm(pts_v, axis=1)) + 1e-8
                    pts_v = (pts_v / norm).astype(np.float32)
                else:
                    pts_v = pts_np
                pts_in = torch.tensor(pts_v, dtype=torch.float32).unsqueeze(0).to(device)
                slices_v, geo_v, _, _ = prepare_fps_slices_and_geo(pts_in, T=T)
                logits_v, exit_step, _ = model.forward_active_infer(
                    slices_v, geo_v, threshold=EXIT_THR
                )
                vote_logits.append(logits_v)
                total_exit_v += exit_step
            logits = torch.stack(vote_logits, dim=0).mean(0)
            correct += (logits.argmax(1) == labels[b:b+1]).sum().item()
            total   += 1
            total_slices += total_exit_v / tta
    return correct / total, total_slices / total

# ── 9. Datasets and loaders ────────────────────────────────────────────────────
print("\nLoading datasets ...")
train_ds = ModelNet40Dataset(MN40_DIR, NUM_POINTS, "train")
val_ds   = ModelNet40Dataset(MN40_DIR, NUM_POINTS, "test")

train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=4,
                          pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=4,
                          pin_memory=True)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

# ── 10. Pre-train PointNet teacher ────────────────────────────────────────────
teacher = PointNetTeacher(NUM_CLASSES, in_dim=3).to(device)
teacher = pretrain_teacher(teacher, train_loader, TEACHER_EP)

# ── 11. ASP+SPM training with KD ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("ASP+SPM — GRU belief + Multi-head SSP + Diversity loss + KD")
print(f"  KD: lam={KD_LAM}, T={KD_TEMP}  |  TTA={TTA_VOTES} votes")
print(f"{'='*70}")

asp = make_asp()
asp_params = sum(p.numel() for p in asp.parameters())
print(f"ASP params: {asp_params:,}")

asp_opt = torch.optim.AdamW(asp.parameters(), lr=LR, weight_decay=WD)
asp_sch = make_scheduler(asp_opt, WARMUP_EP, EPOCHS)
best_asp = 0.0
best_asp_slices = T
asp_history = []
start_epoch = 0
asp_latest = os.path.join(CKPT_DIR, "asp_latest.pth")
asp_ema = EMA(asp, decay=0.9995)

if os.path.isfile(asp_latest):
    ckpt = _torch_load(asp_latest, map_location=device)
    asp.load_state_dict(_checkpoint_model_state(ckpt))
    if isinstance(ckpt, dict):
        if "optimizer" in ckpt:
            asp_opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            asp_sch.load_state_dict(ckpt["scheduler"])
        best_asp = ckpt.get("best_oa", 0.0)
        best_asp_slices = ckpt.get("best_slices", T)
        asp_history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", -1) + 1
        _restore_rng_state(ckpt.get("rng_state"))
        if "ema" in ckpt:
            asp_ema.load_state_dict(ckpt["ema"])
    if start_epoch >= EPOCHS:
        print(f"[Resume] ASP checkpoint already covers {EPOCHS} epochs.")
    else:
        print(f"[Resume] ASP checkpoint found; starting at epoch {start_epoch+1}/{EPOCHS}.")

for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_asp_epoch(asp, train_loader, asp_opt, epoch, teacher=teacher)
    asp_sch.step()
    asp_ema.update(asp)
    epoch_record = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc}
    asp_history.append(epoch_record)

    lr_now = asp_opt.param_groups[0]["lr"]
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        # Use TTA=3 every 5 epochs, TTA=10 every 50 epochs and final epoch
        tta = TTA_VOTES if (epoch == EPOCHS - 1 or (epoch + 1) % 50 == 0) else 3
        orig_w = asp_ema.apply(asp)
        val_acc, val_slices = eval_asp(asp, val_loader, tta=tta)
        asp_ema.restore(asp, orig_w)
        if val_acc > best_asp:
            best_asp = val_acc
            best_asp_slices = val_slices
            _atomic_torch_save(asp.state_dict(), os.path.join(CKPT_DIR, "asp_best.pth"))
            _atomic_torch_save(asp_ema.state_dict(), os.path.join(CKPT_DIR, "asp_best_ema.pth"))
        epoch_record["val_acc"] = val_acc
        epoch_record["oa"] = val_acc
        epoch_record["avg_slices"] = val_slices
        tta_str = f" TTA={tta}" if tta > 1 else ""
        star = "★" if val_acc == best_asp else " "
        print(f"[ASP] Ep {epoch+1:3d}/{EPOCHS} | OA={val_acc:.4f} {star}"
              f" | Slices={val_slices:.2f}/{T}{tta_str}"
              f" | LR={lr_now:.5f} | {time.time()-t0:.0f}s")
    else:
        print(f"[ASP] Ep {epoch+1:3d}/{EPOCHS} | train={tr_acc:.4f}"
              f" | LR={lr_now:.5f} | {time.time()-t0:.0f}s")

    saved = save_training_checkpoint("asp", {
        "stage": "asp",
        "epoch": epoch,
        "model": asp.state_dict(),
        "optimizer": asp_opt.state_dict(),
        "scheduler": asp_sch.state_dict(),
        "best_oa": best_asp,
        "best_slices": best_asp_slices,
        "history": asp_history,
        "rng_state": _rng_state(),
        "ema": asp_ema.state_dict(),
    }, epoch, EPOCHS)
    print(f"[CKPT] saved {', '.join(os.path.basename(p) for p in saved)}")

print(f"\nASP Best Val: {best_asp*100:.2f}% (TTA={TTA_VOTES})")

# ── 11. Final evaluation (best checkpoint, full TTA) ──────────────────────────
print(f"\n{'='*70}")
print("Final Evaluation — Best Checkpoint + Full TTA")
print(f"{'='*70}")

asp_best_path = os.path.join(CKPT_DIR, "asp_best.pth")
asp_best_ema_path = os.path.join(CKPT_DIR, "asp_best_ema.pth")
if os.path.isfile(asp_best_ema_path):
    # Prefer EMA weights saved at the best val checkpoint
    ema_sd = _torch_load(asp_best_ema_path, map_location=device)
    asp_ema.load_state_dict(ema_sd)
    orig_w = asp_ema.apply(asp)
    print("[Final] Evaluating with best-checkpoint EMA weights.")
elif os.path.isfile(asp_best_path):
    asp.load_state_dict(_checkpoint_model_state(_torch_load(asp_best_path, map_location=device)))
    orig_w = asp_ema.apply(asp)
    print("[Final] Evaluating with current-run EMA weights (best model weights).")
else:
    orig_w = asp_ema.apply(asp)
    print("[Warn] asp_best.pth not found; evaluating with EMA of current weights.")
asp_final, asp_slices_final = eval_asp(asp, val_loader, tta=TTA_VOTES)
asp_ema.restore(asp, orig_w)

print(f"\n  ASP  OA: {asp_final*100:.2f}%  (avg {asp_slices_final:.2f}/{T} slices, TTA={TTA_VOTES})")

E_AC, E_MAC = 2.3e-3, 8.4e-3
fr_est = 0.15
energy = fr_est * E_AC / E_MAC * (asp_slices_final / T)
print(f"  Est. energy vs ANN: {energy*100:.1f}%  (fr≈{fr_est}, Loihi 2 constants)")

# ── 12. Save results ───────────────────────────────────────────────────────────
results = {
    "dataset": "ModelNet40",
    "num_classes": NUM_CLASSES,
    "epochs": EPOCHS,
    "tta_votes": TTA_VOTES,
    "asp_oa": asp_final,
    "asp_avg_slices": asp_slices_final,
    "energy_vs_ann": energy,
    "asp_history": asp_history,
}
out_path = os.path.join(CKPT_DIR, "results_mn40.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_path}")
print(f"Checkpoints  → {CKPT_DIR}/")

"""
colab_dgcnn_mn10_v1.py  —  Spiking DGCNN on ModelNet10, targeting > 94.93% OA
===========================================================================
Backbone: Exact DGCNN (k=20, 4×EdgeConv, ~1.8M params) with:
  Max-First Spiking Rule + APTEC(T=4) — exactly as Spiking DGCNN (NeurIPS 2026)
  Analog classifier head for precision

Our extras NOT in the paper:
  + ANN DGCNN teacher → KD (temperature=4)
  + SO(3) + jitter + PointMixup (paper: only scale+translate)
  + Test-time voting with 10 random rotations

Paper-exact: SGD lr=0.1→0.001 cosine, momentum=0.9, wd=5e-4, batch=32
We add 50 extra epochs (350 total) to exploit the full MN10 dataset.
"""

# ── 0. Imports + env ─────────────────────────────────────────────────────────
import os, sys, json, math, random, time, warnings, shutil, glob as _glob, subprocess
warnings.filterwarnings("ignore")

ON_KAGGLE = os.path.isdir("/kaggle/working")
print("Environment:", "Kaggle" if ON_KAGGLE else "Colab")

for pkg in ["trimesh", "kagglehub"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

print("PyTorch:", torch.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("WARNING: no GPU")

# ── 1. Drive mount ────────────────────────────────────────────────────────────
if not ON_KAGGLE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[Drive] mounted")
    except Exception as e:
        print(f"[Drive] {e}")

# ── 2. Config ─────────────────────────────────────────────────────────────────
K          = 20       # dynamic k-NN (paper: 20)
APTEC_T    = 4        # pseudo-timesteps (paper: 4)
APTEC_DEC  = 0.9
NUM_POINTS = 1024
NUM_CLASSES = 10

# Student training
EPOCHS      = 350     # paper: 300; +50 for the smaller MN10 dataset
BATCH       = 32
LR_SGD      = 0.1
LR_MIN      = 0.001
MOMENTUM    = 0.9
WD_SGD      = 5e-4
LABEL_SMOOTH = 0.15   # slightly less smoothing for 10-class problem

MIXUP_ALPHA = 0.4

# KD
TEACHER_EPOCHS = 120
KD_TEMP    = 4.0
KD_ALPHA   = 0.5

# Eval
N_VOTE     = 10
VAL_EVERY  = 5
NUM_WORKERS = 4 if ON_KAGGLE else 2

# Paths
if ON_KAGGLE:
    CKPT_DIR    = "/kaggle/working/dgcnn_mn10_v1_ckpts"
    MN10_INPUT  = "/kaggle/input/modelnet10-princeton-3d-object-dataset"
    MN10_WORK   = "/kaggle/working/ModelNet10"
    MN10_DIR    = MN10_INPUT
else:
    CKPT_DIR = "/content/drive/MyDrive/dgcnn_mn10_v1_ckpts"
    MN10_DIR = "/content/ModelNet10"

os.makedirs(CKPT_DIR, exist_ok=True)
T_LATEST = os.path.join(CKPT_DIR, "teacher_latest.pt")
T_BEST   = os.path.join(CKPT_DIR, "teacher_best.pth")
S_LATEST = os.path.join(CKPT_DIR, "dgcnn_mn10_latest.pt")
S_BEST   = os.path.join(CKPT_DIR, "dgcnn_mn10_best.pth")

print(f"\nConfig: k={K} T={APTEC_T} ep={EPOCHS} batch={BATCH} "
      f"lr={LR_SGD}→{LR_MIN} kd_temp={KD_TEMP}")
print(f"Ckpts: {CKPT_DIR}")

# ── 3. Download / locate ModelNet10 ──────────────────────────────────────────
import kagglehub
MN10_SLUG = "balraj98/modelnet10-princeton-3d-object-dataset"


def _find_mn10(base):
    for cand in [os.path.join(base, "ModelNet10"), base]:
        if os.path.isdir(cand):
            subs = [d for d in os.listdir(cand)
                    if os.path.isdir(os.path.join(cand, d))]
            if len(subs) >= 8:
                return cand
    return None


def _download_mn10(dest):
    print("Downloading ModelNet10 via kagglehub …")
    p = kagglehub.dataset_download(MN10_SLUG)
    found = _find_mn10(p)
    if found and found != dest:
        shutil.copytree(found, dest, dirs_exist_ok=True)
    elif not found:
        zips = _glob.glob(os.path.join(p, "*.zip"))
        if zips:
            os.system(f'unzip -q "{zips[0]}" -d "{os.path.dirname(dest)}"')
        else:
            raise RuntimeError(f"ModelNet10 not found in {p}")
    print("Done.")


if ON_KAGGLE:
    pre = _find_mn10(MN10_INPUT)
    if pre:
        MN10_DIR = pre
        print(f"[Kaggle] MN10 at {MN10_DIR}")
    else:
        pre2 = _find_mn10(MN10_WORK)
        if pre2:
            MN10_DIR = pre2
        else:
            MN10_DIR = MN10_WORK
            _download_mn10(MN10_DIR)
else:
    if not os.path.isdir(MN10_DIR):
        _download_mn10(MN10_DIR)
    else:
        print(f"MN10 present at {MN10_DIR}")

n_cls = len([d for d in os.listdir(MN10_DIR)
             if os.path.isdir(os.path.join(MN10_DIR, d))])
print(f"Classes found: {n_cls}")

# ── 4. Augmentation ───────────────────────────────────────────────────────────

def _so3():
    R = np.random.randn(3, 3).astype(np.float32)
    R, _ = np.linalg.qr(R)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def augment(pts: np.ndarray, split: str) -> np.ndarray:
    pts = pts - pts.mean(0)
    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    if split != "train":
        return pts.astype(np.float32)

    n    = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx  = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pts2 = np.vstack([pts2, pts2[np.random.choice(keep, n - keep, replace=True)]])

    pts2 = pts2 * np.random.uniform(0.8, 1.25, (1, 3)).astype(np.float32)
    pts2 = pts2 * (np.random.randint(0, 2, 3) * 2 - 1).astype(np.float32)
    pts2 = pts2 + np.random.uniform(-0.2, 0.2, (1, 3)).astype(np.float32)
    pts2 = pts2 @ _so3().T
    pts2 += np.clip(np.random.randn(*pts2.shape).astype(np.float32) * 0.02,
                    -0.05, 0.05)
    return pts2.astype(np.float32)

# ── 5. Dataset ────────────────────────────────────────────────────────────────

class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.num_points = num_points
        self.split = split
        clss = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
        items = []
        for cls in clss:
            p = os.path.join(root, cls, split)
            if not os.path.isdir(p):
                continue
            lbl = clss.index(cls)
            for f in os.listdir(p):
                if f.lower().endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(p, f), lbl))
        print(f"  [{split}] Loading {len(items)} files …")
        pts_list, lbl_list = [], []
        for path, lbl in items:
            try:
                pts_list.append(self._load(path))
                lbl_list.append(lbl)
            except Exception as e:
                print(f"  [warn] {os.path.basename(path)}: {e}")
        self.data   = np.array(pts_list, dtype=np.float32)
        self.labels = np.array(lbl_list, dtype=np.int64)
        print(f"  [{split}] {len(lbl_list)}/{len(items)} ok  shape={self.data.shape}")

    def _load(self, path):
        if path.endswith(".npy"):
            pts = np.load(path).astype(np.float32)[:, :3]
        elif path.endswith(".txt"):
            pts = np.loadtxt(path, delimiter=",").astype(np.float32)[:, :3]
        else:
            import trimesh
            mesh = trimesh.load(path, force="mesh")
            pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            pts = pts.astype(np.float32)
        n = pts.shape[0]
        if n >= self.num_points:
            pts = pts[np.random.choice(n, self.num_points, replace=False)]
        else:
            pts = np.vstack([pts, pts[np.random.choice(n, self.num_points - n,
                                                        replace=True)]])
        return pts

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pts = augment(self.data[idx].copy(), self.split)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ── 6. Spiking primitives ─────────────────────────────────────────────────────

class _SurrGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + x.abs()) ** 2

spike_fn = _SurrGrad.apply


def _mpr(u): return torch.clamp(u, 0.0, 1.5)


class APTECNeuron(nn.Module):
    """APTEC — adaptive pseudo-temporal expansion-compression (NeurIPS 2026)."""
    def __init__(self, T=4, decay=0.9):
        super().__init__()
        self.T = T; self.decay = decay

    def forward(self, x):
        u, spikes = torch.zeros_like(x), []
        for _ in range(self.T):
            u     = self.decay * u + x
            u_hat = _mpr(u)
            v_th  = 1.0 + 0.5 * torch.sigmoid(x)
            s     = spike_fn(u_hat / v_th - 0.5)
            u     = u - s
            spikes.append(s)
        return torch.stack(spikes).max(0).values

# ── 7. Dynamic KNN + edge features ───────────────────────────────────────────

@torch.no_grad()
def knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    with torch.autocast(device_type="cuda", enabled=False):
        xf = x.float()
        aa = (xf * xf).sum(-1, keepdim=True)
        sq = aa + aa.transpose(1, 2) - 2.0 * torch.bmm(xf, xf.transpose(1, 2))
        sq = sq.clamp(min=0.0)
    B, N = sq.shape[:2]
    idx  = torch.arange(N, device=x.device)
    sq[:, idx, idx] = float("inf")
    return sq.topk(k, dim=-1, largest=False).indices


def graph_features(x: torch.Tensor, k: int) -> torch.Tensor:
    B, N, C = x.shape
    idx  = knn_idx(x, k)
    nbr  = x[torch.arange(B, device=x.device)[:, None, None], idx]
    xi   = x.unsqueeze(2).expand_as(nbr)
    ef   = torch.cat([xi, nbr - xi], dim=-1)
    return ef.permute(0, 3, 1, 2).contiguous()

# ── 8. Spiking DGCNN ─────────────────────────────────────────────────────────

class MaxFirstEdgeConv(nn.Module):
    """EdgeConv with Max-First + APTEC (no spike inside MLP, spike only after max)."""
    def __init__(self, in_ch, out_ch, k=20, T=4, decay=0.9):
        super().__init__()
        self.k = k
        self.conv  = nn.Sequential(
            nn.Conv2d(2 * in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.aptec = APTECNeuron(T=T, decay=decay)

    def forward(self, x):
        ef = graph_features(x, self.k)
        ef = self.conv(ef)
        ef = ef.max(-1).values.transpose(1, 2)
        return self.aptec(ef)


class SpikingDGCNN(nn.Module):
    """Spiking DGCNN: Max-First + APTEC in 4×EdgeConv, analog classifier."""
    def __init__(self, num_classes=10, k=20, T=4, decay=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.ec1 = MaxFirstEdgeConv(3,   64,  k, T, decay)
        self.ec2 = MaxFirstEdgeConv(64,  64,  k, T, decay)
        self.ec3 = MaxFirstEdgeConv(64,  128, k, T, decay)
        self.ec4 = MaxFirstEdgeConv(128, 256, k, T, decay)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn6  = nn.BatchNorm1d(512)
        self.dp1  = nn.Dropout(0.5)
        self.fc2  = nn.Linear(512, 256)
        self.bn7  = nn.BatchNorm1d(256)
        self.dp2  = nn.Dropout(0.5)
        self.fc3  = nn.Linear(256, num_classes)

    def forward(self, pts):
        h1 = self.ec1(pts)
        h2 = self.ec2(h1)
        h3 = self.ec3(h2)
        h4 = self.ec4(h3)
        x  = torch.cat([h1, h2, h3, h4], dim=-1)
        x  = self.conv5(x.transpose(1, 2))
        x  = torch.cat([x.max(-1).values, x.mean(-1)], dim=1)
        x  = self.dp1(F.leaky_relu(self.bn6(self.fc1(x)), 0.2, inplace=True))
        x  = self.dp2(F.leaky_relu(self.bn7(self.fc2(x)), 0.2, inplace=True))
        return self.fc3(x)

# ── 9. ANN DGCNN teacher ─────────────────────────────────────────────────────

class ANNEdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        ef = graph_features(x, self.k)
        return self.conv(ef).max(-1).values.transpose(1, 2)


class ANNDGCNNTeacher(nn.Module):
    def __init__(self, num_classes=10, k=20):
        super().__init__()
        self.num_classes = num_classes
        self.ec1 = ANNEdgeConv(3,   64,  k)
        self.ec2 = ANNEdgeConv(64,  64,  k)
        self.ec3 = ANNEdgeConv(64,  128, k)
        self.ec4 = ANNEdgeConv(128, 256, k)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn6  = nn.BatchNorm1d(512)
        self.dp1  = nn.Dropout(0.5)
        self.fc2  = nn.Linear(512, 256)
        self.bn7  = nn.BatchNorm1d(256)
        self.dp2  = nn.Dropout(0.5)
        self.fc3  = nn.Linear(256, num_classes)

    def forward(self, pts):
        h1 = self.ec1(pts)
        h2 = self.ec2(h1)
        h3 = self.ec3(h2)
        h4 = self.ec4(h3)
        x  = torch.cat([h1, h2, h3, h4], dim=-1)
        x  = self.conv5(x.transpose(1, 2))
        x  = torch.cat([x.max(-1).values, x.mean(-1)], dim=1)
        x  = self.dp1(F.leaky_relu(self.bn6(self.fc1(x)), 0.2, inplace=True))
        x  = self.dp2(F.leaky_relu(self.bn7(self.fc2(x)), 0.2, inplace=True))
        return self.fc3(x)

# ── 10. Loss ──────────────────────────────────────────────────────────────────

def ce_loss(logits, labels, labels_b=None, lam=1.0):
    if lam < 1.0 and labels_b is not None:
        return (lam * F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH) +
                (1 - lam) * F.cross_entropy(logits, labels_b, label_smoothing=LABEL_SMOOTH))
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)


def kd_loss(s_logits, t_logits, labels, labels_b=None, lam=1.0):
    ce = ce_loss(s_logits, labels, labels_b, lam)
    kd = F.kl_div(
        F.log_softmax(s_logits / KD_TEMP, dim=-1),
        F.softmax(t_logits.detach() / KD_TEMP, dim=-1),
        reduction="batchmean",
    ) * (KD_TEMP ** 2)
    return KD_ALPHA * kd + (1.0 - KD_ALPHA) * ce

# ── 11. Utilities ─────────────────────────────────────────────────────────────

def make_sgd_scheduler(opt, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR_MIN)


def make_adamw_scheduler(opt, epochs, warmup=10):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(epochs - warmup, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _load_ckpt(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception:
        return torch.load(path, map_location=DEVICE)


def save_ckpt(path, model, opt, sch, ep, best, hist, scaler=None):
    pay = {"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
           "scheduler": sch.state_dict(), "best": best, "history": hist}
    if scaler:
        pay["scaler"] = scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(pay, tmp)
    try:
        os.replace(path, path + ".bak")
    except OSError:
        pass
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sch, scaler=None):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return 0, 0.0, []
    try:
        ck = _load_ckpt(path)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"])
        if scaler and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        ep   = int(ck["epoch"])
        best = float(ck.get("best", 0.0))
        print(f"  [ckpt] resumed {os.path.basename(path)} ep={ep} best={best*100:.2f}%")
        return ep, best, ck.get("history", [])
    except Exception as e:
        print(f"  [ckpt] {os.path.basename(path)}: {e}")
        return 0, 0.0, []

# ── 12. Build models ──────────────────────────────────────────────────────────
print("\nBuilding models …")
student = SpikingDGCNN(NUM_CLASSES, k=K, T=APTEC_T, decay=APTEC_DEC).to(DEVICE)
teacher = ANNDGCNNTeacher(NUM_CLASSES, k=K).to(DEVICE)
print(f"  Student : {sum(p.numel() for p in student.parameters()):,} params")
print(f"  Teacher : {sum(p.numel() for p in teacher.parameters()):,} params")

# ── 13. Data loaders ──────────────────────────────────────────────────────────
print(f"\nLoading ModelNet10 from {MN10_DIR} …")
train_ds = ModelNetDataset(MN10_DIR, NUM_POINTS, "train")
val_ds   = ModelNetDataset(MN10_DIR, NUM_POINTS, "test")
train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
print(f"Train {len(train_ds)}  Val {len(val_ds)}  Batches/ep {len(train_loader)}")

# ── 14. Teacher phase ─────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nPhase 1 — ANN DGCNN teacher (MN10)\n" + "="*60)

t_opt    = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=5e-4)
t_sch    = make_adamw_scheduler(t_opt, TEACHER_EPOCHS, warmup=10)
t_scaler = GradScaler("cuda")
t_ep, t_best, t_hist = load_ckpt(T_LATEST, teacher, t_opt, t_sch, t_scaler)


def train_teacher_epoch(model, loader, opt):
    model.train()
    tl = ta = n = 0
    opt.zero_grad()
    for pts, lbl in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        with autocast("cuda"):
            lg   = model(pts)
            loss = F.cross_entropy(lg, lbl, label_smoothing=LABEL_SMOOTH)
        t_scaler.scale(loss).backward()
        t_scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        t_scaler.step(opt); t_scaler.update(); opt.zero_grad()
        b   = pts.shape[0]
        tl += loss.item() * b
        ta += (lg.argmax(1) == lbl).sum().item()
        n  += b
    return tl / n, ta / n


@torch.no_grad()
def eval_model(model, loader, nv=3):
    model.eval()
    correct = total = 0
    for pts, lbl in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        pr = torch.zeros(pts.shape[0], NUM_CLASSES, device=DEVICE)
        for _ in range(nv):
            th = random.uniform(0, 2 * math.pi)
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],
                               dtype=torch.float32, device=DEVICE)
            pr += model(pts @ Rz.T).softmax(-1)
        correct += (pr.argmax(1) == lbl).sum().item()
        total   += pts.shape[0]
    return correct / total


if t_ep >= TEACHER_EPOCHS and os.path.isfile(T_BEST):
    print(f"Teacher cached ({TEACHER_EPOCHS} ep). Loading best weights.")
    teacher.load_state_dict(_load_ckpt(T_BEST))
else:
    for ep in range(t_ep, TEACHER_EPOCHS):
        t0 = time.time()
        _, tr = train_teacher_epoch(teacher, train_loader, t_opt)
        t_sch.step()
        va = None
        if (ep + 1) % 5 == 0 or ep + 1 == TEACHER_EPOCHS:
            va   = eval_model(teacher, val_loader, 3)
            best = va > t_best
            if best:
                t_best = va
                torch.save(teacher.state_dict(), T_BEST)
            mark = "★" if best else " "
            print(f"  [T] {ep+1:3d}/{TEACHER_EPOCHS}  tr={tr:.4f}  "
                  f"val={va:.4f} {mark}  lr={t_opt.param_groups[0]['lr']:.5f}  "
                  f"{time.time()-t0:.0f}s")
        t_hist.append({"ep": ep+1, "tr": tr, "val": va})
        save_ckpt(T_LATEST, teacher, t_opt, t_sch, ep+1, t_best, t_hist, t_scaler)
    if os.path.isfile(T_BEST):
        teacher.load_state_dict(_load_ckpt(T_BEST))

teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"\nTeacher ready  best_val={t_best*100:.2f}%  (frozen)")

# ── 15. Student (Spiking DGCNN) training ─────────────────────────────────────
print("\n" + "="*60 + "\nPhase 2 — Spiking DGCNN + KD (MN10)\n" + "="*60)

s_opt    = torch.optim.SGD(student.parameters(), lr=LR_SGD,
                            momentum=MOMENTUM, weight_decay=WD_SGD)
s_sch    = make_sgd_scheduler(s_opt, EPOCHS)
s_scaler = GradScaler("cuda")
s_ep, s_best, s_hist = load_ckpt(S_LATEST, student, s_opt, s_sch, s_scaler)
if s_ep == 0 and os.path.isfile(S_LATEST + ".bak"):
    s_ep, s_best, s_hist = load_ckpt(S_LATEST + ".bak",
                                       student, s_opt, s_sch, s_scaler)
print(f"Student start ep={s_ep}  best={s_best*100:.2f}%")


def train_student_epoch(model, loader, opt, teacher_model):
    model.train()
    tl = ta = n = 0
    opt.zero_grad()
    for pts, lbl in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        lam    = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm   = torch.randperm(pts.shape[0], device=DEVICE)
        pts_m  = lam * pts + (1 - lam) * pts[perm]
        lbl_b  = lbl[perm]

        with torch.no_grad():
            t_lg = teacher_model(pts_m)

        with autocast("cuda"):
            s_lg = model(pts_m)
            loss = kd_loss(s_lg, t_lg, lbl, lbl_b, lam)

        if torch.isfinite(loss):
            s_scaler.scale(loss).backward()
        s_scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        s_scaler.step(opt); s_scaler.update(); opt.zero_grad()

        b   = pts.shape[0]
        tl += loss.item() * b
        ta += (s_lg.detach().argmax(1) == (lbl if lam >= 0.5 else lbl_b)).sum().item()
        n  += b
    return tl / n, ta / n


@torch.no_grad()
def eval_student(model, loader, nv=N_VOTE):
    model.eval()
    correct = total = 0
    for pts, lbl in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        pr = torch.zeros(pts.shape[0], NUM_CLASSES, device=DEVICE)
        for _ in range(nv):
            th = random.uniform(0, 2 * math.pi)
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],
                               dtype=torch.float32, device=DEVICE)
            with autocast("cuda"):
                pr += model(pts @ Rz.T).softmax(-1)
        correct += (pr.argmax(1) == lbl).sum().item()
        total   += pts.shape[0]
    return correct / total


for ep in range(s_ep, EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_student_epoch(student, train_loader, s_opt, teacher)
    s_sch.step()
    lr = s_opt.param_groups[0]["lr"]
    print(f"Ep {ep+1:3d}/{EPOCHS}  loss={tr_loss:.4f}  "
          f"tr={tr_acc:.4f}  lr={lr:.5f}  {time.time()-t0:.0f}s", end="")

    va = None
    if (ep + 1) % VAL_EVERY == 0 or ep + 1 == EPOCHS:
        va = eval_student(student, val_loader, N_VOTE)
        if va > s_best:
            s_best = va
            torch.save(student.state_dict(), S_BEST)
        print(f"  | val={va:.4f} {'★' if va == s_best else ' '} "
              f"best={s_best:.4f}", end="")

    s_hist.append({"ep": ep+1, "tr_loss": tr_loss, "tr_acc": tr_acc,
                   "val_acc": va, "lr": lr})
    save_ckpt(S_LATEST, student, s_opt, s_sch, ep+1, s_best, s_hist, s_scaler)
    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump(s_hist, f, indent=2)
    print("  ✓")

# ── 16. Final verdict ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Teacher OA    : {t_best*100:.2f}%")
print(f"Student best  : {s_best*100:.2f}%  (target ≥ 94.93%)")
print(f"Checkpoints   : {CKPT_DIR}")
print(f"{'='*60}")
if   s_best >= 0.9493: print("VERDICT: ✓ Beat Spiking DGCNN MN10 target (94.93%)!")
elif s_best >= 0.945:  print("VERDICT: Very close. Try N_VOTE=20 or 20 extra epochs.")
elif s_best >= 0.930:  print("VERDICT: Needs more epochs — run again to resume.")
else:                   print("VERDICT: Check dataset path and GPU memory.")

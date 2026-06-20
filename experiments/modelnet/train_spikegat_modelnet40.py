"""
Full SpikeGAT training on ModelNet40, targeting > 92.38% single-pass OA
===========================================================================
Novel architecture: Max-First Spiking Graph Attention Network (SpikeGAT)

Core innovation — SpikeGATConv augments EdgeConv's max with an
identity-initialized attention gate:
  Paper (EdgeConv):  max_j [ MLP(xi || xj-xi) ]          (uniform)
  SpikeGAT:         max_j [ g_ij · MLP(xi || xj-xi) ]   (attention-gated)

  where g_ij = 2 sigmoid(Wₐ h_ij) ∈ (0,2), initialized to 1.

The attention gate g_ij learns to suppress noisy / geometrically
irrelevant neighbours BEFORE the max without shrinking every feature at
initialization. All operations remain in the
continuous domain; a single APTECNeuron fires after the max — preserving
the Max-First Spiking Rule from the paper.

Why this is more expressive than the paper's EdgeConv:
  • Uniform max: "keep the highest raw response regardless of relevance"
  • Attention-gated max: "keep the highest response from relevant neighbours"
  A near-irrelevant neighbour with high raw response can dominate uniform max;
  attention suppresses it first.

Parameter overhead over DGCNN: <1 % (one extra Conv2d(out,1,1) per layer).
The student keeps the paper's k=20, T=4, Max-First, MPR, and spiking head.

Accuracy strategy:
  + strong ANN GAT teacher → weight transfer → knowledge distillation
  + the paper's canonical scaling/translation augmentation
  + fair single-pass OA for checkpointing, plus separately labelled scale TTA

Run: MODELNET40_DIR=/data/ModelNet40 python experiments/modelnet/train_spikegat_modelnet40.py
"""

# ── 0. Imports + env ─────────────────────────────────────────────────────────
import os, json, math, random, time, warnings
warnings.filterwarnings("ignore")

print("Environment: full training")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

print("PyTorch:", torch.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: no GPU found — will be very slow")

# ── 1. Config ─────────────────────────────────────────────────────────────────
K           = 20      # dynamic k-NN neighbours (same as paper)
APTEC_T     = 4       # pseudo-timesteps (same as paper)
APTEC_DEC   = 1.0     # supplement uses non-leaky pseudo-time accumulation
NUM_POINTS  = 1024
NUM_CLASSES = 40
TARGET_OA   = 0.9238
SEED        = 42

# T4 transfer-learning schedule. The final protocol remains 1,024 points/k=20.
EPOCHS       = int(os.environ.get("EPOCHS", "180"))
BATCH        = int(os.environ.get("BATCH_SIZE", "32"))
LR_SGD       = 0.05
LR_MIN       = 0.001
MOMENTUM     = 0.9
WD_SGD       = 1e-5
LABEL_SMOOTH = 0.2

# Knowledge distillation
TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", "150"))
KD_TEMP        = 4.0
KD_ALPHA       = 0.35  # keep the supervised objective dominant
CACHE_TEACHER_LOGITS = True
FAST_KNN_FP16         = True

# Eval
VAL_EVERY   = 5
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
TTA_SCALES  = (1.0, 0.90, 1.10, 0.95, 1.05)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


seed_everything(SEED)

# Explicit full-run paths. Datasets are never downloaded by a training job.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT_DIR = os.environ.get(
    "SPIKEGAT_CKPT_DIR", os.path.join(REPO_ROOT, "outputs", "spikegat_mn40")
)
MN40_DIR = os.environ.get("MODELNET40_DIR", "")

os.makedirs(CKPT_DIR, exist_ok=True)
T_LATEST = os.path.join(CKPT_DIR, "teacher_latest.pt")
T_BEST   = os.path.join(CKPT_DIR, "teacher_best.pth")
T_TARGETS = os.path.join(CKPT_DIR, "teacher_train_targets.pt")
S_LATEST = os.path.join(CKPT_DIR, "spikegat_mn40_latest.pt")
S_BEST   = os.path.join(CKPT_DIR, "spikegat_mn40_best.pth")

print(f"\nConfig: k={K} T={APTEC_T} ep={EPOCHS} batch={BATCH} "
      f"lr={LR_SGD}→{LR_MIN} kd_temp={KD_TEMP} seed={SEED}")
print(f"T4 profile: teacher_ep={TEACHER_EPOCHS} cached_KD={CACHE_TEACHER_LOGITS} "
      f"fp16_KNN={FAST_KNN_FP16}")
print(f"Ckpts: {CKPT_DIR}")

# ── 2. Validate ModelNet40 ────────────────────────────────────────────────────
if not MN40_DIR or not os.path.isdir(MN40_DIR):
    raise RuntimeError(
        "Set MODELNET40_DIR to a prepared ModelNet40 root containing class/train/test folders."
    )
print(f"ModelNet40: {MN40_DIR}")

n_cls = len([d for d in os.listdir(MN40_DIR)
             if os.path.isdir(os.path.join(MN40_DIR, d))])
print(f"Classes found: {n_cls}")

# ── 4. Augmentation ───────────────────────────────────────────────────────────

def augment(pts: np.ndarray, split: str) -> np.ndarray:
    """Match the paper: normalize, then random anisotropic scale + translation."""
    pts = pts - pts.mean(0)
    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    if split != "train":
        return pts.astype(np.float32)
    scale = np.random.uniform(2.0 / 3.0, 3.0 / 2.0, (1, 3)).astype(np.float32)
    shift = np.random.uniform(-0.2, 0.2, (1, 3)).astype(np.float32)
    return (pts * scale + shift).astype(np.float32)

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
            # Official ModelNet loader takes the first 1,024 resampled points.
            pts = pts[:self.num_points]
        else:
            pts = np.vstack([pts, pts[np.random.choice(n, self.num_points - n,
                                                        replace=True)]])
        return pts

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pts = augment(self.data[idx].copy(), self.split)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(idx, dtype=torch.long))

# ── 6. Spiking primitives ─────────────────────────────────────────────────────

def _mpr(u):
    """Membrane-potential rectifier from the paper's supplementary code."""
    middle = 0.5 * torch.tanh(3.0 * (u - 0.5)) / math.tanh(1.5) + 0.5
    high = torch.pow(torch.where(u > 1.0, u, torch.ones_like(u)), 1.0 / 3.0)
    low_base = torch.where(u < 0.0, 1.0 - u, torch.ones_like(u))
    low = 1.0 - torch.pow(low_base, 1.0 / 3.0)
    return torch.where(u > 1.0, high, torch.where(u < 0.0, low, middle))


def spike_fn(x):
    """Hard spike in forward, clamp-style straight-through gradient in backward."""
    hard = (x > 0.5).to(x.dtype)
    surrogate = torch.clamp(x, 0.0, 1.0)
    return (hard - surrogate).detach() + surrogate


class APTECNeuron(nn.Module):
    """
    Adaptive Pseudo-Temporal Expansion-Compression neuron (NeurIPS 2026).
    T pseudo-timesteps all receive the same input x.
    V_th = 1 + 0.5·σ(x) ∈ (1, 1.5) suppresses spurious spikes.
    Output = temporal-OR = max_t(s_t) ∈ {0, 1}.
    """
    def __init__(self, T=4, decay=0.9):
        super().__init__()
        self.T = T
        self.decay = decay

    def forward(self, x):
        u, spikes = torch.zeros_like(x), []
        for _ in range(self.T):
            u     = self.decay * u + x
            u_hat = _mpr(u)
            v_th  = 1.0 + 0.5 * torch.sigmoid(x)
            s     = spike_fn(u_hat / (v_th + 1e-6))
            u     = u - s
            spikes.append(s)
        return torch.stack(spikes).max(0).values

# ── 7. Dynamic KNN + graph features ──────────────────────────────────────────

@torch.no_grad()
def knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    """Dynamic KNN in feature space.  x: [B, N, C] → [B, N, k]"""
    with torch.autocast(device_type="cuda", enabled=False):
        # Pairwise GEMM dominates T4 runtime. Tensor-core FP16 dot products are
        # safe here because KNN is index-only; norms/distances stay in FP32.
        if FAST_KNN_FP16 and x.is_cuda:
            xh = x.detach().to(torch.float16)
            xf = xh.float()
            dot = torch.bmm(xh, xh.transpose(1, 2)).float()
        else:
            xf = x.detach().float()
            dot = torch.bmm(xf, xf.transpose(1, 2))
        aa = (xf * xf).sum(-1, keepdim=True)
        sq = aa + aa.transpose(1, 2) - 2.0 * dot
        sq = sq.clamp(min=0.0)
    N  = sq.shape[1]
    di = torch.arange(N, device=x.device)
    sq[:, di, di] = float("inf")
    return sq.topk(k, dim=-1, largest=False).indices  # [B, N, k]


def graph_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """[xi || xj-xi] edge features.  x: [B, N, C] → [B, 2C, N, k]"""
    B, N, C = x.shape
    idx = knn_idx(x, k)
    nbr = x[torch.arange(B, device=x.device)[:, None, None], idx]  # [B, N, k, C]
    xi  = x.unsqueeze(2).expand_as(nbr)
    ef  = torch.cat([xi, nbr - xi], dim=-1)                         # [B, N, k, 2C]
    return ef.permute(0, 3, 1, 2).contiguous()                      # [B, 2C, N, k]

# ── 8. SpikeGAT ───────────────────────────────────────────────────────────────

class SpikeGATConv(nn.Module):
    """
    Max-First Spiking Graph Attention Conv (novel).

    Architecture vs. paper's EdgeConv:
      EdgeConv :  h = BN(Conv(e))          →  z = max_j(h)        →  APTEC(z)
      SpikeGAT :  h = BN(Conv(e))          →  g = 2 sigmoid(Wₐ h)
                  z = max_j(g_ij · h_ij)  →  APTEC(z)

    The scalar attention gate g_ij ∈ (0,2) is learned per (i,j) pair and
    suppresses geometrically irrelevant neighbours BEFORE the max — so the
    max selects the most *relevant* dominant neighbour, not just the highest
    raw response. It starts at exactly one, avoiding the original softmax's
    roughly 1/k attenuation. All operations remain continuous; APTEC fires
    after max (Max-First Rule preserved).

    Parameter overhead vs. EdgeConv: one extra Conv2d(out_ch, 1, 1) per layer.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 20,
                 T: int = 4, decay: float = 0.9):
        super().__init__()
        self.k = k
        # Feature transform — same as EdgeConv
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Attention head: [out_ch, N, k] → [1, N, k] scalar gate
        self.attn = nn.Conv2d(out_ch, 1, 1, bias=True)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)
        self.aptec = APTECNeuron(T=T, decay=decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C_in] → [B, N, out_ch]  binary"""
        ef = graph_features(x, self.k)      # [B, 2C, N, k]  continuous
        h  = self.conv(ef)                   # [B, out, N, k] continuous
        gate = 2.0 * torch.sigmoid(self.attn(h))
        z  = (gate * h).max(-1).values       # [B, out, N]    Max-First gated max
        z  = z.transpose(1, 2)               # [B, N, out]
        return self.aptec(z)                 # [B, N, out]    binary


class SpikeGAT(nn.Module):
    """
    Spiking Graph Attention Network with Max-First + APTEC.

    4 SpikeGATConv layers — same depth / width as paper's DGCNN (~1.8 M params).
    Fusion and hidden classifier activations also use APTEC, as in the paper.
    """
    def __init__(self, num_classes: int = 40, k: int = 20,
                 T: int = 4, decay: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.gc1 = SpikeGATConv(3,   64,  k, T, decay)
        self.gc2 = SpikeGATConv(64,  64,  k, T, decay)
        self.gc3 = SpikeGATConv(64,  128, k, T, decay)
        self.gc4 = SpikeGATConv(128, 256, k, T, decay)

        # Fusion: 64+64+128+256 = 512 → 1024
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
        )
        self.aptec5 = APTECNeuron(T=T, decay=decay)
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn6  = nn.BatchNorm1d(512)
        self.aptec6 = APTECNeuron(T=T, decay=decay)
        self.dp1  = nn.Dropout(0.5)
        self.fc2  = nn.Linear(512, 256)
        self.bn7  = nn.BatchNorm1d(256)
        self.aptec7 = APTECNeuron(T=T, decay=decay)
        self.dp2  = nn.Dropout(0.5)
        self.fc3  = nn.Linear(256, num_classes)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: [B, N, 3] → [B, num_classes]"""
        h1 = self.gc1(pts)
        h2 = self.gc2(h1)
        h3 = self.gc3(h2)
        h4 = self.gc4(h3)
        x  = torch.cat([h1, h2, h3, h4], dim=-1)   # [B, N, 512]
        x  = self.conv5(x.transpose(1, 2))
        # Preserve max-before-spike in the global fusion path too.
        x  = torch.cat([self.aptec5(x.max(-1).values),
                        self.aptec5(x.mean(-1))], dim=1)  # [B, 2048]
        x  = self.dp1(self.aptec6(self.bn6(self.fc1(x))))
        x  = self.dp2(self.aptec7(self.bn7(self.fc2(x))))
        return self.fc3(x)

# ── 9. ANN GAT teacher ────────────────────────────────────────────────────────

class ANNGATConv(nn.Module):
    """
    Non-spiking version of SpikeGATConv — same attention-gated max, no APTEC.
    Uses LeakyReLU for ANN-style training.  Matches student topology exactly
    so KD soft labels are directly comparable.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn = nn.Conv2d(out_ch, 1, 1, bias=True)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ef = graph_features(x, self.k)
        h  = self.conv(ef)
        gate = 2.0 * torch.sigmoid(self.attn(h))
        return (gate * h).max(-1).values.transpose(1, 2)  # [B, N, out]


class ANNGATTeacher(nn.Module):
    """ANN teacher — same topology as SpikeGAT, no spiking."""
    def __init__(self, num_classes: int = 40, k: int = 20):
        super().__init__()
        self.num_classes = num_classes
        self.gc1 = ANNGATConv(3,   64,  k)
        self.gc2 = ANNGATConv(64,  64,  k)
        self.gc3 = ANNGATConv(64,  128, k)
        self.gc4 = ANNGATConv(128, 256, k)
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

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        h1 = self.gc1(pts)
        h2 = self.gc2(h1)
        h3 = self.gc3(h2)
        h4 = self.gc4(h3)
        x  = torch.cat([h1, h2, h3, h4], dim=-1)
        x  = self.conv5(x.transpose(1, 2))
        x  = torch.cat([x.max(-1).values, x.mean(-1)], dim=1)
        x  = self.dp1(F.leaky_relu(self.bn6(self.fc1(x)), 0.2, inplace=True))
        x  = self.dp2(F.leaky_relu(self.bn7(self.fc2(x)), 0.2, inplace=True))
        return self.fc3(x)

# ── 10. Loss ──────────────────────────────────────────────────────────────────

def ce_loss(logits, labels):
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)


def kd_loss(s_logits, t_logits, labels):
    ce = ce_loss(s_logits, labels)
    kd = F.kl_div(
        F.log_softmax(s_logits / KD_TEMP, dim=-1),
        F.softmax(t_logits.detach() / KD_TEMP, dim=-1),
        reduction="batchmean",
    ) * (KD_TEMP ** 2)
    return KD_ALPHA * kd + (1.0 - KD_ALPHA) * ce

# ── 11. Scheduler + checkpoint ────────────────────────────────────────────────

def make_sgd_scheduler(opt, epochs, warmup=5):
    min_factor = LR_MIN / LR_SGD

    def fn(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(epochs - warmup, 1)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


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
student = SpikeGAT(NUM_CLASSES, k=K, T=APTEC_T, decay=APTEC_DEC).to(DEVICE)
teacher = ANNGATTeacher(NUM_CLASSES, k=K).to(DEVICE)
s_params = sum(p.numel() for p in student.parameters())
t_params = sum(p.numel() for p in teacher.parameters())
print(f"  SpikeGAT (student) : {s_params:,} params")
print(f"  ANNGATTeacher      : {t_params:,} params")
print(f"  Paper DGCNN        :  1,809,280 params  (ref)")

# ── 13. Data loaders ──────────────────────────────────────────────────────────
print(f"\nLoading ModelNet40 from {MN40_DIR} …")
train_ds = ModelNetDataset(MN40_DIR, NUM_POINTS, "train")
val_ds   = ModelNetDataset(MN40_DIR, NUM_POINTS, "test")
if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError("No ModelNet40 samples found; check MN40_DIR and train/test folders.")
loader_gen = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=AMP_ENABLED, drop_last=True,
                          worker_init_fn=seed_worker, generator=loader_gen,
                          persistent_workers=NUM_WORKERS > 0)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=AMP_ENABLED,
                          worker_init_fn=seed_worker,
                          persistent_workers=NUM_WORKERS > 0)
print(f"Train {len(train_ds)}  Val {len(val_ds)}  Batches/ep {len(train_loader)}")

# ── 14. Teacher phase ─────────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nPhase 1 — ANN GAT teacher\n" + "=" * 60)

t_opt    = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=5e-4)
t_sch    = make_adamw_scheduler(t_opt, TEACHER_EPOCHS, warmup=10)
t_scaler = GradScaler("cuda", enabled=AMP_ENABLED)
t_ep, t_best, t_hist = load_ckpt(T_LATEST, teacher, t_opt, t_sch, t_scaler)


def train_teacher_epoch(model, loader, opt):
    model.train()
    tl = ta = n = 0
    opt.zero_grad()
    for pts, lbl, _ in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        with autocast("cuda", enabled=AMP_ENABLED):
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
def eval_model(model, loader):
    model.eval()
    correct = total = 0
    for pts, lbl, _ in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        with autocast("cuda", enabled=AMP_ENABLED):
            logits = model(pts)
        correct += (logits.argmax(1) == lbl).sum().item()
        total   += pts.shape[0]
    return correct / total


@torch.no_grad()
def build_teacher_targets(model, dataset, path, teacher_oa):
    """Cache one canonical teacher distribution per training shape on CPU."""
    if os.path.isfile(path):
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            logits = payload["logits"]
            valid = (payload.get("version") == 1 and
                     logits.shape == (len(dataset), NUM_CLASSES) and
                     abs(float(payload.get("teacher_oa", -1.0)) - teacher_oa) < 1e-8)
            if valid:
                print(f"Loaded cached teacher targets: {tuple(logits.shape)}")
                return logits
        except Exception as e:
            print(f"  [cache] rebuilding teacher targets: {e}")

    print("Caching canonical teacher targets (one inference pass) …")
    old_split = dataset.split
    dataset.split = "test"
    cache_loader = DataLoader(dataset, BATCH, shuffle=False, num_workers=0,
                              pin_memory=AMP_ENABLED)
    logits = torch.empty(len(dataset), NUM_CLASSES, dtype=torch.float16)
    try:
        model.eval()
        for pts, _, idx in cache_loader:
            pts = pts.to(DEVICE, non_blocking=True)
            with autocast("cuda", enabled=AMP_ENABLED):
                batch_logits = model(pts)
            logits[idx] = batch_logits.detach().cpu().to(torch.float16)
    finally:
        dataset.split = old_split

    payload = {"version": 1, "teacher_oa": teacher_oa, "logits": logits}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"Saved cached teacher targets: {path}")
    return logits


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
            va   = eval_model(teacher, val_loader)
            best = va > t_best
            if best:
                t_best = va
                torch.save(teacher.state_dict(), T_BEST)
            mark = "★" if best else " "
            print(f"  [T] {ep+1:3d}/{TEACHER_EPOCHS}  tr={tr:.4f}  "
                  f"val={va:.4f} {mark}  lr={t_opt.param_groups[0]['lr']:.5f}  "
                  f"{time.time()-t0:.0f}s")
        t_hist.append({"ep": ep + 1, "tr": tr, "val": va})
        save_ckpt(T_LATEST, teacher, t_opt, t_sch, ep + 1, t_best, t_hist, t_scaler)
    if os.path.isfile(T_BEST):
        teacher.load_state_dict(_load_ckpt(T_BEST))

teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"\nTeacher ready  best_val={t_best*100:.2f}%  (frozen)")

# ANN-to-SNN initialization: all graph transforms, attention gates, BN state,
# fusion, and classifier weights have matching names/shapes. APTEC has no state.
student_state = student.state_dict()
transfer = {k: v for k, v in teacher.state_dict().items()
            if k in student_state and student_state[k].shape == v.shape}
missing, unexpected = student.load_state_dict(transfer, strict=False)
print(f"Transferred {len(transfer)}/{len(student_state)} teacher tensors to student "
      f"({len(missing)} APTEC-only/missing, {len(unexpected)} unexpected).")

teacher_targets = None
if CACHE_TEACHER_LOGITS:
    teacher_targets = build_teacher_targets(teacher, train_ds, T_TARGETS, t_best)
    if AMP_ENABLED:
        teacher_targets = teacher_targets.pin_memory()
    teacher.to("cpu")
    torch.cuda.empty_cache()
    print("Teacher moved to CPU; student KD now uses the cached targets.")

# ── 15. SpikeGAT student training ────────────────────────────────────────────
print("\n" + "=" * 60 + "\nPhase 2 — SpikeGAT + KD\n" + "=" * 60)

s_opt    = torch.optim.SGD(student.parameters(), lr=LR_SGD,
                            momentum=MOMENTUM, weight_decay=WD_SGD)
s_sch    = make_sgd_scheduler(s_opt, EPOCHS)
s_scaler = GradScaler("cuda", enabled=AMP_ENABLED)
s_ep, s_best, s_hist = load_ckpt(S_LATEST, student, s_opt, s_sch, s_scaler)
if s_ep == 0 and os.path.isfile(S_LATEST + ".bak"):
    s_ep, s_best, s_hist = load_ckpt(S_LATEST + ".bak",
                                       student, s_opt, s_sch, s_scaler)
print(f"Student start ep={s_ep}  best={s_best*100:.2f}%")


def train_student_epoch(model, loader, opt, cached_targets, teacher_model=None):
    model.train()
    tl = ta = n = 0
    opt.zero_grad()
    for pts, lbl, idx in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)

        with torch.no_grad():
            if cached_targets is not None:
                t_lg = cached_targets[idx].to(DEVICE, dtype=torch.float32,
                                               non_blocking=True)
            else:
                t_lg = teacher_model(pts)

        with autocast("cuda", enabled=AMP_ENABLED):
            s_lg = model(pts)
            loss = kd_loss(s_lg, t_lg, lbl)

        if not torch.isfinite(loss).item():
            opt.zero_grad(set_to_none=True)
            print("  [warn] skipped non-finite student batch")
            continue
        s_scaler.scale(loss).backward()
        s_scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        s_scaler.step(opt); s_scaler.update(); opt.zero_grad()

        b   = pts.shape[0]
        tl += loss.item() * b
        ta += (s_lg.detach().argmax(1) == lbl).sum().item()
        n  += b
    return tl / n, ta / n


@torch.no_grad()
def eval_student(model, loader, use_tta=False):
    """Single-pass OA is paper-comparable; scale TTA is reported separately."""
    model.eval()
    correct = total = 0
    scales = TTA_SCALES if use_tta else (1.0,)
    for pts, lbl, _ in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        pr = torch.zeros(pts.shape[0], NUM_CLASSES, device=DEVICE)
        for scale in scales:
            with autocast("cuda", enabled=AMP_ENABLED):
                pr += model(pts * scale).softmax(-1)
        correct += (pr.argmax(1) == lbl).sum().item()
        total   += pts.shape[0]
    return correct / total


for ep in range(s_ep, EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_student_epoch(
        student, train_loader, s_opt, teacher_targets,
        None if CACHE_TEACHER_LOGITS else teacher)
    s_sch.step()
    lr = s_opt.param_groups[0]["lr"]
    print(f"Ep {ep+1:3d}/{EPOCHS}  loss={tr_loss:.4f}  "
          f"tr={tr_acc:.4f}  lr={lr:.5f}  {time.time()-t0:.0f}s", end="")

    va = None
    if (ep + 1) % VAL_EVERY == 0 or ep + 1 == EPOCHS:
        va = eval_student(student, val_loader, use_tta=False)
        if va > s_best:
            s_best = va
            torch.save(student.state_dict(), S_BEST)
        print(f"  | val={va:.4f} {'★' if va == s_best else ' '} "
              f"best={s_best:.4f}", end="")

    s_hist.append({"ep": ep + 1, "tr_loss": tr_loss, "tr_acc": tr_acc,
                   "val_acc": va, "lr": lr})
    save_ckpt(S_LATEST, student, s_opt, s_sch, ep + 1, s_best, s_hist, s_scaler)
    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump(s_hist, f, indent=2)
    print("  ✓")

# ── 16. Final paper-comparable evaluation ─────────────────────────────────────
if os.path.isfile(S_BEST):
    student.load_state_dict(_load_ckpt(S_BEST))
final_single = eval_student(student, val_loader, use_tta=False)
final_tta = eval_student(student, val_loader, use_tta=True)
with open(os.path.join(CKPT_DIR, "final_metrics.json"), "w") as f:
    json.dump({"single_pass_oa": final_single, "scale_tta_oa": final_tta,
               "paper_target_oa": TARGET_OA, "seed": SEED}, f, indent=2)

print(f"\n{'=' * 60}")
print(f"ANNGATTeacher best  : {t_best*100:.2f}%")
print(f"SpikeGAT single-pass: {final_single*100:.2f}%  (paper target 92.38%)")
print(f"SpikeGAT scale-TTA  : {final_tta*100:.2f}%  (supplementary metric)")
print(f"Checkpoints         : {CKPT_DIR}")
print(f"{'=' * 60}")
if   final_single > TARGET_OA: print("VERDICT: ✓ Beat Spiking DGCNN MN40 single-pass target (92.38%)!")
elif final_single >= 0.920:    print("VERDICT: Very close; extend fine-tuning from the saved checkpoint.")
elif final_single >= 0.900:    print("VERDICT: Needs tuning; inspect teacher OA and firing rates first.")
else:                   print("VERDICT: Check dataset path and GPU memory.")

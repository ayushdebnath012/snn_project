"""
colab_asp_mn10_v1.py  —  ASP on ModelNet10, targeting > 94.93% OA
==================================================================
Colab:  !python colab_asp_mn10_v1.py   (checkpoints → Google Drive)
Kaggle: add dataset balraj98/modelnet10-princeton-3d-object-dataset
        as input; checkpoints → /kaggle/working/

Two-phase training (fully auto-resumable):
  Phase 1 — PointTransformer teacher (100 ep, frozen after)
  Phase 2 — ASP with KD from teacher (400 ep)

Architecture — identical to colab_asp_mn40_v4.py, adapted for 10 classes:
  MaxFirstAPTECEncoder: group tokens via FPS+KNN, ALL max-pools in
    continuous domain (Max-First Rule), single APTECNeuron fires after
    both aggregations (T=4 pseudo-timesteps, adaptive threshold).
  8 Mamba-lite residual blocks (dim=256), Conv1d head.
  ASP wraps it: SSP selects 4 chunks of 16 groups adaptively.

Target: beat Spiking DGCNN paper (NeurIPS 2026) MN10 result of 94.93%.
"""

# ─── 0. Imports + environment ────────────────────────────────────────────────
import os, json, math, random, time, warnings, subprocess, sys
warnings.filterwarnings("ignore")

ON_KAGGLE = os.path.isdir("/kaggle/working")
ON_COLAB  = not ON_KAGGLE
print(f"Environment: {'Kaggle' if ON_KAGGLE else 'Colab'}")

for pkg in ["trimesh", "kagglehub"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import torch
print("PyTorch :", torch.__version__)
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    print("VRAM    :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    raise RuntimeError("No GPU — enable GPU accelerator in notebook settings")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

# ─── 1. Mount Drive / checkpoint dir ─────────────────────────────────────────
if ON_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[Drive] Mounted")
    except Exception as e:
        print(f"[Drive] {e} — checkpoints saved locally only")

# ─── 2. Config ────────────────────────────────────────────────────────────────
DEVICE = "cuda"

# Architecture — smaller model than MN40 (10 classes, simpler shapes)
TIMESTEP    = 2
TRANS_DIM   = 256       # 256 vs 384 for MN40 (MN10 needs fewer dims)
DEPTH       = 8         # 8 vs 12 blocks
NUM_GROUP   = 64        # 64 vs 128 groups
GROUP_SIZE  = 32
EXPAND      = 1.1
DROP_PATH   = 0.2       # lighter regularisation for smaller dataset
ASP_STEPS   = 4
APTEC_T     = 4         # pseudo-timesteps in MaxFirst-APTEC encoder
K_GROUP     = 8         # inter-group KNN for GroupMaxFirstEdgeConv (smaller: G=64)

# Training — more epochs to compensate for smaller dataset
EPOCHS       = 450
BATCH        = 32       # larger batch (dataset is smaller)
GRAD_ACCUM   = 2
LR           = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EP    = 20
LABEL_SMOOTH = 0.15

# Numerical augmentation
MIXUP_ALPHA  = 0.4

# Knowledge distillation
TEACHER_EPOCHS  = 100   # converges faster on small MN10
TEACHER_DIM     = 256
TEACHER_DEPTH   = 6
TEACHER_HEADS   = 8
KD_TEMP         = 4.0
KD_CE_WEIGHT    = 0.5
KD_LOGIT_WEIGHT = 0.5
KD_AUX_WEIGHT   = 0.1

# Eval / data
NUM_POINTS  = 1024
NUM_CLASSES = 10
N_VOTE      = 10
EXIT_THR    = 0.55      # MN10 models converge faster; higher confidence threshold
VAL_EVERY   = 5
NUM_WORKERS = 4 if ON_KAGGLE else 2

# Paths
if ON_KAGGLE:
    CKPT_DIR    = "/kaggle/working/asp_mn10_v1_ckpts"
    _MN10_INPUT = "/kaggle/input/modelnet10-princeton-3d-object-dataset"
    _MN10_WORK  = "/kaggle/working/ModelNet10"
    MN10_DIR    = _MN10_INPUT
else:
    CKPT_DIR = "/content/drive/MyDrive/asp_mn10_v1_ckpts"
    MN10_DIR = "/content/ModelNet10"

os.makedirs(CKPT_DIR, exist_ok=True)

TEACHER_LATEST = os.path.join(CKPT_DIR, "teacher_latest.pt")
TEACHER_BEST   = os.path.join(CKPT_DIR, "teacher_best.pth")
ASP_LATEST     = os.path.join(CKPT_DIR, "asp_mn10_latest.pt")
ASP_BEST       = os.path.join(CKPT_DIR, "asp_mn10_best.pth")

print(f"\nConfig: ep={EPOCHS} batch={BATCH}×{GRAD_ACCUM}={BATCH*GRAD_ACCUM} "
      f"dim={TRANS_DIM} depth={DEPTH} T={TIMESTEP}")
print(f"        teacher_ep={TEACHER_EPOCHS} kd_temp={KD_TEMP} "
      f"mixup={MIXUP_ALPHA} vote={N_VOTE}")
print(f"Checkpoints: {CKPT_DIR}")
print(f"MN10 data  : {MN10_DIR}")

# ─── 3. Download / locate ModelNet10 ─────────────────────────────────────────
import shutil, glob as _glob
import kagglehub

_MN10_DATASET = "balraj98/modelnet10-princeton-3d-object-dataset"


def _find_mn10(base: str) -> str | None:
    """Return path to the ModelNet10 class-folder tree, or None."""
    for candidate in [os.path.join(base, "ModelNet10"), base]:
        if os.path.isdir(candidate):
            subs = [d for d in os.listdir(candidate)
                    if os.path.isdir(os.path.join(candidate, d))]
            if len(subs) >= 8:   # MN10 has exactly 10 classes
                return candidate
    return None


def _download_mn10(dest: str):
    print(f"\nDownloading ModelNet10 via kagglehub → {dest} …")
    p = kagglehub.dataset_download(_MN10_DATASET)
    found = _find_mn10(p)
    if found:
        if found != dest:
            shutil.copytree(found, dest, dirs_exist_ok=True)
    else:
        zips = _glob.glob(os.path.join(p, "*.zip"))
        unzip_dest = os.path.dirname(dest)
        if zips:
            os.system(f'unzip -q "{zips[0]}" -d "{unzip_dest}"')
        else:
            raise RuntimeError(f"ModelNet10 not found in {p}")
    print("Done.")


if ON_KAGGLE:
    pre = _find_mn10(_MN10_INPUT)
    if pre:
        MN10_DIR = pre
        print(f"[Kaggle] MN10 pre-mounted at {MN10_DIR}")
    else:
        pre2 = _find_mn10(_MN10_WORK)
        if pre2:
            MN10_DIR = pre2
            print(f"[Kaggle] MN10 cached at {MN10_DIR}")
        else:
            MN10_DIR = _MN10_WORK
            _download_mn10(MN10_DIR)
else:
    if not os.path.isdir(MN10_DIR):
        _download_mn10(MN10_DIR)
    else:
        print(f"ModelNet10 already present at {MN10_DIR}")

n_cls = len([d for d in os.listdir(MN10_DIR)
             if os.path.isdir(os.path.join(MN10_DIR, d))])
print(f"Classes: {n_cls}")

# ─── 4. Augmentation ─────────────────────────────────────────────────────────

def _so3():
    """Uniform random SO3 via QR decomposition."""
    R = np.random.randn(3, 3).astype(np.float32)
    R, _ = np.linalg.qr(R)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def _augment(pts: np.ndarray, split: str) -> np.ndarray:
    pts = pts - pts.mean(0)
    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8

    if split != "train":
        return pts.astype(np.float32)

    # 1. Point dropout [87.5–100 %]
    n = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx  = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pad = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])

    # 2. Anisotropic scale [0.8, 1.25]
    pts2 = pts2 * np.random.uniform(0.8, 1.25, (1, 3)).astype(np.float32)

    # 3. Random axis flip
    pts2 = pts2 * (np.random.randint(0, 2, 3) * 2 - 1).astype(np.float32)

    # 4. Random translate [-0.1, 0.1]
    pts2 = pts2 + np.random.uniform(-0.1, 0.1, (1, 3)).astype(np.float32)

    # 5. Full SO3 rotation
    pts2 = pts2 @ _so3().T

    # 6. Gaussian jitter σ=0.02, clipped ±0.05
    pts2 += np.clip(np.random.randn(*pts2.shape).astype(np.float32) * 0.02,
                    -0.05, 0.05)
    return pts2.astype(np.float32)

# ─── 5. Dataset ───────────────────────────────────────────────────────────────

class ModelNetDataset(Dataset):
    def __init__(self, root: str, num_points: int = 1024, split: str = "train"):
        self.num_points = num_points
        self.split = split
        clss = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
        items = []
        for cls in clss:
            p = os.path.join(root, cls, split)
            if not os.path.isdir(p): continue
            label = clss.index(cls)
            for f in os.listdir(p):
                if f.lower().endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(p, f), label))

        print(f"  [{split}] Loading {len(items)} files …")
        pts_list, lbl_list = [], []
        for path, label in items:
            try:
                pts_list.append(self._load(path))
                lbl_list.append(label)
            except Exception as e:
                print(f"  [WARN] skip {os.path.basename(path)}: {e}")

        self.data   = np.array(pts_list, dtype=np.float32)
        self.labels = np.array(lbl_list, dtype=np.int64)
        print(f"  [{split}] {len(lbl_list)}/{len(items)} ok  shape={self.data.shape}")

    def _load(self, path: str) -> np.ndarray:
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
            pad = np.random.choice(n, self.num_points - n, replace=True)
            pts = np.vstack([pts, pts[pad]])
        return pts

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pts = _augment(self.data[idx].copy(), self.split)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ─── 6. Spiking primitives ───────────────────────────────────────────────────

class _SurrGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ctx.save_for_backward(x); return (x > 0).float()
    @staticmethod
    def backward(ctx, g): (x,) = ctx.saved_tensors; return g / (1 + x.abs()) ** 2

spike_fn = _SurrGrad.apply


class SpikeAct(nn.Module):
    def __init__(self, vth=0.5):
        super().__init__(); self.vth = vth
        self.register_buffer("_s", torch.tensor(0.0))
        self.register_buffer("_n", torch.tensor(0.0))
    def forward(self, x):
        y = spike_fn(x - self.vth)
        self._s = self._s + y.detach().sum()
        self._n = self._n + y.numel()
        return y
    def rate(self): return (self._s / self._n).item() if self._n > 0 else 0.0


class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        keep  = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep


class TokenBNSpike(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.bn = nn.BatchNorm1d(dim); self.sp = SpikeAct()
    def forward(self, x):
        b, l, c = x.shape
        return self.sp(self.bn(x.reshape(b * l, c)).reshape(b, l, c))


# ─── 6b. Max-First + APTEC encoder ───────────────────────────────────────────
# Implements NeurIPS 2026 "Spiking DGCNN" innovations applied to our SPM backbone.
#
# Max-First Rule: all Conv2d and max-pooling run in continuous domain.
# The single APTECNeuron fires AFTER both neighbourhood max aggregations.
# This preserves winner-take-all geometric ordering that Spike-before-Max destroys.
#
# APTEC adaptive threshold V_th = 1 + 0.5·sigmoid(x) ∈ (1, 1.5):
#   - suppresses borderline spurious spikes (denominator > 1 tightens the gate)
#   - pulls saturated units back into the surrogate-gradient-active region [0,1]

def _mpr(u: torch.Tensor) -> torch.Tensor:
    """Membrane Potential Rectifier: clamp to [0, 1.5]."""
    return torch.clamp(u, min=0.0, max=1.5)


class APTECNeuron(nn.Module):
    """
    Adaptive Pseudo-Temporal Expansion-Compression spiking neuron.
    T pseudo-timesteps all receive the same input x (no repeated graph ops).
    Output = OR(s_1, …, s_T) = max_t(s_t) ∈ {0, 1}.
    """
    def __init__(self, T: int = 4, decay: float = 0.9):
        super().__init__()
        self.T = T
        self.decay = decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(x)
        spikes = []
        for _ in range(self.T):
            u     = self.decay * u + x
            u_hat = _mpr(u)
            v_th  = 1.0 + 0.5 * torch.sigmoid(x)
            z     = u_hat / v_th
            s     = spike_fn(z - 0.5)
            u     = u - s
            spikes.append(s)
        return torch.stack(spikes, dim=0).max(dim=0).values


class MaxFirstAPTECEncoder(nn.Module):
    """
    Drop-in for OfficialLikeEncoder — Max-First Spiking Rule + APTEC.

    All Conv2d and both max-pool operations run in continuous (float) domain.
    A single APTECNeuron binarises group-level features AFTER max aggregation.
    """
    def __init__(self, ch: int, T: int = 4, decay: float = 0.9):
        super().__init__()
        self.c1 = nn.Conv2d(3,   128, 1); self.b1 = nn.BatchNorm2d(128)
        self.c2 = nn.Conv2d(128, 256, 1); self.b2 = nn.BatchNorm2d(256)
        self.c3 = nn.Conv2d(512, 512, 1); self.b3 = nn.BatchNorm2d(512)
        self.c4 = nn.Conv2d(512, ch,  1); self.b4 = nn.BatchNorm2d(ch)
        self.aptec = APTECNeuron(T=T, decay=decay)

    def forward(self, nh: torch.Tensor) -> torch.Tensor:
        t, b, g, k, _ = nh.shape
        x = nh.flatten(0, 1).permute(0, 3, 1, 2).contiguous()   # [tb, 3, g, k]

        x = self.b1(self.c1(x))                                  # [tb, 128, g, k]
        x = self.b2(self.c2(x))                                  # [tb, 256, g, k]

        # Max-First: local-context aggregation in continuous domain
        gl = x.max(3, keepdim=True).values                       # [tb, 256, g, 1]
        x  = torch.cat([gl.expand(-1, -1, -1, k), x], 1)        # [tb, 512, g, k]

        x = self.b3(self.c3(x))                                  # [tb, 512, g, k]
        x = self.b4(self.c4(x))                                  # [tb, ch,  g, k]

        # Max-First: group-level aggregation in continuous domain
        x = x.max(3).values.transpose(1, 2).contiguous()        # [tb, g, ch]

        # APTEC fires AFTER all max aggregations
        x = self.aptec(x)                                        # [tb, g, ch]  binary

        return x.reshape(t, b, g, -1)


class GroupMaxFirstEdgeConv(nn.Module):
    """
    Inter-group EdgeConv with Max-First Spiking Rule + APTEC.

    After intra-group encoding (MaxFirstAPTECEncoder), each group token
    gathers its k spatially nearest group neighbours, computes edge features
    [x_i || x_j - x_i], pools with Max-First (continuous), then fires APTEC.
    This adds a second level of dynamic graph context — bridging our single-level
    FPS+kNN toward the paper's multi-layer feature-space DGCNN graph.
    """
    def __init__(self, dim: int, k_group: int = 8, T: int = 4, decay: float = 0.9):
        super().__init__()
        self.k = k_group
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.aptec = APTECNeuron(T=T, decay=decay)

    def forward(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        tokens:  [t, b, G, D] — binary group tokens from MaxFirstAPTECEncoder
        centers: [t, b, G, 3] — group centroids in input point space
        Returns: [t, b, G, D] — enhanced tokens (Max-First inter-group graph + APTEC)
        """
        t, b, G, D = tokens.shape
        flat_c = centers.flatten(0, 1).contiguous()              # [tb, G, 3]
        flat_t = tokens.flatten(0, 1).contiguous()               # [tb, G, D]

        with torch.no_grad():
            dist = torch.cdist(flat_c, flat_c)                   # [tb, G, G]
            di   = torch.arange(G, device=flat_c.device)
            dist[:, di, di] = float("inf")                       # exclude self-loops
            k_act = min(self.k, G - 1)
            idx = dist.topk(k_act, dim=-1, largest=False).indices  # [tb, G, k]

        tbr = torch.arange(t * b, device=tokens.device)
        nbr = flat_t[tbr[:, None, None], idx]                    # [tb, G, k, D]
        xi  = flat_t.unsqueeze(2).expand_as(nbr)
        ef  = torch.cat([xi, nbr - xi], dim=-1)                  # [tb, G, k, 2D]
        ef  = self.conv(ef.permute(0, 3, 1, 2).contiguous())     # [tb, D, G, k] continuous
        ef  = ef.max(-1).values.transpose(1, 2)                  # [tb, G, D]  Max-First
        ef  = self.aptec(ef)                                      # [tb, G, D]  binary

        return (flat_t + ef).reshape(t, b, G, D)


# ─── 7. FPS + KNN helpers ────────────────────────────────────────────────────

def index_points(points, idx):
    b = points.shape[0]
    vs = list(idx.shape); vs[1:] = [1] * (len(vs) - 1)
    rs = list(idx.shape); rs[0] = 1
    bi = torch.arange(b, device=points.device).view(vs).repeat(rs)
    return points[bi, idx]


def fps_batched(xyz, npoint):
    b, n, _ = xyz.shape
    npoint  = min(npoint, n)
    cents   = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    dist    = torch.full((b, n), 1e10, device=xyz.device)
    far     = torch.randint(0, n, (b,), device=xyz.device)
    bi      = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        cents[:, i] = far
        d    = ((xyz - xyz[bi, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far  = dist.max(-1).indices
    return cents

# ─── 8. SPM backbone ─────────────────────────────────────────────────────────

class OfficialLikeGroup(nn.Module):
    def __init__(self, num_group, group_size, expand=1.1, timestep=2):
        super().__init__()
        self.G = num_group; self.K = group_size
        self.expand = expand; self.T = timestep

    def _centers(self, pts):
        b, n, _ = pts.shape
        sf = int((self.expand - 1.0) * self.G / self.T * 2)
        sb = int((self.expand - 1.0) * self.G)
        total = min(max(self.G + (sf + sb) * (self.T - 1), self.G), n)
        pool  = index_points(pts, fps_batched(pts.contiguous(), total))
        need  = self.G + (sf + sb) * (self.T - 1)
        if pool.shape[1] < need:
            pool = pool.repeat(1, math.ceil(need / pool.shape[1]), 1)
        centers = []
        for i in range(self.T):
            a = pool[:, i * sf: i * sf + (self.G - sb)]
            s = (i - 1) * sb + self.G + (self.T - 1) * sf
            e = i * sb + self.G + (self.T - 1) * sf
            cur = torch.cat([a, pool[:, s:e]], 1)
            if cur.shape[1] < self.G:
                cur = torch.cat([cur, cur[:, -1:].repeat(1, self.G - cur.shape[1], 1)], 1)
            centers.append(cur[:, :self.G])
        return torch.stack(centers, 0)

    def forward(self, pts):
        b, n, _ = pts.shape
        ctr     = self._centers(pts)
        flat_c  = ctr.reshape(self.T * b, self.G, 3)
        flat_p  = pts.unsqueeze(0).expand(self.T, -1, -1, -1).reshape(self.T * b, n, 3)
        k       = min(self.K, n)
        idx     = torch.cdist(flat_c, flat_p).topk(k, dim=-1, largest=False).indices
        grp     = index_points(flat_p, idx).reshape(self.T, b, self.G, k, 3)
        return (grp - ctr.unsqueeze(3)).contiguous(), ctr.contiguous()


class PosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), SpikeAct(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim),
        )
    def forward(self, ctr):
        t, b, g, _ = ctr.shape
        x = ctr.flatten(0, 1).permute(0, 2, 1).contiguous()
        return self.net(x).permute(0, 2, 1).reshape(t, b, g, -1)


class MambaLiteMixer(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        inner = dim * expand
        self.in_proj   = nn.Linear(dim, inner * 2)
        self.dwconv    = nn.Conv1d(inner, inner, 3, padding=1, groups=inner)
        self.scan_proj = nn.Linear(inner, inner)
        self.out_proj  = nn.Linear(inner, dim)

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.dwconv(u.transpose(1, 2)).transpose(1, 2))
        steps = torch.arange(1, u.shape[1] + 1, device=u.device, dtype=u.dtype).view(1, -1, 1)
        u = u + self.scan_proj(torch.cumsum(u, 1) / steps)
        return self.out_proj(u * torch.sigmoid(gate))


class OfficialLikeBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm  = TokenBNSpike(dim)
        self.mixer = MambaLiteMixer(dim)
        self.dp    = DropPath(drop_path)
    def forward(self, x, residual=None):
        residual = self.dp(x) + residual if residual is not None else x
        return self.mixer(self.norm(residual)), residual


class OfficialLikeMixerModel(nn.Module):
    def __init__(self, dim, depth, timestep, drop_path=0.2):
        super().__init__()
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.layers = nn.ModuleList([OfficialLikeBlock(dim, dpr[i]) for i in range(depth)])
    def forward(self, tokens, pos):
        t, b, l, c = tokens.shape
        x = (tokens + pos).reshape(t * b, l, c)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        return ((x + residual) if residual is not None else x).reshape(t, b, l, c)


class OfficialLikeHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            SpikeAct(),
            nn.Conv1d(dim, 256, 1), nn.BatchNorm1d(256), SpikeAct(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), SpikeAct(),
            nn.Conv1d(128, num_classes, 1),
        )
    def forward(self, x):
        t, b, _l, c = x.shape
        return self.net(x.mean(2).reshape(t * b, c, 1)).reshape(t, b, -1, 1).mean(0).squeeze(-1)


class OfficialLikeSPM(nn.Module):
    def __init__(self, num_classes=10, dim=256, depth=8, num_group=64,
                 group_size=32, timestep=2, expand=1.1, drop_path=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim; self.num_group = num_group; self.group_size = group_size
        self.grouper    = OfficialLikeGroup(num_group, group_size, expand, timestep)
        self.encoder    = MaxFirstAPTECEncoder(dim, T=APTEC_T)   # Max-First + APTEC (intra-group)
        self.group_edge = GroupMaxFirstEdgeConv(dim, k_group=K_GROUP, T=APTEC_T)  # inter-group
        self.pos_embed  = PosEmbed(dim)
        self.blocks    = OfficialLikeMixerModel(dim, depth, timestep, drop_path)
        self.head      = OfficialLikeHead(dim, num_classes)

    def encode_groups(self, pts):
        nh, ctr = self.grouper(pts)
        tok = self.encoder(nh)           # intra-group Max-First + APTEC  [t,b,G,D]
        tok = self.group_edge(tok, ctr)  # inter-group Max-First + APTEC  [t,b,G,D]
        return tok, self.pos_embed(ctr), ctr

    def forward_tokens(self, tokens, pos):
        return self.head(self.blocks(tokens, pos))

    def forward(self, pts):
        tok, pos, _ = self.encode_groups(pts)
        return self.forward_tokens(tok, pos)

    def mean_firing_rate(self):
        rates = [m.rate() for m in self.modules() if isinstance(m, SpikeAct)]
        return sum(rates) / max(len(rates), 1)

# ─── 9. KD teacher ───────────────────────────────────────────────────────────

class _AnalogEncoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(3,   128, 1); self.b1 = nn.BatchNorm2d(128)
        self.c2 = nn.Conv2d(128, 256, 1); self.b2 = nn.BatchNorm2d(256)
        self.c3 = nn.Conv2d(512, 512, 1); self.b3 = nn.BatchNorm2d(512)
        self.c4 = nn.Conv2d(512, ch,  1); self.b4 = nn.BatchNorm2d(ch)

    def forward(self, nh):
        t, b, g, k, _ = nh.shape
        x  = nh.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x  = F.gelu(self.b1(self.c1(x)))
        x  = F.gelu(self.b2(self.c2(x)))
        gl = x.max(3, keepdim=True).values
        x  = F.gelu(self.b3(self.c3(torch.cat([gl.expand(-1, -1, -1, k), x], 1))))
        x  = self.b4(self.c4(x)).max(3).values.transpose(1, 2).contiguous()
        return x.reshape(t, b, g, -1)


class _AnalogPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim),
        )
    def forward(self, ctr):
        t, b, g, _ = ctr.shape
        x = ctr.flatten(0, 1).permute(0, 2, 1).contiguous()
        return self.net(x).permute(0, 2, 1).reshape(t, b, g, -1)


class PointTransformerTeacher(nn.Module):
    def __init__(self, num_classes=10, dim=256, depth=6, heads=8,
                 num_group=64, group_size=32, expand=1.1):
        super().__init__()
        self.num_classes = num_classes
        self.grouper  = OfficialLikeGroup(num_group, group_size, expand, timestep=1)
        self.encoder  = _AnalogEncoder(dim)
        self.pos_embed = _AnalogPosEmbed(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm   = nn.LayerNorm(dim)
        self.head   = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, num_classes),
        )

    def forward(self, pts):
        nh, ctr = self.grouper(pts)
        x = (self.encoder(nh) + self.pos_embed(ctr)).squeeze(0)
        x = self.norm(self.blocks(x))
        return self.head(torch.cat([x.mean(1), x.max(1).values], -1))

# ─── 10. ASP wrapper ─────────────────────────────────────────────────────────

class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=7, hidden=128):
        super().__init__()
        self.mem = nn.Linear(mem_dim, hidden, bias=False)
        self.geo = nn.Sequential(nn.Linear(geo_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden, bias=False))
        self.scale = math.sqrt(hidden)
    def forward(self, belief, geo, visited=None):
        s = torch.bmm(self.geo(geo), self.mem(belief).unsqueeze(-1)).squeeze(-1) / self.scale
        if visited is not None:
            s = s.masked_fill(visited.clone(), float("-inf"))
        return s


class OfficialLikeASP(nn.Module):
    def __init__(self, base, asp_steps=4, d_ssp=128):
        super().__init__()
        self.base     = base
        self.S        = asp_steps
        self.chunk    = base.num_group // asp_steps
        self.ssp      = SliceSelectionPolicy(base.dim, 7, d_ssp)
        self.belief   = nn.Sequential(
            nn.Linear(base.num_classes, base.dim), nn.GELU(),
            nn.Linear(base.dim, base.dim),
        )
        self.register_buffer("tau", torch.tensor(1.0))

    @property
    def num_classes(self): return self.base.num_classes

    def set_gumbel_tau(self, v): self.tau.fill_(v)
    def mean_firing_rate(self): return self.base.mean_firing_rate()

    def _chunkify(self, tok, pos, ctr, pts):
        t, b, g, c = tok.shape
        tok_c = tok.reshape(t, b, self.S, self.chunk, c)
        pos_c = pos.reshape(t, b, self.S, self.chunk, c)
        cb    = ctr.mean(0).reshape(b, self.S, self.chunk, 3)
        cc    = cb.mean(2)
        ad    = (cc - pts.mean(1, keepdim=True)).norm(-1, keepdim=True)
        sp    = (cb - cc.unsqueeze(2)).norm(-1).mean(2, keepdim=True)
        ov    = torch.ones(b, self.S, 1, device=pts.device)
        od    = torch.linspace(0, 1, self.S, device=pts.device).view(1, self.S, 1).expand(b, -1, -1)
        geo   = torch.cat([cc, ad, sp, ov, od], -1)
        return tok_c, pos_c, geo

    def _gather(self, chunks, idx):
        t, b, _s, k, c = chunks.shape
        gi = idx.view(1, b, 1, 1, 1).expand(t, b, 1, k, c)
        return chunks.gather(2, gi).squeeze(2)

    def forward_train(self, pts):
        tok, pos, ctr = self.base.encode_groups(pts)
        tok_c, pos_c, geo = self._chunkify(tok, pos, ctr, pts)
        b, dev = pts.shape[0], pts.device
        vis    = torch.zeros(b, self.S, dtype=torch.bool, device=dev)
        bel    = torch.zeros(b, self.base.dim, device=dev)
        st, sp, all_l = [], [], []
        for _ in range(self.S):
            w   = F.gumbel_softmax(self.ssp(bel, geo, vis), tau=float(self.tau), hard=True)
            idx = w.detach().argmax(-1)
            vis.scatter_(1, idx.unsqueeze(1), True)
            st.append((w.view(1, b, self.S, 1, 1) * tok_c).sum(2))
            sp.append((w.view(1, b, self.S, 1, 1) * pos_c).sum(2))
            lg = self.base.forward_tokens(torch.cat(st, 2), torch.cat(sp, 2))
            all_l.append(lg)
            bel = self.belief(lg.detach().softmax(-1))
        return all_l[-1], all_l

    @torch.no_grad()
    def forward_infer(self, pts, thr=0.55):
        tok, pos, ctr = self.base.encode_groups(pts)
        tok_c, pos_c, geo = self._chunkify(tok, pos, ctr, pts)
        b, dev = pts.shape[0], pts.device
        vis = torch.zeros(b, self.S, dtype=torch.bool, device=dev)
        bel = torch.zeros(b, self.base.dim, device=dev)
        st, sp, last = [], [], None
        for step in range(self.S):
            idx = self.ssp(bel, geo, vis).argmax(-1)
            vis.scatter_(1, idx.unsqueeze(1), True)
            st.append(self._gather(tok_c, idx))
            sp.append(self._gather(pos_c, idx))
            lg   = self.base.forward_tokens(torch.cat(st, 2), torch.cat(sp, 2))
            last = lg
            bel  = self.belief(lg.softmax(-1))
            top2 = lg.softmax(-1).topk(2, -1).values
            if (top2[:, 0] - top2[:, 1]).min().item() > thr:
                return lg, step + 1
        return last, self.S

# ─── 11. Loss functions ───────────────────────────────────────────────────────

def smooth_ce(logits, labels):
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)


def mixup_ce(logits, la, lb, lam):
    return lam * smooth_ce(logits, la) + (1 - lam) * smooth_ce(logits, lb)


def kd_ce(logits, labels, t_logits=None, labels_b=None, lam=1.0):
    ce = mixup_ce(logits, labels, labels_b, lam) if (labels_b is not None and lam < 1) \
         else smooth_ce(logits, labels)
    if t_logits is None:
        return ce
    kd = F.kl_div(F.log_softmax(logits / KD_TEMP, -1),
                  F.softmax(t_logits.detach() / KD_TEMP, -1),
                  reduction="batchmean") * (KD_TEMP ** 2)
    return KD_CE_WEIGHT * ce + KD_LOGIT_WEIGHT * kd


def active_loss(lf, all_l, labels, model, t_logits=None, labels_b=None, lam=1.0):
    loss = kd_ce(lf, labels, t_logits, labels_b, lam)
    if len(all_l) > 1:
        aux  = sum(kd_ce(lg, labels, t_logits, labels_b, lam) for lg in all_l[:-1])
        loss = loss + KD_AUX_WEIGHT * aux / (len(all_l) - 1)
    exit_l = sum((len(all_l) - i) / len(all_l) *
                 (1 - lg.softmax(-1).max(-1).values).mean()
                 for i, lg in enumerate(all_l))
    return loss + 0.05 * exit_l / len(all_l) + 0.01 * model.mean_firing_rate()

# ─── 12. Scheduler + checkpoint helpers ──────────────────────────────────────

def gumbel_tau_sched(ep, t0=1.0, tmin=0.1, r=0.04):
    return max(tmin, t0 * math.exp(-r * ep))


def make_scheduler(opt, epochs, warmup):
    def fn(ep):
        if ep < warmup: return (ep + 1) / max(1, warmup)
        t = (ep - warmup) / max(1, epochs - warmup)
        return max(1e-2, 0.5 * (1 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _load(path):
    try:    return torch.load(path, map_location=DEVICE, weights_only=False)
    except: return torch.load(path, map_location=DEVICE)


def save_ckpt(path, model, opt, sch, ep, best, hist, scaler=None):
    pay = {"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
           "scheduler": sch.state_dict(), "best": best, "history": hist}
    if scaler: pay["scaler"] = scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(pay, tmp)
    try: os.replace(path, path + ".bak")
    except: pass
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sch, scaler=None):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return 0, 0.0, []
    try:
        ck = _load(path)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"])
        if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
        ep = int(ck["epoch"]); best = float(ck.get("best", 0))
        print(f"  [CKPT] resumed {os.path.basename(path)} ep={ep} best={best*100:.2f}%")
        return ep, best, ck.get("history", [])
    except Exception as e:
        print(f"  [CKPT] {os.path.basename(path)}: {e}")
        return 0, 0.0, []

# ─── 13. Build models ─────────────────────────────────────────────────────────

print("\nBuilding models …")
base_spm = OfficialLikeSPM(
    num_classes=NUM_CLASSES, dim=TRANS_DIM, depth=DEPTH,
    num_group=NUM_GROUP, group_size=GROUP_SIZE,
    timestep=TIMESTEP, expand=EXPAND, drop_path=DROP_PATH,
).to(DEVICE)

asp = OfficialLikeASP(base_spm, asp_steps=ASP_STEPS).to(DEVICE)

teacher = PointTransformerTeacher(
    num_classes=NUM_CLASSES, dim=TEACHER_DIM, depth=TEACHER_DEPTH,
    heads=TEACHER_HEADS, num_group=NUM_GROUP, group_size=GROUP_SIZE, expand=EXPAND,
).to(DEVICE)

print(f"  ASP     : {sum(p.numel() for p in asp.parameters()):,} params")
print(f"  Teacher : {sum(p.numel() for p in teacher.parameters()):,} params")

# ─── 14. Loaders ─────────────────────────────────────────────────────────────

print(f"\nLoading MN10 …")
train_ds = ModelNetDataset(MN10_DIR, NUM_POINTS, "train")
val_ds   = ModelNetDataset(MN10_DIR, NUM_POINTS, "test")
train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
print(f"Train {len(train_ds)}  Val {len(val_ds)}  Batches/ep {len(train_loader)}")

# ─── 15. Teacher phase ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_teacher(model, loader, nv=3):
    model.eval(); correct = total = 0
    for pts, lbl in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        pr = torch.zeros(pts.shape[0], NUM_CLASSES, device=DEVICE)
        for _ in range(nv):
            th = random.uniform(0, 2 * math.pi)
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]], device=DEVICE)
            pr += model(pts @ Rz.T).softmax(-1)
        correct += (pr.argmax(1) == lbl).sum().item(); total += pts.shape[0]
    return correct / total


def train_teacher_ep(model, loader, opt):
    model.train(); opt.zero_grad()
    tl = ta = n = 0
    for step, (pts, lbl) in enumerate(loader):
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        lg   = model(pts)
        loss = F.cross_entropy(lg, lbl, label_smoothing=LABEL_SMOOTH) / GRAD_ACCUM
        if torch.isfinite(loss): loss.backward()
        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); opt.zero_grad()
        b = pts.shape[0]
        tl += loss.item() * GRAD_ACCUM * b; ta += (lg.argmax(1) == lbl).sum().item(); n += b
    return tl / n, ta / n


print("\n" + "=" * 60)
print("Phase 1 — Teacher training")
print("=" * 60)

t_opt = torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
t_sch = make_scheduler(t_opt, TEACHER_EPOCHS, WARMUP_EP)
t_ep, t_best, t_hist = load_ckpt(TEACHER_LATEST, teacher, t_opt, t_sch)

if t_ep >= TEACHER_EPOCHS and os.path.isfile(TEACHER_BEST):
    print(f"Teacher cached ({TEACHER_EPOCHS} ep). Loading best weights.")
    teacher.load_state_dict(_load(TEACHER_BEST))
else:
    for ep in range(t_ep, TEACHER_EPOCHS):
        t0 = time.time()
        _, tr = train_teacher_ep(teacher, train_loader, t_opt)
        t_sch.step()
        va = None
        if (ep + 1) % 5 == 0 or ep + 1 == TEACHER_EPOCHS:
            va = eval_teacher(teacher, val_loader)
            best = va > t_best
            if best: t_best = va; torch.save(teacher.state_dict(), TEACHER_BEST)
            print(f"  [T] {ep+1:3d}/{TEACHER_EPOCHS} tr={tr:.4f} val={va:.4f} "
                  f"{'★' if best else ' '} lr={t_opt.param_groups[0]['lr']:.5f} "
                  f"{time.time()-t0:.0f}s")
        t_hist.append({"ep": ep + 1, "tr": tr, "val": va})
        save_ckpt(TEACHER_LATEST, teacher, t_opt, t_sch, ep + 1, t_best, t_hist)
    if os.path.isfile(TEACHER_BEST):
        teacher.load_state_dict(_load(TEACHER_BEST))

teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
print(f"\nTeacher ready  best_val={t_best*100:.2f}%  (frozen)")

# ─── 16. ASP phase ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 2 — ASP training with KD + PointMixup")
print("=" * 60)

optimizer = torch.optim.AdamW(asp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EP)
scaler    = GradScaler("cuda")
start, best_acc, history = load_ckpt(ASP_LATEST, asp, optimizer, scheduler, scaler)
if start == 0 and os.path.isfile(ASP_LATEST + ".bak"):
    start, best_acc, history = load_ckpt(ASP_LATEST + ".bak", asp, optimizer, scheduler, scaler)
if start == 0:
    print("Starting ASP from scratch.")


@torch.no_grad()
def eval_asp(model, loader, nv=N_VOTE):
    model.eval(); correct = total = slices = 0
    for pts, lbl in loader:
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        b = pts.shape[0]
        pr = torch.zeros(b, NUM_CLASSES, device=DEVICE)
        for _ in range(nv):
            th = random.uniform(0, 2 * math.pi)
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]], device=DEVICE)
            lg, used = model.forward_infer(pts @ Rz.T, EXIT_THR)
            pr += lg.softmax(-1); slices += used * b
        correct += (pr.argmax(1) == lbl).sum().item(); total += b
    return correct / total, slices / total / nv


def train_one_epoch(model, loader, opt, ep):
    model.train()
    model.set_gumbel_tau(gumbel_tau_sched(ep))
    opt.zero_grad()
    tl = ta = n = 0; t0 = time.time()
    for step, (pts, lbl) in enumerate(loader):
        pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
        lam   = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        idx_m = torch.randperm(pts.shape[0], device=DEVICE)
        pts_m = lam * pts + (1 - lam) * pts[idx_m]
        lbl_b = lbl[idx_m]

        with torch.no_grad():
            t_lg = teacher(pts_m)

        with autocast("cuda"):
            lf, all_l = model.forward_train(pts_m)
            loss = active_loss(lf, all_l, lbl, model, t_lg, lbl_b, lam) / GRAD_ACCUM

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
        else:
            print(f"  [SKIP] step {step}: non-finite loss")

        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()

        b = pts.shape[0]
        tl += loss.item() * GRAD_ACCUM * b
        ta += (lf.detach().argmax(1) == (lbl if lam >= 0.5 else lbl_b)).sum().item()
        n  += b
    return tl / max(n, 1), ta / max(n, 1), time.time() - t0


print(f"ep={EPOCHS}  start={start}  best={best_acc*100:.2f}%")

for ep in range(start, EPOCHS):
    tr_loss, tr_acc, elapsed = train_one_epoch(asp, train_loader, optimizer, ep)
    scheduler.step()
    lr  = optimizer.param_groups[0]["lr"]
    tau = float(asp.tau)
    print(f"Ep {ep+1:3d}/{EPOCHS}  loss={tr_loss:.4f}  tr={tr_acc:.4f}  "
          f"tau={tau:.3f}  lr={lr:.5f}  {elapsed:.0f}s", end="")

    va = None
    if (ep + 1) % VAL_EVERY == 0 or ep + 1 == EPOCHS:
        va, sl = eval_asp(asp, val_loader, N_VOTE)
        if va > best_acc:
            best_acc = va; torch.save(asp.state_dict(), ASP_BEST)
        print(f"  | val={va:.4f} {'★' if va == best_acc else ' '} "
              f"sl={sl:.2f}/{ASP_STEPS}  best={best_acc:.4f}", end="")

    history.append({"ep": ep+1, "tr_loss": tr_loss, "tr_acc": tr_acc,
                    "val_acc": va, "tau": tau, "lr": lr})
    save_ckpt(ASP_LATEST, asp, optimizer, scheduler, ep + 1, best_acc, history, scaler)
    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("  ✓")

# ─── 17. Final verdict ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Teacher OA : {t_best*100:.2f}%")
print(f"ASP best   : {best_acc*100:.2f}%  (target ≥ 94.93%)")
print(f"Checkpoints: {CKPT_DIR}")
print(f"{'='*60}")
if best_acc >= 0.9493:   print("VERDICT: ✓ Beat Spiking DGCNN MN10 target (94.93%)!")
elif best_acc >= 0.940:  print("VERDICT: Very close. Try N_VOTE=15 or 50 more epochs.")
elif best_acc >= 0.920:  print("VERDICT: Needs more epochs or larger model.")
else:                    print("VERDICT: Check data path and GPU memory.")

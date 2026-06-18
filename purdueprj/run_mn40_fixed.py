"""
run_mn40_fixed.py
=================
Self-contained Colab script — paste as ONE cell or run with:
    !python run_mn40_fixed.py

Bugs fixed vs previous run:
  1. active_snn.py  : removed spurious double lif1 call inside
                      forward_active_train (was firing lif1 twice
                      per timestep → membrane explosion by epoch 23)
  2. train_active.py: torch.isfinite(loss) guard skips backward on
                      any remaining non-finite batches
  3. active_snn.py  : visited_mask uses .clone() to avoid in-place
                      mutation on computation graph tensor
"""

# ── 0. Mount Google Drive (must happen before any checkpoint I/O) ─────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("[Drive] Mounted at /content/drive")
except Exception as e:
    print(f"[Drive] Could not mount ({e}) — checkpoints will be LOCAL only")

# ── 0. Imports & deps ────────────────────────────────────────────────────────
import os, sys, subprocess, shutil, math, time, json, random, warnings
warnings.filterwarnings("ignore")

for pkg in ["trimesh", "kagglehub"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ── 1. Write all source modules to /content/asp ──────────────────────────────
ASP = "/content/asp"
for d in ["data", "models", "training", "inference"]:
    os.makedirs(f"{ASP}/{d}", exist_ok=True)
    open(f"{ASP}/{d}/__init__.py", "w").close()

# --------------------------------------------------------------------------- #
# data/modelnet.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/data/modelnet.py", "w").write(
r"""
import os, torch, numpy as np
from torch.utils.data import Dataset
import trimesh

class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.root       = root
        self.num_points = num_points
        self.split      = split
        self.files      = self._scan_files()
        self.data, self.labels = self._load_all()

    def _scan_files(self):
        items = []
        for cls in sorted(os.listdir(self.root)):
            cp = os.path.join(self.root, cls, self.split)
            if not os.path.isdir(cp):
                continue
            label = sorted(os.listdir(self.root)).index(cls)
            for f in os.listdir(cp):
                if f.endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(cp, f), label))
        return items

    def _load_points(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        if path.endswith(".txt"):
            return np.loadtxt(path).astype(np.float32)
        mesh = trimesh.load(path)
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_labels = [], []
        for path, label in self.files:
            pts = self._load_points(path)
            if not path.endswith(".off"):
                if pts.shape[0] >= self.num_points:
                    idx = np.random.choice(pts.shape[0], self.num_points, replace=False)
                    pts = pts[idx]
                else:
                    pad = np.random.choice(pts.shape[0], self.num_points - pts.shape[0], replace=True)
                    pts = np.vstack([pts, pts[pad]])
            all_pts.append(pts)
            all_labels.append(label)
        return np.array(all_pts), np.array(all_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))
""")

# --------------------------------------------------------------------------- #
# data/slicing.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/data/slicing.py", "w").write(
r"""
import math, torch

def farthest_point_sample(pts, n_samples):
    N = pts.shape[0]
    n_samples = min(n_samples, N)
    device    = pts.device
    selected  = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest  = torch.randint(0, N, (1,), device=device).item()
    for i in range(n_samples):
        selected[i] = farthest
        dist        = ((pts - pts[farthest]) ** 2).sum(-1)
        distances   = torch.minimum(distances, dist)
        farthest    = distances.argmax().item()
    return selected

def slice_fps_hierarchical_batch(points, T=16):
    B, N, _ = points.shape
    pps    = N // T
    device = points.device
    all_slices = []
    for b in range(B):
        pts_b   = points[b]
        fps_idx = farthest_point_sample(pts_b, T)
        centres = pts_b[fps_idx]
        diff    = pts_b.unsqueeze(0) - centres.unsqueeze(1)   # [T, N, 3]
        assign  = (diff ** 2).sum(-1).argmin(dim=0)           # [N]
        slices_b = []
        for t in range(T):
            mask = (assign == t).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                mask = torch.randperm(N, device=device)[:pps]
            elif mask.numel() < pps:
                mask = mask.repeat(math.ceil(pps / mask.numel()))[:pps]
            else:
                mask = mask[:pps]
            slices_b.append(pts_b[mask])
        all_slices.append(torch.stack(slices_b))
    return torch.stack(all_slices)   # [B, T, pps, 3]
""")

# --------------------------------------------------------------------------- #
# models/snn_layers.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/models/snn_layers.py", "w").write(
r"""
import math, torch, torch.nn as nn

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + torch.abs(x)) ** 2

spike_fn = SurrogateSpike.apply


class LearnableLIFLayer(nn.Module):
    def __init__(self, in_features, out_features, tau_init=0.9, vth_init=1.0):
        super().__init__()
        self.fc          = nn.Linear(in_features, out_features)
        self.out_features = out_features
        tau_raw = math.log(tau_init / (1.0 - tau_init))
        self.tau_raw = nn.Parameter(torch.full((out_features,), tau_raw))
        self.vth_raw = nn.Parameter(torch.full((out_features,), float(vth_init)))
        self.register_buffer("mem",         None)
        self.register_buffer("spike_count", torch.tensor(0.0))
        self.register_buffer("step_count",  torch.tensor(0))

    @property
    def tau(self): return torch.sigmoid(self.tau_raw)
    @property
    def vth(self): return torch.nn.functional.softplus(self.vth_raw)

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.mem         = torch.zeros(batch_size, self.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        denom = self.out_features * self.step_count * getattr(self, "batch_size", 1)
        return (self.spike_count / denom).item()

    def forward(self, x):
        cur      = self.fc(x)
        self.mem = self.tau * self.mem + cur
        spk      = spike_fn(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count  + 1
        return spk, self.mem


# Alias kept for any code that imports LIFLayer
class LIFLayer(nn.Module):
    def __init__(self, in_features, out_features, tau=0.9):
        super().__init__()
        self.fc  = nn.Linear(in_features, out_features)
        self.tau = tau
        self.register_buffer("mem", None)

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.mem         = torch.zeros(batch_size, self.fc.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        denom = self.fc.out_features * self.step_count * getattr(self, "batch_size", 1)
        return (self.spike_count / denom).item()

    def forward(self, x):
        cur      = self.fc(x)
        self.mem = self.tau * self.mem + cur
        spk      = spike_fn(self.mem - 1.0)
        self.mem = self.mem * (1 - spk)
        if not hasattr(self, "spike_count"):
            self.spike_count = torch.tensor(0.0, device=cur.device)
            self.step_count  = torch.tensor(0,   device=cur.device)
            self.batch_size  = x.shape[0]
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count  + 1
        return spk, self.mem
""")

# --------------------------------------------------------------------------- #
# models/pointnet_backbone.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/models/pointnet_backbone.py", "w").write(
r"""
import torch, torch.nn as nn
from models.snn_layers import LIFLayer, LearnableLIFLayer


class PointNetBackbone(nn.Module):
    # PointNet-style shared-MLP backbone with LIF neurons + global max-pool.
    # Inspired by Spiking PointNet (ICCV 2023, 88.6% MN40):
    #   - Shared MLP applied pointwise (permutation invariant)
    #   - Global max-pool aggregates N points -> single descriptor
    #   - Max-pool >> mean-pool for shape classification (PointNet ablation)
    # Input:  [B, N, 3]
    # Output: [B, out_dim]  global descriptor
    def __init__(self, hidden_dims=[64, 128, 512], learnable_lif=True):
        super().__init__()
        Cls = LearnableLIFLayer if learnable_lif else LIFLayer
        self.out_dim = hidden_dims[-1]
        in_dim = 3
        self.layers = nn.ModuleList()
        for h in hidden_dims:
            self.layers.append(Cls(in_dim, h))
            in_dim = h

    def reset_state(self, batch_size, device=None):
        for layer in self.layers:
            layer.reset_state(batch_size, device)

    def forward(self, pts):
        # pts: [B, N, 3]  ->  [B, out_dim] via shared MLP + max-pool
        B, N, _ = pts.shape
        x = pts.reshape(B * N, 3)
        self.reset_state(B * N, pts.device)
        for layer in self.layers:
            _, x = layer(x)                  # LearnableLIF -> (spike, mem); use mem
        x = x.reshape(B, N, self.out_dim)
        return x.max(dim=1).values           # global max-pool -> [B, out_dim]

    def firing_rates(self):
        return {f"pn_layer_{i}": l.firing_rate()
                for i, l in enumerate(self.layers)
                if hasattr(l, "firing_rate")}
""")

# --------------------------------------------------------------------------- #
# models/temporal_snn.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/models/temporal_snn.py", "w").write(
r"""
import torch.nn as nn
from models.snn_layers import LIFLayer, LearnableLIFLayer

class TemporalSNN(nn.Module):
    def __init__(self, dim=256, num_classes=10, learnable_lif=False):
        super().__init__()
        Cls       = LearnableLIFLayer if learnable_lif else LIFLayer
        self.lif1 = Cls(dim, dim)
        self.lif2 = Cls(dim, dim)
        self.fc   = nn.Linear(dim, num_classes)

    def reset_state(self, batch_size, device=None):
        self.lif1.reset_state(batch_size, device)
        self.lif2.reset_state(batch_size, device)

    def forward(self, x):
        spk1, mem1 = self.lif1(x)
        spk2, mem2 = self.lif2(mem1 + x)   # residual skip: improves gradient flow
        return self.fc(mem2)

    def firing_rates(self):
        return {n: l.firing_rate()
                for n, l in [("temporal_lif1", self.lif1), ("temporal_lif2", self.lif2)]
                if hasattr(l, "firing_rate")}
""")

# --------------------------------------------------------------------------- #
# models/slice_selection_policy.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/models/slice_selection_policy.py", "w").write(
r"""
import math, torch, torch.nn as nn, torch.nn.functional as F

GEO_DIM = 6

def compute_geometry_descriptors(pts, fps_anchors, anchor_assignments):
    B, N, _ = pts.shape
    M       = fps_anchors.size(1)
    centroid    = pts.mean(dim=1, keepdim=True)
    anchor_dist = (fps_anchors - centroid).norm(dim=-1)          # [B, M]
    mean_intra  = torch.zeros(B, M, device=pts.device)
    point_count = torch.zeros(B, M, device=pts.device)
    for m in range(M):
        mask = (anchor_assignments == m).float()
        point_count[:, m] = mask.sum(dim=1)
        for b in range(B):
            cp = pts[b][anchor_assignments[b] == m]
            if cp.size(0) > 1:
                mean_intra[b, m] = (cp - cp.mean(0)).norm(dim=-1).mean()
    norm_count = point_count / (N / M + 1e-6)
    return torch.cat([
        fps_anchors,
        anchor_dist.unsqueeze(-1),
        mean_intra.unsqueeze(-1),
        norm_count.unsqueeze(-1),
    ], dim=-1)                                                     # [B, M, 6]


class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=GEO_DIM, d_ssp=64):
        super().__init__()
        self.d_ssp = d_ssp
        self.scale = math.sqrt(d_ssp)
        self.W_k   = nn.Linear(mem_dim, d_ssp, bias=False)
        self.W_q   = nn.Linear(geo_dim,  d_ssp, bias=False)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)

    def forward(self, mem, geo, visited_mask=None):
        key    = self.W_k(mem)                                     # [B, d]
        query  = self.W_q(geo)                                     # [B, M, d]
        scores = torch.bmm(query, key.unsqueeze(-1)).squeeze(-1) / self.scale
        if visited_mask is not None:
            scores = scores.masked_fill(visited_mask, float("-inf"))
        return scores

    def select_gumbel(self, scores, tau=1.0):
        return F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)

    def select_greedy(self, scores):
        return F.one_hot(scores.argmax(dim=-1), num_classes=scores.size(-1)).float()
""")

# --------------------------------------------------------------------------- #
# models/active_snn.py  ← BUG FIX 1 + 3
# --------------------------------------------------------------------------- #
open(f"{ASP}/models/active_snn.py", "w").write(
r"""
import torch, torch.nn as nn, torch.nn.functional as F
from models.pointnet_backbone    import PointNetBackbone
from models.temporal_snn         import TemporalSNN
from models.slice_selection_policy import SliceSelectionPolicy, compute_geometry_descriptors


class ActiveSNN(nn.Module):
    # Architecture changes vs v1 (based on literature survey):
    #   1. PointNetBackbone (SharedMLP + MaxPool) replaces LocalKNNBackbone
    #      -> Spiking PointNet (ICCV 2023) showed this gets 88.6% vs our 70%
    #   2. Global context feature: backbone(full_cloud) added to each slice feat
    #      -> P2SResLNet (AAAI 2024) showed global spatial context is critical
    #   3. Residual skip in TemporalSNN (see temporal_snn.py)
    def __init__(self, point_dims=[64, 128, 512], temporal_dim=512,
                 num_classes=10, d_ssp=64):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.num_classes  = num_classes
        self.backbone = PointNetBackbone(hidden_dims=point_dims, learnable_lif=True)
        assert self.backbone.out_dim == temporal_dim, (
            f"backbone out_dim {self.backbone.out_dim} must match temporal_dim {temporal_dim}")
        self.temporal = TemporalSNN(dim=temporal_dim, num_classes=num_classes, learnable_lif=True)
        self.ssp      = SliceSelectionPolicy(mem_dim=temporal_dim, d_ssp=d_ssp)
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    def reset_state(self, batch_size, device=None):
        self.backbone.reset_state(batch_size, device)
        self.temporal.reset_state(batch_size, device)

    def _get_membrane(self):
        lif = self.temporal.lif2
        return lif.mem.detach() if lif.mem is not None else None

    def _global_feat(self, pts_slices, device):
        # Reconstruct full point cloud from all slices and extract global descriptor
        B, T, n_pts, C = pts_slices.shape
        all_pts = pts_slices.reshape(B, T * n_pts, C)   # [B, N_total, 3]
        return self.backbone(all_pts)                    # [B, temporal_dim]

    # ------------------------------------------------------------------ #
    # Training forward                                                    #
    # ------------------------------------------------------------------ #

    def forward_active_train(self, pts_slices, geo_descriptors):
        B, T, n_pts, _ = pts_slices.shape
        device = pts_slices.device

        # Global context: full reconstructed cloud -> [B, D]
        global_feat = self._global_feat(pts_slices, device)

        # Precompute per-slice backbone features [B, T, D]
        pts_flat  = pts_slices.reshape(B * T, n_pts, 3)
        all_feats = self.backbone(pts_flat).reshape(B, T, -1)

        self.temporal.reset_state(B, device)
        visited_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mem_state    = torch.zeros(B, self.temporal_dim, device=device)
        logits_all   = []

        for t in range(T):
            scores = self.ssp(mem_state, geo_descriptors, visited_mask)
            w      = self.ssp.select_gumbel(scores, tau=self.gumbel_tau.item())

            selected_idx = scores.masked_fill(visited_mask, float("-inf")).argmax(dim=-1)
            new_mask = visited_mask.clone()
            for b in range(B):
                new_mask[b, selected_idx[b]] = True
            visited_mask = new_mask

            e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)   # [B, D]
            e_t = e_t + global_feat                           # add global context
            logits_t = self.temporal(e_t)
            logits_all.append(logits_t)

            mem_state = self._get_membrane()
            if mem_state is None:
                mem_state = torch.zeros(B, self.temporal_dim, device=device)

        return logits_all[-1], logits_all

    # ------------------------------------------------------------------ #
    # Inference forward                                                   #
    # ------------------------------------------------------------------ #

    def forward_active_infer(self, pts_slices, geo_descriptors, threshold=0.7):
        B, T, n_pts, _ = pts_slices.shape
        device = pts_slices.device

        global_feat = self._global_feat(pts_slices, device)

        self.temporal.reset_state(B, device)
        visited_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mem_state    = torch.zeros(B, self.temporal_dim, device=device)
        slice_order  = []
        last_logits  = None

        for t in range(T):
            with torch.no_grad():
                scores = self.ssp(mem_state, geo_descriptors, visited_mask)
                w      = self.ssp.select_greedy(scores)

            selected_idx = w.argmax(dim=-1)
            chosen = selected_idx[0].item()
            slice_order.append(chosen)

            new_mask = visited_mask.clone()
            for b in range(B):
                new_mask[b, selected_idx[b]] = True
            visited_mask = new_mask

            with torch.no_grad():
                e_t      = self.backbone(pts_slices[:, chosen, :, :])   # [B, D]
                e_t      = e_t + global_feat
                logits_t = self.temporal(e_t)
                last_logits = logits_t

            mem_state = self._get_membrane()
            if mem_state is None:
                mem_state = torch.zeros(B, self.temporal_dim, device=device)

            probs  = F.softmax(logits_t, dim=-1)
            top2   = probs.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            if margin.min().item() > threshold:
                return last_logits, t + 1, slice_order

        return last_logits, T, slice_order

    # ------------------------------------------------------------------ #

    def get_firing_rates(self):
        r = {}
        if hasattr(self.backbone, "firing_rates"): r.update(self.backbone.firing_rates())
        if hasattr(self.temporal, "firing_rates"): r.update(self.temporal.firing_rates())
        return r

    def mean_firing_rate(self):
        r = self.get_firing_rates()
        return sum(r.values()) / len(r) if r else 0.0

    def set_gumbel_tau(self, tau):
        self.gumbel_tau.fill_(tau)

    def param_count(self):
        bb   = sum(p.numel() for p in self.backbone.parameters())
        temp = sum(p.numel() for p in self.temporal.parameters())
        ssp  = sum(p.numel() for p in self.ssp.parameters())
        return {"backbone": bb, "temporal": temp, "ssp": ssp, "total": bb + temp + ssp}
""")

# --------------------------------------------------------------------------- #
# training/loss_active.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/training/loss_active.py", "w").write(
r"""
import torch, torch.nn.functional as F

def active_loss(logits_final, logits_all, labels, model,
                lam_aux=0.3, lam_exit=0.1, lam_fr=0.05):
    T      = len(logits_all)
    device = logits_final.device

    l_ce = F.cross_entropy(logits_final, labels)

    if T > 1:
        l_aux = lam_aux * sum(
            F.cross_entropy(logits_all[t], labels) for t in range(T - 1)
        ) / (T - 1)
    else:
        l_aux = torch.tensor(0.0, device=device)

    total_exit = torch.tensor(0.0, device=device)
    for t, lg in enumerate(logits_all):
        max_p = F.softmax(lg, dim=-1).max(dim=-1).values
        total_exit = total_exit + ((T - t) / T) * (1.0 - max_p).mean()
    l_exit = lam_exit * total_exit / T

    if hasattr(model, "mean_firing_rate"):
        r = model.mean_firing_rate()
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32, device=device)
        l_fr = lam_fr * r.to(device)
    else:
        l_fr = torch.tensor(0.0, device=device)

    total = l_ce + l_aux + l_exit + l_fr
    return total, dict(
        loss_ce=l_ce.item(), loss_aux=l_aux.item(),
        loss_exit=l_exit.item(),
        loss_fr=l_fr.item() if isinstance(l_fr, torch.Tensor) else float(l_fr),
        loss_total=total.item(),
    )
""")

# --------------------------------------------------------------------------- #
# training/metrics.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/training/metrics.py", "w").write(
r"""
def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()
""")

# --------------------------------------------------------------------------- #
# training/optimizers.py
# --------------------------------------------------------------------------- #
open(f"{ASP}/training/optimizers.py", "w").write(
r"""
import torch
def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
""")

# --------------------------------------------------------------------------- #
# training/train_active.py  ← BUG FIX 2
# --------------------------------------------------------------------------- #
open(f"{ASP}/training/train_active.py", "w").write(
r"""
import torch, time, math
from data.slicing import slice_fps_hierarchical_batch
from training.loss_active import active_loss
from training.metrics import accuracy
from models.slice_selection_policy import compute_geometry_descriptors


def gumbel_tau(epoch, tau_0=1.0, tau_min=0.1, anneal_rate=0.05):
    return max(tau_min, tau_0 * math.exp(-anneal_rate * epoch))


def prepare_fps_slices_and_geo(pts, T):
    pts_slices  = slice_fps_hierarchical_batch(pts, T=T)          # [B,T,pps,3]
    fps_anchors = pts_slices.mean(dim=2)                          # [B,T,3]
    diffs       = pts.unsqueeze(2) - fps_anchors.unsqueeze(1)     # [B,N,T,3]
    assignments = (diffs ** 2).sum(-1).argmin(dim=-1)             # [B,N]
    geo = compute_geometry_descriptors(pts, fps_anchors, assignments)
    return pts_slices, geo, fps_anchors, assignments


def train_active_epoch(model, dataloader, optimizer, device, epoch,
                       num_slices=16, lam_aux=0.3, lam_exit=0.1, lam_fr=0.05,
                       tau_0=1.0, tau_min=0.1, anneal_rate=0.05, verbose_every=20):
    model.train()
    tau = gumbel_tau(epoch, tau_0, tau_min, anneal_rate)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)

    total_ce = total_aux = total_exit = total_fr = total_acc = total_ent = 0.0
    count = 0
    start = time.time()

    for batch_idx, (pts, labels) in enumerate(dataloader):
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)

        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=num_slices)
        logits_final, logits_all = model.forward_active_train(pts_slices, geo)
        loss, bd = active_loss(logits_final, logits_all, labels, model,
                               lam_aux=lam_aux, lam_exit=lam_exit, lam_fr=lam_fr)

        # FIX 2: skip backward on any non-finite loss; don't corrupt weights
        if not torch.isfinite(loss):
            print(f"  [SKIP] batch {batch_idx}: non-finite loss={loss.item():.2e}")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            # Use actual last membrane state, not zeros, so entropy is meaningful
            mem_last = model._get_membrane()
            if mem_last is None:
                mem_last = torch.zeros(B, model.temporal_dim, device=device)
            scores = model.ssp(mem_last, geo, visited_mask=None)
            probs  = torch.softmax(scores, dim=-1)
            ent    = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()

        total_ce   += bd["loss_ce"]
        total_aux  += bd["loss_aux"]
        total_exit += bd["loss_exit"]
        total_fr   += bd["loss_fr"]
        total_acc  += accuracy(logits_final, labels)
        total_ent  += ent.item()
        count      += 1

        if (batch_idx + 1) % verbose_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{batch_idx+1}/{len(dataloader)}] "
                f"CE={bd['loss_ce']:.4f}  Aux={bd['loss_aux']:.4f}  "
                f"Exit={bd['loss_exit']:.4f}  FR={bd['loss_fr']:.4f}  "
                f"Acc={total_acc/count:.3f}  Ent={total_ent/count:.3f}  "
                f"tau={tau:.3f}  LR={lr:.6f}  {time.time()-start:.0f}s"
            )

    n = max(count, 1)
    return dict(loss_ce=total_ce/n, loss_aux=total_aux/n, loss_exit=total_exit/n,
                loss_fr=total_fr/n, acc_final=total_acc/n,
                policy_entropy=total_ent/n, gumbel_tau=tau)


def validate_active(model, dataloader, device, num_slices=16, threshold=0.7):
    model.eval()
    correct = total = 0
    total_exit = total_fr = count = 0.0

    with torch.no_grad():
        for pts, labels in dataloader:
            pts, labels = pts.to(device), labels.to(device)
            B = pts.size(0)
            pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=num_slices)
            for b in range(B):
                logits, exit_step, _ = model.forward_active_infer(
                    pts_slices[b].unsqueeze(0),
                    geo[b].unsqueeze(0),
                    threshold=threshold,
                )
                correct    += (logits.argmax(-1) == labels[b]).sum().item()
                total      += 1
                total_exit += exit_step
            fr = model.mean_firing_rate()
            total_fr += fr if isinstance(fr, float) else fr.item()
            count    += 1

    n         = max(total, 1)
    mean_exit = total_exit / n
    mean_fr   = total_fr   / max(count, 1)
    energy    = mean_fr * 0.274 * (mean_exit / num_slices)
    return dict(acc=correct/n, mean_exit=mean_exit, mean_fr=mean_fr,
                energy_ratio=energy, savings=1.0 / max(energy, 1e-9))
""")

# Minimal inference stub (not used during training)
open(f"{ASP}/inference/active_inference.py", "w").write("# inference utilities\n")

print("[OK] All source files written to", ASP)

# ── 2. Download ModelNet40 ────────────────────────────────────────────────────
sys.path.insert(0, ASP)

MN40_DIR = "/content/ModelNet40"
if not os.path.isdir(MN40_DIR):
    print("Downloading ModelNet40 via kagglehub (~1.9 GB)...")
    import kagglehub
    p = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
    for root, dirs, _ in os.walk(p):
        if "ModelNet40" in dirs:
            shutil.copytree(os.path.join(root, "ModelNet40"), MN40_DIR)
            print(f"  Copied: {os.path.join(root,'ModelNet40')} -> {MN40_DIR}")
            break
    print("Done.")
else:
    print("ModelNet40 already present.")

classes = sorted(os.listdir(MN40_DIR))
print(f"Classes ({len(classes)}):", classes[:6], "...")

# ── 3. Config ─────────────────────────────────────────────────────────────────
EPOCHS      = 150
BATCH_SIZE  = 32
LR          = 1e-3
NUM_SLICES  = 8     # T=8: 128 pts/slice vs old T=16 64 pts/slice; more context per step
NUM_POINTS  = 1024
NUM_CLASSES = 40
VAL_EVERY   = 5
VERBOSE     = 20

print(f"\nConfig: epochs={EPOCHS} bs={BATCH_SIZE} lr={LR} T={NUM_SLICES}")

# ── 4. Datasets & dataloaders ────────────────────────────────────────────────
from torch.utils.data import DataLoader
from data.modelnet import ModelNetDataset

import trimesh  # suppress trimesh warnings once
train_ds = ModelNetDataset(MN40_DIR, num_points=NUM_POINTS, split="train")
val_ds   = ModelNetDataset(MN40_DIR, num_points=NUM_POINTS, split="test")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
print(f"MN40 — Train: {len(train_ds)}  Val: {len(val_ds)}  "
      f"Batches/epoch: {len(train_loader)}")

# ── 5. Model, optimizer, scheduler ───────────────────────────────────────────
from models.active_snn     import ActiveSNN
from training.optimizers   import build_optimizer
from training.train_active import train_active_epoch, validate_active

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = ActiveSNN(point_dims=[64, 128, 512], temporal_dim=512,
                   num_classes=NUM_CLASSES, d_ssp=64).to(device)

pc = model.param_count()
print(f"Params — backbone: {pc['backbone']:,}  total: {pc['total']:,}")

optimizer = build_optimizer(model, lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── 6. Checkpoint — resume from Drive if available ───────────────────────────
DRIVE_DIR  = "/content/drive/MyDrive/asp_mn40_v2_checkpoints"  # v2: PointNetBackbone + global feat + T=8
RESULTS    = "/content/asp_results_mn40"
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(RESULTS,   exist_ok=True)

drive_ckpt   = os.path.join(DRIVE_DIR, "epoch_latest.pth")
best_ckpt    = os.path.join(DRIVE_DIR, "best_model_mn40.pth")
history_path = os.path.join(DRIVE_DIR, "history_mn40.json")

start_epoch  = 0
best_val_acc = 0.0
history      = []

# Find best available checkpoint: prefer epoch_latest, fall back to highest epoch_NNN
_resume_path = None
if os.path.exists(drive_ckpt):
    _resume_path = drive_ckpt
else:
    # Scan for per-epoch checkpoints and pick the latest
    import glob as _glob
    _per_epoch = sorted(_glob.glob(os.path.join(DRIVE_DIR, "epoch_[0-9][0-9][0-9].pth")))
    if _per_epoch:
        _resume_path = _per_epoch[-1]   # highest-numbered = most recent

if _resume_path:
    ckpt = torch.load(_resume_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    for _ in range(start_epoch):      # warm scheduler to correct step
        scheduler.step()
    print(f"Resumed from {os.path.basename(_resume_path)} — "
          f"epoch {start_epoch}, best_val={best_val_acc:.4f}")
else:
    print("No Drive checkpoint found — starting from scratch.")

# ── 7. Training loop ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Starting MN40 training — {EPOCHS} epochs (from epoch {start_epoch})")
print("=" * 60)

for epoch in range(start_epoch, EPOCHS):
    print(f"\n--- Epoch {epoch}/{EPOCHS - 1} ---")

    train_m = train_active_epoch(
        model, train_loader, optimizer, device,
        epoch=epoch, num_slices=NUM_SLICES,
        lam_aux=0.3, lam_exit=0.1, lam_fr=0.05,
        tau_0=1.0, tau_min=0.1, anneal_rate=0.05,
        verbose_every=VERBOSE,
    )
    scheduler.step()

    val_m = {}
    if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
        val_m    = validate_active(model, val_loader, device,
                                   num_slices=NUM_SLICES, threshold=0.7)
        val_acc  = val_m["acc"]
        print(
            f"[Val] Acc={val_acc:.4f}  "
            f"MeanExit={val_m['mean_exit']:.2f}/{NUM_SLICES}  "
            f"FR={val_m['mean_fr']:.3f}  "
            f"EnergyRatio={val_m['energy_ratio']:.4f}  "
            f"Savings={val_m['savings']:.1f}x"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, best_ckpt)
            print(f"  *** New best: {val_acc:.4f} -> {best_ckpt}")

    history.append({"epoch": epoch, **train_m, **val_m})
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    ckpt_data = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc":    best_val_acc,
    }
    # Always overwrite the latest (for fast resume)
    torch.save(ckpt_data, drive_ckpt)
    # Also save a permanent per-epoch copy (never overwritten)
    epoch_ckpt = os.path.join(DRIVE_DIR, f"epoch_{epoch:03d}.pth")
    torch.save(ckpt_data, epoch_ckpt)
    print(f"  [Drive] epoch_latest.pth + epoch_{epoch:03d}.pth saved")

print(f"\nDone. Best val acc: {best_val_acc:.4f}")

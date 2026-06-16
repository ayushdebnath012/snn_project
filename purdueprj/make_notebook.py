"""
make_notebook.py
================
Run this script once (locally or on Colab) to generate
ASP_Colab.ipynb -- a fully self-contained Colab notebook
for Active Spiking Perception.

    python make_notebook.py
"""

import json, os, textwrap

# ---------------------------------------------------------------------------
# All source file contents (keyed by relative path)
# ---------------------------------------------------------------------------

FILES = {}

# ── models/__init__.py ──────────────────────────────────────────────────────
FILES["models/__init__.py"] = ""

# ── models/snn_layers.py ───────────────────────────────────────────────────
FILES["models/snn_layers.py"] = textwrap.dedent("""\
import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = 1.0 / (1 + torch.abs(x))**2
        return grad_output * grad


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    def __init__(self, in_features, out_features, tau=0.9):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.register_buffer("mem", None)

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem = torch.zeros(batch_size, self.fc.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        return (self.spike_count / (self.fc.out_features * self.step_count *
                getattr(self, "batch_size", 1))).item()

    def forward(self, x):
        cur = self.fc(x)
        self.mem = self.tau * self.mem + cur
        spk = spike_fn(self.mem - 1.0)
        self.mem = self.mem * (1 - spk)
        if not hasattr(self, "spike_count"):
            self.spike_count = torch.tensor(0.0, device=cur.device)
            self.step_count  = torch.tensor(0,   device=cur.device)
            self.batch_size  = x.shape[0]
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count + 1
        return spk, self.mem


class LearnableLIFLayer(nn.Module):
    def __init__(self, in_features, out_features, tau_init=0.9, vth_init=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.out_features = out_features
        import math
        tau_raw_init = math.log(tau_init / (1 - tau_init))
        self.tau_raw = nn.Parameter(torch.full((out_features,), tau_raw_init))
        self.vth_raw = nn.Parameter(torch.full((out_features,), float(vth_init)))
        self.register_buffer("mem", None)
        self.register_buffer("spike_count", torch.tensor(0.0))
        self.register_buffer("step_count",  torch.tensor(0))

    @property
    def tau(self):
        return torch.sigmoid(self.tau_raw)

    @property
    def vth(self):
        return torch.nn.functional.softplus(self.vth_raw)

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem         = torch.zeros(batch_size, self.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        return (self.spike_count / (self.out_features * self.step_count *
                getattr(self, "batch_size", 1))).item()

    def forward(self, x):
        cur = self.fc(x)
        self.mem = self.tau * self.mem + cur
        spk = spike_fn(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count + 1
        return spk, self.mem
""")

# ── models/pointnet_backbone.py ─────────────────────────────────────────────
FILES["models/pointnet_backbone.py"] = textwrap.dedent("""\
import torch
import torch.nn as nn
from models.snn_layers import LIFLayer, LearnableLIFLayer


def knn_graph(pts, k):
    B, N, _ = pts.shape
    diff  = pts.unsqueeze(2) - pts.unsqueeze(1)
    dist2 = (diff ** 2).sum(-1)
    eye   = torch.eye(N, device=pts.device, dtype=torch.bool).unsqueeze(0)
    dist2 = dist2.masked_fill(eye, float('inf'))
    idx   = dist2.topk(k, dim=-1, largest=False).indices
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    neighbours = torch.gather(
        pts.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_exp)
    return neighbours


class LocalKNNBackbone(nn.Module):
    def __init__(self, hidden_dims=[128, 256, 512], k=16, learnable_lif=True):
        super().__init__()
        self.k = k
        LayerCls = LearnableLIFLayer if learnable_lif else LIFLayer
        in_dim = 3 + 3 * k
        self.layers = nn.ModuleList()
        for h in hidden_dims:
            self.layers.append(LayerCls(in_dim, h))
            in_dim = h
        self.out_dim = hidden_dims[-1]

    def reset_state(self, batch_size, device=None):
        for layer in self.layers:
            layer.reset_state(batch_size, device)

    def forward(self, pts):
        B, N, _ = pts.shape
        neighbours = knn_graph(pts, self.k)
        rel        = neighbours - pts.unsqueeze(2)
        rel_flat   = rel.reshape(B, N, self.k * 3)
        x = torch.cat([pts, rel_flat], dim=-1)
        self.reset_state(B * N, pts.device)
        x = x.reshape(B * N, -1)
        for layer in self.layers:
            spk, mem = layer(x)
            x = mem
        return mem.reshape(B, N, -1)

    def firing_rates(self):
        rates = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'firing_rate'):
                rates[f"knn_layer_{i}"] = layer.firing_rate()
        return rates
""")

# ── models/temporal_snn.py ──────────────────────────────────────────────────
FILES["models/temporal_snn.py"] = textwrap.dedent("""\
import torch
import torch.nn as nn
from models.snn_layers import LIFLayer, LearnableLIFLayer


class TemporalSNN(nn.Module):
    def __init__(self, dim=512, num_classes=10, learnable_lif=True):
        super().__init__()
        LayerCls = LearnableLIFLayer if learnable_lif else LIFLayer
        self.lif1 = LayerCls(dim, dim)
        self.lif2 = LayerCls(dim, dim)
        self.fc   = nn.Linear(dim, num_classes)

    def reset_state(self, batch_size, device=None):
        self.lif1.reset_state(batch_size, device)
        self.lif2.reset_state(batch_size, device)

    def forward(self, x):
        spk1, mem1 = self.lif1(x)
        spk2, mem2 = self.lif2(mem1)
        return self.fc(mem2)

    def firing_rates(self):
        rates = {}
        for name, layer in [("temporal_lif1", self.lif1), ("temporal_lif2", self.lif2)]:
            if hasattr(layer, 'firing_rate'):
                rates[name] = layer.firing_rate()
        return rates
""")

# ── models/slice_selection_policy.py ────────────────────────────────────────
FILES["models/slice_selection_policy.py"] = textwrap.dedent("""\
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

GEO_DIM = 6


def compute_geometry_descriptors(pts, fps_anchors, anchor_assignments):
    B, N, _ = pts.shape
    M = fps_anchors.size(1)
    cloud_centroid = pts.mean(dim=1, keepdim=True)
    anchor_dist = (fps_anchors - cloud_centroid).norm(dim=-1)
    mean_intra  = torch.zeros(B, M, device=pts.device)
    point_count = torch.zeros(B, M, device=pts.device)
    for m in range(M):
        mask  = (anchor_assignments == m)
        count = mask.float().sum(dim=1)
        point_count[:, m] = count
        for b in range(B):
            cluster_pts = pts[b][mask[b]]
            if cluster_pts.size(0) > 1:
                centroid = cluster_pts.mean(dim=0)
                mean_intra[b, m] = (cluster_pts - centroid).norm(dim=-1).mean()
    avg_count  = N / M
    norm_count = point_count / (avg_count + 1e-6)
    G = torch.cat([
        fps_anchors,
        anchor_dist.unsqueeze(-1),
        mean_intra.unsqueeze(-1),
        norm_count.unsqueeze(-1),
    ], dim=-1)
    return G


class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=GEO_DIM, d_ssp=64):
        super().__init__()
        self.d_ssp = d_ssp
        self.scale = math.sqrt(d_ssp)
        self.W_k   = nn.Linear(mem_dim, d_ssp, bias=False)
        self.W_q   = nn.Linear(geo_dim, d_ssp, bias=False)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)

    def forward(self, mem, geo, visited_mask=None):
        key   = self.W_k(mem)
        query = self.W_q(geo)
        scores = torch.bmm(query, key.unsqueeze(-1)).squeeze(-1) / self.scale
        if visited_mask is not None:
            scores = scores.masked_fill(visited_mask, float("-inf"))
        return scores

    def select_gumbel(self, scores, tau=1.0):
        return F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)

    def select_greedy(self, scores):
        idx = scores.argmax(dim=-1)
        return F.one_hot(idx, num_classes=scores.size(-1)).float()
""")

# ── models/active_snn.py ────────────────────────────────────────────────────
FILES["models/active_snn.py"] = textwrap.dedent("""\
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_backbone import LocalKNNBackbone
from models.temporal_snn import TemporalSNN
from models.slice_selection_policy import SliceSelectionPolicy, compute_geometry_descriptors


class ActiveSNN(nn.Module):
    def __init__(self, point_dims=[128, 256, 512], temporal_dim=512,
                 num_classes=10, knn_k=16, d_ssp=64):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.num_classes  = num_classes
        self.backbone = LocalKNNBackbone(hidden_dims=point_dims, k=knn_k, learnable_lif=True)
        self.temporal = TemporalSNN(dim=temporal_dim, num_classes=num_classes, learnable_lif=True)
        self.ssp      = SliceSelectionPolicy(mem_dim=temporal_dim, d_ssp=d_ssp)
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    def reset_state(self, batch_size, device=None):
        self.backbone.reset_state(batch_size, device)
        self.temporal.reset_state(batch_size, device)

    def _get_membrane(self):
        lif = self.temporal.lif2
        if lif.mem is None:
            return None
        return lif.mem.detach()

    def forward_active_train(self, pts_slices, geo_descriptors):
        B, T, n_pts, _ = pts_slices.shape
        device = pts_slices.device
        self.reset_state(B, device)
        pts_flat = pts_slices.reshape(B * T, n_pts, 3)
        self.backbone.reset_state(B * T, device)
        feat_per_point = self.backbone.forward(pts_flat)
        all_feats = feat_per_point.mean(dim=1).reshape(B, T, -1)
        self.backbone.reset_state(B, device)
        self.temporal.reset_state(B, device)
        visited_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        logits_all   = []
        mem_state    = torch.zeros(B, self.temporal_dim, device=device)
        for t in range(T):
            scores = self.ssp(mem_state, geo_descriptors, visited_mask)
            tau = self.gumbel_tau.item()
            w   = self.ssp.select_gumbel(scores, tau=tau)
            selected_idx = scores.masked_fill(visited_mask, float("-inf")).argmax(dim=-1)
            for b in range(B):
                visited_mask[b, selected_idx[b]] = True
            e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)
            logits_t = self.temporal(e_t)
            logits_all.append(logits_t)
            mem_state = self._get_membrane()
            if mem_state is None:
                mem_state = torch.zeros(B, self.temporal_dim, device=device)
        return logits_all[-1], logits_all

    def forward_active_infer(self, pts_slices, geo_descriptors, threshold=0.7):
        B, T, n_pts, _ = pts_slices.shape
        device = pts_slices.device
        self.reset_state(B, device)
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
            for b in range(B):
                visited_mask[b, selected_idx[b]] = True
            slice_b = pts_slices[:, chosen, :, :]
            with torch.no_grad():
                self.backbone.reset_state(B, device)
                feat_pp = self.backbone(slice_b)
                e_t     = feat_pp.mean(dim=1)
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

    def get_firing_rates(self):
        rates = {}
        if hasattr(self.backbone, "firing_rates"):
            rates.update(self.backbone.firing_rates())
        if hasattr(self.temporal, "firing_rates"):
            rates.update(self.temporal.firing_rates())
        return rates

    def mean_firing_rate(self):
        rates = self.get_firing_rates()
        return sum(rates.values()) / len(rates) if rates else 0.0

    def set_gumbel_tau(self, tau):
        self.gumbel_tau.fill_(tau)

    def param_count(self):
        bb   = sum(p.numel() for p in self.backbone.parameters())
        temp = sum(p.numel() for p in self.temporal.parameters())
        ssp  = sum(p.numel() for p in self.ssp.parameters())
        return {"backbone": bb, "temporal": temp, "ssp": ssp, "total": bb + temp + ssp}
""")

# ── data/__init__.py ─────────────────────────────────────────────────────────
FILES["data/__init__.py"] = ""

# ── data/modelnet.py ─────────────────────────────────────────────────────────
FILES["data/modelnet.py"] = textwrap.dedent("""\
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, split='train'):
        self.root       = root
        self.num_points = num_points
        self.split      = split
        self.files      = self._scan_files()
        self.data, self.labels = self._load_all()

    def _scan_files(self):
        items = []
        for class_name in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_name, self.split)
            if not os.path.isdir(class_path):
                continue
            label = sorted(os.listdir(self.root)).index(class_name)
            for f in os.listdir(class_path):
                if f.endswith(('.npy', '.txt', '.off')):
                    items.append((os.path.join(class_path, f), label))
        return items

    def _load_points(self, path):
        if path.endswith('.npy'):
            return np.load(path).astype(np.float32)
        elif path.endswith('.txt'):
            return np.loadtxt(path).astype(np.float32)
        elif path.endswith('.off'):
            return self._load_off(path)
        raise ValueError(f"Unsupported: {path}")

    def _load_off(self, path):
        try:
            import trimesh
            mesh = trimesh.load(path)
            pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            return pts.astype(np.float32)
        except Exception:
            # fallback: parse OFF manually
            with open(path) as f:
                lines = f.read().splitlines()
            start = 1
            if lines[0].strip() == 'OFF':
                start = 1
            elif lines[0].strip().startswith('OFF'):
                lines[0] = lines[0][3:]
                start = 0
            n_verts = int(lines[start].split()[0])
            verts = []
            for i in range(start + 1, start + 1 + n_verts):
                verts.append([float(v) for v in lines[i].split()[:3]])
            pts = np.array(verts, dtype=np.float32)
            idx = np.random.choice(len(pts), self.num_points,
                                   replace=(len(pts) < self.num_points))
            return pts[idx]

    def _load_all(self):
        all_pts, all_labels = [], []
        for path, label in self.files:
            pts = self._load_points(path)
            if not path.endswith('.off'):
                if pts.shape[0] >= self.num_points:
                    idx = np.random.choice(pts.shape[0], self.num_points, replace=False)
                    pts = pts[idx]
                else:
                    pad = self.num_points - pts.shape[0]
                    rep = np.random.choice(pts.shape[0], pad, replace=True)
                    pts = np.vstack([pts, pts[rep]])
            all_pts.append(pts)
            all_labels.append(label)
        return np.array(all_pts), np.array(all_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts   = self.data[idx].copy()
        label = self.labels[idx]
        np.random.shuffle(pts)
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
""")

# ── data/slicing.py ──────────────────────────────────────────────────────────
FILES["data/slicing.py"] = textwrap.dedent("""\
import math
import torch


def farthest_point_sample(pts, n_samples):
    N = pts.shape[0]
    n_samples = min(n_samples, N)
    device    = pts.device
    selected  = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    farthest  = torch.randint(0, N, (1,), device=device).item()
    for i in range(n_samples):
        selected[i] = farthest
        centroid = pts[farthest]
        dist = ((pts - centroid) ** 2).sum(-1)
        distances = torch.minimum(distances, dist)
        farthest  = distances.argmax().item()
    return selected


def slice_fps_hierarchical_batch(points, T=16):
    B, N, C = points.shape
    points_per_slice = N // T
    device = points.device
    all_slices = []
    for b in range(B):
        pts_b   = points[b]
        fps_idx = farthest_point_sample(pts_b, T)
        centres = pts_b[fps_idx]
        diff    = pts_b.unsqueeze(0) - centres.unsqueeze(1)
        dist2   = (diff ** 2).sum(-1)
        assign  = dist2.argmin(dim=0)
        slices_b = []
        for t in range(T):
            mask = (assign == t).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                mask = torch.randperm(N, device=device)[:points_per_slice]
            elif mask.numel() < points_per_slice:
                reps = math.ceil(points_per_slice / mask.numel())
                mask = mask.repeat(reps)[:points_per_slice]
            else:
                mask = mask[:points_per_slice]
            slices_b.append(pts_b[mask])
        all_slices.append(torch.stack(slices_b, dim=0))
    return torch.stack(all_slices, dim=0)
""")

# ── training/__init__.py ─────────────────────────────────────────────────────
FILES["training/__init__.py"] = ""

# ── training/optimizers.py ──────────────────────────────────────────────────
FILES["training/optimizers.py"] = textwrap.dedent("""\
import torch

def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
""")

# ── training/metrics.py ─────────────────────────────────────────────────────
FILES["training/metrics.py"] = textwrap.dedent("""\
import torch

def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
""")

# ── training/loss_active.py ─────────────────────────────────────────────────
FILES["training/loss_active.py"] = textwrap.dedent("""\
import torch
import torch.nn.functional as F


def active_loss(logits_final, logits_all, labels, model,
                lam_aux=0.3, lam_exit=0.1, lam_fr=0.05):
    # L_CE
    l_ce = F.cross_entropy(logits_final, labels)

    # L_aux
    T = len(logits_all)
    if T > 1:
        l_aux = sum(F.cross_entropy(logits_all[t], labels)
                    for t in range(T - 1)) / (T - 1)
        l_aux = lam_aux * l_aux
    else:
        l_aux = torch.tensor(0.0, device=labels.device)

    # L_exit
    l_exit = torch.tensor(0.0, device=labels.device)
    for t, lg in enumerate(logits_all):
        max_prob = F.softmax(lg, dim=-1).max(dim=-1).values
        weight_t = (T - t) / T
        l_exit = l_exit + weight_t * (1.0 - max_prob).mean()
    l_exit = lam_exit * l_exit / T

    # L_fr
    if hasattr(model, "mean_firing_rate"):
        r = model.mean_firing_rate()
        l_fr = lam_fr * (torch.tensor(r, dtype=torch.float32, device=labels.device)
                         if not isinstance(r, torch.Tensor) else r.to(labels.device))
    else:
        l_fr = torch.tensor(0.0, device=labels.device)

    total = l_ce + l_aux + l_exit + l_fr
    breakdown = {
        "loss_ce":    l_ce.item(),
        "loss_aux":   l_aux.item(),
        "loss_exit":  l_exit.item(),
        "loss_fr":    l_fr.item() if isinstance(l_fr, torch.Tensor) else float(l_fr),
        "loss_total": total.item(),
    }
    return total, breakdown
""")

# ── training/train_active.py ─────────────────────────────────────────────────
FILES["training/train_active.py"] = textwrap.dedent("""\
import torch
import torch.nn.functional as F
import time
import math

from data.slicing import slice_fps_hierarchical_batch
from training.loss_active import active_loss
from training.metrics import accuracy
from models.slice_selection_policy import compute_geometry_descriptors


def gumbel_tau(epoch, tau_0=1.0, tau_min=0.1, anneal_rate=0.05):
    return max(tau_min, tau_0 * math.exp(-anneal_rate * epoch))


def prepare_fps_slices_and_geo(pts, T):
    B, N, _ = pts.shape
    pts_slices  = slice_fps_hierarchical_batch(pts, T=T)   # [B, T, N//T, 3]
    fps_anchors = pts_slices.mean(dim=2)                   # [B, T, 3]
    diffs       = pts.unsqueeze(2) - fps_anchors.unsqueeze(1)
    dists       = (diffs ** 2).sum(-1)
    assignments = dists.argmin(dim=-1)
    geo = compute_geometry_descriptors(pts, fps_anchors, assignments)
    return pts_slices, geo, fps_anchors, assignments


def train_active_epoch(model, dataloader, optimizer, device, epoch,
                       num_slices=16, lam_aux=0.3, lam_exit=0.1, lam_fr=0.05,
                       tau_0=1.0, tau_min=0.1, anneal_rate=0.05, verbose_every=20):
    model.train()
    tau = gumbel_tau(epoch, tau_0, tau_min, anneal_rate)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)

    total_ce = total_aux = total_exit = total_fr = total_tot = 0.0
    total_acc = total_ent = 0.0
    count = 0
    start = time.time()

    for batch_idx, (pts, labels) in enumerate(dataloader):
        pts    = pts.to(device)
        labels = labels.to(device)
        B      = pts.size(0)

        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=num_slices)
        logits_final, logits_all = model.forward_active_train(pts_slices, geo)

        loss, breakdown = active_loss(
            logits_final, logits_all, labels, model,
            lam_aux=lam_aux, lam_exit=lam_exit, lam_fr=lam_fr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            mem_zero = torch.zeros(B, model.temporal_dim, device=device)
            scores   = model.ssp(mem_zero, geo, visited_mask=None)
            probs    = torch.softmax(scores, dim=-1)
            ent      = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()

        total_ce   += breakdown["loss_ce"]
        total_aux  += breakdown["loss_aux"]
        total_exit += breakdown["loss_exit"]
        total_fr   += breakdown["loss_fr"]
        total_tot  += breakdown["loss_total"]
        total_acc  += accuracy(logits_final, labels)
        total_ent  += ent.item()
        count += 1

        if (batch_idx + 1) % verbose_every == 0:
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [{batch_idx+1}/{len(dataloader)}] "
                  f"CE={breakdown['loss_ce']:.4f} "
                  f"Aux={breakdown['loss_aux']:.4f} "
                  f"Exit={breakdown['loss_exit']:.4f} "
                  f"FR={breakdown['loss_fr']:.4f} "
                  f"Acc={total_acc/count:.3f} "
                  f"Ent={total_ent/count:.3f} "
                  f"tau={tau:.3f} LR={lr:.6f} {elapsed:.0f}s")

    n = max(count, 1)
    return {
        "loss_ce": total_ce/n, "loss_aux": total_aux/n,
        "loss_exit": total_exit/n, "loss_fr": total_fr/n,
        "loss_total": total_tot/n, "acc_final": total_acc/n,
        "policy_entropy": total_ent/n, "gumbel_tau": tau,
    }


def validate_active(model, dataloader, device, num_slices=16, threshold=0.7):
    model.eval()
    correct = total = 0
    total_exit = total_fr = count = 0.0

    with torch.no_grad():
        for pts, labels in dataloader:
            pts    = pts.to(device)
            labels = labels.to(device)
            B      = pts.size(0)
            pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=num_slices)
            for b in range(B):
                pts_b = pts_slices[b].unsqueeze(0)
                geo_b = geo[b].unsqueeze(0)
                lbl_b = labels[b].unsqueeze(0)
                logits, exit_step, _ = model.forward_active_infer(pts_b, geo_b, threshold=threshold)
                pred = logits.argmax(dim=-1)
                correct += (pred == lbl_b).sum().item()
                total   += 1
                total_exit += exit_step
            fr = model.mean_firing_rate()
            total_fr += fr if isinstance(fr, float) else fr.item()
            count    += 1

    n = max(total, 1)
    mean_fr    = total_fr / max(count, 1)
    mean_exit  = total_exit / n
    energy_r   = mean_fr * 0.274 * (mean_exit / num_slices)
    return {
        "acc":          correct / n,
        "mean_exit":    mean_exit,
        "mean_fr":      mean_fr,
        "energy_ratio": energy_r,
        "savings":      1.0 / max(energy_r, 1e-9),
    }
""")

# ── inference/__init__.py ────────────────────────────────────────────────────
FILES["inference/__init__.py"] = ""

# ── inference/active_inference.py ────────────────────────────────────────────
FILES["inference/active_inference.py"] = textwrap.dedent("""\
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

E_AC  = 2.3e-3
E_MAC = 8.4e-3
EFFICIENCY_RATIO = E_AC / E_MAC


def energy_ratio(firing_rate, exit_fraction):
    return firing_rate * EFFICIENCY_RATIO * exit_fraction


def pareto_curve(model, dataset, device, num_slices=16, thresholds=None, prepare_fn=None):
    if thresholds is None:
        thresholds = [round(i * 0.05, 2) for i in range(21)]
    if prepare_fn is None:
        from training.train_active import prepare_fps_slices_and_geo
        prepare_fn = prepare_fps_slices_and_geo

    curve = []
    for theta in thresholds:
        correct = total = exit_sum = 0
        fr_sum  = 0.0
        model.eval()
        with torch.no_grad():
            for pts, label in dataset:
                if pts.dim() == 2:
                    pts = pts.unsqueeze(0)
                pts = pts.to(device)
                label_idx = label if isinstance(label, int) else label.item()
                pts_slices, geo, _, _ = prepare_fn(pts, T=num_slices)
                logits, exit_step, _ = model.forward_active_infer(pts_slices, geo, threshold=theta)
                pred = logits.argmax(-1).item()
                correct  += int(pred == label_idx)
                total    += 1
                exit_sum += exit_step
                fr_sum   += model.mean_firing_rate()
        n = max(total, 1)
        mean_fr   = fr_sum / n
        mean_exit = exit_sum / n
        e_r = energy_ratio(mean_fr, mean_exit / num_slices)
        m = {"threshold": theta, "accuracy": correct/n, "mean_exit": mean_exit,
             "mean_fr": mean_fr, "energy_ratio": e_r, "savings": 1.0/max(e_r, 1e-9)}
        curve.append(m)
        print(f"  theta={theta:.2f}  acc={m['accuracy']:.4f}  "
              f"mean_exit={mean_exit:.2f}/{num_slices}  "
              f"energy={e_r:.4f}  savings={m['savings']:.1f}x")
    curve.sort(key=lambda x: x["energy_ratio"])
    return curve
""")

# ── plots_active.py ──────────────────────────────────────────────────────────
FILES["plots_active.py"] = textwrap.dedent("""\
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

COLORS = {
    "asp":       "#E6550D",
    "ours_full": "#FD8D3C",
    "ours_knn":  "#FDBE85",
    "spt":       "#3182BD",
    "spm":       "#6BAED6",
}


def plot_pareto(asp_curve, fixed_baselines, save_path="results/active/fig1_pareto.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    energies = [p["energy_ratio"] for p in asp_curve]
    accs     = [p["accuracy"] * 100 for p in asp_curve]
    ax.plot(energies, accs, "o-", color=COLORS["asp"], linewidth=2.5,
            markersize=5, label="ASP (adaptive)")
    baseline_styles = {
        "ours_full": ("s", COLORS["ours_full"], "ours_full (8.4x)"),
        "ours_knn":  ("^", COLORS["ours_knn"],  "ours_knn (11.1x)"),
        "spt":       ("D", COLORS["spt"],        "SPT (6.4x)"),
        "spm":       ("P", COLORS["spm"],        "SPM (3.5x)"),
    }
    for name, meta in fixed_baselines.items():
        if name in baseline_styles:
            marker, color, label = baseline_styles[name]
            ax.scatter(meta["energy_ratio"], meta["accuracy"]*100,
                       marker=marker, color=color, s=100, zorder=6,
                       label=label, edgecolors="black", linewidths=0.7)
    ax.set_xlabel("E_SNN / E_ANN", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Energy-Accuracy Pareto Frontier (ModelNet10)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved -> {save_path}")


def plot_exit_distribution(exit_steps, num_slices=16, save_path="results/active/fig2_exit.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bins = range(1, num_slices + 2)
    ax1.hist(exit_steps, bins=bins, align="left", rwidth=0.8,
             color=COLORS["asp"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.axvline(x=np.mean(exit_steps), color="black", linestyle="--",
                linewidth=1.5, label=f"Mean = {np.mean(exit_steps):.1f}")
    ax1.set_xlabel("Exit Timestep"); ax1.set_ylabel("Samples")
    ax1.set_title("Exit Timestep Distribution"); ax1.legend()
    exits_sorted = np.sort(exit_steps)
    p = np.arange(1, len(exits_sorted)+1) / len(exits_sorted)
    ax2.plot(exits_sorted, p*100, color=COLORS["asp"], linewidth=2.5)
    ax2.set_xlabel("Exit Timestep"); ax2.set_ylabel("Cumulative %")
    ax2.set_title("Exit CDF"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved -> {save_path}")


def plot_training_history(history, save_path="results/active/fig3_history.png"):
    epochs = [h["epoch"] for h in history]
    acc    = [h.get("acc_final", 0)*100 for h in history]
    loss   = [h.get("loss_total", 0) for h in history]
    tau    = [h.get("gumbel_tau", 1.0) for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, acc, color=COLORS["asp"], linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Accuracy (%)")
    ax1.set_title("Training Accuracy"); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, loss, color=COLORS["ours_full"], linewidth=2, label="Total Loss")
    ax2_t = ax2.twinx()
    ax2_t.plot(epochs, tau, color="grey", linestyle="--", linewidth=1.5, label="Gumbel tau")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2_t.set_ylabel("Gumbel tau")
    ax2.set_title("Loss + Gumbel Annealing"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved -> {save_path}")
""")

# ---------------------------------------------------------------------------
# Notebook cell definitions
# ---------------------------------------------------------------------------

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# Active Spiking Perception (ASP) — Colab Notebook\n"
    "**Membrane-Guided Adaptive Slice Selection for Anytime Energy-Efficient 3D Recognition**\n\n"
    "This notebook is fully self-contained. Run cells top-to-bottom.\n\n"
    "Steps:\n"
    "1. **Runtime setup** — GPU check + install deps\n"
    "2. **Write source files** — all modules written to `/content/asp/`\n"
    "3. **Download dataset** — ModelNet10 (.off files)\n"
    "4. **Train** — 30-epoch quick run (change `--epochs` for full run)\n"
    "5. **Evaluate** — accuracy + energy ratio\n"
    "6. **Pareto sweep** — full energy-accuracy curve\n"
    "7. **Figures** — inline plots\n"
    "8. **PDF report** — download\n"
))

# ── 1. GPU check + install ────────────────────────────────────────────────────
cells.append(md("## 1. Runtime Setup"))
cells.append(code(
    "import torch\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('GPU:', torch.cuda.get_device_name(0))\n"
    "else:\n"
    "    print('WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU')\n"
))
cells.append(code(
    "# Install dependencies\n"
    "import subprocess, sys\n"
    "pkgs = ['trimesh', 'reportlab']\n"
    "for p in pkgs:\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', p, '-q'], check=True)\n"
    "print('Dependencies installed.')\n"
))

# ── 2. Write source files ─────────────────────────────────────────────────────
cells.append(md("## 2. Write All Source Files\nThis cell writes every module to `/content/asp/`."))

write_cell_lines = [
    "import os, sys, textwrap\n",
    "\n",
    "ROOT = '/content/asp'\n",
    "os.makedirs(ROOT, exist_ok=True)\n",
    "sys.path.insert(0, ROOT)\n",
    "\n",
    "FILES = {}\n",
]

for rel_path, content in FILES.items():
    # JSON-encode the content string to safely embed it
    encoded = json.dumps(content)
    write_cell_lines.append(f"FILES[{json.dumps(rel_path)}] = {encoded}\n")

write_cell_lines += [
    "\n",
    "for rel_path, content in FILES.items():\n",
    "    full_path = os.path.join(ROOT, rel_path)\n",
    "    os.makedirs(os.path.dirname(full_path), exist_ok=True)\n",
    "    with open(full_path, 'w') as f:\n",
    "        f.write(content)\n",
    "\n",
    "print(f'Written {len(FILES)} files to {ROOT}')\n",
    "for k in sorted(FILES.keys()):\n",
    "    print(' ', k)\n",
]

cells.append(code("".join(write_cell_lines)))

# ── 3. Download ModelNet10 ────────────────────────────────────────────────────
cells.append(md(
    "## 3. Download ModelNet10\n"
    "Downloads the Princeton ModelNet10 dataset (~50 MB, .off mesh files).\n"
    "Surface points are sampled from meshes by the dataset loader.\n\n"
    "> **Alternative:** If the Princeton URL is unavailable, uncomment the Kaggle block."
))
cells.append(code(
    "import os\n"
    "\n"
    "DATASET_DIR = '/content/ModelNet10'\n"
    "\n"
    "if not os.path.isdir(DATASET_DIR):\n"
    "    print('Downloading ModelNet10...')\n"
    "    os.system('wget -q https://modelnet.cs.princeton.edu/ModelNet10.zip -O /tmp/MN10.zip')\n"
    "    os.system('unzip -q /tmp/MN10.zip -d /content/')\n"
    "    print('Done.')\n"
    "else:\n"
    "    print('ModelNet10 already present.')\n"
    "\n"
    "# Verify structure\n"
    "classes = sorted([d for d in os.listdir(DATASET_DIR)\n"
    "                  if os.path.isdir(os.path.join(DATASET_DIR, d))])\n"
    "print(f'Classes ({len(classes)}):', classes)\n"
))
cells.append(code(
    "# --- Kaggle alternative (run if wget fails) ---\n"
    "# from google.colab import files\n"
    "# files.upload()  # upload kaggle.json\n"
    "# os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)\n"
    "# os.system('cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')\n"
    "# os.system('pip install kaggle -q')\n"
    "# os.system('kaggle datasets download -d balraj98/modelnet10-princeton-3d-object-dataset')\n"
    "# os.system('unzip -q modelnet10*.zip -d /content/ModelNet10')\n"
))

# ── 4. Sanity check ───────────────────────────────────────────────────────────
cells.append(md("## 4. Sanity Check — Model Instantiation"))
cells.append(code(
    "import sys\n"
    "sys.path.insert(0, '/content/asp')\n"
    "\n"
    "import torch\n"
    "from models.active_snn import ActiveSNN\n"
    "\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print('Device:', device)\n"
    "\n"
    "model = ActiveSNN(\n"
    "    point_dims=[128, 256, 512],\n"
    "    temporal_dim=512,\n"
    "    num_classes=10,\n"
    "    knn_k=16,\n"
    "    d_ssp=64,\n"
    ").to(device)\n"
    "\n"
    "params = model.param_count()\n"
    "print(f'Backbone: {params[\"backbone\"]:,}')\n"
    "print(f'Temporal: {params[\"temporal\"]:,}')\n"
    "print(f'SSP:      {params[\"ssp\"]:,}  ({params[\"ssp\"]/params[\"total\"]*100:.2f}% of total)')\n"
    "print(f'Total:    {params[\"total\"]:,}')\n"
    "\n"
    "# Quick forward pass test\n"
    "pts_test   = torch.randn(2, 1024, 3).to(device)\n"
    "from training.train_active import prepare_fps_slices_and_geo\n"
    "pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts_test, T=16)\n"
    "with torch.no_grad():\n"
    "    logits_f, logits_all = model.forward_active_train(pts_slices, geo)\n"
    "print(f'Training forward OK: logits shape = {logits_f.shape}')\n"
    "logits_i, exit_step, order = model.forward_active_infer(pts_slices[:1], geo[:1], threshold=0.7)\n"
    "print(f'Inference forward OK: exit_step={exit_step}, order={order[:4]}...')\n"
))

# ── 5. Load dataset ───────────────────────────────────────────────────────────
cells.append(md("## 5. Load Dataset"))
cells.append(code(
    "from data.modelnet import ModelNetDataset\n"
    "from torch.utils.data import DataLoader\n"
    "\n"
    "print('Loading training set (sampling surfaces from .off meshes)...')\n"
    "train_ds = ModelNetDataset(root='/content/ModelNet10', split='train', num_points=1024)\n"
    "val_ds   = ModelNetDataset(root='/content/ModelNet10', split='test',  num_points=1024)\n"
    "\n"
    "train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)\n"
    "val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)\n"
    "\n"
    "print(f'Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples')\n"
    "print(f'Batches per epoch: {len(train_loader)}')\n"
))

# ── 6. Train ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 6. Train ASP\n"
    "Default: **30 epochs** (quick run ~10-20 min on T4). "
    "Change `EPOCHS = 150` for the full SOTA run."
))
cells.append(code(
    "import os, json, torch\n"
    "from models.active_snn import ActiveSNN\n"
    "from training.optimizers import build_optimizer\n"
    "from training.train_active import train_active_epoch, validate_active\n"
    "\n"
    "# ---- Config ----\n"
    "EPOCHS     = 30        # change to 150 for SOTA\n"
    "NUM_SLICES = 16\n"
    "THRESHOLD  = 0.7\n"
    "SAVE_DIR   = '/content/asp_results'\n"
    "os.makedirs(SAVE_DIR, exist_ok=True)\n"
    "\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "\n"
    "model = ActiveSNN(\n"
    "    point_dims=[128, 256, 512], temporal_dim=512,\n"
    "    num_classes=10, knn_k=16, d_ssp=64\n"
    ").to(device)\n"
    "\n"
    "optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)\n"
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)\n"
    "\n"
    "history      = []\n"
    "best_val_acc = 0.0\n"
    "best_ckpt    = os.path.join(SAVE_DIR, 'best_model.pth')\n"
    "\n"
    "for epoch in range(EPOCHS):\n"
    "    print(f'\\n--- Epoch {epoch}/{EPOCHS-1} ---')\n"
    "    train_m = train_active_epoch(\n"
    "        model, train_loader, optimizer, device, epoch=epoch,\n"
    "        num_slices=NUM_SLICES, verbose_every=10)\n"
    "    scheduler.step()\n"
    "\n"
    "    val_m = {}\n"
    "    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:\n"
    "        val_m = validate_active(model, val_loader, device,\n"
    "                                num_slices=NUM_SLICES, threshold=THRESHOLD)\n"
    "        val_acc = val_m['acc']\n"
    "        print(f'[Val] Acc={val_acc:.4f}  '\n"
    "              f'MeanExit={val_m[\"mean_exit\"]:.2f}/{NUM_SLICES}  '\n"
    "              f'FR={val_m[\"mean_fr\"]:.3f}  '\n"
    "              f'EnergyRatio={val_m[\"energy_ratio\"]:.4f}  '\n"
    "              f'Savings={val_m[\"savings\"]:.1f}x')\n"
    "        if val_acc > best_val_acc:\n"
    "            best_val_acc = val_acc\n"
    "            torch.save({'epoch': epoch, 'model_state': model.state_dict(),\n"
    "                        'val_acc': val_acc}, best_ckpt)\n"
    "            print(f'  *** New best: {val_acc:.4f} -> saved')\n"
    "\n"
    "    history.append({'epoch': epoch, **train_m, **val_m})\n"
    "    with open(os.path.join(SAVE_DIR, 'history.json'), 'w') as f:\n"
    "        json.dump(history, f, indent=2)\n"
    "\n"
    "print(f'\\nBest Val Acc: {best_val_acc:.4f}')\n"
))

# ── 7. Final evaluation ────────────────────────────────────────────────────────
cells.append(md("## 7. Final Evaluation (Best Checkpoint)"))
cells.append(code(
    "import torch\n"
    "from models.active_snn import ActiveSNN\n"
    "from training.train_active import validate_active\n"
    "\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "ckpt   = torch.load('/content/asp_results/best_model.pth', map_location=device)\n"
    "model  = ActiveSNN(point_dims=[128, 256, 512], temporal_dim=512,\n"
    "                   num_classes=10, knn_k=16, d_ssp=64).to(device)\n"
    "model.load_state_dict(ckpt['model_state'])\n"
    "print(f'Loaded checkpoint from epoch {ckpt[\"epoch\"]}  (val_acc={ckpt[\"val_acc\"]:.4f})')\n"
    "\n"
    "for theta in [0.3, 0.5, 0.7, 0.9]:\n"
    "    m = validate_active(model, val_loader, device, num_slices=16, threshold=theta)\n"
    "    print(f'theta={theta:.1f}  acc={m[\"acc\"]:.4f}  '\n"
    "          f'mean_exit={m[\"mean_exit\"]:.2f}  '\n"
    "          f'savings={m[\"savings\"]:.1f}x  '\n"
    "          f'energy_ratio={m[\"energy_ratio\"]:.4f}')\n"
))

# ── 8. Pareto sweep ────────────────────────────────────────────────────────────
cells.append(md(
    "## 8. Pareto Sweep\n"
    "Sweeps exit threshold theta from 0 to 1 to trace the full energy-accuracy curve. "
    "Uses a 200-sample subset for speed (remove `[:200]` for full validation set)."
))
cells.append(code(
    "import json\n"
    "from inference.active_inference import pareto_curve\n"
    "from training.train_active import prepare_fps_slices_and_geo\n"
    "\n"
    "# Use 200-sample subset for speed; remove slice for full curve\n"
    "val_subset = torch.utils.data.Subset(val_ds, range(min(200, len(val_ds))))\n"
    "\n"
    "curve = pareto_curve(\n"
    "    model, val_subset, device,\n"
    "    num_slices=16,\n"
    "    thresholds=[round(i*0.1, 1) for i in range(11)],\n"
    "    prepare_fn=prepare_fps_slices_and_geo,\n"
    ")\n"
    "\n"
    "with open('/content/asp_results/pareto_curve.json', 'w') as f:\n"
    "    json.dump(curve, f, indent=2)\n"
    "print('Pareto curve saved.')\n"
))

# ── 9. Figures ─────────────────────────────────────────────────────────────────
cells.append(md("## 9. Figures (Inline)"))
cells.append(code(
    "import json, sys\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.image as mpimg\n"
    "sys.path.insert(0, '/content/asp')\n"
    "import plots_active as PA\n"
    "\n"
    "os.makedirs('/content/asp_results/figs', exist_ok=True)\n"
    "\n"
    "# Fig 1: Pareto frontier\n"
    "with open('/content/asp_results/pareto_curve.json') as f:\n"
    "    curve = json.load(f)\n"
    "\n"
    "fixed_baselines = {\n"
    "    'ours_full': {'energy_ratio': 0.119, 'accuracy': 0.9064},\n"
    "    'ours_knn':  {'energy_ratio': 0.090, 'accuracy': 0.8987},\n"
    "    'spt':       {'energy_ratio': 0.156, 'accuracy': 0.914},\n"
    "    'spm':       {'energy_ratio': 0.286, 'accuracy': 0.923},\n"
    "}\n"
    "PA.plot_pareto(curve, fixed_baselines, save_path='/content/asp_results/figs/fig1_pareto.png')\n"
    "\n"
    "# Fig 2: Exit distribution\n"
    "# Collect exit steps from final val run\n"
    "exit_steps = []\n"
    "model.eval()\n"
    "with torch.no_grad():\n"
    "    for pts, labels in val_loader:\n"
    "        pts = pts.to(device)\n"
    "        from training.train_active import prepare_fps_slices_and_geo\n"
    "        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=16)\n"
    "        for b in range(pts.size(0)):\n"
    "            _, es, _ = model.forward_active_infer(\n"
    "                pts_slices[b:b+1], geo[b:b+1], threshold=0.7)\n"
    "            exit_steps.append(es)\n"
    "        if len(exit_steps) >= 200:\n"
    "            break\n"
    "PA.plot_exit_distribution(exit_steps, save_path='/content/asp_results/figs/fig2_exit.png')\n"
    "\n"
    "# Fig 3: Training history\n"
    "with open('/content/asp_results/history.json') as f:\n"
    "    history = json.load(f)\n"
    "PA.plot_training_history(history, save_path='/content/asp_results/figs/fig3_history.png')\n"
    "\n"
    "# Display all figures inline\n"
    "for figpath in ['fig1_pareto', 'fig2_exit', 'fig3_history']:\n"
    "    path = f'/content/asp_results/figs/{figpath}.png'\n"
    "    if os.path.exists(path):\n"
    "        img = mpimg.imread(path)\n"
    "        plt.figure(figsize=(10, 6))\n"
    "        plt.imshow(img); plt.axis('off')\n"
    "        plt.title(figpath); plt.tight_layout(); plt.show()\n"
))

# ── 10. Generate PDF ───────────────────────────────────────────────────────────
cells.append(md("## 10. Generate PDF Report + Download"))
cells.append(code(
    "# Copy the generate_pdf.py from the source project\n"
    "# (already written by make_notebook.py if run locally,\n"
    "#  or paste it here if running fresh on Colab)\n"
    "import subprocess\n"
    "result = subprocess.run(\n"
    "    ['python', '/content/asp/generate_pdf.py'],\n"
    "    capture_output=True, text=True, cwd='/content/asp'\n"
    ")\n"
    "print(result.stdout or 'generate_pdf.py not found in /content/asp')\n"
    "if result.stderr:\n"
    "    print('STDERR:', result.stderr[:500])\n"
))
cells.append(code(
    "# Download results to your local machine\n"
    "import os, shutil\n"
    "from google.colab import files\n"
    "\n"
    "# Zip results + figures\n"
    "shutil.make_archive('/content/asp_results_export', 'zip', '/content/asp_results')\n"
    "files.download('/content/asp_results_export.zip')\n"
    "\n"
    "# Also download checkpoint\n"
    "if os.path.exists('/content/asp_results/best_model.pth'):\n"
    "    files.download('/content/asp_results/best_model.pth')\n"
))

# ── 11. Resume from checkpoint ────────────────────────────────────────────────
cells.append(md(
    "## 11. Resume Training from Checkpoint\n"
    "If your Colab session resets, upload `best_model.pth` and continue training."
))
cells.append(code(
    "# from google.colab import files\n"
    "# uploaded = files.upload()  # upload best_model.pth\n"
    "#\n"
    "# ckpt = torch.load('best_model.pth', map_location=device)\n"
    "# model.load_state_dict(ckpt['model_state'])\n"
    "# start_epoch = ckpt['epoch'] + 1\n"
    "# print(f'Resumed from epoch {ckpt[\"epoch\"]}  acc={ckpt[\"val_acc\"]:.4f}')\n"
    "# # Then re-run the training cell from epoch=start_epoch\n"
))

# ---------------------------------------------------------------------------
# Assemble + write notebook
# ---------------------------------------------------------------------------

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ASP_Colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook written -> {out_path}")
print(f"  Cells:       {len(cells)}")
print(f"  Source files embedded: {len(FILES)}")
print(f"\nUpload ASP_Colab.ipynb to https://colab.research.google.com and run top-to-bottom.")

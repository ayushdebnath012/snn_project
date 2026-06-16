"""
spt_block.py
============
SPT (Spiking Point Transformer) components adapted from arXiv 2502.15811
"SPT: Spiking Point Transformer for Point Cloud Processing".

Key algorithmic ideas implemented:

  Q-SDE  → Queue-Driven Sampling Direct Encoding:
           A FIFO queue accumulates point cloud events over time.
           At each timestep, the top-K highest-confidence points are
           popped and fed to the backbone. This creates a natural
           curriculum: salient regions first.

  Spiking KNN Attention:
           Instead of standard softmax attention, spike-coded keys and
           queries are computed via LIF neurons. Attention weights are
           binary (spike/no-spike), making the operation purely additive
           (AC instead of MAC).

  HD-IF    → Hybrid Dynamics IF neuron (from neuron_zoo.py):
           Fuses LIF, IF, EIF via learned gate.  Used in place of
           standard LIF throughout the SPT pipeline.

  SPTBackbone:
           Q-SDE sampling → spiking KNN attention block →
           HD-IF projection → global pooling

Paper differences vs ours:
  - Paper uses full attention on neighbourhood graph (like PointTransformer)
  - Paper uses Q-SDE with hardware-friendly FIFO for event cameras
  - We adapt Q-SDE to a priority queue sorted by local density
    (denser = more informative = process first)
  - Paper uses 3-stage hierarchical processing; we use 1 stage for speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neuron_zoo import HDIFNeuron, tri_spike


# ---------------------------------------------------------------------------
# Q-SDE: Queue-Driven Sampling Direct Encoding
# ---------------------------------------------------------------------------

def local_density(pts, k=8):
    """
    Estimate local density for each point as 1 / mean_knn_distance.

    pts : [N, 3]
    Returns : [N]  density scores (higher = denser neighbourhood)
    """
    # Pairwise distances [N, N]
    diff = pts.unsqueeze(0) - pts.unsqueeze(1)        # [N, N, 3]
    dist2 = (diff ** 2).sum(-1)                        # [N, N]

    # k-nearest distances (excluding self)
    k_actual = min(k, pts.size(0) - 1)
    topk_dist, _ = dist2.topk(k_actual + 1, dim=-1, largest=False)
    knn_dist = topk_dist[:, 1:]                        # exclude self (dist=0)
    mean_dist = knn_dist.mean(dim=-1).clamp(min=1e-6)

    return 1.0 / mean_dist                             # [N]


class QSDEQueue:
    """
    Queue-Driven Sampling Direct Encoding.

    Maintains a priority queue of point indices sorted by local density.
    Each call to `pop_slice(K)` returns the K highest-density points
    not yet processed, simulating the FIFO event camera buffer.

    Usage:
      queue = QSDEQueue(pts)            # [N, 3]
      idx = queue.pop_slice(K)          # [K] indices
      pts_slice = pts[idx]              # [K, 3]
    """
    def __init__(self, pts, k_density=8):
        """
        pts : [N, 3]  single point cloud (not batched)
        """
        density = local_density(pts, k=k_density)     # [N]
        # Sort by density descending
        self._order = density.argsort(descending=True).tolist()
        self._pos = 0
        self._N = pts.size(0)

    def pop_slice(self, K):
        """Return next K indices from the priority queue."""
        start = self._pos
        end   = min(self._pos + K, self._N)
        idx   = self._order[start:end]
        self._pos = end
        # Pad if fewer than K remain
        if len(idx) < K:
            idx = idx + self._order[:K - len(idx)]
        return idx

    def remaining(self):
        return self._N - self._pos

    def is_empty(self):
        return self._pos >= self._N


def qsde_slice(pts, T, k_density=8):
    """
    Slice a single point cloud [N, 3] into T slices using Q-SDE.

    Returns list of T index lists, each of length N//T.
    High-density (salient) points appear in early slices.
    """
    N = pts.size(0)
    K = max(1, N // T)
    queue = QSDEQueue(pts, k_density=k_density)
    slices = []
    for _ in range(T):
        idx = queue.pop_slice(K)
        slices.append(idx)
    return slices


# ---------------------------------------------------------------------------
# Spiking KNN Attention Block
# ---------------------------------------------------------------------------

class SpikingKNNAttention(nn.Module):
    """
    Spiking KNN-based attention (SPT §3.2).

    For each point, aggregate features from its K nearest neighbours
    using spike-coded query-key matching.  Binary attention weights
    (from LIF) make this purely additive at inference.

    Forward:
      pts   : [B, N, 3]
      feats : [B, N, C]
      → [B, N, C]  enriched features

    Steps:
      1. Compute KNN graph
      2. Linear project feats → Q, K, V
      3. Q, K through LIF → binary spikes
      4. Attention score = spike_Q · spike_K (dot, binary)
      5. Aggregate: weighted sum of V using score
    """
    def __init__(self, in_ch, out_ch, k=16, tau=0.9, vth=1.0):
        super().__init__()
        self.k   = k
        self.tau = tau
        self.vth = vth

        self.Q_proj = nn.Linear(in_ch, out_ch, bias=False)
        self.K_proj = nn.Linear(in_ch, out_ch, bias=False)
        self.V_proj = nn.Linear(in_ch, out_ch, bias=False)

        # Learnable threshold for Q and K spiking gates (vth per channel)
        self.q_vth = nn.Parameter(torch.full((out_ch,), vth))
        self.k_vth = nn.Parameter(torch.full((out_ch,), vth))
        self.out_ch = out_ch

    def reset_state(self, batch_size, device):
        pass  # Q/K gates are stateless (fire based on threshold only)

    def _knn_idx(self, pts):
        """
        pts : [B, N, 3]
        Returns idx : [B, N, k]
        """
        B, N, _ = pts.shape
        diff  = pts.unsqueeze(2) - pts.unsqueeze(1)   # [B, N, N, 3]
        dist2 = (diff ** 2).sum(-1)                    # [B, N, N]
        k_actual = min(self.k, N - 1)
        _, idx = dist2.topk(k_actual + 1, dim=-1, largest=False)
        return idx[:, :, 1:]                           # exclude self

    def forward(self, pts, feats):
        B, N, C = feats.shape
        k_actual = min(self.k, N - 1)

        # Project
        Q = self.Q_proj(feats)   # [B, N, out_ch]
        K = self.K_proj(feats)
        V = self.V_proj(feats)

        # Spiking Q and K: stateless threshold firing (STE surrogate)
        # q_vth/k_vth are per-channel learnable thresholds [out_ch]
        from models.neuron_zoo import tri_spike
        spk_Q = tri_spike(Q - self.q_vth)   # [B, N, out_ch]
        spk_K = tri_spike(K - self.k_vth)   # [B, N, out_ch]

        # KNN index
        idx = self._knn_idx(pts)                       # [B, N, k_actual]

        # Gather K spikes for neighbours
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, self.out_ch)  # [B, N, k, out_ch]
        spk_K_nb = torch.gather(
            spk_K.unsqueeze(2).expand(-1, -1, k_actual, -1),
            1, idx_exp
        )                                              # [B, N, k, out_ch]
        V_nb = torch.gather(
            V.unsqueeze(2).expand(-1, -1, k_actual, -1),
            1, idx_exp
        )                                              # [B, N, k, out_ch]

        # Binary attention: Q · K_nb  (inner product of binary spikes)
        # → [B, N, k]  (scalar attention per neighbour)
        attn = (spk_Q.unsqueeze(2) * spk_K_nb).sum(-1)   # [B, N, k]
        attn = attn / (self.out_ch ** 0.5 + 1e-6)
        attn = F.softmax(attn, dim=-1)

        # Aggregate
        out = (attn.unsqueeze(-1) * V_nb).sum(dim=2)     # [B, N, out_ch]
        return out


# ---------------------------------------------------------------------------
# SPT Backbone Block
# ---------------------------------------------------------------------------

class SPTBlock(nn.Module):
    """
    One SPT processing stage:
      - Spiking KNN attention enriches per-point features
      - HD-IF projection (linear + HD-IF neuron)
      - Residual connection
    """
    def __init__(self, in_ch, out_ch, k=16, tau=0.9, vth=1.0):
        super().__init__()
        self.attn    = SpikingKNNAttention(in_ch, out_ch, k=k, tau=tau, vth=vth)
        self.hdif    = HDIFNeuron(out_ch, vth=vth, tau=tau)
        self.norm    = nn.LayerNorm(out_ch)
        self.proj_res = nn.Linear(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()
        self.out_ch  = out_ch

    def reset_state(self, batch_size, device):
        self.attn.reset_state(batch_size, device)
        self.hdif.reset(batch_size, device)

    def forward(self, pts, feats):
        """
        pts   : [B, N, 3]
        feats : [B, N, in_ch]
        Returns : [B, N, out_ch]
        """
        B, N, _ = feats.shape

        # Spiking attention
        attn_out = self.attn(pts, feats)           # [B, N, out_ch]

        # HD-IF: run on flattened [B*N, out_ch]
        # Reset neuron state to match B*N effective batch size each call
        flat = attn_out.view(B * N, self.out_ch)
        self.hdif.reset(B * N, flat.device)
        spk, _ = self.hdif(flat)
        spk = spk.view(B, N, self.out_ch)

        # Residual + norm
        res = self.proj_res(feats)                 # [B, N, out_ch]
        out = self.norm(spk + res)
        return out


# ---------------------------------------------------------------------------
# SPT Backbone: Q-SDE + SPTBlock + global pool
# ---------------------------------------------------------------------------

class SPTBackbone(nn.Module):
    """
    SPT-inspired backbone for point cloud slice processing.

    Pipeline (per slice):
      pts [B, N, 3] → initial linear embed → SPTBlock × 2 → global avg pool → [B, out_dim]

    Q-SDE is applied externally (at the dataset/slicing level) to order slices.
    The backbone itself processes whatever slice it receives.

    For Q-SDE ordered inference, use qsde_slice() in data/slicing.py
    (or call it here via get_qsde_slices()).
    """
    def __init__(self, in_dim=3, hidden_ch=64, out_dim=256, k=16, tau=0.9, vth=1.0):
        super().__init__()
        self.k = k

        # Initial embedding: raw XYZ → hidden features
        self.embed = nn.Linear(in_dim, hidden_ch)

        # Two SPT blocks
        self.block1 = SPTBlock(hidden_ch,     hidden_ch * 2, k=k, tau=tau, vth=vth)
        self.block2 = SPTBlock(hidden_ch * 2, out_dim,       k=k, tau=tau, vth=vth)

        self.out_dim = out_dim

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.embed.parameters()).device
        self.block1.reset_state(batch_size, dev)
        self.block2.reset_state(batch_size, dev)

    def forward(self, pts):
        """
        pts : [B, N, 3]
        Returns : [B, out_dim]
        """
        feats = self.embed(pts)              # [B, N, hidden_ch]
        feats = self.block1(pts, feats)      # [B, N, hidden_ch*2]
        feats = self.block2(pts, feats)      # [B, N, out_dim]
        return feats.mean(dim=1)             # [B, out_dim]  global avg pool


# ---------------------------------------------------------------------------
# Full SPT model (backbone + temporal classifier)
# ---------------------------------------------------------------------------

class SPTModel(nn.Module):
    """
    Full SPT-style model for point cloud classification.

    Uses:
      - SPTBackbone (Q-SDE compatible, spiking KNN attention, HD-IF)
      - Simple LIF temporal integration over slices
      - Linear classifier

    Compatible with the PointNetSNN interface:
      forward_step(pts_slice) → logits
      reset_state(batch_size, device)
    """
    def __init__(self, hidden_ch=64, out_dim=256, num_classes=40,
                 k=16, tau=0.9, vth=1.0):
        super().__init__()
        self.backbone  = SPTBackbone(in_dim=3, hidden_ch=hidden_ch,
                                     out_dim=out_dim, k=k, tau=tau, vth=vth)
        # Temporal LIF integration
        self.tau = tau
        self.vth = vth
        self.fc  = nn.Linear(out_dim, num_classes)
        self.out_dim = out_dim

        self.register_buffer("mem", None, persistent=False)

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.backbone.reset_state(batch_size, dev)
        self.mem = torch.zeros(batch_size, self.out_dim, device=dev)

    def forward_step(self, pts):
        """
        pts : [B, N, 3]  (one slice)
        Returns logits [B, num_classes]
        """
        feat = self.backbone(pts)             # [B, out_dim]
        self.mem = self.tau * self.mem + feat
        spk = tri_spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        return self.fc(spk)

    def forward(self, pts_slices):
        """
        pts_slices : [B, T, N, 3]
        Returns logits [B, num_classes]
        """
        B, T, N, _ = pts_slices.shape
        for t in range(T):
            logits = self.forward_step(pts_slices[:, t])
        return logits

    def get_qsde_slice_order(self, pts, T, k_density=8):
        """
        Return Q-SDE ordered slice indices for a single point cloud.
        pts : [N, 3]
        Returns list of T lists of point indices.
        """
        return qsde_slice(pts, T, k_density=k_density)

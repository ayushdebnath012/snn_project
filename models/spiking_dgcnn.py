"""
Spiking DGCNN — Max-First Graph Spiking Network for Point Cloud Classification.

Implements two innovations from "Spiking DGCNN: Max-First Graph Spiking Networks
for Efficient Point Cloud Learning" (NeurIPS 2026 submission):

  1. Max-First Spiking Rule: neighbourhood max aggregation (continuous domain)
     happens BEFORE the spiking nonlinearity in EdgeConv.  Spike-before-Max
     collapses ordering; Max-before-Spike preserves it.

  2. APTEC (Adaptive Pseudo-Temporal Expansion-Compression): lightweight
     pseudo-temporal dynamics via feature repetition + leaky LIF integration
     + input-conditioned adaptive threshold + logical-OR compression over T steps.

Reference accuracy (T=4, k=20):
    ModelNet40: 92.38%   ModelNet10: 94.93%
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.snn_layers import spike_fn


# ── Membrane Potential Rectifier (MPR) ─────────────────────────────────────────

def _mpr(u: torch.Tensor) -> torch.Tensor:
    """Clip membrane potential to [0, 1.5] for binary-friendly quantisation."""
    return torch.clamp(u, min=0.0, max=1.5)


# ── APTEC Neuron ───────────────────────────────────────────────────────────────

class APTECNeuron(nn.Module):
    """
    Adaptive Pseudo-Temporal Expansion-Compression spiking neuron.

    Pipeline for each of the T pseudo-timesteps (all see the same input x):
        u_t  = decay * u_{t-1} + x                        # leaky integration
        û_t  = MPR(u_t)                                    # clip to [0, 1.5]
        V_th = 1 + 0.5 * sigmoid(x)    ∈ (1, 1.5)         # adaptive threshold
        z_t  = û_t / V_th                                  # normalised variable
        s_t  = H(z_t − 0.5)            via surrogate grad  # fire / no-fire
        u_t  ← u_t − s_t                                   # soft reset

    Output = logical OR over {s_1, …, s_T} = max_t s_t ∈ {0, 1}.

    The adaptive threshold introduces negative feedback:
    • borderline units (z near 0.5) are suppressed by a slightly enlarged V_th,
      preventing repetition-induced spurious spikes.
    • saturated units (z > 1) are pulled back into the surrogate-gradient-active
      region [0, 1], restoring trainability.
    """

    def __init__(self, T: int = 4, decay: float = 0.9):
        super().__init__()
        self.T = T
        self.decay = decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(x)
        spikes: list[torch.Tensor] = []

        for _ in range(self.T):
            u = self.decay * u + x                          # integrate
            u_hat = _mpr(u)                                 # MPR
            v_th = 1.0 + 0.5 * torch.sigmoid(x)            # adaptive threshold
            z = u_hat / v_th                                # normalised variable
            s = spike_fn(z - 0.5)                          # fire when z > 0.5
            u = u - s                                       # soft reset
            spikes.append(s)

        # Temporal compression: OR over timesteps
        return torch.stack(spikes, dim=0).max(dim=0).values


# ── kNN graph helpers ──────────────────────────────────────────────────────────

def _knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-nearest-neighbour indices in feature space.

    Args:
        x: [B, N, C]
        k: number of neighbours (self excluded)

    Returns:
        idx: [B, N, k]
    """
    B, N, C = x.shape
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
    sq = (x ** 2).sum(dim=-1, keepdim=True)                # [B, N, 1]
    dist = sq + sq.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
    # k+1 smallest distances; drop index 0 (self)
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    return idx[:, :, 1:].contiguous()                      # [B, N, k]


def _edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Construct edge features: cat(x_i, x_j − x_i) for each neighbour j.

    Args:
        x:   [B, N, C]
        idx: [B, N, k]

    Returns:
        edge: [B, N, k, 2C]
    """
    B, N, C = x.shape
    k = idx.shape[-1]

    # Gather neighbour features efficiently via a flat gather
    idx_flat = idx.reshape(B, -1)                           # [B, N*k]
    x_nbrs = torch.gather(
        x,
        dim=1,
        index=idx_flat.unsqueeze(-1).expand(-1, -1, C),
    ).reshape(B, N, k, C)                                  # [B, N, k, C]

    x_center = x.unsqueeze(2).expand(-1, -1, k, -1)        # [B, N, k, C]
    return torch.cat([x_center, x_nbrs - x_center], dim=-1)  # [B, N, k, 2C]


# ── EdgeConv with Max-First Spiking Rule ───────────────────────────────────────

class EdgeConvMaxFirst(nn.Module):
    """
    EdgeConv reordered for spiking compatibility.

    Standard EdgeConv:  MLP → LeakyReLU → max (over k neighbours)
    Max-First Spiking:  MLP → BN        → max (continuous) → APTEC spike

    Moving max before spike preserves winner-take-all neighbour competition in
    the continuous domain so the spiking binarisation acts on the STRONGEST
    local response rather than collapsing all supra-threshold responses to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
        T: int = 4,
        decay: float = 0.9,
    ):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.aptec = APTECNeuron(T=T, decay=decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C_in]   (binary spikes after first block; raw xyz for block 1)
        Returns:
            out: [B, N, C_out]  binary spikes
        """
        B, N, C = x.shape

        idx = _knn_idx(x, self.k)                          # [B, N, k]
        edge = _edge_features(x, idx)                      # [B, N, k, 2C]

        # Shared linear projection on all edge features
        edge_flat = edge.reshape(B * N * self.k, 2 * C)
        h = self.linear(edge_flat)                         # [B*N*k, out]
        h = self.bn(h)                                     # BN on flat dim
        h = h.reshape(B, N, self.k, -1)                   # [B, N, k, out]

        # ── Max-First: aggregate over neighbours in the CONTINUOUS domain ──
        h_max, _ = h.max(dim=2)                            # [B, N, out]

        # ── APTEC spike applied AFTER max aggregation ──
        return self.aptec(h_max)                           # [B, N, out]  binary


# ── Full Spiking DGCNN Classifier ─────────────────────────────────────────────

class SpikingDGCNN(nn.Module):
    """
    Spiking DGCNN for 3-D point cloud classification.

    Architecture (follows original DGCNN, LeakyReLU → Max-First APTEC):
        EdgeConv × 4  [64, 64, 128, 256]  k=20
        → concat  [B, N, 512]
        → point-wise Linear(512→1024) + BN + APTEC
        → global max-pool over N points  [B, 1024]
        → FC(1024→512) + BN + APTEC + Dropout
        → FC(512→256)  + BN + APTEC + Dropout
        → Linear(256→num_classes)

    Paper settings: k=20, T=4, decay=0.9, dropout=0.5
    SGD lr=0.1 → 0.001 cosine, 300 epochs, batch 32.
    """

    def __init__(
        self,
        num_classes: int = 40,
        k: int = 20,
        T: int = 4,
        decay: float = 0.9,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.T = T
        self.k = k

        # ── 4 EdgeConv blocks ───────────────────────────────────────────────
        self.edge1 = EdgeConvMaxFirst(3,   64,  k=k, T=T, decay=decay)
        self.edge2 = EdgeConvMaxFirst(64,  64,  k=k, T=T, decay=decay)
        self.edge3 = EdgeConvMaxFirst(64,  128, k=k, T=T, decay=decay)
        self.edge4 = EdgeConvMaxFirst(128, 256, k=k, T=T, decay=decay)

        # ── Point-wise projection: 64+64+128+256=512 → 1024 ─────────────────
        self.proj = nn.Linear(512, 1024, bias=False)
        self.bn_proj = nn.BatchNorm1d(1024)
        self.aptec_proj = APTECNeuron(T=T, decay=decay)

        # ── Classifier head ─────────────────────────────────────────────────
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.aptec1 = APTECNeuron(T=T, decay=decay)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.aptec2 = APTECNeuron(T=T, decay=decay)
        self.drop2 = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: [B, N, 3]  normalised point cloud

        Returns:
            logits: [B, num_classes]
        """
        # ── 4-block EdgeConv tower ───────────────────────────────────────────
        x1 = self.edge1(pts)    # [B, N, 64]
        x2 = self.edge2(x1)     # [B, N, 64]
        x3 = self.edge3(x2)     # [B, N, 128]
        x4 = self.edge4(x3)     # [B, N, 256]

        # ── Multi-scale feature concatenation ────────────────────────────────
        x = torch.cat([x1, x2, x3, x4], dim=-1)   # [B, N, 512]

        # ── Point-wise projection + APTEC ────────────────────────────────────
        B, N, _ = x.shape
        x = self.bn_proj(self.proj(x.reshape(B * N, -1))).reshape(B, N, -1)
        x = self.aptec_proj(x)                     # [B, N, 1024]  binary

        # ── Global max-pool over all N points ───────────────────────────────
        #    For binary spikes: max = logical OR — fires if ANY point fired
        x = x.max(dim=1).values                    # [B, 1024]

        # ── FC classifier ────────────────────────────────────────────────────
        x = self.aptec1(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = self.aptec2(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        return self.classifier(x)                  # [B, num_classes]

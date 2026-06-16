"""
spiking_mamba.py
================
Spiking Point Mamba (SPM) inspired architecture adapted from arXiv:2504.14371
(ICCV 2025).

Key components:
  1. HierarchicalDynamicEncoding (HDE):
     - Improved direct encoding that introduces dynamic temporal mechanisms
     - Divides FPS-ordered points into 3 temporal stages:
       Early (diverse anchors), Middle (stable skeleton), Late (surface detail)
     - Each stage gets a learnable temporal embedding

  2. SpikingMambaBlock (SMB):
     - Builds on selective SSM (Mamba) with inter-time-step spike features
     - Minimises information loss from spikes via residual connections
     - Architecture: LayerNorm → Linear → SN → SSM → SN → Linear + skip

  3. SPMModel:
     - Full model: Backbone encoder → HDE → SMB temporal → classifier
     - Compatible with forward_step / reset_state interface
     - Can use either our FPS slicing or radial slicing

Paper: "Efficient Spiking Point Mamba for Point Cloud Analysis"
       arXiv:2504.14371, Accepted at ICCV 2025
       Key result: 92.3% MN40, +6.2% over prior SNN SOTA on ScanObjectNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.neuron_zoo import tri_spike


# ---------------------------------------------------------------------------
# Spiking Neuron (Batch Norm + LIF) — used inside SMB
# ---------------------------------------------------------------------------

class SpikingNeuron(nn.Module):
    """LIF neuron with optional batch norm, used inside Spiking Mamba Block."""
    def __init__(self, dim, tau=0.9, vth=1.0, use_bn=True):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.bn = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.register_buffer("mem", None, persistent=False)

    def reset(self, batch_size, device):
        self.mem = torch.zeros(batch_size, self.bn.num_features
                               if isinstance(self.bn, nn.BatchNorm1d)
                               else 0, device=device)
        self.spike_count = torch.tensor(0.0, device=device)
        self.step_count = torch.tensor(0, device=device)
        self.batch_size = batch_size
        self.out_features = self.bn.num_features if isinstance(self.bn, nn.BatchNorm1d) else 1

    def firing_rate(self):
        if not hasattr(self, "step_count") or self.step_count == 0:
            return 0.0
        return (self.spike_count / (self.out_features * self.step_count * getattr(self, "batch_size", 1))).item()

    def forward(self, x):
        """x: [B, D] → spike: [B, D]"""
        x = self.bn(x)
        if self.mem is None or self.mem.shape[0] != x.shape[0]:
            self.mem = torch.zeros_like(x)
        self.mem = self.tau * self.mem + x
        spk = tri_spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)  # hard reset

        if not hasattr(self, "spike_count"):
            self.spike_count = torch.tensor(0.0, device=x.device)
            self.step_count = torch.tensor(0, device=x.device)
            self.batch_size = x.shape[0]
            self.out_features = x.shape[1]

        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count = self.step_count + 1

        return spk


# ---------------------------------------------------------------------------
# Selective SSM Core (Mamba-style S6)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective SSM for temporal processing.

    Unlike the fixed S4D in spiking_ssm.py, the selective SSM has:
      - Input-dependent B, C (selective — content-aware)
      - Input-dependent dt (step size)
      - Gated output via D parameter

    This is the key difference: the SSM parameters adapt to the input,
    making it better at selectively remembering/forgetting information.
    """
    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # Input-dependent projections (selective)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, self.dt_rank, bias=False),
            nn.Linear(self.dt_rank, d_model, bias=True),
            nn.Softplus(),
        )

        # Learned A (log-space for stability)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        # clone() is required because expand() returns a shared-storage view;
        # optimizers with in-place weight decay cannot update that safely.
        A_log = A_log.unsqueeze(0).expand(d_model, -1).clone()  # [d_model, d_state]
        self.A_log = nn.Parameter(A_log)

        # Skip parameter D
        self.D = nn.Parameter(torch.ones(d_model))

        # Hidden state
        self.register_buffer("h", None, persistent=False)

    def reset(self, batch_size, device):
        self.h = torch.zeros(batch_size, self.d_model, self.d_state, device=device)

    def forward(self, x):
        """
        x: [B, d_model] (single timestep)
        Returns: [B, d_model]
        """
        B_batch = x.size(0)
        if self.h is None or self.h.shape[0] != B_batch:
            self.reset(B_batch, x.device)

        # Selective parameters (input-dependent)
        dt = self.dt_proj(x)                                # [B, d_model]
        B_in = self.B_proj(x)                               # [B, d_state]
        C_in = self.C_proj(x)                               # [B, d_state]

        # Discretise A
        A = -torch.exp(self.A_log)                          # [d_model, d_state]
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_model, d_state]

        # ZOH discretisation for B
        B_bar = dt.unsqueeze(-1) * B_in.unsqueeze(1)       # [B, d_model, d_state]

        # SSM recurrence
        self.h = A_bar * self.h + B_bar                     # [B, d_model, d_state]

        # Output: y = C * h + D * x
        y = (self.h * C_in.unsqueeze(1)).sum(-1)            # [B, d_model]
        y = y + self.D * x

        return y


# ---------------------------------------------------------------------------
# Spiking Mamba Block (SMB) — core of SPM
# ---------------------------------------------------------------------------

class SpikingMambaBlock(nn.Module):
    """
    Spiking Mamba Block (SMB) from SPM paper.

    Architecture:
        x → LayerNorm → Linear → SN → SelectiveSSM → SN → Linear → + x (skip)

    Key features vs plain SpikingSSMCell:
      1. Residual connection preserves information lost by spiking
      2. Selective SSM (input-dependent B, C, dt) instead of fixed S4D
      3. Two spiking neurons sandwich the SSM for spike-driven processing
      4. LayerNorm before projection for training stability
    """
    def __init__(self, dim, d_state=16, tau=0.9, expand=2):
        super().__init__()
        inner_dim = dim * expand

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner_dim)
        self.sn1 = SpikingNeuron(inner_dim, tau=tau)
        self.ssm = SelectiveSSM(inner_dim, d_state=d_state)
        self.sn2 = SpikingNeuron(inner_dim, tau=tau)
        self.out_proj = nn.Linear(inner_dim, dim)

        # Gate projection for selective gating
        self.gate_proj = nn.Linear(dim, inner_dim)

    def reset(self, batch_size, device):
        self.sn1.reset(batch_size, device)
        self.ssm.reset(batch_size, device)
        self.sn2.reset(batch_size, device)

    def forward(self, x):
        """x: [B, dim] → out: [B, dim]"""
        residual = x
        x_norm = self.norm(x)

        # Main path: Linear → SN → SSM → SN → Linear
        h = self.in_proj(x_norm)
        h = self.sn1(h)
        h = self.ssm(h)
        h = self.sn2(h)

        # Gate (like Mamba's multiplicative gate)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        h = h * gate

        out = self.out_proj(h)
        return out + residual  # Skip connection


# ---------------------------------------------------------------------------
# Hierarchical Dynamic Encoding (HDE)
# ---------------------------------------------------------------------------

class HierarchicalDynamicEncoding(nn.Module):
    """
    Hierarchical Dynamic Encoding from SPM.

    Adds learnable temporal embeddings to slice features based on which
    temporal stage the slice belongs to:
      - Early  (t < T/3):  diverse spatial anchors from FPS
      - Middle (T/3 ≤ t < 2T/3): stable skeleton structure
      - Late   (t ≥ 2T/3): surface detail / infill points

    Each stage gets a learned embedding vector added to the slice features,
    providing the model with explicit temporal context about *what kind*
    of spatial information this slice contains.
    """
    def __init__(self, dim, max_T=32):
        super().__init__()
        # Stage embeddings (early, middle, late)
        self.stage_embed = nn.Parameter(torch.randn(3, dim) * 0.02)
        # Fine-grained positional encoding (per-timestep)
        self.pos_embed = nn.Parameter(torch.randn(max_T, dim) * 0.02)
        self.max_T = max_T
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat, t, T):
        """
        feat: [B, dim] — slice embedding at timestep t
        t: int (scalar) OR torch.Tensor [B] of original FPS anchor indices
        T: total number of timesteps
        Returns: [B, dim] — temporally encoded feature
        """
        if isinstance(t, torch.Tensor):
            # Per-sample path: t is [B] int tensor of original FPS anchor indices.
            # Used by ASPWrapper to preserve HDE's geometric meaning when ASP
            # reorders processing (each sample in the batch may select a different
            # anchor, so stage/pos embedding must be per-sample).
            t_long = t.long()
            stage = torch.where(
                t_long < T // 3,
                torch.zeros_like(t_long),
                torch.where(
                    t_long < 2 * T // 3,
                    torch.ones_like(t_long),
                    torch.full_like(t_long, 2),
                ),
            )                                                    # [B]
            feat = feat + self.stage_embed[stage]               # [B, dim]
            pos_idx = t_long.clamp(0, self.max_T - 1)          # [B]
            feat = feat + self.pos_embed[pos_idx]               # [B, dim]
        else:
            # Scalar path (original code) — used by SPMModel.forward_step.
            if t < T // 3:
                stage = 0
            elif t < 2 * T // 3:
                stage = 1
            else:
                stage = 2
            feat = feat + self.stage_embed[stage]
            pos_idx = min(t, self.max_T - 1)
            feat = feat + self.pos_embed[pos_idx]

        # Project and normalise
        feat = self.norm(self.proj(feat))
        return feat


# ---------------------------------------------------------------------------
# SPM Full Model
# ---------------------------------------------------------------------------

class SPMModel(nn.Module):
    """
    Spiking Point Mamba (SPM) full model.

    Architecture:
        Point cloud slice [B, N, 3]
            ↓
        Backbone (PointNet or KNN) → per-point features → mean pool
            ↓
        HDE (temporal encoding)
            ↓
        2× SpikingMambaBlock (temporal integration)
            ↓
        Linear classifier → logits [B, num_classes]

    This model can use either our FPS slicing or radial slicing.
    The key experiment: does FPS slicing improve SPM over radial?
    """
    def __init__(self, num_classes=40, point_dims=(128, 256, 512),
                 d_state=16, tau=0.9, n_smb_layers=2, local_knn=True,
                 knn_k=16, learnable_lif=True, pooling="mean"):
        super().__init__()
        from models.pointnet_backbone import PointNetBackbone, LocalKNNBackbone

        feat_dim = point_dims[-1]
        self.pooling = pooling

        # Backbone — use BN-LIF to match SPM paper (Linear → BN → LIF pattern)
        if local_knn:
            self.backbone = LocalKNNBackbone(
                hidden_dims=list(point_dims), k=knn_k,
                learnable_lif=learnable_lif, use_bn=True
            )
        else:
            self.backbone = PointNetBackbone(
                hidden_dims=list(point_dims),
                learnable_lif=learnable_lif, use_bn=True
            )

        if pooling == "meanmax":
            self.pool_proj = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
            )
        elif pooling in ("mean", "max"):
            self.pool_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown SPM pooling mode: {pooling}")

        # HDE temporal encoding
        self.hde = HierarchicalDynamicEncoding(feat_dim)

        # Stacked Spiking Mamba Blocks
        self.smb_layers = nn.ModuleList([
            SpikingMambaBlock(feat_dim, d_state=d_state, tau=tau)
            for _ in range(n_smb_layers)
        ])

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

        self._current_t = 0
        self._total_T = 16

    def pool_points(self, per_point):
        """Pool per-point slice features into one temporal token."""
        if self.pooling == "mean":
            return per_point.mean(dim=1)
        if self.pooling == "max":
            return per_point.max(dim=1).values
        mean = per_point.mean(dim=1)
        maxv = per_point.max(dim=1).values
        return self.pool_proj(torch.cat([mean, maxv], dim=-1))

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc[0].parameters()).device
        self.backbone.reset_state(batch_size, dev)
        for smb in self.smb_layers:
            smb.reset(batch_size, dev)
        self._current_t = 0

    def forward_step(self, pts_slice):
        """
        Process one temporal slice.
        pts_slice: [B, N, 3]
        Returns: logits [B, num_classes]
        """
        # Backbone: per-point features -> slice token
        per_point = self.backbone(pts_slice)         # [B, N, D]
        feat = self.pool_points(per_point)           # [B, D]

        # HDE: add temporal encoding
        feat = self.hde(feat, self._current_t, self._total_T)

        # Spiking Mamba Blocks
        for smb in self.smb_layers:
            feat = smb(feat)

        self._current_t += 1
        return self.fc(feat)

    def forward_step_feat(self, feat, orig_t=None, T=None):
        """
        Process a precomputed backbone embedding through HDE + SMB layers + fc.
        Used by ASPWrapper, which precomputes all backbone embeddings and then
        feeds them in SSP-selected order.

        Args:
            feat   : [B, feat_dim]  precomputed backbone mean-pool embedding
            orig_t : int scalar, [B] int tensor, or None
                     Original FPS anchor index for HDE geometric encoding.
                     Pass the ANCHOR index (not processing step) so HDE's
                     stage/positional embeddings reflect spatial meaning.
                     If None, uses internal self._current_t (scalar path).
            T      : int or None  total slices; uses self._total_T if None.

        Returns:
            logits : [B, num_classes]
        """
        t_idx = orig_t if orig_t is not None else self._current_t
        total  = T     if T      is not None else self._total_T

        feat = self.hde(feat, t_idx, total)
        for smb in self.smb_layers:
            feat = smb(feat)

        # Advance counter only in non-ASP (default sequential) mode
        if orig_t is None:
            self._current_t += 1

        return self.fc(feat)

    def forward(self, pts_slices):
        """
        pts_slices: [B, T, N, 3]
        Returns: logits [B, num_classes]
        """
        B, T, N, _ = pts_slices.shape
        self._total_T = T
        self.reset_state(B, pts_slices.device)
        for t in range(T):
            logits = self.forward_step(pts_slices[:, t])
        return logits

    def get_firing_rates(self):
        """Collect firing rates from backbone and SMB layers."""
        rates = {}
        if hasattr(self.backbone, 'firing_rates'):
            rates.update(self.backbone.firing_rates())
            
        for i, smb in enumerate(self.smb_layers):
            if hasattr(smb.sn1, 'firing_rate'):
                rates[f'smb_layer_{i}_sn1'] = smb.sn1.firing_rate()
            if hasattr(smb.sn2, 'firing_rate'):
                rates[f'smb_layer_{i}_sn2'] = smb.sn2.firing_rate()
                
        return rates

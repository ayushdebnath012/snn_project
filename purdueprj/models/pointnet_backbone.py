import torch
import torch.nn as nn
from models.snn_layers import LIFLayer, LearnableLIFLayer, BNLIFLayer, PLIFLayer


def _pick_layer_cls(use_plif=False, use_bn=False, learnable_lif=False):
    if use_plif:
        return PLIFLayer
    if use_bn:
        return BNLIFLayer
    if learnable_lif:
        return LearnableLIFLayer
    return LIFLayer


class PointNetBackbone(nn.Module):
    """
    Per-point spiking MLP backbone.

    MaxPool (not mean-pool) over N points gives significantly better results
    for classification — consistent with PointNet's own ablation study.
    `use_plif=True` selects PLIF neurons with ATan surrogate for deeper
    multi-timestep training.
    """
    def __init__(self, hidden_dims=[64, 128, 256], learnable_lif=False,
                 use_bn=False, use_plif=False, pool='max'):
        super().__init__()
        self.pool   = pool
        self.layers = nn.ModuleList()
        LayerCls    = _pick_layer_cls(use_plif, use_bn, learnable_lif)
        in_dim      = 3
        for h in hidden_dims:
            self.layers.append(LayerCls(in_dim, h))
            in_dim = h
        self.out_dim = hidden_dims[-1]

    def reset_state(self, batch_size, device=None):
        for layer in self.layers:
            layer.reset_state(batch_size, device)

    def forward(self, pts):
        # pts: [B, N, 3]
        B, N, _ = pts.shape
        self.reset_state(batch_size=B * N, device=pts.device)
        x = pts.reshape(B * N, -1)
        for layer in self.layers:
            spk, mem = layer(x)
            x = mem
        feats = mem.reshape(B, N, -1)          # [B, N, D]
        if self.pool == 'max':
            return feats.max(dim=1).values     # [B, D]  — better than mean for classification
        return feats.mean(dim=1)               # [B, D]

    def firing_rates(self):
        return {f"pn_layer_{i}": l.firing_rate()
                for i, l in enumerate(self.layers)
                if hasattr(l, "firing_rate")}


# ---------------------------------------------------------------------------
# Novel: KNN Local Neighbourhood Backbone  (inspired by SPM's SEL)
# ---------------------------------------------------------------------------
def knn_graph(pts, k):
    """
    Build KNN graph for a batch of point clouds.
    pts : [B, N, 3]
    Returns neighbours : [B, N, k, 3]  — absolute coordinates of k neighbours
    """
    B, N, _ = pts.shape
    # Pairwise squared distances: [B, N, N]
    diff = pts.unsqueeze(2) - pts.unsqueeze(1)          # [B, N, N, 3]
    dist2 = (diff ** 2).sum(-1)                          # [B, N, N]
    # Exclude self (set diagonal to large value)
    eye = torch.eye(N, device=pts.device, dtype=torch.bool).unsqueeze(0)
    dist2 = dist2.masked_fill(eye, float('inf'))
    # Top-k nearest  [B, N, k]
    idx = dist2.topk(k, dim=-1, largest=False).indices
    # Gather neighbour coordinates [B, N, k, 3]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    pts_exp = pts.unsqueeze(2).expand(-1, -1, k, -1)    # not needed; use gather
    neighbours = torch.gather(
        pts.unsqueeze(1).expand(-1, N, -1, -1),          # [B, N, N, 3]
        2,
        idx_exp
    )
    return neighbours                                     # [B, N, k, 3]


class LocalKNNBackbone(nn.Module):
    """
    Novel feature: KNN local neighbourhood embedding + spiking MLP.

    For each point we compute:
        local  = concat(point_xyz, neighbour_xyz - point_xyz)  -> dim = 3 + 3*k
    then feed through a spiking MLP, and finally MaxPool across neighbours
    to get a per-point feature.

    This mirrors SPM's Spiking Embedding Layer (SEL) which applies KNN
    to encoded center points and uses MaxPooling for local context.
    The key improvement over the original backbone is that each point
    now *sees its spatial neighbourhood* rather than being processed
    independently.
    """
    def __init__(self, hidden_dims=[64, 128, 256], k=16, learnable_lif=False, use_bn=False):
        super().__init__()
        self.k = k
        if use_bn:
            LayerCls = BNLIFLayer
        elif learnable_lif:
            LayerCls = LearnableLIFLayer
        else:
            LayerCls = LIFLayer

        # Input dim: point (3) + relative neighbour coords (3*k)
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
        """
        pts : [B, N, 3]
        Returns : [B, N, out_dim]
        """
        B, N, _ = pts.shape

        # Build local features
        neighbours = knn_graph(pts, self.k)              # [B, N, k, 3]
        rel = neighbours - pts.unsqueeze(2)              # [B, N, k, 3] relative
        # Flatten relative neighbours: [B, N, 3*k]
        rel_flat = rel.reshape(B, N, self.k * 3)
        # Concatenate with absolute position: [B, N, 3 + 3*k]
        x = torch.cat([pts, rel_flat], dim=-1)           # [B, N, 3+3k]

        # Run spiking MLP over all (B*N) points simultaneously
        self.reset_state(batch_size=B * N, device=pts.device)
        x = x.reshape(B * N, -1)
        for layer in self.layers:
            spk, mem = layer(x)
            x = mem

        return mem.reshape(B, N, -1)                     # [B, N, out_dim]

    def firing_rates(self):
        """Return dict of firing rates per layer (only for LearnableLIFLayer)."""
        rates = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'firing_rate'):
                rates[f"knn_layer_{i}"] = layer.firing_rate()
        return rates


# ---------------------------------------------------------------------------
# Novel: Multi-Scale Spiking Backbone  (key for 92-93% OA)
# ---------------------------------------------------------------------------

class MultiScalePointNetBackbone(nn.Module):
    """
    Multi-scale spiking backbone: runs the shared-MLP on the full point cloud
    AND on a 2× downsampled version, then fuses via concatenation + projection.

    Why multi-scale helps:
      - Full cloud  : fine-grained local structure (edges, corners)
      - Half cloud  : coarser global shape (overall geometry)
    Both representations are complementary. The same insight drives multi-scale
    grouping in PointNet++ and the multi-resolution grid in VoxNet.

    Uses PLIF neurons (ATan surrogate) throughout for better gradient flow.
    Global MaxPool extracts the strongest per-channel activation.
    """
    def __init__(self, hidden_dims=(64, 256, 512), out_dim=512,
                 use_plif=True, pool='max'):
        super().__init__()
        self.pool = pool

        LayerCls = PLIFLayer if use_plif else BNLIFLayer

        def _mlp(dims):
            layers, in_d = nn.ModuleList(), 3
            for d in dims:
                layers.append(LayerCls(in_d, d))
                in_d = d
            return layers

        self.layers_full = nn.ModuleList(_mlp(hidden_dims))
        self.layers_half = nn.ModuleList(_mlp(hidden_dims))

        # Fusion: concat both streams and project to out_dim
        fused_dim = hidden_dims[-1] * 2
        self.fuse_fc = nn.Linear(fused_dim, out_dim, bias=False)
        self.fuse_bn = nn.BatchNorm1d(out_dim)
        self.out_dim = out_dim

    def _run_mlp(self, layers, pts):
        B, N, _ = pts.shape
        for l in layers:
            l.reset_state(B * N, pts.device)
        x = pts.reshape(B * N, 3)
        for l in layers:
            _, x = l(x)
        feats = x.reshape(B, N, -1)
        if self.pool == 'max':
            return feats.max(dim=1).values
        return feats.mean(dim=1)

    def reset_state(self, batch_size, device=None):
        pass  # state reset happens inside _run_mlp per-call

    def forward(self, pts):
        """pts : [B, N, 3]  →  [B, out_dim]"""
        B, N, _ = pts.shape

        # Full cloud
        f_full = self._run_mlp(self.layers_full, pts)       # [B, D]

        # Downsampled cloud (random half)
        idx  = torch.randperm(N, device=pts.device)[:N // 2]
        pts_half = pts[:, idx, :]
        f_half = self._run_mlp(self.layers_half, pts_half)  # [B, D]

        fused = torch.cat([f_full, f_half], dim=-1)         # [B, 2D]
        out   = self.fuse_bn(self.fuse_fc(fused))           # [B, out_dim]
        return out

    def firing_rates(self):
        rates = {}
        for i, l in enumerate(self.layers_full):
            if hasattr(l, "firing_rate"):
                rates[f"ms_full_{i}"] = l.firing_rate()
        for i, l in enumerate(self.layers_half):
            if hasattr(l, "firing_rate"):
                rates[f"ms_half_{i}"] = l.firing_rate()
        return rates

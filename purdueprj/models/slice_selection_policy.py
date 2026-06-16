"""
slice_selection_policy.py
=========================
Slice Selection Policy (SSP) for Active Spiking Perception.

The SSP is a lightweight dot-product attention module (~2K parameters) that
decides WHICH FPS anchor region to observe next, given:
  - u: current LIF membrane state (belief over class)           [B, D]
  - G: geometry descriptors for all M candidate anchors         [B, M, 6]
  - visited_mask: boolean mask of already-selected anchors      [B, M]

The output is a score vector [B, M] (masked visited → -inf) used for:
  - Training: Gumbel-softmax with straight-through for differentiable selection
  - Inference: hard argmax for O(1) greedy selection

Architecture:
  key   = W_k @ u          ∈ R^{B × d_ssp}
  query_m = W_q @ g_m      ∈ R^{B × M × d_ssp}
  score_m = key · query_m  ∈ R^{B × M}   (scaled dot-product)

Geometry descriptor g_m ∈ R^6:
  [anchor_x, anchor_y, anchor_z,           (FPS centroid xyz)
   mean_dist_from_cloud_centroid,           (how far from object centre)
   mean_intra_cluster_dist,                 (how spread-out the cluster is)
   normalised_point_count]                  (cluster size / (N/M))

Why this is neuromorphic-compatible:
  - W_k is applied to spike outputs of the previous temporal head step
    (binary spikes → only AC operations, no MACs).
  - G is precomputed from geometry (static, computed once per inference).
  - The dot-product score reduces to a popcount on binarised keys/queries
    on Loihi 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


GEO_DIM = 9   # upgraded from 6 → 9 (adds local density, height, eccentricity)


def compute_geometry_descriptors(pts, fps_anchors, anchor_assignments):
    """
    Precompute 9-D geometry descriptors for each FPS anchor.

    Dimensions:
      0-2  anchor_xyz           — FPS centroid position
      3    anchor_dist          — distance of anchor from cloud centroid
      4    mean_intra_dist      — mean intra-cluster spread (compactness)
      5    norm_count           — normalised point count per cluster
      6    local_density        — 1 / mean distance to 4 nearest anchors
                                  (how "crowded" this anchor region is)
      7    height_norm          — normalised y-height above cloud min-y
                                  (discriminates top/bottom/middle regions)
      8    eccentricity         — std(cluster_pts - anchor) / mean_intra_dist
                                  (elongation of cluster, ~0 for ball, >1 for slab)

    Compared to the old 6-D:
      • Local density and height help the SSP distinguish structurally
        important regions (handles, edges, top faces) from plain surfaces.
      • Eccentricity identifies flat regions (tables, floors) vs. thin
        structures (chair legs, lamp poles) — key for MN40 hard classes.
    """
    B, N, _ = pts.shape
    M = fps_anchors.size(1)
    device = pts.device

    # Cloud centroid
    cloud_centroid = pts.mean(dim=1, keepdim=True)   # [B, 1, 3]

    # Distance of each anchor from cloud centroid
    anchor_dist = (fps_anchors - cloud_centroid).norm(dim=-1)  # [B, M]

    # Intra-cluster spread, point count, eccentricity per anchor
    mean_intra   = torch.zeros(B, M, device=device)
    point_count  = torch.zeros(B, M, device=device)
    eccentricity = torch.zeros(B, M, device=device)

    for m in range(M):
        mask  = (anchor_assignments == m)          # [B, N]
        count = mask.float().sum(dim=1)            # [B]
        point_count[:, m] = count

        for b in range(B):
            cluster_pts = pts[b][mask[b]]          # [K, 3]
            if cluster_pts.size(0) > 1:
                centroid = cluster_pts.mean(dim=0)
                diffs    = cluster_pts - centroid   # [K, 3]
                d        = diffs.norm(dim=-1)       # [K]
                mu       = d.mean()
                mean_intra[b, m] = mu
                # eccentricity: std / (mean + ε) — high → elongated cluster
                eccentricity[b, m] = d.std() / (mu + 1e-6)

    # Normalised point count
    avg_count  = N / M
    norm_count = point_count / (avg_count + 1e-6)   # [B, M]

    # Local density: 1 / mean distance to nearest other anchors (k=4)
    # [B, M, M] pairwise anchor distances
    a2a = (fps_anchors.unsqueeze(2) - fps_anchors.unsqueeze(1)).norm(dim=-1)  # [B,M,M]
    # exclude self by large fill value
    inf_diag = torch.eye(M, device=device).unsqueeze(0) * 1e9
    a2a       = a2a + inf_diag
    k_nbr     = min(4, M - 1)
    nn_dists, _ = a2a.topk(k_nbr, dim=-1, largest=False)  # [B, M, k]
    local_density = 1.0 / (nn_dists.mean(dim=-1) + 1e-6)   # [B, M]

    # Normalised height (y-axis): (anchor_y - y_min) / (y_range + ε)
    y_min = pts[:, :, 1].min(dim=1, keepdim=True).values          # [B, 1]
    y_max = pts[:, :, 1].max(dim=1, keepdim=True).values          # [B, 1]
    y_range = (y_max - y_min).clamp(min=1e-6)                     # [B, 1]
    height_norm = (fps_anchors[:, :, 1] - y_min) / y_range        # [B, M]

    # Stack into 9-D descriptor [B, M, 9]
    G = torch.cat([
        fps_anchors,                               # [B, M, 3]
        anchor_dist.unsqueeze(-1),                 # [B, M, 1]
        mean_intra.unsqueeze(-1),                  # [B, M, 1]
        norm_count.unsqueeze(-1),                  # [B, M, 1]
        local_density.unsqueeze(-1),               # [B, M, 1]
        height_norm.unsqueeze(-1),                 # [B, M, 1]
        eccentricity.unsqueeze(-1),                # [B, M, 1]
    ], dim=-1)                                     # [B, M, 9]

    return G


class SliceSelectionPolicy(nn.Module):
    """
    Multi-Head Slice Selection Policy (MH-SSP).

    Upgrades over the original single-head dot-product SSP:

    1. Multi-head attention  (n_heads=4 by default)
       Each head independently attends from a different subspace of the
       belief state to a different subspace of the 9-D geometry.
       Heads specialise: one for spatial position, one for density, etc.

    2. Diversity bias
       After computing attention scores, we subtract a small diversity
       penalty proportional to the cosine similarity of each unvisited
       anchor to the set of already-visited anchor embeddings.
       This discourages the policy from re-selecting geometrically
       identical regions (e.g. all flat table surfaces), improving
       coverage of the object's structure within T steps.

    3. LayerNorm on belief input
       Stabilises training when the belief state comes from noisy early
       logits; equivalent to normalising the key before projection.

    Parameters
    ----------
    mem_dim   : int   belief state dimension (= temporal_dim of base SNN)
    geo_dim   : int   geometry descriptor dimension (default 9)
    d_ssp     : int   total projection dimension (split across n_heads)
    n_heads   : int   number of attention heads (default 4)
    diversity : float weight for the diversity penalty (default 0.1)
    """

    def __init__(self, mem_dim: int, geo_dim: int = GEO_DIM,
                 d_ssp: int = 128, n_heads: int = 4, diversity: float = 0.1):
        super().__init__()
        assert d_ssp % n_heads == 0, "d_ssp must be divisible by n_heads"
        self.d_ssp     = d_ssp
        self.n_heads   = n_heads
        self.d_head    = d_ssp // n_heads
        self.scale     = math.sqrt(self.d_head)
        self.diversity = diversity

        # Per-head projections
        self.W_k = nn.Linear(mem_dim, d_ssp, bias=False)   # belief → keys
        self.W_q = nn.Linear(geo_dim,  d_ssp, bias=False)  # geo    → queries

        # Post-aggregation score refinement (learns to weight heads)
        self.score_proj = nn.Linear(n_heads, 1, bias=True)

        # LayerNorm on belief for stability
        self.ln_belief = nn.LayerNorm(mem_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.score_proj.bias)
        nn.init.constant_(self.score_proj.weight, 1.0 / self.n_heads)

    def forward(
        self,
        mem: torch.Tensor,
        geo: torch.Tensor,
        visited_mask=None,
    ) -> torch.Tensor:
        """
        Compute multi-head slice selection scores.

        Args:
            mem          : [B, D]     belief state (LIF mem or GRU hidden)
            geo          : [B, M, G]  geometry descriptors (G=9)
            visited_mask : [B, M]     bool — True = already visited

        Returns:
            scores : [B, M]  raw logits
        """
        B, M, _ = geo.shape

        # Normalise belief before projecting (stabilises early training)
        mem_n = self.ln_belief(mem)                            # [B, D]

        # Project to multi-head keys / queries
        K = self.W_k(mem_n)                                    # [B, d_ssp]
        Q = self.W_q(geo)                                      # [B, M, d_ssp]

        # Reshape to [B, n_heads, d_head] and [B, M, n_heads, d_head]
        K = K.view(B, self.n_heads, self.d_head)               # [B, H, dh]
        Q = Q.view(B, M, self.n_heads, self.d_head)            # [B, M, H, dh]

        # Per-head dot-product: [B, M, H]
        # K: [B, H, dh] → [B, 1, H, dh] → broadcast with Q [B, M, H, dh]
        head_scores = (Q * K.unsqueeze(1)).sum(-1) / self.scale  # [B, M, H]

        # Aggregate heads with learned weights → [B, M]
        scores = self.score_proj(head_scores).squeeze(-1)         # [B, M]

        # Diversity penalty: subtract cosine similarity to visited anchors
        if self.diversity > 0 and visited_mask is not None and visited_mask.any():
            geo_flat = geo                                         # [B, M, G]
            geo_norm = F.normalize(geo_flat, dim=-1)              # [B, M, G]
            # Visited anchor mean embedding [B, G]
            vis_float = visited_mask.float()                      # [B, M]
            vis_sum   = vis_float.sum(dim=1, keepdim=True).clamp(min=1)
            vis_mean  = (geo_norm * vis_float.unsqueeze(-1)).sum(1) / vis_sum
            # Similarity of each anchor to mean visited [B, M]
            sim = (geo_norm * vis_mean.unsqueeze(1)).sum(-1)      # [B, M]
            scores = scores - self.diversity * sim

        # Mask visited anchors
        if visited_mask is not None:
            scores = scores.masked_fill(visited_mask, float("-inf"))

        return scores                                              # [B, M]

    def select_gumbel(self, scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)

    def select_greedy(self, scores: torch.Tensor) -> torch.Tensor:
        idx = scores.argmax(dim=-1)
        return F.one_hot(idx, num_classes=scores.size(-1)).float()

    def extra_repr(self) -> str:
        return (f"mem_dim→{self.W_k.in_features}, geo_dim→{self.W_q.in_features}, "
                f"d_ssp={self.d_ssp}, n_heads={self.n_heads}, "
                f"diversity={self.diversity}")

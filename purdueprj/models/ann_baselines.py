"""
ann_baselines.py
================
ANN SOTA baselines for point cloud classification on ModelNet10/40.

Implemented here (lightweight versions that run without special deps):
  DGCNN-lite   — Dynamic Graph CNN (Wang et al. 2019, arXiv 1801.07829)
                 EdgeConv on KNN graph, 3 stages, global max pool.
                 Reference acc: 92.9% MN40 (full model with 1M pts, 50 epochs).
                 Our lite version uses fewer channels for fair param comparison.

  PCT          — Point Cloud Transformer (Guo et al. 2021, arXiv 2012.09688)
                 Self-attention on KNN neighbourhood, 4-head attention, 3 blocks.
                 Reference acc: 93.2% MN40.

  PointNetPP   — PointNet++ MSG (Qi et al. 2017) simplified set-abstraction.
                 Reference acc: 90.7% MN40.

All have a `forward_full(pts)` and `forward_step(pts_slice)` interface
matching the PointNetANN / PointNetSNN conventions.

These serve two roles:
  1. Direct ANN SOTA comparison (what is the ANN ceiling on our data split?)
  2. Input to ANN→SNN conversion (ann_to_snn.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def knn_graph(pts, k):
    """
    Build KNN graph.
    pts : [B, N, 3]
    Returns idx [B, N, k], neighbours [B, N, k, 3]
    """
    B, N, _ = pts.shape
    k = min(k, N - 1)
    diff  = pts.unsqueeze(2) - pts.unsqueeze(1)   # [B, N, N, 3]
    dist2 = (diff ** 2).sum(-1)                    # [B, N, N]
    _, idx = dist2.topk(k + 1, dim=-1, largest=False)
    idx = idx[:, :, 1:]                            # [B, N, k]  exclude self
    return idx


def gather_neighbours(pts, idx):
    """
    pts : [B, N, C]
    idx : [B, N, k]
    Returns [B, N, k, C]
    """
    B, N, C = pts.shape
    k = idx.size(2)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)   # [B, N, k, C]
    pts_exp = pts.unsqueeze(2).expand(-1, -1, k, -1)    # [B, N, k, C]
    # We need to gather along dim=1 (point dimension)
    # idx_exp[b, i, j, c] = idx[b, i, j]  → gather point idx[b,i,j]
    pts_expanded = pts.unsqueeze(2).expand(B, N, k, C)
    idx_f = idx.unsqueeze(-1).expand(B, N, k, C)
    neighbours = torch.gather(
        pts.unsqueeze(2).expand(B, N, N, C),
        1,
        idx.unsqueeze(-1).expand(B, N, k, C)
    )
    return neighbours


# ---------------------------------------------------------------------------
# 1. DGCNN-lite (EdgeConv)
# ---------------------------------------------------------------------------

class EdgeConv(nn.Module):
    """
    EdgeConv block: for each point, aggregate from K nearest neighbours.
    h(x_i, x_j) = MLP([x_i, x_j - x_i])  → max pool over j
    """
    def __init__(self, in_ch, out_ch, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Linear(in_ch * 2, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Linear(out_ch, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.out_ch = out_ch

    def forward(self, x, pts=None):
        """
        x    : [B, N, in_ch]
        pts  : [B, N, 3]  for KNN (uses x if None)
        Returns [B, N, out_ch]
        """
        src = pts if pts is not None else x
        B, N, _ = x.shape
        k = min(self.k, N - 1)

        idx = knn_graph(src, k)                           # [B, N, k]
        # Gather neighbour features
        idx_exp = idx.unsqueeze(-1).expand(B, N, k, x.size(-1))
        x_exp   = x.unsqueeze(2).expand(B, N, k, x.size(-1))
        # Gather: for each (b,i,j) → x[b, idx[b,i,j], :]
        x_nb = torch.zeros(B, N, k, x.size(-1), device=x.device)
        for b in range(B):
            x_nb[b] = x[b][idx[b].reshape(-1)].reshape(N, k, x.size(-1))

        # Edge feature: [x_i, x_j - x_i]
        x_rep    = x.unsqueeze(2).expand_as(x_nb)         # [B, N, k, C]
        edge_feat = torch.cat([x_rep, x_nb - x_rep], dim=-1)  # [B, N, k, 2C]

        # MLP on edges
        BN, k2, C2 = B * N, k, edge_feat.size(-1)
        out = self.conv(edge_feat.view(B * N * k, C2))     # [BNk, out_ch]
        out = out.view(B, N, k, self.out_ch)
        out = out.max(dim=2).values                        # [B, N, out_ch]
        return out


class DGCNNLite(nn.Module):
    """
    DGCNN-lite: 3 EdgeConv stages + global max+avg pool + MLP classifier.

    Channels scaled down from original (1024→512) for fair param comparison.
    Reference (full DGCNN): 92.9% on MN40.
    """
    def __init__(self, k=20, num_classes=40, channels=(64, 128, 256)):
        super().__init__()
        self.k = k
        ch1, ch2, ch3 = channels
        emb_dim = ch1 + ch2 + ch3  # concatenated

        self.ec1 = EdgeConv(3,   ch1, k=k)
        self.ec2 = EdgeConv(ch1, ch2, k=k)
        self.ec3 = EdgeConv(ch2, ch3, k=k)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 512, bias=False),  # *2 for max+avg
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.emb_dim = emb_dim

    def _encode(self, pts):
        """pts [B, N, 3] → [B, emb_dim*2] global descriptor"""
        f1 = self.ec1(pts,  pts)   # [B, N, ch1]
        f2 = self.ec2(f1,   pts)   # [B, N, ch2]
        f3 = self.ec3(f2,   pts)   # [B, N, ch3]
        cat = torch.cat([f1, f2, f3], dim=-1)  # [B, N, emb_dim]
        g_max = cat.max(dim=1).values           # [B, emb_dim]
        g_avg = cat.mean(dim=1)                 # [B, emb_dim]
        return torch.cat([g_max, g_avg], dim=-1)

    def forward_full(self, pts):
        """pts [B, N, 3] → logits [B, num_classes]"""
        g = self._encode(pts)
        return self.mlp(g)

    def forward_step(self, pts_slice):
        """Process one slice. Stateless (ANN)."""
        g = self._encode(pts_slice)
        return self.mlp(g)

    def reset_state(self, *args, **kwargs):
        pass  # ANN has no state


# ---------------------------------------------------------------------------
# 2. PCT — Point Cloud Transformer
# ---------------------------------------------------------------------------

class OffsetAttention(nn.Module):
    """
    Offset-attention from PCT: attention on (Q - softmax(Q K^T) V) offset.
    Improves local structure capture vs standard attention.
    """
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x [B, N, dim] → [B, N, dim]"""
        B, N, C = x.shape
        H, D = self.n_heads, self.head_dim

        Q = self.q(x).view(B, N, H, D).transpose(1, 2)   # [B, H, N, D]
        K = self.k(x).view(B, N, H, D).transpose(1, 2)
        V = self.v(x).view(B, N, H, D).transpose(1, 2)

        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(D), dim=-1)
        agg  = attn @ V                                    # [B, H, N, D]
        agg  = agg.transpose(1, 2).reshape(B, N, C)

        # Offset: x - agg  (captures what attention missed)
        offset = self.out(x - agg)
        return self.norm(x + offset)


class PCTBlock(nn.Module):
    """PCT: input embedding + 4 offset-attention layers."""
    def __init__(self, in_ch=3, dim=128, n_heads=4, k=16):
        super().__init__()
        self.k = k
        # Input embedding: encode [xyz, relative_knn] → dim
        in_feat = 3 + 3 * k
        self.embed = nn.Sequential(
            nn.Linear(in_feat, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.attn1 = OffsetAttention(dim, n_heads)
        self.attn2 = OffsetAttention(dim, n_heads)
        self.attn3 = OffsetAttention(dim, n_heads)
        self.attn4 = OffsetAttention(dim, n_heads)
        self.dim = dim

    def forward(self, pts):
        """pts [B, N, 3] → [B, N, dim]"""
        B, N, _ = pts.shape
        k = min(self.k, N - 1)

        # KNN relative features
        idx = knn_graph(pts, k)                            # [B, N, k]
        # Gather neighbours
        nb = torch.zeros(B, N, k, 3, device=pts.device)
        for b in range(B):
            nb[b] = pts[b][idx[b].reshape(-1)].reshape(N, k, 3)
        rel = (nb - pts.unsqueeze(2)).reshape(B, N, k * 3)  # [B, N, k*3]
        x = torch.cat([pts, rel], dim=-1)                   # [B, N, 3+k*3]

        x = self.embed(x)     # [B, N, dim]
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.attn3(x)
        x = self.attn4(x)
        return x              # [B, N, dim]


class PCT(nn.Module):
    """
    Point Cloud Transformer (simplified).
    Reference acc: 93.2% MN40 (full model).

    Our version uses 1 PCT block (4 attention layers) + global pool + MLP.
    For the larger model variant, increase dim (default 128→256).
    """
    def __init__(self, dim=128, n_heads=4, k=16, num_classes=40):
        super().__init__()
        self.encoder = PCTBlock(in_ch=3, dim=dim, n_heads=n_heads, k=k)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _encode(self, pts):
        """pts [B, N, 3] → [B, dim*2]"""
        feat = self.encoder(pts)               # [B, N, dim]
        g_max = feat.max(dim=1).values
        g_avg = feat.mean(dim=1)
        return torch.cat([g_max, g_avg], dim=-1)

    def forward_full(self, pts):
        return self.mlp(self._encode(pts))

    def forward_step(self, pts_slice):
        return self.mlp(self._encode(pts_slice))

    def reset_state(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# 3. PointNet++ (simplified MSG set abstraction — 2 levels)
# ---------------------------------------------------------------------------

def ball_query(pts, centres, radius, K):
    """
    pts     : [B, N, 3]
    centres : [B, M, 3]
    Returns idx [B, M, K] of points within radius (padded with first index)
    """
    B, N, _ = pts.shape
    M = centres.size(1)
    diff  = pts.unsqueeze(1) - centres.unsqueeze(2)   # [B, M, N, 3]
    dist2 = (diff ** 2).sum(-1)                        # [B, M, N]
    # For each centre, find points within radius
    mask  = dist2 < radius ** 2                        # [B, M, N]
    # Take top K closest within radius
    dist2[~mask] = 1e10
    _, idx = dist2.topk(K, dim=-1, largest=False)      # [B, M, K]
    return idx


def farthest_point_sample(pts, n_samples):
    """pts [N, 3] → indices [n_samples]"""
    N = pts.size(0)
    selected = torch.zeros(n_samples, dtype=torch.long, device=pts.device)
    distances = torch.full((N,), float("inf"), device=pts.device)
    farthest = torch.randint(0, N, (1,), device=pts.device).item()
    for i in range(n_samples):
        selected[i] = farthest
        centroid = pts[farthest]
        dist = ((pts - centroid) ** 2).sum(-1)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax().item()
    return selected


class SetAbstraction(nn.Module):
    """Single-scale set abstraction (PointNet++ §3.2)."""
    def __init__(self, n_centres, radius, K, in_ch, out_ch):
        super().__init__()
        self.n_centres = n_centres
        self.radius    = radius
        self.K         = K
        self.mlp = nn.Sequential(
            nn.Linear(in_ch + 3, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.out_ch = out_ch

    def forward(self, pts, feats=None):
        """
        pts   : [B, N, 3]
        feats : [B, N, C] or None
        Returns centres [B, M, 3], new_feats [B, M, out_ch]
        """
        B, N, _ = pts.shape
        M = min(self.n_centres, N)

        # FPS to select centres
        centres_list = []
        for b in range(B):
            idx = farthest_point_sample(pts[b], M)
            centres_list.append(pts[b][idx])
        centres = torch.stack(centres_list, dim=0)     # [B, M, 3]

        # Ball query
        K = min(self.K, N)
        idx = ball_query(pts, centres, self.radius, K)  # [B, M, K]

        # Gather points in each ball
        grouped_pts = torch.zeros(B, M, K, 3, device=pts.device)
        for b in range(B):
            grouped_pts[b] = pts[b][idx[b].reshape(-1)].reshape(M, K, 3)

        # Relative coordinates
        grouped_pts = grouped_pts - centres.unsqueeze(2)  # [B, M, K, 3]

        if feats is not None:
            grouped_feats = torch.zeros(B, M, K, feats.size(-1), device=pts.device)
            for b in range(B):
                grouped_feats[b] = feats[b][idx[b].reshape(-1)].reshape(M, K, -1)
            grouped = torch.cat([grouped_pts, grouped_feats], dim=-1)
        else:
            grouped = grouped_pts

        # MLP + max pool over K
        BM, K2, Cin = B * M, K, grouped.size(-1)
        out = self.mlp(grouped.reshape(BM * K2, Cin))   # [BMK, out_ch]
        out = out.reshape(B, M, K, self.out_ch)
        out = out.max(dim=2).values                      # [B, M, out_ch]

        return centres, out


class PointNetPP(nn.Module):
    """
    PointNet++ (2-level MSG, simplified single-scale).
    Reference acc: 90.7% MN40.
    """
    def __init__(self, num_classes=40):
        super().__init__()
        self.sa1 = SetAbstraction(n_centres=512, radius=0.2, K=32,
                                   in_ch=0,   out_ch=64)
        self.sa2 = SetAbstraction(n_centres=128, radius=0.4, K=64,
                                   in_ch=64,  out_ch=128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def _encode(self, pts):
        c1, f1 = self.sa1(pts)               # [B, 512, 3], [B, 512, 64]
        c2, f2 = self.sa2(c1, f1)            # [B, 128, 3], [B, 128, 128]
        return f2.max(dim=1).values           # [B, 128]

    def forward_full(self, pts):
        return self.mlp(self._encode(pts))

    def forward_step(self, pts_slice):
        return self.mlp(self._encode(pts_slice))

    def reset_state(self, *args, **kwargs):
        pass

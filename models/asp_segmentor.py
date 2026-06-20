"""ASP segmentor for ShapeNetPart and S3DIS training scripts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn_layers import LIFRecurrentStack
from models.ssp import SSP


def _head_count(hidden_dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    return requested if hidden_dim % requested == 0 else 1


class ASPSegmentor(nn.Module):
    """Slice encoder with SSP-guided context and per-point logits."""

    def __init__(self, cfg):
        super().__init__()
        in_channels = int(getattr(cfg, "in_channels", 6))
        hidden_dim = int(getattr(cfg, "hidden_dim", getattr(cfg, "feat_dim", 512)))
        geo_dim = int(getattr(cfg, "geo_dim", 8))
        num_classes = int(getattr(cfg, "num_classes", 50))
        heads = _head_count(hidden_dim, getattr(cfg, "transformer_heads", 4))

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_category = bool(getattr(cfg, "use_category", False))
        self.T = int(getattr(cfg, "T", 6))
        self.slice_pool = str(getattr(cfg, "slice_pool", "meanmax")).lower()
        self.context_ensemble = max(1, int(getattr(cfg, "context_ensemble", 1)))
        self.temporal_backend = str(
            getattr(cfg, "temporal_backend", "gru")
        ).lower()
        self.register_buffer("gumbel_tau", torch.tensor(1.0))
        self.initial_belief = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.initial_belief, std=0.02)

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        if self.slice_pool == "meanmax":
            self.slice_pool_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        elif self.slice_pool in {"mean", "max"}:
            self.slice_pool_proj = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported slice_pool={self.slice_pool!r}; "
                "expected 'mean', 'max', or 'meanmax'"
            )
        self.pos_proj = nn.Linear(geo_dim, hidden_dim)
        self.slice_token_norm = nn.LayerNorm(hidden_dim)
        self.slice_token_dropout = nn.Dropout(
            float(getattr(cfg, "slice_token_dropout", 0.0))
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=int(getattr(cfg, "transformer_ffn_dim", hidden_dim * 2)),
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.slice_transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.ssp = SSP(
            belief_dim=hidden_dim,
            geo_dim=geo_dim,
            d_ssp=int(getattr(cfg, "d_ssp", 128)),
        )
        if self.temporal_backend == "lif":
            self.temporal = LIFRecurrentStack(
                hidden_dim,
                num_layers=int(getattr(cfg, "num_lif_layers", 1)),
                leak=float(getattr(cfg, "lif_leak", 0.9)),
                threshold=float(getattr(cfg, "lif_threshold", 1.0)),
            )
        elif self.temporal_backend == "gru":
            self.temporal = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            raise ValueError(
                f"Unsupported temporal_backend={self.temporal_backend!r}; "
                "expected 'lif' or 'gru'"
            )

        num_categories = int(getattr(cfg, "num_categories", 0))
        self.num_categories = num_categories
        self.category_embed = (
            nn.Embedding(max(num_categories, 1), hidden_dim)
            if self.use_category
            else None
        )
        point_in_channels = int(
            getattr(cfg, "point_in_channels", in_channels)
        )
        point_feat_dim = int(getattr(cfg, "point_feat_dim", 128))
        self.point_in_channels = point_in_channels
        self.point_encoder = nn.Sequential(
            nn.Linear(point_in_channels + 3, point_feat_dim),
            nn.LayerNorm(point_feat_dim),
            nn.GELU(),
            nn.Linear(point_feat_dim, point_feat_dim),
            nn.GELU(),
        )
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + point_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(getattr(cfg, "seg_head_dropout", 0.1))),
            nn.Linear(hidden_dim, num_classes),
        )
        self.last_selection_entropy = None
        self.last_selection_coverage = None
        self.last_spike_rate = None

    def _pool_slice_points(self, x: torch.Tensor) -> torch.Tensor:
        if self.slice_pool == "mean":
            return x.mean(dim=2)
        if self.slice_pool == "max":
            return x.max(dim=2).values
        mean = x.mean(dim=2)
        maxv = x.max(dim=2).values
        return self.slice_pool_proj(torch.cat([mean, maxv], dim=-1))

    def _encode_slices(self, slices: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        bsz, n_slices, pts_per_slice, channels = slices.shape
        x = slices.reshape(bsz * n_slices * pts_per_slice, channels)
        x = self.feature_extractor(x)
        x = x.reshape(bsz, n_slices, pts_per_slice, self.hidden_dim)
        x = self._pool_slice_points(x)
        x = self.slice_token_norm(x + self.pos_proj(geo))
        x = self.slice_token_dropout(x)
        return self.slice_transformer(x)

    def _active_context(
        self,
        slice_feats: torch.Tensor,
        geo: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bsz, n_slices, _ = slice_feats.shape
        steps = min(n_slices, max(1, int(getattr(self, "T", n_slices))))
        device = slice_feats.device
        belief = self.initial_belief.unsqueeze(0).expand(bsz, -1)
        visited = torch.zeros(bsz, n_slices, dtype=torch.bool, device=device)
        states = []
        membranes = None
        entropies = []
        spike_rates = []

        for _ in range(steps):
            scores = self.ssp(belief, geo, visited.clone())
            probs = scores.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropies.append(entropy / max(1.0, math.log(n_slices)))
            if training:
                weights = F.gumbel_softmax(
                    scores, tau=float(self.gumbel_tau.item()), hard=True, dim=-1
                )
            else:
                idx = scores.argmax(dim=-1)
                weights = F.one_hot(idx, num_classes=n_slices).float()
            selected = weights.argmax(dim=-1)
            visited.scatter_(1, selected.unsqueeze(1), True)
            chosen = (weights.unsqueeze(-1) * slice_feats).sum(dim=1)
            if self.temporal_backend == "lif":
                belief, membranes, spikes = self.temporal.forward_step(
                    chosen, membranes
                )
                spike_rates.append(
                    torch.stack([spike.float().mean() for spike in spikes]).mean()
                )
            else:
                belief = self.temporal(chosen, belief)
            states.append(belief)

        self.last_selection_entropy = torch.stack(entropies).mean().detach()
        self.last_selection_coverage = visited.float().mean().detach()
        self.last_spike_rate = (
            torch.stack(spike_rates).mean().detach() if spike_rates else None
        )
        return belief, states

    def forward(
        self,
        slices: torch.Tensor,
        geo: torch.Tensor,
        sid_arr: torch.Tensor,
        cat_ids: torch.Tensor,
        pts_features: torch.Tensor,
        training: bool = True,
        ):
        slice_feats = self._encode_slices(slices, geo)
        global_ctx, states = self._active_context(slice_feats, geo, training)
        if self.context_ensemble > 1 and states:
            global_ctx = torch.stack(states[-self.context_ensemble :], dim=0).mean(dim=0)

        bsz, n_points = sid_arr.shape
        if pts_features.size(-1) != self.point_in_channels:
            raise ValueError(
                f"Expected {self.point_in_channels} per-point channels, got "
                f"{pts_features.size(-1)}"
            )
        idx = sid_arr.clamp(0, slice_feats.size(1) - 1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        point_ctx = slice_feats.gather(1, gather_idx)
        centroid_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
        slice_centroids = geo[..., :3].gather(1, centroid_idx)
        relative_xyz = pts_features[..., :3] - slice_centroids
        point_features = self.point_encoder(
            torch.cat([pts_features, relative_xyz], dim=-1)
        )

        global_ctx = global_ctx.unsqueeze(1).expand(-1, n_points, -1)
        context = point_ctx + global_ctx
        if self.category_embed is not None:
            cat_ctx = self.category_embed(
                cat_ids.long().clamp(0, max(0, self.num_categories - 1))
            )
            context = context + cat_ctx.unsqueeze(1)

        logits = self.point_head(
            torch.cat([point_ctx, context, point_features], dim=-1)
        )
        return logits, states

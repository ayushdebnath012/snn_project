"""ASP classifier used by ScanObjectNN training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssp import SSP


def _head_count(hidden_dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    return requested if hidden_dim % requested == 0 else 1


class ASPClassifier(nn.Module):
    """Slice encoder + SSP-guided temporal classifier."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_channels = int(getattr(cfg, "in_channels", 6))
        hidden_dim = int(getattr(cfg, "hidden_dim", getattr(cfg, "feat_dim", 512)))
        geo_dim = int(getattr(cfg, "geo_dim", 8))
        num_classes = int(getattr(cfg, "num_classes", 15))
        heads = _head_count(hidden_dim, getattr(cfg, "transformer_heads", 4))

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.T = int(getattr(cfg, "T", 6))
        self.slice_pool = str(getattr(cfg, "slice_pool", "meanmax")).lower()
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
        self.temporal = nn.GRUCell(hidden_dim, hidden_dim)

        head_dims = list(getattr(cfg, "cls_head_dims", [256, 128]))
        dropouts = list(getattr(cfg, "cls_head_dropout", [0.3, 0.2]))
        layers = []
        prev = hidden_dim
        for i, dim in enumerate(head_dims):
            layers.extend(
                [
                    nn.Linear(prev, int(dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropouts[i] if i < len(dropouts) else 0.0)),
                ]
            )
            prev = int(dim)
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def aux_weights(t_steps: int) -> list[float]:
        if t_steps <= 1:
            return [1.0]
        return [0.5 + 0.5 * ((i + 1) / t_steps) for i in range(t_steps)]

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

    def forward(self, slices: torch.Tensor, geo: torch.Tensor, training: bool = True):
        slice_feats = self._encode_slices(slices, geo)
        bsz, n_slices, _ = slice_feats.shape
        steps = min(int(getattr(self, "T", n_slices)), n_slices)
        device = slices.device

        belief = self.initial_belief.unsqueeze(0).expand(bsz, -1)
        visited = torch.zeros(bsz, n_slices, dtype=torch.bool, device=device)
        logits_all = []

        for _ in range(steps):
            scores = self.ssp(belief, geo, visited.clone())
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
            belief = self.temporal(chosen, belief)
            logits_all.append(self.classifier(belief))

        return logits_all

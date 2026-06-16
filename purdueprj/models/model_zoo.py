"""
model_zoo.py
============
Unified model registry for comparing all SNN architectures on ModelNet40.

Registered models:
  --- Our SNN models (scaled) ---
  "ours_base"      — PointNetSNN (standard LIF, radial slicing)
  "ours_learnable" — PointNetSNN (learnable tau+vth, radial slicing)
  "ours_knn"       — PointNetSNN (KNN backbone, learnable LIF)
  "ours_bidir"     — PointNetSNN (bidirectional temporal, learnable LIF)
  "ours_full"      — PointNetSNN (KNN + bidirectional + learnable LIF)
  "ours_large"     — ours_full with wider channels (256→512→1024)
  "ours_xl"        — ours_full with deeper channels (256→512→1024→1024)
  "ours_pct_snn"   — PCT-style spiking transformer (attention + LIF)

  --- Compared SNN papers ---
  "e3dsnn"         — E3DBackbone (SVC + SSC + I-LIF, from 2412.07360)
  "spiking_ssm"    — SpikingSSM temporal (from 2408.14909)
  "spt"            — SPT spiking transformer (from 2502.15811)

  --- ANN baselines for SOTA comparison ---
  "ann_pointnet"   — PointNetANN (our ANN counterpart)
  "ann_dgcnn"      — DGCNN-lite (EdgeConv, ref 92.9% MN40)
  "ann_pct"        — Point Cloud Transformer (ref 93.2% MN40)
  "ann_pointnetpp" — PointNet++ simplified (ref 90.7% MN40)

Usage:
  from models.model_zoo import build_model, MODEL_CONFIGS, count_params

  model = build_model("ours_full", num_classes=40)
  print(count_params(model))
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def count_params(model):
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Wrapper to give E3DBackbone the forward_step / reset_state interface
# ---------------------------------------------------------------------------

class E3DSNNModel(nn.Module):
    """
    E3DBackbone wrapped with a temporal LIF accumulator and linear classifier,
    giving it the same forward_step / reset_state interface as PointNetSNN.
    """
    def __init__(self, grid_size=8, hidden_ch=64, out_dim=256,
                 num_classes=40, D=4, tau=0.9):
        super().__init__()
        from models.e3dsnn_backbone import E3DBackbone
        from models.neuron_zoo import tri_spike

        self.backbone  = E3DBackbone(grid_size=grid_size, hidden_ch=hidden_ch,
                                     out_dim=out_dim, D=D, tau=tau)
        self.tau = tau
        self.vth = 1.0
        self.fc  = nn.Linear(out_dim, num_classes)
        self.out_dim = out_dim
        self._tri_spike = tri_spike
        self.register_buffer("mem", None, persistent=False)

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.backbone.reset_state(batch_size, dev)
        self.mem = torch.zeros(batch_size, self.out_dim, device=dev)

    def forward_step(self, pts):
        """pts: [B, N, 3] → logits [B, num_classes]"""
        feat = self.backbone(pts)                     # [B, out_dim]
        if self.mem is None:
            self.reset_state(pts.size(0), pts.device)
        # Detach top-level membrane state (TBPTT) to match the block-level
        # detach in E3DBlock.forward and prevent "backward a second time" errors.
        self.mem = self.tau * self.mem.detach() + feat
        spk = self._tri_spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        return self.fc(spk)

    def forward(self, pts_slices):
        """pts_slices: [B, T, N, 3] → logits [B, num_classes]"""
        B, T, N, _ = pts_slices.shape
        self.reset_state(B, pts_slices.device)
        for t in range(T):
            logits = self.forward_step(pts_slices[:, t])
        return logits


# ---------------------------------------------------------------------------
# Wrapper to give SpikingSSMTemporal the full model interface
# ---------------------------------------------------------------------------

class SpikingSSMModel(nn.Module):
    """
    PointNetBackbone + SpikingSSMTemporal as a full model.
    Same interface as PointNetSNN.
    """
    def __init__(self, point_dims=(128, 256, 512), d_state=16,
                 num_classes=40, tau=0.9):
        super().__init__()
        from models.pointnet_backbone import PointNetBackbone
        from models.spiking_ssm import SpikingSSMTemporal

        self.backbone = PointNetBackbone(hidden_dims=list(point_dims))
        temporal_dim  = point_dims[-1]
        self.temporal = SpikingSSMTemporal(
            in_dim=temporal_dim, hidden_dim=temporal_dim,
            num_classes=num_classes, d_state=d_state, tau=tau
        )

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.temporal.fc.parameters()).device
        self.backbone.reset_state(batch_size, dev)
        self.temporal.reset_state(batch_size, dev)

    def forward_step(self, pts_slice):
        feat = self.backbone(pts_slice).mean(dim=1)   # [B, D]
        return self.temporal.forward_step(feat)

    def forward(self, pts_slices):
        B, T, N, _ = pts_slices.shape
        self.reset_state(B, pts_slices.device)
        for t in range(T):
            logits = self.forward_step(pts_slices[:, t])
        return logits


# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------

def build_model(name, num_classes=40, **kwargs):
    """
    Build a model by name from the registry.

    Args:
      name        : one of the keys in MODEL_CONFIGS
      num_classes : 10 for ModelNet10, 40 for ModelNet40
      **kwargs    : override any default config values

    Returns:
      nn.Module with reset_state() and forward_step() interface
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS.keys())}")

    cfg = {**MODEL_CONFIGS[name]["defaults"], **kwargs}
    builder = MODEL_CONFIGS[name]["builder"]
    return builder(num_classes=num_classes, **cfg)


def _build_ours(num_classes, learnable_lif=False, local_knn=False,
                knn_k=16, bidirectional=False,
                point_dims=(128, 256, 512), temporal_dim=512):
    from models.pointnet_snn import PointNetSNN
    return PointNetSNN(
        point_dims=list(point_dims),
        temporal_dim=temporal_dim,
        num_classes=num_classes,
        learnable_lif=learnable_lif,
        local_knn=local_knn,
        knn_k=knn_k,
        bidirectional=bidirectional,
    )


def _build_e3dsnn(num_classes, grid_size=8, hidden_ch=64, out_dim=256,
                  D=4, tau=0.9):
    return E3DSNNModel(grid_size=grid_size, hidden_ch=hidden_ch,
                       out_dim=out_dim, num_classes=num_classes, D=D, tau=tau)


def _build_spiking_ssm(num_classes, point_dims=(128, 256, 512),
                        d_state=16, tau=0.9):
    return SpikingSSMModel(point_dims=point_dims, d_state=d_state,
                           num_classes=num_classes, tau=tau)


def _build_spt(num_classes, hidden_ch=64, out_dim=256, k=16, tau=0.9, vth=1.0):
    from models.spt_block import SPTModel
    return SPTModel(hidden_ch=hidden_ch, out_dim=out_dim,
                    num_classes=num_classes, k=k, tau=tau, vth=vth)


def _build_spm(num_classes, point_dims=(128, 256, 512), d_state=16,
               tau=0.9, n_smb_layers=2, local_knn=True, knn_k=16,
               learnable_lif=True, pooling="mean"):
    from models.spiking_mamba import SPMModel
    return SPMModel(num_classes=num_classes, point_dims=list(point_dims),
                    d_state=d_state, tau=tau, n_smb_layers=n_smb_layers,
                    local_knn=local_knn, knn_k=knn_k,
                    learnable_lif=learnable_lif, pooling=pooling)


def _build_asp_spm(num_classes, point_dims=(128, 256, 512), d_ssp=64,
                   d_state=16, tau=0.9, n_smb_layers=2,
                   local_knn=True, knn_k=16, learnable_lif=True,
                   pooling="mean"):
    """ASP wrapper around SPMModel: SSP-guided adaptive slice ordering + early exit."""
    from models.spiking_mamba import SPMModel
    from models.asp_wrapper import ASPWrapper
    base = SPMModel(
        num_classes=num_classes, point_dims=list(point_dims),
        d_state=d_state, tau=tau, n_smb_layers=n_smb_layers,
        local_knn=local_knn, knn_k=knn_k, learnable_lif=learnable_lif,
        pooling=pooling,
    )
    return ASPWrapper(base, feat_dim=point_dims[-1],
                      num_classes=num_classes, d_ssp=d_ssp)


def _build_asp_spn(num_classes, point_dims=(128, 256, 512), d_ssp=64,
                   knn_k=16, learnable_lif=True, local_knn=True):
    """ASP wrapper around PointNetSNN (SpikingPointNet [8]): SSP + early exit."""
    from models.pointnet_snn import PointNetSNN
    from models.asp_wrapper import ASPWrapper
    base = PointNetSNN(
        num_classes=num_classes, point_dims=list(point_dims),
        temporal_dim=point_dims[-1], local_knn=True,
        learnable_lif=learnable_lif, knn_k=knn_k,
        bidirectional=False,   # causal only — SSP provides adaptive context
    )
    return ASPWrapper(base, feat_dim=point_dims[-1],
                      num_classes=num_classes, d_ssp=d_ssp)


def _build_asp_foveater_imagenet(num_classes, image_size=224, feature_grid=14,
                                 embed_dim=192, depth=9, num_heads=3,
                                 max_fixations=5, max_tokens=29,
                                 dropout=0.0):
    """FoveaTer-style ASP model for ImageNet images."""
    from models.foveater_asp import FoveaTerASP
    return FoveaTerASP(
        num_classes=num_classes,
        image_size=image_size,
        feature_grid=feature_grid,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        max_fixations=max_fixations,
        max_tokens=max_tokens,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Scaled model builders
# ---------------------------------------------------------------------------

def _build_ours_large(num_classes, knn_k=16, tau=0.9):
    """Wider channels: 256→512→1024. ~4× params vs ours_full."""
    from models.pointnet_snn import PointNetSNN
    return PointNetSNN(
        point_dims=[256, 512, 1024],
        temporal_dim=1024,
        num_classes=num_classes,
        learnable_lif=True,
        local_knn=True,
        knn_k=knn_k,
        bidirectional=True,
    )


def _build_ours_xl(num_classes, knn_k=16, tau=0.9):
    """Extra-wide + extra-deep backbone: 256→512→1024→1024. ~6× params."""
    from models.pointnet_snn import PointNetSNN
    return PointNetSNN(
        point_dims=[256, 512, 1024, 1024],
        temporal_dim=1024,
        num_classes=num_classes,
        learnable_lif=True,
        local_knn=True,
        knn_k=knn_k,
        bidirectional=True,
    )


class PCTSNNModel(nn.Module):
    """
    PCT-style spiking transformer:
      PCT encoder (offset-attention, dim=256) → LIF temporal integration → classifier.

    Replaces the PointNet MLP backbone with a transformer encoder.
    Each slice is encoded by 4 offset-attention layers, then integrated
    via LIF neurons over time slices.
    """
    def __init__(self, dim=256, n_heads=4, k=16, num_classes=40, tau=0.9):
        super().__init__()
        from models.ann_baselines import PCTBlock
        from models.neuron_zoo import tri_spike

        self.encoder = PCTBlock(in_ch=3, dim=dim, n_heads=n_heads, k=k)
        self.tau = tau
        self.vth = 1.0
        self._spike = tri_spike

        self.fc1 = nn.Linear(dim * 2, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, num_classes)
        self.dim = dim

        self.register_buffer("mem", None, persistent=False)

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc1.parameters()).device
        self.encoder.reset_state(batch_size, dev) if hasattr(self.encoder, 'reset_state') else None
        self.mem = torch.zeros(batch_size, self.dim, device=dev)

    def forward_step(self, pts):
        """pts [B, N, 3] → logits [B, num_classes]"""
        feat = self.encoder(pts)                     # [B, N, dim]
        g = torch.cat([feat.max(1).values,
                        feat.mean(1)], dim=-1)       # [B, dim*2]
        h = torch.relu(self.bn1(self.fc1(g)))        # [B, dim]

        if self.mem is None:
            self.reset_state(pts.size(0), pts.device)
        self.mem = self.tau * self.mem + h
        spk = self._spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        return self.fc2(spk)

    def forward(self, pts_slices):
        B, T, N, _ = pts_slices.shape
        self.reset_state(B, pts_slices.device)
        for t in range(T):
            logits = self.forward_step(pts_slices[:, t])
        return logits


def _build_pct_snn(num_classes, dim=256, n_heads=4, k=16, tau=0.9):
    return PCTSNNModel(dim=dim, n_heads=n_heads, k=k,
                       num_classes=num_classes, tau=tau)


# ---------------------------------------------------------------------------
# ANN baseline builders
# ---------------------------------------------------------------------------

def _build_ann_pointnet(num_classes, point_dims=(128, 256, 512), temporal_dim=512):
    from models.pointnet_ann import PointNetANN
    return PointNetANN(point_dims=list(point_dims),
                       temporal_dim=temporal_dim,
                       num_classes=num_classes)


def _build_ann_dgcnn(num_classes, k=20, channels=(64, 128, 256)):
    from models.ann_baselines import DGCNNLite
    return DGCNNLite(k=k, num_classes=num_classes, channels=channels)


def _build_ann_pct(num_classes, dim=128, n_heads=4, k=16):
    from models.ann_baselines import PCT
    return PCT(dim=dim, n_heads=n_heads, k=k, num_classes=num_classes)


def _build_ann_pointnetpp(num_classes):
    from models.ann_baselines import PointNetPP
    return PointNetPP(num_classes=num_classes)


MODEL_CONFIGS = {
    # -----------------------------------------------------------------------
    # Our models (PointNetSNN variants)
    # -----------------------------------------------------------------------
    "ours_base": {
        "builder": _build_ours,
        "defaults": {"learnable_lif": False, "local_knn": False, "bidirectional": False},
        "description": "Baseline PointNetSNN (fixed LIF, radial slicing)",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_learnable": {
        "builder": _build_ours,
        "defaults": {"learnable_lif": True, "local_knn": False, "bidirectional": False},
        "description": "PointNetSNN + learnable tau/vth per neuron",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_knn": {
        "builder": _build_ours,
        "defaults": {"learnable_lif": True, "local_knn": True, "knn_k": 16, "bidirectional": False},
        "description": "PointNetSNN + KNN backbone (like SPM SEL) + learnable LIF",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_bidir": {
        "builder": _build_ours,
        "defaults": {"learnable_lif": True, "local_knn": False, "bidirectional": True},
        "description": "PointNetSNN + bidirectional temporal (like SPM Time Flip) + learnable LIF",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_full": {
        "builder": _build_ours,
        "defaults": {"learnable_lif": True, "local_knn": True, "knn_k": 16, "bidirectional": True},
        "description": "Full model: KNN + bidirectional + learnable LIF",
        "paper": "Ours",
        "type": "SNN",
    },
    # -----------------------------------------------------------------------
    # Scaled variants of our model (for scaling ablation)
    # -----------------------------------------------------------------------
    "ours_large": {
        "builder": _build_ours_large,
        "defaults": {"knn_k": 16},
        "description": "ours_full scaled: channels 256→512→1024 (~4× params)",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_xl": {
        "builder": _build_ours_xl,
        "defaults": {"knn_k": 16},
        "description": "ours_full XL: channels 256→512→1024→1024 (~6× params)",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_pct_snn": {
        "builder": _build_pct_snn,
        "defaults": {"dim": 256, "n_heads": 4, "k": 16},
        "description": "PCT-style spiking transformer backbone + LIF temporal",
        "paper": "Ours",
        "type": "SNN",
    },
    "ours_transformer_small": {
        "builder": _build_pct_snn,
        "defaults": {"dim": 128, "n_heads": 4, "k": 16},
        "description": "Small PCT-style spiking transformer (dim=128)",
        "paper": "Ours",
        "type": "SNN",
    },
    # -----------------------------------------------------------------------
    # ANN baselines (for SOTA comparison)
    # -----------------------------------------------------------------------
    "ann_pointnet": {
        "builder": _build_ann_pointnet,
        "defaults": {},
        "description": "PointNetANN — our architecture's ANN counterpart",
        "paper": "Ours (ANN)",
        "type": "ANN",
    },
    "ann_dgcnn": {
        "builder": _build_ann_dgcnn,
        "defaults": {"k": 20, "channels": (64, 128, 256)},
        "description": "DGCNN-lite (EdgeConv, ref 92.9% MN40)",
        "paper": "DGCNN 2019",
        "type": "ANN",
    },
    "ann_pct": {
        "builder": _build_ann_pct,
        "defaults": {"dim": 128, "n_heads": 4, "k": 16},
        "description": "Point Cloud Transformer (ref 93.2% MN40)",
        "paper": "PCT 2021",
        "type": "ANN",
    },
    "ann_pointnetpp": {
        "builder": _build_ann_pointnetpp,
        "defaults": {},
        "description": "PointNet++ simplified set abstraction (ref 90.7% MN40)",
        "paper": "PointNet++ 2017",
        "type": "ANN",
    },
    # -----------------------------------------------------------------------
    # E-3DSNN (arXiv 2412.07360)
    # -----------------------------------------------------------------------
    "e3dsnn": {
        "builder": _build_e3dsnn,
        "defaults": {"grid_size": 8, "hidden_ch": 64, "out_dim": 256, "D": 4, "tau": 0.9},
        "description": "E-3DSNN: Spike Voxel Coding + Sparse Conv + I-LIF",
        "paper": "E-3DSNN (2412.07360)",
        "type": "SNN",
    },
    # -----------------------------------------------------------------------
    # SpikingSSMs (arXiv 2408.14909)
    # -----------------------------------------------------------------------
    "spiking_ssm": {
        "builder": _build_spiking_ssm,
        "defaults": {"point_dims": (128, 256, 512), "d_state": 16, "tau": 0.9},
        "description": "SpikingSSM: SSM state-space + LIF spike gate (diagonal S4D)",
        "paper": "SpikingSSMs (2408.14909)",
        "type": "SNN",
    },
    # -----------------------------------------------------------------------
    # SPT (arXiv 2502.15811)
    # -----------------------------------------------------------------------
    "spt": {
        "builder": _build_spt,
        "defaults": {"hidden_ch": 64, "out_dim": 256, "k": 16, "tau": 0.9, "vth": 1.0},
        "description": "SPT: Q-SDE + spiking KNN attention + HD-IF neurons",
        "paper": "SPT (2502.15811)",
        "type": "SNN",
    },
    # -----------------------------------------------------------------------
    # SPM: Spiking Point Mamba (arXiv 2504.14371)
    # -----------------------------------------------------------------------
    "spm": {
        "builder": _build_spm,
        "defaults": {"local_knn": True, "knn_k": 16, "learnable_lif": True},
        "description": "Spiking Point Mamba (SPM) with HDE and SMB",
        "paper": "SPM (2504.14371)",
        "type": "SNN",
    },
    # -----------------------------------------------------------------------
    # ASP plug-in variants (Ours — this work)
    # ASP wraps any temporal SNN with SSP-guided adaptive slice ordering +
    # early exit.  Pareto-dominates fixed-order baselines at matched accuracy.
    # -----------------------------------------------------------------------
    "asp_spm": {
        "builder": _build_asp_spm,
        "defaults": {"local_knn": True, "knn_k": 16, "learnable_lif": True,
                     "d_ssp": 64},
        "description": "ASP + SPM: SSP-guided slice ordering over SPMModel (HDE+SMB)",
        "paper": "Ours",
        "type": "SNN",
    },
    "asp_spm_accuracy": {
        "builder": _build_asp_spm,
        "defaults": {"point_dims": (256, 512, 1024), "local_knn": True,
                     "knn_k": 20, "learnable_lif": False, "d_ssp": 128,
                     "d_state": 16, "n_smb_layers": 2, "pooling": "meanmax"},
        "description": "Accuracy-first ASP + SPM: wider SPM backbone, larger SSP, full-slice eval",
        "paper": "Ours",
        "type": "SNN",
    },
    "asp_spn": {
        "builder": _build_asp_spn,
        "defaults": {"local_knn": True, "knn_k": 16, "learnable_lif": True,
                     "d_ssp": 64},
        "description": "ASP + SpikingPointNet [8]: SSP-guided slice ordering over PointNetSNN",
        "paper": "Ours",
        "type": "SNN",
    },
    "asp_foveater_imagenet": {
        "builder": _build_asp_foveater_imagenet,
        "defaults": {"image_size": 224, "feature_grid": 14, "embed_dim": 192,
                     "depth": 9, "num_heads": 3, "max_fixations": 5,
                     "max_tokens": 29},
        "description": "ASP + FoveaTer: foveated transformer fixations for ImageNet",
        "paper": "FoveaTer (ICLR 2022 submission)",
        "type": "ImageNet",
    },
}


# ---------------------------------------------------------------------------
# Published baselines (for comparison table — no code needed)
# ---------------------------------------------------------------------------

PUBLISHED_RESULTS = {
    # ANNs
    "PointNet":      {"type": "ANN", "mn40": 89.2, "mn10": None, "paper": "2017"},
    "PointNet++":    {"type": "ANN", "mn40": 90.7, "mn10": None, "paper": "2017"},
    "PointMLP":      {"type": "ANN", "mn40": 94.1, "mn10": None, "paper": "2022"},
    "PointMamba":    {"type": "ANN", "mn40": 92.4, "mn10": None, "paper": "2024"},
    # SNNs
    "SpikingPtNet":  {"type": "SNN", "mn40": 88.2, "mn10": None, "paper": "2310.06232"},
    "P2SResLNet-B":  {"type": "SNN", "mn40": 88.7, "mn10": None, "paper": "2023"},
    "SPT":           {"type": "SNN", "mn40": 91.4, "mn10": None, "paper": "2502.15811"},
    "SPM":           {"type": "SNN", "mn40": 92.3, "mn10": None, "paper": "2504.14371"},
}


def print_model_summary():
    """Print parameter counts and descriptions for all registered models."""
    print(f"\n{'Model':<20} {'#Params':>10}  Description")
    print("-" * 80)
    for name, cfg in MODEL_CONFIGS.items():
        try:
            model = build_model(name, num_classes=40)
            n = count_params(model)
            desc = cfg["description"]
            print(f"{name:<20} {n:>10,}  {desc}")
        except Exception as e:
            print(f"{name:<20} {'ERROR':>10}  {e}")
    print()


if __name__ == "__main__":
    print_model_summary()

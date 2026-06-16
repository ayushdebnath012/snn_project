import torch
import torch.nn as nn
from models.snn_layers import LIFLayer, LearnableLIFLayer, BNLIFLayer


class TemporalSNN(nn.Module):
    """
    Original temporal SNN accumulator.
    Fixed bug: num_classes was hardcoded to 10 — now a parameter.
    Works for both ModelNet10 and ModelNet40.
    """
    def __init__(self, dim=256, num_classes=10, learnable_lif=False, use_bn=False):
        super().__init__()
        if use_bn:
            LayerCls = BNLIFLayer
        elif learnable_lif:
            LayerCls = LearnableLIFLayer
        else:
            LayerCls = LIFLayer
        self.lif1 = LayerCls(dim, dim)
        self.lif2 = LayerCls(dim, dim)
        self.fc = nn.Linear(dim, num_classes)

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


class BidirectionalTemporalSNN(nn.Module):
    """
    Novel: Bidirectional Temporal SNN.

    Inspired by SPM's Spiking Mamba Block (SMB) 'Time Flip' strategy:
    process the slice sequence both forward and backward, then fuse.

    Forward pass  : slices t=0..T-1   (causal, sees past context)
    Backward pass : slices t=T-1..0   (anti-causal, sees future context)
    Fusion        : element-wise sum of both membrane potentials before
                    the final classifier.

    This allows the model to capture bidirectional temporal context —
    e.g., a chair seat slice is more meaningful knowing both the
    earlier legs and the later backrest slices.

    Usage in forward_step:
        Call fwd_step / bwd_step per-step, then call classify() after
        all slices to fuse. See PointNetSNN.forward_step for usage.
    """
    def __init__(self, dim=256, num_classes=10, learnable_lif=False, use_bn=False):
        super().__init__()
        if use_bn:
            LayerCls = BNLIFLayer
        elif learnable_lif:
            LayerCls = LearnableLIFLayer
        else:
            LayerCls = LIFLayer

        # Forward temporal LIF pair
        self.fwd_lif1 = LayerCls(dim, dim)
        self.fwd_lif2 = LayerCls(dim, dim)

        # Backward temporal LIF pair (processes reversed slice order)
        self.bwd_lif1 = LayerCls(dim, dim)
        self.bwd_lif2 = LayerCls(dim, dim)

        # Fusion + classify
        self.fc = nn.Linear(dim, num_classes)

        # Accumulate backward embeddings for deferred fusion
        self._fwd_mems = []
        self._slice_features = []   # store slice feats for backward pass

    def reset_state(self, batch_size, device=None):
        self.fwd_lif1.reset_state(batch_size, device)
        self.fwd_lif2.reset_state(batch_size, device)
        self.bwd_lif1.reset_state(batch_size, device)
        self.bwd_lif2.reset_state(batch_size, device)
        self._fwd_mems = []
        self._slice_features = []

    def forward_step(self, x):
        """
        Process one slice in the forward direction.
        Stores both the slice feature and the resulting membrane potential.
        Returns forward logits (for aux loss).
        """
        self._slice_features.append(x)
        _, mem1 = self.fwd_lif1(x)
        _, mem2 = self.fwd_lif2(mem1)
        self._fwd_mems.append(mem2)
        return self.fc(mem2)    # forward-only logits at this timestep

    def classify(self):
        """
        Called after all forward_step() calls.
        Runs the backward pass over stored slices (reversed order),
        fuses with forward mems, returns final logits.
        """
        T = len(self._slice_features)
        bwd_mems = []
        for x in reversed(self._slice_features):
            _, mem1 = self.bwd_lif1(x)
            _, mem2 = self.bwd_lif2(mem1)
            bwd_mems.append(mem2)
        bwd_mems = list(reversed(bwd_mems))  # re-align with forward order

        # Fuse: sum of forward and backward final membrane potentials
        fused = self._fwd_mems[-1] + bwd_mems[-1]
        return self.fc(fused)

    def forward(self, x):
        """Single-step forward (used in forward_full mode)."""
        _, mem1 = self.fwd_lif1(x)
        _, mem2 = self.fwd_lif2(mem1)
        return self.fc(mem2)

    def firing_rates(self):
        rates = {}
        pairs = [("fwd_lif1", self.fwd_lif1), ("fwd_lif2", self.fwd_lif2),
                 ("bwd_lif1", self.bwd_lif1), ("bwd_lif2", self.bwd_lif2)]
        for name, layer in pairs:
            if hasattr(layer, 'firing_rate'):
                rates[f"temporal_{name}"] = layer.firing_rate()
        return rates

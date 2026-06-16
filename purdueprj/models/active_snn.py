"""
active_snn.py
=============
ActiveSNN: Full Active Spiking Perception model.

Combines:
  - LocalKNNBackbone     (per-slice spatial feature extraction)
  - TemporalSNN          (causal LIF temporal head with learnable tau/vth)
  - SliceSelectionPolicy (membrane-guided adaptive slice ordering)

The model operates in two modes:

  TRAINING MODE (forward_active_train):
    1. Precompute backbone features for ALL T slices in parallel.
    2. At each timestep, SSP selects a slice via Gumbel-softmax
       (differentiable, straight-through) from precomputed features.
    3. Selected feature updates the temporal head.
    4. All T logits are returned for the joint loss.
    No early exit during training (exit is encouraged via L_exit loss term).

  INFERENCE MODE (forward_active_infer):
    1. SSP selects the best unvisited anchor greedily (argmax).
    2. Backbone processes ONLY the selected slice (energy savings!).
    3. Temporal head updates membrane state.
    4. Early exit fires when margin(top1 - top2) > threshold θ.
    Sequential processing: only T_exit ≤ T backbone passes run.

The gap between training (all slices precomputed) and inference (sequential)
is bridged by the fact that the backbone is a deterministic function of the
input slice. The SSP learns a policy over features; at inference it receives
the same features sequentially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet_backbone import LocalKNNBackbone
from models.temporal_snn import TemporalSNN
from models.slice_selection_policy import (
    SliceSelectionPolicy,
    compute_geometry_descriptors,
)


class ActiveSNN(nn.Module):
    """
    Active Spiking Perception model.

    Parameters
    ----------
    point_dims   : list[int]   hidden dims for KNN backbone MLP
    temporal_dim : int         membrane dimensionality (= last point_dim)
    num_classes  : int         number of output classes
    knn_k        : int         K nearest neighbours in each slice
    d_ssp        : int         SSP internal projection dim (default 64)
    """

    def __init__(
        self,
        point_dims: list = [128, 256, 512],
        temporal_dim: int = 512,
        num_classes: int = 10,
        knn_k: int = 16,
        d_ssp: int = 64,
    ):
        super().__init__()

        self.temporal_dim = temporal_dim
        self.num_classes  = num_classes

        # Per-slice spatial feature extractor (KNN + learnable LIF)
        self.backbone = LocalKNNBackbone(
            hidden_dims=point_dims,
            k=knn_k,
            learnable_lif=True,
        )

        # Causal temporal head (TemporalSNN, learnable LIF, no bidirectional)
        # We use causal-only because the SSP already provides "future" context
        # by choosing the most informative next slice — no need to buffer.
        self.temporal = TemporalSNN(
            dim=temporal_dim,
            num_classes=num_classes,
            learnable_lif=True,
        )

        # Slice selection policy
        self.ssp = SliceSelectionPolicy(
            mem_dim=temporal_dim,
            d_ssp=d_ssp,
        )

        # Gumbel temperature (annealed during training)
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int, device=None):
        """Reset all LIF membrane states and SSP hidden state."""
        self.backbone.reset_state(batch_size, device)
        self.temporal.reset_state(batch_size, device)

    def _get_membrane(self) -> torch.Tensor:
        """
        Extract the current temporal head membrane potential.
        Used as input to SSP at each step.
        Returns [B, temporal_dim] or zeros if not yet initialised.
        """
        lif = self.temporal.lif2
        if lif.mem is None:
            return None
        return lif.mem.detach()    # detached: gradient flows through lif, not mem history

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_active_train(
        self,
        pts_slices: torch.Tensor,
        geo_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Active training forward pass.

        Args:
            pts_slices      : [B, T, n_pts, 3]  all FPS slices (pre-sliced)
            geo_descriptors : [B, T, 6]          geometry descriptors for each anchor

        Returns:
            logits_final  : [B, num_classes]     final logit (from last selected slice)
            logits_all    : list of T tensors     intermediate logits (for aux + exit loss)
        """
        B, T, n_pts, _ = pts_slices.shape
        device = pts_slices.device

        self.reset_state(B, device)

        # --- Step 1: Precompute backbone features for all T slices ---
        # We process all in one shot for efficiency.
        # [B*T, n_pts, 3] → backbone → [B*T, n_pts, D] → mean pool → [B*T, D]
        pts_flat = pts_slices.reshape(B * T, n_pts, 3)
        # Reset backbone state for all B*T "independent" forward passes
        self.backbone.reset_state(B * T, device)
        feat_per_point = self.backbone.forward(pts_flat)           # [B*T, n_pts, D]
        all_feats = feat_per_point.mean(dim=1)                     # [B*T, D]
        all_feats = all_feats.reshape(B, T, -1)                    # [B, T, D]

        # Note: backbone membrane states are internal to each slice's processing;
        # they are reset for each slice independently. The temporal LIF memory
        # accumulates across slices below.
        self.backbone.reset_state(B, device)

        # --- Step 2: Re-initialise temporal head for sequence processing ---
        # (reset_state was called above, but backbone reset changed things)
        self.temporal.reset_state(B, device)

        # --- Step 3: SSP-guided sequential selection ---
        visited_mask  = torch.zeros(B, T, dtype=torch.bool, device=device)
        logits_all    = []
        mem_state     = torch.zeros(B, self.temporal_dim, device=device)

        for t in range(T):
            # Compute SSP scores from current membrane belief
            scores = self.ssp(mem_state, geo_descriptors, visited_mask)   # [B, T]

            # Differentiable selection via Gumbel-softmax (straight-through)
            tau    = self.gumbel_tau.item()
            w      = self.ssp.select_gumbel(scores, tau=tau)              # [B, T] ≈ one-hot

            # Mark selected anchors as visited (hard argmax for mask update)
            selected_idx = scores.masked_fill(visited_mask, float("-inf")).argmax(dim=-1)  # [B]
            for b in range(B):
                visited_mask[b, selected_idx[b]] = True

            # Soft-select feature (differentiable): weighted sum over all T feats
            # w is ~one-hot so this ≈ all_feats[:, selected_idx, :]
            e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)               # [B, D]

            # Update temporal head and get logits
            logits_t = self.temporal(e_t)                                 # [B, num_classes]
            logits_all.append(logits_t)

            # Update mem_state for next SSP call (detached to avoid retain_graph)
            mem_state = self._get_membrane()
            if mem_state is None:
                mem_state = torch.zeros(B, self.temporal_dim, device=device)

        logits_final = logits_all[-1]
        return logits_final, logits_all

    # ------------------------------------------------------------------
    # Inference forward (energy-efficient)
    # ------------------------------------------------------------------

    def forward_active_infer(
        self,
        pts_slices: torch.Tensor,
        geo_descriptors: torch.Tensor,
        threshold: float = 0.7,
        return_all: bool = False,
    ) -> tuple[torch.Tensor, int, list[int]]:
        """
        Active inference: sequential slice selection with early exit.

        At each step:
          1. SSP selects best unvisited anchor (argmax, O(M) dot products).
          2. Backbone processes ONLY that slice (main energy cost).
          3. Temporal head updates membrane.
          4. Exit if margin(top1 - top2) > threshold.

        Args:
            pts_slices      : [B, T, n_pts, 3]
            geo_descriptors : [B, T, 6]
            threshold       : float  margin threshold for early exit

        Returns:
            logits      : [B, num_classes]  final (or early exit) logit
            exit_step   : int              timestep at which we exited (1-indexed)
            slice_order : list[int]        order of anchor indices selected
        """
        B, T, n_pts, _ = pts_slices.shape
        device = pts_slices.device

        self.reset_state(B, device)

        visited_mask  = torch.zeros(B, T, dtype=torch.bool, device=device)
        mem_state     = torch.zeros(B, self.temporal_dim, device=device)
        slice_orders  = [[] for _ in range(B)]
        last_logits   = None
        logits_all    = []
        batch_idx     = torch.arange(B, device=device)

        for t in range(T):
            # SSP: pick best unvisited anchor
            with torch.no_grad():
                scores = self.ssp(mem_state, geo_descriptors, visited_mask)
                w = self.ssp.select_greedy(scores)                       # [B, T] one-hot

            selected_idx = w.argmax(dim=-1)                              # [B]
            for b in range(B):
                slice_orders[b].append(selected_idx[b].item())

            for b in range(B):
                visited_mask[b, selected_idx[b]] = True

            # Process each sample's selected slice through the backbone.
            slice_b = pts_slices[batch_idx, selected_idx, :, :]          # [B, n_pts, 3]
            with torch.no_grad():
                self.backbone.reset_state(B, device)
                feat_pp = self.backbone(slice_b)                         # [B, n_pts, D]
                e_t = feat_pp.mean(dim=1)                                # [B, D]

                logits_t = self.temporal(e_t)                            # [B, num_classes]
                last_logits = logits_t
                logits_all.append(logits_t)

            # Update membrane belief
            mem_state = self._get_membrane()
            if mem_state is None:
                mem_state = torch.zeros(B, self.temporal_dim, device=device)

            # Early exit check
            probs = F.softmax(logits_t, dim=-1)                         # [B, C]
            top2  = probs.topk(2, dim=-1).values                        # [B, 2]
            margin = (top2[:, 0] - top2[:, 1])                          # [B]

            if margin.min().item() > threshold:
                order = slice_orders[0] if B == 1 else slice_orders
                if return_all:
                    return last_logits, t + 1, order, logits_all
                return last_logits, t + 1, order

        order = slice_orders[0] if B == 1 else slice_orders
        if return_all:
            return last_logits, T, order, logits_all
        return last_logits, T, order

    # ------------------------------------------------------------------
    # Compatibility wrappers (for existing eval loops)
    # ------------------------------------------------------------------

    def forward_step(self, pts_slice: torch.Tensor) -> torch.Tensor:
        """
        Single-step forward without SSP (for compatibility with existing
        train_loop.py and eval scripts).
        pts_slice: [B, n_pts, 3]
        """
        feat_pp = self.backbone(pts_slice)                               # [B, n_pts, D]
        e = feat_pp.mean(dim=1)                                          # [B, D]
        return self.temporal(e)                                          # [B, num_classes]

    def forward_full(self, pts: torch.Tensor) -> torch.Tensor:
        """Single-pass forward on full cloud (no slicing)."""
        feat_pp = self.backbone(pts)
        e = feat_pp.mean(dim=1)
        return self.temporal(e)

    # ------------------------------------------------------------------
    # Efficiency utilities
    # ------------------------------------------------------------------

    def get_firing_rates(self) -> dict:
        """Collect firing rates from all LearnableLIF layers."""
        rates = {}
        if hasattr(self.backbone, "firing_rates"):
            rates.update(self.backbone.firing_rates())
        if hasattr(self.temporal, "firing_rates"):
            rates.update(self.temporal.firing_rates())
        return rates

    def mean_firing_rate(self) -> float:
        """Scalar mean firing rate across all layers."""
        rates = self.get_firing_rates()
        if not rates:
            return 0.0
        return sum(rates.values()) / len(rates)

    def set_gumbel_tau(self, tau: float):
        """Update Gumbel temperature (called by training loop each epoch)."""
        self.gumbel_tau.fill_(tau)

    def param_count(self) -> dict:
        """Parameter counts by submodule."""
        bb   = sum(p.numel() for p in self.backbone.parameters())
        temp = sum(p.numel() for p in self.temporal.parameters())
        ssp  = sum(p.numel() for p in self.ssp.parameters())
        return {"backbone": bb, "temporal": temp, "ssp": ssp, "total": bb + temp + ssp}

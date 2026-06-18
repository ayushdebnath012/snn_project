"""
asp_wrapper.py
==============
ASPWrapper — plug-in Active Spiking Perception adapter for any temporal SNN.

Usage
-----
    from models.asp_wrapper import ASPWrapper
    from models.spiking_mamba import SPMModel
    from models.pointnet_snn import PointNetSNN

    base = SPMModel(num_classes=40, ...)
    model = ASPWrapper(base, feat_dim=512, num_classes=40)

    # Training
    logits_final, logits_all, selection_weights = model.forward_active_train(pts_slices, geo_desc)

    # Inference with early exit
    logits, exit_step, order = model.forward_active_infer(pts_slices, geo_desc,
                                                           threshold=0.7)

Interface contract for base_model
----------------------------------
  base_model.backbone
      — nn.Module with:
          reset_state(batch_size, device)
          forward(pts: [B, N, 3]) → [B, N, D]  per-point features

  base_model.forward_step_feat(feat, **kwargs) → [B, num_classes]
      — Runs the temporal head on a precomputed backbone embedding.
        SPMModel additionally accepts orig_t=[B] and T=int for HDE.
        PointNetSNN ignores extra kwargs.

  base_model.reset_state(batch_size, device)

  base_model.get_firing_rates() → dict  (optional)

ASP belief state
-----------------
Unlike ActiveSNN (which reads lif2.mem directly), ASPWrapper maintains a
model-agnostic belief state by projecting the softmax output of the temporal
head to feat_dim via a small Linear + ReLU layer.  This is equivalent in spirit
to the membrane potential — "what class does the model currently favour?" — and
requires no introspection of model internals.

For SPMModel (HDE path):
    forward_step_feat receives orig_t=[B] (original FPS anchor index) so that
    the stage/positional embeddings in HDE reflect the *spatial geometry* of the
    selected slice, not its processing order.  ASP reorders processing order;
    HDE encodes geometry.  The two concerns are separated cleanly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slice_selection_policy import SliceSelectionPolicy, GEO_DIM


class ASPWrapper(nn.Module):
    """
    Plug-in ASP adapter for any temporal SNN.

    Improvements over v1
    --------------------
    1. GRU belief state (replaces the Linear(C→D) softmax projection).
       A GRU maintains a running hidden state across the T slice steps,
       giving the SSP a richer, history-aware view of what has been seen.
       Input at each step: concat(backbone_feat [D], softmax_logits [C]).

    2. Upgraded SSP (multi-head, 9-D geometry, diversity penalty).
       SliceSelectionPolicy now uses n_heads=4 heads, LayerNorm on belief,
       and a diversity penalty to spread anchor selections across the shape.

    3. geo_dim auto-detected from SliceSelectionPolicy.W_q.in_features.
       Old code hardcoded 6; new descriptors are 9-D.

    Parameters
    ----------
    base_model  : nn.Module  the wrapped temporal SNN (SPMModel or PointNetSNN)
    feat_dim    : int        backbone embedding dimension (= last point_dim, e.g. 512)
    num_classes : int        number of output classes
    d_ssp       : int        SSP total projection dimension (default 128, split across heads)
    n_heads     : int        SSP attention heads (default 4)
    diversity   : float      SSP diversity penalty weight (default 0.1)
    """

    def __init__(self, base_model: nn.Module, feat_dim: int,
                 num_classes: int, d_ssp: int = 128,
                 n_heads: int = 4, diversity: float = 0.1):
        super().__init__()
        self.base_model  = base_model
        self.feat_dim    = feat_dim
        self.num_classes = num_classes
        self.temporal_dim = feat_dim   # read by train_active.py for zero vectors

        # Multi-head SSP with 9-D geometry descriptors and diversity penalty
        self.ssp = SliceSelectionPolicy(
            mem_dim=feat_dim, geo_dim=GEO_DIM,
            d_ssp=d_ssp, n_heads=n_heads, diversity=diversity,
        )

        # GRU belief state: input = [backbone_feat | softmax(logits)]
        # hidden = feat_dim  (same dimensionality as before → SSP unchanged)
        gru_input_dim = feat_dim + num_classes
        self.belief_gru = nn.GRUCell(gru_input_dim, feat_dim)

        # Gumbel temperature (annealed during training via set_gumbel_tau)
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    # ------------------------------------------------------------------
    # Backbone / temporal property shims (for active_inference.py compat)
    # ------------------------------------------------------------------

    @property
    def backbone(self):
        return self.base_model.backbone

    @property
    def temporal(self):
        """Returns base_model.temporal if it exists, else None (e.g. SPMModel)."""
        return getattr(self.base_model, "temporal", None)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int, device=None):
        self.base_model.reset_state(batch_size, device)

    # ------------------------------------------------------------------
    # Feature-step helper — dispatches to the right base_model method
    # ------------------------------------------------------------------

    def _step_feat(self, feat: torch.Tensor,
                   orig_t=None, T: int = None) -> torch.Tensor:
        """
        Call base_model.forward_step_feat with the appropriate signature.

        For SPMModel (has HDE): passes orig_t and T so HDE uses the original
        FPS anchor index for geometric encoding.
        For PointNetSNN (no HDE): ignores orig_t / T.
        """
        if hasattr(self.base_model, "hde"):
            return self.base_model.forward_step_feat(feat, orig_t=orig_t, T=T)
        return self.base_model.forward_step_feat(feat)

    def _pool_points(self, feat_pp: torch.Tensor) -> torch.Tensor:
        """Use the base model's slice pooling when it defines one."""
        if hasattr(self.base_model, "pool_points"):
            return self.base_model.pool_points(feat_pp)
        return feat_pp.mean(dim=1)

    # ------------------------------------------------------------------
    # Training forward (all slices precomputed, SSP reorders)
    # ------------------------------------------------------------------

    def forward_active_train(
        self,
        pts_slices: torch.Tensor,
        geo_descriptors: torch.Tensor,
    ):
        """
        Active training forward pass.

        Args:
            pts_slices      : [B, T, N, 3]   all FPS slices
            geo_descriptors : [B, T, G]       geometry descriptors per anchor (G=9)

        Returns:
            logits_final      : [B, num_classes]  logit from last selected slice
            logits_all        : list[T tensors]   intermediate logits (for aux/exit loss)
            selection_weights : list[T tensors]   Gumbel-softmax weights [B, T] (for diversity loss)
        """
        B, T, N, _ = pts_slices.shape
        device = pts_slices.device

        # ── Phase 1: precompute backbone embeddings for ALL T slices ────────
        pts_flat = pts_slices.reshape(B * T, N, 3)
        self.base_model.backbone.reset_state(B * T, device)
        feat_pp   = self.base_model.backbone(pts_flat)         # [B*T, N, D]
        all_feats = self._pool_points(feat_pp).reshape(B, T, -1)  # [B, T, D]

        # ── Phase 2: reset backbone + full model for sequential pass ────────
        self.base_model.backbone.reset_state(B, device)
        self.base_model.reset_state(B, device)
        if hasattr(self.base_model, "_total_T"):
            self.base_model._total_T = T

        # ── Phase 3: SSP-guided sequential selection ─────────────────────────
        visited = [[False] * T for _ in range(B)]
        logits_all        = []
        selection_weights = []
        belief = torch.zeros(B, self.feat_dim, device=device)

        for t in range(T):
            visited_mask = torch.tensor(visited, dtype=torch.bool, device=device)

            scores = self.ssp(belief, geo_descriptors, visited_mask)  # [B, T]

            tau = self.gumbel_tau.item()
            w   = self.ssp.select_gumbel(scores, tau=tau)             # [B, T]
            selection_weights.append(w)

            selected_idx = scores.detach().argmax(dim=-1)             # [B]
            for b in range(B):
                visited[b][selected_idx[b].item()] = True

            e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)           # [B, D]

            logits_t = self._step_feat(e_t, orig_t=selected_idx, T=T)
            logits_all.append(logits_t)

            gru_in = torch.cat([
                e_t.detach(),
                logits_t.detach().softmax(dim=-1),
            ], dim=-1)                                                # [B, D+C]
            belief = self.belief_gru(gru_in, belief)                  # [B, D]

        if (hasattr(self.base_model, "bidirectional") and
                self.base_model.bidirectional and
                hasattr(getattr(self.base_model, "temporal", None), "classify")):
            logits_all[-1] = self.base_model.temporal.classify()

        return logits_all[-1], logits_all, selection_weights

    # ------------------------------------------------------------------
    # Inference forward (energy-efficient: only selected slices processed)
    # ------------------------------------------------------------------

    def forward_active_infer(
        self,
        pts_slices: torch.Tensor,
        geo_descriptors: torch.Tensor,
        threshold: float = 0.7,
        return_all: bool = False,
    ):
        """
        Active inference: sequential slice selection with early exit.

        At each step:
          1. SSP selects the best unvisited anchor (argmax).
          2. Backbone processes ONLY that slice (main energy cost).
          3. Temporal head updates internal state.
          4. Exit if margin(top1 - top2) > threshold.

        Args:
            pts_slices      : [B, T, N, 3]
            geo_descriptors : [B, T, 6]
            threshold       : float  margin for early exit

        Returns:
            logits      : [B, num_classes]
            exit_step   : int   timestep at which we exited (1-indexed)
            slice_order : list[int]  original FPS anchor indices selected
        """
        B, T, N, _ = pts_slices.shape
        device = pts_slices.device

        self.base_model.backbone.reset_state(B, device)
        self.base_model.reset_state(B, device)
        if hasattr(self.base_model, "_total_T"):
            self.base_model._total_T = T

        visited       = [[False] * T for _ in range(B)]
        belief        = torch.zeros(B, self.feat_dim, device=device)
        slice_orders  = [[] for _ in range(B)]
        last_logits   = None
        logits_all    = []
        batch_idx     = torch.arange(B, device=device)

        with torch.no_grad():
            for t in range(T):
                visited_mask = torch.tensor(visited, dtype=torch.bool, device=device)
                # SSP: greedy argmax selection
                scores = self.ssp(belief, geo_descriptors, visited_mask)
                w      = self.ssp.select_greedy(scores)             # [B, T] one-hot
                selected_idx = w.argmax(dim=-1)                     # [B]
                for b in range(B):
                    slice_orders[b].append(selected_idx[b].item())

                for b in range(B):
                    visited[b][selected_idx[b].item()] = True

                # Process each sample's selected slice. The earlier version
                # used selected_idx[0] for the whole batch, which made batched
                # ASP validation follow the first sample's policy.
                self.base_model.backbone.reset_state(B, device)
                feat_pp = self.base_model.backbone(
                    pts_slices[batch_idx, selected_idx]             # [B, N, 3]
                )
                e_t = self._pool_points(feat_pp)                    # [B, D]

                # Temporal head — pass original anchor indices for HDE
                logits_t = self._step_feat(e_t, orig_t=selected_idx, T=T)
                last_logits = logits_t
                logits_all.append(logits_t)

                # GRU belief update (same as training path)
                gru_in = torch.cat([
                    e_t,
                    logits_t.softmax(dim=-1),
                ], dim=-1)
                belief = self.belief_gru(gru_in, belief)

                # Early exit check
                probs  = F.softmax(logits_t, dim=-1)
                top2   = probs.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1])
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
    # Compatibility wrappers (for eval loops that call model(pts_slices))
    # ------------------------------------------------------------------

    def forward(self, pts_slices: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: runs training forward with zero geometry descriptors.
        Used by eval scripts that call model(pts_slices) directly.
        Geometry descriptors = 0 → SSP produces uniform-ish scores (random order),
        which is still valid for accuracy evaluation (not energy-optimal).
        """
        B, T = pts_slices.shape[:2]
        geo = torch.zeros(B, T, GEO_DIM, device=pts_slices.device)
        logits_final, _, _ = self.forward_active_train(pts_slices, geo)
        return logits_final

    # ------------------------------------------------------------------
    # Efficiency utilities
    # ------------------------------------------------------------------

    def set_gumbel_tau(self, tau: float):
        """Anneal Gumbel temperature (called by training loop each epoch)."""
        self.gumbel_tau.fill_(tau)

    def get_firing_rates(self) -> dict:
        if hasattr(self.base_model, "get_firing_rates"):
            return self.base_model.get_firing_rates()
        return {}

    def mean_firing_rate(self) -> float:
        rates = self.get_firing_rates()
        if not rates:
            return 0.0
        return sum(rates.values()) / len(rates)

    def param_count(self) -> dict:
        bb   = sum(p.numel() for p in self.base_model.backbone.parameters())
        base = sum(p.numel() for p in self.base_model.parameters())
        ssp  = sum(p.numel() for p in self.ssp.parameters())
        gru  = sum(p.numel() for p in self.belief_gru.parameters())
        return {
            "backbone":    bb,
            "temporal":    base - bb,
            "ssp":         ssp,
            "belief_gru":  gru,
            "total":       base + ssp + gru,
        }

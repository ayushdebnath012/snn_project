"""
spiking_ssm.py
==============
SpikingSSMs temporal block adapted from arXiv 2408.14909
"SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State
Space Models".

Key ideas implemented here:
  SpikingSSMCell:
    - SSM recurrence: h_t = A * h_{t-1} + B * x_t
                      y_t = C * h_t
    - Then y_t passes through LIF neuron to produce binary spike s_t
    - s_t is the temporal context passed to the classifier

  SpikingSSMTemporal:
    - Wraps SpikingSSMCell for slice-by-slice point cloud processing
    - Compatible with PointNetSNN forward_step / reset_state interface

Architecture differences vs our TemporalSNN:
  - TemporalSNN: two stacked LIF layers (standard RNN-like)
  - SpikingSSMTemporal: SSM (structured matrix multiply) + LIF
    → better at long-range dependencies (exponential decay in A)
    → parallel training possible (parallel scan)
    → sparser activations due to LIF gate after SSM output

Paper training detail:
  - Uses Surrogate Dynamic Network (SDN) for backward pass
  - A is diagonal, initialised with exponential basis: exp(-exp(log_dt + A_log))
  - B, C are linear projections learned from input
  - Here we use a simplified real-valued diagonal SSM (S4-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.neuron_zoo import tri_spike


# ---------------------------------------------------------------------------
# Diagonal SSM core (S4D-Real)
# ---------------------------------------------------------------------------

class DiagonalSSM(nn.Module):
    """
    Real-valued diagonal SSM.

    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C * h_t + D * x_t

    A_bar = exp(dt * A)   (ZOH discretisation, A is negative real → stable)
    B_bar = (A_bar - 1) / A * B   (ZOH)

    Parameters:
      d_model : input/output feature dimension
      d_state : SSM state dimension N (default 16 as in S4D paper)
      dt_min/max : range for dt initialisation
    """
    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A: initialised as log of negative real values
        # A = -exp(A_log) ensures stability
        A_log = torch.rand(d_model, d_state) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.A_log = nn.Parameter(A_log)

        # B projection: input → shared state  [d_model → d_state]
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        # C is a per-feature-dimension weight: [d_model, d_state]
        # y[b,d] = sum_n(C[d,n] * h[b,d,n])  (no mixing across d)
        self.C_weight = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Skip connection D (direct term)
        self.D = nn.Parameter(torch.ones(d_model))

        # Log step size (learnable scalar per feature)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Hidden state buffer
        self.register_buffer("h", None, persistent=False)

    def reset(self, batch_size, device):
        self.h = torch.zeros(batch_size, self.d_model, self.d_state, device=device)

    def forward(self, x):
        """
        x : [B, d_model]  (single timestep)
        Returns : [B, d_model]
        """
        B_batch = x.size(0)
        if self.h is None or self.h.shape[0] != B_batch:
            self.reset(B_batch, x.device)

        # Build discretised A
        dt = torch.exp(self.log_dt)                          # [d_model]
        A  = -torch.exp(self.A_log)                          # [d_model, d_state]  negative
        A_bar = torch.exp(dt.unsqueeze(-1) * A)             # [d_model, d_state]

        # B_bar: ZOH for input matrix
        # B_in: [B, d_state]
        B_in = self.B_proj(x)                               # [B, d_state]
        # B_bar = (A_bar - 1) / A * B  → broadcast
        B_bar = (A_bar - 1) / (A + 1e-8)                   # [d_model, d_state]
        # h update: [B, d_model, d_state]
        # h = A_bar * h + B_bar * B_in
        #   A_bar: [d_model, d_state]  → broadcast over B
        #   B_in:  [B, d_state] → need [B, d_model, d_state]
        B_term = B_bar.unsqueeze(0) * B_in.unsqueeze(1)    # [B, d_model, d_state]
        self.h = A_bar.unsqueeze(0) * self.h + B_term      # [B, d_model, d_state]

        # Output: y[b,d] = sum_n(C[d,n] * h[b,d,n]) + D[d]*x[b,d]
        # h: [B, d_model, d_state], C_weight: [d_model, d_state]
        y = (self.h * self.C_weight.unsqueeze(0)).sum(-1)   # [B, d_model]
        y = y + self.D * x
        return y


# ---------------------------------------------------------------------------
# SpikingSSM Cell: SSM output → LIF neuron → spike
# ---------------------------------------------------------------------------

class SpikingSSMCell(nn.Module):
    """
    Single SpikingSSM layer:
      x_t (input features) → DiagonalSSM → y_t → LIF → spike s_t

    The LIF here acts as a threshold gate on the SSM output, producing
    binary/sparse activations that are energy-efficient at inference.
    """
    def __init__(self, d_model, d_state=16, tau=0.9, vth=1.0):
        super().__init__()
        self.ssm = DiagonalSSM(d_model, d_state=d_state)
        self.tau = tau
        self.vth = vth
        self.register_buffer("mem", None, persistent=False)

    def reset(self, batch_size, device):
        self.ssm.reset(batch_size, device)
        self.mem = torch.zeros(batch_size, self.ssm.d_model, device=device)

    def forward(self, x):
        """
        x   : [B, d_model]
        Returns spk : [B, d_model]
        """
        y = self.ssm(x)                      # [B, d_model]
        self.mem = self.tau * self.mem + y
        spk = tri_spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)      # hard reset
        return spk


# ---------------------------------------------------------------------------
# SpikingSSM Temporal module — replaces TemporalSNN
# ---------------------------------------------------------------------------

class SpikingSSMTemporal(nn.Module):
    """
    Temporal processing using two stacked SpikingSSM cells.

    Replaces TemporalSNN in PointNetSNN when spiking_ssm=True.

    Input: backbone embedding per slice [B, in_dim]
    Output: classification logits [B, num_classes]

    The two-cell stack (like 2-layer RNN) allows richer temporal dynamics:
      cell1: first-level SSM integration + LIF spike
      cell2: second-level SSM integration + LIF spike → fc → logits
    """
    def __init__(self, in_dim=256, hidden_dim=256, num_classes=40,
                 d_state=16, tau=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Project from backbone dim to SSM dim if needed
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # Two stacked SpikingSSM cells
        self.cell1 = SpikingSSMCell(hidden_dim, d_state=d_state, tau=tau)
        self.cell2 = SpikingSSMCell(hidden_dim, d_state=d_state, tau=tau)

        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Track last output for early exit
        self._last_logits = None

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.cell1.reset(batch_size, dev)
        self.cell2.reset(batch_size, dev)
        self._last_logits = None

    def forward_step(self, feat):
        """
        Process one slice embedding.
        feat : [B, in_dim]
        Returns logits [B, num_classes]
        """
        x = self.in_proj(feat)
        s1 = self.cell1(x)
        s2 = self.cell2(s1)
        logits = self.fc(s2)
        self._last_logits = logits
        return logits

    def forward(self, feats):
        """
        feats : [B, T, in_dim]  (all slices)
        Returns logits [B, num_classes]
        """
        B, T, _ = feats.shape
        for t in range(T):
            logits = self.forward_step(feats[:, t])
        return logits


# ---------------------------------------------------------------------------
# Parallel training version (for batch efficiency — not slice-by-slice)
# ---------------------------------------------------------------------------

class SpikingSSMParallel(nn.Module):
    """
    Parallel-scan version of SpikingSSM for faster training.

    When all slices are available at once (training), the SSM recurrence
    can be parallelised via associative scan.  Here we use a simple
    unrolled loop but structure it for potential future optimisation.

    For inference (slice-by-slice), use SpikingSSMTemporal.

    Input:  feats [B, T, in_dim]
    Output: logits [B, num_classes]  (uses last timestep's output)
    """
    def __init__(self, in_dim=256, hidden_dim=256, num_classes=40,
                 d_state=16, tau=0.9):
        super().__init__()
        self.temporal = SpikingSSMTemporal(
            in_dim=in_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, d_state=d_state, tau=tau
        )

    def reset_state(self, batch_size, device=None):
        self.temporal.reset_state(batch_size, device)

    def forward(self, feats):
        """feats: [B, T, in_dim] → logits [B, num_classes]"""
        B = feats.size(0)
        self.temporal.reset_state(B, feats.device)
        return self.temporal.forward(feats)

    def forward_step(self, feat):
        return self.temporal.forward_step(feat)

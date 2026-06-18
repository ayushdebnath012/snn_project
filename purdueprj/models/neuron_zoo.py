"""
neuron_zoo.py
=============
All spiking neuron variants used across the compared papers, implemented
from scratch in pure PyTorch for fair comparison.

Papers sourced from:
  - Spiking PointNet  (2310.06232)  : PerturbedLIFLayer, TanhSurrogate
  - SPM ablation      (2504.14371)  : PLIFLayer (PLIF = Parametric LIF)
  - E-3DSNN           (2412.07360)  : ILIFLayer (Integer LIF)
  - SpikingSSMs       (2408.14909)  : SpikingSSMCell
  - SPT               (2502.15811)  : HDIFLayer (Hybrid Dynamics IF)
  - Our own work                    : LearnableLIFLayer (in snn_layers.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Surrogate gradient functions
# ---------------------------------------------------------------------------

class TanhSurrogate(torch.autograd.Function):
    """
    Spiking PointNet (2310.06232) Eq.5:
    φ(x) = 0.5 * tanh(k*(x - Vth)) + 0.5   with k=5
    Smoother than the rectangular window — gives better gradients for
    single-step training.
    """
    @staticmethod
    def forward(ctx, x, k=5.0):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        ctx.k = k
        return out

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        grad_phi = 0.5 * ctx.k * (1 - torch.tanh(ctx.k * x) ** 2)
        return grad * grad_phi, None


class TriangularSurrogate(torch.autograd.Function):
    """Rectangular window / triangular surrogate (standard, used in SPM/SPT)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (1.0 / (1 + torch.abs(x)) ** 2)


tanh_spike   = TanhSurrogate.apply
tri_spike    = TriangularSurrogate.apply


# ---------------------------------------------------------------------------
# 1. Standard LIF  (baseline, from our snn_layers.py — repeated here for
#    completeness so neuron_zoo is self-contained)
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """Standard LIF with fixed tau and Vth. Triangular surrogate."""
    def __init__(self, tau=0.9, vth=1.0):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.register_buffer("mem", None, persistent=False)

    def reset(self, batch_size, n_features, device):
        self.mem = torch.zeros(batch_size, n_features, device=device)

    def forward(self, x):
        self.mem = self.tau * self.mem + x
        spk = tri_spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        return spk, self.mem


# ---------------------------------------------------------------------------
# 2. PLIF — Parametric LIF  (SPM ablation Table 10, ref [7] Wei Fang 2021)
#    tau is a single GLOBAL learnable scalar (not per-neuron like our LLIF).
#    tau = sigmoid(w) so it stays in (0,1).
#    SPM PLIF achieves 90.5% OBJ-BG vs LIF 90.2% — +0.3%.
# ---------------------------------------------------------------------------

class PLIFNeuron(nn.Module):
    """
    Parametric LIF (PLIF) — Wei Fang et al. ICCV 2021.
    Single learnable tau (global for the layer).
    """
    def __init__(self, vth=1.0, tau_init=0.9):
        super().__init__()
        self.vth = vth
        tau_raw = math.log(tau_init / (1.0 - tau_init))
        self.tau_raw = nn.Parameter(torch.tensor(tau_raw))
        self.register_buffer("mem", None, persistent=False)

    @property
    def tau(self):
        return torch.sigmoid(self.tau_raw)

    def reset(self, batch_size, n_features, device):
        self.mem = torch.zeros(batch_size, n_features, device=device)

    def forward(self, x):
        self.mem = self.tau * self.mem + x
        spk = tri_spike(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)
        return spk, self.mem


class PLIFLayer(nn.Module):
    """PLIF wrapped with a linear projection."""
    def __init__(self, in_f, out_f, vth=1.0, tau_init=0.9):
        super().__init__()
        self.fc      = nn.Linear(in_f, out_f)
        self.neuron  = PLIFNeuron(vth=vth, tau_init=tau_init)
        self.out_f   = out_f

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.neuron.reset(batch_size, self.out_f, dev)

    def forward(self, x):
        cur = self.fc(x)
        return self.neuron(cur)


# ---------------------------------------------------------------------------
# 3. I-LIF — Integer LIF  (E-3DSNN 2412.07360)
#    Output = round(clip(U, 0, D)) — integer spike values in [0,D].
#    Enables pure spike-driven sparse addition inference.
#    D=4 is typical (4-bit integer spikes).
# ---------------------------------------------------------------------------

class ILIFNeuron(nn.Module):
    """
    Integer LIF (I-LIF) from E-3DSNN.
    S = round(clip(U, 0, D)) where D is the integer depth.
    The surrogate for training approximates round() with STE
    (straight-through estimator) and uses soft clipping gradient.
    """
    def __init__(self, D=4, tau=0.9):
        super().__init__()
        self.D   = D
        self.tau = tau
        self.register_buffer("mem", None, persistent=False)

    def reset(self, batch_size, n_features, device):
        self.mem = torch.zeros(batch_size, n_features, device=device)

    def forward(self, x):
        self.mem = self.tau * self.mem + x
        # Soft clip for gradient flow, hard for forward
        mem_clipped = torch.clamp(self.mem, 0.0, float(self.D))
        # STE: forward = round, backward = identity
        spk = (torch.round(mem_clipped) - mem_clipped).detach() + mem_clipped
        # Soft reset: subtract D when saturated
        self.mem = self.mem - spk
        return spk, self.mem


class ILIFLayer(nn.Module):
    """I-LIF wrapped with linear projection."""
    def __init__(self, in_f, out_f, D=4, tau=0.9):
        super().__init__()
        self.fc     = nn.Linear(in_f, out_f)
        self.neuron = ILIFNeuron(D=D, tau=tau)
        self.out_f  = out_f

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.neuron.reset(batch_size, self.out_f, dev)

    def forward(self, x):
        cur = self.fc(x)
        return self.neuron(cur)


# ---------------------------------------------------------------------------
# 4. Perturbed LIF  (Spiking PointNet 2310.06232)
#    Standard LIF but membrane is randomly perturbed at reset to simulate
#    the ensemble effect of multi-step inference during single-step training.
#    λ (leak) = 0.25 by default (much lower than usual 0.9).
#    Surrogate = tanh (k=5).
# ---------------------------------------------------------------------------

class PerturbedLIFLayer(nn.Module):
    """
    Spiking PointNet perturbed LIF:
      - Low leakage λ ~ 0.25
      - Membrane perturbation δ ~ U(0, 0.5) during training
      - Tanh surrogate with k=5
      - Trained T=1, inferred T>1 (set self.T at inference)
    """
    def __init__(self, in_f, out_f, lam=0.25, vth=1.0, perturb=0.5):
        super().__init__()
        self.fc      = nn.Linear(in_f, out_f)
        self.lam     = lam        # leakage (λ in paper)
        self.vth     = vth
        self.perturb = perturb    # max perturbation magnitude
        self.out_f   = out_f
        self.register_buffer("mem", None, persistent=False)

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.mem = torch.zeros(batch_size, self.out_f, device=dev)
        if self.training and self.perturb > 0:
            self.mem = self.mem + torch.rand_like(self.mem) * self.perturb

    def forward(self, x):
        cur = self.fc(x)
        self.mem = self.lam * (self.mem - self.vth * tri_spike(self.mem - self.vth)) + cur
        spk = tanh_spike(self.mem - self.vth)
        return spk, self.mem


# ---------------------------------------------------------------------------
# 5. HD-IF — Hybrid Dynamics IF  (SPT 2502.15811)
#    Fuses LIF, IF, and EIF membrane dynamics via learned gating weights.
#    At inference, top-2 neuron types are selected.
#    We implement the training-time fusion version here.
# ---------------------------------------------------------------------------

class HDIFNeuron(nn.Module):
    """
    Hybrid Dynamics IF (SPT).
    Maintains 3 parallel membrane potentials (LIF, IF, EIF),
    fuses via softmax gating weights α, generates one spike.
    """
    def __init__(self, n_features, vth=1.0, tau=0.9, delta=0.5):
        super().__init__()
        self.vth   = vth
        self.tau   = tau
        self.delta = delta   # EIF sharpness
        self.n     = n_features
        # Gating weights: which neuron type to trust
        self.gate  = nn.Parameter(torch.ones(3) / 3)
        self.register_buffer("mem_lif", None, persistent=False)
        self.register_buffer("mem_if",  None, persistent=False)
        self.register_buffer("mem_eif", None, persistent=False)

    def reset(self, batch_size, device):
        z = torch.zeros(batch_size, self.n, device=device)
        self.mem_lif = z.clone()
        self.mem_if  = z.clone()
        self.mem_eif = z.clone()

    def forward(self, x):
        # LIF: leaky integrate
        self.mem_lif = self.tau * self.mem_lif + x

        # IF: perfect integrator (no leak)
        self.mem_if  = self.mem_if + x

        # EIF: exponential integrate (sharpness delta)
        exp_term = self.delta * torch.exp(
            (self.mem_eif - self.vth) / (self.delta + 1e-6)
        ).clamp(max=10.0)
        self.mem_eif = self.tau * self.mem_eif + x + exp_term

        # Fuse via softmax gate
        w = F.softmax(self.gate, dim=0)
        mem_fused = w[0] * self.mem_lif + w[1] * self.mem_if + w[2] * self.mem_eif

        spk = tri_spike(mem_fused - self.vth)

        # Hard reset all
        self.mem_lif = self.mem_lif * (1 - spk)
        self.mem_if  = self.mem_if  * (1 - spk)
        self.mem_eif = self.mem_eif * (1 - spk)

        return spk, mem_fused


class HDIFLayer(nn.Module):
    """HD-IF wrapped with linear projection."""
    def __init__(self, in_f, out_f, vth=1.0, tau=0.9):
        super().__init__()
        self.fc     = nn.Linear(in_f, out_f)
        self.neuron = HDIFNeuron(out_f, vth=vth, tau=tau)
        self.out_f  = out_f

    def reset_state(self, batch_size, device=None):
        dev = device or next(self.fc.parameters()).device
        self.neuron.reset(batch_size, dev)

    def forward(self, x):
        cur = self.fc(x)
        return self.neuron(cur)


# ---------------------------------------------------------------------------
# Neuron factory — build any layer type by name
# ---------------------------------------------------------------------------

NEURON_REGISTRY = {
    "lif"       : None,          # uses LIFLayer from snn_layers.py
    "learnable" : None,          # uses LearnableLIFLayer from snn_layers.py
    "plif"      : PLIFLayer,
    "ilif"      : ILIFLayer,
    "perturbed" : PerturbedLIFLayer,
    "hdif"      : HDIFLayer,
}


def build_layer(neuron_type, in_f, out_f, **kwargs):
    """
    Factory: build a spiking linear layer of the given neuron type.
    """
    if neuron_type in ("lif", None):
        from models.snn_layers import LIFLayer
        return LIFLayer(in_f, out_f)
    if neuron_type == "learnable":
        from models.snn_layers import LearnableLIFLayer
        return LearnableLIFLayer(in_f, out_f)
    cls = NEURON_REGISTRY.get(neuron_type)
    if cls is None:
        raise ValueError(f"Unknown neuron type: {neuron_type}")
    return cls(in_f, out_f, **kwargs)

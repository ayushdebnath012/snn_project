import math
import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = 1.0 / (1 + torch.abs(x))**2
        return grad_output * grad


spike_fn = SurrogateSpike.apply


class ATanSpike(torch.autograd.Function):
    """
    Arctangent surrogate gradient — produces wider, smoother gradient signal
    than the 1/(1+|x|)^2 surrogate, which helps in deep multi-timestep SNNs.
    Reference: Fang et al. "Incorporating Learnable Membrane Time Constant to
    Enhance Learning of Spiking Neural Networks" (ICCV 2021).
    alpha controls gradient width; larger alpha → narrower but taller gradient.
    """
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2.0 * (1.0 + (math.pi / 2.0 * alpha * x) ** 2))
        return grad_output * grad, None


def atan_spike(x, alpha=2.0):
    return ATanSpike.apply(x, alpha)


class BNLIFLayer(nn.Module):
    """LIF layer with BatchNorm before the membrane update (Linear -> BN -> LIF).
    Matches the BN-LIF pattern from Spiking PointNet (Wu et al., NeurIPS 2023).
    BN prevents membrane explosion and stabilises spike thresholds across layers.
    Uses fixed tau=0.25 and vth=0.5 (paper defaults).
    """
    def __init__(self, in_features, out_features, tau=0.25, vth=0.5):
        super().__init__()
        self.fc  = nn.Linear(in_features, out_features, bias=False)
        self.bn  = nn.BatchNorm1d(out_features)
        self.tau = tau
        self.vth = vth
        self.register_buffer("mem", None, persistent=False)

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem = torch.zeros(batch_size, self.fc.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        return (self.spike_count / (self.fc.out_features * self.step_count * getattr(self, "batch_size", 1))).item()

    def forward(self, x):
        cur      = self.bn(self.fc(x))
        self.mem = self.tau * self.mem + cur
        spk      = spike_fn(self.mem - self.vth)
        self.mem = self.mem * (1.0 - spk.detach())
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count + 1
        return spk, self.mem


class LIFLayer(nn.Module):
    """Original LIF with fixed tau and threshold."""
    def __init__(self, in_features, out_features, tau=0.9):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.register_buffer("mem", None, persistent=False)

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem = torch.zeros(batch_size, self.fc.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        return (self.spike_count / (self.fc.out_features * self.step_count * getattr(self, "batch_size", 1))).item()

    def forward(self, x):
        cur = self.fc(x)
        self.mem = self.tau * self.mem + cur
        spk = spike_fn(self.mem - 1.0)
        self.mem = self.mem * (1 - spk)
        
        if not hasattr(self, "spike_count"):
            self.spike_count = torch.tensor(0.0, device=cur.device)
            self.step_count = torch.tensor(0, device=cur.device)
            self.batch_size = x.shape[0]
            
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count + 1
        return spk, self.mem


class LearnableLIFLayer(nn.Module):
    """
    Novel: LIF neuron with LEARNABLE tau (leak) and threshold (V_th).

    From your notes: "v, T should be learnable — both threshold and leak
    are learnable parameters." This is inspired by SPM which learns
    per-neuron firing thresholds for better spike utilization.

    tau  is parameterized via sigmoid so it stays in (0,1).
    V_th is parameterized via softplus so it stays positive.
    Both are per-neuron (shape [out_features]), giving the model
    fine-grained control over each neuron's spiking behaviour.

    Also tracks:
        self.spike_count  — cumulative spike count since last reset_state
        self.total_neurons — denominator for spike rate
    so you can compute firing_rate = spike_count / (total_neurons * T).
    """
    def __init__(self, in_features, out_features, tau_init=0.9, vth_init=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.out_features = out_features

        # Learnable parameters (unconstrained; mapped through activation below)
        # tau_raw -> sigmoid -> (0,1)
        import math
        tau_raw_init = math.log(tau_init / (1 - tau_init))   # inverse sigmoid
        self.tau_raw = nn.Parameter(torch.full((out_features,), tau_raw_init))

        # vth_raw -> softplus -> positive
        # softplus(x) ≈ x for large x, so init to vth_init works
        self.vth_raw = nn.Parameter(torch.full((out_features,), float(vth_init)))

        self.register_buffer("mem", None, persistent=False)
        # Spike tracking (not parameters, just accumulators)
        self.register_buffer("spike_count", torch.tensor(0.0), persistent=False)
        self.register_buffer("step_count", torch.tensor(0), persistent=False)

    @property
    def tau(self):
        return torch.sigmoid(self.tau_raw)          # (0, 1)

    @property
    def vth(self):
        return torch.nn.functional.softplus(self.vth_raw)   # > 0

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem = torch.zeros(batch_size, self.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        """Average firing rate across neurons and timesteps since reset."""
        if self.step_count == 0:
            return 0.0
        return (self.spike_count / (self.out_features * self.step_count * getattr(self, "batch_size", 1))).item()

    def forward(self, x):
        cur = self.fc(x)
        # tau and vth are [out_features]; broadcast over batch dim
        self.mem = self.tau * self.mem + cur
        spk = spike_fn(self.mem - self.vth)
        self.mem = self.mem * (1 - spk)

        # Accumulate spike stats
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count + 1

        return spk, self.mem


class PLIFLayer(nn.Module):
    """
    Parametric LIF (PLIF) — tau stored as a single scalar parameter per layer
    rather than per-neuron.  Substantially fewer parameters than LearnableLIFLayer
    but still adapts the leak constant, which is often the most important knob.
    Uses ATan surrogate gradient for better gradient flow in deep networks.

    Reference: Fang et al. "Incorporating Learnable Membrane Time Constant to
    Enhance Learning of Spiking Neural Networks" (ICCV 2021).
    """
    def __init__(self, in_features, out_features, tau_init=0.5, vth=1.0, atan_alpha=2.0):
        super().__init__()
        self.fc          = nn.Linear(in_features, out_features, bias=False)
        self.bn          = nn.BatchNorm1d(out_features)
        self.out_features = out_features
        self.vth         = vth
        self.atan_alpha  = atan_alpha
        # tau encoded as w = 1/tau - 1  (ensures tau in (0,1) via sigmoid-like inversion)
        w_init = 1.0 / max(tau_init, 1e-3) - 1.0
        self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float32))
        self.register_buffer("mem",         None, persistent=False)
        self.register_buffer("spike_count", torch.tensor(0.0), persistent=False)
        self.register_buffer("step_count",  torch.tensor(0),   persistent=False)

    @property
    def tau(self):
        return 1.0 / (1.0 + self.w.exp())   # in (0, 1)

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem         = torch.zeros(batch_size, self.out_features, device=dev)
        self.spike_count = torch.tensor(0.0, device=dev)
        self.step_count  = torch.tensor(0,   device=dev)
        self.batch_size  = batch_size

    def firing_rate(self):
        if self.step_count == 0:
            return 0.0
        denom = self.out_features * self.step_count * getattr(self, "batch_size", 1)
        return (self.spike_count / denom).item()

    def forward(self, x):
        cur      = self.bn(self.fc(x))
        self.mem = self.tau * self.mem + cur
        spk      = atan_spike(self.mem - self.vth, self.atan_alpha)
        self.mem = self.mem * (1.0 - spk.detach())
        self.spike_count = self.spike_count + spk.detach().sum()
        self.step_count  = self.step_count + 1
        return spk, self.mem


class BNPLIFLayer(PLIFLayer):
    """Alias kept for backwards compatibility — same as PLIFLayer."""
    pass

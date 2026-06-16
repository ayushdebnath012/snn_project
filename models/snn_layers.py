import torch
import torch.nn as nn

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Preserve input dtype so AMP stays in float16 through the temporal loop.
        out = (x > 0).to(x.dtype)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Compute surrogate gradient in float32 for numerical stability,
        # then cast back to match the incoming gradient dtype.
        grad = 1.0 / (1.0 + torch.abs(x.float())) ** 2
        return grad_output * grad.to(grad_output.dtype)

spike_fn = SurrogateSpike.apply

class LIFLayer(nn.Module):
    def __init__(self, in_features, out_features, tau=0.9):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.register_buffer("mem", None)

    def reset_state(self, batch_size, device=None):
        dev = device if device else next(self.fc.parameters()).device
        self.mem = torch.zeros(batch_size, self.fc.out_features, device=dev)

    def forward(self, x):
        cur = self.fc(x)
        self.mem = self.tau * self.mem + cur
        spk = spike_fn(self.mem - 1.0)
        self.mem = self.mem * (1 - spk)
        return spk, self.mem


class LIFRecurrentStack(nn.Module):
    """Stateless multi-layer LIF recurrence driven one timestep at a time."""

    def __init__(self, dim: int, num_layers: int = 1, leak: float = 0.9,
                 threshold: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(max(1, int(num_layers)))]
        )
        self.leak = float(leak)
        self.threshold = float(threshold)

    def forward_step(self, x: torch.Tensor,
                     membranes: list[torch.Tensor] | None = None):
        if membranes is None:
            membranes = [torch.zeros_like(x) for _ in self.layers]

        next_membranes = []
        spikes = []
        for layer, previous in zip(self.layers, membranes):
            membrane = self.leak * previous + layer(x)
            spike = spike_fn(membrane - self.threshold)
            membrane = membrane - spike * self.threshold
            x = membrane
            next_membranes.append(membrane)
            spikes.append(spike)
        return x, next_membranes, spikes

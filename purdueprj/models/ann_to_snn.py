"""
ann_to_snn.py
=============
ANN → SNN conversion via threshold balancing (weight-based).

Method (standard pipeline — see "Conversion of Continuous-Valued Deep Networks
to Efficient Event-Driven Networks", Diehl et al. 2016 and follow-up work):

  1. Train an ANN normally (ReLU activations).
  2. Replace every ReLU with an IF neuron (integrate-and-fire, no leak).
  3. Calibrate firing thresholds per layer:
       threshold_l = max activation reaching layer l over calibration set.
     This ensures the firing rate can represent the full activation range.
  4. Run inference for T timesteps with rate-coded inputs:
       x_input[t] ~ Poisson(rate=pixel_value/max_pixel)  (images)
       or directly feed the same static feature every timestep (point clouds).
  5. Read out: predicted class = argmax over accumulated membrane potential
     at the last layer.

For our point cloud pipeline:
  - Input is a point cloud slice [B, N, 3] — treated as a static frame.
  - We feed the same slice T times (rate = 1 per timestep for all pts).
  - The IF neurons integrate and fire; outputs accumulate over timesteps.
  - We compare this converted SNN against our native SNN and the source ANN.

Usage:
  from models.ann_to_snn import convert_ann_to_snn, run_converted_snn

  # 1. Train your ANN
  ann = DGCNNLite(num_classes=40); train(ann, ...)

  # 2. Convert
  snn = convert_ann_to_snn(ann, calibration_loader, device, T=8)

  # 3. Inference
  acc = eval_converted_snn(snn, val_loader, device, T=8)
"""

import torch
import torch.nn as nn
import copy


# ---------------------------------------------------------------------------
# IF neuron — no leak, hard reset, threshold = 1.0 (normalised by calibration)
# ---------------------------------------------------------------------------

class IFNeuron(nn.Module):
    """
    Integrate-and-Fire neuron for ANN→SNN conversion.

    Unlike LIF, there is NO leak (τ=∞), matching the ReLU's summing behaviour
    when averaged over T timesteps.

    Threshold is set to 1.0 (normalised after threshold balancing).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("mem", None, persistent=False)
        self.vth = 1.0

    def reset(self, like):
        self.mem = torch.zeros_like(like)

    def forward(self, x):
        if self.mem is None or self.mem.shape != x.shape:
            self.reset(x)
        self.mem = self.mem + x
        # Hard-reset spike
        spk = (self.mem >= self.vth).float()
        self.mem = self.mem - spk * self.vth
        return spk


# ---------------------------------------------------------------------------
# ReLU → IFNeuron replacement
# ---------------------------------------------------------------------------

def _replace_relu_with_if(module):
    """
    Recursively replace all ReLU (and LeakyReLU, GELU) activations
    in a module with IFNeuron. Returns a new module (modifies in-place on
    a deep copy).
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.GELU)):
            setattr(module, name, IFNeuron())
        else:
            _replace_relu_with_if(child)
    return module


# ---------------------------------------------------------------------------
# Threshold balancing via calibration forward pass
# ---------------------------------------------------------------------------

class _ActivationRecorder(nn.Module):
    """Hook wrapper that records max activation after each replaced layer."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.max_act = 0.0

    def forward(self, x):
        out = self.layer(x)
        self.max_act = max(self.max_act, out.abs().max().item())
        return out


def calibrate_thresholds(snn_model, calibration_loader, device, n_batches=10):
    """
    Run N calibration batches through the (already converted) SNN model
    with IF neurons, recording the maximum pre-threshold activation at each
    IF layer.  Then rescale weights entering each IF layer so the max
    activation equals the IF threshold (= 1.0 after normalisation).

    This is the "weight normalisation" approach (Diehl 2015, Rueckauer 2017).

    Steps:
      - Insert recording wrappers before each IFNeuron
      - Run calibration data (no spikes yet — use raw linear outputs)
      - Divide each layer's weights by its max activation (scale to [0,1])
      - Remove wrappers

    Args:
      snn_model         : model with ReLU replaced by IFNeuron
      calibration_loader: DataLoader yielding (pts, labels)
      device            : torch device
      n_batches         : how many batches to calibrate on

    Returns:
      snn_model with rescaled weights (in-place)
    """
    snn_model.eval()

    # Collect all linear/conv layers paired with subsequent IFNeurons
    # We walk through named modules in order
    layer_pairs = []   # list of (linear_module, if_neuron)
    mods = list(snn_model.named_modules())
    for i, (name, mod) in enumerate(mods):
        if isinstance(mod, IFNeuron):
            # Find the immediately preceding linear/conv layer
            for j in range(i - 1, -1, -1):
                prev_name, prev_mod = mods[j]
                if isinstance(prev_mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    layer_pairs.append((prev_mod, mod))
                    break

    if not layer_pairs:
        print("[ANN→SNN] No (Linear, IFNeuron) pairs found — skipping calibration.")
        return snn_model

    # Replace each IFNeuron temporarily with a pass-through + max recorder
    max_acts = [0.0] * len(layer_pairs)

    def make_hook(idx):
        def hook(module, inp, out):
            max_acts[idx] = max(max_acts[idx], out.detach().abs().max().item())
        return hook

    handles = []
    for i, (lin, if_neu) in enumerate(layer_pairs):
        h = lin.register_forward_hook(make_hook(i))
        handles.append(h)

    # Run calibration (temporarily suppress IF firing)
    orig_forwards = []
    for _, if_neu in layer_pairs:
        orig_forwards.append(if_neu.forward)
        if_neu.forward = lambda x, neu=if_neu: (neu.mem if neu.mem is not None else x) * 0 + x

    with torch.no_grad():
        for batch_i, (pts, _) in enumerate(calibration_loader):
            if batch_i >= n_batches:
                break
            pts = pts.to(device)
            try:
                snn_model.forward_full(pts)
            except Exception:
                try:
                    snn_model.forward_step(pts)
                except Exception:
                    pass

    # Restore IF forward methods
    for i, (_, if_neu) in enumerate(layer_pairs):
        if_neu.forward = orig_forwards[i]
    for h in handles:
        h.remove()

    # Rescale weights: divide by max activation
    scale_prev = 1.0
    for i, (lin, _) in enumerate(layer_pairs):
        if max_acts[i] > 0:
            scale = max_acts[i]
            if hasattr(lin, "weight") and lin.weight is not None:
                lin.weight.data /= scale
            if hasattr(lin, "bias") and lin.bias is not None:
                lin.bias.data /= scale
            print(f"  [Calib] Layer {i}: max_act={scale:.4f} → weights rescaled")
        else:
            print(f"  [Calib] Layer {i}: max_act=0 — skipping")

    return snn_model


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def convert_ann_to_snn(ann_model, calibration_loader, device,
                        T=8, n_calib_batches=10):
    """
    Convert a trained ANN to an SNN via threshold-balanced weight normalisation.

    Args:
      ann_model          : trained ANN with ReLU/GELU activations
      calibration_loader : DataLoader for threshold calibration
      device             : torch device
      T                  : number of inference timesteps (used in eval)
      n_calib_batches    : batches to use for max-activation calibration

    Returns:
      ConvertedSNN — a wrapper with the same reset_state / forward_step
                     interface as PointNetSNN, plus .T attribute.
    """
    print(f"[ANN→SNN] Converting {type(ann_model).__name__} → SNN (T={T})")

    # Deep copy to avoid modifying the original ANN
    snn_model = copy.deepcopy(ann_model)
    snn_model = snn_model.to(device)

    # Replace activations
    _replace_relu_with_if(snn_model)
    print(f"[ANN→SNN] ReLU/GELU → IFNeuron replacements done.")

    # Calibrate thresholds
    print(f"[ANN→SNN] Running threshold calibration ({n_calib_batches} batches)...")
    calibrate_thresholds(snn_model, calibration_loader, device, n_calib_batches)

    # Wrap for standard interface
    return ConvertedSNN(snn_model, T=T)


# ---------------------------------------------------------------------------
# Inference wrapper for converted SNN
# ---------------------------------------------------------------------------

class ConvertedSNN(nn.Module):
    """
    Wraps a converted ANN (with IFNeurons) for rate-coded inference.

    Rate coding for point clouds:
      - The same static point cloud slice is fed T times.
      - IF neurons integrate over T timesteps.
      - Final logits = sum of spike outputs over T (vote counting).

    This is the simplest and most common ANN→SNN inference scheme.
    """
    def __init__(self, converted_model, T=8):
        super().__init__()
        self.model = converted_model
        self.T = T
        self._if_neurons = [m for m in converted_model.modules()
                            if isinstance(m, IFNeuron)]

    def _reset_if_neurons(self):
        for neu in self._if_neurons:
            neu.mem = None

    def reset_state(self, batch_size=None, device=None):
        self._reset_if_neurons()

    def forward_step(self, pts_slice):
        """
        Rate-coded inference: feed pts_slice T times, return accumulated logits.
        pts_slice : [B, N, 3]
        Returns   : logits [B, num_classes]
        """
        self._reset_if_neurons()
        acc_logits = None
        for _ in range(self.T):
            out = self.model.forward_step(pts_slice)
            acc_logits = out if acc_logits is None else acc_logits + out
        return acc_logits / self.T

    def forward_full(self, pts):
        """Same as forward_step but calls forward_full."""
        self._reset_if_neurons()
        acc_logits = None
        for _ in range(self.T):
            out = self.model.forward_full(pts)
            acc_logits = out if acc_logits is None else acc_logits + out
        return acc_logits / self.T

    def forward(self, pts):
        return self.forward_full(pts)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_converted_snn(converted_snn, val_loader, device, T=None):
    """
    Evaluate a ConvertedSNN on a DataLoader.
    Returns accuracy (float).
    """
    if T is not None:
        converted_snn.T = T
    converted_snn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pts, labels in val_loader:
            pts    = pts.to(device)
            labels = labels.to(device).long()
            converted_snn.reset_state()
            logits = converted_snn.forward_full(pts)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

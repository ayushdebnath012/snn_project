"""
metrics.py
==========
Accuracy, spike-rate, and energy-efficiency metrics.

Energy constants:
  Horowitz 2014 (theoretical 45nm CMOS):
    E_MAC = 4.6 pJ,  E_AC = 0.9 pJ  → ratio 5.1×

  Lemaire et al. 2022 (hardware-measured, Intel Loihi 2):
    "An Analytical Estimation of Spiking Neural Networks Energy Efficiency"
    https://arxiv.org/abs/2206.10569
    E_MAC ≈ 8.4e-3 pJ,  E_AC ≈ 2.3e-3 pJ  → ratio 3.65×

  Christensen et al. 2022 (Intel Loihi benchmarks):
    "2022 roadmap on neuromorphic computing and engineering"
    Confirms AC-only operation provides 3–8× energy savings over GPU inference.

Use `efficiency_ratio(..., hardware="loihi")` for paper results.
"""

import torch

# ---------------------------------------------------------------------------
# Energy constants
# ---------------------------------------------------------------------------

ENERGY = {
    "45nm": {
        "E_MAC": 4.6,     # pJ — Horowitz 2014
        "E_AC":  0.9,     # pJ
        "ref":   "Horowitz 2014 (45nm CMOS theoretical)",
    },
    "loihi": {
        "E_MAC": 8.4e-3,  # pJ — Lemaire et al. 2022 (Intel Loihi 2)
        "E_AC":  2.3e-3,  # pJ
        "ref":   "Lemaire et al. 2022 (Intel Loihi 2, arXiv:2206.10569)",
    },
}


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def margin(logits):
    probs = logits.softmax(dim=-1)
    top2 = probs.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).mean().item()


# ---------------------------------------------------------------------------
# Novel: Efficiency Metrics
# From notes: "total spike rate = all spikes / total no. of neurons"
#             "Acc vs timestep, ANN efficiency, SNN efficiency, how calc?"
# ---------------------------------------------------------------------------

def spike_rate(model):
    """
    Compute the average firing rate across all LearnableLIF layers.
    Rate = total spikes fired / (total neurons * total timesteps).

    This is the direct proxy for energy efficiency: lower firing rate
    means fewer synaptic events (AC operations) and lower power draw.
    Returns: mean rate across all tracked layers (float), or None if
             model has no LearnableLIF layers.
    """
    rates = []
    if hasattr(model, 'get_firing_rates'):
        r = model.get_firing_rates()
        rates = list(r.values())
    if not rates:
        return None
    return sum(rates) / len(rates)


def efficiency_ratio(model, num_params_ann=None, hardware="loihi"):
    """
    Estimate the SNN vs ANN energy efficiency ratio.

    hardware : "loihi"  → Lemaire et al. 2022 (Intel Loihi 2) [DEFAULT for papers]
               "45nm"   → Horowitz 2014 theoretical 45nm CMOS

    Returns dict:
        {
          'firing_rate'      : mean firing rate f,
          'ann_energy_unit'  : 1.0  (normalised),
          'snn_energy_unit'  : f * (E_AC / E_MAC),
          'speedup'          : E_MAC / (f * E_AC),
          'per_layer_rates'  : dict of per-layer rates,
          'hardware'         : which energy model was used,
          'energy_ref'       : citation string,
          'E_MAC_pJ'         : E_MAC used,
          'E_AC_pJ'          : E_AC used,
        }
    """
    hw    = ENERGY.get(hardware, ENERGY["loihi"])
    E_MAC = hw["E_MAC"]
    E_AC  = hw["E_AC"]

    per_layer = {}
    if hasattr(model, 'get_firing_rates'):
        per_layer = model.get_firing_rates()

    rates = list(per_layer.values())
    f = sum(rates) / len(rates) if rates else 1.0

    snn_energy = f * (E_AC / E_MAC)
    speedup    = E_MAC / (f * E_AC) if f > 0 else float('inf')

    return {
        'firing_rate'     : f,
        'ann_energy_unit' : 1.0,
        'snn_energy_unit' : snn_energy,
        'speedup'         : speedup,
        'per_layer_rates' : per_layer,
        'hardware'        : hardware,
        'energy_ref'      : hw["ref"],
        'E_MAC_pJ'        : E_MAC,
        'E_AC_pJ'         : E_AC,
    }


def learnable_lif_stats(model):
    """
    Return summary stats of learned tau and V_th across all
    LearnableLIF layers. Useful to verify they are actually learning.
    Returns dict: {layer_name: {tau_mean, tau_std, vth_mean, vth_std}}
    """
    from models.snn_layers import LearnableLIFLayer
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, LearnableLIFLayer):
            tau  = module.tau.detach()
            vth  = module.vth.detach()
            stats[name] = {
                'tau_mean' : tau.mean().item(),
                'tau_std'  : tau.std().item(),
                'vth_mean' : vth.mean().item(),
                'vth_std'  : vth.std().item(),
            }
    return stats

"""
loss_active.py
==============
Joint five-term loss for Active Spiking Perception (ASP).

L = L_CE(ŷ_T, y)                                    [final classification]
  + λ_aux  × (1/T) Σ_{t<T} L_CE(ŷ_t, y)             [anytime auxiliary]
  + λ_exit × (1/T) Σ_t (1 - max_softmax(ŷ_t))        [early confidence]
  + λ_fr   × r̄                                        [firing-rate sparsity]
  + λ_div  × geo_diversity_penalty                    [SSP diversity — NEW]

geo_diversity_penalty
---------------------
Penalises the SSP for selecting geometrically similar slices consecutively.
At each timestep t we compute the mean cosine similarity between the selected
anchor's geo descriptor and all previously selected anchors.  Minimising this
encourages the policy to explore structurally diverse regions of the object,
which leads to faster information gain and earlier confident exit.

Term details
------------
L_CE (final):
    Standard cross-entropy on the last timestep's logit.
    Primary accuracy signal.

L_aux (anytime):
    Each intermediate timestep t < T also supervised with cross-entropy.
    Weight λ_aux = 0.3 by default (same as original report).
    This ensures that the model can exit at any timestep and still be accurate.
    Also trains the SSP: if an informative slice is processed early,
    intermediate accuracy is high, reinforcing that ordering.

L_exit (early confidence):
    Penalises low maximum softmax probability at each timestep.
    Minimising (1 - max_p_t) ≡ maximising max_p_t ≡ encouraging confidence.
    A model that reaches high confidence at t=3 instead of t=16 is rewarded.
    Weight λ_exit = 0.1 by default.

L_fr (firing-rate):
    Penalises the mean firing rate r̄ across all LearnableLIF layers.
    This closes the gap seen in ours_base/ours_bidir where r ≈ 0.7.
    With λ_fr = 0.05 and longer training, r converges to ~0.15–0.2.
    Weight λ_fr = 0.05 by default.
"""

import torch
import torch.nn.functional as F


def loss_final(
    logits_T: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy on final timestep logit."""
    return F.cross_entropy(logits_T, labels, label_smoothing=label_smoothing)


def loss_aux(
    logits_all: list[torch.Tensor],
    labels: torch.Tensor,
    lam: float = 0.3,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Anytime auxiliary loss: CE at every intermediate timestep.

    Args:
        logits_all : list of T tensors, each [B, num_classes]
        labels     : [B]
        lam        : weight per intermediate step

    Returns:
        scalar loss
    """
    T = len(logits_all)
    if T <= 1:
        return torch.tensor(0.0, device=labels.device)

    total = torch.tensor(0.0, device=labels.device)
    for t in range(T - 1):    # exclude final (covered by loss_final)
        total = total + F.cross_entropy(
            logits_all[t], labels, label_smoothing=label_smoothing
        )
    return lam * total / (T - 1)


def loss_exit(
    logits_all: list[torch.Tensor],
    lam: float = 0.1,
    progress: float = 0.0,
) -> torch.Tensor:
    """
    Progressive early-confidence loss.

    The effective weight ramps from 0.5×lam at epoch 0 to lam at the final
    epoch. This ensures the model first learns to classify correctly before
    being pushed to exit early, preventing early over-confidence collapse.

    Args:
        logits_all : list of T tensors, each [B, num_classes]
        lam        : base weight
        progress   : float in [0, 1] — epoch / max_epochs

    Returns:
        scalar loss
    """
    effective_lam = lam * (0.5 + 0.5 * progress)
    T = len(logits_all)
    total = torch.tensor(0.0, device=logits_all[0].device)

    for t, logits_t in enumerate(logits_all):
        max_prob = F.softmax(logits_t, dim=-1).max(dim=-1).values   # [B]
        weight_t = (T - t) / T
        total = total + weight_t * (1.0 - max_prob).mean()

    return effective_lam * total / T


def loss_diversity(
    geo_descriptors: torch.Tensor,
    selection_weights: list,
    lam: float = 0.05,
) -> torch.Tensor:
    """
    SSP diversity loss: penalise selecting geometrically similar anchors.

    Computes mean cosine similarity between consecutively selected anchor
    geometry vectors (via soft Gumbel weights). Minimising this encourages
    the SSP to spread selections across structurally different regions.

    Args:
        geo_descriptors   : [B, M, G]  per-anchor 9-D geometry
        selection_weights : list of T [B, M] Gumbel-softmax weights
        lam               : weight

    Returns:
        scalar diversity penalty
    """
    if lam <= 0 or len(selection_weights) < 2:
        return torch.tensor(0.0, device=geo_descriptors.device)

    geo_norm = F.normalize(geo_descriptors, dim=-1)        # [B, M, G]
    total    = torch.tensor(0.0, device=geo_descriptors.device)
    cumul    = torch.zeros_like(geo_norm[:, 0, :])         # [B, G]

    for t, w_t in enumerate(selection_weights):
        sel_geo = (w_t.unsqueeze(-1) * geo_norm).sum(dim=1)  # [B, G]
        sel_geo = F.normalize(sel_geo, dim=-1)
        if t > 0:
            prior = F.normalize(cumul / t, dim=-1)
            total = total + (sel_geo * prior).sum(dim=-1).mean()
        cumul = cumul + sel_geo.detach()

    return lam * total / max(len(selection_weights) - 1, 1)


def loss_firing_rate(
    model,
    lam: float = 0.05,
) -> torch.Tensor:
    """
    Firing-rate regularisation: penalise mean spike rate across all layers.

    Args:
        model : ActiveSNN instance (must implement mean_firing_rate())
        lam   : weight

    Returns:
        scalar loss (or 0.0 if model has no firing rate tracking)
    """
    if not hasattr(model, "mean_firing_rate"):
        return torch.tensor(0.0)
    r = model.mean_firing_rate()
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, dtype=torch.float32)
    return lam * r


def active_loss(
    logits_final: torch.Tensor,
    logits_all: list[torch.Tensor],
    labels: torch.Tensor,
    model,
    lam_aux: float  = 0.3,
    lam_exit: float = 0.1,
    lam_fr: float   = 0.05,
    lam_div: float  = 0.05,
    label_smoothing: float = 0.0,
    progress: float = 0.0,
    geo_descriptors: torch.Tensor = None,
    selection_weights: list = None,
) -> tuple[torch.Tensor, dict]:
    """
    Full joint five-term loss for Active Spiking Perception.

    Args:
        logits_final      : [B, C]           final timestep logit
        logits_all        : list[T × [B, C]] all timestep logits
        labels            : [B]
        model             : ActiveSNN
        lam_aux           : weight for anytime auxiliary loss
        lam_exit          : base weight for progressive early-confidence loss
        lam_fr            : weight for firing-rate regularisation
        lam_div           : weight for SSP diversity penalty
        label_smoothing   : label smoothing for CE terms
        progress          : float in [0, 1] — epoch / max_epochs (for exit ramp)
        geo_descriptors   : [B, M, G]  geometry descriptors (for diversity loss)
        selection_weights : list of T [B, M] Gumbel weights (for diversity loss)

    Returns:
        total_loss : scalar tensor
        breakdown  : dict with individual loss values (for logging)
    """
    l_ce   = loss_final(logits_final, labels, label_smoothing=label_smoothing)
    l_aux  = loss_aux(
        logits_all, labels, lam=lam_aux, label_smoothing=label_smoothing
    )
    l_exit = loss_exit(logits_all, lam=lam_exit, progress=progress)
    l_fr   = loss_firing_rate(model, lam=lam_fr)

    device = logits_final.device
    l_fr   = l_fr.to(device)

    if (lam_div > 0 and geo_descriptors is not None
            and selection_weights is not None and len(selection_weights) >= 2):
        l_div = loss_diversity(geo_descriptors, selection_weights, lam=lam_div)
    else:
        l_div = torch.tensor(0.0, device=device)

    total  = l_ce + l_aux + l_exit + l_fr + l_div

    breakdown = {
        "loss_ce":    l_ce.item(),
        "loss_aux":   l_aux.item(),
        "loss_exit":  l_exit.item(),
        "loss_fr":    l_fr.item() if isinstance(l_fr, torch.Tensor) else float(l_fr),
        "loss_div":   l_div.item(),
        "loss_total": total.item(),
    }

    return total, breakdown

"""
active_inference.py
===================
Inference utilities for Active Spiking Perception (ASP).

Provides:
  1. active_eval_dataset  — full evaluation with SSP on a DataLoader
  2. pareto_curve         — threshold sweep → energy–accuracy Pareto frontier
  3. visualise_attention  — per-class SSP attention map (for qualitative analysis)
  4. compare_orderings    — compare SSP ordering vs fixed (FPS/radial) ordering

The key metric is the Energy-Accuracy Pareto frontier:
  x-axis: E_SNN / E_ANN (energy ratio, lower = better)
  y-axis: Accuracy (%)
  Each point = one (threshold, mean_exit_time) operating point.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# -----------------------------------------------------------------------
# Energy model (Lemaire et al. 2022, Loihi 2 constants)
# -----------------------------------------------------------------------

E_AC  = 2.3e-3   # pJ, spike-triggered accumulate on Loihi 2
E_MAC = 8.4e-3   # pJ, multiply-accumulate (ANN reference)
EFFICIENCY_RATIO = E_AC / E_MAC   # = 0.274


def energy_ratio(firing_rate: float, exit_fraction: float) -> float:
    """
    Compute SNN/ANN energy ratio.

    E_SNN / E_ANN = r × (E_AC / E_MAC) × (T_exit / T)

    Args:
        firing_rate   : mean firing rate r ∈ [0, 1]
        exit_fraction : T_exit / T ∈ [0, 1]  (fraction of slices consumed)

    Returns:
        ratio : float  (< 1 means SNN is cheaper, e.g. 0.07 = 14× cheaper)
    """
    return firing_rate * EFFICIENCY_RATIO * exit_fraction


# -----------------------------------------------------------------------
# Full dataset evaluation
# -----------------------------------------------------------------------

def active_eval_dataset(
    model,
    dataset,
    device,
    num_slices: int = 16,
    threshold: float = 0.7,
    prepare_fn=None,
) -> dict:
    """
    Evaluate ActiveSNN on a dataset (list of (pts, label) pairs).

    Args:
        model      : ActiveSNN
        dataset    : iterable of (pts [N,3], label scalar)
        device     : torch device
        num_slices : T
        threshold  : margin threshold for early exit
        prepare_fn : function(pts_batch, T) → (pts_slices, geo, ...)
                     If None, uses train_active.prepare_fps_slices_and_geo

    Returns:
        metrics : dict
    """
    if prepare_fn is None:
        from training.train_active import prepare_fps_slices_and_geo
        prepare_fn = prepare_fps_slices_and_geo

    model.eval()

    correct       = 0
    total         = 0
    exit_steps    = []
    firing_rates  = []
    slice_orders  = []   # record order each sample was processed

    with torch.no_grad():
        for pts, label in dataset:
            if pts.dim() == 2:
                pts = pts.unsqueeze(0)    # [1, N, 3]
            pts   = pts.to(device)
            label = label if isinstance(label, int) else label.item()

            pts_slices, geo, _, _ = prepare_fn(pts, T=num_slices)

            logits, exit_step, order = model.forward_active_infer(
                pts_slices, geo, threshold=threshold
            )

            pred = logits.argmax(dim=-1).item()
            correct += int(pred == label)
            total   += 1

            exit_steps.append(exit_step)
            slice_orders.append(order)

            fr = model.mean_firing_rate()
            firing_rates.append(fr if isinstance(fr, float) else fr.item())

    mean_exit = np.mean(exit_steps)
    mean_fr   = np.mean(firing_rates)
    acc       = correct / max(total, 1)
    e_ratio   = energy_ratio(mean_fr, mean_exit / num_slices)

    return {
        "accuracy":     acc,
        "mean_exit":    mean_exit,
        "mean_fr":      mean_fr,
        "energy_ratio": e_ratio,
        "savings":      1.0 / max(e_ratio, 1e-9),
        "exit_steps":   exit_steps,
        "slice_orders": slice_orders,
        "n_samples":    total,
    }


# -----------------------------------------------------------------------
# Pareto curve: threshold sweep
# -----------------------------------------------------------------------

def pareto_curve(
    model,
    dataset,
    device,
    num_slices: int = 16,
    thresholds: list = None,
    prepare_fn=None,
) -> list[dict]:
    """
    Generate the energy–accuracy Pareto frontier by sweeping exit threshold θ.

    At θ=0: always exit at t=1 (minimum energy, low accuracy)
    At θ=1: never exit early (maximum energy = fixed-order baseline)

    Args:
        thresholds : list of floats ∈ [0, 1]  (default: 21 evenly spaced)

    Returns:
        curve : list of dicts, sorted by energy_ratio ascending
    """
    if thresholds is None:
        thresholds = [round(i * 0.05, 2) for i in range(21)]

    if prepare_fn is None:
        from training.train_active import prepare_fps_slices_and_geo
        prepare_fn = prepare_fps_slices_and_geo

    curve = []
    for theta in thresholds:
        m = active_eval_dataset(
            model, dataset, device,
            num_slices=num_slices, threshold=theta, prepare_fn=prepare_fn
        )
        m["threshold"] = theta
        curve.append(m)
        print(
            f"  θ={theta:.2f}  acc={m['accuracy']:.4f}  "
            f"mean_exit={m['mean_exit']:.2f}/{num_slices}  "
            f"energy={m['energy_ratio']:.4f}  "
            f"savings={m['savings']:.1f}×"
        )

    # Sort by energy ratio (low to high) to get the frontier ordering
    curve.sort(key=lambda x: x["energy_ratio"])
    return curve


# -----------------------------------------------------------------------
# SSP attention visualisation
# -----------------------------------------------------------------------

def visualise_attention(
    model,
    dataset,
    device,
    class_names: list,
    num_slices: int = 16,
    n_samples_per_class: int = 10,
    prepare_fn=None,
) -> dict:
    """
    For each class, record which slices the SSP selects first across
    n_samples_per_class examples. Returns per-class anchor priority maps.

    This reveals what the model "looks at first" for each object category.
    Use the returned maps for qualitative analysis and paper visualisations.

    Returns:
        attention_maps : {class_name: [T] array of mean first-selection frequency}
    """
    if prepare_fn is None:
        from training.train_active import prepare_fps_slices_and_geo
        prepare_fn = prepare_fps_slices_and_geo

    model.eval()

    class_orders  = defaultdict(list)   # {class_idx: [[order1], [order2], ...]}
    class_counts  = defaultdict(int)

    with torch.no_grad():
        for pts, label in dataset:
            label_idx = label if isinstance(label, int) else label.item()
            if class_counts[label_idx] >= n_samples_per_class:
                continue

            if pts.dim() == 2:
                pts = pts.unsqueeze(0)
            pts = pts.to(device)

            pts_slices, geo, fps_anchors, _ = prepare_fn(pts, T=num_slices)

            _, _, order = model.forward_active_infer(
                pts_slices, geo, threshold=0.0    # θ=0 → never exit early → get full order
            )
            class_orders[label_idx].append(order)
            class_counts[label_idx] += 1

    # Build attention maps: for each class, what is the mean position
    # of each anchor in the selection order? (position 0 = first, best)
    attention_maps = {}
    for cls_idx, orders in class_orders.items():
        priority = np.zeros(num_slices)
        for order in orders:
            for rank, anchor_idx in enumerate(order):
                # Lower rank → selected earlier → higher priority score
                priority[anchor_idx] += (num_slices - rank)
        priority /= (len(orders) * num_slices)   # normalise to [0, 1]
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        attention_maps[name] = priority

    return attention_maps


# -----------------------------------------------------------------------
# Comparison: SSP ordering vs fixed ordering
# -----------------------------------------------------------------------

def compare_orderings(
    model,
    dataset,
    device,
    num_slices: int = 16,
    threshold: float = 0.7,
    n_samples: int = 200,
    prepare_fn=None,
) -> dict:
    """
    Compare three strategies:
      1. SSP (adaptive, learned policy)
      2. Random ordering (baseline: random permutation each sample)
      3. Fixed FPS ordering (t=0,1,...,T-1, standard approach)

    Returns:
        comparison : dict with acc, mean_exit, energy for each strategy
    """
    if prepare_fn is None:
        from training.train_active import prepare_fps_slices_and_geo
        prepare_fn = prepare_fps_slices_and_geo

    model.eval()

    results = {
        "ssp":    {"correct": 0, "total": 0, "exit_sum": 0},
        "random": {"correct": 0, "total": 0, "exit_sum": 0},
        "fixed":  {"correct": 0, "total": 0, "exit_sum": 0},
    }

    count = 0
    with torch.no_grad():
        for pts, label in dataset:
            if count >= n_samples:
                break
            label_idx = label if isinstance(label, int) else label.item()

            if pts.dim() == 2:
                pts = pts.unsqueeze(0)
            pts = pts.to(device)

            pts_slices, geo, _, _ = prepare_fn(pts, T=num_slices)

            # --- SSP ---
            logits, exit_step, _ = model.forward_active_infer(
                pts_slices, geo, threshold=threshold
            )
            pred = logits.argmax(-1).item()
            results["ssp"]["correct"] += int(pred == label_idx)
            results["ssp"]["total"]   += 1
            results["ssp"]["exit_sum"] += exit_step

            # --- Fixed ordering (t=0,1,...,T-1) ---
            model.reset_state(1, device)
            last_logits = None
            fixed_exit  = num_slices
            for t in range(num_slices):
                sl = pts_slices[:, t, :, :]
                model.backbone.reset_state(1, device)
                fp = model.backbone(sl).mean(dim=1)
                lg = model.temporal(fp)
                last_logits = lg
                probs = F.softmax(lg, dim=-1)
                top2  = probs.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).item()
                if margin > threshold:
                    fixed_exit = t + 1
                    break
            pred_fixed = last_logits.argmax(-1).item()
            results["fixed"]["correct"] += int(pred_fixed == label_idx)
            results["fixed"]["total"]   += 1
            results["fixed"]["exit_sum"] += fixed_exit

            # --- Random ordering ---
            perm = torch.randperm(num_slices)
            model.reset_state(1, device)
            last_logits = None
            rand_exit   = num_slices
            for t in range(num_slices):
                sl = pts_slices[:, perm[t].item(), :, :]
                model.backbone.reset_state(1, device)
                fp = model.backbone(sl).mean(dim=1)
                lg = model.temporal(fp)
                last_logits = lg
                probs = F.softmax(lg, dim=-1)
                top2  = probs.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).item()
                if margin > threshold:
                    rand_exit = t + 1
                    break
            pred_rand = last_logits.argmax(-1).item()
            results["random"]["correct"] += int(pred_rand == label_idx)
            results["random"]["total"]   += 1
            results["random"]["exit_sum"] += rand_exit

            count += 1

    summary = {}
    for strategy, r in results.items():
        n = max(r["total"], 1)
        mean_exit = r["exit_sum"] / n
        fr = model.mean_firing_rate()
        e_r = energy_ratio(fr, mean_exit / num_slices)
        summary[strategy] = {
            "accuracy":     r["correct"] / n,
            "mean_exit":    mean_exit,
            "energy_ratio": e_r,
            "savings":      1.0 / max(e_r, 1e-9),
        }

    print("\n=== Ordering Strategy Comparison ===")
    for strategy, m in summary.items():
        print(
            f"  {strategy:8s}  acc={m['accuracy']:.4f}  "
            f"mean_exit={m['mean_exit']:.2f}  "
            f"savings={m['savings']:.1f}×"
        )

    return summary

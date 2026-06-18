"""
train_active.py
===============
Training loop for Active Spiking Perception (ASP).

Key differences from the original train_loop.py:
  1. Uses ActiveSNN.forward_active_train() instead of forward_step() loop.
  2. Precomputes geometry descriptors for FPS anchors.
  3. Uses the 4-term active_loss() instead of CE + aux.
  4. Anneals Gumbel temperature τ each epoch.
  5. Logs per-term loss breakdown + firing rates + policy entropy.

Gumbel annealing schedule:
    τ(epoch) = max(τ_min, τ_0 × exp(-anneal_rate × epoch))
    Default: τ_0=1.0, τ_min=0.1, anneal_rate=0.05
    → τ reaches 0.1 after ~46 epochs (essentially hard selection)
"""

import torch
import torch.nn.functional as F
import time
import math
import numpy as np

from data.slicing import slice_fps_hierarchical_batch
from data.modelnet import augment_point_cloud
from training.loss_active import active_loss
from training.metrics import accuracy
from models.slice_selection_policy import compute_geometry_descriptors


# -----------------------------------------------------------------------
# Gumbel temperature schedule
# -----------------------------------------------------------------------

def gumbel_tau(epoch: int, tau_0: float = 1.0, tau_min: float = 0.1,
               anneal_rate: float = 0.05) -> float:
    return max(tau_min, tau_0 * math.exp(-anneal_rate * epoch))


# -----------------------------------------------------------------------
# FPS slice + geometry descriptor helper
# -----------------------------------------------------------------------

def prepare_fps_slices_and_geo(pts: torch.Tensor, T: int):
    """
    FPS-slice a batch of point clouds and compute geometry descriptors.

    Args:
        pts : [B, N, 3]
        T   : number of temporal slices

    Returns:
        pts_slices      : [B, T, N//T, 3]
        geo_descriptors : [B, T, 6]
        fps_anchors     : [B, T, 3]
        assignments     : [B, N]
    """
    B, N, _ = pts.shape
    n_per_slice = N // T
    device = pts.device

    # FPS hierarchical slicing (from existing data/slicing.py)
    # Returns [B, T, n_per_slice, 3]
    pts_slices = slice_fps_hierarchical_batch(pts, T=T)   # [B, T, n, 3]

    # Compute FPS anchor as centroid of each slice
    fps_anchors = pts_slices.mean(dim=2)                  # [B, T, 3]

    # Build approximate assignment: each point → nearest anchor
    # [B, N, 1, 3] - [B, 1, T, 3] → [B, N, T]
    diffs = pts.unsqueeze(2) - fps_anchors.unsqueeze(1)
    dists = (diffs ** 2).sum(-1)                          # [B, N, T]
    assignments = dists.argmin(dim=-1)                    # [B, N]

    # Geometry descriptors
    geo = compute_geometry_descriptors(pts, fps_anchors, assignments)  # [B, T, 9]

    return pts_slices, geo, fps_anchors, assignments


def aggregate_logits(logits: torch.Tensor, logits_all=None, last_k: int = 1):
    """Average the last K timestep logits for steadier full-slice accuracy."""
    if logits_all is None or last_k <= 1:
        return logits
    k = min(int(last_k), len(logits_all))
    return torch.stack(logits_all[-k:], dim=0).mean(dim=0)


# -----------------------------------------------------------------------
# One training epoch
# -----------------------------------------------------------------------

def train_active_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch: int,
    max_epochs: int = 200,
    num_slices: int = 16,
    lam_aux:  float = 0.3,
    lam_exit: float = 0.1,
    lam_fr:   float = 0.05,
    lam_div:  float = 0.05,
    label_smoothing: float = 0.0,
    tau_0:    float = 1.0,
    tau_min:  float = 0.1,
    anneal_rate: float = 0.05,
    verbose_every: int = 20,
) -> dict:
    """
    Run one epoch of active SNN training.

    Returns:
        metrics : dict with mean losses, accuracy, firing rate, policy entropy
    """
    model.train()

    progress = epoch / max(max_epochs - 1, 1)   # [0, 1]

    tau = gumbel_tau(epoch, tau_0, tau_min, anneal_rate)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)

    total_ce   = 0.0
    total_aux  = 0.0
    total_exit = 0.0
    total_fr   = 0.0
    total_div  = 0.0
    total_tot  = 0.0
    total_acc  = 0.0
    total_acc1 = 0.0
    total_ent  = 0.0
    count = 0

    start = time.time()

    for batch_idx, (pts, labels) in enumerate(dataloader):
        pts    = pts.to(device)
        labels = labels.to(device)
        B      = pts.size(0)

        pts_slices, geo, fps_anchors, assignments = prepare_fps_slices_and_geo(
            pts, T=num_slices
        )

        logits_final, logits_all, selection_weights = model.forward_active_train(
            pts_slices, geo
        )

        loss, breakdown = active_loss(
            logits_final, logits_all, labels, model,
            lam_aux=lam_aux, lam_exit=lam_exit, lam_fr=lam_fr, lam_div=lam_div,
            label_smoothing=label_smoothing,
            progress=progress,
            geo_descriptors=geo,
            selection_weights=selection_weights,
        )

        if not torch.isfinite(loss):
            print(f"  [SKIP] batch {batch_idx}: non-finite loss={loss.item():.2f}, skipping")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            mem_zero = torch.zeros(B, model.temporal_dim, device=device)
            scores = model.ssp(mem_zero, geo, visited_mask=None)
            policy_probs = torch.softmax(scores, dim=-1)
            policy_ent = -(policy_probs * (policy_probs + 1e-9).log()).sum(dim=-1).mean()

        total_ce   += breakdown["loss_ce"]
        total_aux  += breakdown["loss_aux"]
        total_exit += breakdown["loss_exit"]
        total_fr   += breakdown["loss_fr"]
        total_div  += breakdown["loss_div"]
        total_tot  += breakdown["loss_total"]
        total_acc  += accuracy(logits_final, labels)
        total_acc1 += accuracy(logits_all[0], labels)
        total_ent  += policy_ent.item()
        count += 1

        if (batch_idx + 1) % verbose_every == 0:
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{batch_idx+1}/{len(dataloader)}] "
                f"CE={breakdown['loss_ce']:.4f}  "
                f"Aux={breakdown['loss_aux']:.4f}  "
                f"Exit={breakdown['loss_exit']:.4f}  "
                f"Div={breakdown['loss_div']:.4f}  "
                f"FR={breakdown['loss_fr']:.4f}  "
                f"AccEnd={total_acc/count:.3f}  "
                f"Acc1={total_acc1/count:.3f}  "
                f"PolicyEnt={total_ent/count:.3f}  "
                f"τ={tau:.3f}  prog={progress:.2f}  LR={lr:.6f}  {elapsed:.0f}s"
            )

    n = max(count, 1)
    return {
        "loss_ce":    total_ce   / n,
        "loss_aux":   total_aux  / n,
        "loss_exit":  total_exit / n,
        "loss_fr":    total_fr   / n,
        "loss_div":   total_div  / n,
        "loss_total": total_tot  / n,
        "acc_final":  total_acc  / n,
        "acc_first":  total_acc1 / n,
        "policy_entropy": total_ent / n,
        "gumbel_tau": tau,
        "progress":   progress,
    }


# -----------------------------------------------------------------------
# Validation (fixed-order, no SSP, for clean eval)
# -----------------------------------------------------------------------

def validate_active(
    model,
    dataloader,
    device,
    num_slices: int = 16,
    threshold: float = 0.7,
    logit_ensemble: int = 1,
    tta_votes: int = 1,
) -> dict:
    """
    Validate with active inference (SSP + early exit).

    Args:
        tta_votes : number of TTA random-rotation votes (1 = no TTA)

    Returns:
        metrics : dict with accuracy, mean exit step, mean firing rate
    """
    model.eval()

    correct = 0
    total   = 0
    total_exit_step = 0.0
    total_fr = 0.0
    count = 0

    with torch.no_grad():
        for pts, labels in dataloader:
            pts    = pts.to(device)
            labels = labels.to(device)
            B      = pts.size(0)

            for b in range(B):
                pts_np = pts[b].cpu().numpy()   # [N, 3]
                lbl_b  = labels[b].unsqueeze(0)

                vote_logits = []
                total_exit_v = 0.0

                for _ in range(max(tta_votes, 1)):
                    if tta_votes > 1:
                        aug_np = augment_point_cloud(pts_np, mode="vote")
                        pts_v  = torch.from_numpy(aug_np).float().to(device)
                    else:
                        pts_v = pts[b]

                    pts_in = pts_v.unsqueeze(0)  # [1, N, 3]
                    pts_slices_v, geo_v, _, _ = prepare_fps_slices_and_geo(
                        pts_in, T=num_slices
                    )

                    out = model.forward_active_infer(
                        pts_slices_v, geo_v, threshold=threshold,
                        return_all=logit_ensemble > 1,
                    )
                    if len(out) == 4:
                        logits_v, exit_step, _, logits_all_v = out
                    else:
                        logits_v, exit_step, _ = out
                        logits_all_v = None

                    logits_v = aggregate_logits(logits_v, logits_all_v, last_k=logit_ensemble)
                    vote_logits.append(logits_v)
                    total_exit_v += exit_step

                logits = torch.stack(vote_logits, dim=0).mean(dim=0)
                pred = logits.argmax(dim=-1)
                correct += (pred == lbl_b).sum().item()
                total   += 1
                total_exit_step += total_exit_v / max(tta_votes, 1)

            fr = model.mean_firing_rate()
            total_fr += fr if isinstance(fr, float) else fr.item()
            count += 1

    n = max(total, 1)
    energy_ratio_value = (total_fr / max(count, 1)) * 0.274 * (total_exit_step / n / num_slices)
    return {
        "acc":        correct / n,
        "mean_exit":  total_exit_step / n,
        "mean_fr":    total_fr / max(count, 1),
        "energy_ratio": energy_ratio_value,
        "savings": 1.0 / max(energy_ratio_value, 1e-9),
    }


# -----------------------------------------------------------------------
# Threshold sweep for Pareto curve
# -----------------------------------------------------------------------

def sweep_threshold(
    model,
    dataloader,
    device,
    num_slices: int = 16,
    thresholds: list = None,
) -> list[dict]:
    """
    Evaluate model at multiple exit thresholds to produce the Pareto curve.

    Args:
        thresholds : list of float thresholds to try (default: 21 values 0..1)

    Returns:
        results : list of dicts, each with {'threshold', 'acc', 'mean_exit', 'energy_ratio'}
    """
    if thresholds is None:
        thresholds = [i / 20.0 for i in range(21)]    # 0.0, 0.05, ..., 1.0

    results = []
    for theta in thresholds:
        metrics = validate_active(
            model, dataloader, device, num_slices=num_slices, threshold=theta
        )
        metrics["threshold"] = theta
        results.append(metrics)
        print(
            f"  θ={theta:.2f}  acc={metrics['acc']:.4f}  "
            f"mean_exit={metrics['mean_exit']:.2f}/{num_slices}  "
            f"energy={metrics['energy_ratio']:.4f}"
        )

    return results

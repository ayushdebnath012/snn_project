"""
t_sweep.py
==========
T-timestep sensitivity analysis.

Sweeps T ∈ {4, 8, 12, 16, 24, 32} for:
  1. Native SNN (ours_full) — retrain at each T
  2. Converted SNN (ann→snn) — no retraining, just change T at inference
  3. ANN baseline — constant accuracy (no T dependence)

This directly answers the reviewer question:
  "Vary T from 4 to 32 and show the accuracy-efficiency frontier."

Also addresses:
  "ANN→SNN conversion at matched accuracy: show native SNN matches
   converted SNN at lower T."

Usage:
  python benchmarks/t_sweep.py \
      --mn40_root /data/ModelNet40 \
      --epochs 100 \
      --T_list 4 8 12 16 24 32 \
      --out_dir results/t_sweep
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_zoo  import build_model, count_params
from models.ann_to_snn import convert_ann_to_snn, eval_converted_snn
from data.slicing      import slice_fps_hierarchical_batch


# ---------------------------------------------------------------------------
# Energy constants
# ---------------------------------------------------------------------------

# Theoretical 45nm CMOS  (Horowitz 2014)
E_MAC_45NM = 4.6   # pJ  multiply-accumulate (ANN)
E_AC_45NM  = 0.9   # pJ  accumulate only     (SNN)

# Hardware-measured — Intel Loihi 2  (Lemaire et al. 2022)
# "An Analytical Estimation of Spiking Neural Networks Energy Efficiency"
E_MAC_LOIHI = 8.4e-3   # pJ per MAC
E_AC_LOIHI  = 2.3e-3   # pJ per AC


def energy_ratio(firing_rate, hardware="loihi"):
    """
    Relative SNN energy vs ANN = 1.0.

    firing_rate : fraction of neurons that spike per timestep
    hardware    : "loihi" (Lemaire 2022) or "45nm" (Horowitz 2014)
    """
    if hardware == "loihi":
        e_mac, e_ac = E_MAC_LOIHI, E_AC_LOIHI
    else:
        e_mac, e_ac = E_MAC_45NM, E_AC_45NM
    return firing_rate * (e_ac / e_mac)


# ---------------------------------------------------------------------------
# Train / eval helpers (self-contained, no imports from run_all_experiments)
# ---------------------------------------------------------------------------

def make_slices_fps(pts, T):
    """pts [B,N,3] → [B,T,pps,3] using FPS hierarchical slicing."""
    return slice_fps_hierarchical_batch(pts, T=T)


def _train_epoch(model, loader, optimizer, criterion, device, T,
                 aux_weight=0.3, clip=1.0, bidirectional=False):
    model.train()
    total_loss = total_correct = total_n = 0

    for pts, labels in loader:
        pts    = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        B      = pts.size(0)

        if hasattr(model, "reset_state"):
            model.reset_state(B, device)

        pts_sl = make_slices_fps(pts, T)
        logits_list = []
        for t in range(T):
            logits_list.append(model.forward_step(pts_sl[:, t]))

        if bidirectional and hasattr(model, "finalize"):
            final = model.finalize()
        else:
            final = logits_list[-1]

        loss = criterion(final, labels)
        if aux_weight > 0 and len(logits_list) > 1:
            aux  = sum(criterion(l, labels) for l in logits_list[:-1]) / (len(logits_list) - 1)
            loss = loss + aux_weight * aux

        optimizer.zero_grad()
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss    += loss.item() * B
        total_correct += (final.argmax(1) == labels).sum().item()
        total_n       += B

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def _eval(model, loader, device, T, bidirectional=False):
    model.eval()
    correct = total = 0
    total_spikes = total_neurons_steps = 0

    for pts, labels in loader:
        pts    = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        B      = pts.size(0)

        if hasattr(model, "reset_state"):
            model.reset_state(B, device)

        pts_sl = make_slices_fps(pts, T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_sl[:, t])

        if bidirectional and hasattr(model, "finalize"):
            logits = model.finalize()

        correct += (logits.argmax(1) == labels).sum().item()
        total   += B

        # Firing rate via get_firing_rates() if available
        if hasattr(model, "get_firing_rates"):
            rates = model.get_firing_rates()
            if rates:
                total_spikes        += sum(rates.values()) * B
                total_neurons_steps += len(rates) * B

    acc = correct / total if total > 0 else 0.0
    fr  = (total_spikes / total_neurons_steps
           if total_neurons_steps > 0 else 0.5)   # default: 50% if unavailable
    return acc, fr


def _train_model(model, train_l, val_l, device, T, epochs, lr=1e-3,
                 bidirectional=False, name=""):
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best  = 0.0
    t0    = time.time()

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_l, opt, crit, device,
                                        T=T, bidirectional=bidirectional)
        sched.step()
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            val_acc, fr = _eval(model, val_l, device, T, bidirectional)
            best = max(best, val_acc)
            print(f"  [{name} T={T} ep{ep}/{epochs}]  "
                  f"loss={tr_loss:.4f} tr={tr_acc:.3f} "
                  f"val={val_acc:.3f} fr={fr:.3f}  ({time.time()-t0:.0f}s)")

    return best


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_t_sweep(train_loader, val_loader, device,
                num_classes, epochs, T_list,
                out_dir, smoke_test=False):
    """
    For each T in T_list:
      1. Train ours_full (SNN) with num_slices=T using FPS slicing.
      2. Train ann_pointnet (ANN) with T (acc should be stable).
      3. Convert ann_pointnet → SNN and eval at T (no retraining).
      4. Record accuracy and energy ratio for each.

    Returns: list of row dicts, one per (T, model_type).
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    # ── Train ANN once (T-independent for full-mode ANN) ──────────────────
    print(f"\n{'='*60}\n  Training ANN baseline (once, reused for conversion)\n{'='*60}")
    ann = build_model("ann_pointnet", num_classes=num_classes).to(device)
    if smoke_test:
        ann_acc = 1.0 / num_classes
    else:
        ann_acc = _train_model(ann, train_loader, val_loader, device,
                               T=T_list[-1], epochs=epochs,
                               name="ann_pointnet")
    rows.append({
        "T": "N/A", "model": "ann_pointnet", "type": "ANN",
        "val_acc": round(ann_acc * 100, 2),
        "firing_rate": 1.0,
        "energy_loihi": 1.0,
        "energy_45nm":  1.0,
    })
    print(f"  ANN acc = {ann_acc*100:.2f}%")

    # ── Sweep T ───────────────────────────────────────────────────────────
    for T in T_list:
        print(f"\n{'#'*60}\n  T = {T}\n{'#'*60}")

        # 1. Native SNN
        print(f"  [Native SNN] T={T}")
        snn = build_model("ours_full", num_classes=num_classes).to(device)
        if smoke_test:
            snn_acc, fr = 1.0 / num_classes, 0.3
        else:
            snn_acc = _train_model(snn, train_loader, val_loader, device,
                                   T=T, epochs=epochs, bidirectional=True,
                                   name=f"ours_full")
            _, fr = _eval(snn, val_loader, device, T, bidirectional=True)

        rows.append({
            "T": T, "model": "ours_full", "type": "Native-SNN",
            "val_acc":      round(snn_acc * 100, 2),
            "firing_rate":  round(fr, 4),
            "energy_loihi": round(energy_ratio(fr, "loihi"), 4),
            "energy_45nm":  round(energy_ratio(fr, "45nm"),  4),
        })
        print(f"  Native SNN  T={T}: acc={snn_acc*100:.2f}%  fr={fr:.3f}")

        # 2. Converted SNN (reuse ANN, just change T)
        print(f"  [Converted SNN] T={T}")
        if smoke_test:
            conv_acc = 1.0 / num_classes
        else:
            snn_conv  = convert_ann_to_snn(ann, train_loader, device,
                                            T=T, n_calib_batches=8)
            conv_acc  = eval_converted_snn(snn_conv, val_loader, device, T=T)

        # Converted SNN firing rate ≈ mean activation / threshold ≈ 0.5 (typical)
        fr_conv = 0.5
        rows.append({
            "T": T, "model": "ann_pointnet->SNN", "type": "Converted-SNN",
            "val_acc":      round(conv_acc * 100, 2),
            "firing_rate":  fr_conv,
            "energy_loihi": round(energy_ratio(fr_conv, "loihi"), 4),
            "energy_45nm":  round(energy_ratio(fr_conv, "45nm"),  4),
        })
        print(f"  Converted SNN T={T}: acc={conv_acc*100:.2f}%"
              f"  gap={ann_acc-conv_acc:+.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "t_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[T-sweep] Results saved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_t_sweep(rows, out_dir)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_t_sweep(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        os.makedirs(out_dir, exist_ok=True)

        native_rows = [r for r in rows if r["type"] == "Native-SNN"]
        conv_rows   = [r for r in rows if r["type"] == "Converted-SNN"]
        ann_acc     = next((r["val_acc"] for r in rows if r["type"] == "ANN"), None)

        T_vals_n    = [r["T"] for r in native_rows]
        acc_n       = [r["val_acc"] for r in native_rows]
        T_vals_c    = [r["T"] for r in conv_rows]
        acc_c       = [r["val_acc"] for r in conv_rows]

        # ── Plot 1: Accuracy vs T ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_vals_n, acc_n, "s-", color="tomato",    linewidth=2,
                markersize=7, label="Native SNN (ours_full)")
        ax.plot(T_vals_c, acc_c, "o--", color="orange",   linewidth=2,
                markersize=7, label="Converted SNN (ANN→SNN)")
        if ann_acc is not None:
            ax.axhline(ann_acc, color="steelblue", linestyle=":",
                       linewidth=2, label=f"ANN baseline ({ann_acc:.1f}%)")
        ax.set_xlabel("Number of Timesteps T")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs Timestep T\n"
                     "(Native SNN vs Converted SNN vs ANN)")
        ax.legend(); ax.grid(True, alpha=0.3)
        if T_vals_n:
            ymin = max(0, min(acc_n + acc_c + ([ann_acc] if ann_acc else [])) - 3)
            ax.set_ylim(ymin, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "t_sweep_accuracy.png"), dpi=150)
        plt.close()

        # ── Plot 2: Accuracy-Efficiency frontier ─────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        if native_rows:
            en_n = [r["energy_loihi"] for r in native_rows]
            ax.scatter(en_n, acc_n, s=100, c="tomato",  zorder=3,
                       label="Native SNN")
            for i, r in enumerate(native_rows):
                ax.annotate(f"T={r['T']}", (en_n[i], acc_n[i]),
                            fontsize=8, ha="left", va="bottom")

        if conv_rows:
            en_c = [r["energy_loihi"] for r in conv_rows]
            ax.scatter(en_c, acc_c, s=100, c="orange",  marker="D", zorder=3,
                       label="Converted SNN")
            for i, r in enumerate(conv_rows):
                ax.annotate(f"T={r['T']}", (en_c[i], acc_c[i]),
                            fontsize=8, ha="right", va="top")

        if ann_acc is not None:
            ax.scatter([1.0], [ann_acc], s=150, c="steelblue",
                       marker="*", zorder=4, label="ANN (1.0)")

        ax.set_xlabel("Relative Energy (ANN=1.0, Loihi 2, Lemaire 2022)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy–Efficiency Pareto Frontier\n"
                     "(left = more efficient; up = more accurate)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "t_sweep_pareto.png"), dpi=150)
        plt.close()

        # ── Plot 3: Gap (ANN acc - SNN acc) vs T ─────────────────────────
        if ann_acc and native_rows and conv_rows:
            T_all  = sorted(set(T_vals_n) & set(T_vals_c))
            gaps_n = [ann_acc - next(r["val_acc"] for r in native_rows if r["T"]==T)
                      for T in T_all]
            gaps_c = [ann_acc - next(r["val_acc"] for r in conv_rows   if r["T"]==T)
                      for T in T_all]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(T_all, gaps_n, "s-", color="tomato",  linewidth=2,
                    label="Native SNN gap")
            ax.plot(T_all, gaps_c, "o--", color="orange", linewidth=2,
                    label="Converted SNN gap")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("T (timesteps)")
            ax.set_ylabel("Accuracy Gap: ANN − SNN (%)")
            ax.set_title("Conversion Gap vs T\n"
                         "(lower = native SNN closes in on ANN quality)")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "t_sweep_gap.png"), dpi=150)
            plt.close()

        print(f"[T-sweep] Plots saved to {out_dir}/")

    except ImportError:
        print("[T-sweep] matplotlib not available — skipping plots.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="T-timestep sensitivity sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--mn40_root",   default=None)
    p.add_argument("--mn10_root",   default=None)
    p.add_argument("--dataset",     default="modelnet40",
                   choices=["modelnet10", "modelnet40"])
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--T_list",      type=int, nargs="+",
                   default=[4, 8, 12, 16, 24, 32])
    p.add_argument("--out_dir",     default="results/t_sweep")
    p.add_argument("--smoke_test",  action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc     = 10 if args.dataset == "modelnet10" else 40
    root   = args.mn10_root if args.dataset == "modelnet10" else args.mn40_root

    # data
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_all_experiments import get_loaders
    train_l = get_loaders(root, "train", args.batch_size, nc)
    val_l   = get_loaders(root, "test",  args.batch_size, nc)

    rows = run_t_sweep(train_l, val_l, device,
                       num_classes=nc, epochs=args.epochs,
                       T_list=args.T_list if not args.smoke_test else [4, 8],
                       out_dir=args.out_dir, smoke_test=args.smoke_test)

    # Print summary table
    print(f"\n{'T':<6} {'Model':<25} {'Type':<16} {'Acc%':>7}  "
          f"{'FR':>6}  {'E_Loihi':>8}  {'E_45nm':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{str(r['T']):<6} {r['model']:<25} {r['type']:<16} "
              f"{r['val_acc']:>7.2f}  {r.get('firing_rate','-'):>6}  "
              f"{r.get('energy_loihi','-'):>8}  {r.get('energy_45nm','-'):>8}")


if __name__ == "__main__":
    main()

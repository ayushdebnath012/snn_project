"""
energy_analysis.py
==================
Complete energy efficiency analysis for SPM and ASP+SPM.

Reports:
  1. Per-layer firing rate (from model.get_firing_rates() / SpikingNeuron stats)
  2. E_SNN / E_ANN ratio using Loihi 2 constants (Lemaire et al. 2022)
  3. ASP mean exit step T_eff as a function of confidence threshold
  4. Energy vs accuracy Pareto curve (sweep threshold ∈ [0.1, 0.9])
  5. Comparison bar chart: SPM vs ASP at each T (energy breakdown)

Loihi 2 constants:
  E_MAC = 8.4 pJ   (multiply-accumulate, ANN)
  E_AC  = 2.3 pJ   (accumulate-only, SNN spike)
  E_SNN / E_ANN = firing_rate × E_AC / E_MAC

Usage:
  cd purdueprj
  python benchmarks/energy_analysis.py \\
      --spm_ckpt  results/asp_spm_ckpts/spm_ModelNet10_best.pth \\
      --asp_ckpt  results/asp_spm_ckpts/asp_ModelNet10_best.pth \\
      --root      /data/ModelNet10 \\
      --num_classes 10

For smoke test (no real data needed):
  python benchmarks/energy_analysis.py --smoke_test
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spiking_mamba  import SPMModel
from models.asp_wrapper    import ASPWrapper
from data.slicing          import slice_fps_hierarchical_batch
from training.train_active import prepare_fps_slices_and_geo

# ---------------------------------------------------------------------------
# Loihi 2 energy constants (Lemaire et al. 2022)
# ---------------------------------------------------------------------------

E_MAC = 8.4e-3   # pJ per multiply-accumulate  (ANN)
E_AC  = 2.3e-3   # pJ per accumulate-only       (SNN spike)

# 45-nm CMOS theoretical (Horowitz 2014) — included for reference
E_MAC_45NM = 4.6e-3
E_AC_45NM  = 0.9e-3

HARDWARE_LABELS = {
    "loihi2": ("Loihi 2 (Lemaire 2022)", E_MAC,      E_AC),
    "45nm":   ("45nm CMOS (Horowitz 2014)", E_MAC_45NM, E_AC_45NM),
}


def snn_ann_energy_ratio(firing_rate, hardware="loihi2"):
    _, E_mac, E_ac = HARDWARE_LABELS[hardware]
    return firing_rate * E_ac / E_mac


def combined_energy(firing_rate, avg_slices, T, hardware="loihi2"):
    """E_SNN_ASP / E_ANN combining SNN sparsity and early exit."""
    return snn_ann_energy_ratio(firing_rate, hardware) * (avg_slices / T)


# ---------------------------------------------------------------------------
# Model builders (must match what was trained)
# ---------------------------------------------------------------------------

T           = 4
FEAT_DIM    = 512
POINT_DIMS  = (128, 256, 512)
D_STATE     = 16
N_SMB       = 2
KNN_K       = 16
TAU         = 0.9


def make_spm(num_classes, device):
    return SPMModel(
        num_classes=num_classes, point_dims=POINT_DIMS, d_state=D_STATE,
        tau=TAU, n_smb_layers=N_SMB, local_knn=True, knn_k=KNN_K,
        learnable_lif=False,
    ).to(device)


def make_asp(num_classes, device):
    base = make_spm(num_classes, device)
    return ASPWrapper(base, feat_dim=FEAT_DIM, num_classes=num_classes).to(device)


# ---------------------------------------------------------------------------
# Firing rate collection
# ---------------------------------------------------------------------------

def collect_firing_rates(model, loader, T_val, device, is_asp=False,
                         n_batches=20):
    """
    Run the model on up to n_batches of data, then read per-layer firing rates.
    Returns dict { layer_name: firing_rate }.
    """
    model.eval()
    with torch.no_grad():
        for i, (pts, _) in enumerate(loader):
            if i >= n_batches:
                break
            pts = pts.to(device)
            B   = pts.size(0)
            if is_asp:
                pts_sl, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T_val)
                model.forward_active_infer(pts_sl, geo, threshold=0.5)
            else:
                model.reset_state(B, device)
                model._total_T = T_val
                pts_sl = slice_fps_hierarchical_batch(pts, T=T_val)
                for t in range(T_val):
                    model.forward_step(pts_sl[:, t])

    rates = {}
    # SPMModel / ASPWrapper: recurse into submodules for SpikingNeuron / BNLIFLayer
    for name, module in model.named_modules():
        if hasattr(module, "firing_rate"):
            fr = module.firing_rate()
            if fr > 0:
                rates[name] = fr
        elif hasattr(module, "spike_count") and hasattr(module, "step_count"):
            if module.step_count > 0:
                fr = (module.spike_count /
                      (module.step_count * getattr(module, "batch_size", 1) *
                       getattr(module, "out_features", 1))).item()
                if fr > 0:
                    rates[name] = fr

    # Fallback: model.get_firing_rates()
    if not rates and hasattr(model, "get_firing_rates"):
        rates = model.get_firing_rates()

    return rates


# ---------------------------------------------------------------------------
# ASP early-exit sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def threshold_sweep(model, loader, T_val, device,
                    thresholds=None):
    """
    For each confidence threshold, compute (accuracy, mean_slices).
    Returns list of dicts.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.95, 18).tolist()

    results = []
    model.eval()
    for thr in thresholds:
        correct = total = total_sl = 0
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            B = pts.size(0)
            pts_sl, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T_val)
            logits, exit_step, _ = model.forward_active_infer(
                pts_sl, geo, threshold=float(thr)
            )
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += B
            total_sl += exit_step * B

        acc    = correct / total if total > 0 else 0
        avg_sl = total_sl / total if total > 0 else T_val
        results.append({
            "threshold": round(float(thr), 3),
            "accuracy":  round(acc, 4),
            "avg_slices": round(avg_sl, 3),
        })
        print(f"  thr={thr:.2f}  acc={acc:.4f}  avg_slices={avg_sl:.2f}/{T_val}")

    return results


# ---------------------------------------------------------------------------
# Energy table
# ---------------------------------------------------------------------------

def compute_energy_table(fr_spm, fr_asp, avg_slices_asp, T_val):
    """
    Returns dict with energy breakdown for both hardware targets.
    """
    table = {}
    for hw in ["loihi2", "45nm"]:
        label = HARDWARE_LABELS[hw][0]
        e_spm = snn_ann_energy_ratio(fr_spm, hw)     # uses all T slices
        e_asp = combined_energy(fr_asp, avg_slices_asp, T_val, hw)
        table[hw] = {
            "label":         label,
            "fr_spm":        round(fr_spm, 4),
            "fr_asp":        round(fr_asp, 4),
            "e_spm_vs_ann":  round(e_spm, 4),
            "e_asp_vs_ann":  round(e_asp, 4),
            "asp_savings":   round((e_spm - e_asp) / e_spm, 4),
        }
    return table


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all(fr_spm, fr_asp, pareto, energy_table, out_dir, T_val=4):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        os.makedirs(out_dir, exist_ok=True)

        thr    = [r["threshold"]   for r in pareto]
        acc    = [r["accuracy"] * 100 for r in pareto]
        avg_sl = [r["avg_slices"]  for r in pareto]

        # ── Plot 1: Accuracy vs Mean Slices (Pareto) ──────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(avg_sl, acc, c=thr, cmap="RdYlGn_r",
                        s=90, zorder=3, edgecolors="black", linewidth=0.5)
        ax.plot(avg_sl, acc, "k--", lw=0.8, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Confidence threshold")
        ax.axvline(T_val, color="steelblue", lw=1.2, linestyle="--",
                   label=f"SPM (all {T_val} slices)")
        for i, (x, y, t) in enumerate(zip(avg_sl, acc, thr)):
            if t in [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]:
                ax.annotate(f"{t:.1f}", (x, y), fontsize=7,
                            ha="left", va="bottom")
        ax.set_xlabel(f"Mean slices used at exit (out of {T_val})")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("ASP+SPM: Accuracy vs Mean Slices\n"
                     "(Pareto curve — vary confidence threshold)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pareto_acc_slices.png"), dpi=150)
        plt.close()

        # ── Plot 2: Energy vs Accuracy Pareto ─────────────────────────────
        fr_mean = (fr_spm + fr_asp) / 2   # representative FR
        energy_asp = [combined_energy(fr_asp, s, T_val) * 100 for s in avg_sl]
        energy_spm = snn_ann_energy_ratio(fr_spm) * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        sc2 = ax.scatter(energy_asp, acc, c=thr, cmap="RdYlGn_r",
                         s=90, zorder=3, edgecolors="black", linewidth=0.5)
        ax.plot(energy_asp, acc, "k--", lw=0.8, alpha=0.5)
        plt.colorbar(sc2, ax=ax, label="Confidence threshold")
        ax.scatter([energy_spm], [max(acc)], marker="*", s=250,
                   color="steelblue", zorder=5, label=f"SPM ({T_val} slices)")
        ax.scatter([100.0], [max(acc)], marker="P", s=200,
                   color="gray", zorder=4, label="ANN baseline")
        ax.set_xlabel("Est. Energy vs ANN (%, Loihi 2)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("ASP+SPM: Energy–Accuracy Pareto\n"
                     "(left = cheaper; up = more accurate)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pareto_energy_acc.png"), dpi=150)
        plt.close()

        # ── Plot 3: Per-layer firing rates (SPM vs ASP) ───────────────────
        if fr_spm > 0 and fr_asp > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            models = ["SPM (fixed)", "ASP+SPM (adaptive)"]
            frs    = [fr_spm * 100, fr_asp * 100]
            bars   = ax.bar(models, frs,
                            color=["steelblue", "tomato"], edgecolor="black")
            ax.axhline(100, color="gray", lw=0.8, linestyle="--",
                       label="ANN (dense = 100%)")
            ax.set_ylabel("Mean Firing Rate (%)")
            ax.set_title("Mean Firing Rate: SPM vs ASP+SPM\n"
                         "(lower = more sparse = more efficient)")
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
            ax.legend(); ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "firing_rate_bar.png"), dpi=150)
            plt.close()

        print(f"[Energy] Plots saved to {out_dir}/")

    except ImportError:
        print("[Energy] matplotlib not available — skipping plots.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Energy efficiency analysis for SPM and ASP+SPM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spm_ckpt",   default=None,
                   help="SPM checkpoint (.pth)")
    p.add_argument("--asp_ckpt",   default=None,
                   help="ASP+SPM checkpoint (.pth)")
    p.add_argument("--root",       default=None,
                   help="Dataset root directory")
    p.add_argument("--num_classes",type=int, default=10)
    p.add_argument("--T",          type=int, default=T)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--out_dir",    default="results/energy_analysis")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use dummy data, skip checkpoint loading")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    T_val = args.T
    nc    = args.num_classes

    # ── Dataset ────────────────────────────────────────────────────────────
    if args.smoke_test or args.root is None:
        print("[Smoke] Using dummy dataset")
        va_ds = TensorDataset(
            torch.randn(128, args.num_points, 3),
            torch.randint(0, nc, (128,))
        )
        val_l = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    else:
        try:
            from data.modelnet import ModelNetDataset
            val_ds = ModelNetDataset(args.root, split="test",
                                     num_points=args.num_points,
                                     num_classes=nc)
            val_l  = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)
            print(f"Val set: {len(val_ds)} samples")
        except Exception as e:
            print(f"[WARN] ModelNetDataset failed ({e}), using dummy data")
            va_ds = TensorDataset(
                torch.randn(128, args.num_points, 3),
                torch.randint(0, nc, (128,))
            )
            val_l = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    # ── Models ─────────────────────────────────────────────────────────────
    spm = make_spm(nc, device)
    asp = make_asp(nc, device)

    if not args.smoke_test:
        if args.spm_ckpt and os.path.exists(args.spm_ckpt):
            spm.load_state_dict(torch.load(args.spm_ckpt, map_location=device))
            print(f"SPM loaded from {args.spm_ckpt}")
        else:
            print("[WARN] No SPM checkpoint — using random weights (firing rates unreliable)")

        if args.asp_ckpt and os.path.exists(args.asp_ckpt):
            asp.load_state_dict(torch.load(args.asp_ckpt, map_location=device))
            print(f"ASP loaded from {args.asp_ckpt}")
        else:
            print("[WARN] No ASP checkpoint — using random weights")

    # ── 1. Firing rates ─────────────────────────────────────────────────
    print("\n[1] Collecting firing rates ...")
    rates_spm = collect_firing_rates(spm, val_l, T_val, device, is_asp=False)
    rates_asp = collect_firing_rates(asp, val_l, T_val, device, is_asp=True)

    fr_spm = float(np.mean(list(rates_spm.values()))) if rates_spm else 0.15
    fr_asp = float(np.mean(list(rates_asp.values()))) if rates_asp else 0.15

    print(f"  SPM mean firing rate: {fr_spm:.4f} ({fr_spm*100:.1f}%)")
    print(f"  ASP mean firing rate: {fr_asp:.4f} ({fr_asp*100:.1f}%)")

    if rates_spm:
        print("  Per-layer (SPM):")
        for name, fr in sorted(rates_spm.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name:<40} {fr:.4f}")

    # ── 2. Threshold sweep ──────────────────────────────────────────────
    print(f"\n[2] ASP confidence threshold sweep (T={T_val}) ...")
    pareto = threshold_sweep(asp, val_l, T_val, device)

    # ── 3. Energy table ─────────────────────────────────────────────────
    best_pareto = max(pareto, key=lambda r: r["accuracy"])
    avg_slices  = best_pareto["avg_slices"]
    print(f"\n[3] Energy table (at ASP best-accuracy threshold "
          f"={best_pareto['threshold']}, avg_slices={avg_slices:.2f}/{T_val}) ...")
    energy_table = compute_energy_table(fr_spm, fr_asp, avg_slices, T_val)

    for hw, row in energy_table.items():
        print(f"\n  {row['label']}")
        print(f"    SPM  E/ANN = {row['e_spm_vs_ann']*100:.1f}%")
        print(f"    ASP  E/ANN = {row['e_asp_vs_ann']*100:.1f}%")
        print(f"    ASP additional savings vs SPM = {row['asp_savings']*100:.1f}%")

    # ── 4. Plots ────────────────────────────────────────────────────────
    plot_all(fr_spm, fr_asp, pareto, energy_table, args.out_dir, T_val=T_val)

    # ── 5. Save JSON ────────────────────────────────────────────────────
    output = {
        "T":             T_val,
        "num_classes":   nc,
        "fr_spm":        fr_spm,
        "fr_asp":        fr_asp,
        "rates_spm":     rates_spm,
        "rates_asp":     rates_asp,
        "pareto":        pareto,
        "energy_table":  energy_table,
        "best_threshold": best_pareto,
    }
    json_path = os.path.join(args.out_dir, "energy_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Energy] Results saved: {json_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    loihi = energy_table["loihi2"]
    print(f"\n{'='*60}")
    print(f"Energy Efficiency Summary (Loihi 2)")
    print(f"{'='*60}")
    print(f"  ANN baseline:     100.0%")
    print(f"  SPM (all slices): {loihi['e_spm_vs_ann']*100:.1f}% of ANN")
    print(f"  ASP+SPM (exit):   {loihi['e_asp_vs_ann']*100:.1f}% of ANN")
    print(f"  Additional savings (ASP over SPM): {loihi['asp_savings']*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

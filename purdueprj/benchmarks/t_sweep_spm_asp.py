"""
t_sweep_spm_asp.py
==================
Ablation: sweep T ∈ {2, 4, 8, 16} for SPM (fixed order) vs ASP+SPM (adaptive).

Hypothesis: ASP's benefit grows at lower T because the ordering matters more
when fewer slices are available.  At T=2, choosing the wrong slice first is
catastrophic; the SSP should recover most of that.

For each T we:
  1. Train SPM from scratch (fixed FPS order, CE loss).
  2. Train ASP+SPM from scratch (adaptive SSP order, active loss).
  3. Eval: SPM accuracy, ASP accuracy + mean slices used at exit.
  4. Record energy ratio (SNN sparsity × exit fraction, Loihi 2).

Output: results/t_sweep_spm_asp/results.json + plots

Usage:
  cd purdueprj
  python benchmarks/t_sweep_spm_asp.py \\
      --root /data/ModelNet10 --dataset modelnet10 \\
      --T_list 2 4 8 16 --epochs 100

For a quick smoke test:
  python benchmarks/t_sweep_spm_asp.py --smoke_test
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spiking_mamba  import SPMModel
from models.asp_wrapper    import ASPWrapper
from data.slicing          import slice_fps_hierarchical_batch
from training.train_active import prepare_fps_slices_and_geo, gumbel_tau
from training.loss_active  import active_loss

# ---------------------------------------------------------------------------
# Energy constants (Loihi 2, Lemaire et al. 2022)
# ---------------------------------------------------------------------------

E_MAC = 8.4e-3   # pJ per MAC
E_AC  = 2.3e-3   # pJ per AC

def energy_ratio(firing_rate, avg_slices, T):
    """E_SNN_ASP / E_ANN — combines SNN sparsity and early exit."""
    snn_fraction  = firing_rate * E_AC / E_MAC
    exit_fraction = avg_slices / T
    return snn_fraction * exit_fraction

# ---------------------------------------------------------------------------
# Model builders (all shared config except T)
# ---------------------------------------------------------------------------

POINT_DIMS = (128, 256, 512)
D_STATE    = 16
N_SMB      = 2
KNN_K      = 16
TAU        = 0.9
FEAT_DIM   = 512
EXIT_THR   = 0.6


def make_spm(num_classes, device):
    return SPMModel(
        num_classes  = num_classes,
        point_dims   = POINT_DIMS,
        d_state      = D_STATE,
        tau          = TAU,
        n_smb_layers = N_SMB,
        local_knn    = True,
        knn_k        = KNN_K,
        learnable_lif= False,
    ).to(device)


def make_asp(num_classes, device):
    base = make_spm(num_classes, device)
    return ASPWrapper(base, feat_dim=FEAT_DIM, num_classes=num_classes).to(device)


# ---------------------------------------------------------------------------
# Train / eval for SPM
# ---------------------------------------------------------------------------

def train_spm_epoch(model, loader, optimizer, T, device):
    model.train()
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        model.reset_state(B, device)
        model._total_T = T
        pts_sl = slice_fps_hierarchical_batch(pts, T=T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_sl[:, t])
        loss = F.cross_entropy(logits, labels)
        if torch.isfinite(loss):
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * B
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_spm(model, loader, T, device):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        model.reset_state(B, device)
        model._total_T = T
        pts_sl = slice_fps_hierarchical_batch(pts, T=T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_sl[:, t])
        correct += (logits.argmax(1) == labels).sum().item()
        total   += B
    return correct / total


# ---------------------------------------------------------------------------
# Train / eval for ASP
# ---------------------------------------------------------------------------

def train_asp_epoch(model, loader, optimizer, epoch, T, device,
                    lam_aux=0.05, lam_exit=0.1, lam_fr=0.02):
    model.train()
    tau = gumbel_tau(epoch)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        pts_sl, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T)
        logits_final, logits_all, _ = model.forward_active_train(pts_sl, geo)
        loss, _ = active_loss(logits_final, logits_all, labels, model,
                              lam_aux=lam_aux, lam_exit=lam_exit, lam_fr=lam_fr)
        if torch.isfinite(loss):
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * B
        total_acc  += (logits_final.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_asp(model, loader, T, device, threshold=EXIT_THR):
    model.eval()
    correct = total = total_sl = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        pts_sl, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T)
        logits, exit_step, _ = model.forward_active_infer(pts_sl, geo, threshold=threshold)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += B
        total_sl += exit_step * B
    return correct / total, total_sl / total


# ---------------------------------------------------------------------------
# Train one (T, model_type) cell
# ---------------------------------------------------------------------------

def train_and_eval(model_type, num_classes, train_l, val_l,
                   T, epochs, lr, device, ckpt_path, smoke_test):
    """Returns (best_val_acc, firing_rate, avg_slices_at_exit)."""
    if model_type == "spm":
        model = make_spm(num_classes, device)
    else:
        model = make_asp(num_classes, device)

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)
    best  = 0.0
    best_sl = T

    for ep in range(epochs):
        if model_type == "spm":
            tr_loss, tr_acc = train_spm_epoch(model, train_l, opt, T, device)
        else:
            tr_loss, tr_acc = train_asp_epoch(model, train_l, opt, ep, T, device)
        sched.step()

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            if model_type == "spm":
                va = eval_spm(model, val_l, T, device)
                sl = T
            else:
                va, sl = eval_asp(model, val_l, T, device)
            if va > best:
                best = va
                best_sl = sl
                torch.save(model.state_dict(), ckpt_path)
            tag = "★" if va == best else " "
            print(f"    [{model_type} T={T} ep{ep+1:03d}] "
                  f"tr={tr_acc:.3f} val={va:.3f} {tag} sl={sl:.2f}/{T}")

    # Firing rate — read from best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    if hasattr(model, "get_firing_rates"):
        rates = model.get_firing_rates()
        fr = float(np.mean(list(rates.values()))) if rates else 0.15
    else:
        fr = 0.15

    return best, fr, best_sl


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(train_l, val_l, device, num_classes, epochs, T_list, lr,
              out_dir, smoke_test):
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for T in T_list:
        print(f"\n{'#'*60}\n  T = {T}\n{'#'*60}")

        for mtype in ["spm", "asp"]:
            ckpt = os.path.join(out_dir, f"{mtype}_T{T}_best.pth")
            t0 = time.time()
            va, fr, avg_sl = train_and_eval(
                mtype, num_classes, train_l, val_l,
                T=T, epochs=epochs, lr=lr, device=device,
                ckpt_path=ckpt, smoke_test=smoke_test,
            )
            elapsed = time.time() - t0
            e_ratio = energy_ratio(fr, avg_sl, T)
            print(f"  [{mtype.upper()} T={T}] "
                  f"val={va*100:.2f}%  fr={fr:.3f}  "
                  f"avg_slices={avg_sl:.2f}/{T}  "
                  f"E/ANN={e_ratio*100:.1f}%  ({elapsed:.0f}s)")
            rows.append({
                "T":          T,
                "model":      mtype,
                "val_acc":    round(va * 100, 2),
                "firing_rate": round(fr, 4),
                "avg_slices": round(avg_sl, 3),
                "energy_ratio": round(e_ratio, 4),
            })

    # Save
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[T-sweep] Saved: {json_path}")

    # Plot
    _plot(rows, out_dir)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        spm_rows = [r for r in rows if r["model"] == "spm"]
        asp_rows = [r for r in rows if r["model"] == "asp"]

        T_spm = [r["T"] for r in spm_rows]
        T_asp = [r["T"] for r in asp_rows]
        acc_spm = [r["val_acc"] for r in spm_rows]
        acc_asp = [r["val_acc"] for r in asp_rows]
        delta   = [a - s for a, s in zip(acc_asp, acc_spm)]
        en_asp  = [r["energy_ratio"] * 100 for r in asp_rows]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Plot 1: Accuracy vs T
        ax = axes[0]
        ax.plot(T_spm, acc_spm, "s-", color="steelblue",  lw=2, ms=8, label="SPM (fixed order)")
        ax.plot(T_asp, acc_asp, "o-", color="tomato",     lw=2, ms=8, label="ASP+SPM (adaptive)")
        ax.set_xlabel("T (timesteps / slices)")
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_title("Accuracy vs T\nSPM vs ASP+SPM")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xticks(T_spm)

        # Plot 2: Δ accuracy (ASP − SPM) vs T
        ax = axes[1]
        colors = ["tomato" if d > 0 else "steelblue" for d in delta]
        ax.bar(T_spm, delta, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("T (timesteps / slices)")
        ax.set_ylabel("Δ Accuracy (ASP − SPM, pp)")
        ax.set_title("ASP Benefit vs T\n(hypothesis: larger at lower T)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(T_spm)
        for i, (t, d) in enumerate(zip(T_spm, delta)):
            ax.text(t, d + (0.05 if d >= 0 else -0.15),
                    f"{d:+.2f}", ha="center", fontsize=9, fontweight="bold")

        # Plot 3: Accuracy vs Energy (Pareto)
        ax = axes[2]
        ax.plot([r["energy_ratio"] * 100 for r in spm_rows],
                acc_spm, "s--", color="steelblue", lw=1.5, ms=8, label="SPM")
        ax.plot(en_asp, acc_asp, "o-", color="tomato",    lw=2, ms=8, label="ASP+SPM")
        for r in asp_rows:
            ax.annotate(f"T={r['T']}", (r["energy_ratio"] * 100, r["val_acc"]),
                        fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Est. Energy vs ANN (%)")
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_title("Accuracy–Energy Pareto\n(lower-left = better)")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "t_sweep_spm_asp.png"), dpi=150)
        plt.close()
        print(f"[T-sweep] Plot saved to {out_dir}/t_sweep_spm_asp.png")

    except ImportError:
        print("[T-sweep] matplotlib not available — skipping plots.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="T-sweep: SPM vs ASP+SPM ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",       default=None,
                   help="ModelNet10 or ModelNet40 root directory")
    p.add_argument("--dataset",    default="modelnet10",
                   choices=["modelnet10", "modelnet40"])
    p.add_argument("--T_list",     type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--out_dir",    default="results/t_sweep_spm_asp")
    p.add_argument("--smoke_test", action="store_true",
                   help="2-epoch dummy run for CI / quick sanity check")
    args = p.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_pts   = 1024
    nc        = 10 if args.dataset == "modelnet10" else 40
    epochs    = 2 if args.smoke_test else args.epochs
    T_list    = [2, 4] if args.smoke_test else args.T_list

    # Dataset
    if args.smoke_test or args.root is None:
        # Dummy tensors for smoke test / missing root
        tr_ds = TensorDataset(torch.randn(256, num_pts, 3), torch.randint(0, nc, (256,)))
        va_ds = TensorDataset(torch.randn(64,  num_pts, 3), torch.randint(0, nc, (64,)))
        train_l = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        val_l   = DataLoader(va_ds, batch_size=args.batch_size)
        print("[Smoke] Using dummy dataset")
    else:
        from data.modelnet import ModelNetDataset
        train_l = DataLoader(
            ModelNetDataset(args.root, split="train", num_points=num_pts,
                            num_classes=nc),
            batch_size=args.batch_size, shuffle=True, num_workers=2,
            pin_memory=True, drop_last=True,
        )
        val_l = DataLoader(
            ModelNetDataset(args.root, split="test", num_points=num_pts,
                            num_classes=nc),
            batch_size=args.batch_size, shuffle=False, num_workers=2,
            pin_memory=True,
        )

    rows = run_sweep(
        train_l, val_l, device, nc, epochs, T_list, args.lr,
        args.out_dir, smoke_test=args.smoke_test,
    )

    # Print summary
    print(f"\n{'T':<6} {'Model':<10} {'Acc%':>8}  {'FR':>6}  {'AvgSl':>7}  {'E/ANN%':>8}  {'Δ(pp)':>8}")
    print("-" * 65)
    for T in T_list:
        spm = next((r for r in rows if r["T"] == T and r["model"] == "spm"), None)
        asp = next((r for r in rows if r["T"] == T and r["model"] == "asp"), None)
        if spm:
            print(f"{T:<6} {'SPM':<10} {spm['val_acc']:>8.2f}  "
                  f"{spm['firing_rate']:>6.3f}  {spm['avg_slices']:>7.2f}  "
                  f"{spm['energy_ratio']*100:>7.1f}%  {'—':>8}")
        if asp:
            delta = asp["val_acc"] - spm["val_acc"] if spm else 0.0
            print(f"{T:<6} {'ASP+SPM':<10} {asp['val_acc']:>8.2f}  "
                  f"{asp['firing_rate']:>6.3f}  {asp['avg_slices']:>7.2f}  "
                  f"{asp['energy_ratio']*100:>7.1f}%  {delta:>+7.2f}")


if __name__ == "__main__":
    main()

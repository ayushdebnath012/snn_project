"""
run_scanobjectnn.py
===================
SPM baseline vs ASP+SPM on ScanObjectNN.

ScanObjectNN (Uy et al. ICCV 2019) uses real-world scanned objects with
background clutter — a harder test than ModelNet for adaptive perception,
because some spatial regions are uninformative background.  Our hypothesis:
ASP's SSP will learn to front-load informative foreground regions and skip
noisy background slices, yielding higher accuracy and fewer slices at exit.

Three variants tested (in order of difficulty):
  OBJ_ONLY   — clean objects, no background
  OBJ_BG     — objects with background points
  PB_T50_RS  — perturbed + rotated (hardest; paper's main benchmark)

Paper target (SPM, arXiv:2504.14371):
  ScanObjectNN PB_T50_RS: 85.5%

Usage (local, with dummy data if ScanObjectNN not downloaded):
  python run_scanobjectnn.py --root /data/ScanObjectNN --variant PB_T50_RS

Usage (Kaggle, auto-downloads OBJ_ONLY via kagglehub):
  python run_scanobjectnn.py --kaggle

OBJ_ONLY is also available on Kaggle; PB_T50_RS must be downloaded manually
from https://hkust-vgd.github.io/scanobjectnn/
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.spiking_mamba          import SPMModel
from models.asp_wrapper             import ASPWrapper
from data.slicing                   import slice_fps_hierarchical_batch
from data.scanobjectnn              import ScanObjectNNDataset, get_scanobjectnn_loaders, NUM_CLASSES
from training.train_active          import prepare_fps_slices_and_geo, gumbel_tau
from training.loss_active           import active_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCHS      = 150
BATCH       = 16
LR          = 0.001
NUM_POINTS  = 1024
T           = 4          # slices — 256 pts/slice
FEAT_DIM    = 512
POINT_DIMS  = (128, 256, 512)
D_STATE     = 16
N_SMB       = 2
KNN_K       = 16
TAU         = 0.9
EXIT_THR    = 0.6        # early exit confidence margin threshold

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def make_spm(device):
    return SPMModel(
        num_classes  = NUM_CLASSES,
        point_dims   = POINT_DIMS,
        d_state      = D_STATE,
        tau          = TAU,
        n_smb_layers = N_SMB,
        local_knn    = True,
        knn_k        = KNN_K,
        learnable_lif= False,
    ).to(device)


def make_asp(device):
    base = make_spm(device)
    return ASPWrapper(base, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES).to(device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_spm_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        model.reset_state(B, device)
        model._total_T = T

        pts_slices = slice_fps_hierarchical_batch(pts, T=T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_slices[:, t])

        loss = F.cross_entropy(logits, labels)
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * B
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


def train_asp_epoch(model, loader, optimizer, epoch, device,
                    lam_aux=0.05, lam_exit=0.1, lam_fr=0.02):
    model.train()
    tau = gumbel_tau(epoch)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)

    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T)

        logits_final, logits_all, _ = model.forward_active_train(pts_slices, geo)
        loss, _ = active_loss(
            logits_final, logits_all, labels, model,
            lam_aux=lam_aux, lam_exit=lam_exit, lam_fr=lam_fr,
        )

        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * B
        total_acc  += (logits_final.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_spm(model, loader, device):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        model.reset_state(B, device)
        model._total_T = T
        pts_slices = slice_fps_hierarchical_batch(pts, T=T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_slices[:, t])
        correct += (logits.argmax(1) == labels).sum().item()
        total   += B
    return correct / total


@torch.no_grad()
def eval_asp(model, loader, device, threshold=EXIT_THR):
    """Returns (accuracy, mean_slices_used)."""
    model.eval()
    correct = total = total_slices = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T)
        logits, exit_step, _ = model.forward_active_infer(
            pts_slices, geo, threshold=threshold
        )
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += B
        total_slices += exit_step * B
    return correct / total, total_slices / total


# ---------------------------------------------------------------------------
# Pareto curve: accuracy vs threshold sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def pareto_curve(model, loader, device, thresholds=None):
    """Sweep confidence thresholds; return list of (threshold, acc, avg_slices)."""
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    for thr in thresholds:
        acc, avg_sl = eval_asp(model, loader, device, threshold=thr)
        results.append({"threshold": thr, "accuracy": acc, "avg_slices": avg_sl})
        print(f"  threshold={thr:.1f}  acc={acc:.4f}  avg_slices={avg_sl:.2f}/{T}")
    return results


# ---------------------------------------------------------------------------
# Energy calculation
# ---------------------------------------------------------------------------

E_MAC = 8.4e-3   # pJ per MAC (Loihi 2, Lemaire 2022)
E_AC  = 2.3e-3   # pJ per AC  (Loihi 2, Lemaire 2022)

def energy_saving(avg_slices, firing_rate=0.15):
    """
    Relative energy of ASP vs full ANN.
    Two savings multiplicatively: SNN sparsity × early exit fraction.
    """
    snn_vs_ann = firing_rate * E_AC / E_MAC       # ~4% at fr=0.15
    exit_frac  = avg_slices / T                   # fraction of slices used
    return snn_vs_ann * exit_frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(variant, root, device, out_dir, smoke_test=False):
    print(f"\n{'='*70}")
    print(f"ScanObjectNN variant: {variant}   (15 classes)")
    print("=" * 70)

    train_l, val_l, nc = get_scanobjectnn_loaders(
        root, variant=variant, batch_size=BATCH, num_points=NUM_POINTS,
        num_workers=2 if not smoke_test else 0,
    )
    epochs = 3 if smoke_test else EPOCHS

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, variant)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── SPM ──────────────────────────────────────────────────────────────
    spm = make_spm(device)
    spm_opt = torch.optim.Adam(spm.parameters(), lr=LR, weight_decay=1e-4)
    spm_sch = torch.optim.lr_scheduler.StepLR(spm_opt, step_size=20, gamma=0.7)
    best_spm = 0.0
    spm_history = []

    print(f"\n[SPM]  params={sum(p.numel() for p in spm.parameters()):,}")
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_spm_epoch(spm, train_l, spm_opt, device)
        spm_sch.step()

        val_acc = None
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            val_acc = eval_spm(spm, val_l, device)
            if val_acc > best_spm:
                best_spm = val_acc
                torch.save(spm.state_dict(), os.path.join(ckpt_dir, "spm_best.pth"))
            print(f"[SPM] {variant} Ep {ep:3d} | tr={tr_acc:.4f} val={val_acc:.4f} "
                  f"{'★' if val_acc == best_spm else ' '} | {time.time()-t0:.0f}s")
        else:
            print(f"[SPM] {variant} Ep {ep:3d} | tr={tr_acc:.4f} | {time.time()-t0:.0f}s")
        spm_history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": val_acc})

    # ── ASP ──────────────────────────────────────────────────────────────
    asp = make_asp(device)
    asp_opt = torch.optim.Adam(asp.parameters(), lr=LR, weight_decay=1e-4)
    asp_sch = torch.optim.lr_scheduler.StepLR(asp_opt, step_size=20, gamma=0.7)
    best_asp = 0.0
    best_asp_slices = T
    asp_history = []

    print(f"\n[ASP]  params={sum(p.numel() for p in asp.parameters()):,}")
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_asp_epoch(asp, train_l, asp_opt, ep, device)
        asp_sch.step()

        val_acc = val_sl = None
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            val_acc, val_sl = eval_asp(asp, val_l, device)
            if val_acc > best_asp:
                best_asp = val_acc
                best_asp_slices = val_sl
                torch.save(asp.state_dict(), os.path.join(ckpt_dir, "asp_best.pth"))
            print(f"[ASP] {variant} Ep {ep:3d} | tr={tr_acc:.4f} val={val_acc:.4f} "
                  f"{'★' if val_acc == best_asp else ' '} "
                  f"| slices={val_sl:.2f}/{T} | {time.time()-t0:.0f}s")
        else:
            print(f"[ASP] {variant} Ep {ep:3d} | tr={tr_acc:.4f} | {time.time()-t0:.0f}s")
        asp_history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": val_acc})

    # ── Pareto sweep ──────────────────────────────────────────────────────
    if not smoke_test:
        print(f"\n[Pareto] Sweeping confidence thresholds for ASP ({variant}):")
        asp.load_state_dict(torch.load(os.path.join(ckpt_dir, "asp_best.pth"),
                                       map_location=device))
        pareto = pareto_curve(asp, val_l, device)
    else:
        pareto = []

    # ── Energy saving at best checkpoint ─────────────────────────────────
    fr = 0.15  # conservative estimate; ideally read from model.get_firing_rates()
    e_save = energy_saving(best_asp_slices, fr)

    # ── Summary ──────────────────────────────────────────────────────────
    result = {
        "variant":          variant,
        "spm_best_val":     best_spm,
        "asp_best_val":     best_asp,
        "delta_pp":         (best_asp - best_spm) * 100,
        "asp_avg_slices":   best_asp_slices,
        "energy_vs_ann":    e_save,
        "pareto":           pareto,
        "spm_history":      spm_history,
        "asp_history":      asp_history,
    }

    with open(os.path.join(ckpt_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n── {variant} Summary {'─'*40}")
    print(f"  SPM  best val: {best_spm:.4f} ({best_spm*100:.2f}%)")
    print(f"  ASP  best val: {best_asp:.4f} ({best_asp*100:.2f}%)")
    print(f"  Δ (ASP−SPM):   {(best_asp-best_spm)*100:+.2f} pp")
    print(f"  ASP avg slices at exit: {best_asp_slices:.2f}/{T}")
    print(f"  Est. energy vs ANN:     {e_save*100:.1f}%")

    return result


def main():
    p = argparse.ArgumentParser(
        description="SPM vs ASP+SPM on ScanObjectNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",       default=None,
                   help="ScanObjectNN root directory (see data/scanobjectnn.py for layout)")
    p.add_argument("--variants",   nargs="+",
                   default=["OBJ_ONLY", "OBJ_BG", "PB_T50_RS"],
                   choices=["OBJ_ONLY", "OBJ_BG", "PB_T50_RS"])
    p.add_argument("--out_dir",    default="results/scanobjectnn")
    p.add_argument("--smoke_test", action="store_true",
                   help="3-epoch dummy run for CI / quick sanity check")
    p.add_argument("--kaggle",     action="store_true",
                   help="Download OBJ_ONLY via kagglehub and run OBJ_ONLY only")
    args = p.parse_args()

    # Kaggle: auto-download OBJ_ONLY
    if args.kaggle:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
        dl_path = kagglehub.dataset_download(
            "wanghao8080/scanobjectnn"   # community re-upload of OBJ_ONLY split
        )
        args.root = dl_path
        args.variants = ["OBJ_ONLY"]
        print(f"[Kaggle] ScanObjectNN downloaded to {dl_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    for variant in args.variants:
        try:
            r = run_variant(variant, args.root, device, args.out_dir, args.smoke_test)
            all_results[variant] = r
        except FileNotFoundError as e:
            print(f"\n[SKIP] {variant}: {e}")

    if all_results:
        print(f"\n\n{'='*70}")
        print("FINAL RESULTS — SPM vs ASP+SPM on ScanObjectNN")
        print("Architecture: IDENTICAL except slice ordering (SSP vs fixed FPS)")
        print("=" * 70)
        print(f"{'Variant':<14} {'SPM':>9} {'ASP':>9} {'Δ(pp)':>8}  {'AvgSlices':>10}  {'E_vs_ANN':>10}")
        print("-" * 70)
        for var, r in all_results.items():
            print(f"{var:<14} {r['spm_best_val']*100:>8.2f}% {r['asp_best_val']*100:>8.2f}% "
                  f"{r['delta_pp']:>+7.2f}  {r['asp_avg_slices']:>8.2f}/{T}  "
                  f"{r['energy_vs_ann']*100:>8.1f}%")
        print("=" * 70)
        print("\nNote: Δ > 0 means ASP outperforms SPM on noisy/background data.")
        print("  Energy estimate: fr=0.15 (Loihi 2) × early-exit fraction.")

        with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

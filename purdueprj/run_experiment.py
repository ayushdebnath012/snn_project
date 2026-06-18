# -*- coding: utf-8 -*-
"""
Full training + inference + plotting pipeline.
Trains ANN and SNN in slice mode on ModelNet10, then runs all 4 inference modes
and saves training curves + 5 inference plots to ./results/.
"""

import sys
import os

# Ensure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── config ──────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.normpath(
    "c:/Users/USER_HP/Desktop/Purdue Project/SNN"
    "/ModelNet10-20260219T070651Z-1-001/ModelNet10"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPOCHS     = 10
BATCH_SIZE = 16
NUM_POINTS = 1024
NUM_SLICES = 16
AUX_WEIGHT = 0.3
LR         = 1e-3
# ────────────────────────────────────────────────────────────────────────────

from data.modelnet         import ModelNetDataset
from training.optimizers   import build_optimizer
from training.train_loop   import train_one_epoch
from models.pointnet_snn   import PointNetSNN
from models.pointnet_ann   import PointNetANN
from inference.infer_modes import (
    infer_ann_full, infer_snn_full,
    infer_ann_slice, infer_snn_slice,
)
from inference.plotting    import plot_all_metrics

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Output: {OUTPUT_DIR}\n")

# ── data ─────────────────────────────────────────────────────────────────────
print("Loading datasets ...")
train_ds    = ModelNetDataset(root=DATA_ROOT, split="train", num_points=NUM_POINTS)
test_ds     = ModelNetDataset(root=DATA_ROOT, split="test",  num_points=NUM_POINTS)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32,         shuffle=False, num_workers=0)
print(f"  Train samples : {len(train_ds)}")
print(f"  Test  samples : {len(test_ds)}\n")


# ── training helper ──────────────────────────────────────────────────────────
def train_model(model_type: str):
    print("=" * 62)
    print(f"  Training {model_type.upper()}  ({EPOCHS} epochs, slice mode)")
    print("=" * 62)

    if model_type == "snn":
        model = PointNetSNN(
            point_dims=[128, 256, 512], temporal_dim=512, num_classes=10
        ).to(device)
    else:
        model = PointNetANN(
            point_dims=[128, 256, 512], temporal_dim=512, num_classes=10
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}\n")

    optimizer = build_optimizer(model, lr=LR)
    history   = {"loss": [], "aux": [], "acc": []}

    for epoch in range(EPOCHS):
        print(f"\n--- {model_type.upper()} Epoch {epoch + 1}/{EPOCHS} ---")
        loss, aux, acc = train_one_epoch(
            model, train_loader, optimizer, device,
            num_slices=NUM_SLICES, aux_weight=AUX_WEIGHT,
        )
        history["loss"].append(loss)
        history["aux"].append(aux)
        history["acc"].append(acc)
        print(f"  >> Loss: {loss:.4f} | Aux: {aux:.4f} | Acc: {acc:.4f}")

        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"{model_type}_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  >> Checkpoint saved: {ckpt}")

    final_ckpt = os.path.join(OUTPUT_DIR, f"{model_type}_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"\n  Final checkpoint: {final_ckpt}")
    return model, history


# ── run training ─────────────────────────────────────────────────────────────
ann_model, ann_hist = train_model("ann")
snn_model, snn_hist = train_model("snn")


# ── training curves plot ──────────────────────────────────────────────────────
print("\nSaving training curves ...")
epochs_x = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(epochs_x, ann_hist["loss"], "b-o", label="ANN")
ax.plot(epochs_x, snn_hist["loss"], "r-s", label="SNN")
ax.set_xlabel("Epoch"); ax.set_ylabel("CE Loss (final slice)")
ax.set_title("Training Loss"); ax.legend(); ax.grid(True)

ax = axes[1]
ax.plot(epochs_x, ann_hist["acc"], "b-o", label="ANN")
ax.plot(epochs_x, snn_hist["acc"], "r-s", label="SNN")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Training Accuracy"); ax.legend(); ax.grid(True)

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"  Saved: {curve_path}")


# ── inference ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Inference (all 4 modes)")
print("=" * 62)
results = {}

print("\n[1/4] ANN + Full ...")
r = infer_ann_full(ann_model, test_loader, device)
results["ANN+Full"] = r
print(f"  Accuracy: {r['final_accuracy']:.4f}")

print("\n[2/4] ANN + Slice ...")
r = infer_ann_slice(ann_model, test_loader, device, num_slices=NUM_SLICES)
results["ANN+Slice"] = r
print(f"  Accuracy: {r['final_accuracy']:.4f}  |  Mean exit step: {r['mean_exit']:.2f}")

print("\n[3/4] SNN + Full ...")
r = infer_snn_full(snn_model, test_loader, device)
results["SNN+Full"] = r
print(f"  Accuracy: {r['final_accuracy']:.4f}")

print("\n[4/4] SNN + Slice ...")
r = infer_snn_slice(snn_model, test_loader, device, num_slices=NUM_SLICES)
results["SNN+Slice"] = r
print(f"  Accuracy: {r['final_accuracy']:.4f}  |  Mean exit step: {r['mean_exit']:.2f}")


# ── inference plots ───────────────────────────────────────────────────────────
print("\nGenerating inference plots ...")
plot_all_metrics(results, OUTPUT_DIR)
print(f"  Saved to: {OUTPUT_DIR}")


# ── results summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  RESULTS SUMMARY")
print("=" * 62)
for name, res in results.items():
    line = f"  {name:<14}  Acc = {res['final_accuracy']:.4f}"
    if "mean_exit" in res:
        line += f"  |  Mean exit step = {res['mean_exit']:.2f} / {NUM_SLICES}"
    print(line)

print("\nPlots saved:")
for fname in [
    "training_curves.png",
    "accuracy_vs_timestep.png",
    "exit_histogram_snn.png",
    "threshold_tradeoff.png",
    "exit_cdf.png",
    "confidence_growth.png",
]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    status = "OK" if os.path.exists(fpath) else "MISSING"
    print(f"  [{status}]  {fpath}")

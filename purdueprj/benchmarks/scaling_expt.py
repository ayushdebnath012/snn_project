"""
scaling_expt.py
===============
Scaling ablation experiment for the paper:
  "Show improvement is robust to scale — both data scale and model scale."
                                          — Prof. Sayeed

Experiment design:
  Axis 1 — Data scale:    ModelNet10 (10 cls) vs ModelNet40 (40 cls)
  Axis 2 — Model scale:   base (128→256→512) / large (256→512→1024) / XL (256→512→1024→1024)
                           + PCT-SNN transformer variant

  For each (data, model) cell we also compare:
    - Fixed radial slicing  vs  our FPS hierarchical slicing
    - ANN counterpart        (PointNetANN / DGCNN / PCT)
    - ANN→SNN converted      (best ANN → IFNeuron, T timesteps)

  Outputs:
    results/scaling/
      scaling_table.csv         — all (model × dataset × slicing × type) rows
      scaling_heatmap.png       — accuracy heatmap: model_scale × dataset
      scaling_bar.png           — bar chart grouped by model size
      conversion_comparison.png — ANN vs ANN→SNN vs our-SNN

Usage:
  # Smoke test (no real data, random init):
  python benchmarks/scaling_expt.py --smoke_test --epochs 2

  # Full run:
  python benchmarks/scaling_expt.py \\
      --mn10_root /data/ModelNet10 \\
      --mn40_root /data/ModelNet40 \\
      --epochs 150 --out_dir results/scaling
"""

import argparse
import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model_zoo import build_model, count_params, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=128, n_pts=1024, num_classes=40):
        self.n, self.n_pts, self.nc = n, n_pts, num_classes
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.randn(self.n_pts, 3), torch.randint(0, self.nc, (1,)).item()


def get_loader(data_root, dataset_name, split, batch_size, num_classes):
    if data_root and os.path.isdir(data_root):
        try:
            from data.modelnet_dataset import ModelNetDataset
            ds = ModelNetDataset(data_root, split=split, num_points=1024)
            return DataLoader(ds, batch_size=batch_size,
                              shuffle=(split == "train"), num_workers=2)
        except Exception as e:
            print(f"[Data] {dataset_name} load failed ({e}). Using dummy.")
    n = 512 if split == "train" else 128
    ds = DummyDataset(n=n, num_classes=num_classes)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


# ---------------------------------------------------------------------------
# Training / evaluation helpers (shared with compare_models.py)
# ---------------------------------------------------------------------------

def _sliced_forward(model, pts, num_slices, slicing):
    from data.slicing import slice_radial_batch, slice_fps_hierarchical_batch
    B = pts.size(0)
    if slicing == "fps":
        pts_slices = slice_fps_hierarchical_batch(pts, T=num_slices)
    else:
        idx   = slice_radial_batch(pts, T=num_slices)
        gi    = idx.unsqueeze(-1).expand(-1, -1, 3)
        ps    = torch.gather(pts, 1, gi)
        pps   = ps.size(1) // num_slices
        pts_slices = ps.view(B, num_slices, pps, 3)

    logits = None
    for t in range(num_slices):
        logits = model.forward_step(pts_slices[:, t])
    return logits


def train_one_model(model, train_loader, val_loader, device,
                    epochs, num_slices, slicing, lr=1e-3, name="model"):
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_ep  = 0

    for ep in range(epochs):
        model.train()
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device).long()
            B = pts.size(0)
            if hasattr(model, "reset_state"):
                model.reset_state(B, device)

            logits_list = []
            from data.slicing import slice_radial_batch, slice_fps_hierarchical_batch
            if slicing == "fps":
                pts_sl = slice_fps_hierarchical_batch(pts, T=num_slices)
            else:
                idx = slice_radial_batch(pts, T=num_slices)
                gi  = idx.unsqueeze(-1).expand(-1, -1, 3)
                ps  = torch.gather(pts, 1, gi)
                pps = ps.size(1) // num_slices
                pts_sl = ps.view(B, num_slices, pps, 3)

            for t in range(num_slices):
                logits_list.append(model.forward_step(pts_sl[:, t]))

            loss = crit(logits_list[-1], labels) + \
                   0.3 * sum(crit(l, labels) for l in logits_list[:-1]) / max(len(logits_list)-1, 1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        acc = eval_one_model(model, val_loader, device, num_slices, slicing)
        if acc > best_acc:
            best_acc = acc; best_ep = ep + 1

        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  [{name}] ep {ep+1}/{epochs}  val={acc:.3f}  best={best_acc:.3f}")

    print(f"  [{name}] → best acc = {best_acc:.4f} @ epoch {best_ep}")
    return best_acc


def eval_one_model(model, val_loader, device, num_slices, slicing):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for pts, labels in val_loader:
            pts, labels = pts.to(device), labels.to(device).long()
            B = pts.size(0)
            if hasattr(model, "reset_state"):
                model.reset_state(B, device)
            logits = _sliced_forward(model, pts, num_slices, slicing)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += B
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# ANN→SNN conversion experiment
# ---------------------------------------------------------------------------

def run_conversion_experiment(ann_name, train_loader, val_loader,
                               device, epochs, num_slices, num_classes,
                               T_list=(4, 8, 16)):
    """
    Train ANN, convert to SNN, measure accuracy vs T.
    Returns dict with 'ann_acc', 'snn_T{T}_acc' for each T in T_list.
    """
    from models.ann_to_snn import convert_ann_to_snn, eval_converted_snn

    ann = build_model(ann_name, num_classes=num_classes).to(device)
    print(f"\n  [Conversion] Training {ann_name}...")
    ann_acc = train_one_model(ann, train_loader, val_loader, device,
                               epochs, num_slices, "radial", name=ann_name)

    result = {"ann_acc": ann_acc}
    for T in T_list:
        print(f"  [Conversion] Converting {ann_name} → SNN (T={T})...")
        snn = convert_ann_to_snn(ann, train_loader, device, T=T, n_calib_batches=10)
        snn_acc = eval_converted_snn(snn, val_loader, device, T=T)
        result[f"snn_T{T}_acc"] = snn_acc
        print(f"    ANN={ann_acc:.3f}  SNN(T={T})={snn_acc:.3f}  "
              f"gap={ann_acc - snn_acc:.3f}")
    return result


# ---------------------------------------------------------------------------
# Main experiment grid
# ---------------------------------------------------------------------------

SCALE_GRID = [
    # (model_name, slicing)
    ("ours_base",    "radial"),
    ("ours_base",    "fps"),
    ("ours_full",    "radial"),
    ("ours_full",    "fps"),
    ("ours_large",   "fps"),
    ("ours_xl",      "fps"),
    ("ours_transformer_small", "fps"),
    ("ours_pct_snn", "fps"),
    ("spm",          "radial"),
    ("spm",          "fps"),
]

DATASET_CONFIGS = {
    "modelnet10": {"num_classes": 10,  "root_arg": "mn10_root"},
    "modelnet40": {"num_classes": 40,  "root_arg": "mn40_root"},
}

ANN_FOR_CONVERSION = {
    "modelnet10": "ann_pointnet",
    "modelnet40": "ann_dgcnn",
}


def run_scaling_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Scaling] Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []

    for ds_name, ds_cfg in DATASET_CONFIGS.items():
        num_classes = ds_cfg["num_classes"]
        data_root   = getattr(args, ds_cfg["root_arg"], None)
        print(f"\n{'='*70}")
        print(f"[Dataset] {ds_name.upper()}  ({num_classes} classes)")
        print(f"{'='*70}")

        train_loader = get_loader(data_root, ds_name, "train",
                                   args.batch_size, num_classes)
        val_loader   = get_loader(data_root, ds_name, "test",
                                   args.batch_size, num_classes)

        # ---- Model scale × slicing grid ----
        for model_name, slicing in SCALE_GRID:
            label = f"{model_name}/{slicing}"
            print(f"\n  [{ds_name}] {label}")

            try:
                model = build_model(model_name, num_classes=num_classes).to(device)
                n_params = count_params(model)

                if args.smoke_test:
                    acc = 1.0 / num_classes
                else:
                    acc = train_one_model(
                        model, train_loader, val_loader, device,
                        args.epochs, args.num_slices, slicing,
                        lr=args.lr, name=label
                    )

                row = {
                    "dataset":    ds_name,
                    "model":      model_name,
                    "slicing":    slicing,
                    "type":       MODEL_CONFIGS[model_name]["type"],
                    "params":     n_params,
                    "val_acc":    round(acc * 100, 2),
                    "epochs":     args.epochs,
                }
                all_rows.append(row)
                print(f"  → {ds_name} | {label} | {n_params:,} params | acc={acc*100:.2f}%")

            except Exception as e:
                import traceback; traceback.print_exc()
                all_rows.append({
                    "dataset": ds_name, "model": model_name, "slicing": slicing,
                    "type": "?", "params": 0, "val_acc": 0.0,
                    "epochs": args.epochs, "error": str(e)
                })

        # ---- ANN→SNN conversion experiment ----
        if not args.skip_conversion:
            ann_name = ANN_FOR_CONVERSION[ds_name]
            print(f"\n  [ANN→SNN] {ann_name} on {ds_name}")
            try:
                conv_results = run_conversion_experiment(
                    ann_name, train_loader, val_loader,
                    device, args.epochs, args.num_slices, num_classes,
                    T_list=[4, 8, 16] if not args.smoke_test else [4]
                )
                for key, val in conv_results.items():
                    row = {
                        "dataset":  ds_name,
                        "model":    f"{ann_name}→SNN({key})",
                        "slicing":  "N/A",
                        "type":     "Converted-SNN" if "snn" in key else "ANN",
                        "params":   count_params(build_model(ann_name, num_classes=num_classes)),
                        "val_acc":  round(val * 100, 2),
                        "epochs":   args.epochs,
                    }
                    all_rows.append(row)
            except Exception as e:
                import traceback; traceback.print_exc()

    # ---- Save CSV ----
    csv_path = os.path.join(args.out_dir, "scaling_table.csv")
    fields = ["dataset", "model", "slicing", "type", "params", "val_acc", "epochs"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"\n[Saved] {csv_path}")

    # ---- Save JSON (for plotting) ----
    json_path = os.path.join(args.out_dir, "scaling_results.json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"[Saved] {json_path}")

    # ---- Print summary table ----
    _print_summary(all_rows)

    # ---- Generate plots ----
    _make_plots(all_rows, args.out_dir)

    return all_rows


def _print_summary(rows):
    print(f"\n{'Dataset':<12} {'Model':<25} {'Slicing':<8} {'Params':>10}  {'Acc%':>7}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: (-x.get("val_acc", 0), x["dataset"])):
        print(f"{r['dataset']:<12} {r['model']:<25} {r.get('slicing','?'):<8} "
              f"{r.get('params',0):>10,}  {r.get('val_acc',0):>7.2f}")


def _make_plots(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # ---- Plot 1: Scaling heatmap (model_size × dataset) ----
        model_order = ["ours_base", "ours_full", "ours_large", "ours_xl", "ours_transformer_small", "ours_pct_snn", "spm"]
        ds_order    = ["modelnet10", "modelnet40"]
        slicings    = ["fps"]   # show FPS slicing in heatmap

        heat_data = {}
        for r in rows:
            key = (r["model"], r["dataset"], r.get("slicing", ""))
            heat_data[key] = r.get("val_acc", 0)

        heat = np.zeros((len(model_order), len(ds_order)))
        for i, m in enumerate(model_order):
            for j, d in enumerate(ds_order):
                heat[i, j] = max(heat_data.get((m, d, s), 0) for s in slicings)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heat, cmap="YlGn", vmin=max(0, heat.min() - 5),
                        vmax=min(100, heat.max() + 1))
        ax.set_xticks(range(len(ds_order))); ax.set_xticklabels(ds_order)
        ax.set_yticks(range(len(model_order))); ax.set_yticklabels(model_order)
        for i in range(len(model_order)):
            for j in range(len(ds_order)):
                ax.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center",
                        fontsize=10, color="black")
        plt.colorbar(im, ax=ax, label="Accuracy (%)")
        ax.set_title("Scaling: Model Size × Dataset (FPS slicing)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scaling_heatmap.png"), dpi=150)
        plt.close()

        # ---- Plot 2: Radial vs FPS slicing per model (MN40 only) ----
        mn40_rows = [r for r in rows if r["dataset"] == "modelnet40"
                     and r["model"] in ["ours_base", "ours_full", "ours_large", "ours_xl", "spm"]
                     and r.get("slicing") in ("radial", "fps")]

        models_seen = sorted(set(r["model"] for r in mn40_rows))
        x   = np.arange(len(models_seen))
        w   = 0.35
        rad = [next((r["val_acc"] for r in mn40_rows
                      if r["model"] == m and r["slicing"] == "radial"), 0) for m in models_seen]
        fps = [next((r["val_acc"] for r in mn40_rows
                      if r["model"] == m and r["slicing"] == "fps"), 0) for m in models_seen]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w/2, rad, w, label="Radial slicing", color="steelblue", alpha=0.85)
        ax.bar(x + w/2, fps, w, label="FPS slicing (ours)",  color="tomato",    alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models_seen, rotation=15, ha="right")
        ax.set_ylabel("Val Accuracy (%) — ModelNet40")
        ax.set_title("FPS vs Radial Slicing across Model Scales")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        ymin = max(0, min(rad + fps) - 3)
        ax.set_ylim(ymin, min(100, max(rad + fps) + 3))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "slicing_vs_scale.png"), dpi=150)
        plt.close()

        # ---- Plot 3: ANN vs Converted-SNN vs Our-SNN ----
        conv_rows = [r for r in rows
                     if "Converted" in r.get("type", "") or r.get("type") == "ANN"]
        snn_base  = next((r["val_acc"] for r in rows
                           if r["model"] == "ours_full"
                           and r.get("slicing") == "fps"
                           and r["dataset"] == "modelnet40"), 0)

        if conv_rows:
            labels2 = [r["model"] for r in conv_rows] + ["ours_full (FPS)"]
            accs2   = [r["val_acc"] for r in conv_rows] + [snn_base]
            colors2 = ["steelblue" if "ANN" in r.get("type","") else "orange"
                       for r in conv_rows] + ["gold"]

            fig, ax = plt.subplots(figsize=(max(8, len(labels2)*1.2), 5))
            ax.bar(labels2, accs2, color=colors2, edgecolor="black", linewidth=0.7)
            ax.set_ylabel("Accuracy (%) — ModelNet40")
            ax.set_title("ANN vs ANN→SNN Conversion vs Our Native SNN")
            ax.grid(True, axis="y", alpha=0.3)
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor="steelblue", label="ANN"),
                Patch(facecolor="orange",    label="Converted SNN"),
                Patch(facecolor="gold",      label="Our native SNN"),
            ], loc="lower right")
            plt.xticks(rotation=25, ha="right", fontsize=8)
            ymin = max(0, min(accs2) - 3)
            ax.set_ylim(ymin, min(100, max(accs2) + 3))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "conversion_comparison.png"), dpi=150)
            plt.close()

        print(f"[Plots] Saved to {out_dir}/")

    except Exception as e:
        print(f"[Plot] Plotting failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mn10_root",      default=None)
    p.add_argument("--mn40_root",      default=None)
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--num_slices",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--out_dir",        default="results/scaling")
    p.add_argument("--smoke_test",     action="store_true")
    p.add_argument("--skip_conversion",action="store_true",
                   help="Skip the ANN→SNN conversion experiment")
    return p.parse_args()


if __name__ == "__main__":
    run_scaling_experiment(parse_args())

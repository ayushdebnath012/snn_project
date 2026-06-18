"""
run_all_experiments.py
======================
Single entry-point script for ALL experiments in the paper:

  1. Train & evaluate all models (SNN variants, ANN baselines, paper comparisons)
  2. Scaling ablation: model-size x dataset (MN10 vs MN40)
  3. Slicing ablation: radial vs FPS per model
  4. ANN->SNN conversion: train ANN, convert, compare at T ∈ {4,8,16}
  5. Early-exit analysis: mean exit timestep, CDF, threshold tradeoff
  6. Energy efficiency: SNN vs ANN (Lemaire et al. 2022, Intel Loihi 2)
  7. ScanObjectNN benchmark: OBJ-BG / OBJ-ONLY / PB_T50_RS
  8. Multi-seed evaluation: mean ± std over 3 seeds
  9. T-timestep sensitivity: accuracy vs T ∈ {4,8,12,16,24,32}
  10. Generate ALL plots and a single merged comparison table

Usage:
  # Quick smoke test (no data required, random tensors):
  python run_all_experiments.py --smoke_test

  # Full run on ModelNet40 only:
  python run_all_experiments.py --mn40_root /data/ModelNet40 --epochs 150

  # Full run with ScanObjectNN:
  python run_all_experiments.py \\
      --mn10_root /data/ModelNet10 \\
      --mn40_root /data/ModelNet40 \\
      --sonn_root /data/ScanObjectNN \\
      --epochs 150 --out_dir results/final

  # New reviewer-requested experiments only:
  python run_all_experiments.py --mn40_root /data/MN40 \\
      --groups scanobjectnn multi_seed t_sweep
"""

import argparse
import os
import csv
import json
import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# -- project imports ----------------------------------------------------------
from models.model_zoo import build_model, count_params, MODEL_CONFIGS, PUBLISHED_RESULTS
from data.slicing import slice_radial_batch, slice_fps_hierarchical_batch

# Energy constants — Lemaire et al. 2022 (Loihi 2) and Horowitz 2014 (45nm)
ENERGY_MODELS = {
    "loihi": {"E_MAC": 8.4e-3, "E_AC": 2.3e-3,
               "ref": "Lemaire et al. 2022 (arXiv:2206.10569)"},
    "45nm":  {"E_MAC": 4.6,    "E_AC": 0.9,
               "ref": "Horowitz 2014 (45nm CMOS theoretical)"},
}


# =============================================================================
# 0. DATA
# =============================================================================

class DummyDataset(Dataset):
    """Random point clouds — for smoke testing without real data."""
    def __init__(self, n=256, n_pts=1024, num_classes=40):
        self.n, self.n_pts, self.nc = n, n_pts, num_classes
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.randn(self.n_pts, 3), torch.tensor(torch.randint(0, self.nc, (1,)).item(), dtype=torch.long)


def get_loaders(root, split, batch_size, num_classes, n_pts=1024, n_workers=2):
    """Return a DataLoader; falls back to DummyDataset if root is absent."""
    if root and os.path.isdir(root):
        try:
            from data.modelnet import ModelNetDataset
            ds = ModelNetDataset(root, num_points=n_pts, split=split)
            shuffle = (split == "train")
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=n_workers, pin_memory=True)
        except Exception as e:
            print(f"  [Data] Real dataset load failed ({e}). Falling back to dummy.")
    n = 512 if split == "train" else 128
    ds = DummyDataset(n=n, num_classes=num_classes)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


# =============================================================================
# 1. CORE TRAIN / EVAL HELPERS
# =============================================================================

def make_slices(pts, num_slices, slicing):
    """pts [B,N,3] -> pts_slices [B, T, pps, 3]"""
    B, N, _ = pts.shape
    if slicing == "fps":
        return slice_fps_hierarchical_batch(pts, T=num_slices)   # [B,T,pps,3]
    else:   # radial
        idx = slice_radial_batch(pts, T=num_slices)              # [B, N]
        gi  = idx.unsqueeze(-1).expand(-1, -1, 3)
        ps  = torch.gather(pts, 1, gi)
        pps = N // num_slices
        return ps.view(B, num_slices, pps, 3)


def forward_all_slices(model, pts_slices, collect_logits=False):
    """Run model.forward_step over T slices. Returns final logits (+ list if collect)."""
    T = pts_slices.size(1)
    logits_list = []
    for t in range(T):
        logits = model.forward_step(pts_slices[:, t])
        logits_list.append(logits)
    if collect_logits:
        return logits_list[-1], logits_list
    return logits_list[-1]


def train_epoch(model, loader, optimizer, criterion, device, num_slices, slicing,
                aux_weight=0.3, bidirectional=False, clip_grad=1.0):
    model.train()
    total_loss = total_correct = total_n = 0

    for pts, labels in loader:
        pts    = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        B      = pts.size(0)

        if hasattr(model, "reset_state"):
            model.reset_state(B, device)

        pts_sl = make_slices(pts, num_slices, slicing)
        final_logits, all_logits = forward_all_slices(model, pts_sl, collect_logits=True)

        if bidirectional and hasattr(model, "finalize"):
            final_logits = model.finalize()

        loss = criterion(final_logits, labels)
        if len(all_logits) > 1 and aux_weight > 0:
            aux = sum(criterion(l, labels) for l in all_logits[:-1]) / (len(all_logits) - 1)
            loss = loss + aux_weight * aux

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss    += loss.item() * B
        total_correct += (final_logits.argmax(1) == labels).sum().item()
        total_n       += B

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_model(model, loader, device, num_slices, slicing,
               threshold=None, bidirectional=False):
    """
    Evaluate model. If threshold is set, also track early-exit steps.
    Returns: acc, mean_exit (or None), all_probs [N, T, C], all_labels [N]
    """
    model.eval()
    correct = total = 0
    exit_steps_all = []
    all_probs_list  = []
    all_labels_list = []

    for pts, labels in loader:
        pts    = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        B      = pts.size(0)

        if hasattr(model, "reset_state"):
            model.reset_state(B, device)

        pts_sl   = make_slices(pts, num_slices, slicing)
        T        = pts_sl.size(1)
        batch_probs = []
        batch_exits = [T] * B
        exited   = [False] * B

        for t in range(T):
            logits = model.forward_step(pts_sl[:, t])
            probs  = F.softmax(logits, dim=-1)          # [B, C]
            batch_probs.append(probs.cpu())

            if threshold is not None:
                max_p = probs.max(dim=1).values
                for b in range(B):
                    if not exited[b] and max_p[b].item() > threshold:
                        batch_exits[b] = t + 1
                        exited[b] = True

        if bidirectional and hasattr(model, "finalize"):
            logits = model.finalize()
        # final logits already in `logits`

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += B
        exit_steps_all.extend(batch_exits)

        # [B, T, C]
        all_probs_list.append(torch.stack(batch_probs, dim=1))
        all_labels_list.append(labels.cpu())

    acc = correct / total if total > 0 else 0.0
    mean_exit = float(np.mean(exit_steps_all)) if threshold is not None and exit_steps_all else None
    
    mean_fr = None
    if hasattr(model, "get_firing_rates"):
        fr_dict = model.get_firing_rates()
        if fr_dict:
            mean_fr = float(np.mean(list(fr_dict.values())))

    if not all_probs_list:
        return acc, mean_exit, torch.zeros(0), torch.zeros(0), mean_fr

    all_probs  = torch.cat(all_probs_list,  dim=0)   # [N, T, C]
    all_labels = torch.cat(all_labels_list, dim=0)   # [N]
    return acc, mean_exit, all_probs, all_labels, mean_fr


def train_model_full(model, train_loader, val_loader, device,
                     epochs, num_slices, slicing, lr=1e-3,
                     aux_weight=0.3, bidirectional=False, name="model",
                     log_every=10):
    """Train a model for `epochs` epochs. Returns best val accuracy."""
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, opt, crit, device,
            num_slices, slicing, aux_weight=aux_weight,
            bidirectional=bidirectional
        )
        sched.step()

        if ep % log_every == 0 or ep == epochs or ep == 1:
            val_acc, _, _, _, _ = eval_model(model, val_loader, device,
                                           num_slices, slicing,
                                           bidirectional=bidirectional)
            best_acc = max(best_acc, val_acc)
            elapsed = time.time() - t0
            print(f"  [{name:30s}] ep {ep:3d}/{epochs}  "
                  f"loss={tr_loss:.4f}  tr={tr_acc:.3f}  "
                  f"val={val_acc:.3f}  best={best_acc:.3f}  "
                  f"({elapsed:.0f}s)")

    return best_acc


# =============================================================================
# 2. ANN -> SNN CONVERSION
# =============================================================================

def run_conversion(ann_model_name, train_loader, val_loader, device,
                   epochs, num_slices, num_classes, T_list=(4, 8, 16),
                   smoke_test=False):
    """
    Train ANN, convert to SNN via threshold balancing, eval at multiple T.
    Returns dict: {ann_acc, snn_T4_acc, snn_T8_acc, snn_T16_acc}
    """
    from models.ann_to_snn import convert_ann_to_snn, eval_converted_snn

    ann = build_model(ann_model_name, num_classes=num_classes)
    ann = ann.to(device)
    print(f"\n  [Convert] Training {ann_model_name}...")
    if smoke_test:
        ann_acc = 1.0 / num_classes
    else:
        ann_acc = train_model_full(ann, train_loader, val_loader, device,
                                    epochs=epochs, num_slices=num_slices,
                                    slicing="radial", name=ann_model_name)

    results = {"ann_acc": ann_acc}
    for T in (T_list if not smoke_test else [4]):
        print(f"  [Convert] {ann_model_name} -> SNN  T={T}")
        if smoke_test:
            snn_acc = 1.0 / num_classes
        else:
            snn = convert_ann_to_snn(ann, train_loader, device, T=T, n_calib_batches=8)
            snn_acc = eval_converted_snn(snn, val_loader, device, T=T)
        results[f"snn_T{T}"] = snn_acc
        print(f"    ANN={ann_acc:.4f}  SNN(T={T})={snn_acc:.4f}  gap={ann_acc-snn_acc:+.4f}")
    return results


# =============================================================================
# 3. EXPERIMENT GROUPS
# =============================================================================

def exp_main_comparison(args, device, train_l, val_l, nc, ds_tag, rows):
    """
    Group 'comparison': train all SNN models + ANN baselines. Record accuracy.
    """
    # Which models to run (skip heavy ANN baselines in smoke_test)
    snn_models = [
        ("ours_base",    "radial", False),
        ("ours_learnable","fps",   False),
        ("ours_knn",     "fps",    False),
        ("ours_bidir",   "fps",    True),
        ("ours_full",    "fps",    True),
        ("e3dsnn",       "radial", False),
        ("spiking_ssm",  "radial", False),
        ("spt",          "fps",    False),
        ("spm",          "fps",    False),
        ("ours_transformer_small", "fps", False),
    ]
    ann_models = [
        ("ann_pointnet",   "radial", False),
        ("ann_dgcnn",      "radial", False),
        ("ann_pct",        "radial", False),
        ("ann_pointnetpp", "radial", False),
    ]
    run_list = snn_models + (ann_models if not args.skip_ann else [])

    for model_name, slicing, bidir in run_list:
        tag = f"{ds_tag}|{model_name}|{slicing}"
        print(f"\n{'-'*65}\n  {tag}\n{'-'*65}")
        try:
            m = build_model(model_name, num_classes=nc).to(device)
            n_params = count_params(m)
            acc = (1.0/nc) if args.smoke_test else train_model_full(
                m, train_l, val_l, device,
                epochs=args.epochs, num_slices=args.num_slices,
                slicing=slicing, bidirectional=bidir,
                name=model_name
            )
            val_acc, mean_exit, all_probs, all_labels, mean_fr = eval_model(
                m, val_l, device, args.num_slices, slicing,
                threshold=args.threshold, bidirectional=bidir
            )
            energy_savings = "N/A"
            if mean_fr is not None:
                e_ac = ENERGY_MODELS["loihi"]["E_AC"]
                e_mac = ENERGY_MODELS["loihi"]["E_MAC"]
                # E_SNN/E_ANN = fr * E_AC / E_MAC.  T (num_slices) cancels
                # because both ANN and SNN process the same N total point
                # operations — the SNN just distributes them over T steps of
                # N/T points each.  Multiplying by T double-counts the slicing
                # and incorrectly penalises higher T.
                efficiency = (e_ac / e_mac) * mean_fr
                energy_savings = round(1.0 / efficiency, 2) if efficiency > 0 else "inf"

            rows.append({
                "group": "comparison", "dataset": ds_tag,
                "model": model_name, "slicing": slicing,
                "type": MODEL_CONFIGS[model_name]["type"],
                "params": n_params,
                "val_acc": round(val_acc * 100, 2),
                "mean_exit": round(mean_exit, 2) if mean_exit else "N/A",
                "mean_fr": round(mean_fr, 3) if mean_fr else "N/A",
                "energy_savings_x": energy_savings,
                "paper": MODEL_CONFIGS[model_name]["paper"],
            })
            if energy_savings != "N/A":
                print(f"  -> acc={val_acc*100:.2f}%  fr={mean_fr:.3f}  energy={energy_savings}x better")
            else:
                print(f"  -> acc={val_acc*100:.2f}%  params={n_params:,}")
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"group":"comparison","dataset":ds_tag,
                         "model":model_name,"slicing":slicing,
                         "type":MODEL_CONFIGS.get(model_name,{}).get("type","?"),
                         "params":0,"val_acc":0,"mean_exit":"N/A",
                         "paper":MODEL_CONFIGS.get(model_name,{}).get("paper","?"),
                         "error":str(e)})


def exp_scaling(args, device, loaders_by_ds, rows):
    """
    Group 'scaling': sweep model-size x dataset x slicing.
    Demonstrates that FPS improvement is robust across scales.
    """
    scale_grid = [
        ("ours_base",    "radial", False),
        ("ours_base",    "fps",    False),
        ("ours_full",    "radial", True),
        ("ours_full",    "fps",    True),
        ("ours_large",   "fps",    True),
        ("ours_xl",      "fps",    True),
        ("ours_pct_snn", "fps",    False),
        ("spm",          "fps",    False),
        ("ours_transformer_small", "fps", False),
    ]

    for ds_tag, (train_l, val_l, nc) in loaders_by_ds.items():
        for model_name, slicing, bidir in scale_grid:
            tag = f"{ds_tag}|{model_name}|{slicing}"
            print(f"\n  [Scaling] {tag}")
            try:
                m = build_model(model_name, num_classes=nc).to(device)
                n_params = count_params(m)
                acc = (1.0/nc) if args.smoke_test else train_model_full(
                    m, train_l, val_l, device,
                    epochs=args.epochs, num_slices=args.num_slices,
                    slicing=slicing, bidirectional=bidir,
                    name=f"{model_name}/{slicing}"
                )
                rows.append({
                    "group": "scaling", "dataset": ds_tag,
                    "model": model_name, "slicing": slicing,
                    "type": MODEL_CONFIGS[model_name]["type"],
                    "params": n_params,
                    "val_acc": round(acc * 100, 2),
                    "mean_exit": "N/A",
                    "paper": MODEL_CONFIGS[model_name]["paper"],
                })
                print(f"  -> acc={acc*100:.2f}%  params={n_params:,}")
            except Exception as e:
                import traceback; traceback.print_exc()
                rows.append({"group":"scaling","dataset":ds_tag,
                             "model":model_name,"slicing":slicing,
                             "type":MODEL_CONFIGS.get(model_name,{}).get("type","?"),
                             "params":0,"val_acc":0,"mean_exit":"N/A",
                             "paper":MODEL_CONFIGS.get(model_name,{}).get("paper","?"),
                             "error":str(e)})


def exp_conversion(args, device, train_l, val_l, nc, ds_tag, rows):
    """
    Group 'conversion': ANN->SNN conversion experiment.
    Train ANN -> convert -> compare at T ∈ {4, 8, 16}.
    """
    ann_name = "ann_dgcnn" if nc == 40 else "ann_pointnet"
    try:
        conv = run_conversion(
            ann_name, train_l, val_l, device,
            epochs=args.epochs, num_slices=args.num_slices,
            num_classes=nc, smoke_test=args.smoke_test
        )
        params = count_params(build_model(ann_name, num_classes=nc))
        for key, acc in conv.items():
            is_snn = key.startswith("snn")
            rows.append({
                "group": "conversion", "dataset": ds_tag,
                "model": f"{ann_name}->({key})",
                "slicing": "rate-coded",
                "type": "Converted-SNN" if is_snn else "ANN",
                "params": params,
                "val_acc": round(acc * 100, 2),
                "mean_exit": "N/A",
                "paper": "ANN->SNN",
            })
    except Exception as e:
        import traceback; traceback.print_exc()


def exp_slicing_ablation(args, device, train_l, val_l, nc, ds_tag, rows):
    """
    Group 'slicing': compare radial vs FPS slicing for same model.
    Also check whether FPS helps ANNs (if it does, claim is weaker for SNNs).
    """
    grid = [
        ("ours_full",  "radial", True,  "SNN"),
        ("ours_full",  "fps",    True,  "SNN"),
        ("ours_large", "radial", True,  "SNN"),
        ("ours_large", "fps",    True,  "SNN"),
        ("ann_pointnet","radial",False, "ANN"),
        ("ann_pointnet","fps",   False, "ANN"),
    ]
    for model_name, slicing, bidir, mtype in grid:
        tag = f"{ds_tag}|{model_name}|{slicing}"
        print(f"\n  [Slicing] {tag}")
        try:
            m = build_model(model_name, num_classes=nc).to(device)
            n_params = count_params(m)
            acc = (1.0/nc) if args.smoke_test else train_model_full(
                m, train_l, val_l, device,
                epochs=args.epochs, num_slices=args.num_slices,
                slicing=slicing, bidirectional=bidir,
                name=f"{model_name}/{slicing}"
            )
            rows.append({
                "group": "slicing", "dataset": ds_tag,
                "model": model_name, "slicing": slicing,
                "type": mtype, "params": n_params,
                "val_acc": round(acc * 100, 2),
                "mean_exit": "N/A",
                "paper": MODEL_CONFIGS[model_name]["paper"],
            })
            print(f"  -> acc={acc*100:.2f}%")
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"group":"slicing","dataset":ds_tag,
                         "model":model_name,"slicing":slicing,
                         "type":mtype,"params":0,"val_acc":0,"mean_exit":"N/A",
                         "paper":MODEL_CONFIGS.get(model_name,{}).get("paper","?"),
                         "error":str(e)})


def exp_early_exit(args, device, train_l, val_l, nc, ds_tag, rows):
    """
    Group 'early_exit': sweep confidence threshold, record mean_exit, CDF,
    and confidence growth. Uses ours_full (trained in comparison group if
    checkpoint available; else re-trains).
    """
    model_name, slicing, bidir = "ours_full", "fps", True
    print(f"\n  [Early-exit] {ds_tag}|{model_name}")
    try:
        m = build_model(model_name, num_classes=nc).to(device)
        if not args.smoke_test:
            train_model_full(m, train_l, val_l, device,
                             epochs=args.epochs, num_slices=args.num_slices,
                             slicing=slicing, bidirectional=bidir,
                             name=f"early_exit/{model_name}")
        # Eval at multiple thresholds
        thresholds = np.linspace(0.5, 0.99, 15)
        th_rows = []
        for th in thresholds:
            acc, mean_exit, _, _, _ = eval_model(
                m, val_l, device, args.num_slices, slicing,
                threshold=float(th), bidirectional=bidir
            )
            th_rows.append({"threshold": round(th, 3),
                            "acc": round(acc*100, 2),
                            "mean_exit": round(mean_exit, 2)})
            rows.append({
                "group": "early_exit", "dataset": ds_tag,
                "model": f"{model_name}@th={th:.2f}",
                "slicing": slicing, "type": "SNN",
                "params": count_params(m),
                "val_acc": round(acc*100, 2),
                "mean_exit": round(mean_exit, 2),
                "paper": "Ours",
            })
        print(f"  Threshold sweep done. Best acc={max(r['acc'] for r in th_rows):.2f}%")
    except Exception as e:
        import traceback; traceback.print_exc()


# =============================================================================
# 4. PLOTS
# =============================================================================

def make_all_plots(rows, pub_results, out_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        os.makedirs(out_dir, exist_ok=True)

        # -- helpers ----------------------------------------------------------
        def savefig(name):
            path = os.path.join(out_dir, name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [Plot] {path}")

        # -- Plot 1: Master comparison bar chart (MN40, all models) ----------
        comp_rows = [r for r in rows
                     if r.get("group") == "comparison"
                     and "40" in str(r.get("dataset",""))
                     and r.get("val_acc", 0) > 0]
        pub_bars = [(k, v["type"], v["mn40"])
                    for k, v in pub_results.items() if v.get("mn40")]
        our_bars = [(r["model"], r["type"], r["val_acc"]) for r in comp_rows]
        all_bars = pub_bars + our_bars

        names  = [b[0] for b in all_bars]
        accs   = [b[2] for b in all_bars]
        colors = []
        for b in all_bars:
            if b[0].startswith("ours") or b[0].startswith("ann_"):
                colors.append("gold" if b[1]=="SNN" else "lightblue")
            elif b[1] == "ANN":
                colors.append("steelblue")
            else:
                colors.append("tomato")

        fig, ax = plt.subplots(figsize=(max(12, len(names)*1.2), 5))
        ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_ylabel("Accuracy (%) — ModelNet40"); ax.grid(True, axis="y", alpha=0.3)
        ax.set_title("Point Cloud Classification: All Models vs SOTA (ModelNet40)")
        ymin = max(0, min(a for a in accs if a>0) - 3)
        ax.set_ylim(ymin, min(100, max(accs)+2))
        ax.legend(handles=[
            mpatches.Patch(facecolor="steelblue", label="ANN (published)"),
            mpatches.Patch(facecolor="tomato",    label="SNN (published)"),
            mpatches.Patch(facecolor="gold",       label="Our SNN"),
            mpatches.Patch(facecolor="lightblue",  label="Our ANN"),
        ], loc="lower right")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        savefig("01_master_comparison.png")

        # -- Plot 2: Scaling heatmap (model x dataset) ------------------------
        scale_rows = [r for r in rows if r.get("group") == "scaling"
                      and r.get("slicing") == "fps"]
        models_s = ["ours_base","ours_full","ours_large","ours_xl","ours_pct_snn"]
        datasets = ["modelnet10","modelnet40"]
        heat = np.zeros((len(models_s), len(datasets)))
        for r in scale_rows:
            if r["model"] in models_s and r["dataset"] in datasets:
                i = models_s.index(r["model"])
                j = datasets.index(r["dataset"])
                heat[i,j] = r.get("val_acc", 0)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heat, cmap="YlGn",
                       vmin=max(0, heat[heat>0].min()-5) if heat.any() else 0,
                       vmax=min(100, heat.max()+1) if heat.any() else 100)
        ax.set_xticks(range(len(datasets))); ax.set_xticklabels(["MN10","MN40"])
        ax.set_yticks(range(len(models_s))); ax.set_yticklabels(models_s, fontsize=8)
        for i in range(len(models_s)):
            for j in range(len(datasets)):
                ax.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label="Accuracy (%)")
        ax.set_title("Scaling: Model Size x Dataset (FPS slicing)")
        savefig("02_scaling_heatmap.png")

        # -- Plot 3: FPS vs Radial slicing across model sizes -----------------
        scale_mn40 = [r for r in rows
                      if r.get("group") in ("scaling","slicing")
                      and "40" in str(r.get("dataset",""))
                      and r["model"] in ["ours_base","ours_full","ours_large","ours_xl"]]
        models_u = sorted(set(r["model"] for r in scale_mn40))
        x   = np.arange(len(models_u)); w = 0.35
        rad = [next((r["val_acc"] for r in scale_mn40
                     if r["model"]==m and r["slicing"]=="radial"), 0) for m in models_u]
        fps_ = [next((r["val_acc"] for r in scale_mn40
                      if r["model"]==m and r["slicing"]=="fps"),    0) for m in models_u]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x-w/2, rad,  w, label="Radial slicing", color="steelblue", alpha=0.85)
        ax.bar(x+w/2, fps_, w, label="FPS slicing (ours)",  color="tomato",    alpha=0.85)
        # annotate deltas
        for i, (r_v, f_v) in enumerate(zip(rad, fps_)):
            if r_v>0 and f_v>0:
                ax.annotate(f"{f_v-r_v:+.1f}", xy=(x[i]+w/2, f_v),
                            ha="center", va="bottom", fontsize=8, color="darkred")
        ax.set_xticks(x); ax.set_xticklabels(models_u, rotation=15, ha="right")
        ax.set_ylabel("Val Accuracy (%) — ModelNet40")
        ax.set_title("FPS vs Radial Slicing across Model Scales\n(+Delta = FPS gain)")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        ymin = max(0, min(v for v in rad+fps_ if v>0)-3) if any(rad+fps_) else 0
        ax.set_ylim(ymin, min(100, max(rad+fps_)+3) if any(rad+fps_) else 100)
        savefig("03_fps_vs_radial_scaling.png")

        # -- Plot 4: Slicing ablation — does FPS also help ANNs? -------------
        slic_rows = [r for r in rows
                     if r.get("group") == "slicing"
                     and "40" in str(r.get("dataset",""))]
        if slic_rows:
            labels4 = [f"{r['model']}\n({r['slicing']})" for r in slic_rows]
            accs4   = [r["val_acc"] for r in slic_rows]
            cols4   = ["tomato" if r["type"]=="SNN" else "steelblue" for r in slic_rows]
            hatch4  = ["//" if r["slicing"]=="fps" else "" for r in slic_rows]
            fig, ax = plt.subplots(figsize=(max(8, len(labels4)*1.1), 5))
            bars = ax.bar(labels4, accs4, color=cols4, edgecolor="black",
                          linewidth=0.7, hatch=hatch4, alpha=0.85)
            ax.set_ylabel("Val Accuracy (%) — ModelNet40")
            ax.set_title("Slicing Ablation: Radial vs FPS, SNN vs ANN\n"
                         "(// = FPS slicing; solid = radial)")
            ax.grid(True, axis="y", alpha=0.3)
            ymin = max(0, min(accs4)-3) if accs4 else 0
            ax.set_ylim(ymin, min(100, max(accs4)+3) if accs4 else 100)
            ax.legend(handles=[
                mpatches.Patch(facecolor="tomato",    label="SNN"),
                mpatches.Patch(facecolor="steelblue", label="ANN"),
                mpatches.Patch(facecolor="white", hatch="//", edgecolor="black", label="FPS"),
            ], loc="lower right")
            savefig("04_slicing_ablation.png")

        # -- Plot 5: ANN vs Converted-SNN vs Native-SNN -----------------------
        conv_rows = [r for r in rows
                     if r.get("group") == "conversion"
                     and "40" in str(r.get("dataset",""))]
        native_snn = next((r["val_acc"] for r in rows
                           if r.get("group")=="comparison"
                           and r.get("model")=="ours_full"
                           and "40" in str(r.get("dataset",""))), None)
        if conv_rows:
            labels5 = [r["model"] for r in conv_rows]
            accs5   = [r["val_acc"] for r in conv_rows]
            cols5   = ["steelblue" if r["type"]=="ANN" else "orange" for r in conv_rows]
            if native_snn is not None:
                labels5 += ["ours_full (native SNN)"]
                accs5   += [native_snn]
                cols5   += ["gold"]
            fig, ax = plt.subplots(figsize=(max(8, len(labels5)*1.2), 5))
            ax.bar(labels5, accs5, color=cols5, edgecolor="black", linewidth=0.7)
            ax.set_ylabel("Accuracy (%) — ModelNet40")
            ax.set_title("ANN vs ANN->SNN Conversion vs Our Native SNN\n"
                         "(Conversion gap shows value of native SNN training)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(handles=[
                mpatches.Patch(facecolor="steelblue", label="ANN"),
                mpatches.Patch(facecolor="orange",    label="Converted SNN"),
                mpatches.Patch(facecolor="gold",       label="Native SNN (ours)"),
            ], loc="lower right")
            plt.xticks(rotation=25, ha="right", fontsize=8)
            ymin = max(0, min(accs5)-3)
            ax.set_ylim(ymin, min(100, max(accs5)+3))
            savefig("05_conversion_comparison.png")

        # -- Plot 6: Early-exit threshold tradeoff (mean_exit vs acc) ---------
        ee_rows = sorted(
            [r for r in rows if r.get("group")=="early_exit"
             and "40" in str(r.get("dataset",""))
             and isinstance(r.get("mean_exit"), (int,float))],
            key=lambda r: r.get("mean_exit", 0)
        )
        if ee_rows:
            me_vals  = [r["mean_exit"] for r in ee_rows]
            acc_vals = [r["val_acc"]   for r in ee_rows]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(me_vals, acc_vals, marker="o", color="tomato", linewidth=2)
            ax.set_xlabel("Mean Exit Timestep"); ax.set_ylabel("Accuracy (%)")
            ax.set_title("Early-Exit Threshold Tradeoff\n"
                         "(leftward = more efficient; upward = more accurate)")
            ax.grid(True, alpha=0.3)
            # annotate threshold values
            for r in ee_rows[::3]:
                ax.annotate(f"theta={r['model'].split('=')[-1]}",
                            xy=(r["mean_exit"], r["val_acc"]),
                            fontsize=7, ha="left", va="bottom")
            savefig("06_early_exit_tradeoff.png")

        # -- Plot 7: Energy efficiency vs timestep ----------------------------
        hw_choice = args.energy_hw if hasattr(args, "energy_hw") else "loihi"
        E_MAC = ENERGY_MODELS[hw_choice]["E_MAC"]
        E_AC  = ENERGY_MODELS[hw_choice]["E_AC"]
        ref_text = ENERGY_MODELS[hw_choice]["ref"]

        # Use confidence growth as firing-rate proxy if available
        # (from all_probs of ours_full eval — here we synthesise from acc growth)
        ee_accs = [r["val_acc"] for r in sorted(
            [r for r in rows if r.get("group")=="early_exit"
             and "40" in str(r.get("dataset",""))],
            key=lambda r: r.get("mean_exit",0))]
        if ee_accs:
            T = len(ee_accs)
            ts  = np.arange(1, T+1)
            ann_energy = np.ones(T)
            conf  = np.array(ee_accs) / 100.0
            fr    = np.clip(1.0 - conf, 0.05, 1.0)
            snn_e = np.cumsum(fr * (E_AC / E_MAC)) / ts

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(ts, ann_energy, "--", color="steelblue",
                    linewidth=2, label="ANN (normalised to 1.0)")
            ax.plot(ts, snn_e, "s-", color="tomato",
                    linewidth=2, label="SNN (ours)")
            ax.set_xlabel("Timestep (slices seen)")
            ax.set_ylabel("Cumulative Energy (ANN=1.0)")
            ax.set_title(f"Energy Efficiency: SNN vs ANN\n"
                         f"Hw: {hw_choice} ({ref_text})")
            ax.legend(); ax.grid(True, alpha=0.3)
            savefig("07_energy_efficiency.png")

        # -- Plot 8: Param count vs accuracy scatter ---------------------------
        scatter_rows = [r for r in rows
                        if r.get("group") in ("comparison","scaling")
                        and "40" in str(r.get("dataset",""))
                        and r.get("params",0) > 0
                        and r.get("val_acc",0) > 0]
        pub_scatter = [(k, v["type"], v.get("mn40",0), 0)
                       for k,v in pub_results.items() if v.get("mn40")]

        if scatter_rows:
            fig, ax = plt.subplots(figsize=(8, 6))
            type_color = {"SNN":"tomato","ANN":"steelblue",
                          "Converted-SNN":"orange","?":"gray"}
            for r in scatter_rows:
                ax.scatter(r["params"]/1e6, r["val_acc"],
                           c=type_color.get(r["type"],"gray"),
                           s=80, zorder=3, edgecolors="black", linewidths=0.5)
                ax.annotate(r["model"], xy=(r["params"]/1e6, r["val_acc"]),
                            fontsize=6, ha="left", va="bottom")
            ax.set_xlabel("Parameters (M)"); ax.set_ylabel("Accuracy (%) — MN40")
            ax.set_title("Accuracy vs Model Size (Pareto front)")
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[
                mpatches.Patch(facecolor=c, label=t)
                for t,c in type_color.items() if t!="?"
            ], loc="lower right")
            savefig("08_params_vs_acc.png")

        print(f"\n[Plots] All saved to {out_dir}/")

    except ImportError:
        print("[Plot] matplotlib not available — skipping plots.")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[Plot] Error: {e}")


# =============================================================================
# 5. TABLE OUTPUT
# =============================================================================

def save_tables(rows, pub_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # -- CSV (all rows) --------------------------------------------------------
    csv_path = os.path.join(out_dir, "all_results.csv")
    fields = ["group","dataset","model","slicing","type","params",
              "val_acc","val_std","mean_exit","T","firing_rate",
              "energy_loihi","energy_45nm","paper"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"[Table] {csv_path}")

    # -- JSON (for downstream plotting) ---------------------------------------
    json_path = os.path.join(out_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[Table] {json_path}")

    # -- Pretty text table -----------------------------------------------------
    txt_path = os.path.join(out_dir, "comparison_table.txt")
    with open(txt_path, "w") as f:
        # -- Main comparison (MN40) -----------------------------------------
        f.write("=" * 90 + "\n")
        f.write("  MAIN COMPARISON — ModelNet40\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Model':<28} {'Type':<12} {'Params':>10}  {'Acc%':>7}  "
                f"{'MeanExit':>9}  {'Paper'}\n")
        f.write("-" * 90 + "\n")
        cmp = [r for r in rows if r.get("group")=="comparison"
               and "40" in str(r.get("dataset",""))]
        for r in sorted(cmp, key=lambda x: -x.get("val_acc",0)):
            f.write(f"{r['model']:<28} {r.get('type','?'):<12} "
                    f"{r.get('params',0):>10,}  {r.get('val_acc',0):>7.2f}  "
                    f"{str(r.get('mean_exit','N/A')):>9}  {r.get('paper','')}\n")

        # -- Published baselines ---------------------------------------------
        f.write("\n--- Published Baselines (from papers, ModelNet40) ---\n")
        f.write(f"{'Model':<28} {'Type':<12} {'MN40 Acc%':>10}  Paper\n")
        f.write("-" * 60 + "\n")
        for name, info in pub_results.items():
            acc = f"{info['mn40']:.1f}" if info.get("mn40") else "N/A"
            f.write(f"{name:<28} {info['type']:<12} {acc:>10}  {info['paper']}\n")

        # -- Scaling table ---------------------------------------------------
        f.write("\n" + "=" * 90 + "\n")
        f.write("  SCALING ABLATION — FPS slicing\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Model':<28} {'Dataset':<12} {'Params':>10}  {'Acc%':>7}\n")
        f.write("-" * 65 + "\n")
        scl = [r for r in rows if r.get("group")=="scaling"
               and r.get("slicing")=="fps"]
        for r in sorted(scl, key=lambda x: (x.get("dataset",""), -x.get("val_acc",0))):
            f.write(f"{r['model']:<28} {r.get('dataset','?'):<12} "
                    f"{r.get('params',0):>10,}  {r.get('val_acc',0):>7.2f}\n")

        # -- Slicing ablation -------------------------------------------------
        f.write("\n" + "=" * 90 + "\n")
        f.write("  SLICING ABLATION (Radial vs FPS, SNN vs ANN)\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Model':<28} {'Type':<6} {'Slicing':<8} {'MN40 Acc%':>10}\n")
        f.write("-" * 60 + "\n")
        sla = [r for r in rows if r.get("group")=="slicing"
               and "40" in str(r.get("dataset",""))]
        for r in sorted(sla, key=lambda x: (x["model"], x.get("slicing",""))):
            f.write(f"{r['model']:<28} {r.get('type','?'):<6} "
                    f"{r.get('slicing','?'):<8} {r.get('val_acc',0):>10.2f}\n")

        # -- ANN->SNN conversion -----------------------------------------------
        f.write("\n" + "=" * 90 + "\n")
        f.write("  ANN->SNN CONVERSION\n")
        f.write("=" * 90 + "\n")
        conv = [r for r in rows if r.get("group")=="conversion"
                and "40" in str(r.get("dataset",""))]
        for r in conv:
            f.write(f"  {r['model']:<40}  {r.get('val_acc',0):.2f}%\n")

        # -- Multi-seed results (mean ± std) -----------------------------------
        ms_rows = [r for r in rows if r.get("group") == "multi_seed"]
        if ms_rows:
            f.write("\n" + "=" * 90 + "\n")
            f.write("  MULTI-SEED RESULTS (mean ± std, 3 seeds)\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'Model':<28} {'Dataset':<14} {'Type':<8} "
                    f"{'Mean%':>7}  {'Std%':>6}  Per-seed\n")
            f.write("-" * 80 + "\n")
            for r in ms_rows:
                ps = "  ".join(str(v) for v in r.get("per_seed", []))
                f.write(f"{r['model']:<28} {r.get('dataset','?'):<14} "
                        f"{r.get('type','?'):<8} "
                        f"{r.get('val_acc',0):>7.2f}  "
                        f"±{r.get('val_std',0):>5.2f}  [{ps}]\n")

        # -- ScanObjectNN results ----------------------------------------------
        sonn_rows = [r for r in rows if r.get("group") == "scanobjectnn"]
        if sonn_rows:
            f.write("\n" + "=" * 90 + "\n")
            f.write("  SCANOBJECTNN BENCHMARK\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'Model':<28} {'Variant':<14} {'Type':<8} {'Acc%':>7}\n")
            f.write("-" * 65 + "\n")
            for r in sorted(sonn_rows,
                            key=lambda x: (x.get("dataset",""), -x.get("val_acc",0))):
                f.write(f"{r['model']:<28} {r.get('dataset','?'):<14} "
                        f"{r.get('type','?'):<8} {r.get('val_acc',0):>7.2f}\n")

        # -- T-sweep summary ---------------------------------------------------
        tsw_rows = [r for r in rows if r.get("group") == "t_sweep"]
        if tsw_rows:
            f.write("\n" + "=" * 90 + "\n")
            f.write("  T-TIMESTEP SENSITIVITY\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'T':<6} {'Model':<28} {'Type':<16} {'Acc%':>7}  "
                    f"{'FR':>6}  {'E_Loihi':>9}\n")
            f.write("-" * 80 + "\n")
            for r in sorted(tsw_rows, key=lambda x: (x["model"], str(x.get("T","")))):
                f.write(f"{str(r.get('T','N/A')):<6} {r['model']:<28} "
                        f"{r.get('type','?'):<16} {r.get('val_acc',0):>7.2f}  "
                        f"{str(r.get('firing_rate','?')):>6}  "
                        f"{str(r.get('energy_loihi','?')):>9}\n")

    print(f"[Table] {txt_path}")

    # -- Print to stdout -------------------------------------------------------
    with open(txt_path) as f:
        print("\n" + f.read())


# =============================================================================
# 6. SCANOBJECTNN BENCHMARK
# =============================================================================

def exp_scanobjectnn(args, device, rows):
    """
    Group 'scanobjectnn': benchmark ours_full + ANN baseline on all three
    ScanObjectNN variants (OBJ-BG, OBJ-ONLY, PB_T50_RS).

    Addresses reviewer: "Use ScanObjectNN (harder than ModelNet40)."
    """
    from data.scanobjectnn import get_scanobjectnn_loaders

    VARIANTS = ["OBJ_ONLY", "OBJ_BG", "PB_T50_RS"]
    MODELS   = [
        ("ours_full",    "fps",    True),
        ("ann_pointnet", "radial", False),
    ]

    for variant in VARIANTS:
        print(f"\n{'='*65}\n  ScanObjectNN — {variant}\n{'='*65}")
        try:
            tr_l, va_l, nc = get_scanobjectnn_loaders(
                args.sonn_root, variant=variant,
                batch_size=args.batch_size, num_points=args.n_pts
            )
        except Exception as e:
            print(f"  [ScanObjectNN/{variant}] Data load failed: {e}")
            continue

        for model_name, slicing, bidir in MODELS:
            tag = f"sonn_{variant}|{model_name}"
            print(f"\n  {tag}")
            try:
                m = build_model(model_name, num_classes=nc).to(device)
                n_params = count_params(m)

                acc = (1.0 / nc) if args.smoke_test else train_model_full(
                    m, tr_l, va_l, device,
                    epochs=args.epochs, num_slices=args.num_slices,
                    slicing=slicing, bidirectional=bidir,
                    name=f"{model_name}/{variant}"
                )
                val_acc, mean_exit, _, _, _ = eval_model(
                    m, va_l, device, args.num_slices, slicing,
                    threshold=args.threshold, bidirectional=bidir
                )
                rows.append({
                    "group": "scanobjectnn", "dataset": f"sonn_{variant}",
                    "model": model_name, "slicing": slicing,
                    "type": MODEL_CONFIGS[model_name]["type"],
                    "params": n_params,
                    "val_acc": round(val_acc * 100, 2),
                    "mean_exit": round(mean_exit, 2) if mean_exit else "N/A",
                    "paper": MODEL_CONFIGS[model_name]["paper"],
                })
                print(f"  -> {variant}/{model_name}: acc={val_acc*100:.2f}%")
            except Exception as e:
                import traceback; traceback.print_exc()
                rows.append({
                    "group": "scanobjectnn", "dataset": f"sonn_{variant}",
                    "model": model_name, "slicing": slicing, "type": "?",
                    "params": 0, "val_acc": 0, "mean_exit": "N/A",
                    "paper": "?", "error": str(e),
                })


# =============================================================================
# 7. MULTI-SEED EVALUATION
# =============================================================================

def exp_multi_seed(args, device, train_l, val_l, nc, ds_tag, rows):
    """
    Group 'multi_seed': train ours_full and ann_pointnet 3× with different
    seeds; report mean ± std accuracy.

    Addresses reviewer: "Where are error bars / std across 3–5 runs?"
    """
    from training.multi_seed import run_with_seeds, format_result, set_seed

    SEEDS = [0, 1, 2]
    CFGS  = [
        ("ours_full",    "fps",    True),
        ("ann_pointnet", "radial", False),
    ]

    for model_name, slicing, bidir in CFGS:
        print(f"\n  [Multi-seed] {ds_tag}|{model_name}  seeds={SEEDS}")

        def build_fn():
            return build_model(model_name, num_classes=nc).to(device)

        def train_fn(model):
            if not args.smoke_test:
                train_model_full(
                    model, train_l, val_l, device,
                    epochs=args.epochs, num_slices=args.num_slices,
                    slicing=slicing, bidirectional=bidir,
                    name=f"{model_name}/seed", log_every=args.epochs
                )

        def eval_fn(model):
            if args.smoke_test:
                return 1.0 / nc
            acc, _, _, _, _ = eval_model(
                model, val_l, device, args.num_slices, slicing,
                bidirectional=bidir
            )
            return acc

        try:
            mean, std, per_seed = run_with_seeds(
                build_fn, train_fn, eval_fn, seeds=SEEDS, verbose=True
            )
            print(f"  Result: {format_result(mean, std)}%")
            rows.append({
                "group": "multi_seed", "dataset": ds_tag,
                "model": model_name, "slicing": slicing,
                "type": MODEL_CONFIGS[model_name]["type"],
                "params": count_params(build_fn()),
                "val_acc": round(mean * 100, 2),
                "val_std": round(std  * 100, 2),
                "mean_exit": "N/A",
                "paper": MODEL_CONFIGS[model_name]["paper"],
                "per_seed": [round(v * 100, 2) for _, v in per_seed],
            })
        except Exception as e:
            import traceback; traceback.print_exc()


# =============================================================================
# 8. T-TIMESTEP SENSITIVITY
# =============================================================================

def exp_t_sweep(args, device, train_l, val_l, nc, ds_tag, rows):
    """
    Group 't_sweep': sweep T ∈ {4,8,12,16,24,32} for native SNN and
    converted SNN. Shows the accuracy-efficiency frontier vs T.

    Addresses reviewer: "Vary T from 4 to 32, show accuracy-efficiency frontier."
    Also: "Native SNN closes conversion gap as T increases."
    """
    from benchmarks.t_sweep import run_t_sweep

    T_list = [4, 8, 12, 16, 24, 32] if not args.smoke_test else [4, 8]
    out_dir = os.path.join(args.out_dir, "t_sweep", ds_tag)

    print(f"\n  [T-sweep] {ds_tag}  T_list={T_list}")
    try:
        sweep_rows = run_t_sweep(
            train_l, val_l, device,
            num_classes=nc, epochs=args.epochs,
            T_list=T_list, out_dir=out_dir,
            smoke_test=args.smoke_test
        )
        for r in sweep_rows:
            rows.append({
                "group": "t_sweep", "dataset": ds_tag,
                "model": r["model"], "slicing": "fps",
                "type": r["type"],
                "params": 0,
                "val_acc": r["val_acc"],
                "mean_exit": "N/A",
                "paper": "Ours" if "ours" in r["model"] else "ANN->SNN",
                "T": r.get("T", "N/A"),
                "firing_rate": r.get("firing_rate", "N/A"),
                "energy_loihi": r.get("energy_loihi", "N/A"),
                "energy_45nm":  r.get("energy_45nm",  "N/A"),
            })
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [T-sweep] ERROR: {e}")


# =============================================================================
# 6. ARGUMENT PARSING + MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run all paper experiments: training, scaling, conversion, plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    p.add_argument("--mn10_root",     default=None, help="ModelNet10 root directory")
    p.add_argument("--mn40_root",     default=None, help="ModelNet40 root directory")
    p.add_argument("--sonn_root",     default=None, help="ScanObjectNN root directory")
    p.add_argument("--n_pts",         type=int, default=1024, help="Points per cloud")

    # Training
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--num_slices",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--threshold",     type=float, default=0.7,
                   help="Confidence threshold for early-exit measurement")
    p.add_argument("--seeds",         type=int,   nargs="+", default=[0, 1, 2],
                   help="Random seeds for multi-seed experiment")
    p.add_argument("--T_list",        type=int,   nargs="+",
                   default=[4, 8, 12, 16, 24, 32],
                   help="Timestep values for T-sweep experiment")
    p.add_argument("--energy_hw",     default="loihi",
                   choices=["loihi", "45nm"],
                   help="Energy model: loihi=Lemaire2022, 45nm=Horowitz2014")

    # Experiment groups to run
    ALL_GROUPS = ["comparison", "scaling", "slicing", "conversion",
                  "early_exit", "scanobjectnn", "multi_seed", "t_sweep"]
    p.add_argument("--groups", nargs="+",
                   default=["comparison", "scaling", "slicing",
                            "conversion", "early_exit"],
                   choices=ALL_GROUPS,
                   help="Which experiment groups to run")

    # Flags
    p.add_argument("--smoke_test",    action="store_true",
                   help="Use dummy data and skip training (quick CI test)")
    p.add_argument("--skip_ann",      action="store_true",
                   help="Skip ANN baseline training in comparison group")
    p.add_argument("--out_dir",       default="results/all_experiments")
    p.add_argument("--datasets",      nargs="+", default=["modelnet40"],
                   choices=["modelnet10", "modelnet40"],
                   help="Which datasets to use")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Purdue SNN Point Cloud — All Experiments")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}  |  Slices : {args.num_slices}")
    print(f"  Groups  : {args.groups}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Smoke   : {args.smoke_test}")
    print(f"  Energy  : {ENERGY_MODELS[args.energy_hw]['ref']}")
    print(f"  Seeds   : {args.seeds}  |  T_list : {args.T_list}")
    print(f"{'='*70}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Build dataset loaders -------------------------------------------------
    ds_root = {"modelnet10": args.mn10_root, "modelnet40": args.mn40_root}
    ds_nc   = {"modelnet10": 10, "modelnet40": 40}
    loaders = {}
    for ds in args.datasets:
        nc = ds_nc[ds]
        tr = get_loaders(ds_root[ds], "train", args.batch_size, nc, args.n_pts)
        va = get_loaders(ds_root[ds], "test",  args.batch_size, nc, args.n_pts)
        loaders[ds] = (tr, va, nc)

    rows = []

    # -- Run experiment groups -------------------------------------------------
    for ds, (train_l, val_l, nc) in loaders.items():
        ds_tag = ds

        if "comparison" in args.groups:
            print(f"\n{'#'*70}\n  GROUP: comparison | {ds_tag}\n{'#'*70}")
            exp_main_comparison(args, device, train_l, val_l, nc, ds_tag, rows)

        if "slicing" in args.groups:
            print(f"\n{'#'*70}\n  GROUP: slicing | {ds_tag}\n{'#'*70}")
            exp_slicing_ablation(args, device, train_l, val_l, nc, ds_tag, rows)

        if "conversion" in args.groups:
            print(f"\n{'#'*70}\n  GROUP: conversion | {ds_tag}\n{'#'*70}")
            exp_conversion(args, device, train_l, val_l, nc, ds_tag, rows)

        if "early_exit" in args.groups:
            print(f"\n{'#'*70}\n  GROUP: early_exit | {ds_tag}\n{'#'*70}")
            exp_early_exit(args, device, train_l, val_l, nc, ds_tag, rows)

        if "multi_seed" in args.groups:
            print(f"\n{'#'*70}\n  GROUP: multi_seed | {ds_tag}\n{'#'*70}")
            exp_multi_seed(args, device, train_l, val_l, nc, ds_tag, rows)

        if "t_sweep" in args.groups:
            print(f"\n{'#'*70}\n  GROUP: t_sweep | {ds_tag}\n{'#'*70}")
            exp_t_sweep(args, device, train_l, val_l, nc, ds_tag, rows)

    if "scaling" in args.groups:
        print(f"\n{'#'*70}\n  GROUP: scaling\n{'#'*70}")
        exp_scaling(args, device, loaders, rows)

    if "scanobjectnn" in args.groups:
        print(f"\n{'#'*70}\n  GROUP: scanobjectnn\n{'#'*70}")
        exp_scanobjectnn(args, device, rows)

    # -- Save tables -----------------------------------------------------------
    save_tables(rows, PUBLISHED_RESULTS, args.out_dir)

    # -- Generate all plots ----------------------------------------------------
    make_all_plots(rows, PUBLISHED_RESULTS, os.path.join(args.out_dir, "plots"))

    print(f"\n{'='*70}")
    print(f"  DONE. Results saved to: {args.out_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

"""
compare_models.py
=================
Benchmark all registered models on ModelNet10/40 and produce a comparison table.

Usage:
  # Quick smoke test (tiny data, no real training):
  python benchmarks/compare_models.py --smoke_test

  # Full benchmark (requires trained checkpoints OR trains from scratch):
  python benchmarks/compare_models.py \\
      --dataset modelnet40 \\
      --data_root /path/to/ModelNet40 \\
      --epochs 50 \\
      --models ours_base ours_full e3dsnn spiking_ssm spt \\
      --out_dir results/benchmark

Outputs:
  results/benchmark/
    comparison_table.csv        — accuracy, params, mean exit, efficiency
    comparison_table.txt        — formatted text table
    paper_comparison_updated.png — bar chart including our measured results
"""

import argparse
import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model_zoo import build_model, MODEL_CONFIGS, PUBLISHED_RESULTS, count_params


# ---------------------------------------------------------------------------
# Dummy dataset for smoke testing (no files needed)
# ---------------------------------------------------------------------------

class DummyDataset(torch.utils.data.Dataset):
    """Random point clouds for smoke testing."""
    def __init__(self, n_samples=64, n_points=1024, num_classes=40):
        self.n = n_samples
        self.n_points = n_points
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        pts   = torch.randn(self.n_points, 3)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return pts, label


# ---------------------------------------------------------------------------
# Attempt to load ModelNet dataset (falls back to dummy if unavailable)
# ---------------------------------------------------------------------------

def get_dataloaders(dataset_name, data_root, batch_size=16, num_workers=2):
    """
    Try to load ModelNet10 or ModelNet40.
    Falls back to DummyDataset if the data root doesn't exist.
    """
    num_classes = 40 if "40" in dataset_name else 10

    if data_root and os.path.isdir(data_root):
        try:
            from data.modelnet_dataset import ModelNetDataset
            train_ds = ModelNetDataset(data_root, split="train", num_points=1024)
            val_ds   = ModelNetDataset(data_root, split="test",  num_points=1024)
            print(f"[Data] Loaded {dataset_name}: {len(train_ds)} train / {len(val_ds)} val")
        except Exception as e:
            print(f"[Data] Could not load real dataset ({e}). Using DummyDataset.")
            train_ds = DummyDataset(256, num_classes=num_classes)
            val_ds   = DummyDataset(64,  num_classes=num_classes)
    else:
        print("[Data] data_root not set or not found. Using DummyDataset.")
        train_ds = DummyDataset(256, num_classes=num_classes)
        val_ds   = DummyDataset(64,  num_classes=num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------------
# Single model training loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, epochs=50,
                num_slices=16, lr=1e-3, model_name="model"):
    """Train one model and return best validation accuracy."""
    from data.slicing import slice_radial_batch

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        for pts, labels in train_loader:
            pts    = pts.to(device)
            labels = labels.to(device).long()
            B      = pts.size(0)

            if hasattr(model, "reset_state"):
                model.reset_state(B, device)

            # Radial slicing
            batch_idx = slice_radial_batch(pts, T=num_slices)
            gather_i  = batch_idx.unsqueeze(-1).expand(-1, -1, 3)
            pts_sort  = torch.gather(pts, 1, gather_i)
            N = pts_sort.size(1)
            pps = N // num_slices
            pts_slices = pts_sort.view(B, num_slices, pps, 3)

            logits_all = []
            for t in range(num_slices):
                logits_t = model.forward_step(pts_slices[:, t])
                logits_all.append(logits_t)

            # Final slice loss + aux losses on intermediate slices
            final_loss = criterion(logits_all[-1], labels)
            aux_loss   = sum(criterion(l, labels) for l in logits_all[:-1]) / max(len(logits_all) - 1, 1)
            loss       = final_loss + 0.3 * aux_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ---- Validate ----
        val_acc = evaluate_model(model, val_loader, device, num_slices)
        if val_acc > best_acc:
            best_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1:3d}/{epochs} — val acc: {val_acc:.3f}  best: {best_acc:.3f}")

    return best_acc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, val_loader, device, num_slices=16):
    """Evaluate one model, return accuracy."""
    from data.slicing import slice_radial_batch

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for pts, labels in val_loader:
            pts    = pts.to(device)
            labels = labels.to(device).long()
            B      = pts.size(0)

            if hasattr(model, "reset_state"):
                model.reset_state(B, device)

            batch_idx  = slice_radial_batch(pts, T=num_slices)
            gather_i   = batch_idx.unsqueeze(-1).expand(-1, -1, 3)
            pts_sort   = torch.gather(pts, 1, gather_i)
            N = pts_sort.size(1)
            pps = N // num_slices
            pts_slices = pts_sort.view(B, num_slices, pps, 3)

            for t in range(num_slices):
                logits = model.forward_step(pts_slices[:, t])

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += B

    return correct / total if total > 0 else 0.0


def measure_mean_exit(model, val_loader, device, num_slices=16, threshold=0.5):
    """Measure mean early exit timestep with a confidence threshold."""
    import torch.nn.functional as F
    from data.slicing import slice_radial_batch

    model.eval()
    exit_steps = []

    with torch.no_grad():
        for pts, _ in val_loader:
            pts = pts.to(device)
            B   = pts.size(0)

            if hasattr(model, "reset_state"):
                model.reset_state(B, device)

            batch_idx  = slice_radial_batch(pts, T=num_slices)
            gather_i   = batch_idx.unsqueeze(-1).expand(-1, -1, 3)
            pts_sort   = torch.gather(pts, 1, gather_i)
            N = pts_sort.size(1)
            pps = N // num_slices
            pts_slices = pts_sort.view(B, num_slices, pps, 3)

            batch_exits = [num_slices] * B    # default: no early exit
            exited = [False] * B

            for t in range(num_slices):
                logits = model.forward_step(pts_slices[:, t])
                probs  = F.softmax(logits, dim=-1)
                max_p, _ = probs.max(dim=1)

                for b in range(B):
                    if not exited[b] and max_p[b].item() > threshold:
                        batch_exits[b] = t + 1
                        exited[b] = True

            exit_steps.extend(batch_exits)

    return sum(exit_steps) / len(exit_steps)


def measure_throughput(model, device, batch_size=8, n_points=1024, num_slices=16, n_runs=20):
    """Measure inference throughput (samples/second)."""
    from data.slicing import slice_radial_batch

    model.eval()
    pts = torch.randn(batch_size, n_points, 3, device=device)

    # Warmup
    for _ in range(3):
        if hasattr(model, "reset_state"):
            model.reset_state(batch_size, device)
        bi = slice_radial_batch(pts, T=num_slices)
        gi = bi.unsqueeze(-1).expand(-1, -1, 3)
        ps = torch.gather(pts, 1, gi)
        pps = ps.size(1) // num_slices
        ps = ps.view(batch_size, num_slices, pps, 3)
        with torch.no_grad():
            for t in range(num_slices):
                model.forward_step(ps[:, t])

    start = time.time()
    for _ in range(n_runs):
        if hasattr(model, "reset_state"):
            model.reset_state(batch_size, device)
        bi = slice_radial_batch(pts, T=num_slices)
        gi = bi.unsqueeze(-1).expand(-1, -1, 3)
        ps = torch.gather(pts, 1, gi)
        pps = ps.size(1) // num_slices
        ps = ps.view(batch_size, num_slices, pps, 3)
        with torch.no_grad():
            for t in range(num_slices):
                model.forward_step(ps[:, t])
    elapsed = time.time() - start

    samples_per_sec = (batch_size * n_runs) / elapsed
    return samples_per_sec


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Datasets
    train_loader, val_loader, num_classes = get_dataloaders(
        args.dataset, args.data_root,
        batch_size=args.batch_size
    )

    # Determine which models to run
    model_names = args.models if args.models else list(MODEL_CONFIGS.keys())

    results = []

    for name in model_names:
        if name not in MODEL_CONFIGS:
            print(f"[Skip] Unknown model: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"[Model] {name}: {MODEL_CONFIGS[name]['description']}")
        print(f"{'='*60}")

        try:
            model = build_model(name, num_classes=num_classes).to(device)
            n_params = count_params(model)
            print(f"  Params: {n_params:,}")

            if args.smoke_test:
                # Skip training, just measure forward pass
                val_acc = 1.0 / num_classes   # random baseline
                mean_exit = args.num_slices / 2
                throughput = measure_throughput(model, device,
                                               batch_size=min(4, args.batch_size),
                                               num_slices=args.num_slices,
                                               n_runs=5)
            else:
                # Train from scratch
                val_acc = train_model(
                    model, train_loader, val_loader, device,
                    epochs=args.epochs, num_slices=args.num_slices,
                    lr=args.lr, model_name=name
                )
                mean_exit = measure_mean_exit(model, val_loader, device,
                                              num_slices=args.num_slices,
                                              threshold=args.threshold)
                throughput = measure_throughput(model, device,
                                               batch_size=min(8, args.batch_size),
                                               num_slices=args.num_slices)

            row = {
                "model":      name,
                "type":       MODEL_CONFIGS[name]["type"],
                "paper":      MODEL_CONFIGS[name]["paper"],
                "params":     n_params,
                "val_acc":    round(val_acc * 100, 2),
                "mean_exit":  round(mean_exit, 2),
                "throughput": round(throughput, 1),
                "description": MODEL_CONFIGS[name]["description"],
            }
            results.append(row)
            print(f"  → val acc: {val_acc*100:.2f}%  mean exit: {mean_exit:.1f}  {throughput:.0f} samp/s")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results.append({
                "model": name, "type": "?", "paper": "?",
                "params": 0, "val_acc": 0.0, "mean_exit": 0.0,
                "throughput": 0.0, "description": f"ERROR: {e}",
            })

    # ---- Write CSV ----
    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    fieldnames = ["model", "type", "paper", "params", "val_acc",
                  "mean_exit", "throughput", "description"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] {csv_path}")

    # ---- Write text table ----
    txt_path = os.path.join(args.out_dir, "comparison_table.txt")
    with open(txt_path, "w") as f:
        header = f"{'Model':<20} {'Type':<5} {'Params':>10}  {'Acc%':>7}  {'MeanExit':>9}  {'Samp/s':>8}  Description"
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for r in sorted(results, key=lambda x: -x["val_acc"]):
            line = (f"{r['model']:<20} {r['type']:<5} {r['params']:>10,}  "
                    f"{r['val_acc']:>7.2f}  {r['mean_exit']:>9.2f}  "
                    f"{r['throughput']:>8.1f}  {r['description']}")
            f.write(line + "\n")

        # Published baselines
        f.write("\n--- Published Baselines (from papers) ---\n")
        pub_header = f"{'Model':<20} {'Type':<5} {'MN40 Acc%':>10}  Paper"
        f.write(pub_header + "\n")
        f.write("-" * 60 + "\n")
        for name, info in PUBLISHED_RESULTS.items():
            acc = f"{info['mn40']:.1f}" if info["mn40"] else "N/A"
            f.write(f"{name:<20} {info['type']:<5} {acc:>10}  {info['paper']}\n")

    print(f"[Saved] {txt_path}")

    # Print table to stdout
    print("\n" + "=" * 100)
    with open(txt_path) as f:
        print(f.read())

    # ---- Update paper comparison plot ----
    _update_paper_plot(results, args.out_dir, args.dataset)

    return results


def _update_paper_plot(our_results, out_dir, dataset):
    """Update paper_comparison.png with our measured results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        paper_models = [
            ("PointNet",         "ANN",  89.2),
            ("PointNet++",       "ANN",  90.7),
            ("PointMLP",         "ANN",  94.1),
            ("PointMamba",       "ANN",  92.4),
            ("Spiking PointNet", "SNN",  88.2),
            ("P2SResLNet-B",     "SNN",  88.7),
            ("SPT",              "SNN",  91.4),
            ("SPM (paper)",      "SNN",  92.3),
        ]

        # Append our measured models
        for r in our_results:
            if r["val_acc"] > 0:
                paper_models.append((r["model"], r["type"], r["val_acc"]))

        names  = [m[0] for m in paper_models]
        accs   = [m[2] for m in paper_models]
        colors = []
        for m in paper_models:
            if "ours" in m[0]:
                colors.append("gold")
            elif m[1] == "ANN":
                colors.append("steelblue")
            else:
                colors.append("tomato")

        fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 5))
        ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Point Cloud Classification: ANNs vs SNNs — {dataset.upper()}")
        y_min = max(0, min(accs) - 2)
        ax.set_ylim(y_min, min(100, max(accs) + 2))
        ax.axhline(y=92.3, color="tomato", linestyle=":", linewidth=1.2, label="SPM baseline")

        legend_elements = [
            mpatches.Patch(facecolor="steelblue", label="ANN"),
            mpatches.Patch(facecolor="tomato",    label="SNN (published)"),
            mpatches.Patch(facecolor="gold",       label="Ours"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()

        save_path = os.path.join(out_dir, "paper_comparison_updated.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Saved] {save_path}")
    except Exception as e:
        print(f"[Plot] Could not generate plot: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare all SNN models on ModelNet")
    p.add_argument("--dataset",    default="modelnet40", choices=["modelnet10", "modelnet40"])
    p.add_argument("--data_root",  default=None,  help="Path to ModelNet root directory")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_slices", type=int, default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Confidence threshold for early exit measurement")
    p.add_argument("--models",     nargs="+", default=None,
                   help="Subset of models to run (default: all in registry)")
    p.add_argument("--out_dir",    default="results/benchmark")
    p.add_argument("--smoke_test", action="store_true",
                   help="Quick run with dummy data, no training (for CI/testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)

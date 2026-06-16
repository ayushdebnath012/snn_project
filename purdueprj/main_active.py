"""
main_active.py
==============
Entry point for Active Spiking Perception (ASP) training and evaluation.

Usage examples:

  # Train on ModelNet10 for 50 epochs (ablation):
  python main_active.py --dataset modelnet10 --epochs 50

  # Full ModelNet40 run (150 epochs for SOTA comparison):
  python main_active.py --dataset modelnet40 --epochs 150

  # Evaluate only (load checkpoint, run Pareto sweep):
  python main_active.py --eval_only --checkpoint /path/to/ckpt.pth

  # Multi-seed experiment (3 seeds, for paper results):
  python main_active.py --seeds 0 1 2 --epochs 150

Command-line flags:
  --dataset        modelnet10 | modelnet40 (default: modelnet10)
  --data_root      path to dataset root
  --epochs         number of training epochs (default: 50)
  --batch_size     batch size (default: 16)
  --num_slices     T, number of temporal slices (default: 16)
  --num_points     points per cloud (default: 1024)
  --d_ssp          SSP projection dimension (default: 64)
  --lam_aux        auxiliary loss weight (default: 0.3)
  --lam_exit       early confidence loss weight (default: 0.1)
  --lam_fr         firing rate loss weight (default: 0.05)
  --tau_0          initial Gumbel temperature (default: 1.0)
  --tau_min        minimum Gumbel temperature (default: 0.1)
  --anneal_rate    Gumbel annealing rate (default: 0.05)
  --threshold      inference exit threshold (default: 0.7)
  --seeds          random seeds for multi-seed runs (default: 0)
  --eval_only      skip training, load checkpoint and evaluate
  --checkpoint     path to load model checkpoint
  --save_dir       directory to save checkpoints (default: ./results/active/)
  --log_every      log frequency in batches (default: 20)
  --pareto_sweep   run full threshold sweep after training (default: True)
"""

import argparse
import os
import json
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from data.modelnet import ModelNetDataset
from training.optimizers import build_optimizer
from training.train_active import train_active_epoch, validate_active, sweep_threshold
from inference.active_inference import (
    pareto_curve,
    compare_orderings,
    visualise_attention,
)
from models.active_snn import ActiveSNN


# -----------------------------------------------------------------------
# Dataset config
# -----------------------------------------------------------------------

DATASET_CLASSES = {"modelnet10": 10, "modelnet40": 40}
DATASET_NAMES = {
    "modelnet10": ["bathtub", "bed", "chair", "desk", "dresser",
                   "monitor", "night_stand", "sofa", "table", "toilet"],
    "modelnet40": [f"class_{i}" for i in range(40)],   # placeholder
}
DEFAULT_ROOTS = {
    "modelnet10": "/content/drive/MyDrive/ModelNet10",
    "modelnet40": "/content/drive/MyDrive/ModelNet40",
}


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Learning rate schedule
# -----------------------------------------------------------------------

def build_lr_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )


def apply_profile(args, parser):
    """
    Expand a high-level run profile into concrete hyperparameters.
    The accuracy profile intentionally spends more compute: wider SPM,
    full-slice inference, no firing-rate penalty, and smoother CE targets.
    """
    if args.profile != "accuracy":
        return

    if args.model_type == parser.get_default("model_type"):
        args.model_type = "asp_spm"
    if args.point_dims == parser.get_default("point_dims"):
        args.point_dims = [256, 512, 1024]
    if args.d_ssp == parser.get_default("d_ssp"):
        args.d_ssp = 128
    if args.num_slices == parser.get_default("num_slices"):
        args.num_slices = 8
    if args.epochs == parser.get_default("epochs"):
        args.epochs = 300 if args.dataset == "modelnet40" else 200
    if args.lam_aux == parser.get_default("lam_aux"):
        args.lam_aux = 0.15
    if args.lam_exit == parser.get_default("lam_exit"):
        args.lam_exit = 0.0
    if args.lam_fr == parser.get_default("lam_fr"):
        args.lam_fr = 0.0
    if args.threshold == parser.get_default("threshold"):
        args.threshold = 1.1
    if args.label_smoothing == parser.get_default("label_smoothing"):
        args.label_smoothing = 0.1
    if args.knn_k == parser.get_default("knn_k"):
        args.knn_k = 20
    if args.aug_mode == parser.get_default("aug_mode"):
        args.aug_mode = "strong"
    if args.pooling == parser.get_default("pooling"):
        args.pooling = "meanmax"
    if args.logit_ensemble == parser.get_default("logit_ensemble"):
        args.logit_ensemble = 3
    args.fixed_lif = True


# -----------------------------------------------------------------------
# Single-seed training run
# -----------------------------------------------------------------------

def run_single_seed(args, seed: int) -> dict:
    """
    Run one full training + evaluation with a given seed.

    Returns:
        final_metrics : dict with best_val_acc, final energy ratio, etc.
    """
    print(f"\n{'='*60}")
    print(f"  SEED {seed}  |  Dataset: {args.dataset.upper()}")
    print(f"{'='*60}\n")

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = DATASET_CLASSES[args.dataset]
    data_root   = args.data_root or DEFAULT_ROOTS[args.dataset]
    save_dir    = os.path.join(args.save_dir, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # --- Datasets ---
    train_dataset = ModelNetDataset(
        root=data_root, split="train", num_points=args.num_points,
        aug_mode=args.aug_mode,
    )
    val_dataset = ModelNetDataset(
        root=data_root, split="test", num_points=args.num_points,
        aug_mode="none",
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # --- Model ---
    model_type = getattr(args, "model_type", "active")
    point_dims = list(args.point_dims)
    if model_type == "active":
        model = ActiveSNN(
            point_dims=point_dims,
            temporal_dim=point_dims[-1],
            num_classes=num_classes,
            knn_k=args.knn_k,
            d_ssp=args.d_ssp,
        ).to(device)
    elif model_type == "asp_spm":
        from models.model_zoo import build_model
        model = build_model(model_type, num_classes=num_classes,
                            point_dims=tuple(point_dims),
                            d_ssp=args.d_ssp,
                            d_state=args.d_state,
                            n_smb_layers=args.n_smb_layers,
                            knn_k=args.knn_k,
                            learnable_lif=not args.fixed_lif,
                            pooling=args.pooling).to(device)
    elif model_type == "asp_spn":
        from models.model_zoo import build_model
        model = build_model(model_type, num_classes=num_classes,
                            point_dims=tuple(point_dims),
                            d_ssp=args.d_ssp,
                            knn_k=args.knn_k,
                            learnable_lif=not args.fixed_lif).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    params = model.param_count()
    print(f"Parameters: backbone={params['backbone']:,}  "
          f"temporal={params.get('temporal', 0):,}  "
          f"ssp={params['ssp']:,}  "
          f"total={params['total']:,}\n")

    # --- Optimizer + scheduler ---
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = build_lr_scheduler(optimizer, args.epochs)

    # --- Load checkpoint if eval_only or warm-start ---
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        if not args.eval_only:
            start_epoch = ckpt.get("epoch", 0) + 1
            optimizer.load_state_dict(ckpt.get("optimizer_state", {}))

    if args.eval_only:
        print("=== Evaluation Only Mode ===")
        metrics = validate_active(model, val_loader, device,
                                  num_slices=args.num_slices,
                                  threshold=args.threshold,
                                  logit_ensemble=args.logit_ensemble)
        print(f"Acc={metrics['acc']:.4f}  MeanExit={metrics['mean_exit']:.2f}  "
              f"EnergyRatio={metrics['energy_ratio']:.4f}  "
              f"Savings={1/max(metrics['energy_ratio'],1e-9):.1f}×")
        return metrics

    # --- Training loop ---
    history = []
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch}/{args.epochs-1} ---")

        # Train
        train_metrics = train_active_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch,
            num_slices=args.num_slices,
            lam_aux=args.lam_aux,
            lam_exit=args.lam_exit,
            lam_fr=args.lam_fr,
            label_smoothing=args.label_smoothing,
            tau_0=args.tau_0,
            tau_min=args.tau_min,
            anneal_rate=args.anneal_rate,
            verbose_every=args.log_every,
        )

        scheduler.step()

        # Validate (every 5 epochs to save time)
        val_metrics = {}
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = validate_active(
                model, val_loader, device,
                num_slices=args.num_slices, threshold=args.threshold,
                logit_ensemble=args.logit_ensemble,
            )
            val_acc = val_metrics["acc"]
            print(
                f"[Val] Acc={val_acc:.4f}  "
                f"MeanExit={val_metrics['mean_exit']:.2f}/{args.num_slices}  "
                f"FR={val_metrics['mean_fr']:.3f}  "
                f"EnergyRatio={val_metrics['energy_ratio']:.4f}  "
                f"Savings={val_metrics.get('savings', 0):.1f}×"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                }, best_ckpt_path)
                print(f"  *** New best: {val_acc:.4f} — saved to {best_ckpt_path}")

        record = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(record)

        # Save history every epoch
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # --- Final evaluation on best checkpoint ---
    print(f"\n=== Final Evaluation (best checkpoint, seed={seed}) ===")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    final_metrics = validate_active(
        model, val_loader, device,
        num_slices=args.num_slices, threshold=args.threshold,
        logit_ensemble=args.logit_ensemble,
    )
    print(
        f"Best Val Acc: {best_val_acc:.4f}  "
        f"Final Acc: {final_metrics['acc']:.4f}  "
        f"MeanExit: {final_metrics['mean_exit']:.2f}  "
        f"Savings: {final_metrics.get('savings', 0):.1f}×"
    )

    # --- Pareto sweep ---
    if args.pareto_sweep:
        print("\n=== Pareto Threshold Sweep ===")
        from training.train_active import prepare_fps_slices_and_geo

        # Use val_dataset directly for per-sample inference
        curve = pareto_curve(
            model, val_dataset, device,
            num_slices=args.num_slices,
            prepare_fn=prepare_fps_slices_and_geo,
        )
        with open(os.path.join(save_dir, "pareto_curve.json"), "w") as f:
            json.dump(curve, f, indent=2)
        print(f"Pareto curve saved to {save_dir}/pareto_curve.json")

    # --- Ordering comparison ---
    print("\n=== Ordering Strategy Comparison ===")
    from training.train_active import prepare_fps_slices_and_geo
    ordering_results = compare_orderings(
        model, val_dataset, device,
        num_slices=args.num_slices, threshold=args.threshold,
        n_samples=200, prepare_fn=prepare_fps_slices_and_geo,
    )
    with open(os.path.join(save_dir, "ordering_comparison.json"), "w") as f:
        json.dump(ordering_results, f, indent=2)

    # --- SSP attention maps ---
    print("\n=== SSP Attention Maps ===")
    class_names = DATASET_NAMES.get(args.dataset, [])
    if class_names:
        from training.train_active import prepare_fps_slices_and_geo
        attn = visualise_attention(
            model, val_dataset, device,
            class_names=class_names,
            num_slices=args.num_slices,
            n_samples_per_class=10,
            prepare_fn=prepare_fps_slices_and_geo,
        )
        # Convert numpy arrays for JSON serialisation
        attn_json = {k: v.tolist() for k, v in attn.items()}
        with open(os.path.join(save_dir, "ssp_attention.json"), "w") as f:
            json.dump(attn_json, f, indent=2)
        print(f"SSP attention maps saved to {save_dir}/ssp_attention.json")

    return {
        "seed": seed,
        "best_val_acc": best_val_acc,
        "final_acc": final_metrics["acc"],
        "mean_exit": final_metrics["mean_exit"],
        "energy_ratio": final_metrics["energy_ratio"],
        "savings": final_metrics.get("savings", 0),
    }


# -----------------------------------------------------------------------
# Multi-seed aggregation
# -----------------------------------------------------------------------

def run_multi_seed(args):
    """Run args.seeds seeds and aggregate results."""
    all_results = []
    for seed in args.seeds:
        r = run_single_seed(args, seed)
        all_results.append(r)

    accs  = [r["best_val_acc"] for r in all_results]
    saves = [r["savings"]      for r in all_results]
    exits = [r["mean_exit"]    for r in all_results]

    print("\n" + "=" * 60)
    print("  MULTI-SEED SUMMARY")
    print("=" * 60)
    print(f"  Best Val Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Savings:       {np.mean(saves):.1f}× ± {np.std(saves):.1f}×")
    print(f"  Mean Exit:     {np.mean(exits):.2f} ± {np.std(exits):.2f}")

    summary_path = os.path.join(args.save_dir, "multi_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "results": all_results,
            "acc_mean":  float(np.mean(accs)),
            "acc_std":   float(np.std(accs)),
            "saves_mean": float(np.mean(saves)),
            "saves_std":  float(np.std(saves)),
            "exit_mean": float(np.mean(exits)),
            "exit_std":  float(np.std(exits)),
        }, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Active Spiking Perception")

    # Dataset
    parser.add_argument("--dataset",    type=str, default="modelnet10",
                        choices=["modelnet10", "modelnet40"])
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--num_points", type=int, default=1024)

    # Architecture
    parser.add_argument("--model_type", type=str, default="active",
                        choices=["active", "asp_spm", "asp_spn"],
                        help=(
                            "active   — original ActiveSNN (LocalKNNBackbone + TemporalSNN + SSP)  "
                            "asp_spm  — ASP plug-in on SPMModel (HDE + Spiking Mamba Blocks)  "
                            "asp_spn  — ASP plug-in on PointNetSNN (SpikingPointNet [8] proxy)"
                        ))
    parser.add_argument("--profile", type=str, default="balanced",
                        choices=["balanced", "accuracy"],
                        help="balanced keeps the energy-aware defaults; accuracy spends more compute")
    parser.add_argument("--d_ssp",      type=int, default=64)
    parser.add_argument("--num_slices", type=int, default=16)
    parser.add_argument("--point_dims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--d_state",    type=int, default=16)
    parser.add_argument("--n_smb_layers", type=int, default=2)
    parser.add_argument("--knn_k",      type=int, default=16)
    parser.add_argument("--fixed_lif",  action="store_true",
                        help="Use fixed BN-LIF in SPM-style backbones")
    parser.add_argument("--pooling",    type=str, default="mean",
                        choices=["mean", "max", "meanmax"],
                        help="Per-slice point pooling for SPM-style backbones")

    # Training
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lam_aux",    type=float, default=0.3)
    parser.add_argument("--lam_exit",   type=float, default=0.1)
    parser.add_argument("--lam_fr",     type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--aug_mode", type=str, default="baseline",
                        choices=["baseline", "strong", "elastic", "none"])

    # Gumbel annealing
    parser.add_argument("--tau_0",       type=float, default=1.0)
    parser.add_argument("--tau_min",     type=float, default=0.1)
    parser.add_argument("--anneal_rate", type=float, default=0.05)

    # Inference
    parser.add_argument("--threshold",  type=float, default=0.7)
    parser.add_argument("--logit_ensemble", type=int, default=1,
                        help="Average the last K ASP logits during validation")
    parser.add_argument("--pareto_sweep", action="store_true", default=True)

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    # Checkpointing
    parser.add_argument("--save_dir",   type=str, default="./results/active/")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_only",  action="store_true")

    # Logging
    parser.add_argument("--log_every",  type=int, default=20)

    args = parser.parse_args()
    apply_profile(args, parser)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n=== Active Spiking Perception (ASP) ===")
    print(f"  Profile:    {args.profile}")
    print(f"  Model:      {args.model_type}")
    print(f"  Dataset:    {args.dataset.upper()}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  point_dims: {tuple(args.point_dims)}")
    print(f"  pooling:    {args.pooling}")
    print(f"  aug:        {args.aug_mode}")
    print(f"  T slices:   {args.num_slices}")
    print(f"  d_ssp:      {args.d_ssp}")
    print(f"  λ_aux:      {args.lam_aux}")
    print(f"  λ_exit:     {args.lam_exit}")
    print(f"  λ_fr:       {args.lam_fr}")
    print(f"  τ_0→τ_min:  {args.tau_0}→{args.tau_min}  (rate={args.anneal_rate})")
    print(f"  smoothing:  {args.label_smoothing}")
    print(f"  threshold:  {args.threshold}\n")
    print(f"  logit ens:  {args.logit_ensemble}\n")

    if len(args.seeds) == 1:
        run_single_seed(args, args.seeds[0])
    else:
        run_multi_seed(args)


if __name__ == "__main__":
    main()

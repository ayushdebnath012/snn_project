"""
Train ASP-SNN on ShapeNetPart part segmentation.

Usage:
    python train_shapenet.py [--config configs/shapenet_seg.yaml] [--resume ckpt.pt]
    python train_shapenet.py --set epochs=30 batch_size=16 grad_accum_steps=2
"""

import glob
import json
import math
import os
import tempfile
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from config import base_argparser, load_config, parse_overrides, set_seed
from datasets.shapenetpart import (
    CATEGORY_NAMES,
    CATEGORY_TO_PARTS,
    NUM_CATEGORIES,
    NUM_PARTS,
    ShapeNetPartDataset,
)
from models.asp_segmentor import ASPSegmentor


PART_VALIDITY = torch.zeros(NUM_CATEGORIES, NUM_PARTS, dtype=torch.bool)
for _cat, _parts in CATEGORY_TO_PARTS.items():
    PART_VALIDITY[_cat, _parts] = True


# ─────────────────────────────────────────────────────────────────────────────
#  GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_info():
    if not torch.cuda.is_available():
        print("[GPU] No CUDA device found — running on CPU")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        bf16 = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        print(
            f"  GPU {i}: {p.name}  "
            f"{p.total_memory / 1e9:.1f} GB  "
            f"cc={p.major}.{p.minor}  amp={bf16}"
        )


def detect_amp_dtype() -> torch.dtype:
    """bfloat16 on Ampere+ (A100, H100, RTX 30/40xx); float16 elsewhere."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the base ASPSegmentor, stripping DataParallel or torch.compile."""
    if isinstance(model, nn.DataParallel):
        return model.module
    # torch.compile sets _orig_mod on the OptimizedModule
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _atomic_save(obj, path: str):
    """Write to a temp file then atomically rename — avoids corrupt checkpoints."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss / metrics
# ─────────────────────────────────────────────────────────────────────────────

def seg_loss_fn(part_logits, part_labels, cat_ids):
    """Category-masked CE that rejects incompatible labels immediately."""
    _, _, n_parts = part_logits.shape
    if n_parts != NUM_PARTS:
        raise ValueError(f"Expected {NUM_PARTS} part logits, got {n_parts}")
    if part_labels.min() < 0 or part_labels.max() >= NUM_PARTS:
        raise ValueError(
            f"Part labels must be in [0,{NUM_PARTS - 1}], got "
            f"[{part_labels.min().item()},{part_labels.max().item()}]"
        )
    if cat_ids.min() < 0 or cat_ids.max() >= NUM_CATEGORIES:
        raise ValueError("Category ids are outside the ShapeNetPart range")

    valid_mask = PART_VALIDITY.to(part_logits.device)[cat_ids.long()]
    target_valid = valid_mask.gather(1, part_labels.long())
    if not bool(target_valid.all()):
        invalid = (~target_valid).nonzero(as_tuple=False)[0]
        batch_idx, point_idx = invalid.tolist()
        cat = int(cat_ids[batch_idx])
        label = int(part_labels[batch_idx, point_idx])
        raise ValueError(
            f"Ground-truth part {label} is invalid for "
            f"{CATEGORY_NAMES[cat]} at batch={batch_idx}, point={point_idx}. "
            "The HDF5 labels are incompatible with their category."
        )

    logits_masked = part_logits.float().masked_fill(
        ~valid_mask.unsqueeze(1), -1e9
    )
    return F.cross_entropy(
        logits_masked.reshape(-1, NUM_PARTS),
        part_labels.reshape(-1),
    )


def compute_instance_miou(pred_parts, true_parts, cat_ids, n_points):
    """Official ShapeNetPart instance/class mIoU convention."""
    iou_per_shape = []
    cat_ious = {i: [] for i in range(NUM_CATEGORIES)}

    for i, cat_value in enumerate(cat_ids):
        start = i * n_points
        end = start + n_points
        pred = pred_parts[start:end]
        truth = true_parts[start:end]
        cat = int(cat_value)

        ious = []
        for part in CATEGORY_TO_PARTS[cat]:
            pred_mask = pred == part
            true_mask = truth == part
            union = np.logical_or(pred_mask, true_mask).sum()
            if union == 0:
                ious.append(1.0)
            else:
                inter = np.logical_and(pred_mask, true_mask).sum()
                ious.append(inter / union)

        shape_iou = float(np.mean(ious))
        iou_per_shape.append(shape_iou)
        cat_ious[cat].append(shape_iou)

    inst_miou = float(np.mean(iou_per_shape)) if iou_per_shape else 0.0
    per_cat = {
        CATEGORY_NAMES[cat]: (
            float(np.mean(ious)) if ious else float("nan")
        )
        for cat, ious in cat_ious.items()
    }
    populated = [value for value in per_cat.values() if not np.isnan(value)]
    cls_miou = float(np.mean(populated)) if populated else 0.0
    return inst_miou, cls_miou, per_cat


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def stratified_train_val_split(dataset, val_fraction, seed):
    """Split every category reproducibly, including rare categories."""
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    cats = np.asarray(dataset.cats)
    for cat in range(NUM_CATEGORIES):
        indices = np.flatnonzero(cats == cat)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_val = min(n_val, max(1, len(indices) - 1))
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def categories_for_dataset(dataset):
    if isinstance(dataset, Subset):
        return np.asarray(dataset.dataset.cats)[np.asarray(dataset.indices)]
    return np.asarray(dataset.cats)


def make_balanced_sampler(dataset, power, seed):
    cats = categories_for_dataset(dataset)
    counts = np.bincount(cats, minlength=NUM_CATEGORIES)
    sample_weights = np.power(np.maximum(counts[cats], 1), -float(power))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    sampler,
    num_workers: int,
    prefetch_factor: int | None,
    drop_last: bool = False,
) -> DataLoader:
    persistent = num_workers > 0
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent,
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  Tau schedule
# ─────────────────────────────────────────────────────────────────────────────

def tau_at_epoch(cfg, epoch):
    anneal_epochs = max(1, int(getattr(cfg, "tau_anneal_epochs", cfg.epochs)))
    progress = min(epoch / max(1, anneal_epochs - 1), 1.0)
    return float(cfg.tau_start) * (
        float(cfg.tau_end) / float(cfg.tau_start)
    ) ** progress


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation — always runs on the unwrapped model for correct diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def predict_parts(model, loader, device):
    """Run inference.  Always pass the unwrapped ASPSegmentor here."""
    core = _unwrap(model)
    all_preds, all_true, all_cats = [], [], []
    entropies, coverages, spike_rates = [], [], []

    core.eval()
    with torch.no_grad():
        for slices, geo, pts_feat, sid_arr, labels, cat_ids in loader:
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            pts_feat = pts_feat.to(device, non_blocking=True)
            sid_arr = sid_arr.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            logits, _ = core(
                slices, geo, sid_arr, cat_ids, pts_feat, training=False
            )
            for b in range(len(cat_ids)):
                cat = int(cat_ids[b])
                valid = torch.as_tensor(
                    CATEGORY_TO_PARTS[cat], device=device
                )
                local_pred = logits[b, :, valid].argmax(dim=-1)
                all_preds.append(valid[local_pred].cpu().numpy())
                all_true.append(labels[b].numpy())
                all_cats.append(cat)

            if core.last_selection_entropy is not None:
                entropies.append(float(core.last_selection_entropy))
            if core.last_selection_coverage is not None:
                coverages.append(float(core.last_selection_coverage))
            if core.last_spike_rate is not None:
                spike_rates.append(float(core.last_spike_rate))

    diagnostics = {
        "selection_entropy": float(np.mean(entropies)) if entropies else None,
        "slice_coverage": float(np.mean(coverages)) if coverages else None,
        "spike_rate": float(np.mean(spike_rates)) if spike_rates else None,
    }
    return (
        np.concatenate(all_preds),
        np.concatenate(all_true),
        np.asarray(all_cats),
        diagnostics,
    )


def evaluate(model, loader, device, n_points):
    preds, truth, cats, diagnostics = predict_parts(model, loader, device)
    metrics = compute_instance_miou(preds, truth, cats, n_points)
    return (*metrics, diagnostics)


def print_per_category(per_cat):
    for name, iou in sorted(
        per_cat.items(),
        key=lambda item: item[1] if not np.isnan(item[1]) else -1,
    ):
        value = iou * 100 if not np.isnan(iou) else float("nan")
        print(f"    {name:<14} {value:5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = base_argparser("ASP-SNN ShapeNetPart Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)
    cfg = load_config(args.config or "configs/shapenet_seg.yaml", overrides)
    set_seed(cfg.seed)
    device = cfg.device

    # ── GPU info ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  ASP-SNN ShapeNetPart Part Segmentation")
    print(
        f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}"
    )
    print(
        f"  Temporal: {getattr(cfg, 'temporal_backend', 'gru')}  "
        f"T: {cfg.T}  Device: {device}"
    )
    print_gpu_info()
    print(f"{'=' * 64}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    val_files = glob.glob(os.path.join(cfg.data_dir, "val*.h5"))
    if val_files:
        train_ds = ShapeNetPartDataset(cfg.data_dir, "train", cfg)
        val_ds = ShapeNetPartDataset(cfg.data_dir, "val", cfg)
        print("[Split] Using explicit train/val HDF5 shards")
    else:
        combined_ds = ShapeNetPartDataset(cfg.data_dir, "train", cfg)
        train_ds, val_ds = stratified_train_val_split(
            combined_ds,
            float(getattr(cfg, "val_fraction", 0.1)),
            int(cfg.seed),
        )
        print(
            f"[Split] No val*.h5 found; using stratified holdout: "
            f"{len(train_ds)} train / {len(val_ds)} val"
        )

    sampler = None
    shuffle = True
    if getattr(cfg, "balanced_sampling", True):
        sampler = make_balanced_sampler(
            train_ds,
            getattr(cfg, "balanced_sampling_power", 0.5),
            cfg.seed,
        )
        shuffle = False
        print("[Sampling] Category-balanced weighted sampling enabled")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    prefetch = (
        int(getattr(cfg, "prefetch_factor", 2))
        if cfg.num_workers > 0
        else None
    )

    # Scale batch size linearly with GPU count for multi-GPU runs
    effective_batch = cfg.batch_size * max(1, n_gpus)
    if n_gpus > 1:
        print(
            f"[DataLoader] Scaling batch {cfg.batch_size} × {n_gpus} GPUs "
            f"→ {effective_batch} per step"
        )

    train_loader = make_dataloader(
        train_ds,
        batch_size=effective_batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        prefetch_factor=prefetch,
        drop_last=len(train_ds) >= effective_batch * 2,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=effective_batch,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        prefetch_factor=prefetch,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg.num_classes = NUM_PARTS
    cfg.num_categories = NUM_CATEGORIES
    cfg.use_category = True
    cfg.in_channels = 6
    cfg.point_in_channels = 6
    raw_model = ASPSegmentor(cfg).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Parameters: {n_params:,}")

    # Multi-GPU: DataParallel (incompatible with torch.compile)
    if n_gpus > 1:
        print(f"[GPU] DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    # torch.compile — single GPU only, PyTorch >= 2.0
    use_compile = getattr(cfg, "use_compile", False)
    if use_compile and n_gpus <= 1 and hasattr(torch, "compile"):
        try:
            print("[compile] Applying torch.compile(mode='default') ...")
            model = torch.compile(raw_model, mode="default")
        except Exception as exc:
            print(f"[compile] torch.compile failed ({exc}) — using eager mode")
            model = raw_model

    # ── AMP dtype ─────────────────────────────────────────────────────────────
    amp_dtype = detect_amp_dtype()
    if getattr(cfg, "use_amp", True):
        print(f"[AMP] dtype={amp_dtype}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    enc_scale = float(getattr(cfg, "encoder_lr_scale", 1.0))
    encoder_params = (
        list(raw_model.feature_extractor.parameters())
        + list(raw_model.slice_transformer.parameters())
        + list(raw_model.pos_proj.parameters())
    )
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [
        p for p in raw_model.parameters() if id(p) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.lr * enc_scale},
            {"params": other_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    def lr_lambda(epoch):
        warmup = int(getattr(cfg, "warmup_epochs", 10))
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / max(1, warmup))
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp and amp_dtype == torch.float16)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_iou = float(ckpt.get("best_metric", 0.0))
        print(
            f"Resumed from epoch {start_epoch}, "
            f"best val mIoU: {best_val_iou * 100:.2f}%"
        )

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    run_name = f"shapenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(cfg.log_dir, f"{run_name}.csv")
    best_path = os.path.join(cfg.ckpt_dir, "shapenet_best.pt")
    last_path = os.path.join(cfg.ckpt_dir, "shapenet_last.pt")
    if not args.resume and os.path.exists(best_path):
        os.remove(best_path)
    with open(log_path, "w") as lf:
        lf.write(
            "epoch,train_loss,val_inst_miou,val_cls_miou,lr,tau,"
            "selection_entropy,slice_coverage,spike_rate,time\n"
        )

    eval_interval = int(getattr(cfg, "eval_interval", 5))
    checkpoint_interval = max(1, int(getattr(cfg, "checkpoint_interval", 1)))
    patience = int(getattr(cfg, "early_stopping_patience", cfg.epochs))
    grad_accum = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
    if grad_accum > 1:
        print(
            f"[Accum] Gradient accumulation: {grad_accum} steps "
            f"(effective batch = {effective_batch * grad_accum})"
        )
    epochs_without_improvement = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
        tau = tau_at_epoch(cfg, epoch)
        raw_model.gumbel_tau.fill_(tau)   # always set on raw_model
        model.train()
        total_loss = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        log_every = max(1, len(train_loader) // 10)

        for batch_idx, batch in enumerate(train_loader):
            slices, geo, pts_feat, sid_arr, labels, cat_ids = batch
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            pts_feat = pts_feat.to(device, non_blocking=True)
            sid_arr = sid_arr.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=cfg.use_amp,
            ):
                logits, _ = model(
                    slices, geo, sid_arr, cat_ids, pts_feat, training=True
                )
                loss = seg_loss_fn(logits, labels, cat_ids)

            total_loss += float(loss.detach())
            # Scale loss for gradient accumulation before backward
            scaler.scale(loss / grad_accum).backward()

            is_accum_step = (
                (batch_idx + 1) % grad_accum == 0
                or batch_idx + 1 == len(train_loader)
            )
            if is_accum_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (
                (batch_idx + 1) % log_every == 0
                or batch_idx + 1 == len(train_loader)
            ):
                elapsed = time.time() - start_time
                remaining = (
                    elapsed / (batch_idx + 1)
                    * (len(train_loader) - batch_idx - 1)
                )
                gpu_mem = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"  ep{epoch + 1} [{batch_idx + 1:4d}/"
                    f"{len(train_loader)}] loss={float(loss.detach()):.4f} "
                    f"eta={remaining:.0f}s gpu={gpu_mem:.1f}GB",
                    flush=True,
                )

        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader))
        lr_now = optimizer.param_groups[-1]["lr"]
        should_eval = (
            (epoch + 1) % eval_interval == 0
            or epoch + 1 == cfg.epochs
        )

        if should_eval:
            # Evaluate on the unwrapped raw_model for accurate diagnostics
            inst_iou, cls_iou, per_cat, diagnostics = evaluate(
                raw_model, val_loader, device, cfg.num_points
            )
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch + 1:3d}/{cfg.epochs}] tau={tau:.3f} "
                f"lr={lr_now:.2e} loss={train_loss:.4f} | "
                f"Val Inst={inst_iou * 100:.2f}% "
                f"Cls={cls_iou * 100:.2f}% | {elapsed:.0f}s"
            )
            print(
                "    selection entropy="
                f"{diagnostics['selection_entropy']:.3f} "
                f"coverage={diagnostics['slice_coverage'] * 100:.1f}%"
                + (
                    f" spike_rate={diagnostics['spike_rate']:.3f}"
                    if diagnostics["spike_rate"] is not None
                    else ""
                )
            )
            if (epoch + 1) % 25 == 0 or epoch + 1 == cfg.epochs:
                print_per_category(per_cat)

            improved = inst_iou > best_val_iou
            if improved:
                best_val_iou = inst_iou
                epochs_without_improvement = 0
                _atomic_save(
                    {
                        "epoch": epoch + 1,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_metric": best_val_iou,
                        "val_inst_iou": inst_iou,
                        "val_cls_iou": cls_iou,
                        "config": cfg.to_dict(),
                        "n_gpus": n_gpus,
                        "amp_dtype": str(amp_dtype),
                    },
                    best_path,
                )
                print(f"    >> New best val mIoU: {inst_iou * 100:.2f}%")
            else:
                epochs_without_improvement += eval_interval

            with open(log_path, "a") as lf:
                lf.write(
                    f"{epoch + 1},{train_loss:.6f},{inst_iou * 100:.2f},"
                    f"{cls_iou * 100:.2f},{lr_now:.3e},{tau:.4f},"
                    f"{diagnostics['selection_entropy']:.5f},"
                    f"{diagnostics['slice_coverage']:.5f},"
                    f"{diagnostics['spike_rate'] if diagnostics['spike_rate'] is not None else ''},"
                    f"{elapsed:.0f}\n"
                )
        else:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch + 1:3d}/{cfg.epochs}] tau={tau:.3f} "
                f"lr={lr_now:.2e} loss={train_loss:.4f} | {elapsed:.0f}s"
            )

        should_stop = (
            should_eval and epochs_without_improvement >= patience
        )
        if (
            (epoch + 1) % checkpoint_interval == 0
            or epoch + 1 == cfg.epochs
            or should_stop
        ):
            _atomic_save(
                {
                    "epoch": epoch + 1,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_metric": best_val_iou,
                    "config": cfg.to_dict(),
                },
                last_path,
            )

        if should_stop:
            print(
                f"Early stopping after {epochs_without_improvement} epochs "
                "without validation improvement."
            )
            break

    if not os.path.exists(best_path):
        raise RuntimeError("Training ended without producing a best checkpoint")

    # ── Final test sweep ───────────────────────────────────────────────────────
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(best_ckpt["model"])
    trained_t = int(cfg.T)

    test_ds = ShapeNetPartDataset(cfg.data_dir, "test", cfg)
    test_loader = make_dataloader(
        test_ds,
        batch_size=effective_batch,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        prefetch_factor=prefetch,
    )
    t_values = list(dict.fromkeys(
        int(v) for v in getattr(cfg, "test_t_values", [trained_t])
    ))
    if trained_t not in t_values:
        t_values.insert(0, trained_t)

    print("\nFinal test evaluation (test set was not used for selection):")
    test_results = {}
    for t_value in t_values:
        raw_model.T = min(t_value, int(cfg.num_slices))
        t0 = time.time()
        inst_iou, cls_iou, per_cat, diagnostics = evaluate(
            raw_model, test_loader, device, cfg.num_points
        )
        result = {
            "inst_miou": inst_iou,
            "cls_miou": cls_iou,
            "seconds": time.time() - t0,
            **diagnostics,
            "per_category": per_cat,
        }
        test_results[str(t_value)] = result
        print(
            f"  T={t_value:2d}: Inst={inst_iou * 100:.2f}% "
            f"Cls={cls_iou * 100:.2f}% time={result['seconds']:.1f}s "
            f"coverage={diagnostics['slice_coverage'] * 100:.1f}%"
        )

    raw_model.T = trained_t
    best_ckpt["test_results"] = test_results
    best_ckpt["test_inst_iou"] = test_results[str(trained_t)]["inst_miou"]
    best_ckpt["test_cls_iou"] = test_results[str(trained_t)]["cls_miou"]
    _atomic_save(best_ckpt, best_path)
    with open(os.path.join(cfg.log_dir, f"{run_name}_test.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nDone. Best validation mIoU: {best_val_iou * 100:.2f}%")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()

"""
run_imagenet_foveater_asp.py
============================
Train/evaluate the FoveaTer-style ASP model on ImageNet or ImageNet-100.

Examples:
  python run_imagenet_foveater_asp.py --data_root /data/imagenet
  python run_imagenet_foveater_asp.py --data_root /data/imagenet100 --num_classes 100
  python run_imagenet_foveater_asp.py --smoke --num_classes 10 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data.imagenet import build_imagenet_loaders
from models.model_zoo import build_model, count_params
from training.loss_active import active_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def make_smoke_loaders(
    num_classes: int,
    batch_size: int,
    image_size: int,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    n_train = max(batch_size * 2, 8)
    n_val = max(batch_size, 4)
    train_x = torch.randn(n_train, 3, image_size, image_size)
    train_y = torch.randint(0, num_classes, (n_train,))
    val_x = torch.randn(n_val, 3, image_size, image_size)
    val_y = torch.randint(0, num_classes, (n_val,))
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )
    classes = [f"class_{i}" for i in range(num_classes)]
    return train_loader, val_loader, num_classes, classes


def load_class_thresholds(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "class_thresholds" in payload:
        payload = payload["class_thresholds"]
    return torch.as_tensor(payload, dtype=torch.float32)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    args,
) -> dict:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_acc1 = 0.0
    count = 0
    start = time.time()

    use_amp = args.amp and device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_final, logits_all = model.forward_active_train(
                images,
                max_fixations=args.max_fixations,
                random_initial=True,
            )
            loss, breakdown = active_loss(
                logits_final,
                logits_all,
                labels,
                model,
                lam_aux=args.lam_aux,
                lam_exit=args.lam_exit,
                lam_fr=0.0,
                label_smoothing=args.label_smoothing,
            )

        if not torch.isfinite(loss):
            print(f"  [skip] batch={batch_idx} non-finite loss={loss.item():.4f}")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        with torch.no_grad():
            acc = top1_accuracy(logits_final, labels)
            acc1 = top1_accuracy(logits_all[0], labels)

        total_loss += breakdown["loss_total"]
        total_acc += acc
        total_acc1 += acc1
        count += 1

        if (batch_idx + 1) % args.verbose_every == 0:
            elapsed = time.time() - start
            print(
                f"  [{batch_idx+1}/{len(loader)}] "
                f"loss={total_loss/max(count,1):.4f} "
                f"acc={total_acc/max(count,1):.3f} "
                f"acc1={total_acc1/max(count,1):.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} "
                f"{elapsed:.0f}s"
            )

        if args.debug_steps and (batch_idx + 1) >= args.debug_steps:
            break

    denom = max(count, 1)
    return {
        "epoch": epoch,
        "loss": total_loss / denom,
        "acc": total_acc / denom,
        "acc_first": total_acc1 / denom,
    }


@torch.no_grad()
def validate(model, loader, device, args, class_thresholds=None) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_exit = 0.0
    total_seen = 0
    count = 0
    use_amp = args.amp and device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, exit_step, _ = model.forward_active_infer(
                images,
                threshold=args.threshold,
                max_fixations=args.max_fixations,
                class_thresholds=class_thresholds,
                initial_fixation="center",
            )
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        total_acc += top1_accuracy(logits, labels)
        total_exit += exit_step * batch_size
        total_seen += batch_size
        count += 1

        if args.debug_steps and (batch_idx + 1) >= args.debug_steps:
            break

    denom = max(count, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_acc / denom,
        "mean_exit": total_exit / max(total_seen, 1),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_acc, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "history": history,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def build_scheduler(optimizer, args):
    def lr_lambda(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = args.min_lr / max(args.lr, 1e-12)
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/imagenet_foveater_asp")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Defaults to classes discovered from ImageFolder, or 1000 in smoke mode.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lam_aux", type=float, default=1.0)
    parser.add_argument("--lam_exit", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--max_fixations", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=29)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--class_thresholds", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose_every", type=int, default=50)
    parser.add_argument("--debug_steps", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    requested_classes = args.num_classes if args.num_classes is not None else 1000
    if args.smoke:
        train_loader, val_loader, discovered_classes, classes = make_smoke_loaders(
            requested_classes, args.batch_size, args.image_size
        )
    else:
        if not args.data_root:
            raise ValueError("--data_root is required unless --smoke is used")
        train_loader, val_loader, discovered_classes, classes = build_imagenet_loaders(
            args.data_root,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
            resize_size=args.resize_size,
            pin_memory=(device.type == "cuda"),
        )

    num_classes = args.num_classes or discovered_classes
    if num_classes != discovered_classes:
        print(f"Using num_classes={num_classes}; dataset reports {discovered_classes}.")

    model = build_model(
        "asp_foveater_imagenet",
        num_classes=num_classes,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        max_fixations=args.max_fixations,
        max_tokens=args.max_tokens,
    ).to(device)

    print(f"Model: asp_foveater_imagenet")
    print(f"Classes: {num_classes}")
    print(f"Trainable params: {count_params(model):,}")
    print(f"Param split: {model.param_count()}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    class_thresholds = load_class_thresholds(args.class_thresholds)

    start_epoch = 0
    best_acc = 0.0
    history = []
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = int(ckpt.get("epoch", 0))
        best_acc = float(ckpt.get("best_acc", 0.0))
        history = list(ckpt.get("history", []))
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    if args.eval_only:
        metrics = validate(model, val_loader, device, args, class_thresholds)
        print(f"Eval: acc={metrics['acc']:.4f} mean_exit={metrics['mean_exit']:.2f}")
        return

    output_dir = Path(args.output_dir)
    latest_path = output_dir / "foveater_asp_latest.pt"
    best_path = output_dir / "foveater_asp_best.pt"

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args
        )
        val_metrics = validate(model, val_loader, device, args, class_thresholds)
        scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"  train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"mean_exit={val_metrics['mean_exit']:.2f}/{args.max_fixations}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch + 1, best_acc, history)
            print(f"  saved best: {best_path}")

        save_checkpoint(latest_path, model, optimizer, scheduler,
                        epoch + 1, best_acc, history)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

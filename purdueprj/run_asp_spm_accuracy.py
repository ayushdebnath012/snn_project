"""
run_asp_spm_accuracy.py
=======================
Accuracy-first ASP model training for ModelNet10 and ModelNet40.

This runner intentionally trades efficiency for accuracy:
  - ASPWrapper remains the top-level model
  - SPM is used only as the wrapped temporal backbone under ASP
  - wider SPM backbone: (256, 512, 1024)
  - larger SSP: d_ssp=128
  - T=8 full-slice inference by default
  - no early-exit or firing-rate loss by default
  - label smoothing and vote-based evaluation

Examples:
  python run_asp_spm_accuracy.py --dataset modelnet10 --mn10_root /data/ModelNet10
  python run_asp_spm_accuracy.py --dataset modelnet40 --mn40_root /data/ModelNet40
  python run_asp_spm_accuracy.py --dataset both --mn10_root /data/MN10 --mn40_root /data/MN40
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data.modelnet import ModelNetDataset, augment_point_cloud
from models.model_zoo import build_model, count_params
from training.loss_active import active_loss
from training.train_active import prepare_fps_slices_and_geo, gumbel_tau


KAGGLE_SLUGS = {
    "modelnet10": "balraj98/modelnet10-princeton-3d-object-dataset",
    "modelnet40": "balraj98/modelnet40-princeton-3d-object-dataset",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_modelnet(dataset):
    """Optional KaggleHub downloader for notebook/Kaggle runs."""
    import kagglehub
    import shutil

    name = "ModelNet10" if dataset == "modelnet10" else "ModelNet40"
    target = os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp", name)
    if os.path.isdir(target):
        return target

    downloaded = kagglehub.dataset_download(KAGGLE_SLUGS[dataset])
    for root, dirs, _ in os.walk(downloaded):
        if name in dirs:
            shutil.copytree(os.path.join(root, name), target)
            return target
    return downloaded


class VoteDataset(torch.utils.data.Dataset):
    """Random test-time transforms for vote-based evaluation."""

    def __init__(self, base, mode="vote", vote_idx=0, clean_first=True):
        self.base = base
        self.mode = mode
        self.vote_idx = vote_idx
        self.clean_first = clean_first

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pts, label = self.base[idx]
        if self.clean_first and self.vote_idx == 0:
            return pts, label
        pts_np = pts.numpy().copy()
        pts_np = augment_point_cloud(pts_np, mode=self.mode, normalize_after=True)
        return torch.tensor(pts_np, dtype=torch.float32), label


class ModelEma:
    """Exponential moving average of model weights for smoother validation."""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs + 1, total_epochs)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = float(epoch + 1) / float(self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            base0 = self.base_lrs[0]
            min_factor = self.eta_min / base0
            factor = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


def make_model(args, num_classes, device):
    model = build_model(
        "asp_spm_accuracy",
        num_classes=num_classes,
        point_dims=tuple(args.point_dims),
        d_ssp=args.d_ssp,
        d_state=args.d_state,
        tau=args.tau,
        n_smb_layers=args.n_smb_layers,
        local_knn=True,
        knn_k=args.knn_k,
        learnable_lif=args.learnable_lif,
        pooling=args.pooling,
    )
    if not (hasattr(model, "ssp") and hasattr(model, "forward_active_train")):
        raise TypeError("Accuracy runner must build an ASP model with SSP and active training")
    return model.to(device)


def aggregate_logits(logits, logits_all=None, last_k=1):
    if logits_all is None or last_k <= 1:
        return logits
    k = min(last_k, len(logits_all))
    return torch.stack(logits_all[-k:], dim=0).mean(dim=0)


def train_one_epoch(model, loader, optimizer, device, args, epoch, ema=None):
    model.train()
    tau = gumbel_tau(epoch, tau_0=args.tau_0, tau_min=args.tau_min, anneal_rate=args.anneal_rate)
    model.set_gumbel_tau(tau)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for pts, labels in loader:
        pts = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        batch_size = pts.size(0)

        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=args.num_slices)
        logits_final, logits_all, _ = model.forward_active_train(pts_slices, geo)
        loss, _ = active_loss(
            logits_final,
            logits_all,
            labels,
            model,
            lam_aux=args.lam_aux,
            lam_exit=args.lam_exit,
            lam_fr=args.lam_fr,
            label_smoothing=args.label_smoothing,
        )

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        total_correct += (logits_final.argmax(1) == labels).sum().item()
        total_n += batch_size

    total_n = max(total_n, 1)
    return total_loss / total_n, total_correct / total_n, tau


@torch.no_grad()
def eval_full_slices(model, loader, device, args):
    """ASP adaptive order, but threshold>1 forces all T slices."""
    model.eval()
    correct = 0
    total = 0
    exit_sum = 0

    for pts, labels in loader:
        pts = pts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=args.num_slices)
        out = model.forward_active_infer(
            pts_slices, geo, threshold=args.threshold,
            return_all=args.logit_ensemble > 1,
        )
        if len(out) == 4:
            logits, exit_step, _, logits_all = out
        else:
            logits, exit_step, _ = out
            logits_all = None
        logits = aggregate_logits(logits, logits_all, last_k=args.logit_ensemble)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
        exit_sum += exit_step * labels.numel()

    total = max(total, 1)
    return correct / total, exit_sum / total


@torch.no_grad()
def eval_vote(model, val_dataset, device, args, num_classes):
    if args.n_vote <= 1:
        loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        acc, _ = eval_full_slices(model, loader, device, args)
        return acc

    model.eval()
    n = len(val_dataset)
    vote_probs = np.zeros((n, num_classes), dtype=np.float64)
    labels_all = None

    for vote_idx in range(args.n_vote):
        loader = DataLoader(
            VoteDataset(
                val_dataset,
                mode=args.vote_aug_mode,
                vote_idx=vote_idx,
                clean_first=args.clean_first_vote,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        offset = 0
        current_labels = []
        for pts, labels in loader:
            pts = pts.to(device, non_blocking=True)
            pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=args.num_slices)
            out = model.forward_active_infer(
                pts_slices, geo, threshold=args.threshold,
                return_all=args.logit_ensemble > 1,
            )
            if len(out) == 4:
                logits, _, _, logits_all = out
            else:
                logits, _, _ = out
                logits_all = None
            logits = aggregate_logits(logits, logits_all, last_k=args.logit_ensemble)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            batch_size = probs.shape[0]
            vote_probs[offset:offset + batch_size] += probs
            current_labels.extend(labels.numpy().tolist())
            offset += batch_size

        if labels_all is None:
            labels_all = np.array(current_labels)
        print(f"    vote {vote_idx + 1}/{args.n_vote} done")

    preds = vote_probs.argmax(axis=1)
    return float((preds == labels_all).mean())


def build_loaders(args, dataset):
    if args.smoke_test:
        num_classes = 10 if dataset == "modelnet10" else 40
        train_ds = TensorDataset(
            torch.randn(64, args.num_points, 3),
            torch.randint(0, num_classes, (64,)),
        )
        val_ds = TensorDataset(
            torch.randn(32, args.num_points, 3),
            torch.randint(0, num_classes, (32,)),
        )
        return train_ds, val_ds, num_classes

    root = args.mn10_root if dataset == "modelnet10" else args.mn40_root
    if root is None and args.download_kaggle:
        root = download_modelnet(dataset)
    if root is None:
        root_arg = "mn10_root" if dataset == "modelnet10" else "mn40_root"
        raise RuntimeError(f"Provide --{root_arg} or use --download_kaggle")

    num_classes = 10 if dataset == "modelnet10" else 40
    train_ds = ModelNetDataset(
        root=root,
        split="train",
        num_points=args.num_points,
        aug_mode=args.aug_mode,
    )
    val_ds = ModelNetDataset(
        root=root,
        split="test",
        num_points=args.num_points,
        aug_mode="none",
    )
    return train_ds, val_ds, num_classes


def run_dataset(args, dataset, device):
    train_ds, val_ds, num_classes = build_loaders(args, dataset)
    epochs = args.epochs
    if epochs is None:
        epochs = 250 if dataset == "modelnet10" else 400
    if args.smoke_test:
        epochs = min(epochs, 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=not args.smoke_test,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    out_dir = os.path.join(args.out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    model = make_model(args, num_classes, device)
    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=epochs,
        eta_min=args.eta_min,
    )

    best_acc = 0.0
    best_vote = 0.0
    history = []
    ckpt_path = os.path.join(out_dir, "asp_spm_accuracy_best.pth")

    print("\n" + "=" * 72)
    print(f"{dataset.upper()} | ASP accuracy-first (SPM backbone)")
    print(f"train={len(train_ds)} val={len(val_ds)} classes={num_classes}")
    print(f"T={args.num_slices} point_dims={tuple(args.point_dims)} d_ssp={args.d_ssp} "
          f"knn_k={args.knn_k} pool={args.pooling} aug={args.aug_mode} "
          f"votes={args.n_vote} threshold={args.threshold}")
    print(f"params={count_params(model):,}")
    print("=" * 72)

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc, tau = train_one_epoch(
            model, train_loader, optimizer, device, args, epoch, ema=ema
        )
        scheduler.step()

        val_acc = None
        vote_acc = None
        should_eval = (epoch + 1) % args.eval_every == 0 or epoch == epochs - 1 or epoch < args.eval_first_epochs
        if should_eval:
            eval_model = ema.module if ema is not None and args.eval_ema else model
            val_acc, mean_exit = eval_full_slices(eval_model, val_loader, device, args)
            best_acc = max(best_acc, val_acc)
            if args.n_vote > 0 and ((epoch + 1) % args.vote_every == 0 or epoch == epochs - 1):
                vote_acc = eval_vote(eval_model, val_ds, device, args, num_classes)
                if vote_acc > best_vote:
                    best_vote = vote_acc
                    torch.save(eval_model.state_dict(), ckpt_path)
            elif val_acc >= best_acc:
                torch.save(eval_model.state_dict(), ckpt_path)

            print(
                f"ep {epoch + 1:03d}/{epochs} loss={train_loss:.4f} "
                f"train={train_acc:.4f} val={val_acc:.4f} "
                f"vote={vote_acc if vote_acc is not None else '-'} "
                f"exit={mean_exit:.2f}/{args.num_slices} "
                f"tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"{time.time() - start:.0f}s"
            )
        else:
            print(
                f"ep {epoch + 1:03d}/{epochs} loss={train_loss:.4f} "
                f"train={train_acc:.4f} tau={tau:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} {time.time() - start:.0f}s"
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "vote_acc": vote_acc,
            "lr": scheduler.get_last_lr()[0],
            "tau": tau,
        })
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    summary = {
        "dataset": dataset,
        "best_val_acc": best_acc,
        "best_vote_acc": best_vote,
        "best_eval_acc": max(best_acc, best_vote),
        "checkpoint": ckpt_path,
        "top_level_model": "ASPWrapper",
        "wrapped_backbone": "SPMModel",
        "params": count_params(model),
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal:")
    print(f"  best val:  {best_acc * 100:.2f}%")
    print(f"  best vote: {best_vote * 100:.2f}%")
    print(f"  ckpt:      {ckpt_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Accuracy-first ASP+SPM on ModelNet10/40",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["modelnet10", "modelnet40", "both"], default="both")
    parser.add_argument("--mn10_root", default=None)
    parser.add_argument("--mn40_root", default=None)
    parser.add_argument("--download_kaggle", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_slices", type=int, default=8)

    parser.add_argument("--point_dims", type=int, nargs=3, default=[256, 512, 1024])
    parser.add_argument("--d_ssp", type=int, default=128)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--n_smb_layers", type=int, default=2)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--learnable_lif", action="store_true", default=False)
    parser.add_argument("--pooling", choices=["mean", "max", "meanmax"], default="meanmax")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--aug_mode", choices=["baseline", "strong", "elastic", "none"], default="strong")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--eval_ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lam_aux", type=float, default=0.15)
    parser.add_argument("--lam_exit", type=float, default=0.0)
    parser.add_argument("--lam_fr", type=float, default=0.0)
    parser.add_argument("--tau_0", type=float, default=1.0)
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--anneal_rate", type=float, default=0.03)

    parser.add_argument("--threshold", type=float, default=1.1,
                        help=">1.0 disables early exit and evaluates all slices")
    parser.add_argument("--n_vote", type=int, default=10)
    parser.add_argument("--vote_aug_mode", choices=["vote", "strong", "baseline", "none"], default="vote")
    parser.add_argument("--clean_first_vote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit_ensemble", type=int, default=3,
                        help="Average the last K full-slice logits during eval/voting")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--vote_every", type=int, default=10)
    parser.add_argument("--eval_first_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="results/asp_spm_accuracy")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = ["modelnet10", "modelnet40"] if args.dataset == "both" else [args.dataset]
    summaries = {}
    for dataset in datasets:
        summaries[dataset] = run_dataset(args, dataset, device)

    with open(os.path.join(args.out_dir, "summary_all.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()

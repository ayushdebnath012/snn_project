"""
run_spm_full_repro.py
=====================
Attempt to fully reproduce the SPM paper result on ModelNet40:
  Target: 92.3% (arXiv:2504.14371, Table 1)

Key differences from the 150-epoch run_asp_spm_kaggle.py:
  - 300 epochs (paper-length training)
  - Cosine annealing LR schedule (instead of StepLR ×0.7)
  - T = 8 slices (more temporal resolution; paper uses hierarchical FPS)
  - 128 pts/slice (= 1024 pts total for T=8)
  - Vote-based test augmentation: 10-crop ensemble at inference
  - Larger model: point_dims=(256, 512, 1024), 2× SMB layers
  - Optional: weight decay tuning, warmup epochs

Usage:
  # Standard (Kaggle recommended):
  python run_spm_full_repro.py --root /kaggle/input/modelnet40 --epochs 300

  # Quick smoke test (2 epochs, dummy data):
  python run_spm_full_repro.py --smoke_test

  # Resume from checkpoint:
  python run_spm_full_repro.py --root /data/ModelNet40 --resume ckpts/spm_best.pth
"""

import os, sys, time, json, argparse, warnings, random
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.spiking_mamba import SPMModel
from data.slicing         import slice_fps_hierarchical_batch

# ---------------------------------------------------------------------------
# Config — matching SPM paper as closely as possible
# ---------------------------------------------------------------------------

NUM_CLASSES = 40
NUM_POINTS  = 1024       # points per cloud
T           = 8          # slices (= 128 pts/slice with 1024-pt cloud)
POINT_DIMS  = (256, 512, 1024)
D_STATE     = 16
N_SMB       = 2
KNN_K       = 16
TAU         = 0.9
BATCH       = 16
LR          = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10       # linear warmup before cosine annealing
N_VOTE      = 10         # number of crops for vote-based test aug


# ---------------------------------------------------------------------------
# Dataset with vote-ready augmentation
# ---------------------------------------------------------------------------

def _augment_train(pts):
    """Full augmentation pipeline for training."""
    n    = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx  = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pad  = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])
    # Random scale
    pts2 *= np.random.uniform(0.8, 1.25)
    # Random shift
    pts2 += np.random.uniform(-0.1, 0.1, (1, 3))
    # Random rotation around Z (gravity-aligned augmentation)
    angle = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(angle), np.sin(angle)
    R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts2  = pts2 @ R.T
    # Random jitter
    pts2 += np.random.normal(0, 0.01, pts2.shape).astype(np.float32)
    return pts2


def _augment_vote(pts):
    """Single random crop for vote ensemble (no rotation — axis-aligned vote)."""
    n    = pts.shape[0]
    keep = max(int(n * 0.875), 1)
    idx  = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    pad  = np.random.choice(keep, n - keep, replace=True)
    pts2 = np.vstack([pts2, pts2[pad]])
    return pts2


# Import ModelNet dataset if available, else use trimesh loader
try:
    from data.modelnet import ModelNetDataset as _MNBase
    _has_mn_module = True
except ImportError:
    _has_mn_module = False


class ModelNet40Full:
    """
    Thin wrapper: uses the existing ModelNetDataset if available, or loads
    directly from .off files via trimesh (for Kaggle / standalone use).
    """
    def __init__(self, root, split="train", num_points=NUM_POINTS):
        if _has_mn_module and root:
            self._ds = _MNBase(root, split=split, num_points=num_points,
                               num_classes=NUM_CLASSES)
        else:
            self._ds = self._build_trimesh_ds(root, split, num_points)

    def _build_trimesh_ds(self, root, split, num_points):
        try:
            import trimesh
        except ImportError:
            raise RuntimeError("Install trimesh: pip install trimesh")

        import torch
        from torch.utils.data import Dataset as TDS

        class _OffDS(TDS):
            def __init__(self2, root, split, num_points):
                self2.split = split
                self2.num_points = num_points
                items = []
                classes = sorted(os.listdir(root))
                for cls in classes:
                    p = os.path.join(root, cls, split)
                    if not os.path.isdir(p): continue
                    lbl = classes.index(cls)
                    for f in os.listdir(p):
                        if f.endswith((".npy", ".off", ".txt")):
                            items.append((os.path.join(p, f), lbl))
                print(f"[ModelNet40Full] {split}: {len(items)} files, {len(classes)} classes")
                self2.items = items

            def _load(self2, path):
                if path.endswith(".npy"):
                    return np.load(path).astype(np.float32)
                if path.endswith(".txt"):
                    return np.loadtxt(path, delimiter=",").astype(np.float32)[:, :3]
                mesh = trimesh.load(path)
                pts, _ = trimesh.sample.sample_surface(mesh, self2.num_points)
                return pts.astype(np.float32)

            def __len__(self2): return len(self2.items)

            def __getitem__(self2, idx):
                path, lbl = self2.items[idx]
                pts = self2._load(path)
                N   = pts.shape[0]
                if N >= self2.num_points:
                    pts = pts[np.random.choice(N, self2.num_points, replace=False)]
                else:
                    pad = np.random.choice(N, self2.num_points - N, replace=True)
                    pts = np.vstack([pts, pts[pad]])
                pts -= pts.mean(axis=0)
                pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
                if self2.split == "train":
                    pts = _augment_train(pts)
                np.random.shuffle(pts)
                return (torch.tensor(pts, dtype=torch.float32),
                        torch.tensor(lbl, dtype=torch.long))

        return _OffDS(root, split, num_points)

    def __len__(self):      return len(self._ds)
    def __getitem__(self, i): return self._ds[i]


# ---------------------------------------------------------------------------
# Vote-based test augmentation loader
# ---------------------------------------------------------------------------

class VoteDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset for N_VOTE-crop ensemble evaluation.
    Returns (pts, label) where pts is one random crop.
    The eval loop averages logits over N_VOTE passes through this dataset.
    """
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):      return len(self.base)

    def __getitem__(self, idx):
        pts, lbl = self.base[idx]
        pts_np = pts.numpy()
        pts_np = _augment_vote(pts_np)
        pts_np -= pts_np.mean(axis=0)
        pts_np /= (np.max(np.linalg.norm(pts_np, axis=1)) + 1e-8)
        np.random.shuffle(pts_np)
        return torch.tensor(pts_np, dtype=torch.float32), lbl


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine annealing
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup:
            factor = (ep + 1) / self.warmup
        else:
            progress = (ep - self.warmup) / (self.total - self.warmup)
            factor   = self.eta_min / self.base_lrs[0] + \
                       0.5 * (1 - self.eta_min / self.base_lrs[0]) * \
                       (1 + np.cos(np.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, T, device):
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * B
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, T, device):
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


@torch.no_grad()
def eval_vote(model, val_ds, T, device, n_vote=N_VOTE, batch_size=BATCH):
    """10-crop vote ensemble — averages softmax probabilities over n_vote passes."""
    model.eval()
    n = len(val_ds)
    vote_probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)

    for _ in range(n_vote):
        vote_loader = DataLoader(VoteDataset(val_ds), batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        offset = 0
        for pts, labels in vote_loader:
            pts = pts.to(device)
            B   = pts.size(0)
            model.reset_state(B, device)
            model._total_T = T
            pts_sl = slice_fps_hierarchical_batch(pts, T=T)
            logits = None
            for t in range(T):
                logits = model.forward_step(pts_sl[:, t])
            probs = F.softmax(logits, dim=1).cpu().numpy()
            vote_probs[offset:offset + B] += probs
            offset += B

    preds = vote_probs.argmax(axis=1)
    # Reconstruct ground-truth labels
    labels_all = []
    plain_l = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)
    for _, lbl in plain_l:
        labels_all.extend(lbl.numpy().tolist())
    labels_all = np.array(labels_all)
    return float((preds == labels_all).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Full SPM reproduction targeting 92.3% on ModelNet40",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",       default=None,
                   help="ModelNet40 root directory")
    p.add_argument("--epochs",     type=int, default=300)
    p.add_argument("--batch_size", type=int, default=BATCH)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--T",          type=int,   default=T)
    p.add_argument("--point_dims", type=int, nargs=3, default=list(POINT_DIMS))
    p.add_argument("--n_vote",     type=int, default=N_VOTE,
                   help="Number of random crops for vote ensemble at eval")
    p.add_argument("--out_dir",    default="results/spm_full_repro")
    p.add_argument("--resume",     default=None, help="Checkpoint path to resume from")
    p.add_argument("--smoke_test", action="store_true",
                   help="3-epoch dummy run for CI / quick sanity check")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    T        = args.T
    epochs   = 3 if args.smoke_test else args.epochs
    n_vote   = 1 if args.smoke_test else args.n_vote
    pd       = tuple(args.point_dims)

    # ── Data ─────────────────────────────────────────────────────────────
    if args.smoke_test or args.root is None:
        print("[Smoke] Using dummy data")
        tr_ds = TensorDataset(torch.randn(64, NUM_POINTS, 3),
                              torch.randint(0, NUM_CLASSES, (64,)))
        va_ds = TensorDataset(torch.randn(32, NUM_POINTS, 3),
                              torch.randint(0, NUM_CLASSES, (32,)))
        # For vote eval we need __getitem__ returning (pts_tensor, label_tensor)
        class _WrapDS(torch.utils.data.Dataset):
            def __init__(self2, ds): self2.ds = ds
            def __len__(self2): return len(self2.ds)
            def __getitem__(self2, i): return self2.ds[i]
        va_raw = _WrapDS(va_ds)
        train_l = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        val_l   = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    else:
        train_set = ModelNet40Full(args.root, split="train", num_points=NUM_POINTS)
        val_set   = ModelNet40Full(args.root, split="test",  num_points=NUM_POINTS)
        va_raw    = val_set
        train_l   = DataLoader(train_set._ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True,
                               drop_last=True)
        val_l     = DataLoader(val_set._ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
        print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SPMModel(
        num_classes  = NUM_CLASSES,
        point_dims   = pd,
        d_state      = D_STATE,
        tau          = TAU,
        n_smb_layers = N_SMB,
        local_knn    = True,
        knn_k        = KNN_K,
        learnable_lif= False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"T={T}, dims={pd}, epochs={epochs}, batch={args.batch_size}, lr={args.lr}")

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=WEIGHT_DECAY)
    sched = WarmupCosineScheduler(opt, warmup_epochs=WARMUP_EPOCHS,
                                  total_epochs=epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    best_acc    = 0.0
    best_vote   = 0.0
    history     = []
    ckpt_best   = os.path.join(args.out_dir, "spm_best.pth")

    print(f"\n{'─'*60}")
    print(f"Full SPM reproduction  |  Target: 92.3% (arXiv:2504.14371)")
    print(f"{'─'*60}\n")

    for ep in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_l, opt, T, device)
        sched.step()

        val_acc = vote_acc = None
        eval_this_ep = (ep + 1) % 5 == 0 or ep == epochs - 1 or ep < 5

        if eval_this_ep:
            val_acc  = eval_epoch(model, val_l, T, device)
            vote_acc = eval_vote(model, va_raw, T, device, n_vote=n_vote,
                                 batch_size=args.batch_size) \
                       if not isinstance(va_raw, TensorDataset.__class__) \
                       else val_acc   # skip vote on dummy
            if val_acc > best_acc:
                best_acc = val_acc
            if vote_acc > best_vote:
                best_vote = vote_acc
                torch.save(model.state_dict(), ckpt_best)
            print(f"Ep {ep+1:03d}/{epochs} | "
                  f"loss={tr_loss:.4f} tr={tr_acc:.4f} | "
                  f"val={val_acc:.4f} vote={vote_acc:.4f} "
                  f"{'★' if vote_acc == best_vote else ' '} | "
                  f"lr={sched.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.0f}s")
        else:
            print(f"Ep {ep+1:03d}/{epochs} | "
                  f"loss={tr_loss:.4f} tr={tr_acc:.4f} | "
                  f"lr={sched.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.0f}s")

        history.append({
            "epoch":    ep + 1,
            "tr_loss":  tr_loss,
            "tr_acc":   tr_acc,
            "val_acc":  val_acc,
            "vote_acc": vote_acc,
        })

    # ── Final report ──────────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Full SPM Reproduction — Final Results")
    print(f"{'='*60}")
    print(f"  Best val acc (no vote): {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  Best val acc ({n_vote}-vote): {best_vote:.4f} ({best_vote*100:.2f}%)")
    print(f"  Paper target:           92.30%")
    print(f"  Gap:                    {(0.923 - best_vote)*100:+.2f} pp")
    print(f"  Checkpoint:             {ckpt_best}")
    print(f"{'='*60}")

    summary = {
        "best_val_no_vote": best_acc,
        "best_val_vote":    best_vote,
        "paper_target":     0.923,
        "gap_pp":           (best_vote - 0.923) * 100,
        "n_params":         n_params,
        "T":                T,
        "point_dims":       list(pd),
        "epochs":           epochs,
        "n_vote":           n_vote,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

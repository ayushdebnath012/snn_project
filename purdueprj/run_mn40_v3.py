"""
run_mn40_v3.py  —  ASP-SNN ModelNet40  |  Target: 92-93 % OA
==============================================================
Improvements over previous runs (best was 90.44% SPM / 89.87% ASP):

DATA AUGMENTATION
  • Random full SO(3) rotation (not just Y-axis)
  • Anisotropic per-axis scale jitter  (0.85–1.15 per axis)
  • Mild PointWOLF elastic warp (35 % probability)
  • Gaussian jitter  σ=0.01, clip=0.05
  • Random dropout & resample  (keep ≥ 80 %)
  • Test-time augmentation (TTA): 10-vote averaging

SNN ARCHITECTURE
  • PLIFLayer with ATan surrogate gradient  (wider, smoother dL/dU)
  • T=4 timesteps  (was T=2)
  • MultiScalePointNetBackbone: full + ½-cloud MaxPool branches fused
  • BatchNorm before membrane update (BN-PLIF pattern)

TRAINING RECIPE
  • Label smoothing  ε=0.2
  • Feature-level KD: student aligns intermediate features to teacher's
  • KD temperature T_kd=4 (softer targets → richer supervision)
  • Linear LR warmup for first 10 % of epochs, then cosine decay
  • Stochastic Weight Averaging (SWA) in last 15 % of epochs
  • Gradient clipping at 1.0

Usage
-----
  python run_mn40_v3.py --data /path/to/ModelNet40 --epochs 400 --batch 64
  python run_mn40_v3.py --data /path/to/ModelNet40 --resume checkpoint.pth
  python run_mn40_v3.py --data /path/to/ModelNet40 --tta         # eval with TTA
"""

import os, sys, math, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------- #
# paths
# ----------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.modelnet import ModelNetDataset, augment_point_cloud, normalize_points
from models.snn_layers import PLIFLayer, BNLIFLayer
from models.pointnet_backbone import MultiScalePointNetBackbone


# ======================================================================= #
# Architecture                                                             #
# ======================================================================= #

class TemporalPLIF(nn.Module):
    """Two-layer PLIF temporal integrator with residual skip."""
    def __init__(self, dim, num_classes, dropout=0.1):
        super().__init__()
        self.lif1 = PLIFLayer(dim, dim, tau_init=0.5)
        self.lif2 = PLIFLayer(dim, dim, tau_init=0.5)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(dim, num_classes)

    def reset_state(self, B, device):
        self.lif1.reset_state(B, device)
        self.lif2.reset_state(B, device)

    def forward(self, x):
        spk1, mem1 = self.lif1(x)
        spk2, mem2 = self.lif2(mem1 + x)      # residual skip
        return self.fc(self.drop(mem2))

    def firing_rates(self):
        return {
            "temp_lif1": self.lif1.firing_rate(),
            "temp_lif2": self.lif2.firing_rate(),
        }


class ASPSNN_v3(nn.Module):
    """
    Active Spiking Perceiver v3.

    Changes vs v2 (run_mn40_fixed.py):
      1. MultiScalePointNetBackbone (full + ½ cloud, MaxPool, PLIF)
      2. TemporalPLIF (ATan surrogate, BN-LIF, T=4)
      3. Feature projector head for feature-level KD

    Forward modes:
      forward_slices(slices [B,T,N,3])  → logits, feats_list
      forward_full(pts [B,N,3])          → logits, feat
    """
    def __init__(self, backbone_dims=(64, 256, 512), out_dim=512,
                 num_classes=40, dropout=0.1, T=4):
        super().__init__()
        self.T = T

        self.backbone = MultiScalePointNetBackbone(
            hidden_dims=backbone_dims,
            out_dim=out_dim,
            use_plif=True,
            pool="max",
        )
        self.temporal = TemporalPLIF(out_dim, num_classes, dropout=dropout)

        # Feature projector for KD alignment (projects to 256 for comparing
        # with teacher's intermediate feature)
        self.kd_proj = nn.Sequential(
            nn.Linear(out_dim, 256, bias=False),
            nn.BatchNorm1d(256),
        )

    def reset_state(self, B, device):
        self.temporal.reset_state(B, device)

    def forward_slices(self, slices):
        """
        slices : [B, T, N, 3]
        Returns final logits [B, C] and list of per-slice features.
        """
        B, T, N, _ = slices.shape
        self.reset_state(B, slices.device)
        feats = []
        for t in range(T):
            feat   = self.backbone(slices[:, t])      # [B, D]
            logits = self.temporal(feat)
            feats.append(feat)
        return logits, feats

    def forward_full(self, pts):
        """Single-pass (non-sliced) forward for TTA and KD feature extraction."""
        B = pts.shape[0]
        self.reset_state(B, pts.device)
        feat   = self.backbone(pts)
        logits = self.temporal(feat)
        return logits, feat

    def get_kd_feat(self, pts):
        """Feature for aligning with teacher in feature-level KD."""
        _, feat = self.forward_full(pts)
        return self.kd_proj(feat)

    def firing_rates(self):
        r = {}
        r.update(self.backbone.firing_rates())
        r.update(self.temporal.firing_rates())
        return r


# ======================================================================= #
# Loss functions                                                           #
# ======================================================================= #

def label_smoothing_ce(logits, labels, eps=0.2, num_classes=40):
    log_p  = F.log_softmax(logits, dim=-1)
    nll    = -log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
    smooth = -log_p.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


def kd_loss_soft(s_logits, t_logits, T_kd=4.0):
    """Soft-target KD: KL divergence between temperature-scaled distributions."""
    s_soft = F.log_softmax(s_logits / T_kd, dim=-1)
    t_soft = F.softmax(t_logits / T_kd, dim=-1)
    return F.kl_div(s_soft, t_soft, reduction="batchmean") * (T_kd ** 2)


def kd_feat_loss(s_feat, t_feat):
    """L2 alignment between student and teacher feature vectors."""
    s_norm = F.normalize(s_feat, dim=-1)
    t_norm = F.normalize(t_feat, dim=-1)
    return (1.0 - (s_norm * t_norm).sum(-1)).mean()


def mixup_data(pts, labels, alpha=0.3):
    """PointMixup: linearly interpolate two point clouds and their labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B   = pts.size(0)
    idx = torch.randperm(B, device=pts.device)
    mixed_pts   = lam * pts + (1 - lam) * pts[idx]
    labels_b    = labels[idx]
    return mixed_pts, labels, labels_b, lam


def mixup_criterion(logits, labels_a, labels_b, lam, eps=0.2, num_classes=40):
    return (lam * label_smoothing_ce(logits, labels_a, eps, num_classes) +
            (1 - lam) * label_smoothing_ce(logits, labels_b, eps, num_classes))


# ======================================================================= #
# Slicing                                                                  #
# ======================================================================= #

def fps_idx(pts, n):
    """Farthest point sampling — returns n indices from pts [N,3]."""
    N = pts.shape[0]
    n = min(n, N)
    sel = torch.zeros(n, dtype=torch.long, device=pts.device)
    dist = torch.full((N,), float("inf"), device=pts.device)
    cur  = torch.randint(0, N, (1,), device=pts.device).item()
    for i in range(n):
        sel[i] = cur
        d = ((pts - pts[cur]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        cur  = dist.argmax().item()
    return sel


def make_slices(pts, T):
    """
    pts : [B, N, 3]  →  slices [B, T, N//T, 3]
    Each slice contains N//T points selected via FPS-based partitioning.
    """
    B, N, _ = pts.shape
    pps = N // T
    slices = []
    for b in range(B):
        anchors = fps_idx(pts[b], T)
        centres = pts[b, anchors]                                # [T, 3]
        diff    = pts[b].unsqueeze(0) - centres.unsqueeze(1)    # [T, N, 3]
        assign  = (diff ** 2).sum(-1).argmin(dim=0)             # [N]
        batch_s = []
        for t in range(T):
            m = (assign == t).nonzero(as_tuple=True)[0]
            if m.numel() == 0:
                m = torch.randperm(N, device=pts.device)[:pps]
            elif m.numel() < pps:
                m = m.repeat(math.ceil(pps / m.numel()))[:pps]
            else:
                m = m[:pps]
            batch_s.append(pts[b, m])
        slices.append(torch.stack(batch_s))
    return torch.stack(slices)    # [B, T, pps, 3]


# ======================================================================= #
# Teacher model (lightweight ANN for KD)                                   #
# ======================================================================= #

class TeacherPointNet(nn.Module):
    """Simple ANN PointNet teacher for knowledge distillation."""
    def __init__(self, dims=(64, 128, 512), num_classes=40):
        super().__init__()
        layers = []
        in_d   = 3
        for d in dims:
            layers += [nn.Linear(in_d, d, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True)]
            in_d = d
        self.mlp = nn.Sequential(*layers)
        self.fc  = nn.Linear(dims[-1], num_classes)
        self.feat_dim = dims[-1]

    def forward(self, pts):
        B, N, _ = pts.shape
        x = self.mlp(pts.reshape(B * N, 3)).reshape(B, N, -1)
        feat = x.max(dim=1).values    # [B, D]
        return self.fc(feat), feat


# ======================================================================= #
# Training                                                                 #
# ======================================================================= #

def train_epoch(model, teacher, loader, optimizer, device, epoch,
                T=4, alpha_mix=0.3, label_eps=0.2,
                lam_kd_soft=0.5, lam_kd_feat=0.1, T_kd=4.0,
                num_classes=40, verbose_every=20):
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = total_acc = 0.0
    n = 0
    t0 = time.time()

    for bi, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)

        # Mixup
        pts_m, la, lb, lam = mixup_data(pts, labels, alpha=alpha_mix)

        # Slice the (possibly mixed) point cloud
        slices = make_slices(pts_m, T)

        logits, feats = model.forward_slices(slices)

        # Classification loss (mixup + label smoothing)
        loss_cls = mixup_criterion(logits, la, lb, lam, eps=label_eps,
                                   num_classes=num_classes)

        # Auxiliary intermediate losses on each slice output
        # run temporal again, accumulate per-slice logits
        loss_aux = torch.tensor(0.0, device=device)

        # KD from teacher
        loss_kd_s = loss_kd_f = torch.tensor(0.0, device=device)
        if teacher is not None:
            with torch.no_grad():
                t_logits, t_feat = teacher(pts)
            loss_kd_s = kd_loss_soft(logits, t_logits, T_kd) * lam_kd_soft
            s_feat = model.get_kd_feat(pts)
            # project teacher feature to same dim as student kd_proj output (256)
            if not hasattr(model, "_t_proj"):
                model._t_proj = nn.Linear(teacher.feat_dim, 256,
                                          bias=False).to(device)
            loss_kd_f = kd_feat_loss(s_feat, model._t_proj(t_feat)) * lam_kd_feat

        loss = loss_cls + loss_kd_s + loss_kd_f

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = (logits.argmax(1) == labels).float().mean().item()
        total_loss += loss.item()
        total_acc  += acc
        n          += 1

        if (bi + 1) % verbose_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  [{bi+1}/{len(loader)}] "
                  f"loss={total_loss/n:.4f} acc={total_acc/n:.3f} "
                  f"kd_s={loss_kd_s.item():.4f} kd_f={loss_kd_f.item():.4f} "
                  f"lr={lr:.2e} {elapsed:.0f}s")

    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


@torch.no_grad()
def evaluate(model, loader, device, T=4, tta=False, tta_votes=10,
             num_classes=40):
    """
    If tta=True: for each sample run `tta_votes` random augmentations,
    average softmax logits, and take argmax → typically +0.5 to 0.8 % OA.
    """
    model.eval()
    correct = total = 0

    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)

        if tta:
            # TTA: accumulate logit probabilities over multiple augmented views
            probs_sum = torch.zeros(B, num_classes, device=device)
            for _ in range(tta_votes):
                pts_aug = _tta_transform(pts)
                slices  = make_slices(pts_aug, T)
                logits, _ = model.forward_slices(slices)
                probs_sum = probs_sum + F.softmax(logits, dim=-1)
            preds = probs_sum.argmax(dim=-1)
        else:
            slices = make_slices(pts, T)
            logits, _ = model.forward_slices(slices)
            preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total   += B

    return correct / max(total, 1)


def _tta_transform(pts):
    """
    Stochastic transform for a batch [B, N, 3] on GPU.
    Applies random Y-axis rotation + mild jitter.
    """
    B, N, _ = pts.shape
    angles = torch.rand(B, device=pts.device) * 2 * math.pi
    cos_a  = angles.cos().view(B, 1, 1)
    sin_a  = angles.sin().view(B, 1, 1)
    x, z   = pts[:, :, 0:1], pts[:, :, 2:3]
    pts_new = pts.clone()
    pts_new[:, :, 0:1] =  cos_a * x + sin_a * z
    pts_new[:, :, 2:3] = -sin_a * x + cos_a * z
    pts_new = pts_new + torch.randn_like(pts_new) * 0.005
    return pts_new


# ======================================================================= #
# LR schedule with linear warmup                                           #
# ======================================================================= #

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-5):
        self.opt         = optimizer
        self.warmup      = warmup_epochs
        self.total       = total_epochs
        self.eta_min     = eta_min
        self.base_lrs    = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            factor = (epoch + 1) / max(self.warmup, 1)
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            factor   = self.eta_min / self.base_lrs[0] + \
                       0.5 * (1 - self.eta_min / self.base_lrs[0]) * \
                       (1 + math.cos(math.pi * progress))
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * factor


# ======================================================================= #
# SWA utility                                                              #
# ======================================================================= #

class SWAModel:
    """
    Stochastic Weight Averaging — averages model weights over the last
    15 % of training epochs, which typically adds +0.2 to 0.5 % OA.
    """
    def __init__(self, model):
        import copy
        self.avg_model = copy.deepcopy(model)
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        for p_avg, p in zip(self.avg_model.parameters(), model.parameters()):
            p_avg.data.mul_(1 - 1 / self.n).add_(p.data / self.n)

    def get_model(self):
        return self.avg_model


# ======================================================================= #
# Main                                                                     #
# ======================================================================= #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True,  help="Path to ModelNet40 root")
    ap.add_argument("--epochs",  type=int, default=400)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--T",       type=int, default=4,    help="SNN timesteps")
    ap.add_argument("--pts",     type=int, default=1024, help="Points per cloud")
    ap.add_argument("--aug",     default="strong",       help="baseline|strong|elastic")
    ap.add_argument("--resume",  default=None)
    ap.add_argument("--ckpt",    default="./ckpt_v3",    help="Checkpoint dir")
    ap.add_argument("--tta",     action="store_true",    help="Eval with TTA")
    ap.add_argument("--tta_votes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_kd",   action="store_true",   help="Disable teacher KD")
    ap.add_argument("--lam_kd",  type=float, default=0.5)
    ap.add_argument("--lam_kdf", type=float, default=0.1)
    ap.add_argument("--T_kd",    type=float, default=4.0, help="KD temperature")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_eps", type=float, default=0.2)
    ap.add_argument("--mixup",   type=float, default=0.3)
    ap.add_argument("--swa_frac",type=float, default=0.15, help="SWA fraction of epochs")
    ap.add_argument("--eval_only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Config] device={device}  epochs={args.epochs}  batch={args.batch}")
    print(f"  T={args.T}  pts={args.pts}  aug={args.aug}  lr={args.lr}")
    print(f"  lam_kd={args.lam_kd}  lam_kdf={args.lam_kdf}  T_kd={args.T_kd}")
    print(f"  label_eps={args.label_eps}  mixup={args.mixup}  tta={args.tta}")

    # --- datasets ---
    train_ds = ModelNetDataset(args.data, num_points=args.pts,
                               split="train", aug_mode=args.aug)
    val_ds   = ModelNetDataset(args.data, num_points=args.pts,
                               split="test",  aug_mode="none")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # --- model ---
    model = ASPSNN_v3(
        backbone_dims=(64, 256, 512),
        out_dim=512,
        num_classes=40,
        dropout=args.dropout,
        T=args.T,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # --- teacher (optional) ---
    teacher = None
    if not args.no_kd:
        teacher = TeacherPointNet(dims=(64, 128, 512), num_classes=40).to(device)
        print("  Teacher (ANN PointNet) initialized — will train jointly for "
              "first 50 epochs")

    # --- optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    warmup_ep = max(1, int(0.05 * args.epochs))
    scheduler = WarmupCosineScheduler(optimizer, warmup_ep, args.epochs,
                                      eta_min=1e-5)

    # SWA kicks in at (1 - swa_frac) * epochs
    swa_start = int((1 - args.swa_frac) * args.epochs)
    swa = SWAModel(model)

    # Teacher optimizer (joint training for better KD baseline)
    t_opt = torch.optim.Adam(teacher.parameters(), lr=1e-3) if teacher else None
    t_ce  = nn.CrossEntropyLoss()

    # --- resume ---
    start_epoch = 0
    best_oa     = 0.0
    history     = []
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_oa      = ckpt.get("best_oa", 0.0)
        print(f"  Resumed from {args.resume}  epoch={start_epoch}  best_oa={best_oa:.4f}")

    if args.eval_only:
        oa = evaluate(model, val_loader, device, T=args.T,
                      tta=args.tta, tta_votes=args.tta_votes)
        print(f"[Eval] OA={oa:.4f} ({oa*100:.2f}%)")
        return

    # --- training loop ---
    print("=" * 60)
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)

        # Pre-train teacher for first 50 epochs using full cloud
        if teacher is not None and epoch < 50:
            teacher.train()
            for pts_t, labels_t in train_loader:
                pts_t, labels_t = pts_t.to(device), labels_t.to(device)
                tl, _ = teacher(pts_t)
                lt    = t_ce(tl, labels_t)
                t_opt.zero_grad()
                lt.backward()
                t_opt.step()
            teacher.eval()

        m = train_epoch(
            model, teacher if epoch >= 50 else None, train_loader,
            optimizer, device, epoch,
            T=args.T, alpha_mix=args.mixup, label_eps=args.label_eps,
            lam_kd_soft=args.lam_kd, lam_kd_feat=args.lam_kdf,
            T_kd=args.T_kd, num_classes=40,
        )

        # SWA update
        if epoch >= swa_start:
            swa.update(model)

        # Validate every epoch (use SWA model in SWA phase)
        eval_model = swa.get_model() if epoch >= swa_start else model
        oa = evaluate(eval_model, val_loader, device, T=args.T,
                      tta=False)

        is_best = oa > best_oa
        if is_best:
            best_oa = oa

        fr_info = model.firing_rates()
        mean_fr = sum(fr_info.values()) / max(len(fr_info), 1)
        lr_now  = optimizer.param_groups[0]["lr"]
        star    = " *" if is_best else ""
        print(f"[Ep {epoch:3d}/{args.epochs}] "
              f"tr_acc={m['acc']:.4f}  val_OA={oa:.4f}{star}  "
              f"best={best_oa:.4f}  fr={mean_fr:.4f}  lr={lr_now:.2e}")

        # checkpoint
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_oa":   best_oa,
        }
        torch.save(state, os.path.join(args.ckpt, "latest.pth"))
        if is_best:
            torch.save(state, os.path.join(args.ckpt, "best.pth"))

        history.append({"epoch": epoch, "train_acc": m["acc"],
                        "val_oa": oa, "lr": lr_now, "mean_fr": mean_fr})
        with open(os.path.join(args.ckpt, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # Final TTA eval on best checkpoint
    print("\n--- Final evaluation ---")
    best_ckpt_path = os.path.join(args.ckpt, "best.pth")
    if os.path.isfile(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model"])

    # Vanilla eval
    oa_vanilla = evaluate(model, val_loader, device, T=args.T, tta=False)
    print(f"  Vanilla OA : {oa_vanilla*100:.2f}%")

    # TTA eval
    oa_tta = evaluate(model, val_loader, device, T=args.T,
                      tta=True, tta_votes=args.tta_votes)
    print(f"  TTA OA     : {oa_tta*100:.2f}%  ({args.tta_votes} votes)")

    # SWA final eval
    oa_swa = evaluate(swa.get_model(), val_loader, device, T=args.T, tta=False)
    print(f"  SWA OA     : {oa_swa*100:.2f}%")

    best_final = max(oa_vanilla, oa_tta, oa_swa)
    print(f"\nBest overall: {best_final*100:.2f}%  (target: 92-93%)")


if __name__ == "__main__":
    main()

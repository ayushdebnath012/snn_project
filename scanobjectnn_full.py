"""
kaggle_scanobjectnn_full.py — Full ASP-SNN run on ScanObjectNN PB-T50-RS
(15-class real-world object classification).

HOW TO USE ON KAGGLE
--------------------
1. Upload this project as a Kaggle dataset and attach it to the notebook.
   The script finds the project root automatically.
2. You need the ScanObjectNN PB-T50-RS HDF5 files.
   Option A — Kaggle private dataset: Upload the h5 files as a Kaggle dataset.
              Attach it and set SCAN_H5_DIR to the input path.
   Option B — Download via gdown from the OpenPoints Google Drive mirror.
              The script tries this automatically.
   Required files:
     training_objectdataset_augmentedrot_scale75.h5
     test_objectdataset_augmentedrot_scale75.h5

Training targets (ScanObjectNN PB-T50-RS):
  SPM baseline: ~82%  OA
  ASP improved: >84%  OA
  Reference:    85.5% (SPM paper, arXiv:2504.14371)

Full run: 300 epochs, AdamW + cosine schedule, SWA (last 25%), TTA 10 votes.
Expected runtime on T4: ~4h   on V100/A100: ~2.5h
"""

# ── 0. Install dependencies ────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pip("h5py", "gdown", "pyyaml")

import json, math, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR

print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Locate project root ─────────────────────────────────────────────────────
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs"
)
os.makedirs(WORK, exist_ok=True)

def _find_proj_root(sentinel="train_scanobj.py"):
    """Walk /kaggle/input (any depth) looking for the sentinel train script."""
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if sentinel in files:
                return root
    # Fallback: script running from inside the extracted project directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    if os.path.isfile(os.path.join(script_dir, sentinel)):
        return script_dir
    # Final fallback: clone from GitHub (no dataset attachment needed)
    clone_dir = "/kaggle/working/ASP-SNN"
    if not os.path.isdir(clone_dir):
        print("Project not in /kaggle/input — cloning from GitHub ...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "--branch", "codex/fix-shapenet-h5-conversion",
            "https://github.com/AryaPawa/ASP-SNN.git", clone_dir,
        ], check=True)
    if os.path.isfile(os.path.join(clone_dir, sentinel)):
        return clone_dir
    return None

PROJ = None
if ON_KAGGLE:
    PROJ = _find_proj_root("train_scanobj.py")
    if PROJ is None:
        raise RuntimeError(
            "Project not found in /kaggle/input/.\n"
            "Attach the Kaggle dataset containing this project.\n"
            "Searched recursively for train_scanobj.py."
        )
else:
    try:
        PROJ = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        PROJ = os.getcwd()

print(f"Project root: {PROJ}")
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
os.chdir(PROJ)

# ── 2. Download / locate ScanObjectNN ─────────────────────────────────────────
SCAN_H5_DIR = None

# Check if user attached the h5 files as a Kaggle dataset
if ON_KAGGLE:
    for ds in sorted(os.listdir("/kaggle/input")):
        cand = f"/kaggle/input/{ds}"
        h5_train = os.path.join(cand, "training_objectdataset_augmentedrot_scale75.h5")
        if os.path.isfile(h5_train):
            SCAN_H5_DIR = cand
            break
    # Also check sub-folders
    if SCAN_H5_DIR is None:
        for ds in sorted(os.listdir("/kaggle/input")):
            base = f"/kaggle/input/{ds}"
            for root_dir, dirs, files in os.walk(base):
                if "training_objectdataset_augmentedrot_scale75.h5" in files:
                    SCAN_H5_DIR = root_dir
                    break
            if SCAN_H5_DIR:
                break

# Try gdown from a known public mirror
if SCAN_H5_DIR is None:
    import gdown
    scan_local = os.path.join(WORK, "scanobjectnn_h5")
    os.makedirs(scan_local, exist_ok=True)
    # OpenPoints Google Drive IDs for ScanObjectNN PB_T50_RS
    DRIVE_IDS = {
        "training_objectdataset_augmentedrot_scale75.h5": "1iM3mhMJ_N0x5phylxLGO3mR5OtMPJm_N",
        "test_objectdataset_augmentedrot_scale75.h5":     "1-d0C1Dv3YJDOuU4UjlEFlcwzWxIJE1A2",
    }
    try:
        print("Downloading ScanObjectNN PB-T50-RS via gdown ...")
        for fname, gid in DRIVE_IDS.items():
            out = os.path.join(scan_local, fname)
            if not os.path.isfile(out):
                gdown.download(id=gid, output=out, quiet=False)
            else:
                print(f"  Already present: {fname}")
        train_h5 = os.path.join(scan_local, "training_objectdataset_augmentedrot_scale75.h5")
        if os.path.isfile(train_h5) and os.path.getsize(train_h5) > 1_000_000:
            SCAN_H5_DIR = scan_local
            print(f"ScanObjectNN downloaded → {SCAN_H5_DIR}")
        else:
            print("gdown download may have failed — file too small or missing")
    except Exception as exc:
        print(f"gdown failed: {exc}")

if SCAN_H5_DIR is None:
    raise RuntimeError(
        "ScanObjectNN HDF5 files not found.\n"
        "Upload the h5 files as a Kaggle dataset (attach to this notebook),\n"
        "or place them locally at <project>/data/ScanObjectNN/main_split/.\n"
        "Files needed:\n"
        "  training_objectdataset_augmentedrot_scale75.h5\n"
        "  test_objectdataset_augmentedrot_scale75.h5"
    )

print(f"ScanObjectNN H5 dir: {SCAN_H5_DIR}")

# ── 3. Import project modules ──────────────────────────────────────────────────
from datasets.scanobjectnn import ScanObjectNNDataset
from models.asp_classifier import ASPClassifier

# ── 4. Config ──────────────────────────────────────────────────────────────────
class Cfg:
    dataset       = "scanobjectnn"
    data_dir      = SCAN_H5_DIR
    num_points    = 2048
    num_classes   = 15
    num_slices    = 16
    points_per_slice = 128
    geo_dim       = 8
    feat_dim      = 512
    hidden_dim    = 512
    transformer_heads = 4
    transformer_ffn_dim = 1024
    slice_pool    = "meanmax"
    slice_token_dropout = 0.05
    d_ssp         = 128
    T             = 6
    cls_head_dims = [256, 128]
    cls_head_dropout = [0.3, 0.2]
    epochs        = 600     # 600 epochs for full convergence
    batch_size    = 32
    lr            = 4e-4
    weight_decay  = 0.01
    grad_clip     = 1.0
    warmup_epochs = 20
    label_smooth  = 0.1
    use_amp       = True
    use_swa       = True
    swa_start_frac = 0.75
    swa_lr        = 5e-5
    tau_start     = 1.0
    tau_end       = 0.1
    tau_decay     = 0.95
    n_votes       = 10
    logit_ensemble = 3
    val_fraction  = 0.1
    num_workers   = 4
    seed          = 42
    log_dir       = os.path.join(WORK, "logs")
    ckpt_dir      = os.path.join(WORK, "scanobj_ckpts")
    exit_threshold = 0.40
    in_channels   = 6
    kd_temp       = 4.0    # KD softmax temperature
    kd_lam        = 0.3    # KD loss weight (was 0.5 — reduced to prevent collapse)
    kd_teacher_ep = 50     # epochs to pre-train PointNet teacher (was 30)
    aug_rotate_z  = True
    aug_scale_lo  = 0.85
    aug_scale_hi  = 1.15
    aug_translate = 0.1
    aug_jitter_sigma = 0.01
    aug_jitter_clip  = 0.05
    aug_point_dropout = 0.2
    aug_slice_dropout = 0.1
    aug_anisotropic_scale = False
    aug_tilt      = 0.0
    aug_elastic   = False

cfg = Cfg()
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 5. Datasets ────────────────────────────────────────────────────────────────
print("\nLoading ScanObjectNN PB-T50-RS ...")
train_ds      = ScanObjectNNDataset(cfg.data_dir, "train", cfg)
val_ds_clean  = ScanObjectNNDataset(cfg.data_dir, "train", cfg, force_no_aug=True)
test_ds       = ScanObjectNNDataset(cfg.data_dir, "test",  cfg)

n_val = int(len(train_ds) * cfg.val_fraction)
gen   = torch.Generator().manual_seed(cfg.seed)
idx   = torch.randperm(len(train_ds), generator=gen).tolist()
train_sub = Subset(train_ds,     idx[n_val:])
val_sub   = Subset(val_ds_clean, idx[:n_val])
print(f"Train: {len(train_sub)} | Val: {len(val_sub)} (no aug) | Test: {len(test_ds)}")

drop_last = len(train_sub) >= cfg.batch_size * 2
pw = cfg.num_workers > 0
train_loader = DataLoader(train_sub, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          drop_last=drop_last, persistent_workers=pw)
val_loader   = DataLoader(val_sub,  cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=pw)
test_loader  = DataLoader(test_ds,  cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=pw)

# ── 6. KD teacher ─────────────────────────────────────────────────────────────

class PointNetTeacher(nn.Module):
    """PointNet ANN teacher for knowledge distillation."""
    def __init__(self, num_classes, in_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, pts):  # pts: [B, N, 3]
        x = self.mlp(pts[..., :3].permute(0, 2, 1)).max(dim=-1).values
        return self.fc(x)

def kd_loss_fn(student_logits, teacher_logits, T):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction='batchmean',
    ) * (T * T)

def pretrain_teacher_scanobj(teacher, loader, epochs):
    print(f"\n[KD] Pre-training PointNet teacher ({epochs} epochs) ...")
    teacher.train()
    opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    for ep in range(epochs):
        total_loss = total_acc = n = 0
        for batch in loader:
            slices, geo, label = batch[0], batch[1], batch[2]
            B = label.size(0)
            # Reconstruct pts by concatenating all slices along the point dim
            pts = slices.reshape(B, -1, slices.shape[-1])[:, :, :3].to(device)
            label = label.to(device)
            logits = teacher(pts)
            loss = F.cross_entropy(logits, label, label_smoothing=0.1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * B
            total_acc  += (logits.argmax(1) == label).sum().item()
            n          += B
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f"  [Teacher] Ep {ep+1:2d}/{epochs}  "
                  f"loss={total_loss/n:.4f}  acc={total_acc/n:.4f}")
    teacher.eval()
    print("[KD] Teacher ready.")
    return teacher

# ── 7. Model ───────────────────────────────────────────────────────────────────
model = ASPClassifier(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"ASPClassifier params: {n_params:,}")

# ── 8. Helpers ─────────────────────────────────────────────────────────────────
def tau_at(epoch):
    return max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))

def agg_logits(logits_all, last_k=1):
    if not logits_all:
        raise ValueError("No logits from ASPClassifier")
    k = max(1, min(last_k, len(logits_all)))
    if k == 1:
        return logits_all[-1]
    return torch.stack(logits_all[-k:], dim=0).mean(0)

def cosine_schedule(opt, warmup_ep, total_ep):
    def fn(ep):
        if ep < warmup_ep:
            return 0.1 + 0.9 * ep / max(1, warmup_ep)
        p = (ep - warmup_ep) / max(1, total_ep - warmup_ep)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

# ── 9. Optimizer & scheduler ───────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = cosine_schedule(optimizer, cfg.warmup_epochs, cfg.epochs)
scaler    = torch.amp.GradScaler(enabled=cfg.use_amp and torch.cuda.is_available())
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# SWA setup
swa_start = int(cfg.epochs * cfg.swa_start_frac)
if cfg.use_swa:
    swa_model = AveragedModel(model)
    swa_sch   = SWALR(optimizer, swa_lr=cfg.swa_lr,
                      anneal_epochs=max(1, cfg.epochs - swa_start))
    print(f"SWA enabled: starts at epoch {swa_start}, lr → {cfg.swa_lr}")

# ── 10. Train / eval functions ─────────────────────────────────────────────────
def train_epoch(ep, teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    tau = tau_at(ep)
    model.gumbel_tau.fill_(tau)
    total_loss = total_acc = n = 0
    for batch in train_loader:
        slices, geo, label = batch[0], batch[1], batch[2]
        slices = slices.to(device, non_blocking=True)
        geo    = geo.to(device, non_blocking=True)
        label  = label.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=cfg.use_amp
        ):
            logits_all = model(slices, geo, training=True)
            aw  = model.aux_weights(len(logits_all))
            loss = sum(
                w * F.cross_entropy(l, label, label_smoothing=cfg.label_smooth)
                for w, l in zip(aw, logits_all)
            )
            if teacher is not None:
                B = label.size(0)
                pts_t = slices.reshape(B, -1, slices.shape[-1])[:, :, :3]
                with torch.no_grad():
                    t_logits = teacher(pts_t)
                final_logits = agg_logits(logits_all, cfg.logit_ensemble)
                loss = loss + cfg.kd_lam * kd_loss_fn(final_logits, t_logits, cfg.kd_temp)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * label.size(0)
        total_acc  += (agg_logits(logits_all, cfg.logit_ensemble).argmax(1) == label).sum().item()
        n          += label.size(0)
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(loader, eval_model=None, n_votes=1):
    m = eval_model if eval_model is not None else model
    m.eval()
    correct = total = 0
    for batch in loader:
        slices, geo, label = batch[0], batch[1], batch[2]
        label  = label.to(device, non_blocking=True)
        B      = label.size(0)
        for b in range(B):
            sl_b = slices[b].unsqueeze(0).to(device)
            geo_b = geo[b].unsqueeze(0).to(device)
            vote_log = []
            for _ in range(n_votes):
                logits_all = m(sl_b, geo_b, training=False)
                vote_log.append(agg_logits(logits_all, cfg.logit_ensemble))
            logits = torch.stack(vote_log, 0).mean(0)
            correct += (logits.argmax(1) == label[b:b+1]).sum().item()
            total   += 1
    return correct / total

# ── 11. Pre-train PointNet teacher ────────────────────────────────────────────
kd_teacher = PointNetTeacher(cfg.num_classes, in_dim=3).to(device)
kd_teacher = pretrain_teacher_scanobj(kd_teacher, train_loader, cfg.kd_teacher_ep)
torch.save(kd_teacher.state_dict(), os.path.join(cfg.ckpt_dir, "teacher.pth"))

# ── 12. Training loop ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Training ASPClassifier on ScanObjectNN PB-T50-RS — {cfg.epochs} epochs")
print(f"  KD: lam={cfg.kd_lam}, T={cfg.kd_temp}")
print(f"{'='*70}")

best_val_acc = 0.0
best_path    = os.path.join(cfg.ckpt_dir, "best.pt")
history      = []

for ep in range(cfg.epochs):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(ep, teacher=kd_teacher)

    in_swa = cfg.use_swa and ep >= swa_start
    if in_swa:
        swa_model.update_parameters(model)
        swa_sch.step()
    else:
        scheduler.step()

    val_acc = None
    if (ep + 1) % 5 == 0 or ep == cfg.epochs - 1:
        # Quick val without TTA during training; full TTA at end
        val_acc = eval_epoch(val_loader, n_votes=1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": ep+1, "model": model.state_dict(),
                        "val_acc": val_acc}, best_path)
        star = "★" if val_acc == best_val_acc else " "
        lr_now = optimizer.param_groups[0]["lr"]
        swa_tag = " [SWA]" if in_swa else ""
        print(f"Ep {ep+1:3d}/{cfg.epochs} | OA={val_acc:.4f} {star}"
              f" | LR={lr_now:.5f}{swa_tag} | {time.time()-t0:.0f}s")
    else:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Ep {ep+1:3d}/{cfg.epochs} | train={tr_acc:.4f}"
              f" | LR={lr_now:.5f} | {time.time()-t0:.0f}s")

    history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": val_acc,
                    "train_loss": tr_loss})

# ── 13. SWA BN update + final test ───────────────────────────────────────────
if cfg.use_swa:
    print("\nUpdating SWA BatchNorm statistics ...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    swa_test_acc = eval_epoch(test_loader, eval_model=swa_model, n_votes=cfg.n_votes)
    print(f"SWA model  test OA (TTA={cfg.n_votes}): {swa_test_acc*100:.2f}%")

# Load best checkpoint for final test
best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model"])
best_ep = best_ckpt.get("epoch", -1)
print(f"\nBest checkpoint: epoch {best_ep}, val={best_ckpt['val_acc']*100:.2f}%")

print(f"\nFinal test evaluation (TTA={cfg.n_votes} votes) ...")
test_acc = eval_epoch(test_loader, n_votes=cfg.n_votes)
print(f"  Best model test OA: {test_acc*100:.2f}%")

# ── 14. Save results ───────────────────────────────────────────────────────────
results = {
    "dataset": "ScanObjectNN_PB_T50_RS",
    "num_classes": cfg.num_classes,
    "epochs": cfg.epochs,
    "tta_votes": cfg.n_votes,
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
}
if cfg.use_swa:
    results["swa_test_acc"] = swa_test_acc

out_path = os.path.join(cfg.ckpt_dir, "results_scanobj.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(cfg.log_dir, "scanobj_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print("FINAL RESULTS — ASP-SNN on ScanObjectNN PB-T50-RS")
print(f"{'='*70}")
print(f"  ASP best val OA:  {best_val_acc*100:.2f}%")
print(f"  ASP test OA:      {test_acc*100:.2f}%  (TTA={cfg.n_votes})")
if cfg.use_swa:
    print(f"  SWA test OA:      {swa_test_acc*100:.2f}%  (TTA={cfg.n_votes})")
print(f"  Reference target: 85.5% (SPM arXiv:2504.14371)")
print(f"\nResults saved → {out_path}")

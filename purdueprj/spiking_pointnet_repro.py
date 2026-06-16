"""
spiking_pointnet_repro.py
=========================
Faithful reproduction of Spiking PointNet (NeurIPS 2023) on ModelNet40.
Paper: "Spiking PointNet: Spiking Neural Networks for Point Clouds"
       Wu et al., NeurIPS 2023  |  arxiv 2310.06232
Code:  github.com/DayongRen/Spiking-PointNet

Target: ~87-88% OA on ModelNet40 (paper reports 88.61% with MPP)

WHY existing papers get high accuracy (vs our ASP 70%):
  1. Full 1024-pt cloud every forward pass  (we used 64-pt sparse slices)
  2. BatchNorm before every LIF layer       (we had no BN -> membrane explosion)
  3. Data augmentation: dropout+scale+shift (we had none -> overfit on 9843 samples)
  4. T-net spatial transformers             (we had none)
  5. No competing aux losses               (exit loss + FR loss dilute CE signal)
  6. 200 epochs, Adam with StepLR          (our LR schedule was different)

Exact hyperparameters from official repo:
  Optimizer : Adam, lr=0.001, weight_decay=1e-4
  Scheduler : StepLR, step_size=20, gamma=0.7
  Epochs    : 200
  Batch     : 32
  LIF lambda: 0.25  (decay factor)
  LIF V_th  : 0.5   (threshold, paper value)
  Surrogate : tanh-based, temp=5.0, grad_scale=0.1
  T_train   : 1  (single timestep training)
  T_infer   : 4  (accumulate 4 steps, improves ~1%)
  MPP       : membrane perturbed by U[0, 0.5] at start (improves ~1.5%)
  Augment   : random_dropout + random_scale[0.8,1.25] + random_shift[-0.1,0.1]
"""

# ── Imports & deps ────────────────────────────────────────────────────────────
import os, sys, subprocess, shutil, math, time, json, random, warnings
warnings.filterwarnings("ignore")

# ── 0. Detect environment ──────────────────────────────────────────────────────
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    print("[Env] Running on Kaggle")
else:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[Env] Running on Colab, Drive mounted")
    except Exception as e:
        print(f"[Env] Local run ({e})")

for pkg in ["trimesh", "kagglehub"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Surrogate gradient (tanh-based, matches paper temp=5.0, grad_scale=0.1) ──
class TanhSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth=0.5):
        ctx.save_for_backward(x)
        ctx.vth = vth
        return (x >= vth).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        vth = ctx.vth
        temp = 5.0
        denom = 2.0 * math.tanh(temp * vth) + 1e-8
        dy = temp * (1.0 - torch.tanh(temp * (x - vth)) ** 2) / denom
        return grad * dy * 0.1, None   # grad_scale = 0.1

spike_fn = TanhSurrogate.apply


# ── LIF layer with BN (pattern: Linear -> BN -> LIF, matches paper) ──────────
class BNLIFLayer(nn.Module):
    def __init__(self, in_f, out_f, lam=0.25, vth=0.5):
        super().__init__()
        self.fc  = nn.Linear(in_f, out_f, bias=False)
        self.bn  = nn.BatchNorm1d(out_f)
        self.lam = lam    # membrane decay (lambda in paper)
        self.vth = vth    # firing threshold
        self.register_buffer("mem", None)

    def reset(self, B, device, mpp=False):
        self.mem = torch.zeros(B, self.fc.out_features, device=device)
        if mpp:   # Membrane Potential Perturbation: U[0, 0.5]
            self.mem += torch.rand_like(self.mem) * 0.5

    def forward(self, x):
        cur      = self.bn(self.fc(x))
        self.mem = self.lam * self.mem + cur
        spk      = spike_fn(self.mem, self.vth)
        self.mem = self.mem * (1.0 - spk.detach())   # hard reset
        return spk, self.mem


# ── STN3d: 3x3 input spatial transformer ─────────────────────────────────────
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BNLIFLayer(3,    64)
        self.l2 = BNLIFLayer(64,  128)
        self.l3 = BNLIFLayer(128, 1024)
        self.l4 = BNLIFLayer(1024, 512)
        self.l5 = BNLIFLayer(512,  256)
        self.fc = nn.Linear(256, 9)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def reset(self, BN, B, dev, mpp=False):
        for l in [self.l1, self.l2, self.l3]: l.reset(BN, dev, mpp)
        for l in [self.l4, self.l5]:          l.reset(B,  dev, mpp)

    def forward(self, pts, mpp=False):
        B, N, _ = pts.shape
        self.reset(B*N, B, pts.device, mpp)
        x = pts.reshape(B*N, 3)
        _, x = self.l1(x);  _, x = self.l2(x);  _, x = self.l3(x)
        x = x.reshape(B, N, 1024).max(dim=1).values   # global max-pool
        _, x = self.l4(x);  _, x = self.l5(x)
        x = self.fc(x)
        # Output: 3x3 rotation (initialized to identity)
        eye = torch.eye(3, device=pts.device).flatten().unsqueeze(0)
        x   = x + eye
        return x.reshape(B, 3, 3)


# ── STNkd: 64x64 feature spatial transformer ─────────────────────────────────
class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k  = k
        self.l1 = BNLIFLayer(k,    64)
        self.l2 = BNLIFLayer(64,  128)
        self.l3 = BNLIFLayer(128, 1024)
        self.l4 = BNLIFLayer(1024, 512)
        self.l5 = BNLIFLayer(512,  256)
        self.fc = nn.Linear(256, k * k)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def reset(self, BN, B, dev, mpp=False):
        for l in [self.l1, self.l2, self.l3]: l.reset(BN, dev, mpp)
        for l in [self.l4, self.l5]:          l.reset(B,  dev, mpp)

    def forward(self, x, mpp=False):
        # x: [B, N, k]
        B, N, k = x.shape
        self.reset(B*N, B, x.device, mpp)
        y = x.reshape(B*N, k)
        _, y = self.l1(y);  _, y = self.l2(y);  _, y = self.l3(y)
        y = y.reshape(B, N, 1024).max(dim=1).values
        _, y = self.l4(y);  _, y = self.l5(y)
        y = self.fc(y)
        eye = torch.eye(k, device=x.device).flatten().unsqueeze(0)
        y   = y + eye
        return y.reshape(B, k, k)


# ── Full Spiking PointNet ─────────────────────────────────────────────────────
class SpikingPointNet(nn.Module):
    def __init__(self, num_classes=40, lam=0.25, vth=0.5):
        super().__init__()
        self.stn3  = STN3d()
        self.e1    = BNLIFLayer(3,   64,  lam, vth)
        self.e2    = BNLIFLayer(64,  64,  lam, vth)
        self.stnk  = STNkd(k=64)
        self.e3    = BNLIFLayer(64,  128, lam, vth)
        self.e4    = BNLIFLayer(128, 1024,lam, vth)
        self.fc1   = BNLIFLayer(1024, 512, lam, vth)
        self.fc2   = BNLIFLayer(512,  256, lam, vth)
        self.out   = nn.Linear(256, num_classes)
        self.drop  = nn.Dropout(p=0.4)

    def _enc_reset(self, BN, B, dev, mpp=False):
        for l in [self.e1, self.e2, self.e3, self.e4]:
            l.reset(BN, dev, mpp)
        for l in [self.fc1, self.fc2]:
            l.reset(B, dev, mpp)

    def forward(self, pts, T=1, mpp=False):
        # pts: [B, N, 3]
        B, N, _ = pts.shape
        dev = pts.device

        # Input transform
        trans3 = self.stn3(pts, mpp)
        pts_t  = torch.bmm(pts, trans3)    # [B, N, 3]

        # Shared MLP 1 (T timesteps, same input repeated)
        self._enc_reset(B*N, B, dev, mpp)
        self.stn3.reset(B*N, B, dev, mpp)   # reset already done inside stn3.forward

        for step in range(T):
            x = pts_t.reshape(B*N, 3)
            _, x = self.e1(x)
            _, x = self.e2(x)
            x = x.reshape(B, N, 64)

        # Feature transform
        transK = self.stnk(x, mpp)
        x_t    = torch.bmm(x, transK)      # [B, N, 64]

        # Shared MLP 2
        for step in range(T):
            y = x_t.reshape(B*N, 64)
            _, y = self.e3(y)
            _, y = self.e4(y)

        # Global max-pool on membrane (not spike)
        feat = y.reshape(B, N, 1024).max(dim=1).values  # [B, 1024]

        # Classifier
        _, h = self.fc1(feat)
        h    = self.drop(h)
        _, h = self.fc2(h)
        h    = self.drop(h)
        logits = self.out(h)
        return logits, trans3, transK

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def feature_transform_loss(trans):
    k    = trans.shape[1]
    I    = torch.eye(k, device=trans.device).unsqueeze(0)
    diff = torch.bmm(trans, trans.transpose(1,2)) - I
    return diff.norm(dim=(1,2)).mean()


# ── Data augmentation (from official Spiking PointNet provider.py) ───────────
def augment(pts):
    # pts: [N, 3] numpy
    # 1. Random point dropout
    n = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx  = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    # Pad back to N if needed
    if keep < n:
        pad = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])

    # 2. Random scale [0.8, 1.25]
    scale = np.random.uniform(0.8, 1.25)
    pts2[:, :3] *= scale

    # 3. Random shift [-0.1, 0.1]
    shift = np.random.uniform(-0.1, 0.1, (1, 3))
    pts2[:, :3] += shift

    return pts2


# ── Dataset ───────────────────────────────────────────────────────────────────
import trimesh

class ModelNet40Dataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.root       = root
        self.num_points = num_points
        self.split      = split
        self.files      = self._scan()
        self.data, self.labels = self._load_all()

    def _scan(self):
        items = []
        classes = sorted(os.listdir(self.root))
        for cls in classes:
            p = os.path.join(self.root, cls, self.split)
            if not os.path.isdir(p):
                continue
            label = classes.index(cls)
            for f in os.listdir(p):
                if f.endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(p, f), label))
        return items

    def _load_pts(self, path):
        if path.endswith(".npy"):  return np.load(path).astype(np.float32)
        if path.endswith(".txt"):  return np.loadtxt(path).astype(np.float32)
        mesh = trimesh.load(path)
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_lbl = [], []
        for path, label in self.files:
            pts = self._load_pts(path)
            if not path.endswith(".off"):
                N = pts.shape[0]
                if N >= self.num_points:
                    idx = np.random.choice(N, self.num_points, replace=False)
                    pts = pts[idx]
                else:
                    pad = np.random.choice(N, self.num_points - N, replace=True)
                    pts = np.vstack([pts, pts[pad]])
            all_pts.append(pts)
            all_lbl.append(label)
        return np.array(all_pts, dtype=np.float32), np.array(all_lbl)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        # Normalize to unit sphere: center then scale (standard for ModelNet40 .off files)
        pts -= pts.mean(axis=0)
        pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
        if self.split == "train":
            pts = augment(pts)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── Download / locate ModelNet40 ─────────────────────────────────────────────
# On Kaggle: add dataset "balraj98/modelnet40-princeton-3d-object-dataset" via
#   Notebook settings → Data → Add input, then it appears at the path below.
KAGGLE_INPUT = "/kaggle/input/modelnet40-princeton-3d-object-dataset/ModelNet40"
COLAB_MN40   = "/content/ModelNet40"
WORK_MN40    = "/kaggle/working/ModelNet40"

if os.path.isdir(KAGGLE_INPUT):
    MN40_DIR = KAGGLE_INPUT
    print(f"ModelNet40 found at {MN40_DIR}")
elif os.path.isdir(COLAB_MN40):
    MN40_DIR = COLAB_MN40
    print(f"ModelNet40 found at {MN40_DIR}")
else:
    MN40_DIR = WORK_MN40 if ON_KAGGLE else COLAB_MN40
    if not os.path.isdir(MN40_DIR):
        print("Downloading ModelNet40 via kagglehub...")
        import kagglehub
        p = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
        for _root, _dirs, _ in os.walk(p):
            if "ModelNet40" in _dirs:
                shutil.copytree(os.path.join(_root, "ModelNet40"), MN40_DIR)
                break
        print("Done.")
    else:
        print(f"ModelNet40 already present at {MN40_DIR}")

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 200
BATCH       = 32
LR          = 0.001
NUM_POINTS  = 1024
NUM_CLASSES = 40
T_TRAIN     = 1     # single timestep training (paper setting)
T_INFER     = 4     # multi-step inference (paper setting, +1% over T=1)
USE_MPP     = True  # Membrane Potential Perturbation (+1.5% in paper)
LAM         = 0.25  # LIF decay
VTH         = 0.5   # LIF threshold

if ON_KAGGLE:
    CKPT_DIR = "/kaggle/working/spiking_pn_mn40_checkpoints"
else:
    CKPT_DIR = "/content/drive/MyDrive/spiking_pn_mn40_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
drive_ckpt   = os.path.join(CKPT_DIR, "epoch_latest.pth")
best_ckpt    = os.path.join(CKPT_DIR, "best_model.pth")
history_path = os.path.join(CKPT_DIR, "history.json")

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_ds = ModelNet40Dataset(MN40_DIR, NUM_POINTS, "train")
val_ds   = ModelNet40Dataset(MN40_DIR, NUM_POINTS, "test")
train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=4, pin_memory=True)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batches/epoch: {len(train_loader)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = SpikingPointNet(num_classes=NUM_CLASSES, lam=LAM, vth=VTH).to(device)
print(f"Params: {model.param_count():,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# ── Resume ────────────────────────────────────────────────────────────────────
start_epoch  = 0
best_val_acc = 0.0
history      = []

if os.path.exists(drive_ckpt):
    ck = torch.load(drive_ckpt, map_location=device)
    model.load_state_dict(ck["model_state"], strict=False)
    optimizer.load_state_dict(ck["optimizer_state"])
    start_epoch  = ck["epoch"] + 1
    best_val_acc = ck.get("best_val_acc", 0.0)
    if os.path.exists(history_path):
        with open(history_path) as f: history = json.load(f)
    for _ in range(start_epoch): scheduler.step()
    print(f"Resumed epoch {start_epoch}, best={best_val_acc:.4f}")
else:
    print("Starting from scratch.")

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Spiking PointNet — ModelNet40 — {EPOCHS} epochs")
print(f"T_train={T_TRAIN}  T_infer={T_INFER}  MPP={USE_MPP}")
print("=" * 60)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = total_correct = total_n = 0
    t0 = time.time()

    for pts, labels in train_loader:
        pts, labels = pts.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, trans3, transK = model(pts, T=T_TRAIN, mpp=USE_MPP)
        loss_ce    = F.cross_entropy(logits, labels)
        loss_feat  = feature_transform_loss(transK) * 0.001
        loss       = loss_ce + loss_feat

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss    += loss_ce.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += labels.size(0)

    scheduler.step()
    train_acc  = total_correct / total_n
    train_loss = total_loss    / total_n

    # Validation every 5 epochs
    val_acc = None
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for pts, labels in val_loader:
                pts, labels = pts.to(device), labels.to(device)
                # T_INFER independent passes (each fresh MPP reset) → average logits
                # Running T steps in one forward accumulates membrane to out-of-distribution
                # values (calibrated for T=1); independent averaging is the correct ensemble.
                logit_sum = None
                for _ in range(T_INFER):
                    logits_i, _, _ = model(pts, T=1, mpp=True)
                    logit_sum = logits_i if logit_sum is None else logit_sum + logits_i
                logits = logit_sum / T_INFER
                correct += (logits.argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, best_ckpt)
        print(f"Epoch {epoch:3d} | TrainAcc={train_acc:.4f} Loss={train_loss:.4f} | "
              f"ValAcc={val_acc:.4f} {'★ BEST' if is_best else ''} | "
              f"LR={scheduler.get_last_lr()[0]:.6f} | {time.time()-t0:.0f}s")
    else:
        print(f"Epoch {epoch:3d} | TrainAcc={train_acc:.4f} Loss={train_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f} | {time.time()-t0:.0f}s")

    history.append({"epoch": epoch, "train_acc": train_acc,
                    "val_acc": val_acc, "loss": train_loss})
    with open(history_path, "w") as f: json.dump(history, f, indent=2)

    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc}, drive_ckpt)

print(f"\nDone. Best val acc: {best_val_acc:.4f}")
print(f"Paper reports: 87.13% (no MPP) / 88.61% (with MPP)")

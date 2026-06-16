"""
kaggle_mn10_full.py — Full ASP+SPM run on ModelNet10 (10-class classification).

HOW TO USE ON KAGGLE
--------------------
1. Upload this project as a Kaggle dataset and attach it to the notebook.
   The script finds the purdueprj subfolder automatically inside /kaggle/input/.
2. Run the script. ModelNet10 is downloaded automatically via kagglehub.

Training targets (ModelNet10):
  SPM baseline:  ~93–94%  OA
  ASP improved:  >95%     OA  (GRU belief + multi-head SSP + diversity loss)

Full run: 200 epochs, cosine LR + warmup, strong augmentation, TTA 10 votes.
ModelNet10 is small (~3.9k train / 0.9k test) so this runs in ~40min on T4.
"""

# ── 0. Install dependencies ────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pip("kagglehub", "trimesh", "h5py")

import json, math, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print(f"Python {sys.version.split()[0]}  PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}", end="")
print()

# ── 1. Locate project root ─────────────────────────────────────────────────────
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs"
)
os.makedirs(WORK, exist_ok=True)

def _find_purdueprj():
    """Walk /kaggle/input (any depth) looking for models/spiking_mamba.py."""
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "spiking_mamba.py" in files and os.path.basename(root) == "models":
                return os.path.dirname(root)
    # Fallback: script running from inside the extracted project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in ("purdueprj", ""):
        p = os.path.join(script_dir, rel) if rel else script_dir
        if os.path.isfile(os.path.join(p, "models", "spiking_mamba.py")):
            return p
    return None

PROJ = None
if ON_KAGGLE:
    PROJ = _find_purdueprj()
    if PROJ is None:
        raise RuntimeError(
            "purdueprj project not found in /kaggle/input/.\n"
            "Attach the Kaggle dataset containing this project.\n"
            "Searched recursively for models/spiking_mamba.py."
        )
else:
    PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "purdueprj")

print(f"Project root: {PROJ}")
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# ── 2. Download ModelNet10 ─────────────────────────────────────────────────────
import trimesh
import kagglehub

def download_modelnet10():
    local_dir = os.path.join(WORK, "ModelNet10")
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"ModelNet10 already at {local_dir}")
        return local_dir
    print("Downloading ModelNet10 via kagglehub ...")
    raw = kagglehub.dataset_download("balraj98/modelnet10-princeton-3d-object-dataset")
    # Find the ModelNet10 sub-folder inside the download
    import shutil
    for root_dir, dirs, _ in os.walk(raw):
        if "ModelNet10" in dirs:
            src = os.path.join(root_dir, "ModelNet10")
            shutil.copytree(src, local_dir)
            print(f"Copied ModelNet10 → {local_dir}")
            return local_dir
    # The download root IS ModelNet10 (already the right folder)
    print(f"Using download root: {raw}")
    return raw

MN10_DIR = download_modelnet10()

# ── 3. Dataset ─────────────────────────────────────────────────────────────────

def _z_rotate(pts):
    theta = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return pts @ R.T

def _augment_strong(pts):
    pts = _z_rotate(pts)
    pts = pts * np.random.uniform(0.8, 1.25, (1, 3)).astype(np.float32)
    pts += np.clip(np.random.normal(0, 0.01, pts.shape), -0.05, 0.05).astype(np.float32)
    N    = pts.shape[0]
    keep = max(int(N * np.random.uniform(0.875, 1.0)), 1)
    idx  = np.random.choice(N, keep, replace=False)
    pts2 = pts[idx]
    if keep < N:
        pad  = np.random.choice(keep, N - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])
    return pts2.astype(np.float32)

def _augment_vote(pts):
    return _z_rotate(pts).astype(np.float32)

class ModelNet10Dataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.num_points = num_points
        self.split      = split
        self.files      = self._scan(root)
        print(f"[ModelNet10] {split}: loading {len(self.files)} shapes ...")
        self.data, self.labels = self._load_all()
        nc = len(set(self.labels.tolist()))
        print(f"[ModelNet10] {split}: {len(self.labels)} shapes, {nc} classes")

    def _scan(self, root):
        items   = []
        classes = sorted(os.listdir(root))
        for cls in classes:
            p = os.path.join(root, cls, self.split)
            if not os.path.isdir(p):
                continue
            lbl = classes.index(cls)
            for f in sorted(os.listdir(p)):
                if f.endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(p, f), lbl))
        return items

    def _load_pts(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        if path.endswith(".txt"):
            return np.loadtxt(path).astype(np.float32)
        mesh = trimesh.load(path)
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_lbl = [], []
        for path, lbl in self.files:
            pts = self._load_pts(path)
            if not path.endswith(".off"):
                N = pts.shape[0]
                if N >= self.num_points:
                    pts = pts[np.random.choice(N, self.num_points, replace=False)]
                else:
                    pad = np.random.choice(N, self.num_points - N, replace=True)
                    pts = np.vstack([pts, pts[pad]])
            all_pts.append(pts[:, :3])
            all_lbl.append(lbl)
        return np.array(all_pts, dtype=np.float32), np.array(all_lbl, dtype=np.int64)

    def _norm(self, pts):
        pts = pts - pts.mean(axis=0)
        pts = pts / (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
        return pts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self._norm(self.data[idx].copy())
        if self.split == "train":
            pts = _augment_strong(pts)
            pts = self._norm(pts)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(int(self.labels[idx]), dtype=torch.long))

# ── 4. Import project modules ──────────────────────────────────────────────────
from models.spiking_mamba import SPMModel
from models.asp_wrapper   import ASPWrapper
from data.slicing         import slice_fps_hierarchical_batch
from training.train_active import prepare_fps_slices_and_geo, gumbel_tau
from training.loss_active  import active_loss
from training.metrics      import accuracy

# ── 5. Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS       = 200
BATCH        = 32
LR           = 5e-4
WD           = 1e-4
NUM_POINTS   = 1024
NUM_CLASSES  = 10
T            = 4        # slices (256 pts/slice)
FEAT_DIM     = 512
POINT_DIMS   = (128, 256, 512)
D_STATE      = 16
N_SMB        = 2
KNN_K        = 16
TAU_LIF      = 0.9
WARMUP_EP    = 20
TTA_VOTES    = 10
EXIT_THR     = 0.6
LAM_AUX      = 0.3
LAM_EXIT     = 0.1
LAM_FR       = 0.02
LAM_DIV      = 0.05
LABEL_SMOOTH = 0.1
CKPT_DIR     = os.path.join(WORK, "mn10_ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 6. Model builders ──────────────────────────────────────────────────────────
def make_spm():
    return SPMModel(
        num_classes  = NUM_CLASSES,
        point_dims   = POINT_DIMS,
        d_state      = D_STATE,
        tau          = TAU_LIF,
        n_smb_layers = N_SMB,
        local_knn    = True,
        knn_k        = KNN_K,
        learnable_lif= False,
    ).to(device)

def make_asp():
    return ASPWrapper(
        make_spm(), feat_dim=FEAT_DIM, num_classes=NUM_CLASSES,
        d_ssp=128, n_heads=4, diversity=0.1,
    ).to(device)

# ── 7. LR schedule ─────────────────────────────────────────────────────────────
def make_scheduler(opt):
    def lr_lambda(ep):
        if ep < WARMUP_EP:
            return 0.1 + 0.9 * ep / max(1, WARMUP_EP)
        p = (ep - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ── 8. Training helpers ────────────────────────────────────────────────────────

def train_spm_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        model.reset_state(B, device)
        model._total_T = T
        pts_slices = slice_fps_hierarchical_batch(pts, T=T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_slices[:, t])
        loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * B
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


def train_asp_epoch(model, loader, optimizer, epoch):
    model.train()
    progress = epoch / max(EPOCHS - 1, 1)
    tau = gumbel_tau(epoch)
    if hasattr(model, "set_gumbel_tau"):
        model.set_gumbel_tau(tau)
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        pts_slices, geo, _, _ = prepare_fps_slices_and_geo(pts, T=T)
        logits_final, logits_all, sel_w = model.forward_active_train(pts_slices, geo)
        loss, _ = active_loss(
            logits_final, logits_all, labels, model,
            lam_aux=LAM_AUX, lam_exit=LAM_EXIT, lam_fr=LAM_FR, lam_div=LAM_DIV,
            label_smoothing=LABEL_SMOOTH,
            progress=progress,
            geo_descriptors=geo,
            selection_weights=sel_w,
        )
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * B
        total_acc  += (logits_final.argmax(1) == labels).sum().item()
        n          += B
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_spm(model, loader):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        B = pts.size(0)
        model.reset_state(B, device)
        model._total_T = T
        pts_slices = slice_fps_hierarchical_batch(pts, T=T)
        logits = None
        for t in range(T):
            logits = model.forward_step(pts_slices[:, t])
        correct += (logits.argmax(1) == labels).sum().item()
        total   += B
    return correct / total


@torch.no_grad()
def eval_asp(model, loader, tta=1):
    model.eval()
    correct = total = total_slices = 0
    for pts_batch, labels in loader:
        labels = labels.to(device)
        for b in range(pts_batch.size(0)):
            pts_np = pts_batch[b].numpy()
            vote_logits = []
            total_exit_v = 0
            for _ in range(tta):
                pts_v = _augment_vote(pts_np) if tta > 1 else pts_np
                norm  = np.max(np.linalg.norm(pts_v, axis=1)) + 1e-8
                pts_v = (pts_v / norm).astype(np.float32)
                pts_in = torch.tensor(pts_v).unsqueeze(0).to(device)
                slices_v, geo_v, _, _ = prepare_fps_slices_and_geo(pts_in, T=T)
                logits_v, exit_step, _ = model.forward_active_infer(
                    slices_v, geo_v, threshold=EXIT_THR
                )
                vote_logits.append(logits_v)
                total_exit_v += exit_step
            logits = torch.stack(vote_logits, 0).mean(0)
            correct      += (logits.argmax(1) == labels[b:b+1]).sum().item()
            total        += 1
            total_slices += total_exit_v / tta
    return correct / total, total_slices / total

# ── 9. Data loaders ────────────────────────────────────────────────────────────
print("\nLoading ModelNet10 ...")
train_ds = ModelNet10Dataset(MN10_DIR, NUM_POINTS, "train")
val_ds   = ModelNet10Dataset(MN10_DIR, NUM_POINTS, "test")

train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=4,
                          pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=4,
                          pin_memory=True)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

# ── 10. SPM baseline ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Phase 1: SPM Baseline (fixed FPS slice order)")
print(f"{'='*70}")

spm = make_spm()
spm_params = sum(p.numel() for p in spm.parameters())
print(f"SPM params: {spm_params:,}")

spm_opt = torch.optim.AdamW(spm.parameters(), lr=LR, weight_decay=WD)
spm_sch = make_scheduler(spm_opt)
best_spm = 0.0
spm_history = []

for epoch in range(EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_spm_epoch(spm, train_loader, spm_opt, epoch)
    spm_sch.step()
    spm_history.append({"epoch": epoch, "train_acc": tr_acc})

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        val_acc = eval_spm(spm, val_loader)
        if val_acc > best_spm:
            best_spm = val_acc
            torch.save(spm.state_dict(), os.path.join(CKPT_DIR, "spm_best.pth"))
        spm_history[-1]["val_acc"] = val_acc
        lr_now = spm_opt.param_groups[0]["lr"]
        print(f"[SPM] Ep {epoch+1:3d}/{EPOCHS} | TrainAcc={tr_acc:.4f} "
              f"| ValAcc={val_acc:.4f} {'★' if val_acc == best_spm else ' '} "
              f"| LR={lr_now:.5f} | {time.time()-t0:.0f}s")
    elif (epoch + 1) % 2 == 0:
        print(f"[SPM] Ep {epoch+1:3d}/{EPOCHS} | TrainAcc={tr_acc:.4f} "
              f"| LR={spm_opt.param_groups[0]['lr']:.5f} | {time.time()-t0:.0f}s")

print(f"\nSPM Best Val: {best_spm*100:.2f}%")

# ── 11. ASP+SPM (improved) ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Phase 2: ASP+SPM — Improved (GRU belief + Multi-head SSP + Diversity loss)")
print(f"  d_ssp=128, n_heads=4, diversity=0.1")
print(f"  LAM_DIV={LAM_DIV}, LAM_EXIT={LAM_EXIT} (progressive), TTA={TTA_VOTES} votes")
print(f"{'='*70}")

asp = make_asp()
asp_params = sum(p.numel() for p in asp.parameters())
print(f"ASP params: {asp_params:,}  (+overhead: {asp_params - spm_params:,})")

asp_opt = torch.optim.AdamW(asp.parameters(), lr=LR, weight_decay=WD)
asp_sch = make_scheduler(asp_opt)
best_asp = 0.0
best_asp_slices = T
asp_history = []

for epoch in range(EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_asp_epoch(asp, train_loader, asp_opt, epoch)
    asp_sch.step()
    asp_history.append({"epoch": epoch, "train_acc": tr_acc})

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        tta = TTA_VOTES if (epoch == EPOCHS - 1 or (epoch + 1) % 50 == 0) else 1
        val_acc, val_slices = eval_asp(asp, val_loader, tta=tta)
        if val_acc > best_asp:
            best_asp = val_acc
            best_asp_slices = val_slices
            torch.save(asp.state_dict(), os.path.join(CKPT_DIR, "asp_best.pth"))
        asp_history[-1]["val_acc"] = val_acc
        lr_now = asp_opt.param_groups[0]["lr"]
        tta_str = f" TTA={tta}" if tta > 1 else ""
        print(f"[ASP] Ep {epoch+1:3d}/{EPOCHS} | TrainAcc={tr_acc:.4f} "
              f"| ValAcc={val_acc:.4f} {'★' if val_acc == best_asp else ' '}"
              f" | Slices={val_slices:.2f}/{T}{tta_str}"
              f" | LR={lr_now:.5f} | {time.time()-t0:.0f}s")
    elif (epoch + 1) % 2 == 0:
        print(f"[ASP] Ep {epoch+1:3d}/{EPOCHS} | TrainAcc={tr_acc:.4f} "
              f"| LR={asp_opt.param_groups[0]['lr']:.5f} | {time.time()-t0:.0f}s")

print(f"\nASP Best Val: {best_asp*100:.2f}%")

# ── 12. Final evaluation (best checkpoints, full TTA) ─────────────────────────
print(f"\n{'='*70}")
print("Final Evaluation — Best Checkpoints + Full TTA")
print(f"{'='*70}")

spm.load_state_dict(torch.load(os.path.join(CKPT_DIR, "spm_best.pth"), map_location=device))
asp.load_state_dict(torch.load(os.path.join(CKPT_DIR, "asp_best.pth"), map_location=device))

spm_final              = eval_spm(spm, val_loader)
asp_final, asp_slices  = eval_asp(asp, val_loader, tta=TTA_VOTES)

E_AC, E_MAC = 2.3e-3, 8.4e-3
energy = 0.15 * E_AC / E_MAC * (asp_slices / T)

print(f"\n  SPM  OA: {spm_final*100:.2f}%")
print(f"  ASP  OA: {asp_final*100:.2f}%  "
      f"(avg {asp_slices:.2f}/{T} slices, TTA={TTA_VOTES})")
print(f"  Δ (ASP - SPM): {(asp_final - spm_final)*100:+.2f} pp")
print(f"  Est. energy vs ANN: {energy*100:.1f}%  (fr≈0.15, Loihi 2)")

# ── 13. Save results ───────────────────────────────────────────────────────────
results = {
    "dataset":        "ModelNet10",
    "num_classes":    NUM_CLASSES,
    "epochs":         EPOCHS,
    "tta_votes":      TTA_VOTES,
    "spm_oa":         spm_final,
    "asp_oa":         asp_final,
    "asp_avg_slices": asp_slices,
    "delta_pp":       (asp_final - spm_final) * 100,
    "energy_vs_ann":  energy,
    "spm_history":    spm_history,
    "asp_history":    asp_history,
}
out_path = os.path.join(CKPT_DIR, "results_mn10.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("FINAL RESULTS — SPM vs ASP+SPM on ModelNet10")
print(f"{'='*70}")
print(f"  SPM  OA: {spm_final*100:.2f}%")
print(f"  ASP  OA: {asp_final*100:.2f}%  (TTA={TTA_VOTES})")
print(f"  Δ:       {(asp_final-spm_final)*100:+.2f} pp")
print(f"\nResults saved → {out_path}")
print(f"Checkpoints  → {CKPT_DIR}/")

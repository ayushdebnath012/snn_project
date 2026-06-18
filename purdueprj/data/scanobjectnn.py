"""
scanobjectnn.py
===============
Data loader for ScanObjectNN — a harder real-world point cloud dataset
(objects scanned from real environments, with background clutter).

Reference: Uy et al. "Revisiting Point Cloud Classification: A New Benchmark
Dataset and Classification Model on Real-World Data", ICCV 2019.

Three standard variants:
  OBJ_BG     — objects WITH background points  (hardest background)
  OBJ_ONLY   — objects WITHOUT background       (clean)
  PB_T50_RS  — perturbed+rotated (hardest overall, used in most papers)

Expected directory layout (download from official source):
  <root>/
    main_split/
      training_objectdataset.h5           ← OBJ_ONLY train
      test_objectdataset.h5               ← OBJ_ONLY test
    main_split_nobg/
      training_objectdataset.h5           ← OBJ_BG train  (confusingly named)
      test_objectdataset.h5               ← OBJ_BG test
    main_split/
      training_objectdataset_augmented25rot.h5   ← PB_T50_RS train
      test_objectdataset_augmented25rot.h5        ← PB_T50_RS test

15 classes, ~15k train / ~2.9k test samples.
Each sample: 2048 points × (x, y, z).

Usage:
  ds = ScanObjectNNDataset(root="/data/ScanObjectNN",
                           variant="PB_T50_RS", split="train",
                           num_points=1024)
  loader = DataLoader(ds, batch_size=32, shuffle=True)
  pts, label = ds[0]  # pts: [1024, 3], label: int
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# File name mapping per variant and split
# ---------------------------------------------------------------------------

_VARIANT_FILES = {
    "OBJ_ONLY": {
        "train": os.path.join("main_split", "training_objectdataset.h5"),
        "test":  os.path.join("main_split", "test_objectdataset.h5"),
    },
    "OBJ_BG": {
        "train": os.path.join("main_split_nobg", "training_objectdataset.h5"),
        "test":  os.path.join("main_split_nobg", "test_objectdataset.h5"),
    },
    "PB_T50_RS": {
        "train": os.path.join("main_split",
                              "training_objectdataset_augmented25rot.h5"),
        "test":  os.path.join("main_split",
                              "test_objectdataset_augmented25rot.h5"),
    },
}

NUM_CLASSES = 15  # ScanObjectNN has 15 object categories


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ScanObjectNNDataset(Dataset):
    """
    ScanObjectNN point cloud dataset.

    Args:
        root        : path to the ScanObjectNN root directory
        variant     : one of "OBJ_BG", "OBJ_ONLY", "PB_T50_RS"
        split       : "train" or "test"
        num_points  : number of points to sample per cloud (default 1024)
        augment     : if True, apply random jitter + random flip during training
    """

    def __init__(self, root, variant="PB_T50_RS", split="train",
                 num_points=1024, augment=False):
        assert variant in _VARIANT_FILES, \
            f"variant must be one of {list(_VARIANT_FILES.keys())}"
        assert split in ("train", "test")

        self.num_points = num_points
        self.augment    = augment and (split == "train")

        h5_path = os.path.join(root, _VARIANT_FILES[variant][split])

        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"ScanObjectNN h5 file not found: {h5_path}\n"
                f"Download from: https://hkust-vgd.github.io/scanobjectnn/"
            )

        self.data, self.labels = self._load_h5(h5_path)
        print(f"[ScanObjectNN] {variant}/{split}: "
              f"{len(self.labels)} samples, {NUM_CLASSES} classes  "
              f"(h5: {os.path.basename(h5_path)})")

    # ------------------------------------------------------------------

    def _load_h5(self, path):
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for ScanObjectNN. Install: pip install h5py"
            )

        with h5py.File(path, "r") as f:
            data   = np.array(f["data"],  dtype=np.float32)   # [N, 2048, 3]
            labels = np.array(f["label"], dtype=np.int64).squeeze()  # [N]

        return data, labels

    # ------------------------------------------------------------------

    def _sample(self, pts):
        """Sub-sample or pad to self.num_points."""
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            idx = np.concatenate([
                np.arange(N),
                np.random.choice(N, self.num_points - N, replace=True)
            ])
        return pts[idx]

    def _augment(self, pts):
        """Random jitter + random x-axis flip."""
        pts = pts + np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        if np.random.rand() > 0.5:
            pts[:, 0] = -pts[:, 0]
        return pts

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts   = self._sample(self.data[idx])   # [num_points, 3]
        label = int(self.labels[idx])

        # Unit-sphere normalisation (centre + scale) — critical for BN-LIF stability
        pts = pts - pts.mean(axis=0)
        pts = pts / (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

        if self.augment:
            pts = self._augment(pts)

        # Random shuffle (like ModelNetDataset)
        np.random.shuffle(pts)

        return torch.tensor(pts, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Convenience factory — mirrors get_loaders() pattern in run_all_experiments.py
# ---------------------------------------------------------------------------

def get_scanobjectnn_loaders(root, variant="PB_T50_RS",
                              batch_size=32, num_points=1024,
                              num_workers=2):
    """
    Returns (train_loader, test_loader, NUM_CLASSES=15).
    Falls back to DummyDataset if root is missing.
    """
    from torch.utils.data import DataLoader

    try:
        if root and os.path.isdir(root):
            tr = ScanObjectNNDataset(root, variant=variant, split="train",
                                     num_points=num_points, augment=True)
            va = ScanObjectNNDataset(root, variant=variant, split="test",
                                     num_points=num_points)
            train_l = DataLoader(tr, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True)
            val_l   = DataLoader(va, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
            return train_l, val_l, NUM_CLASSES
    except Exception as e:
        print(f"[ScanObjectNN] Load failed: {e}. Using dummy data.")

    # Fallback: dummy
    from torch.utils.data import TensorDataset
    import torch
    dummy_tr = TensorDataset(torch.randn(512,  num_points, 3), torch.randint(0, 15, (512,)))
    dummy_va = TensorDataset(torch.randn(128,  num_points, 3), torch.randint(0, 15, (128,)))
    from torch.utils.data import DataLoader
    return (DataLoader(dummy_tr, batch_size=batch_size, shuffle=True),
            DataLoader(dummy_va, batch_size=batch_size, shuffle=False),
            NUM_CLASSES)

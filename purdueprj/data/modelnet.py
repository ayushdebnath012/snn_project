import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import trimesh
except Exception:  # pragma: no cover - only needed for OFF meshes.
    trimesh = None


def normalize_points(pts):
    """Center a point cloud and scale it to the unit sphere."""
    pts = pts.astype(np.float32, copy=True)
    pts -= pts.mean(axis=0, keepdims=True)
    radius = np.max(np.linalg.norm(pts, axis=1))
    pts /= radius + 1e-8
    return pts


def _rotation_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _random_z_rotate(pts, rng):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return pts @ _rotation_matrix("z", angle).T


def _random_small_tilt(pts, rng, max_angle=0.10):
    ax = rng.uniform(-max_angle, max_angle)
    ay = rng.uniform(-max_angle, max_angle)
    return pts @ _rotation_matrix("x", ax).T @ _rotation_matrix("y", ay).T


def _drop_and_resample(pts, rng, keep_min=0.875):
    n = pts.shape[0]
    keep = max(int(n * rng.uniform(keep_min, 1.0)), 1)
    idx = rng.choice(n, keep, replace=False)
    pts = pts[idx]
    if keep < n:
        pad = rng.choice(keep, n - keep, replace=True)
        pts = np.vstack([pts, pts[pad]])
    return pts


def _jitter(pts, rng, sigma=0.01, clip=0.05):
    noise = rng.normal(0.0, sigma, pts.shape).astype(np.float32)
    return pts + np.clip(noise, -clip, clip)


def _point_wolf_light(pts, rng, num_anchors=4, sigma=0.12):
    """Lightweight local elastic warp; useful late in training, but kept mild."""
    n = pts.shape[0]
    if n < num_anchors:
        return pts
    anchor_idx = rng.choice(n, num_anchors, replace=False)
    anchors = pts[anchor_idx]
    displace = rng.normal(0.0, sigma, (num_anchors, 3)).astype(np.float32)
    diff = pts[:, None, :] - anchors[None, :, :]
    dist2 = (diff ** 2).sum(-1)
    weights = np.exp(-dist2 / (2.0 * sigma ** 2))
    weights = weights / (weights.sum(-1, keepdims=True) + 1e-8)
    return pts + weights @ displace


def augment_point_cloud(pts, mode="baseline", rng=np.random, normalize_after=False):
    """
    Numeric point-cloud augmentation for ModelNet training and TTA.

    Modes:
      none      no augmentation
      baseline  dropout/resample + isotropic scale + shift
      strong    z-rotation, anisotropic scale, dropout, tilt, jitter
      elastic   strong + mild local PointWOLF-style warp
      vote      mild stochastic transform for test-time voting
    """
    mode = "none" if mode is None else str(mode).lower()
    pts = pts.astype(np.float32, copy=True)

    if mode in ("none", "false", "0"):
        return normalize_points(pts) if normalize_after else pts

    if mode in ("baseline", "standard"):
        pts = _drop_and_resample(pts, rng, keep_min=0.875)
        pts *= np.float32(rng.uniform(0.8, 1.25))
        pts += rng.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)
    elif mode in ("strong", "sota", "modelnet40", "elastic"):
        pts = _drop_and_resample(pts, rng, keep_min=0.80)
        pts = _random_z_rotate(pts, rng)
        if rng.random() < 0.35:
            pts = _random_small_tilt(pts, rng, max_angle=0.12)
        pts *= rng.uniform(0.80, 1.25, size=(1, 3)).astype(np.float32)
        pts += rng.uniform(-0.10, 0.10, size=(1, 3)).astype(np.float32)
        pts = _jitter(pts, rng, sigma=0.01, clip=0.05)
        if mode == "elastic" and rng.random() < 0.35:
            pts = _point_wolf_light(pts, rng, num_anchors=4, sigma=0.10)
    elif mode in ("vote", "tta"):
        pts = _drop_and_resample(pts, rng, keep_min=0.90)
        pts = _random_z_rotate(pts, rng)
        pts *= rng.uniform(0.95, 1.05, size=(1, 3)).astype(np.float32)
        pts = _jitter(pts, rng, sigma=0.005, clip=0.02)
        normalize_after = True
    else:
        raise ValueError(f"Unknown ModelNet augmentation mode: {mode}")

    if normalize_after:
        pts = normalize_points(pts)
    rng.shuffle(pts)
    return pts.astype(np.float32, copy=False)


# Backwards compatibility for old scripts importing _augment directly.
def _augment(pts):
    return augment_point_cloud(pts, mode="baseline")


class ModelNetDataset(Dataset):
    def __init__(
        self,
        root,
        num_points=1024,
        split="train",
        aug_mode=None,
        augment=None,
        normalize=True,
        return_index=False,
    ):
        """
        aug_mode/augment:
          baseline  original dropout + uniform scale + shift
          strong    accuracy-first numeric augmentation for ModelNet40
          elastic   strong plus mild local warping
          none      no augmentation
        """
        self.root = root
        self.num_points = num_points
        self.split = split
        self.normalize = normalize
        self.return_index = return_index
        self.aug_mode = augment if augment is not None else aug_mode
        if self.aug_mode is None:
            self.aug_mode = "baseline" if split == "train" else "none"

        self.class_names = sorted(
            d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))
        )
        self.files = self._scan_files()
        self.data, self.labels = self._load_all()

    def _scan_files(self):
        items = []
        label_map = {class_name: i for i, class_name in enumerate(self.class_names)}
        for class_name in self.class_names:
            class_path = os.path.join(self.root, class_name, self.split)
            if not os.path.isdir(class_path):
                continue
            label = label_map[class_name]
            for f in os.listdir(class_path):
                if f.endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(class_path, f), label))
        return items

    def _load_points(self, path):
        if path.endswith(".npy"):
            pts = np.load(path).astype(np.float32)
        elif path.endswith(".txt"):
            pts = np.loadtxt(path).astype(np.float32)
        elif path.endswith(".off"):
            pts = self._load_off_points(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        return pts[:, :3]

    def _load_off_points(self, path):
        if trimesh is not None:
            mesh = trimesh.load(path)
            points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            return points.astype(np.float32)

        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        start = 1
        if lines[0].startswith("OFF") and lines[0] != "OFF":
            lines[0] = lines[0][3:].strip()
            start = 0
        n_verts = int(lines[start].split()[0])
        verts = np.array(
            [[float(v) for v in lines[start + 1 + i].split()[:3]] for i in range(n_verts)],
            dtype=np.float32,
        )
        idx = np.random.choice(n_verts, self.num_points, replace=n_verts < self.num_points)
        return verts[idx]

    def _load_all(self):
        all_pts, all_labels = [], []
        for path, label in self.files:
            pts = self._load_points(path)
            if not path.endswith(".off"):
                if pts.shape[0] >= self.num_points:
                    idx = np.random.choice(pts.shape[0], self.num_points, replace=False)
                    pts = pts[idx]
                else:
                    pad = self.num_points - pts.shape[0]
                    repeat_idx = np.random.choice(pts.shape[0], pad, replace=True)
                    pts = np.vstack([pts, pts[repeat_idx]])
            all_pts.append(pts)
            all_labels.append(label)
        return np.asarray(all_pts, dtype=np.float32), np.asarray(all_labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        if self.normalize:
            pts = normalize_points(pts)
        if self.split == "train" and self.aug_mode not in (None, "none", "false", "0"):
            pts = augment_point_cloud(pts, mode=self.aug_mode)
        np.random.shuffle(pts)
        item = (
            torch.tensor(pts, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
        if self.return_index:
            return (*item, idx)
        return item

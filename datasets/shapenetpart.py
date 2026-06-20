"""
datasets/shapenetpart.py — ShapeNetPart HDF5 dataset for part segmentation.

16 categories, 50 part labels, 14007 train / 2874 test shapes.
Each shape has 2048 points with global part labels in [0, 49].

Returns per sample:
    slices       [M, K, 6]   sliced point cloud (xyz + normals)
    geo          [M, 8]      geometry descriptors
    pts_features [N, 6]      normalised xyz + normals for point branch
    sid_arr      [N]          slice assignment per point
    part_labels  [N]          ground truth part ids (0-49)
    cat_id       int          category index (0-15)
"""

import glob
import json
import os
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices, compute_geo
from .transforms import augment_seg


# ── Category and part definitions ──────────────────────────────────────

CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
    'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',
    'Rocket', 'Skateboard', 'Table',
]

# Part label ranges per category (global labels, 0-indexed)
CATEGORY_TO_PARTS = {
    0:  [0, 1, 2, 3],          # Airplane
    1:  [4, 5],                 # Bag
    2:  [6, 7],                 # Cap
    3:  [8, 9, 10, 11],        # Car
    4:  [12, 13, 14, 15],      # Chair
    5:  [16, 17, 18],          # Earphone
    6:  [19, 20, 21],          # Guitar
    7:  [22, 23],              # Knife
    8:  [24, 25, 26, 27],      # Lamp
    9:  [28, 29],              # Laptop
    10: [30, 31, 32, 33, 34, 35],  # Motorbike
    11: [36, 37],              # Mug
    12: [38, 39, 40],          # Pistol
    13: [41, 42, 43],          # Rocket
    14: [44, 45, 46],          # Skateboard
    15: [47, 48, 49],          # Table
}

NUM_PARTS = 50
NUM_CATEGORIES = 16


class ShapeNetPartDataset(Dataset):

    def __init__(self, data_dir: str, split: str, cfg=None):
        """
        Args:
            data_dir: path to shapenet_part_seg_hdf5_data/
            split:    'train', 'val', or 'test'
            cfg:      config object with slicing/aug parameters
        """
        assert split in ('train', 'val', 'test')
        self.split = split
        self.cfg = cfg
        self.n_points = getattr(cfg, 'num_points', 2048)

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        # Discover h5 files dynamically (train0.h5 ... trainN.h5)
        pattern = os.path.join(data_dir, f"{split}*.h5")
        h5_files = sorted(glob.glob(pattern))
        if not h5_files:
            raise FileNotFoundError(
                f"No {split}*.h5 files found in {data_dir}. "
                f"Run: python datasets/download.py --shapenet"
            )
        manifest_count = None
        manifest_path = os.path.join(data_dir, "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as manifest_file:
                manifest = json.load(manifest_file)
            split_info = manifest.get("splits", {}).get(split)
            if split_info is None:
                raise ValueError(
                    f"{manifest_path} does not describe split {split!r}"
                )
            expected = sorted(split_info.get("files", []))
            manifest_count = int(split_info.get("count", -1))
            actual = sorted(os.path.basename(path) for path in h5_files)
            if actual != expected:
                raise ValueError(
                    f"{split} HDF5 files do not match dataset_manifest.json. "
                    f"Expected {expected}, found {actual}. Remove stale shards "
                    "and reconvert."
                )

        self.h5_files = h5_files
        self.file_lengths = []
        all_cat = []
        channels = set()
        for path in h5_files:
            with h5py.File(path, 'r') as f:
                point_shape = f['data'].shape
                cats = f['label'][:].astype(np.int64)
                pids = f['pid'][:].astype(np.int64)
            self._validate_shard(path, point_shape, cats, pids)
            channels.add(point_shape[-1])
            self.file_lengths.append(point_shape[0])
            all_cat.append(cats)

        if len(channels) != 1:
            raise ValueError(
                f"Mixed ShapeNetPart feature widths found in {data_dir}: "
                f"{sorted(channels)}. Remove stale HDF5 shards and reconvert."
            )

        cats = np.concatenate(all_cat, axis=0)
        self.cats = cats.squeeze(-1) if cats.ndim == 2 else cats
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        if manifest_count is not None and len(self.cats) != manifest_count:
            raise ValueError(
                f"{split} contains {len(self.cats)} shapes but the manifest "
                f"declares {manifest_count}"
            )
        self.feature_channels = next(iter(channels))
        self.has_normals = self.feature_channels >= 6
        self._handles = {}
        if getattr(cfg, "use_normals", True) and not self.has_normals:
            print(
                f"[ShapeNetPart] '{split}' has xyz only; normal channels will "
                "be zero. Reconvert the raw *_normal dataset for best accuracy."
            )

        print(f"[ShapeNetPart] '{split}': {len(self)} shapes, "
              f"{NUM_CATEGORIES} categories, {NUM_PARTS} parts")

    @staticmethod
    def _validate_shard(path, point_shape, cats, pids):
        """Fail before training if a shard mixes categories and part ids."""
        if len(point_shape) != 3 or point_shape[-1] not in (3, 6):
            raise ValueError(f"{path}: expected data [B,N,3|6], got {point_shape}")
        cats_flat = cats.reshape(-1)
        if len(cats_flat) != point_shape[0] or pids.shape != point_shape[:2]:
            raise ValueError(
                f"{path}: inconsistent data/label/pid shapes "
                f"{point_shape}, {cats.shape}, {pids.shape}"
            )
        for cat in np.unique(cats_flat):
            cat = int(cat)
            if cat not in CATEGORY_TO_PARTS:
                raise ValueError(f"{path}: invalid category id {cat}")
            observed = set(map(int, np.unique(pids[cats_flat == cat])))
            valid = set(CATEGORY_TO_PARTS[cat])
            if not observed.issubset(valid):
                raise ValueError(
                    f"{path}: {CATEGORY_NAMES[cat]} contains invalid part ids "
                    f"{sorted(observed - valid)}. This usually means stale or "
                    "incompatible HDF5 shards were mixed; reconvert into an "
                    "empty output directory."
                )

    def __len__(self):
        return int(self.cumulative_lengths[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def _get_file(self, file_idx):
        import h5py
        handle = self._handles.get(file_idx)
        if handle is None:
            handle = h5py.File(self.h5_files[file_idx], "r")
            self._handles[file_idx] = handle
        return handle

    def _locate(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        previous = 0 if file_idx == 0 else int(self.cumulative_lengths[file_idx - 1])
        return file_idx, idx - previous

    def _normalise(self, pts):
        """Centre and scale to unit sphere."""
        pts = pts - pts.mean(axis=0)
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts = pts / scale
        return pts.astype(np.float32)

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        handle = self._get_file(file_idx)
        cat_id = int(self.cats[idx])
        part_labels = handle["pid"][local_idx, :self.n_points].astype(np.int64)

        # Normalise xyz and preserve normals when present.
        raw = handle["data"][local_idx, :self.n_points].astype(np.float32)
        raw_xyz = raw[:, :3]
        pts_n = self._normalise(raw_xyz)
        if self.has_normals and getattr(self.cfg, "use_normals", True):
            normals = raw[:, 3:6].astype(np.float32)
            norm = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(norm, 1e-12)
        else:
            normals = np.zeros((len(pts_n), 3), dtype=np.float32)
        pts6 = np.concatenate([pts_n, normals], axis=1)

        # Slice
        M = getattr(self.cfg, 'num_slices', 16)
        K = getattr(self.cfg, 'points_per_slice', 128)
        # Deterministic FPS at test time for reproducible metrics
        fps_seed = idx if self.split != 'train' else None
        slices, geo, anchor_xyz = slice_point_cloud(pts6, M, K, seed=fps_seed)

        # Assign each point to nearest slice
        sid_arr = assign_points_to_slices(pts_n, anchor_xyz)

        # Augment (training only) — shared transform on slices + pts
        if self.split == 'train' and self.cfg is not None:
            slices, pts6, part_labels, sid_arr = augment_seg(
                slices, pts6, self.cfg, part_labels, sid_arr
            )
            # P0 FIX: Recompute geo from augmented slices so positional encoding
            # and SSP see the same geometry as the encoder.
            # Without this, train and test see different (geo, points) distributions.
            geo = np.stack([compute_geo(s) for s in slices])

        return (
            slices.astype(np.float32),        # [M, K, 6]
            geo.astype(np.float32),           # [M, 8]
            pts6.astype(np.float32),          # [N, 6]
            sid_arr.astype(np.int64),         # [N]
            part_labels,                      # [N]
            cat_id,                           # int
        )

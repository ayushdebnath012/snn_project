"""
datasets/convert_shapenet_raw.py — Convert raw ShapeNetPart (PartAnnotation)
to the HDF5 format our loader expects.

Input format (Kaggle "shapenetcore_partanno_segmentation_benchmark_v0_normal"):
    raw_dir/
        synsetoffset2category.txt   # category_name <tab> synset_id
        02691156/                   # synset id (Airplane)
            points/*.pts            # text files: x y z [nx ny nz]
            points_label/*.seg      # text files: local part labels
        02773838/                   # synset id (Bag)
            points/*.pts
            points_label/*.seg
        ...
        train_test_split/
            shuffled_train_file_list.json
            shuffled_val_file_list.json
            shuffled_test_file_list.json

Output format (matches Stanford HDF5 download):
    out_dir/
        train0.h5 .. trainN.h5
        val0.h5 .. valN.h5
        test0.h5 .. testM.h5
    Each h5 contains:
        data:  [num_shapes, 2048, 6]   float32  (xyz + normals)
        label: [num_shapes, 1]         int64  (0-15 category)
        pid:   [num_shapes, 2048]      int64  (0-49 global part)
        all_object_categories.txt
        dataset_manifest.json

Usage:
    python datasets/convert_shapenet_raw.py \
        --raw_dir data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --out_dir data/shapenet_part_seg_hdf5_data
"""

import argparse
import glob
import json
import os

import numpy as np


# Standard 16 ShapeNetPart categories (synset id → category index 0-15)
SYNSET_TO_CAT = {
    "02691156": 0,   # Airplane
    "02773838": 1,   # Bag
    "02954340": 2,   # Cap
    "02958343": 3,   # Car
    "03001627": 4,   # Chair
    "03261776": 5,   # Earphone
    "03467517": 6,   # Guitar
    "03624134": 7,   # Knife
    "03636649": 8,   # Lamp
    "03642806": 9,   # Laptop
    "03790512": 10,  # Motorbike
    "03797390": 11,  # Mug
    "03948459": 12,  # Pistol
    "04099429": 13,  # Rocket
    "04225987": 14,  # Skateboard
    "04379243": 15,  # Table
}

CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
    'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',
    'Rocket', 'Skateboard', 'Table',
]

# Per-category part label offsets — global part label is offset + local part
PART_OFFSET = {
    0: 0,   1: 4,   2: 6,   3: 8,   4: 12,  5: 16,  6: 19,  7: 22,
    8: 24,  9: 28,  10: 30, 11: 36, 12: 38, 13: 41, 14: 44, 15: 47,
}
CAT_TO_GLOBAL_PARTS = {
    0:  [0, 1, 2, 3],
    1:  [4, 5],
    2:  [6, 7],
    3:  [8, 9, 10, 11],
    4:  [12, 13, 14, 15],
    5:  [16, 17, 18],
    6:  [19, 20, 21],
    7:  [22, 23],
    8:  [24, 25, 26, 27],
    9:  [28, 29],
    10: [30, 31, 32, 33, 34, 35],
    11: [36, 37],
    12: [38, 39, 40],
    13: [41, 42, 43],
    14: [44, 45, 46],
    15: [47, 48, 49],
}

NUM_POINTS_PER_SHAPE = 2048
SHAPES_PER_H5 = 2048  # how many shapes per output h5 file
SPLIT_FILES = {
    "train": "shuffled_train_file_list.json",
    "val": "shuffled_val_file_list.json",
    "test": "shuffled_test_file_list.json",
}


def validate_raw_dir(raw_dir: str):
    """Fail early if the extracted raw ShapeNetPart folder is incomplete."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    cat_file = os.path.join(raw_dir, "synsetoffset2category.txt")
    if not os.path.exists(cat_file):
        raise FileNotFoundError(
            f"Missing synsetoffset2category.txt in {raw_dir}. "
            "Did you pass the extracted shapenetcore_partanno_* folder?"
        )

    split_dir = os.path.join(raw_dir, "train_test_split")
    missing = [
        fname for fname in SPLIT_FILES.values()
        if not os.path.exists(os.path.join(split_dir, fname))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing ShapeNetPart split JSON(s): "
            + ", ".join(os.path.join("train_test_split", f) for f in missing)
            + "\nThe converter needs the raw ShapeNetPart/PartAnnotation "
              "archive, not a ShapeNetCore-only or partial Kaggle dataset. "
              "Expected Kaggle mirror: mitkir/shapenet."
        )


def load_shape(pts_path: str, seg_path: str = None) -> tuple:
    """
    Load a .pts file from PartAnnotation.

    Official ShapeNetPart stores xyz/normals in points/*.pts and labels in
    points_label/*.seg. Some mirrors inline the label as the final pts column;
    that layout is supported as a fallback.
    """
    data = np.loadtxt(pts_path).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    xyz = data[:, :3]
    if data.shape[1] >= 6:
        normals = data[:, 3:6]
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norm, 1e-12)
    else:
        normals = np.zeros_like(xyz)
    points = np.concatenate([xyz, normals], axis=1).astype(np.float32)

    if seg_path is not None and os.path.exists(seg_path):
        part = np.loadtxt(seg_path).astype(np.int64)
        if part.ndim > 1:
            part = part.reshape(-1)
        if len(part) != len(xyz):
            raise ValueError(
                f"Point/label length mismatch: {pts_path} has {len(xyz)} "
                f"points but {seg_path} has {len(part)} labels"
            )
    elif data.shape[1] == 4:
        part = data[:, 3].astype(np.int64)
    elif data.shape[1] >= 7:
        part = data[:, -1].astype(np.int64)
    else:
        raise ValueError(
            f"No part labels found for {pts_path}. Expected matching "
            "points_label/*.seg or an inline label column."
        )
    return points, part.astype(np.int64), seg_path is not None


def resample(points: np.ndarray, part: np.ndarray, n: int) -> tuple:
    """Sample exactly n points (with replacement if needed)."""
    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
    else:
        idx = np.random.choice(len(points), n, replace=True)
    return points[idx], part[idx]


def globalize_parts(part: np.ndarray, cat_idx: int,
                    labels_are_local: bool) -> np.ndarray:
    """Convert local 1-indexed labels to global ShapeNetPart part ids."""
    valid_parts = set(CAT_TO_GLOBAL_PARTS[cat_idx])
    observed_parts = set(map(int, np.unique(part)))

    if labels_are_local:
        if part.min() < 1 or part.max() > len(valid_parts):
            raise ValueError(
                f"local labels [{part.min()},{part.max()}] are invalid for "
                f"{CATEGORY_NAMES[cat_idx]} ({len(valid_parts)} parts)"
            )
        part_global = part - 1 + PART_OFFSET[cat_idx]
    elif observed_parts.issubset(valid_parts):
        part_global = part
    elif part.min() >= 1 and part.max() <= len(valid_parts):
        part_global = part - 1 + PART_OFFSET[cat_idx]
    else:
        raise ValueError(
            f"labels [{part.min()},{part.max()}] are neither valid global "
            f"nor local ids for {CATEGORY_NAMES[cat_idx]}"
        )

    converted = set(map(int, np.unique(part_global)))
    if not converted.issubset(valid_parts):
        raise ValueError(
            f"converted labels {sorted(converted)} do not belong to "
            f"{CATEGORY_NAMES[cat_idx]}: {sorted(valid_parts)}"
        )
    return part_global.astype(np.int64)


def load_split_list(raw_dir: str, split: str) -> list:
    """Parse the train_test_split JSON file."""
    fname = SPLIT_FILES[split]
    path = os.path.join(raw_dir, "train_test_split", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    with open(path) as f:
        items = json.load(f)
    # Items look like "shape_data/02691156/abcd1234"
    parsed = []
    for item in items:
        item = item.replace("shape_data/", "").strip()
        if "/" not in item:
            continue
        synset, name = item.split("/", 1)
        if synset not in SYNSET_TO_CAT:
            continue
        name = os.path.splitext(name)[0]
        parsed.append((synset, name))
    return parsed


def find_shape_files(raw_dir: str, synset: str, name: str) -> tuple:
    """Locate the point file and optional segmentation-label file."""
    candidates = [
        os.path.join(raw_dir, synset, "points", f"{name}.pts"),
        os.path.join(raw_dir, synset, "points", f"{name}.txt"),
        os.path.join(raw_dir, "shape_data", synset, "points", f"{name}.pts"),
        os.path.join(raw_dir, "shape_data", synset, "points", f"{name}.txt"),
        os.path.join(raw_dir, synset, f"{name}.pts"),
        os.path.join(raw_dir, synset, f"{name}.txt"),
    ]
    pts_path = None
    for c in candidates:
        if os.path.exists(c):
            pts_path = c
            break
    if pts_path is None:
        return None, None

    seg_candidates = [
        os.path.join(raw_dir, synset, "points_label", f"{name}.seg"),
        os.path.join(raw_dir, synset, "points_label", f"{name}.txt"),
        os.path.join(raw_dir, "shape_data", synset, "points_label", f"{name}.seg"),
        os.path.join(raw_dir, "shape_data", synset, "points_label", f"{name}.txt"),
        os.path.splitext(pts_path)[0] + ".seg",
    ]
    seg_path = next((c for c in seg_candidates if os.path.exists(c)), None)
    return pts_path, seg_path


def write_h5_chunk(out_dir: str, prefix: str, idx: int,
                   data: np.ndarray, label: np.ndarray, pid: np.ndarray):
    """Write one h5 chunk."""
    import h5py
    path = os.path.join(out_dir, f"{prefix}{idx}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
        f.create_dataset("pid", data=pid, compression="gzip")
    print(f"  wrote {path}  ({len(data)} shapes)")


def clean_generated_outputs(out_dir: str):
    """Remove only files generated by this converter."""
    os.makedirs(out_dir, exist_ok=True)
    stale = []
    for pattern in ("train*.h5", "val*.h5", "test*.h5"):
        stale.extend(glob.glob(os.path.join(out_dir, pattern)))
    for name in ("all_object_categories.txt", "dataset_manifest.json"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            stale.append(path)
    for path in sorted(set(stale)):
        os.remove(path)
    if stale:
        print(f"  removed {len(set(stale))} stale generated files")


def validate_h5_output(out_dir: str, expected_counts: dict) -> dict:
    """Validate shapes and category-aware part ids in every output shard."""
    import h5py

    manifest = {"version": 2, "channels": 6, "splits": {}}
    for split, expected_count in expected_counts.items():
        files = sorted(glob.glob(os.path.join(out_dir, f"{split}*.h5")))
        count = 0
        for path in files:
            with h5py.File(path, "r") as f:
                data = f["data"][:]
                cats = f["label"][:].reshape(-1)
                pids = f["pid"][:]
            if data.ndim != 3 or data.shape[1:] != (NUM_POINTS_PER_SHAPE, 6):
                raise ValueError(f"invalid data shape in {path}: {data.shape}")
            if len(cats) != len(data) or pids.shape != data.shape[:2]:
                raise ValueError(f"inconsistent dataset shapes in {path}")
            for cat_idx in np.unique(cats):
                cat_idx = int(cat_idx)
                if cat_idx not in CAT_TO_GLOBAL_PARTS:
                    raise ValueError(f"invalid category {cat_idx} in {path}")
                observed = set(map(int, np.unique(pids[cats == cat_idx])))
                valid = set(CAT_TO_GLOBAL_PARTS[cat_idx])
                if not observed.issubset(valid):
                    raise ValueError(
                        f"{path}: category {CATEGORY_NAMES[cat_idx]} contains "
                        f"invalid parts {sorted(observed - valid)}"
                    )
            count += len(data)
        if count != expected_count:
            raise ValueError(
                f"{split} output contains {count} shapes; expected {expected_count}"
            )
        manifest["splits"][split] = {
            "count": count,
            "files": [os.path.basename(path) for path in files],
        }
    return manifest


def convert_split(raw_dir: str, out_dir: str, split: str, prefix: str,
                  start_idx: int = 0):
    """Convert one split (train/val/test) to HDF5 chunks."""
    pairs = load_split_list(raw_dir, split)
    if not pairs:
        print(f"  [skip] no entries for split '{split}'")
        return 0, start_idx

    np.random.seed(0)  # deterministic resampling

    data_buf, label_buf, pid_buf = [], [], []
    chunk_idx = start_idx
    n_written = 0
    n_missing = 0

    for i, (synset, name) in enumerate(pairs):
        pts_path, seg_path = find_shape_files(raw_dir, synset, name)
        if pts_path is None:
            n_missing += 1
            continue

        try:
            points, part, labels_are_local = load_shape(pts_path, seg_path)
        except Exception as e:
            print(f"  [warn] failed to read {pts_path}: {e}")
            continue

        points_n, part_n = resample(points, part, NUM_POINTS_PER_SHAPE)

        cat_idx = SYNSET_TO_CAT[synset]

        try:
            part_global = globalize_parts(part_n, cat_idx, labels_are_local)
        except ValueError as e:
            print(f"  [warn] {name}: {e}, skipping")
            continue

        data_buf.append(points_n)
        label_buf.append(cat_idx)
        pid_buf.append(part_global)

        if len(data_buf) >= SHAPES_PER_H5:
            arr_data  = np.stack(data_buf).astype(np.float32)
            arr_label = np.array(label_buf, dtype=np.int64).reshape(-1, 1)
            arr_pid   = np.stack(pid_buf).astype(np.int64)
            write_h5_chunk(out_dir, prefix, chunk_idx,
                           arr_data, arr_label, arr_pid)
            data_buf, label_buf, pid_buf = [], [], []
            chunk_idx += 1
            n_written += len(arr_data)

        if (i + 1) % 500 == 0:
            print(f"  ... processed {i+1}/{len(pairs)}")

    # Final partial chunk
    if data_buf:
        arr_data  = np.stack(data_buf).astype(np.float32)
        arr_label = np.array(label_buf, dtype=np.int64).reshape(-1, 1)
        arr_pid   = np.stack(pid_buf).astype(np.int64)
        write_h5_chunk(out_dir, prefix, chunk_idx,
                       arr_data, arr_label, arr_pid)
        n_written += len(arr_data)

    if n_missing:
        print(f"  [warn] missing point files for {n_missing}/{len(pairs)} "
              f"'{split}' entries")

    return n_written, chunk_idx + (1 if data_buf else 0)


def write_metadata(out_dir: str, manifest: dict):
    """Write all_object_categories.txt so loader treats this as valid."""
    path = os.path.join(out_dir, "all_object_categories.txt")
    with open(path, "w") as f:
        for name, sid in zip(CATEGORY_NAMES, SYNSET_TO_CAT.keys()):
            f.write(f"{name}\t{sid}\n")
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description="Convert raw PartAnnotation ShapeNetPart to HDF5"
    )
    p.add_argument("--raw_dir", type=str, required=True,
                   help="Root of raw PartAnnotation (contains synset folders)")
    p.add_argument("--out_dir", type=str,
                   default="data/shapenet_part_seg_hdf5_data",
                   help="Output directory for HDF5 files")
    args = p.parse_args()

    validate_raw_dir(args.raw_dir)
    clean_generated_outputs(args.out_dir)

    print(f"Converting raw -> HDF5")
    print(f"  raw:  {args.raw_dir}")
    print(f"  out:  {args.out_dir}")

    print("\n[train]")
    n_tr, _ = convert_split(args.raw_dir, args.out_dir, "train", "train")
    print(f"  total train: {n_tr} shapes")

    print("\n[val]")
    n_va, _ = convert_split(args.raw_dir, args.out_dir, "val", "val")
    print(f"  total val: {n_va} shapes")

    # Test split
    print("\n[test]")
    n_te, _ = convert_split(args.raw_dir, args.out_dir, "test", "test")
    print(f"  total test: {n_te} shapes")

    if n_tr == 0 or n_va == 0 or n_te == 0:
        raise RuntimeError(
            "Conversion produced an empty train, val, or test split. Check that raw_dir "
            "points/*.pts and points_label/*.seg match the split JSON entries."
        )

    manifest = validate_h5_output(
        args.out_dir, {"train": n_tr, "val": n_va, "test": n_te}
    )
    write_metadata(args.out_dir, manifest)
    print(f"\nDone. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()

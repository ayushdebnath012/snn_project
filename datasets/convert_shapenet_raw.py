"""
datasets/convert_shapenet_raw.py — Convert raw ShapeNetPart (PartAnnotation)
to the HDF5 format our loader expects.

Input format (Kaggle "shapenetcore_partanno_segmentation_benchmark_v0_normal"):
    raw_dir/
        synsetoffset2category.txt   # category_name <tab> synset_id
        02691156/                   # synset id (Airplane)
            points/*.pts            # text files: x y z nx ny nz part_label
        02773838/                   # synset id (Bag)
            points/*.pts
        ...
        train_test_split/
            shuffled_train_file_list.json
            shuffled_val_file_list.json
            shuffled_test_file_list.json

Output format (matches Stanford HDF5 download):
    out_dir/
        train0.h5 .. trainN.h5
        test0.h5 .. testM.h5
    Each h5 contains:
        data:  [num_shapes, 2048, 3]   float32
        label: [num_shapes, 1]         int64  (0-15 category)
        pid:   [num_shapes, 2048]      int64  (0-49 global part)
        all_object_categories.txt

Usage:
    python datasets/convert_shapenet_raw.py \
        --raw_dir data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --out_dir data/shapenet_part_seg_hdf5_data
"""

import argparse
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
    8: 24,  9: 28,  10: 30, 11: 36, 12: 38, 13: 41, 14: 44, 15: 46,
}

NUM_POINTS_PER_SHAPE = 2048
SHAPES_PER_H5 = 2048  # how many shapes per output h5 file


def load_shape(pts_path: str) -> tuple:
    """
    Load a .pts file from PartAnnotation.

    Each row has either:
        x y z part_label                 (4 cols, basic variant)
        x y z nx ny nz part_label        (7 cols, _normal variant)
    """
    data = np.loadtxt(pts_path).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 4:
        xyz, part = data[:, :3], data[:, 3]
    elif data.shape[1] >= 7:
        xyz, part = data[:, :3], data[:, -1]
    else:
        raise ValueError(f"Unexpected column count in {pts_path}: {data.shape}")
    return xyz, part.astype(np.int64)


def resample(xyz: np.ndarray, part: np.ndarray, n: int) -> tuple:
    """Sample exactly n points (with replacement if needed)."""
    if len(xyz) >= n:
        idx = np.random.choice(len(xyz), n, replace=False)
    else:
        idx = np.random.choice(len(xyz), n, replace=True)
    return xyz[idx], part[idx]


def load_split_list(raw_dir: str, split: str) -> list:
    """Parse the train_test_split JSON file."""
    fname = f"shuffled_{split}_file_list.json"
    path = os.path.join(raw_dir, "train_test_split", fname)
    if not os.path.exists(path):
        return []
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
        parsed.append((synset, name))
    return parsed


def find_pts_file(raw_dir: str, synset: str, name: str) -> str:
    """Locate the .pts/.txt file for a given shape."""
    candidates = [
        os.path.join(raw_dir, synset, "points", f"{name}.pts"),
        os.path.join(raw_dir, synset, "points", f"{name}.txt"),
        os.path.join(raw_dir, synset, f"{name}.pts"),
        os.path.join(raw_dir, synset, f"{name}.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


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


def convert_split(raw_dir: str, out_dir: str, split: str, prefix: str):
    """Convert one split (train/val/test) to HDF5 chunks."""
    pairs = load_split_list(raw_dir, split)
    if not pairs:
        print(f"  [skip] no entries for split '{split}'")
        return 0

    np.random.seed(0)  # deterministic resampling

    data_buf, label_buf, pid_buf = [], [], []
    chunk_idx = 0
    n_written = 0

    for i, (synset, name) in enumerate(pairs):
        pts_path = find_pts_file(raw_dir, synset, name)
        if pts_path is None:
            continue

        try:
            xyz, part = load_shape(pts_path)
        except Exception as e:
            print(f"  [warn] failed to read {pts_path}: {e}")
            continue

        xyz_n, part_n = resample(xyz, part, NUM_POINTS_PER_SHAPE)

        cat_idx = SYNSET_TO_CAT[synset]

        # Make part labels GLOBAL (0..49) if the raw labels are 1-indexed local
        if part_n.min() >= 1 and part_n.max() <= 6:
            # Local 1-indexed labels — shift to global
            part_global = part_n - 1 + PART_OFFSET[cat_idx]
        elif part_n.min() >= 0 and part_n.max() <= 49:
            # Already global
            part_global = part_n
        else:
            print(f"  [warn] unexpected label range in {name}: "
                  f"[{part_n.min()},{part_n.max()}], skipping")
            continue

        data_buf.append(xyz_n)
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

    return n_written


def write_metadata(out_dir: str):
    """Write all_object_categories.txt so loader treats this as valid."""
    path = os.path.join(out_dir, "all_object_categories.txt")
    with open(path, "w") as f:
        for name, sid in zip(CATEGORY_NAMES, SYNSET_TO_CAT.keys()):
            f.write(f"{name}\t{sid}\n")


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

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Converting raw -> HDF5")
    print(f"  raw:  {args.raw_dir}")
    print(f"  out:  {args.out_dir}")

    # Train split (combine train + val into train h5)
    print("\n[train + val]")
    n_tr = convert_split(args.raw_dir, args.out_dir, "train", "train")
    n_va = convert_split(args.raw_dir, args.out_dir, "val", "train")
    print(f"  total train+val: {n_tr + n_va} shapes")

    # Test split
    print("\n[test]")
    n_te = convert_split(args.raw_dir, args.out_dir, "test", "test")
    print(f"  total test: {n_te} shapes")

    write_metadata(args.out_dir)
    print(f"\nDone. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()

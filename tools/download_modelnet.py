#!/usr/bin/env python3
"""Download and validate the official ModelNet10/40 archives."""

from __future__ import annotations

import argparse
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive_bytes: int
    class_count: int


DATASETS = {
    "10": DatasetSpec(
        name="ModelNet10",
        url="https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        archive_bytes=473_402_300,
        class_count=10,
    ),
    "40": DatasetSpec(
        name="ModelNet40",
        url="https://modelnet.cs.princeton.edu/ModelNet40.zip",
        archive_bytes=2_039_180_837,
        class_count=40,
    ),
}


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def class_directories(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.is_dir():
        return []
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    )


def validate_dataset(dataset_dir: Path, spec: DatasetSpec) -> None:
    classes = class_directories(dataset_dir)
    if len(classes) != spec.class_count:
        raise RuntimeError(
            f"{dataset_dir} contains {len(classes)} valid class directories; "
            f"expected {spec.class_count}"
        )
    has_off_samples = any(
        sample.is_file() and sample.suffix.lower() == ".off"
        for cls in classes
        for split in (cls / "train", cls / "test")
        for sample in split.iterdir()
    )
    if not has_off_samples:
        raise RuntimeError(f"{dataset_dir} does not contain OFF samples")


def download_archive(spec: DatasetSpec, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file() and archive.stat().st_size == spec.archive_bytes:
        print(f"Using complete archive: {archive}")
        return

    partial = archive.with_suffix(archive.suffix + ".part")
    if partial.is_file() and partial.stat().st_size > spec.archive_bytes:
        partial.unlink()
    if partial.is_file() and partial.stat().st_size == spec.archive_bytes:
        partial.replace(archive)
        print(f"Using completed partial archive: {archive}")
        return
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "SpikeGAT-SNN dataset setup"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    print(f"Downloading {spec.name} ({human_size(spec.archive_bytes)})")
    if offset:
        print(f"Resuming at {human_size(offset)}")

    request = urllib.request.Request(spec.url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = offset > 0 and response.status == 206
        if offset and not resumed:
            offset = 0
        mode = "ab" if resumed else "wb"
        downloaded = offset
        last_report = 0.0
        with partial.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 1:
                    percent = 100 * downloaded / spec.archive_bytes
                    print(
                        f"  {human_size(downloaded)} / {human_size(spec.archive_bytes)} "
                        f"({percent:.1f}%)",
                        end="\r",
                        flush=True,
                    )
                    last_report = now
    print()

    actual_size = partial.stat().st_size
    if actual_size != spec.archive_bytes:
        raise RuntimeError(
            f"incomplete archive: received {actual_size} bytes, expected {spec.archive_bytes}; "
            "rerun the command to resume"
        )
    partial.replace(archive)


def extract_archive(archive: Path, data_root: Path) -> None:
    root = data_root.resolve()
    print(f"Extracting {archive.name} into {root}")
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (root / entry.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive path: {entry.filename}")
            mode = (entry.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"archive contains a symbolic link: {entry.filename}")
        bundle.extractall(root)


def install_dataset(spec: DatasetSpec, data_root: Path, keep_archive: bool) -> Path:
    dataset_dir = data_root / spec.name
    try:
        validate_dataset(dataset_dir, spec)
    except RuntimeError:
        pass
    else:
        print(f"{spec.name} is already ready: {dataset_dir.resolve()}")
        return dataset_dir.resolve()

    archive = data_root / ".downloads" / f"{spec.name}.zip"
    download_archive(spec, archive)
    extract_archive(archive, data_root)
    validate_dataset(dataset_dir, spec)
    if not keep_archive:
        archive.unlink()
    print(f"Ready: {dataset_dir.resolve()}")
    return dataset_dir.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("10", "40", "all"),
        help="dataset to download; required unless --list is used",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="destination parent directory (default: repository data/)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="retain the ZIP after successful extraction",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate existing data without downloading",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show official sources and archive sizes, then exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for key, spec in DATASETS.items():
            print(f"{key}: {spec.name}  {human_size(spec.archive_bytes)}  {spec.url}")
        return 0
    if args.dataset is None:
        print("error: --dataset is required unless --list is used", file=sys.stderr)
        return 2

    keys = DATASETS if args.dataset == "all" else (args.dataset,)
    try:
        for key in keys:
            spec = DATASETS[key]
            if args.check:
                dataset_dir = args.data_root / spec.name
                validate_dataset(dataset_dir, spec)
                print(f"Valid: {dataset_dir.resolve()}")
            else:
                dataset_dir = install_dataset(spec, args.data_root, args.keep_archive)
                variable = f"MODELNET{key}_DIR"
                print(f'POSIX: export {variable}="{dataset_dir}"')
                print(f"PowerShell: $env:{variable}='{dataset_dir}'")
    except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

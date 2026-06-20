#!/usr/bin/env python3
"""Fast repository integrity check; does not start training or download data."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "config.py",
    "configs/scanobj_cls.yaml",
    "configs/shapenet_seg.yaml",
    "configs/s3dis_seg.yaml",
    "experiments/kaggle/spikegat/modelnet10.py",
    "experiments/kaggle/spikegat/modelnet40.py",
    "tasks/train_scanobjectnn.py",
    "tasks/train_shapenetpart.py",
    "tasks/train_s3dis.py",
    "docs/CLUSTER.md",
]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pdf", ".h5", ".hdf5"}
STALE_TEXT = (
    "github.com/" + "AryaPawa/ASP-SNN",
    "codex/" + "fix-shapenet-h5-conversion",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imports", action="store_true", help="Import shared core modules")
    args = parser.parse_args()

    errors: list[str] = []
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required file: {rel}")

    python_files = sorted(
        path for path in ROOT.rglob("*.py") if ".venv" not in path.parts
    )
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(str(exc))

    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/restricted artifact is tracked locally: {path.relative_to(ROOT)}")

    for path in [*python_files, *ROOT.rglob("*.md")]:
        text = path.read_text(encoding="utf-8", errors="replace")
        for stale in STALE_TEXT:
            if stale in text:
                errors.append(f"stale repository reference in {path.relative_to(ROOT)}: {stale}")

    if args.imports:
        sys.path.insert(0, str(ROOT))
        try:
            import config  # noqa: F401
            for package_name in ("data", "datasets", "models", "training"):
                package = importlib.import_module(package_name)
                for module in pkgutil.iter_modules(package.__path__, package_name + "."):
                    importlib.import_module(module.name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"shared-module import failed: {exc!r}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"VALIDATION PASSED: {len(python_files)} Python files compiled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

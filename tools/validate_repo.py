#!/usr/bin/env python3
"""Validate the focused ModelNet10/40 SpikeGAT repository."""

from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPERIMENTS = {
    "experiments/full/train_spikegat_modelnet10.py",
    "experiments/full/train_spikegat_modelnet40.py",
}
REQUIRED = EXPECTED_EXPERIMENTS | {
    "README.md",
    "docs/CLUSTER.md",
    "docs/EXPERIMENTS.md",
    "scripts/slurm/spikegat_mn10.sbatch",
    "scripts/slurm/spikegat_mn40.sbatch",
    "scripts/slurm/submit_all.sh",
    "requirements.txt",
    "environment.yml",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pdf", ".h5", ".hdf5"}
FORBIDDEN_PATH_PARTS = {"configs", "data", "datasets", "models", "tasks", "training"}
STALE_TEXT = (
    "ASPClassifier",
    "ASPSegmentor",
    "train_asp_",
    "ScanObjectNN",
    "ShapeNetPart",
    "S3DIS",
)


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required file: {rel}")

    experiment_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "experiments").rglob("*.py")
    }
    if experiment_files != EXPECTED_EXPERIMENTS:
        errors.append(
            "unexpected experiment surface: "
            f"expected={sorted(EXPECTED_EXPERIMENTS)}, actual={sorted(experiment_files)}"
        )

    python_files = sorted(path for path in ROOT.rglob("*.py") if ".venv" not in path.parts)
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(str(exc))

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in rel.parts):
            errors.append(f"out-of-scope path remains: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/restricted artifact remains: {rel}")

    text_files = [ROOT / "README.md", *ROOT.glob("docs/*.md"), *ROOT.glob("scripts/slurm/*")]
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for stale in STALE_TEXT:
            if stale in text:
                errors.append(f"stale out-of-scope reference in {path.relative_to(ROOT)}: {stale}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"VALIDATION PASSED: {len(python_files)} Python files; MN10/MN40 only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

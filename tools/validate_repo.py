#!/usr/bin/env python3
"""Validate repository health while allowing new dataset integrations."""

from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_EXPERIMENTS = {
    "experiments/modelnet/train_spikegat_modelnet10.py",
    "experiments/modelnet/train_spikegat_modelnet40.py",
}
REQUIRED = BASELINE_EXPERIMENTS | {
    "CONTRIBUTING.md",
    "README.md",
    "docs/CLUSTER.md",
    "docs/EXPERIMENTS.md",
    "scripts/slurm/modelnet/spikegat_mn10.sbatch",
    "scripts/slurm/modelnet/spikegat_mn40.sbatch",
    "scripts/slurm/submit_all.sh",
    "requirements.txt",
    "environment.yml",
}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pdf",
    ".pt",
    ".pth",
}
STALE_ASP_TEXT = ("ASPClassifier", "ASPSegmentor", "train_asp_")
IGNORED_PARTS = {".git", ".venv", "__pycache__", "venv"}


def tracked_files() -> list[Path]:
    """Return files Git would publish, with a filesystem fallback."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        return [ROOT / name.decode() for name in result.stdout.split(b"\0") if name]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    ]


def main() -> int:
    errors: list[str] = []

    for rel in sorted(REQUIRED):
        if not (ROOT / rel).is_file():
            errors.append(f"missing required file: {rel}")

    python_files = sorted(
        path
        for path in ROOT.rglob("*.py")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(str(exc))

    published_files = [path for path in tracked_files() if path.is_file()]
    for path in published_files:
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            errors.append(f"generated/restricted artifact is tracked: {rel}")

        if path == ROOT / "tools/validate_repo.py":
            continue
        if path.suffix.lower() not in {".md", ".py", ".sh", ".sbatch", ".txt", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for stale in STALE_ASP_TEXT:
            if stale in text:
                errors.append(f"stale ASP reference in {rel}: {stale}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    experiment_count = len(list((ROOT / "experiments").rglob("*.py")))
    print(
        "VALIDATION PASSED: "
        f"{len(python_files)} Python files, {experiment_count} experiment files; "
        "additional datasets are supported"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
multi_seed.py
=============
Run any experiment N times with different random seeds and report mean ± std.

This addresses the reviewer concern:
  "Where are error bars / standard deviations across 3–5 runs?"

Usage:
  from training.multi_seed import run_with_seeds, format_result

  mean, std, per_seed = run_with_seeds(
      build_fn   = lambda: build_model("ours_full", num_classes=40).to(device),
      train_fn   = lambda model: train_model_full(model, ...),
      eval_fn    = lambda model: eval_acc(model, ...),
      seeds      = [0, 1, 2],
  )
  print(format_result(mean, std))   # "87.34 ± 0.42"
"""

import time
import copy
import numpy as np
import torch


# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def format_result(mean: float, std: float, scale=100.0, decimals=2) -> str:
    """Return 'XX.XX ± Y.YY' string (e.g. for accuracy in %)."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean * scale)} ± {fmt.format(std * scale)}"


# ---------------------------------------------------------------------------

def run_with_seeds(build_fn, train_fn, eval_fn,
                   seeds=(0, 1, 2), verbose=True):
    """
    Run build → train → eval for each seed. Collect per-seed results.

    Args:
        build_fn  : callable() → nn.Module  (called fresh each seed)
        train_fn  : callable(model) → any   (trains model in-place)
        eval_fn   : callable(model) → float (returns scalar metric, e.g. accuracy)
        seeds     : iterable of ints
        verbose   : print per-seed results

    Returns:
        mean      : float
        std       : float
        per_seed  : list of (seed, metric) tuples
    """
    per_seed = []
    t_start  = time.time()

    for seed in seeds:
        set_seed(seed)
        if verbose:
            print(f"\n  [Seed {seed}] Building model ...")

        model = build_fn()
        t0    = time.time()
        train_fn(model)
        val   = eval_fn(model)
        elapsed = time.time() - t0

        per_seed.append((seed, val))
        if verbose:
            print(f"  [Seed {seed}] val={val*100:.2f}%  ({elapsed:.0f}s)")

    vals = [v for _, v in per_seed]
    mean = float(np.mean(vals))
    std  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    if verbose:
        print(f"\n  Result: {format_result(mean, std)}%  "
              f"(over {len(seeds)} seeds, {time.time()-t_start:.0f}s total)")

    return mean, std, per_seed


# ---------------------------------------------------------------------------
# Higher-level wrapper: multi-seed comparison of multiple model configs
# ---------------------------------------------------------------------------

def compare_configs_multi_seed(configs, train_fn_factory, eval_fn,
                                seeds=(0, 1, 2), verbose=True):
    """
    Compare multiple model configurations with mean ± std.

    Args:
        configs          : list of (name, build_fn) pairs
        train_fn_factory : callable(model) → train_fn
                           (called each seed so state is fresh)
        eval_fn          : callable(model) → float
        seeds            : random seeds
        verbose          : print progress

    Returns:
        results : dict { name: {"mean": float, "std": float,
                                "per_seed": list } }
    """
    results = {}
    for name, build_fn in configs:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Config: {name}")
            print(f"{'='*60}")

        mean, std, per_seed = run_with_seeds(
            build_fn  = build_fn,
            train_fn  = lambda m, _bfn=build_fn: train_fn_factory(m),
            eval_fn   = eval_fn,
            seeds     = seeds,
            verbose   = verbose,
        )
        results[name] = {"mean": mean, "std": std, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Print a results table with confidence intervals
# ---------------------------------------------------------------------------

def print_multi_seed_table(results: dict, scale=100.0):
    """
    Print a formatted table of mean ± std results.

    Args:
        results : dict { name: {"mean", "std", "per_seed"} }
        scale   : multiply mean/std by this (e.g. 100.0 for accuracy %)
    """
    print(f"\n{'Config':<30} {'Mean':>8}  {'Std':>7}  Per-seed values")
    print("-" * 75)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["mean"]):
        per = "  ".join(f"{v*scale:.2f}" for _, v in r["per_seed"])
        print(f"{name:<30} {r['mean']*scale:>8.2f}  "
              f"±{r['std']*scale:>6.2f}  [{per}]")

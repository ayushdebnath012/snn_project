"""
plots_active.py
===============
Plotting utilities for Active Spiking Perception paper figures.

Figures generated:
  Fig 1 — Energy–Accuracy Pareto Frontier (main result)
  Fig 2 — Exit Timestep Distribution histogram
  Fig 3 — Confidence growth over timesteps (SSP vs fixed)
  Fig 4 — SSP Attention Maps per class (heatmap)
  Fig 5 — Ablation bar chart (contribution of each component)
  Fig 6 — Policy entropy over training epochs (how SSP evolves)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


COLORS = {
    "asp":        "#E6550D",   # orange-red (ours)
    "ours_full":  "#FD8D3C",   # light orange
    "ours_knn":   "#FDBE85",   # pale orange
    "spt":        "#3182BD",   # blue (SPT)
    "spm":        "#6BAED6",   # light blue (SPM)
    "ann":        "#636363",   # grey (ANN)
}


# -----------------------------------------------------------------------
# Fig 1: Energy–Accuracy Pareto Frontier
# -----------------------------------------------------------------------

def plot_pareto(
    asp_curve: list[dict],
    fixed_baselines: dict,
    save_path: str = "results/active/fig1_pareto.png",
):
    """
    Plot energy–accuracy Pareto frontier.

    Args:
        asp_curve : list of dicts from pareto_curve() (each has 'energy_ratio', 'accuracy')
        fixed_baselines : dict of {model_name: {'energy_ratio': float, 'accuracy': float}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # ASP Pareto curve
    energies = [p["energy_ratio"] for p in asp_curve]
    accs     = [p["accuracy"] * 100 for p in asp_curve]
    ax.plot(energies, accs, "o-", color=COLORS["asp"], linewidth=2.5,
            markersize=5, label="ASP (Ours, adaptive)", zorder=5)

    # Annotate a few key operating points
    for i, p in enumerate(asp_curve):
        if p.get("threshold") in [0.0, 0.5, 0.8, 1.0]:
            ax.annotate(
                f"θ={p['threshold']:.1f}",
                xy=(p["energy_ratio"], p["accuracy"] * 100),
                xytext=(p["energy_ratio"] + 0.003, p["accuracy"] * 100 + 0.2),
                fontsize=7, color=COLORS["asp"],
            )

    # Fixed-order baselines (single points)
    baseline_styles = {
        "ours_full": ("s", COLORS["ours_full"], "ours_full (8.4×)"),
        "ours_knn":  ("^", COLORS["ours_knn"],  "ours_knn (11.1×)"),
        "spt":       ("D", COLORS["spt"],        "SPT (6.4×)"),
        "spm":       ("P", COLORS["spm"],        "SPM (3.5×)"),
    }
    for name, meta in fixed_baselines.items():
        if name in baseline_styles:
            marker, color, label = baseline_styles[name]
            ax.scatter(
                meta["energy_ratio"], meta["accuracy"] * 100,
                marker=marker, color=color, s=100, zorder=6,
                label=label, edgecolors="black", linewidths=0.7,
            )

    ax.set_xlabel("E_SNN / E_ANN  (lower = more efficient)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Energy–Accuracy Pareto Frontier\n(ModelNet10, T=16 slices)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Shade "dominance region"
    if asp_curve:
        max_energy = max(p["energy_ratio"] for p in asp_curve)
        min_acc    = min(p["accuracy"] * 100 for p in asp_curve)
        ax.fill_betweenx(
            [min_acc - 1, 100],
            [0, 0], [max_energy, max_energy],
            alpha=0.04, color=COLORS["asp"], label="_nolegend_"
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plots] Saved Pareto frontier → {save_path}")


# -----------------------------------------------------------------------
# Fig 2: Exit Timestep Distribution
# -----------------------------------------------------------------------

def plot_exit_distribution(
    exit_steps: list[int],
    num_slices: int = 16,
    save_path: str = "results/active/fig2_exit_dist.png",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    bins = range(1, num_slices + 2)
    ax1.hist(exit_steps, bins=bins, align="left", rwidth=0.8,
             color=COLORS["asp"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.axvline(x=np.mean(exit_steps), color="black", linestyle="--",
                linewidth=1.5, label=f"Mean = {np.mean(exit_steps):.1f}")
    ax1.set_xlabel("Exit Timestep", fontsize=12)
    ax1.set_ylabel("Number of Samples", fontsize=12)
    ax1.set_title("Exit Timestep Distribution\n(ASP, ModelNet10)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    # CDF
    exits_sorted = np.sort(exit_steps)
    p = np.arange(1, len(exits_sorted) + 1) / len(exits_sorted)
    ax2.plot(exits_sorted, p * 100, color=COLORS["asp"], linewidth=2.5)
    ax2.axhline(y=50, color="grey", linestyle=":", linewidth=1)
    ax2.axhline(y=80, color="grey", linestyle=":", linewidth=1)
    ax2.set_xlabel("Exit Timestep", fontsize=12)
    ax2.set_ylabel("Cumulative % of Samples", fontsize=12)
    ax2.set_title("Exit Timestep CDF", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 102)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plots] Saved exit distribution → {save_path}")


# -----------------------------------------------------------------------
# Fig 3: Confidence Growth
# -----------------------------------------------------------------------

def plot_confidence_growth(
    conf_asp: list[float],
    conf_fixed: list[float] = None,
    save_path: str = "results/active/fig3_confidence.png",
):
    fig, ax = plt.subplots(figsize=(8, 5))

    T = len(conf_asp)
    steps = list(range(1, T + 1))

    ax.plot(steps, [c * 100 for c in conf_asp], "o-", color=COLORS["asp"],
            linewidth=2.5, markersize=5, label="ASP (adaptive ordering)")
    if conf_fixed:
        ax.plot(steps, [c * 100 for c in conf_fixed[:T]], "s--",
                color=COLORS["ours_full"], linewidth=2, markersize=5,
                label="Fixed FPS order")

    ax.axhline(y=80, color="grey", linestyle=":", linewidth=1, label="θ=0.8 threshold")
    ax.set_xlabel("Timestep (slices seen)", fontsize=12)
    ax.set_ylabel("Mean Max Softmax Probability (%)", fontsize=12)
    ax.set_title("Confidence Growth: ASP vs Fixed Ordering", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 102)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plots] Saved confidence growth → {save_path}")


# -----------------------------------------------------------------------
# Fig 4: SSP Attention Maps (per class)
# -----------------------------------------------------------------------

def plot_ssp_attention(
    attention_maps: dict,
    num_slices: int = 16,
    save_path: str = "results/active/fig4_attention.png",
):
    """
    Heatmap: rows = classes, columns = slice anchor index,
    cell = mean priority score (how often selected early).
    """
    class_names = list(attention_maps.keys())
    n_classes   = len(class_names)
    data = np.array([attention_maps[c] for c in class_names])   # [C, T]

    fig, ax = plt.subplots(figsize=(12, max(4, n_classes * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=data.max())

    ax.set_xticks(range(num_slices))
    ax.set_xticklabels([f"A{i}" for i in range(num_slices)], fontsize=7)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("FPS Anchor Region Index", fontsize=11)
    ax.set_ylabel("Object Class", fontsize=11)
    ax.set_title("SSP Attention Map: Priority per Anchor Region per Class\n"
                 "(bright = selected early, dark = selected late)", fontsize=12)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Mean Selection Priority")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved SSP attention map → {save_path}")


# -----------------------------------------------------------------------
# Fig 5: Ablation Bar Chart
# -----------------------------------------------------------------------

def plot_ablation(
    ablation_results: dict,
    save_path: str = "results/active/fig5_ablation.png",
):
    """
    ablation_results: dict of {model_name: {'acc': float, 'savings': float}}
    """
    models = list(ablation_results.keys())
    accs   = [ablation_results[m]["acc"] * 100 for m in models]
    saves  = [ablation_results[m]["savings"]    for m in models]

    x = np.arange(len(models))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - 0.2, accs,  0.35, color="#FD8D3C", alpha=0.85,
                    label="Accuracy (%)", edgecolor="black", linewidth=0.7)
    bars2 = ax2.bar(x + 0.2, saves, 0.35, color="#3182BD", alpha=0.85,
                    label="Energy Savings (×)", edgecolor="black", linewidth=0.7)

    ax1.set_xlabel("Model Variant", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color="#FD8D3C")
    ax2.set_ylabel("Energy Savings (×)", fontsize=12, color="#3182BD")
    ax1.set_title("Ablation: Contribution of Each ASP Component", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=9)

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{bar.get_height():.1f}×", ha="center", va="bottom", fontsize=8)

    legend1 = mpatches.Patch(color="#FD8D3C", label="Accuracy (%)")
    legend2 = mpatches.Patch(color="#3182BD", label="Energy Savings (×)")
    ax1.legend(handles=[legend1, legend2], loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plots] Saved ablation chart → {save_path}")


# -----------------------------------------------------------------------
# Fig 6: Policy Entropy over Training
# -----------------------------------------------------------------------

def plot_policy_entropy(
    history: list[dict],
    save_path: str = "results/active/fig6_policy_entropy.png",
):
    epochs = [h["epoch"] for h in history]
    ent    = [h.get("policy_entropy", 0) for h in history]
    tau    = [h.get("gumbel_tau", 1.0)   for h in history]
    acc    = [h.get("acc_final", 0) * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(epochs, ent, color=COLORS["asp"], linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, tau, color="grey", linestyle="--", linewidth=1.5,
                  label="Gumbel τ")
    ax1.set_ylabel("Policy Entropy (nats)", color=COLORS["asp"], fontsize=11)
    ax1_twin.set_ylabel("Gumbel Temperature τ", color="grey", fontsize=11)
    ax1.set_title("SSP Policy Entropy & Gumbel Temperature over Training", fontsize=12)
    ax1_twin.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, acc, color=COLORS["ours_full"], linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Training Accuracy (%)", fontsize=11)
    ax2.set_title("Training Accuracy", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plots] Saved policy entropy plot → {save_path}")


# -----------------------------------------------------------------------
# Generate all paper figures from saved JSON files
# -----------------------------------------------------------------------

def generate_all_figures(results_dir: str = "results/active/seed_0/"):
    """Load saved JSON results and generate all paper figures."""
    print(f"\n=== Generating Paper Figures from {results_dir} ===")

    # Fig 6: Training history
    hist_path = os.path.join(results_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_policy_entropy(history, save_path=os.path.join(results_dir, "fig6_entropy.png"))

    # Fig 1: Pareto curve
    pareto_path = os.path.join(results_dir, "pareto_curve.json")
    if os.path.exists(pareto_path):
        with open(pareto_path) as f:
            curve = json.load(f)
        fixed_baselines = {
            "ours_full": {"energy_ratio": 0.119, "accuracy": 0.9064},
            "ours_knn":  {"energy_ratio": 0.090, "accuracy": 0.8987},
            "spt":       {"energy_ratio": 0.156, "accuracy": 0.914},
            "spm":       {"energy_ratio": 0.286, "accuracy": 0.923},
        }
        plot_pareto(curve, fixed_baselines,
                    save_path=os.path.join(results_dir, "fig1_pareto.png"))

    # Fig 4: SSP attention
    attn_path = os.path.join(results_dir, "ssp_attention.json")
    if os.path.exists(attn_path):
        with open(attn_path) as f:
            attn = json.load(f)
        # Convert lists back to numpy arrays
        attn_np = {k: np.array(v) for k, v in attn.items()}
        plot_ssp_attention(attn_np,
                           save_path=os.path.join(results_dir, "fig4_attention.png"))

    print("All figures generated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/active/seed_0/")
    args = parser.parse_args()
    generate_all_figures(args.results_dir)

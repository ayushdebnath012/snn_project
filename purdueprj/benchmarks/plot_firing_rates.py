"""
plot_firing_rates.py
====================
Generates a bar plot showing the average firing rate for each spiking layer
in the network. Helps verify the energy efficiency per layer.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from models.model_zoo import build_model
from data.slicing import slice_fps_hierarchical_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ours_full")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint to load")
    parser.add_argument("--out", default="results/firing_rates.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, num_classes=40).to(device)
    
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.ckpt}")
    
    model.eval()
    
    # Dummy data forward pass to collect stats
    pts = torch.randn(2, 1024, 3, device=device)
    pts_slices = slice_fps_hierarchical_batch(pts, T=16)
    
    model.reset_state(2, device)
    with torch.no_grad():
        for t in range(16):
            model.forward_step(pts_slices[:, t])
            
    # Collect rates
    rates = model.get_firing_rates()
    
    if not rates:
        print("Model does not report firing_rates.")
        return
        
    # Plot
    layers = list(rates.keys())
    vals = list(rates.values())
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(layers, vals, color="steelblue", edgecolor="black")
    plt.axhline(y=np.mean(vals), color="tomato", linestyle="--", label=f"Mean: {np.mean(vals):.3f}")
    
    plt.ylabel("Average Firing Rate (Spikes/Neuron)")
    plt.title(f"Per-Layer Firing Rates: {args.model}")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(max(vals) * 1.2, 0.1))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Value labels
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=9)
                 
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved firing rate plot to {args.out}")

if __name__ == "__main__":
    main()

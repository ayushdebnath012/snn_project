import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_all_metrics(results, output_dir):
    """
    results: dict key -> metrics dict
    Keys expected: 'ANN+Slice', 'SNN+Slice', 'ANN+Full', 'SNN+Full'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy vs Timestep
    plt.figure(figsize=(8, 6))
    if 'ANN+Slice' in results:
        plt.plot(results['ANN+Slice']['acc_vs_timestep'], label='ANN Slice', marker='o')
    if 'SNN+Slice' in results:
        plt.plot(results['SNN+Slice']['acc_vs_timestep'], label='SNN Slice', marker='s')
        
    # Add baselines
    if 'ANN+Full' in results:
        plt.axhline(y=results['ANN+Full']['final_accuracy'], color='gray', linestyle='--', label='ANN Full')
    if 'SNN+Full' in results:
        plt.axhline(y=results['SNN+Full']['final_accuracy'], color='black', linestyle='--', label='SNN Full')
        
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Timestep")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_timestep.png'))
    plt.close()
    
    # 2. Early Exit Histogram (SNN)
    if 'SNN+Slice' in results:
        plt.figure(figsize=(8, 6))
        plt.hist(results['SNN+Slice']['exit_steps'], bins=range(18), align='left', rwidth=0.8, alpha=0.7, label='SNN')
        plt.xlabel("Exit Timestep")
        plt.ylabel("Count")
        plt.title("Early Exit Histogram (SNN)")
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, 'exit_histogram_snn.png'))
        plt.close()
        
    # 3. Threshold Tradeoff Curve (Mean Exit vs Accuracy)
    # Require raw probs
    plt.figure(figsize=(8, 6))
    thresholds = np.linspace(0.5, 0.99, 20)
    
    for mode in ['ANN+Slice', 'SNN+Slice']:
        if mode in results and 'all_probs' in results[mode]:
            probs = results[mode]['all_probs'] # [N, T, C]
            labels = results[mode]['all_labels']
            num_slices = probs.shape[1]
            
            mean_exits = []
            final_accs = []
            
            for th in thresholds:
                # Calc exit step
                max_probs, _ = probs.max(dim=2)
                exited = max_probs > th
                exit_indices = torch.argmax(exited.int(), dim=1)
                any_exited = exited.any(dim=1)
                exit_indices[~any_exited] = num_slices - 1
                
                # Calc mean exit
                mean_exits.append(exit_indices.float().mean().item())
                
                # Calc accuracy at exit
                # Gather logits at exit step
                # indices: [N, 1, 1] -> [N, 1, C]
                batch_indices = torch.arange(probs.size(0))
                probs_at_exit = probs[batch_indices, exit_indices, :]
                preds = probs_at_exit.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                final_accs.append(acc)
                
            plt.plot(mean_exits, final_accs, marker='.', label=mode)
            
    plt.xlabel("Mean Exit Timestep")
    plt.ylabel("Accuracy")
    plt.title("Threshold Tradeoff Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_tradeoff.png'))
    plt.close()

    # 4. Exit Timestep CDF
    plt.figure(figsize=(8, 6))
    for mode in ['ANN+Slice', 'SNN+Slice']:
        if mode in results:
            exits = np.array(results[mode]['exit_steps'])
            exits_sorted = np.sort(exits)
            p = 1. * np.arange(len(exits)) / (len(exits) - 1)
            plt.plot(exits_sorted, p, label=mode)
            
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Fraction Exited")
    plt.title("Exit Timestep CDF")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'exit_cdf.png'))
    plt.close()

    # 5. Confidence Growth Curve
    plt.figure(figsize=(8, 6))
    if 'SNN+Slice' in results and 'confidence_curve' in results['SNN+Slice']:
        plt.plot(results['SNN+Slice']['confidence_curve'], label='SNN', marker='s')
    if 'ANN+Slice' in results and 'all_probs' in results['ANN+Slice']:
         probs = results['ANN+Slice']['all_probs']
         max_probs, _ = probs.max(dim=2)
         conf_curve = max_probs.mean(dim=0).numpy()
         plt.plot(conf_curve, label='ANN', marker='o')

    plt.xlabel("Timestep")
    plt.ylabel("Average Max Softmax")
    plt.title("Confidence Growth")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'confidence_growth.png'))
    plt.close()

    # -----------------------------------------------------------------------
    # 6. SNN vs ANN Energy Efficiency vs Timestep
    #
    # From notes: "Acc vs timestep, ANN efficiency, SNN efficiency, how calc?"
    # E_MAC = 4.6 pJ (ANN), E_AC = 0.9 pJ (SNN spike).
    # Effective SNN energy at timestep t = firing_rate(t) × E_AC.
    # We normalise both to ANN=1.0 so the plot shows relative cost.
    # -----------------------------------------------------------------------
    E_MAC = 4.6
    E_AC  = 0.9
    if 'SNN+Slice' in results and 'confidence_curve' in results['SNN+Slice']:
        T = len(results['SNN+Slice']['confidence_curve'])
        timesteps = np.arange(1, T + 1)

        # ANN: constant cost per timestep (baseline = 1.0)
        ann_energy = np.ones(T)

        # SNN: cumulative cost grows with timesteps, but each step only
        # fires a fraction f of neurons (use confidence as proxy for f:
        # high confidence → model has learned → neurons fire selectively).
        conf = np.array(results['SNN+Slice']['confidence_curve'])
        # firing rate proxy: 1 - confidence (low conf = high entropy = more spikes)
        firing_rate_proxy = np.clip(1.0 - conf, 0.05, 1.0)
        snn_energy_per_step = firing_rate_proxy * (E_AC / E_MAC)
        snn_cumulative = np.cumsum(snn_energy_per_step) / timesteps

        plt.figure(figsize=(8, 6))
        plt.plot(timesteps, ann_energy, label='ANN (normalised)', color='tab:blue',
                 linestyle='--', linewidth=2)
        plt.plot(timesteps, snn_cumulative, label='SNN (normalised)', color='tab:orange',
                 marker='s', linewidth=2)
        plt.xlabel("Timestep (slices seen)")
        plt.ylabel("Relative Energy Cost (ANN = 1.0)")
        plt.title("SNN vs ANN Energy Efficiency vs Timestep")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'snn_vs_ann_energy.png'))
        plt.close()

    # -----------------------------------------------------------------------
    # 7. Paper Comparison Bar Chart
    #    Contextualises our model against published results on ModelNet40.
    # -----------------------------------------------------------------------
    paper_models = [
        # (name,            type,   MN40 acc)
        ("PointNet",         "ANN",  89.2),
        ("PointNet++",       "ANN",  90.7),
        ("PointMLP",         "ANN",  94.1),
        ("PointMamba",       "ANN",  92.4),
        ("Spiking PointNet", "SNN",  88.2),
        ("P2SResLNet-B",     "SNN",  88.7),
        ("SPT",              "SNN",  91.4),
        ("SPM (paper)",      "SNN",  92.3),
        ("Ours (SNN)",       "SNN",  None),   # fill in after training
    ]
    names   = [m[0] for m in paper_models]
    accs    = [m[2] if m[2] else 0 for m in paper_models]
    colors  = ['steelblue' if m[1] == 'ANN' else 'tomato' for m in paper_models]
    # Highlight ours
    colors[-1] = 'gold'

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel("ModelNet40 Accuracy (%)")
    ax.set_title("ModelNet40 Classification: ANNs vs SNNs (our model = gold)")
    ax.set_ylim(85, 95)
    ax.axhline(y=92.3, color='tomato', linestyle=':', linewidth=1.2, label='SPM baseline')

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='ANN'),
        Patch(facecolor='tomato',    label='SNN'),
        Patch(facecolor='gold',      label='Ours'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_comparison.png'), dpi=150)
    plt.close()
    print(f"[Plots] Saved to {output_dir}/")

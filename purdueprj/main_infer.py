import torch
import argparse
import os
from data.modelnet import ModelNetDataset
from models.pointnet_snn import PointNetSNN
from models.pointnet_ann import PointNetANN
from inference.infer_modes import (
    infer_ann_full,
    infer_snn_full,
    infer_ann_slice,
    infer_snn_slice
)
from inference.plotting import plot_all_metrics

DATASET_CLASSES = {"modelnet10": 10, "modelnet40": 40}
DEFAULT_ROOTS = {
    "modelnet10": "/content/drive/MyDrive/ModelNet10",
    "modelnet40": "/content/drive/MyDrive/ModelNet40",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points",  type=int, default=1024)
    parser.add_argument("--num_slices",  type=int, default=16)
    parser.add_argument("--dataset",     type=str, default="modelnet10",
                        choices=["modelnet10", "modelnet40"])
    parser.add_argument("--data_root",   type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default="./results_inference")

    # Model checkpoints
    parser.add_argument("--checkpoint_ann", type=str, default=None,
                        help="Path to ANN checkpoint (.pth)")
    parser.add_argument("--checkpoint_snn", type=str, default=None,
                        help="Path to SNN checkpoint (.pth)")

    # SNN model flags (must match what was used during training)
    parser.add_argument("--learnable_lif",  action="store_true")
    parser.add_argument("--local_knn",      action="store_true")
    parser.add_argument("--knn_k",          type=int, default=16)
    parser.add_argument("--bidirectional",  action="store_true")

    # Inference options
    parser.add_argument("--exit_threshold", type=float, default=0.8,
                        help="Confidence threshold for early exit (default 0.8)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    num_classes = DATASET_CLASSES[args.dataset]
    data_root   = args.data_root or DEFAULT_ROOTS[args.dataset]

    test_ds = ModelNetDataset(
        root=data_root, split="test", num_points=args.num_points
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    results = {}

    # ----- ANN -----
    if args.checkpoint_ann:
        print(f"\nLoading ANN from {args.checkpoint_ann}...")
        model_ann = PointNetANN(
            point_dims=[128, 256, 512],
            temporal_dim=512,
            num_classes=num_classes
        ).to(device)
        model_ann.load_state_dict(
            torch.load(args.checkpoint_ann, map_location=device), strict=False
        )

        print("Running ANN + Full...")
        results['ANN+Full'] = infer_ann_full(model_ann, test_loader, device)
        print(f"  ANN Full Acc: {results['ANN+Full']['final_accuracy']:.4f}")

        print("Running ANN + Slice...")
        results['ANN+Slice'] = infer_ann_slice(
            model_ann, test_loader, device,
            num_slices=args.num_slices,
            exit_threshold=args.exit_threshold
        )
        print(f"  ANN Slice Acc:   {results['ANN+Slice']['final_accuracy']:.4f}")
        print(f"  ANN Mean Exit:   {results['ANN+Slice']['mean_exit']:.2f} / {args.num_slices}")

    # ----- SNN -----
    if args.checkpoint_snn:
        print(f"\nLoading SNN from {args.checkpoint_snn}...")
        model_snn = PointNetSNN(
            point_dims=[128, 256, 512],
            temporal_dim=512,
            num_classes=num_classes,
            learnable_lif=args.learnable_lif,
            local_knn=args.local_knn,
            knn_k=args.knn_k,
            bidirectional=args.bidirectional,
        ).to(device)
        model_snn.load_state_dict(
            torch.load(args.checkpoint_snn, map_location=device), strict=False
        )

        print("Running SNN + Full...")
        results['SNN+Full'] = infer_snn_full(model_snn, test_loader, device)
        print(f"  SNN Full Acc: {results['SNN+Full']['final_accuracy']:.4f}")

        print("Running SNN + Slice...")
        results['SNN+Slice'] = infer_snn_slice(
            model_snn, test_loader, device,
            num_slices=args.num_slices,
            exit_threshold=args.exit_threshold
        )
        print(f"  SNN Slice Acc:   {results['SNN+Slice']['final_accuracy']:.4f}")
        print(f"  SNN Mean Exit:   {results['SNN+Slice']['mean_exit']:.2f} / {args.num_slices}")
        print(
            f"  Efficiency:      exiting at t={results['SNN+Slice']['mean_exit']:.1f} "
            f"saves {(1 - results['SNN+Slice']['mean_exit']/args.num_slices)*100:.1f}% compute"
        )

    # ----- Plots -----
    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nGenerating plots in {args.output_dir}/...")
        plot_all_metrics(results, args.output_dir)
        print("Done. Plots saved:")
        for f in ['accuracy_vs_timestep.png', 'exit_histogram_snn.png',
                  'threshold_tradeoff.png', 'exit_cdf.png',
                  'confidence_growth.png', 'snn_vs_ann_energy.png',
                  'paper_comparison.png']:
            p = os.path.join(args.output_dir, f)
            if os.path.exists(p):
                print(f"  {f}")
    else:
        print("No checkpoints provided. Use --checkpoint_ann and/or --checkpoint_snn.")


if __name__ == "__main__":
    main()

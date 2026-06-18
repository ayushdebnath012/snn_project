import torch
import time
import argparse
from torch.utils.data import DataLoader

from data.modelnet import ModelNetDataset
from training.optimizers import build_optimizer
from training.train_loop import train_one_epoch
from training.train_full import train_full_epoch
from training.metrics import efficiency_ratio, learnable_lif_stats

from models.pointnet_snn import PointNetSNN
from models.pointnet_ann import PointNetANN


# ModelNet10 has 10 classes; ModelNet40 has 40 classes
DATASET_CLASSES = {"modelnet10": 10, "modelnet40": 40}

DEFAULT_ROOTS = {
    "modelnet10": "/content/drive/MyDrive/ModelNet10",
    "modelnet40": "/content/drive/MyDrive/ModelNet40",
}


def main():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model",  type=str, default="snn", choices=["snn", "ann"])
    parser.add_argument("--mode",   type=str, default="slice", choices=["slice", "full"])

    # Dataset — now selectable between ModelNet10 and ModelNet40
    parser.add_argument("--dataset", type=str, default="modelnet10",
                        choices=["modelnet10", "modelnet40"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override default dataset root path.")

    # Training
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_slices", type=int, default=4)
    parser.add_argument("--slicing",    type=str, default="radial",
                        choices=["radial", "fps"],
                        help="radial=original, fps=novel FPS hierarchical (HDE-inspired)")

    parser.add_argument("--aux_weight",  type=float, default=0.05,
                        help="Weight for auxiliary slice-level CE losses (lower = stronger final CE signal)")

    # Novel SNN features (from notes + SPM paper)
    parser.add_argument("--learnable_lif",  action="store_true",
                        help="Learnable tau and V_th per neuron (from notes: v,T should be learnable)")
    parser.add_argument("--local_knn",      action="store_true",
                        help="KNN neighbourhood backbone (novel, inspired by SPM SEL)")
    parser.add_argument("--knn_k",          type=int, default=16)
    parser.add_argument("--bidirectional",  action="store_true",
                        help="Bidirectional temporal SNN (inspired by SPM Time Flip)")
    parser.add_argument("--use_bn",         action="store_true",
                        help="BatchNorm before every LIF layer (sacrifices some sparsity, improves accuracy)")

    # Logging
    parser.add_argument("--log_efficiency", action="store_true",
                        help="Log spike rates and ANN vs SNN energy efficiency each epoch")

    # Debug: run exactly 1 step then exit with timing report
    parser.add_argument("--debug", action="store_true",
                        help="Run 1 step only and print runtime — use to measure cost per step before committing to a full run")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = DATASET_CLASSES[args.dataset]
    data_root   = args.data_root or DEFAULT_ROOTS[args.dataset]

    print(f"\n=== Training Config ===")
    print(f"  Dataset:       {args.dataset.upper()} ({num_classes} classes)")
    print(f"  Model:         {args.model.upper()}")
    print(f"  Mode:          {args.mode}  |  Slicing: {args.slicing}")
    if args.model == "snn":
        print(f"  LearnableLIF:  {args.learnable_lif}")
        print(f"  BatchNorm-LIF: {args.use_bn}")
        print(f"  LocalKNN:      {args.local_knn} (k={args.knn_k})")
        print(f"  Bidirectional: {args.bidirectional}")
        print(f"  NumSlices:     {args.num_slices}  ({1024 // args.num_slices} pts/slice)")
        print(f"  AuxWeight:     {args.aux_weight}")
    print()

    train_loader = DataLoader(
        ModelNetDataset(root=data_root, split="train", num_points=args.num_points),
        batch_size=args.batch_size, shuffle=True
    )

    if args.model == "snn":
        model = PointNetSNN(
            point_dims=[128, 256, 512],
            temporal_dim=512,
            num_classes=num_classes,
            learnable_lif=args.learnable_lif,
            local_knn=args.local_knn,
            knn_k=args.knn_k,
            bidirectional=args.bidirectional,
            use_bn=args.use_bn,
        ).to(device)
    else:
        model = PointNetANN(
            point_dims=[128, 256, 512],
            temporal_dim=512,
            num_classes=num_classes,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = build_optimizer(model)

    n_epochs = 1 if args.debug else args.epochs

    for epoch in range(n_epochs):
        print(f"\n===== Epoch {epoch} =====")
        epoch_start = time.time()

        if args.mode == "full":
            loss, acc = train_full_epoch(model, train_loader, optimizer, device,
                                         debug=args.debug)
        else:
            loss, aux, acc = train_one_epoch(
                model, train_loader, optimizer, device,
                num_slices=args.num_slices,
                aux_weight=args.aux_weight,
                slicing=args.slicing,
                bidirectional=args.bidirectional,
                debug=args.debug,
            )

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} | Acc {acc:.4f}")

        if args.debug:
            print(f"\n[DEBUG] === Timing report ===")
            print(f"[DEBUG]   1 step wall time : {epoch_time:.3f}s")
            print(f"[DEBUG]   batch_size        : {args.batch_size}")
            print(f"[DEBUG]   device            : {device}")
            print(f"[DEBUG] ========================")
            break

        # Efficiency logging — answers notes question: "ANN efficiency vs SNN efficiency"
        if args.log_efficiency and args.model == "snn" and args.learnable_lif:
            eff = efficiency_ratio(model)
            print(
                f"  [Efficiency] FiringRate={eff['firing_rate']:.4f} | "
                f"SNN_energy={eff['snn_energy_unit']:.4f} (norm. ANN=1.0) | "
                f"Speedup={eff['speedup']:.1f}x"
            )
            for lname, s in learnable_lif_stats(model).items():
                print(
                    f"  [LIF {lname}] "
                    f"tau={s['tau_mean']:.3f}±{s['tau_std']:.3f}  "
                    f"vth={s['vth_mean']:.3f}±{s['vth_std']:.3f}"
                )

        if (epoch + 1) % 5 == 0:
            tag = (
                f"{args.dataset}_{args.model}_{args.mode}"
                + ("_llif"  if args.learnable_lif else "")
                + ("_knn"   if args.local_knn      else "")
                + ("_bidir" if args.bidirectional   else "")
                + ("_fps"   if args.slicing == "fps" else "")
                + f"_ep{epoch+1}.pth"
            )
            torch.save(model.state_dict(), f"/content/drive/MyDrive/{tag}")
            print(f"  Saved: {tag}")


if __name__ == "__main__":
    main()

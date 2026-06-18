"""Evaluate an ASP-SNN ShapeNetPart checkpoint."""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from config import load_config, set_seed
from datasets.shapenetpart import (
    NUM_CATEGORIES,
    NUM_PARTS,
    ShapeNetPartDataset,
)
from models.asp_segmentor import ASPSegmentor
from train_shapenet import evaluate, print_per_category


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShapeNetPart")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default="configs/shapenet_seg.yaml"
    )
    parser.add_argument("--per_cat", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--t_values",
        type=int,
        nargs="*",
        default=None,
        help="Evaluate temporal budgets such as --t_values 6 8 12 16",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        cfg.batch_size = args.batch
    set_seed(cfg.seed)
    device = cfg.device

    test_ds = ShapeNetPartDataset(cfg.data_dir, "test", cfg)
    loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    cfg.num_classes = NUM_PARTS
    cfg.num_categories = NUM_CATEGORIES
    cfg.use_category = True
    cfg.in_channels = 6
    cfg.point_in_channels = 6
    model = ASPSegmentor(cfg).to(device)

    checkpoint = torch.load(
        args.ckpt, map_location=device, weights_only=False
    )
    state = checkpoint.get("model", checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    t_values = args.t_values
    if not t_values:
        t_values = list(
            getattr(cfg, "test_t_values", [int(getattr(cfg, "T", 6))])
        )
    t_values = list(dict.fromkeys(
        min(int(value), int(cfg.num_slices)) for value in t_values
    ))

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Epoch      : {checkpoint.get('epoch', '?')}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    for t_value in t_values:
        model.T = t_value
        start = time.time()
        inst_iou, cls_iou, per_cat, diagnostics = evaluate(
            model, loader, device, cfg.num_points
        )
        elapsed = time.time() - start
        print(f"\n{'=' * 56}")
        print(
            f"  T={t_value:<2d}  Instance mIoU: {inst_iou * 100:.2f}%  "
            f"Class mIoU: {cls_iou * 100:.2f}%"
        )
        print(
            f"  Time: {elapsed:.1f}s  "
            f"Slice coverage: {diagnostics['slice_coverage'] * 100:.1f}%  "
            f"Selection entropy: {diagnostics['selection_entropy']:.3f}"
        )
        if diagnostics["spike_rate"] is not None:
            print(f"  LIF spike rate: {diagnostics['spike_rate']:.3f}")
        print(f"{'=' * 56}")
        if args.per_cat:
            print_per_category(per_cat)


if __name__ == "__main__":
    main()

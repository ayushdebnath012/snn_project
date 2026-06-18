import torch
from data.slicing import slice_radial
from inference.ordering_entropy import entropy_order_slices

def evaluate_entropy_ordering(model, dataset, device, threshold=0.5, T=16):
    model.eval()
    total = 0
    correct = 0
    total_exit = 0

    for pts, label in dataset:
        pts = pts.unsqueeze(0).to(device)

        slice_idx = slice_radial(pts[0], T=T)

        pred, t_exit, order = entropy_order_slices(
            model, pts, slice_idx, device, threshold
        )

        total += 1
        correct += int(pred == label.item())
        total_exit += t_exit

    acc = correct / total
    avg_exit = total_exit / total

    return acc, avg_exit

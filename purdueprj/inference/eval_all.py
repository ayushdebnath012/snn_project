import torch
from data.modelnet import ModelNetDataset
from data.slicing import slice_radial


def eval_full(model, dataset, device):
    model.eval()
    total = 0
    correct = 0

    for pts, label in dataset:
        pts = pts.unsqueeze(0).to(device)
        logits = model.forward_full(pts)
        pred = logits.argmax(dim=-1).item()

        correct += int(pred == label.item())
        total += 1

    return correct/total


def eval_slice_ann(model, dataset, device, num_slices=12):

    model.eval()
    total = 0
    correct = 0

    for pts, label in dataset:

        pts = pts.unsqueeze(0).to(device)
        slice_idx = slice_radial(pts[0], T=num_slices)

        running = torch.zeros(
            1,
            model.temporal.fc1.in_features,
            device=device
        )

        for t in range(num_slices):
            idx = slice_idx[t]
            pts_slice = pts[:, idx, :]

            per_point = model.backbone(pts_slice)
            slice_feat = per_point.mean(dim=1)

            running += slice_feat

        running /= num_slices

        logits = model.forward_step(running)
        pred = logits.argmax(dim=-1).item()

        correct += int(pred == label.item())
        total += 1

    return correct/total

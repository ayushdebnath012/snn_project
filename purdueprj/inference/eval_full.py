import torch
from data.slicing import slice_radial

def accuracy_per_timestep(model, dataset, device, T=12):
    """
    For each t, compute accuracy if the model exited EXACTLY at t.
    """
    correct = [0]*T
    total = 0

    for pts, label in dataset:
        total += 1
        pts = pts.unsqueeze(0).to(device)
        slice_idx = slice_radial(pts[0], T=T)

        model.reset_state(1, device)

        logits_all = []

        for t, idx in enumerate(slice_idx):
            pts_slice = pts[:, idx, :]
            logits = model.forward_step(pts_slice)
            logits_all.append(logits)

        # Compute accuracy at each timestep
        for t in range(T):
            pred = logits_all[t].argmax(dim=-1).item()
            correct[t] += int(pred == label.item())

    acc = [c/total for c in correct]
    return acc

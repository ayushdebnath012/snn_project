import torch
import torch.nn.functional as F
from data.slicing import slice_radial

def fixed_order_infer(model, pts, slice_idx_list, device, threshold=0.5):
    model.reset_state(batch_size=1, device=device)

    for t, idx in enumerate(slice_idx_list):
        pts_slice = pts[:, idx, :]
        logits = model.forward_step(pts_slice)

        # early exit check
        probs = F.softmax(logits, dim=-1)
        top2 = probs.topk(2, dim=-1).values
        margin = (top2[:,0] - top2[:,1]).item()

        if margin > threshold:
            pred = logits.argmax(dim=-1).item()
            return pred, t+1

    # no early exit, return final
    pred = logits.argmax(dim=-1).item()
    return pred, len(slice_idx_list)


def evaluate_fixed_order(model, dataset, device, threshold=0.5, T=16):
    total = 0
    correct = 0
    total_exit = 0

    for pts, label in dataset:
        pts = pts.unsqueeze(0).to(device)
        slice_idx = slice_radial(pts[0], T=T)

        pred, t_exit = fixed_order_infer(model, pts, slice_idx, device, threshold)
        total += 1
        correct += int(pred == label.item())
        total_exit += t_exit

    return correct/total, total_exit/total

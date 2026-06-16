import torch
import torch.nn.functional as F


def entropy_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(dim=-1)


def get_all_lif_layers(model):
    """
    Collect ALL LIF layers from backbone and temporal modules.
    Handles TemporalSNN (.lif1/.lif2) and
    BidirectionalTemporalSNN (.fwd_lif1/.fwd_lif2/.bwd_lif1/.bwd_lif2).
    """
    layers = []

    # Backbone LIF layers — works for PointNetBackbone + LocalKNNBackbone
    if hasattr(model.backbone, 'layers'):
        for lif in model.backbone.layers:
            layers.append(lif)

    # Temporal — handle TemporalSNN and BidirectionalTemporalSNN
    temporal = model.temporal
    if hasattr(temporal, 'lif1') and hasattr(temporal, 'lif2'):
        layers.append(temporal.lif1)
        layers.append(temporal.lif2)
    elif hasattr(temporal, 'fwd_lif1'):
        layers.append(temporal.fwd_lif1)
        layers.append(temporal.fwd_lif2)
        layers.append(temporal.bwd_lif1)
        layers.append(temporal.bwd_lif2)

    return layers


def simulate_slice(model, pts_slice):
    """
    Forward a slice WITHOUT affecting real membrane states.
    """
    lif_layers = get_all_lif_layers(model)

    # Save membrane states
    saved_states = [layer.mem.clone() for layer in lif_layers]

    # Simulate forward
    logits = model.forward_step(pts_slice)

    # Restore states
    for layer, mem in zip(lif_layers, saved_states):
        layer.mem = mem.clone()

    return logits


def entropy_order_slices(model, pts, slice_idx_list, device, threshold=0.5):
    """
    Order slices using entropy and check early exit.
    """
    remaining = list(range(len(slice_idx_list)))
    ordered = []

    model.reset_state(batch_size=1, device=device)

    logits_real = None

    for t in range(len(slice_idx_list)):

        entropies = []

        # Evaluate each remaining slice
        for i in remaining:
            idx = slice_idx_list[i]
            pts_slice = pts[:, idx, :]
            logits_sim = simulate_slice(model, pts_slice)
            H = entropy_from_logits(logits_sim)
            entropies.append((H.item(), i))

        # Pick lowest-entropy slice
        _, best_i = min(entropies, key=lambda x: x[0])

        # Feed it for REAL
        idx = slice_idx_list[best_i]
        pts_slice = pts[:, idx, :]
        logits_real = model.forward_step(pts_slice)

        ordered.append(best_i)
        remaining.remove(best_i)

        # Early exit check (margin-based)
        probs = F.softmax(logits_real, dim=-1)
        top2 = probs.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).item()

        if margin > threshold:
            pred = logits_real.argmax(dim=-1).item()
            return pred, t + 1, ordered

    # No early exit → final prediction
    pred = logits_real.argmax(dim=-1).item()
    return pred, len(slice_idx_list), ordered

import torch
import time
from training.loss_functions import ce_loss_final, ce_loss_aux
from training.metrics import accuracy
from data.slicing import slice_radial_batch, slice_fps_hierarchical_batch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    num_slices=16,
    aux_weight=0.3,
    verbose_every=20,
    slicing="radial",
    bidirectional=False,
    debug=False,
):
    model.train()

    total_final_loss = 0.0
    total_aux_loss   = 0.0
    total_acc        = 0.0
    total_acc_first  = 0.0
    total_acc_mid    = 0.0
    count = 0

    start_time    = time.time()
    total_batches = len(dataloader)

    print(f"\n[TRAIN-SLICE] Total batches: {total_batches}")

    for batch_idx, (pts, labels) in enumerate(dataloader):

        pts    = pts.to(device)
        labels = labels.to(device)
        B = pts.size(0)

        if hasattr(model, "reset_state"):
            model.reset_state(batch_size=B, device=device)

        # --- Slicing: radial (original) or fps (novel HDE-inspired) ---
        if slicing == "fps":
            pts_slices = slice_fps_hierarchical_batch(pts, T=num_slices)
        else:
            batch_indices  = slice_radial_batch(pts, T=num_slices)
            gather_indices = batch_indices.unsqueeze(-1).expand(-1, -1, 3)
            pts_sorted     = torch.gather(pts, 1, gather_indices)
            B2, N, C       = pts_sorted.shape
            points_per_slice = N // num_slices
            pts_slices     = pts_sorted.view(B2, num_slices, points_per_slice, C)

        # --- Forward through all slices ---
        logits_all = []
        for t in range(num_slices):
            logits_t = model.forward_step(pts_slices[:, t, :, :])
            logits_all.append(logits_t)

        # --- Bidirectional: fuse forward+backward to get true final logits ---
        if bidirectional and hasattr(model, "finalize"):
            logits_final = model.finalize()
        else:
            logits_final = logits_all[-1]

        loss_final = ce_loss_final(logits_final, labels)
        loss_aux   = ce_loss_aux(logits_all, labels, aux_weight)
        loss       = loss_final + loss_aux

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc_final = accuracy(logits_final, labels)
        acc_mid   = accuracy(logits_all[num_slices // 2], labels)
        acc_first = accuracy(logits_all[0], labels)

        total_final_loss += loss_final.item()
        total_aux_loss   += loss_aux.item()
        total_acc        += acc_final
        total_acc_first  += acc_first
        total_acc_mid    += acc_mid
        count += 1

        if debug:
            step_time = time.time() - start_time
            print(f"[DEBUG] 1 step done | step_time={step_time:.3f}s | loss={loss_final.item():.4f} | acc={acc_final:.3f}")
            break

        if (batch_idx + 1) % verbose_every == 0:
            elapsed = time.time() - start_time
            lr      = optimizer.param_groups[0]["lr"]
            print(
                f"[{batch_idx+1}/{total_batches}] "
                f"Loss: {loss_final.item():.4f} | "
                f"Aux: {loss_aux.item():.4f} | "
                f"AccEnd: {total_acc/count:.3f} | "
                f"AccMid: {total_acc_mid/count:.3f} | "
                f"Acc1: {total_acc_first/count:.3f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    print("[TRAIN-SLICE] Epoch complete.")
    return (
        total_final_loss / count,
        total_aux_loss   / count,
        total_acc        / count,
    )

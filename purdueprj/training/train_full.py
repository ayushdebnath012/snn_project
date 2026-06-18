import torch
import time
from training.loss_functions import ce_loss_final
from training.metrics import accuracy


def train_full_epoch(model, dataloader, optimizer, device, verbose_every=20, debug=False):

    model.train()

    total_loss = 0
    total_acc = 0
    count = 0

    start_time = time.time()
    total_batches = len(dataloader)

    print(f"\n[TRAIN-FULL] Total batches: {total_batches}")

    for batch_idx, (pts, labels) in enumerate(dataloader):

        pts = pts.to(device)
        labels = labels.to(device)

        B = pts.size(0)

        if hasattr(model, "reset_state"):
            model.reset_state(batch_size=B, device=device)

        logits = model.forward_full(pts)

        loss = ce_loss_final(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        batch_acc = accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += batch_acc
        count += 1

        if debug:
            step_time = time.time() - start_time
            print(f"[DEBUG] 1 step done | step_time={step_time:.3f}s | loss={loss.item():.4f} | acc={batch_acc:.3f}")
            break

        if (batch_idx + 1) % verbose_every == 0:

            elapsed = time.time() - start_time
            avg_loss = total_loss / count
            avg_acc = total_acc / count
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"[{batch_idx+1}/{total_batches}] "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {batch_acc:.4f} | "
                f"AvgLoss: {avg_loss:.4f} | "
                f"AvgAcc: {avg_acc:.4f} | "
                f"GradNorm: {grad_norm:.3f} | "
                f"LR: {lr:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    print("[TRAIN-FULL] Epoch complete.")

    return total_loss / count, total_acc / count

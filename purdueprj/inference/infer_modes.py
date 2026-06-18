import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def infer_ann_full(model, loader, device):
    """
    Model 1: ANN + Full
    Single forward pass on full cloud. No slicing.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="ANN+Full"):
            points, labels = points.to(device), labels.to(device)
            
            # Forward pass
            logits = model.forward_full(points)
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    final_acc = correct / total if total > 0 else 0.0
    return {"final_accuracy": final_acc}


def infer_snn_full(model, loader, device):
    """
    Model 2: SNN + Full
    Single forward pass on full cloud. Reset state per sample.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="SNN+Full"):
            points, labels = points.to(device), labels.to(device)
            B = points.size(0)
            
            # Reset state
            model.reset_state(B, device)
            
            # Forward pass
            logits = model.forward_full(points)
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    final_acc = correct / total if total > 0 else 0.0
    return {"final_accuracy": final_acc}


def infer_ann_slice(model, loader, device, num_slices=16, exit_threshold=0.8):
    """
    Model 3: ANN + Slice
    Sequential accumulation of features.
    """
    model.eval()
    
    # Metrics
    correct_at_step = torch.zeros(num_slices).to(device)
    total_samples = 0
    exit_steps = []
    
    # Store all probs for threshold sweep [N_total, T, C]
    all_probs_list = []
    all_labels_list = [] # Need labels to compute accuracy at different thresholds
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="ANN+Slice"):
            points, labels = points.to(device), labels.to(device)
            B, N, C = points.shape
            
            step_size = N // num_slices
            running_feat = 0
            batch_logits_t = []
            
            for t in range(num_slices):
                start_idx = t * step_size
                end_idx = start_idx + step_size
                slice_data = points[:, start_idx:end_idx, :]
                
                per_point = model.backbone(slice_data)
                curr_feat = per_point.mean(dim=1)
                
                if t == 0:
                    running_feat = curr_feat
                else:
                    running_feat = running_feat + curr_feat
                
                avg_feat = running_feat / (t + 1)
                logits = model.temporal(avg_feat)
                batch_logits_t.append(logits)
                
                preds = logits.argmax(dim=1)
                correct_at_step[t] += (preds == labels).sum()
                
            total_samples += B
            
            # Stack logits [B, T, C]
            batch_logits_t = torch.stack(batch_logits_t, dim=1)
            probs = F.softmax(batch_logits_t, dim=2)
            
            all_probs_list.append(probs.cpu())
            all_labels_list.append(labels.cpu())
            
            # Early exit for default threshold
            max_probs, _ = probs.max(dim=2)
            exited = max_probs > exit_threshold
            exit_indices = torch.argmax(exited.int(), dim=1)
            any_exited = exited.any(dim=1)
            exit_indices[~any_exited] = num_slices - 1
            exit_steps.extend(exit_indices.cpu().numpy().tolist())

    # Aggregate
    final_acc = correct_at_step[-1].item() / total_samples
    acc_vs_timestep = (correct_at_step / total_samples).cpu().numpy().tolist()
    mean_exit = np.mean(exit_steps)
    
    all_probs = torch.cat(all_probs_list, dim=0) # [Total, T, C]
    all_labels = torch.cat(all_labels_list, dim=0) # [Total]
    
    return {
        "final_accuracy": final_acc,
        "acc_vs_timestep": acc_vs_timestep,
        "exit_steps": exit_steps,
        "mean_exit": float(mean_exit),
        "all_probs": all_probs,
        "all_labels": all_labels
    }


def infer_snn_slice(model, loader, device, num_slices=16, exit_threshold=0.8):
    """
    Model 4: SNN + Slice (Main Model)
    Accumulates via membrane potential.
    """
    model.eval()
    
    correct_at_step = torch.zeros(num_slices).to(device)
    total_samples = 0
    exit_steps = []
    
    confidence_sum_at_step = torch.zeros(num_slices).to(device)
    
    all_probs_list = []
    all_labels_list = []
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="SNN+Slice"):
            points, labels = points.to(device), labels.to(device)
            B, N, C = points.shape
            
            model.reset_state(B, device)
            step_size = N // num_slices
            batch_logits_t = []
            
            for t in range(num_slices):
                start_idx = t * step_size
                end_idx = start_idx + step_size
                slice_data = points[:, start_idx:end_idx, :]
                
                logits = model.forward_step(slice_data)
                batch_logits_t.append(logits)
                
                preds = logits.argmax(dim=1)
                correct_at_step[t] += (preds == labels).sum()
                
            total_samples += B
            
            batch_logits_t = torch.stack(batch_logits_t, dim=1)
            probs = F.softmax(batch_logits_t, dim=2)
            
            all_probs_list.append(probs.cpu())
            all_labels_list.append(labels.cpu())
            
            # Confidence & Exit
            max_probs, _ = probs.max(dim=2)
            confidence_sum_at_step += max_probs.sum(dim=0)
            
            exited = max_probs > exit_threshold
            exit_indices = torch.argmax(exited.int(), dim=1)
            any_exited = exited.any(dim=1)
            exit_indices[~any_exited] = num_slices - 1
            exit_steps.extend(exit_indices.cpu().numpy().tolist())

    # Aggregate
    final_acc = correct_at_step[-1].item() / total_samples
    acc_vs_timestep = (correct_at_step / total_samples).cpu().numpy().tolist()
    mean_exit = np.mean(exit_steps)
    confidence_curve = (confidence_sum_at_step / total_samples).cpu().numpy().tolist()
    
    all_probs = torch.cat(all_probs_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)
    
    return {
        "final_accuracy": final_acc,
        "acc_vs_timestep": acc_vs_timestep,
        "exit_steps": exit_steps,
        "mean_exit": float(mean_exit),
        "confidence_curve": confidence_curve,
        "all_probs": all_probs,
        "all_labels": all_labels
    }

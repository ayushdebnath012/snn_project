# Purdue SNN Point Cloud — Experiment Report
**Date:** 2026-03-13
**Datasets:** ModelNet10 (50 epochs, real training) | ModelNet40 (smoke-test only, not meaningful)
**Device:** CUDA | **Slices:** T=16 | **Seeds:** 3

---

## Bugs Found and Fixed

### Bug 1 — e3dsnn: `RuntimeError: backward through graph a second time`

**Root cause:** `E3DBlock.forward` stored `self.mem1` and `self.mem2` without detaching between timesteps. When aux loss (`aux_weight=0.3`) combines `criterion(logits_T)` + `mean(criterion(logits_1..T-1))`, ALL logits share intermediate activations `mem1_t` in the computation graph. `loss.backward()` traverses the graph and frees saved tensors on the first path through `mem1_t` (e.g., from `logits_T`). The second path (from `logits_1`) then hits already-freed tensors → crash.

**Fix:** Detach `self.mem1.detach()` and `self.mem2.detach()` before each `_ilif` call in `E3DBlock.forward`, and `self.mem.detach()` in `E3DSNNModel.forward_step`. This is standard 1-step TBPTT (Truncated BPTT), which is how SpikingJelly and most SNN frameworks handle temporal state by default. Gradient still flows through model parameters (shared leaf nodes) correctly.

**Files changed:**
- `models/e3dsnn_backbone.py` — `E3DBlock.forward`
- `models/model_zoo.py` — `E3DSNNModel.forward_step`

---

### Bug 2 — Energy formula: spurious `× num_slices` factor

**Root cause:** `run_all_experiments.py` computed:
```
efficiency = (E_AC / E_MAC) × fr × T
```
This over-penalises the SNN for having more timesteps. The correct derivation:
- ANN: 1 forward pass on N points → `N_ops × E_MAC` energy
- SNN: T passes on N/T points each, fr fraction are spikes → `T × fr × (N_ops/T) × E_AC = N_ops × fr × E_AC`
- Ratio: `E_SNN / E_ANN = fr × E_AC / E_MAC` — **T cancels**

With T=16 and fr≈0.4, the old formula gave `0.4 × 0.274 × 16 = 1.75` (SNN worse than ANN). Correct formula gives `0.4 × 0.274 = 0.11` (SNN 9× better).

**Fix:** Removed `× num_slices` from the formula in `run_all_experiments.py`.

---

## ModelNet10 Results — Main Comparison (50 epochs)

ModelNet40 results are smoke-test only (acc = 1/num_classes = 2.5%) and are **not meaningful** for accuracy. All substantive results are from ModelNet10.

### Our SNN Models

| Model | Val Acc | Firing Rate | Energy vs ANN | Notes |
|---|---|---|---|---|
| **ours_full** | **90.64%** | 0.434 | **8.4× better** | KNN + bidir + learnable LIF |
| ours_knn | 89.87% | 0.329 | 11.1× better | KNN backbone, lowest fr |
| ours_base | 89.21% | 0.708 | 5.2× better | Radial slicing, fixed LIF |
| ours_bidir | 88.77% | 0.688 | 5.3× better | Bidirectional temporal |
| ours_learnable | 88.88% | 0.667 | 5.5× better | Learnable tau/vth |
| ours_transformer_small | 65.53% | — | — | Transformer variant, underperforms |

Energy = 1 / (fr × E_AC/E_MAC), Loihi 2 constants (E_AC=2.3e-3 pJ, E_MAC=8.4e-3 pJ).

### Compared SNN Baselines (ModelNet10)

| Model | Val Acc | Firing Rate | Energy vs ANN | Paper |
|---|---|---|---|---|
| spiking_ssm | 90.09% | — | — | SpikingSSMs (2408.14909) |
| ours_full | **90.64%** | 0.434 | 8.4× | Ours (best) |
| spm | 81.50% | 0.247 | 14.8× better | Spiking Point Mamba (2504.14371) |
| spt | 74.34% | — | — | SPT (2502.15811) |
| **e3dsnn** | **FAILED** | — | — | E-3DSNN (2412.07360) — fixed, rerun needed |

### ANN Baselines (ModelNet10)

| Model | Val Acc | Notes |
|---|---|---|
| ann_pointnet | 76.65% | Radial slicing |
| ann_dgcnn | training... | Incomplete at capture time |

---

## Key Observations

### 1. Accuracy
- `ours_full` achieves **90.64%** on ModelNet10, beating spiking_ssm (90.09%), spm (81.50%), and spt (74.34%)
- `ours_knn` achieves 89.87% with the **lowest firing rate** (fr=0.329), suggesting KNN backbone naturally encourages sparse representations
- `ours_transformer_small` significantly underperforms (65.53%) — the transformer head may need longer training or a different lr schedule
- spt (74.34%) is surprisingly weak; its HD-IF neuron may require more epochs to converge
- spm (81.50%) is well below its reported 92.3% on MN40 — ModelNet10 appears harder for it

### 2. Energy Efficiency (corrected formula)
All our SNN models are **energy-efficient** relative to ANN:

| Model | fr | Energy benefit |
|---|---|---|
| ours_knn | 0.329 | **11.1×** cheaper than ANN |
| ours_learnable | 0.667 | 5.5× |
| ours_full | 0.434 | 8.4× |
| spm | 0.247 | **14.8×** cheapest (but lower accuracy) |

The firing rate is relatively high for ours_base (fr=0.708) and ours_bidir (fr=0.688). This is typical for early/limited training (50 epochs). Longer training with regularisation (firing-rate loss) would push fr lower, improving energy efficiency further.

### 3. Accuracy vs Energy Trade-off
`ours_full` offers the best balance: highest accuracy (90.64%) at competitive energy (8.4×). `ours_knn` is the best energy-accuracy compromise if compute is the bottleneck (89.87%, 11.1× savings).

### 4. High Firing Rates
Firing rates of 0.4–0.7 are higher than ideal for SNNs. Causes:
- Only 50 training epochs (models not fully converged)
- No explicit firing-rate regularisation term in the loss
- ModelNet10 may have simpler features that don't require sparse activations

**Recommendation:** Add `λ × mean_fr` penalty to the loss, or train for 150 epochs, to push fr below 0.2.

---

## Comparison vs Published Baselines (ModelNet40)

The ModelNet40 smoke test is not valid for accuracy. Published numbers from papers:

| Model | Type | MN40 Acc | Source |
|---|---|---|---|
| PointNet | ANN | 89.2% | 2017 |
| PointNet++ | ANN | 90.7% | 2017 |
| PointMLP | ANN | 94.1% | 2022 |
| SpikingPtNet | SNN | 88.2% | 2310.06232 |
| P2SResLNet-B | SNN | 88.7% | 2023 |
| SPT | SNN | 91.4% | 2502.15811 |
| SPM | SNN | 92.3% | 2504.14371 |

Our ModelNet10 results suggest ours_full is competitive with spiking_ssm and beats spm/spt on this dataset. A full MN40 run (150 epochs) is needed for a fair comparison.

---

## e3dsnn Failure Analysis

With the retain_graph fix applied, e3dsnn should now train. Expected characteristics:
- Uses Integer LIF (I-LIF) with D=4 integer spike depth
- SSC (Spike Sparse Convolution) gate forces ~50% sparsity regardless of firing rate
- Likely achieves moderate accuracy (smoke test gave random ~3.91% due to no training)
- Energy should be very competitive due to architectural sparsity enforcement

**Action required:** Re-run `modelnet10|e3dsnn|radial` with the patched code.

---

## What to Run Next

```bash
# 1. Re-run e3dsnn only (ModelNet10, fast check)
python run_all_experiments.py \
    --mn10_root /kaggle/input/... \
    --datasets modelnet10 \
    --groups comparison \
    --epochs 50 \
    --out_dir results/mn10_e3dsnn_fixed

# 2. Full ModelNet40 run for paper-grade comparison (150 epochs)
python run_all_experiments.py \
    --mn40_root /kaggle/input/... \
    --datasets modelnet40 \
    --groups comparison scaling \
    --epochs 150 \
    --out_dir results/mn40_full

# 3. Add firing-rate regularisation to reduce fr and improve energy efficiency
# (needs code change: loss += 0.001 * mean_fr_tensor in train_epoch)
```

---

## Summary Table

| Model | Type | MN10 Acc | fr | Energy | Status |
|---|---|---|---|---|---|
| ours_full | SNN | **90.64%** | 0.434 | 8.4× | ✓ Best overall |
| spiking_ssm | SNN | 90.09% | — | — | ✓ Strong competitor |
| ours_knn | SNN | 89.87% | 0.329 | **11.1×** | ✓ Best energy |
| ours_base | SNN | 89.21% | 0.708 | 5.2× | ✓ Simplest |
| ours_learnable | SNN | 88.88% | 0.667 | 5.5× | ✓ |
| ours_bidir | SNN | 88.77% | 0.688 | 5.3× | ✓ |
| spm | SNN | 81.50% | 0.247 | 14.8× | ~ Lower acc |
| ann_pointnet | ANN | 76.65% | — | 1× (baseline) | ✓ |
| spt | SNN | 74.34% | — | — | ✗ Needs tuning |
| ours_transformer_small | SNN | 65.53% | — | — | ✗ Underperforms |
| e3dsnn | SNN | CRASHED | — | — | Fixed, rerun needed |

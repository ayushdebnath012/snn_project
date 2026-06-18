# Strict NeurIPS/ICML/ICLR Reviewer Critique

*Use this document to prompt an LLM (e.g., Claude/GPT) to critique the paper
 from the perspective of an adversarial top-venue reviewer.*

---

## Prompt to use

Paste the following into a new Claude conversation with your paper PDF attached:

---

**Prompt:**

You are a **strict, senior reviewer for NeurIPS 2026**.
Your job is to provide a rigorous, adversarial critique of the attached paper.
Do NOT be kind — be specific, technical, and demanding, exactly as a real
NeurIPS reviewer would be. Focus on:

### 1. Novelty
- Is the proposed method (FPS hierarchical slicing + learnable LIF + bidirectional temporal) truly novel?
- How does it differ from SPM (2504.14371), SPT (2502.15811), SpikingSSMs (2408.14909)?
- What is the key unique contribution beyond "we apply SNN to point clouds with a different slicer"?

### 2. Experimental rigour
- Are the baselines fair? Same dataset split, same # of points per sample?
- Is ModelNet40 accuracy the right metric, or should we use OBJ-BG / OBJ-ONLY / PB-T50-RS from ScanObjectNN (harder)?
- Where are the error bars / standard deviations across 3–5 runs?
- Are the hyperparameters (tau, vth, T, num_slices) tuned on the test set? (Cherry-picking concern.)
- The ANN→SNN conversion: was it compared to proper conversion baselines (SNNTorch, spikingjelly)?

### 3. Scaling claims
- "Our method scales better" — scaling in parameters is necessary but not sufficient.
  What is the sample efficiency? The compute budget at equivalent #FLOPs?
- Were larger models trained for the same number of epochs or the same wall-clock time?
  Bigger models need more epochs to converge.
- The XL model likely overfits ModelNet40 (only ~9K training samples). How was regularisation handled?

### 4. Efficiency claims
- The energy proxy (firing_rate × E_AC) is a rough estimate. Real neuromorphic hardware
  latency depends on memory access patterns, not just #AC operations.
- How does the early-exit approach compare to a well-calibrated ANN with fewer layers?
- Is there an actual hardware demo, or just a theoretical estimate?

### 5. Theoretical motivation
- Why does FPS slicing help SNNs specifically (vs ANNs)? Is there a theoretical argument,
  or only empirical evidence on one dataset?
- The bidirectional temporal processing requires buffering all slice embeddings before
  the backward pass. At inference, is this truly "online" (causal)?

### 6. Writing / presentation
- The paper needs a clear problem statement: what specific limitation of prior SNNs
  on point clouds are we solving?
- The method section must clearly differentiate Spike Voxel Coding (SVC, from E-3DSNN)
  from our slicing approach — reviewers will flag apparent overlap.
- All tables need statistical significance tests (Wilcoxon or bootstrap CI).

---

## Key weaknesses to address BEFORE submission

Based on the above critique, here are the experiments/analyses we MUST add:

### Must-have
- [ ] **ScanObjectNN benchmark** (OBJ-BG, OBJ-ONLY, PB-T50-RS) — harder than ModelNet40,
      will differentiate methods more clearly.
- [ ] **3 independent runs** with mean ± std for all main results.
- [ ] **Ablation: radial vs FPS on ANN** — does FPS only help SNNs, or does it help ANNs too?
      If it also helps ANNs, the claim is weaker.
- [ ] **Proper energy analysis**: cite Lemaire et al. 2022 (actual AC/MAC costs on Loihi/BrainScaleS),
      not just theoretical 0.9 pJ / 4.6 pJ estimates.
- [ ] **Same FLOPs / #params comparison**: when comparing ours vs SPM, make sure we're comparing
      at the same parameter count, not arbitrarily larger models.

### Should-have
- [ ] **Continual/online slicing**: demonstrate the model being fed a truly streaming LiDAR-style
      input, not a batch-partitioned slice. This is the real use case.
- [ ] **T-timestep sensitivity**: vary T from 4 to 32 and show the accuracy-efficiency frontier.
- [ ] **Comparison on a neuromorphic-relevant benchmark**: N-ModelNet40 (neuromorphic version),
      or at least report inference latency simulated on Intel Loihi 2.
- [ ] **ANN→SNN conversion at matched accuracy**: show the gap between our native SNN and the
      converted SNN shrinks as T increases, and our native SNN matches it at lower T.

### Nice-to-have
- [ ] **Gradient flow analysis**: do learnable tau/vth parameters actually converge to different
      values per neuron, and does this diversity correlate with performance?
- [ ] **Qualitative: what does each slice "see"?** Visualise which geometric region each
      temporal slice focuses on (point attention maps).
- [ ] **Application demo**: AR/VR real-time object recognition with early-exit latency measured
      end-to-end on a GPU or FPGA.

---

## Response template for rebuttal

When reviewers raise these concerns, respond with:

> "We thank Reviewer X for this important observation. We have added:
>  (1) ScanObjectNN results in Table Y (Appendix A) showing [result];
>  (2) Error bars across 5 seeds in Table Z;
>  (3) An ablation on [concern] in Figure W.
>  These results confirm / do not change our main claim that..."

---

## Self-checklist before submission

- [ ] All baselines use the SAME random seed for data split
- [ ] No hyperparameters tuned on test set
- [ ] Reproducibility: code + pretrained weights released
- [ ] All figures have axis labels, units, and captions that stand alone
- [ ] Related work cites: PointNet, PointNet++, DGCNN, PointMamba, SPM, SPT, E-3DSNN,
      SpikingSSMs, Spiking PointNet (2310.06232), P2SResLNet
- [ ] Energy comparison cites Lemaire 2022 or Christensen 2022 (Loihi benchmarks)
- [ ] Paper clearly states: training T, inference T, threshold θ, all as hyperparameters
      with sensitivity analysis

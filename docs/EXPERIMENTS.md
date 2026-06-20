# SpikeGAT ModelNet experiments

## Architecture

Both full runners use:

- four dynamic `k=20` graph-convolution stages;
- continuous attention-gated Max-First aggregation;
- identity-initialized attention gates;
- MPR/APTEC pseudo-temporal spiking with `T=4`;
- ANN teacher weight transfer and logit distillation;
- paper-aligned scaling and translation augmentation;
- single-pass checkpoint selection.

ModelNet40 additionally caches one canonical teacher distribution per training
shape to avoid a second graph-teacher forward pass in every student batch.

## Required reporting

For every completed run preserve:

- checkpoint directory;
- `history.json`;
- `final_metrics.json`;
- seed, epoch counts, batch size, GPU model, and PyTorch version.

Report `single_pass_oa` as the primary result. Label `scale_tta_oa` separately.
Do not claim either paper target has been beaten until a completed run records a
strictly higher single-pass value.

# Experiment registry

This file records maintained experiment families. Add a section when a new
dataset or task is contributed.

## ModelNet classification

### Architecture

The ModelNet10 and ModelNet40 full runners use:

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

For every completed experiment preserve:

- checkpoint directory;
- training history and final metrics;
- seed, epoch counts, batch size, GPU model, and framework version;
- the dataset's official split and evaluation protocol.

For ModelNet, report `single_pass_oa` as the primary result and label
`scale_tta_oa` separately. Do not claim either paper target has been beaten
until a completed run records a strictly higher single-pass value.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the requirements of a new dataset
integration.

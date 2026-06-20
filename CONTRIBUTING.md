# Contributing datasets and experiments

Contributions for additional point-cloud datasets and tasks are welcome. Keep
each dataset integration isolated so existing experiments remain reproducible.

## Suggested layout

```text
experiments/<dataset>/
  train_<model>_<dataset>.py
scripts/slurm/<dataset>/
  <model>_<dataset>.sbatch
docs/<DATASET>.md
```

Shared components may be added to `datasets/`, `models/`, `training/`, `tasks/`,
or `configs/`. Put reusable logic there rather than copying it between runners.
Do not commit raw datasets, checkpoints, generated metrics, logs, or papers.

## Dataset contribution checklist

1. Document how users obtain the dataset and its expected local directory
   layout. Prefer a separate, resumable downloader under `tools/`; do not
   download data silently during a training job or commit the raw dataset.
2. Add a complete training entrypoint under `experiments/<dataset>/`.
3. Make dataset paths, checkpoint paths, seeds, and important hyperparameters
   configurable through arguments or environment variables.
4. State the official evaluation protocol and distinguish primary metrics from
   test-time augmentation or supplementary metrics.
5. Support checkpoint resume for long cluster runs.
6. Add a Slurm template under `scripts/slurm/<dataset>/` when cluster execution
   is supported.
7. Update `docs/EXPERIMENTS.md` and dependency files when needed.
8. Run `python tools/validate_repo.py` and a small forward/backward smoke test.

The validator requires the maintained ModelNet baselines to remain available,
but it deliberately permits new experiment folders and shared packages.

# Kaggle execution

## Notebook setup

1. Create a GPU notebook and select a T4/P100 accelerator.
2. Enable Internet if the dataset is not attached as a Kaggle Input.
3. Clone this repository or upload it as a Kaggle dataset.
4. Run one training process at a time on a single T4.

```python
!git clone https://github.com/ayushdebnath012/ASP-SNN.git
%cd ASP-SNN
!python experiments/kaggle/spikegat/modelnet40.py
```

The ModelNet runners search attached Kaggle Inputs first and otherwise use
`kagglehub`. Outputs are written under `/kaggle/working`.

## T4 recommendations

```python
# If the 16 GB T4 runs out of memory:
%env BATCH_SIZE=16
%env NUM_WORKERS=2
!python experiments/kaggle/spikegat/modelnet40.py
```

ModelNet40 defaults to the T4 profile:

- 150 ANN-teacher epochs;
- 180 transferred student epochs;
- cached teacher logits for student KD;
- FP16 tensor-core KNN with FP32 distance accumulation;
- 1,024 points and `k=20` for final evaluation.

## Resume and preserve results

Rerun the same script in the same session to resume. Before ending the session,
save a Kaggle notebook version with outputs or publish the checkpoint directory
as a private Kaggle dataset. `/kaggle/working` is not durable across unrelated
sessions.

The final paper-comparable number is `single_pass_oa` in
`final_metrics.json`; `scale_tta_oa` is supplementary.

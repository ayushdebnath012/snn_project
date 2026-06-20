# Experiments and metrics

## ModelNet classification

| Runner | Dataset | Primary output |
|---|---|---|
| `experiments/kaggle/spikegat/modelnet10.py` | ModelNet10 | `final_metrics.json` |
| `experiments/kaggle/spikegat/modelnet40.py` | ModelNet40 | `final_metrics.json` |
| `experiments/kaggle/dgcnn/modelnet*.py` | ModelNet10/40 | baseline checkpoints/history |
| `experiments/kaggle/asp/modelnet*.py` | ModelNet10/40 | ASP checkpoints/history |
| `experiments/cluster/train_modelnet_a100.py` | ModelNet10/40 | aggregate results/history |

The updated SpikeGAT code preserves Max-First graph aggregation, uses the
supplementary MPR/APTEC equations, initializes attention as identity, transfers
ANN teacher weights, and separates single-pass OA from scale-TTA OA.

ModelNet40 additionally caches one canonical teacher distribution per training
shape, avoiding a second dynamic-graph forward pass during every student batch.

## Other point-cloud tasks

| Task | Config-driven command | Standalone command |
|---|---|---|
| ScanObjectNN | `python -m tasks.train_scanobjectnn --config configs/scanobj_cls.yaml` | `python experiments/standalone/scanobjectnn.py` |
| ShapeNetPart | `python -m tasks.train_shapenetpart --config configs/shapenet_seg.yaml` | `python experiments/standalone/shapenetpart.py` |
| S3DIS | `python -m tasks.train_s3dis --config configs/s3dis_seg.yaml` | `python experiments/standalone/s3dis.py` |

Use config-driven jobs when data already exists on a cluster. Standalone jobs
are useful on Kaggle or a fresh node because they locate or download data.

## Fair comparison checklist

1. Record the exact train/test split and point count.
2. Report single-pass metrics first.
3. Label voting or TTA metrics explicitly.
4. Keep the random seed and best-checkpoint selection rule in the result file.
5. Do not treat a smoke test or teacher accuracy as a student result.
6. Preserve the complete checkpoint and `final_metrics.json` for auditability.

# Local datasets

This directory is the default location for downloaded datasets. Its contents
are ignored by Git; only this file is tracked.

Download a maintained ModelNet baseline from the repository root:

```bash
python tools/download_modelnet.py --dataset 10
python tools/download_modelnet.py --dataset 40
```

The commands create `data/ModelNet10` or `data/ModelNet40`. Point the training
runners at those directories with `MODELNET10_DIR` or `MODELNET40_DIR`.

Do not commit raw datasets here. A dataset contribution should add acquisition
instructions or a downloader, while respecting the source dataset's license
and terms.

# Data layout

Annotations, split files, and GloVe vocabularies are included in this
folder. Large video-feature files are **not** redistributed; obtain them
from their original sources (below) and place them (or symlinks) at the
listed paths.

```
data/
├── charades/                 annotations + vocab (included)
├── activitynet/              annotations + vocab (included)
├── tacos/                    annotations + vocab (included)
├── ego4d/slowfast/           annotations + vocab (included)
├── qvhighlights/             annotations, vocab, GT file (included)
│   ├── clip_slowfast.hdf5        ← QVHighlights features (download)
│   └── cache/qv_val_75.npy/.index.json   ← built by scripts/build_feature_cache.py
└── features/                 (download / build)
    ├── i3d_features.hdf5             ← Charades-CG I3D
    ├── sub_activitynet_v1-3.c3d.hdf5 ← ActivityNet-CG C3D
    ├── tacos_features_c3d.hdf5       ← TACoS C3D
    ├── ego4d_features_sf.hdf5        ← Ego4D-NLQ SlowFast
    └── cache/                        ← built by scripts/build_feature_cache.py
```

## Feature sources

| File | Dataset | Source |
|---|---|---|
| `i3d_features.hdf5` | Charades-CG | I3D features following [QMN](https://github.com/mingyao1120/QMN) (1024-d) |
| `sub_activitynet_v1-3.c3d.hdf5` | ActivityNet-CG | C3D features following [QMN](https://github.com/mingyao1120/QMN) (500-d) |
| `tacos_features_c3d.hdf5` | TACoS | C3D features following [SnAG](https://github.com/fmu2/snag_release) (4096-d) |
| `ego4d_features_sf.hdf5` | Ego4D-NLQ | SlowFast features following [SnAG](https://github.com/fmu2/snag_release) (2304-d, 200-frame protocol) |
| `clip_slowfast.hdf5` | QVHighlights | CLIP + SlowFast features following [QD-DETR](https://github.com/wjun0830/QD-DETR) (2816-d; originally distributed with Moment-DETR) |

Charades-CG and ActivityNet-CG read the raw HDF5 files directly. TACoS,
Ego4D-NLQ, and QVHighlights additionally use a deterministic frame-pooling
cache (uniform pooling to the protocol frame count), built once with:

```bash
python scripts/build_feature_cache.py --config configs/tacos.json \
    --output data/features/cache/tacos_c3d_200.npy \
    --index  data/features/cache/tacos_c3d_200.index.json
python scripts/build_feature_cache.py --config configs/ego4d.json \
    --output data/features/cache/ego4d_slowfast_200.npy \
    --index  data/features/cache/ego4d_slowfast_200.index.json
python scripts/build_feature_cache.py --config configs/qvhighlights.json \
    --output data/qvhighlights/cache/qv_val_75.npy \
    --index  data/qvhighlights/cache/qv_val_75.index.json
```

The builder pools every video appearing in the annotation files of the
config's dataset; no temporal labels are used.

## Splits

- Charades-CG / ActivityNet-CG: `test_trivial` (TT), `novel_comp` (NC),
  `novel_word` (NW) from the official
  [Compositional Temporal Grounding](https://github.com/YYJMJC/Compositional-Temporal-Grounding)
  release.
- TACoS: standard train/val split; the val split is the report split.
- Ego4D-NLQ: val split as the report split.
- QVHighlights: val split (official evaluator; `qvhl_val_gt.jsonl` is the
  public GT file consumed by the evaluator only).

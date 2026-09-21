#!/usr/bin/env python3
"""Build label-free video-level mmap caches for deterministic frame pooling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf8")
    temporary.replace(path)


def pool_frames(frames, num_clips):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or len(frames) == 0:
        raise ValueError("video feature must be a non-empty [time, dim] array")
    boundaries = np.round(
        np.arange(num_clips + 1) / num_clips * len(frames)).astype(np.int64)
    boundaries[boundaries >= len(frames)] = len(frames) - 1
    starts, ends = boundaries[:-1], boundaries[1:]
    prefix = np.concatenate((
        np.zeros((1, frames.shape[1]), dtype=np.float32),
        np.cumsum(frames, axis=0, dtype=np.float32)), axis=0)
    lengths = ends - starts
    safe_ends = np.maximum(ends, starts + 1)
    pooled = (prefix[safe_ends] - prefix[starts]) / np.maximum(
        lengths, 1)[:, None]
    repeated = lengths == 0
    if np.any(repeated):
        pooled[repeated] = frames[starts[repeated]]
    return np.asarray(pooled, dtype=np.float32)


def source_videos(dataset):
    videos, paths = set(), []
    for key in ("train_data", "test_data", "val_data"):
        path = dataset.get(key)
        if not path or path in paths:
            continue
        paths.append(path)
        for row in read_json(path):
            videos.add(str(row[0]))
    return sorted(videos), paths


def resolve_dataset_paths(dataset, root):
    resolved = dict(dataset)
    for key in ("feature_path", "train_data", "test_data", "val_data"):
        value = resolved.get(key)
        if value and not Path(value).is_absolute():
            resolved[key] = str((root / value).resolve())
    return resolved


def build(config_path, output, index_path, force=False):
    config_path = Path(config_path).resolve()
    dataset = resolve_dataset_paths(read_json(config_path)["dataset"],
                                    config_path.parent.parent)
    videos, sources = source_videos(dataset)
    shape = (len(videos), int(dataset["max_num_frames"]),
             int(dataset["frame_dim"]))
    output, index_path = Path(output).resolve(), Path(index_path).resolve()
    if output.exists() and index_path.exists() and not force:
        existing = np.load(output, mmap_mode="r")
        index = read_json(index_path)
        if (tuple(existing.shape) == shape and
                len(index["video_to_index"]) == len(videos)):
            print(f"cache already complete: {output} {shape}", flush=True)
            return
        raise ValueError("existing cache does not match; pass --force to rebuild")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npy")
    if temporary.exists():
        temporary.unlink()
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape)
    feature_path = Path(dataset["feature_path"])
    with h5py.File(feature_path, "r") as handle:
        for index, video in enumerate(videos):
            if video not in handle:
                raise KeyError(f"{video} is absent from {feature_path}")
            item = handle[video]
            if isinstance(item, h5py.Group):
                item = item["c3d_features"]
            target[index] = pool_frames(item, int(dataset["max_num_frames"]))
            if index % 25 == 0 or index + 1 == len(videos):
                print(f"{dataset['dataset']}: {index + 1}/{len(videos)} videos",
                      flush=True)
    target.flush()
    del target
    os.replace(temporary, output)

    digest = hashlib.sha256()
    for path in sources:
        digest.update(Path(path).read_bytes())
    write_json(index_path, {
        "schema_version": 1,
        "dataset": dataset["dataset"],
        "feature_path": str(feature_path),
        "shape": list(shape),
        "dtype": "float32",
        "pooling": "legacy_round_mean_200_vectorized",
        "uses_temporal_labels": False,
        "source_annotation_sha256": digest.hexdigest(),
        "source_annotations": sources,
        "video_to_index": {video: index for index, video in enumerate(videos)},
    })
    print(f"wrote {output} ({output.stat().st_size / 1024 ** 3:.2f} GiB)",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.config, args.output, args.index, args.force)


if __name__ == "__main__":
    main()

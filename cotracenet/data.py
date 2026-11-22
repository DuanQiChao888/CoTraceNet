"""Paired image manifests and binary classification metrics."""

import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_splits(manifest, wanted):
    """Validate source-group and path identities, then open only requested splits."""
    wanted = tuple(wanted)
    groups = {key: [] for key in wanted}
    sources, paths_seen = set(), set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"source", "split", "cover", "stego"} <= set(reader.fieldnames or ()):
            raise ValueError("Expected source,split,cover,stego columns")
        for row in reader:
            source, split = row["source"], row["split"]
            if not source or source in sources or split not in ("train", "val", "test"):
                raise ValueError(f"Duplicate source or invalid split: {source!r}, {split!r}")
            paths = tuple((manifest.parent / row[k]).resolve() for k in ("cover", "stego"))
            if paths[0] == paths[1] or paths_seen.intersection(paths):
                raise ValueError(f"Reused cover/stego path for {source}")
            sources.add(source)
            paths_seen.update(paths)
            if split in groups:
                groups[split].append((source, paths))
    if not all(groups.values()):
        raise ValueError(f"Missing images for a required split in {wanted}")
    return groups


def tuning_splits(manifest):
    return read_splits(manifest, ("train", "val"))


def load_image(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (256, 256):
            raise ValueError(f"Expected an 8-bit grayscale 256x256 image: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


class ImagePairs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source, paths = self.rows[index]
        pair = []
        for path in paths:
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (256, 256):
                    raise ValueError(f"Expected an 8-bit grayscale 256x256 image: {path}")
                pair.append(np.array(image, dtype=np.uint8))
        return source, torch.from_numpy(np.stack(pair)[:, None])


def classification_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite prediction scores")
    pos, neg = scores[labels == 1], np.sort(scores[labels == 0])
    # Mann-Whitney ROC AUC, awarding half a point per tied positive-negative pair.
    auc = (
        np.searchsorted(neg, pos, side="left") + np.searchsorted(neg, pos, side="right")
    ).mean() / (2 * len(neg))
    specificity = float(np.mean(neg <= 0))
    sensitivity = float(np.mean(pos > 0))
    return {
        "auc": float(auc),
        "accuracy": float(np.mean((scores > 0) == labels)),
        "balanced_accuracy": (specificity + sensitivity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

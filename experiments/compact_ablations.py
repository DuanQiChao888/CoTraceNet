"""Reproduce both exploratory compression families from paired 256×256 grayscale images.

Fit uses grouped training folds and a separate validation split. Score applies
the frozen pooled view and residual-cell view independently to each test image.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotracenet import detector, lite
from cotracenet.data import read_splits
from cotracenet.experts import fit_action, paired_discriminant
from compact_views import MODES, descriptor, fit_model, project


def combined(path):
    image, _, _, _, joint = detector.image_features(path)
    return np.concatenate((joint, descriptor(image)))


def collect(rows, workers):
    values = lite.feature_pairs(rows, workers, combined)
    return values[..., :34398], values[..., 34398:]


def run_folds(x, training_count, fit_fn):
    tr = np.arange(training_count)
    val = np.arange(training_count, len(x))
    folds = np.array_split(np.random.default_rng(lite.SEED).permutation(tr), 4)
    scores = np.empty((len(x), 2))
    for held in [*folds, val]:
        active = tr if np.array_equal(held, val) else np.setdiff1d(tr, held)
        w, b = fit_fn(x[active])
        scores[held] = (x[held] @ w + b).cpu().numpy()
    return scores, w.cpu().numpy(), float(b)


def performance(scores, n):
    return {"oof": lite.metrics(scores[:n]), "val": lite.metrics(scores[n:])}


def pooled_fit(joint, n, device, directory):
    # The original pooling study took all square roots at float32 precision
    # before accumulating in float64. At inference it takes the public joint
    # descriptor at its native precision, as does the original locked scorer.
    training_view = joint.astype(np.float32).astype(np.float64)
    trials, weights, finalists = [], [], []
    for mode in MODES:
        start = len(trials)
        for floor in (0.0, 0.0002):
            x = torch.as_tensor(project(training_view, mode, floor), device=device)
            for ridge in (1.5, 10.0):
                scores, w, b = run_folds(x, n, lambda subset: fit_model(subset, ridge))
                trials.append(
                    dict(mode=mode, floor=floor, ridge=ridge, dim=len(w), **performance(scores, n))
                )
                weights.append((w, b))
        finalists.append(
            max(
                range(start, len(trials)),
                key=lambda i: (trials[i]["oof"]["accuracy"], trials[i]["oof"]["auc"]),
            )
        )
    chosen = max(
        finalists,
        key=lambda i: (
            trials[i]["val"]["accuracy"],
            trials[i]["val"]["auc"],
            trials[i]["oof"]["accuracy"],
            -trials[i]["dim"],
        ),
    )
    row = trials[chosen]
    w, b = weights[chosen]
    np.savez_compressed(directory / "pool.npz", w=w, b=b, mode=row["mode"], floor=row["floor"])
    return {"selected": row, "candidates": trials}


def cells_fit(joint, cells, n, device, directory):
    def teacher(data):
        return lite.average(paired_discriminant(data, 1.5, 1), fit_action(data, 0.12, 512))

    reference, teacher_w, teacher_b = run_folds(torch.as_tensor(joint, device=device), n, teacher)
    # The cell head ran on CPU, the reference head on the requested device.
    # Make each source row contiguous, matching the original saved descriptors.
    x = torch.from_numpy(np.ascontiguousarray(cells))
    trials, fits = [], {}
    for ridge in (0.15, 1.5, 8.0):
        raw, w, b = run_folds(x, n, lambda subset: paired_discriminant(subset, ridge, 1))
        fits[ridge] = (w, b)
        for share in (0.2, 0.4, 0.65, 1.0):
            scores = (1 - share) * reference + share * raw
            count = len(w) + 1 + (len(teacher_w) if share < 1 else 0)
            trials.append(
                dict(ridge=ridge, share=share, parameters=count, **performance(scores, n))
            )
    winner = max(
        trials,
        key=lambda r: (
            r["oof"]["accuracy"] + r["val"]["accuracy"],
            r["oof"]["auc"] + r["val"]["auc"],
            -r["parameters"],
        ),
    )
    w, b = fits[winner["ridge"]]
    share = winner["share"]
    np.savez_compressed(
        directory / "cells.npz",
        joint_weight=teacher_w * (1 - share) if share < 1 else np.empty(0),
        cell_weight=share * w,
        bias=(1 - share) * teacher_b + share * b,
    )
    return {"selected": winner, "candidates": trials}


def fit(args):
    groups = read_splits(args.manifest, ("train", "val"))
    train, val = (sorted(groups[k], key=lambda row: row[0]) for k in ("train", "val"))
    if len(train) < 4:
        raise ValueError("Four source-grouped folds require at least four training pairs")
    joint, cells = collect(train + val, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "models"
        stage.mkdir()
        pooled = pooled_fit(joint, len(train), args.device, stage)
        two_d = cells_fit(joint, cells, len(train), args.device, stage)
        lock = {
            "seed": lite.SEED,
            "source_ids": [lite.source_hash(s) for s, _ in train + val],
            "pooled": pooled,
            "cells": two_d,
            "sha256": {p.name: detector.digest(p) for p in stage.glob("*.npz")},
        }
        (stage / "lock.json").write_text(json.dumps(lock, indent=2), encoding="utf8")
        stage.rename(args.out)
    return {"pooled": pooled["selected"], "cells": two_d["selected"]}


def score(args):
    lock = json.loads((args.fit / "lock.json").read_text(encoding="utf8"))
    pairs = read_splits(args.manifest, ("test",))["test"]
    if set(lock["source_ids"]).intersection(lite.source_hash(s) for s, _ in pairs):
        raise ValueError("Test source IDs overlap training or validation")
    for name in ("pool.npz", "cells.npz"):
        if detector.digest(args.fit / name) != lock["sha256"][name]:
            raise ValueError(f"Model differs from its training lock: {name}")
    joint, cells = collect(pairs, args.workers)
    with np.load(args.fit / "pool.npz", allow_pickle=False) as m:
        pooled = project(joint, str(m["mode"]), float(m["floor"])) @ m["w"] + m["b"]
    with np.load(args.fit / "cells.npz", allow_pickle=False) as m:
        cell_score = cells @ m["cell_weight"] + m["bias"]
        if m["joint_weight"].size:
            cell_score += joint @ m["joint_weight"]
    result = {"pooled": lite.metrics(pooled), "cells": lite.metrics(cell_score)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "label", "pooled", "cells"])
        for (name, _), p, c in zip(pairs, pooled, cell_score):
            writer.writerows((name, y, float(p[y]), float(c[y])) for y in (0, 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    training, testing = subs.add_parser("fit"), subs.add_parser("score")
    for cmd in (training, testing):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--workers", type=int, default=2)
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    testing.add_argument("--fit", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.workers < 0:
        parser.error("Choose a new output and nonnegative workers")
    torch.set_num_threads(3)
    torch.manual_seed(lite.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps((fit if args.action == "fit" else score)(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

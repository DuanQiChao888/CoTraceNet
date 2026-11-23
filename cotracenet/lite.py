"""Fit and apply a compact residual-feature detector without CNN weights."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .data import classification_metrics, read_splits
from .detector import digest, image_features
from .experts import fit_action, paired_discriminant


SEED = 73


def joint_features(path):
    return image_features(path)[-1]


def feature_pairs(rows, workers, extractor=joint_features):
    paths = [path for _, pair in rows for path in pair]
    if workers > 0:
        with ProcessPoolExecutor(workers) as pool:
            values = list(pool.map(extractor, paths, chunksize=2))
    else:
        values = [extractor(path) for path in paths]
    return np.stack(values).reshape(len(rows), 2, -1)


def average(first, second):
    return tuple(0.5 * a + 0.5 * b for a, b in zip(first, second))


def candidate_heads(x):
    """Fit all candidates inside the current fold, including feature selection."""
    all_indices = torch.arange(x.shape[-1], device=x.device)
    dense = paired_discriminant(x, 1.5, 1)
    bank = fit_action(x, 0.12, 512)
    teacher = average(dense, bank)
    yield "folded_Q", all_indices, teacher
    second = fit_action(x, 0.35, 768, count=64, seed=SEED + 1)
    yield "balanced_bank", all_indices, average(dense, second)
    yield "two_banks", all_indices, average(teacher, second)

    scale = x.flatten(0, 1).std(0, unbiased=False)
    importance = (teacher[0] * scale).abs()
    for width in (4096, 8192, 16384):
        ix = torch.topk(importance, width).indices.sort().values
        sub = x[:, :, ix]
        local = fit_action(sub, 0.12, 512)
        refit = average(paired_discriminant(sub, 1.5, 1), local)
        yield f"refit_{width}", ix, refit
        # Re-center and normalize the restricted teacher using only training sources.
        keep = teacher[0][ix]
        raw = sub @ keep
        magnitude = raw.std(unbiased=False).clamp_min(1e-8)
        restricted = keep / magnitude, -raw.mean() / magnitude
        yield f"distill_{width}", ix, average(refit, restricted)
        yield f"midpoint_{width}", ix, average(paired_discriminant(sub, 1.5, 0.25), local)


def source_hash(name):
    return hashlib.sha256(name.encode("utf8")).hexdigest()


def metrics(scores):
    labels = np.tile([0, 1], len(scores))
    return classification_metrics(labels, scores.ravel())


def fit(manifest, out, device, workers):
    if out.exists():
        raise FileExistsError(f"Choose a new model directory: {out}")
    groups = read_splits(manifest, ("train", "val"))
    train, val = (sorted(groups[k], key=lambda row: row[0]) for k in ("train", "val"))
    n = len(train)
    if n < 4:
        raise ValueError("Four source-grouped folds require at least four training pairs")
    x = torch.as_tensor(feature_pairs(train + val, workers), device=device)
    folds = np.array_split(np.random.default_rng(SEED).permutation(n), 4)
    validation = np.arange(n, len(x))
    oof_scores, candidates, parameters = {}, {}, {}
    for step, held in enumerate([*folds, validation]):
        final_fit = step == len(folds)
        available = np.arange(n) if final_fit else np.setdiff1d(np.arange(n), held)
        for name, ix, (weight, bias) in candidate_heads(x[available]):
            pred = (x[held][:, :, ix] @ weight + bias).cpu().numpy()
            if final_fit:
                parameters[name] = (
                    ix.cpu().numpy().astype(np.uint16),
                    weight.cpu().numpy(),
                    float(bias),
                )
                candidates[name] = {
                    "oof": metrics(oof_scores[name]),
                    "validation": metrics(pred),
                    "parameters": int(weight.numel()) + 1,
                }
            else:
                if name not in oof_scores:
                    oof_scores[name] = np.empty((n, 2))
                oof_scores[name][held] = pred

    def rank(name):
        row = candidates[name]
        return (
            row["oof"]["accuracy"] + row["validation"]["accuracy"],
            row["oof"]["auc"] + row["validation"]["auc"],
            -row["parameters"],
        )

    chosen = max(candidates, key=rank)
    indices, weights, bias = parameters[chosen]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        np.savez_compressed(stage / "model.npz", indices=indices, weight=weights, bias=bias)
        lock = {
            "seed": SEED,
            "selected": chosen,
            "training_sources": n,
            "validation_sources": len(val),
            "source_ids": [source_hash(s) for s, _ in train + val],
            "model_sha256": digest(stage / "model.npz"),
            "trials": candidates,
        }
        (stage / "lock.json").write_text(json.dumps(lock, indent=2), encoding="utf8")
        stage.rename(out)
    return {"selected": chosen, **candidates[chosen], "model_sha256": lock["model_sha256"]}


def load_model(folder):
    lock = json.loads((folder / "lock.json").read_text(encoding="utf8"))
    path = folder / "model.npz"
    if digest(path) != lock["model_sha256"]:
        raise ValueError("The compact model differs from its lock")
    with np.load(path, allow_pickle=False) as data:
        model = data["indices"], data["weight"], float(data["bias"])
    return lock, model


def decision(features, model):
    ix, weight, bias = model
    return features[..., ix] @ weight + bias


def score(manifest, fit_dir, out, workers):
    if out.exists():
        raise FileExistsError(f"Choose a new score CSV: {out}")
    lock, model = load_model(fit_dir)
    pairs = read_splits(manifest, ("test",))["test"]
    if set(lock["source_ids"]).intersection(source_hash(source) for source, _ in pairs):
        raise ValueError("Test source IDs overlap training or validation")
    scores = decision(feature_pairs(pairs, workers), model)
    result = metrics(scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        stage = Path(scratch) / "predictions.csv"
        with stage.open("w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "label", "cotracenet_lite"])
            for (name, _), values in zip(pairs, scores):
                writer.writerows((name, label, float(value)) for label, value in enumerate(values))
        # Link creation is exclusive even if two writers finish simultaneously.
        out.hardlink_to(stage)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    train = subs.add_parser("fit", help="Fit, rank and lock the seed-73 candidates")
    test = subs.add_parser("score", help="Evaluate a locked compact model on paired images")
    one = subs.add_parser("predict", help="Classify a single 256 × 256 grayscale image")
    for cmd in (train, test):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--workers", type=int, default=2)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for cmd in (test, one):
        cmd.add_argument("--fit", type=Path, required=True)
    one.add_argument("--image", type=Path, required=True)
    args = p.parse_args()
    if getattr(args, "workers", 0) < 0:
        p.error("--workers must be nonnegative")
    if args.action == "fit":
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        result = fit(args.manifest, args.out, torch.device(args.device), args.workers)
    elif args.action == "score":
        result = score(args.manifest, args.fit, args.out, args.workers)
    else:
        _, model = load_model(args.fit)
        value = float(decision(joint_features(args.image), model))
        result = {"score": value, "prediction": "stego" if value > 0 else "cover"}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

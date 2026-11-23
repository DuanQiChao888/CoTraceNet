"""Train and apply CoTraceNet-Lite on source-grouped pairs of grayscale images."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, softplus

from . import fullscale_fisher
from .data import read_splits
from .detector import digest
from .lite import decision, joint_features, load_model, metrics, source_hash


WIDTH = 34_398
RETAINED = 16_384
CHECKPOINTS = (5, 15, 40, 100, 200)
PENALTIES = (1.0, 0.1, 0.01)
SELECTOR_PENALTY = 0.1


def descriptors(rows, workers):
    """Yield features in source order with bounded work and buffered results."""
    paths = [path for _, pair in rows for path in pair]
    if not workers:
        yield from map(joint_features, paths)
        return
    chunk = 16 * workers
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), chunk):
            yield from pool.map(joint_features, paths[start : start + chunk], chunksize=2)


def matrix(rows, workers):
    # Fill a single array. Stacking 12,000 individual arrays would double peak RAM.
    x = torch.empty((2 * len(rows), WIDTH), dtype=torch.float64)
    for i, row in enumerate(descriptors(rows, workers)):
        if row.shape != (WIDTH,) or row.dtype != np.float64 or not np.isfinite(row).all():
            raise ValueError("Unexpected residual feature shape, dtype, or nonfinite value")
        x[i].copy_(torch.from_numpy(row))
    return x


def optimize(x, labels, penalty, limit, callback=None, *, fisher_indices=False):
    """Unconstrained FP64 L-BFGS-B, full batch, unregularized intercept."""

    def objective(candidate):
        theta = torch.as_tensor(candidate, dtype=x.dtype, device=x.device)
        weight, bias = theta[:-1], theta[-1]
        logits = x @ weight + bias
        residual = logits.sigmoid() - labels
        # Preserve the floating-point evaluation order of each published ablation.
        if fisher_indices:
            ce = (softplus(logits) - labels * logits).mean()
            dw, db = (x.T @ residual) / len(labels), residual.mean()
        else:
            ce = binary_cross_entropy_with_logits(logits, labels)
            error = residual / len(labels)
            dw, db = x.T @ error, error.sum()
        loss = ce + 0.5 * penalty * weight.dot(weight)
        gradient = torch.cat((dw + penalty * weight, db.view(1)))
        return float(loss), gradient.cpu().numpy()

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": limit, "maxcor": 15, "ftol": 1e-12, "gtol": 1e-7, "maxls": 35},
    )
    if result.status not in (0, 1) or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"L-BFGS-B stopped before producing a valid solution: {result.message}")
    return result


def readout(x, labels, val, recipe, fisher_indices):
    if recipe is not None:
        penalty, limit = recipe
        if limit == 0:
            # A stationary all-zero candidate can win on a degenerate source set.
            return (
                np.zeros(x.shape[1] + 1),
                penalty,
                0,
                [dict(ridge=penalty, iteration=0, training_loss=float(np.log(2)))],
            )
        result = optimize(x, labels, penalty, limit, fisher_indices=fisher_indices)
        # No held-out features are read in this branch.
        return (
            result.x,
            penalty,
            int(result.nit),
            [dict(ridge=penalty, iteration=int(result.nit), training_loss=float(result.fun))],
        )

    best, winner, history = None, None, []
    for penalty in PENALTIES:
        step, last_recorded = 0, -1

        def record(theta):
            nonlocal best, winner, last_recorded
            t = torch.as_tensor(theta, device=val.device)
            scores = (val @ t[:-1] + t[-1]).cpu().numpy().reshape(-1, 2)
            m = metrics(scores)
            history.append(
                dict(ridge=penalty, iteration=step, val_auc=m["auc"], val_accuracy=m["accuracy"])
            )
            key = (m["auc"], penalty, -step)
            if best is None or key > best:
                best, winner = key, theta.copy()
            last_recorded = step

        def observe(theta):
            nonlocal step
            step += 1
            if step in CHECKPOINTS:
                record(theta)

        result = optimize(x, labels, penalty, CHECKPOINTS[-1], observe, fisher_indices=fisher_indices)
        if result.nit != step:
            raise RuntimeError("L-BFGS-B iteration counter disagrees with the checkpoint history")
        if last_recorded != step:
            record(result.x)
    return winner, best[1], -best[2], history


def fit(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a new output directory: {args.out}")
    groups = read_splits(
        args.manifest, ("train", "val") if args.action == "fit" and not args.recipe_from else ("train",)
    )
    train = sorted(groups["train"])
    valrows = sorted(groups.get("val", []))
    manifest_sha = digest(args.manifest)
    info = {
        "manifest_sha256": manifest_sha,
        "training_sources": len(train),
        "validation_sources_used": len(valrows),
        "training_ids": [source_hash(s) for s, _ in train],
        "validation_ids": [source_hash(s) for s, _ in valrows],
        "threshold": 0,
        "parameters": RETAINED + 1,
        "seed": 73,
    }
    # Legacy Lite predict/score checks this union when evaluating held-out test images.
    info["source_ids"] = info["training_ids"] + info["validation_ids"]
    device = torch.device(args.device)
    history = []
    if args.action == "fit-fisher":
        info["selector"] = "fisher"
        info["selected"] = "distill_16384"
        raw = matrix(train, args.workers).reshape(len(train), 2, WIDTH)
        keep, weight, bias = fullscale_fisher.fit(raw, RETAINED, device)
    else:
        recipe = None
        keep = None
        if args.indices_from:
            lock, source = load_model(args.indices_from)
            if lock.get("selected") != "distill_16384" or lock.get("manifest_sha256") != manifest_sha:
                raise ValueError("--indices-from requires fit-fisher trained with this manifest")
            keep = np.asarray(source[0], dtype=np.int64)
            info["indices_from_sha256"] = lock["model_sha256"]
        info["selector"] = "fisher" if keep is not None else "logistic"
        info["selected"] = "iterative_readout"
        if args.recipe_from:
            reference, _ = load_model(args.recipe_from)
            if (
                reference.get("selected") != "iterative_readout"
                or reference.get("selector") != info["selector"]
                or reference.get("selection") != "validation_auc"
            ):
                raise ValueError(
                    "--recipe-from must name a validation-selected model with the same selector"
                )
            recipe = reference["readout_lambda"], reference["readout_iterations"]
            info["recipe_from_sha256"] = reference["model_sha256"]

        raw = matrix(train, args.workers)
        labels = torch.arange(len(raw), device=device).remainder(2).to(torch.float64)
        if keep is None:
            x = raw.to(device)
            del raw
            var, mean = torch.var_mean(x, dim=0, unbiased=False)
            std = var.sqrt().clamp_min(1e-7)
            x.sub_(mean).div_(std)
            selector = optimize(x, labels, SELECTOR_PENALTY, CHECKPOINTS[-1])
            keep = np.sort(np.argsort(-np.abs(selector.x[:-1]), kind="stable")[:RETAINED])
            info.update(selector_lambda=SELECTOR_PENALTY, selector_iterations=int(selector.nit))
            idx = torch.as_tensor(keep, device=device)
            x = x.index_select(1, idx)
            mean, std = mean[idx], std[idx]
        else:
            idx = torch.as_tensor(keep)
            x = raw.index_select(1, idx).to(device)
            del raw
            std, mean = torch.std_mean(x, dim=0, unbiased=False)
            std.clamp_min_(1e-7)
            x.sub_(mean).div_(std)
        val = None
        if valrows:
            val = matrix(valrows, args.workers).index_select(1, idx.cpu()).to(device).sub_(mean).div_(std)
        theta, penalty, steps, history = readout(x, labels, val, recipe, args.indices_from is not None)
        theta = torch.as_tensor(theta, device=device)
        w = theta[:-1] / std
        weight, bias = w.cpu().numpy(), float(theta[-1] - mean.dot(w))
        info.update(
            selection="frozen_recipe" if recipe else "validation_auc",
            readout_lambda=penalty,
            readout_iterations=steps,
        )
        if valrows:
            info["validation_auc"] = max(row["val_auc"] for row in history)

    if not np.isfinite(weight).all() or not np.isfinite(bias):
        raise FloatingPointError("Nonfinite fitted weights")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        np.savez_compressed(stage / "model.npz", indices=keep.astype(np.uint16), weight=weight, bias=bias)
        info["model_sha256"] = digest(stage / "model.npz")
        if history:
            with (stage / "history.csv").open("w", newline="", encoding="utf8") as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0])
                writer.writeheader()
                writer.writerows(history)
        (stage / "lock.json").write_text(
            json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        stage.rename(args.out)
    return {k: v for k, v in info.items() if k not in ("source_ids", "training_ids", "validation_ids")}


def score(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a new score CSV: {args.out}")
    lock, model = load_model(args.fit)
    pairs = sorted(read_splits(args.manifest, (args.split,))[args.split])
    ids = {source_hash(s) for s, _ in pairs}
    if args.split == "test":
        if ids.intersection(lock["source_ids"]):
            raise ValueError("Test source IDs overlap training or validation")
    elif "training_ids" not in lock or ids.intersection(lock["training_ids"]):
        raise ValueError("Cannot verify separation of validation and training source IDs")
    scores = np.fromiter(
        (decision(x, model) for x in descriptors(pairs, args.workers)),
        dtype=np.float64,
        count=2 * len(pairs),
    ).reshape(-1, 2)
    report = metrics(scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        path = Path(tmp) / "scores.csv"
        with path.open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source", "label", "cotracenet_lite"))
            for (name, _), pair in zip(pairs, scores):
                writer.writerows((name, label, float(s)) for label, s in enumerate(pair))
        args.out.hardlink_to(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    train = subs.add_parser("fit", help="Fit the current iterative Lite model")
    fisher = subs.add_parser("fit-fisher", help="Fit the frozen full-scale closed-form comparator")
    test = subs.add_parser("score", help="Write validation or test image scores")
    one = subs.add_parser("predict", help="Score one 256 x 256 grayscale image")
    for cmd in (train, fisher, test):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--workers", type=int, default=2, help="CPU feature workers; 0 runs inline")
    for cmd in (train, fisher):
        cmd.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--recipe-from", type=Path, help="Reuse a validated ridge and update count")
    train.add_argument("--indices-from", type=Path, help="Use this split's frozen Fisher coordinates")
    for cmd in (test, one):
        cmd.add_argument("--fit", type=Path, required=True)
    test.add_argument("--split", choices=("val", "test"), default="test")
    one.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "workers", 0) < 0:
        parser.error("--workers must be nonnegative")
    torch.set_num_threads(1 if getattr(args, "indices_from", None) else 2)
    with torch.inference_mode():
        if args.action in ("fit", "fit-fisher"):
            result = fit(args)
        elif args.action == "score":
            result = score(args)
        else:
            _, model = load_model(args.fit)
            value = float(decision(joint_features(args.image), model))
            result = {"score": value, "prediction": "stego" if value > 0 else "cover"}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

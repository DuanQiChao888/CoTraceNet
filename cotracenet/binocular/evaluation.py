"""Evaluate test logits for same- and cross-embedding experiments.

For each direction give --case TRAIN TEST predictions.npz. Each archive contains
one-dimensional source, label, and one signed logit vector per method. An
optional scalar string named @fit:METHOD identifies the same fitted model
across target conditions. Model identifiers and raw scores are never exported.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_case(path, selected):
    with np.load(path, allow_pickle=False) as data:
        source = data["source"]
        label = data["label"]
        if source.ndim != 1 or source.dtype.kind not in "iuUS":
            raise ValueError(f"{path}: source must be integer or string IDs")
        if label.shape != source.shape or not np.isin(label, [0, 1]).all():
            raise ValueError(f"{path}: expected one binary label per source entry")
        ids = source.astype("U")
        order = np.lexsort((label, ids))
        ids = ids[order]
        if ids.size == 0 or ids.size % 2 or (ids == "").any():
            raise ValueError(f"{path}: expected non-empty cover/stego pairs")
        ids = ids.reshape(-1, 2)
        if not (ids[:, 0] == ids[:, 1]).all() or len(np.unique(ids[:, 0])) != len(ids):
            raise ValueError(f"{path}: source IDs must form distinct pairs")
        if not (label[order].reshape(-1, 2) == [0, 1]).all():
            raise ValueError(f"{path}: each source must have both labels")
        methods = selected or [
            k for k in data.files if k not in ("source", "label") and not k.startswith("@")
        ]
        if not methods or len(set(methods)) != len(methods):
            raise ValueError(f"{path}: invalid model list {methods}")
        scores, fits = {}, {}
        for method in methods:
            v = data[method]
            if (
                v.ndim != 1
                or v.size != ids.size
                or v.dtype.kind not in "fiu"
                or not np.isfinite(v).all()
            ):
                raise ValueError(f"{path}: {method} must hold one finite signed logit per input")
            scores[method] = v[order].reshape(-1, 2)
            key = "@fit:" + method
            if key in data:
                value = data[key]
                if (
                    value.ndim != 0
                    or value.dtype.kind not in "US"
                    or not str(value.astype("U").item())
                ):
                    raise ValueError(f"{path}: invalid fit identifier for {method}")
                fits[method] = str(value.astype("U").item())
        return ids[:, 0], scores, fits


def summarize(scores):
    cover, stego = scores.T
    cover_sorted = np.sort(cover)
    ties_weighted = np.searchsorted(cover_sorted, stego, "left") + np.searchsorted(
        cover_sorted, stego, "right"
    )
    hits = (scores > 0) == [0, 1]
    tn, tp = map(int, hits.sum(axis=0))
    n = len(cover)
    return {
        "pairs": n,
        "images": 2 * n,
        "correct": tn + tp,
        "accuracy": (tn + tp) / (2 * n),
        "auc": float(ties_weighted.mean() / (2 * n)),
        "true_negative": tn,
        "false_positive": n - tn,
        "true_positive": tp,
        "false_negative": n - tp,
    }, hits


def contrast(candidate, reference, n_boot, seed):
    diff = candidate.astype(np.int8) - reference.astype(np.int8)
    by_source = diff.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    n = len(diff)
    batch = max(1, min(1000, 1_000_000 // n))
    for start in range(0, n_boot, batch):
        count = min(batch, n_boot - start)
        sample = rng.integers(n, size=(count, n))
        draws[start : start + count] = by_source[sample].mean(axis=1)
    return {
        "delta_accuracy_pp": float(100 * by_source.mean()),
        "ci95_pp": (100 * np.quantile(draws, [0.025, 0.975])).tolist(),
        "corrected": int((diff == 1).sum()),
        "introduced": int((diff == -1).sum()),
        "net_cover_correct": int(diff[:, 0].sum()),
        "net_stego_correct": int(diff[:, 1].sum()),
    }


def evaluate(cases, selected, comparisons, n_boot=10000, seed=7319):
    if n_boot < 1 or seed < 0:
        raise ValueError("Expected a positive bootstrap count and a nonnegative seed")
    matrix, fit_ids, reference_cover = {}, {}, {}
    align = {}
    sources, expected_methods = None, None
    for train, test, path in cases:
        cells = matrix.setdefault(train, {})
        if test in cells:
            raise ValueError(f"Test direction supplied twice: {train} -> {test}")
        ids, logits, fits = load_case(path, selected)
        if sources is None:
            sources, expected_methods = ids, set(logits)
        if not np.array_equal(sources, ids):
            raise ValueError(f"{path}: test sources differ from first condition")
        if set(logits) != expected_methods:
            raise ValueError(f"{path}: compared methods differ across conditions")
        cell = {"models": {}, "paired_comparisons": []}
        cells[test] = cell
        hits = {}
        for method, v in logits.items():
            cell["models"][method], hits[method] = summarize(v)
            key = train, method
            if key not in reference_cover:
                fit_ids[key] = fits.get(method)
                reference_cover[key] = (test, v[:, 0].copy())
                continue
            if fit_ids[key] != fits.get(method):
                raise ValueError(
                    f"{path}: fit identifier inconsistent/missing for {train}/{method}"
                )
            previous, cover = reference_cover[key]
            delta = np.abs(v[:, 0].astype(np.float64) - cover)
            align.setdefault(train, {}).setdefault(method, []).append(
                {
                    "target_a": previous,
                    "target_b": test,
                    "same_supplied_fit_id": True if method in fits else None,
                    "max_abs_difference_on_shared_cover": float(delta.max()),
                }
            )
        for candidate, reference in comparisons:
            if candidate == reference or candidate not in hits or reference not in hits:
                raise ValueError(f"Invalid comparison: {candidate} vs {reference}")
            cell["paired_comparisons"].append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "delta_auc": cell["models"][candidate]["auc"]
                    - cell["models"][reference]["auc"],
                    **contrast(hits[candidate], hits[reference], n_boot, seed),
                }
            )
    return {
        "decision": "logit > 0 predicts stego; a zero predicts cover",
        "bootstrap": {"unit": "source pair", "draws": n_boot, "seed": seed},
        "train_then_test": matrix,
        "cross_target_checks": align,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs=3, action="append", required=True, metavar=("TRAIN", "TEST", "NPZ")
    )
    parser.add_argument("--models", nargs="+", help="Optional exact list to include")
    parser.add_argument(
        "--compare", nargs=2, action="append", default=[], metavar=("CANDIDATE", "REFERENCE")
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.case, args.models, args.compare, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

"""Summarize aligned validation and test score CSVs for one embedding condition."""

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np

from summarize import auc, intervals, read_table


def metrics(scores, threshold):
    tn = int(np.count_nonzero(scores[:, 0] <= threshold))
    tp = int(np.count_nonzero(scores[:, 1] > threshold))
    pairs = len(scores)
    return {
        "pairs": pairs, "images": pairs * 2, "correct": tn + tp,
        "true_negative": tn, "true_positive": tp, "false_positive": pairs - tn,
        "false_negative": pairs - tp, "accuracy": (tn + tp) / (2 * pairs),
        "specificity": tn / pairs, "sensitivity": tp / pairs, "auc": auc(scores),
    }


def contrasts(candidate, reference):
    labels = np.array([False, True])
    hit = ((candidate > 0) == labels).astype(np.int8)
    changed = hit - ((reference > 0) == labels)
    lo, hi = intervals(changed.mean(axis=1))
    return {
        "delta_accuracy_pp": float(100 * changed.mean()),
        "ci95_pp": [float(lo), float(hi)],
        "corrected": int((changed == 1).sum()),
        "introduced": int((changed == -1).sum()),
        "net_cover_correct": int(changed[:, 0].sum()),
        "net_stego_correct": int(changed[:, 1].sum()),
    }


def evaluate(models, comparisons, fpr):
    cohorts = None
    test_scores, model_results = {}, {}
    for name, validation, test, column in models:
        if name in model_results:
            raise ValueError(f"Model name repeated: {name}")
        val_keys, val = read_table(Path(validation), {name: column})
        test_keys, test = read_table(Path(test), {name: column})
        if not set(val_keys).isdisjoint(test_keys):
            raise ValueError(f"Validation and test sources overlap for {name}")
        if cohorts is not None and cohorts != (val_keys, test_keys):
            raise ValueError(f"Source cohorts differ across models for {name}")
        cohorts = val_keys, test_keys
        val, test_scores[name] = val[name], test[name]
        # For the strict '>' decision, ties can only lower validation FPR.
        rank = ceil((1 - fpr) * len(val))
        cutoff = float(np.partition(val[:, 0], rank - 1)[rank - 1])
        calibrated = metrics(test_scores[name], cutoff)
        model_results[name] = {
            "validation": metrics(val, 0), "test": metrics(test_scores[name], 0),
            "validation_calibrated_threshold": {
                "target_fpr": fpr, "cutoff": cutoff,
                "validation_false_positives": int((val[:, 0] > cutoff).sum()),
                "test_false_positives": calibrated["false_positive"],
                "test_true_positives": calibrated["true_positive"],
                "test_fpr": calibrated["false_positive"] / calibrated["pairs"],
                "test_tpr": calibrated["sensitivity"],
            },
        }
    paired = []
    for candidate, reference in comparisons:
        if candidate == reference or candidate not in test_scores or reference not in test_scores:
            raise ValueError(f"Unknown or identical model names: {candidate}, {reference}")
        paired.append({"candidate": candidate, "reference": reference,
                       **contrasts(test_scores[candidate], test_scores[reference])})
    return {"decision": "score > 0 predicts stego for validation and test metrics",
            "bootstrap": {"unit": "source pair", "draws": 10000, "seed": 7319},
            "models": model_results, "test_contrasts": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=4, required=True, action="append",
                        metavar=("NAME", "VAL_CSV", "TEST_CSV", "SCORE_COLUMN"))
    parser.add_argument("--compare", nargs=2, action="append", default=[],
                        metavar=("CANDIDATE", "REFERENCE"))
    parser.add_argument("--validation-fpr", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True, help="A new aggregate JSON file")
    args = parser.parse_args()
    if not 0 <= args.validation_fpr < 1:
        parser.error("--validation-fpr must be a fraction in [0, 1)")
    result = evaluate(args.model, args.compare, args.validation_fpr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

"""Summarize single-image detector scores and bootstrap paired accuracy differences."""
import argparse
import csv
from pathlib import Path

import numpy as np


CONDITIONS = ("WOW_01", "WOW_04", "HILL_01", "HILL_04", "JPEG_03")
# Column names of the frozen CoTraceNet scoring pipeline. Probabilities and
# diagnostic fields in the input table do not enter the detector comparison.
FINAL_COLUMNS = {
    "texture_expert_C": "texture",
    "previous_fixed_F": "fixed_matched",
    "full_fisher_E": "directional",
    "subspace_fisher_B": "subspace",
    "same_expert_fixed_Q": "strict_fixed",
    "mean_gate": "mean_gate",
    "no_reward": "gate_ce",
    "CoTraceNet": "cotracenet",
}


def read_table(path, columns):
    pairs = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if columns is FINAL_COLUMNS and "baseline" in fields:
            columns = {"ShuffleBANet": "baseline", **columns}
        missing = ({"source", "label"} | set(columns.values())) - fields
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            key = row["source"], row["label"]
            if not key[0] or key[1] not in ("0", "1") or key in pairs:
                raise ValueError(f"Missing source, invalid label or duplicate row in {path}: {key}")
            values = {method: float(row[column]) for method, column in columns.items()}
            if not all(np.isfinite(list(values.values()))):
                raise ValueError(f"Non-finite scores in {path} for {key}")
            pairs[key] = values
    sources = sorted({source for source, _ in pairs})
    expected = {(source, label) for source in sources for label in ("0", "1")}
    if not sources or set(pairs) != expected:
        raise ValueError(f"Each source must have exactly one cover and one stego image in {path}")
    scores = {
        method: np.array([[pairs[source, label][method] for label in ("0", "1")] for source in sources])
        for method in columns
    }
    return sources, scores


def auc(scores):
    cover = np.sort(scores[:, 0])
    stego = scores[:, 1]
    rank = np.searchsorted(cover, stego, side="left") + np.searchsorted(cover, stego, side="right")
    return float(rank.mean() / (2 * len(cover)))


def intervals(by_source):
    # Each bootstrap draw uses the same source indices for both detectors and
    # both members of each cover/stego pair. Bound the size of the index array.
    rng = np.random.default_rng(7319)
    n = len(by_source)
    estimates = np.empty(10000)
    batch = max(1, min(1000, 1_000_000 // n))
    for start in range(0, len(estimates), batch):
        size = min(batch, len(estimates) - start)
        draws = rng.integers(n, size=(size, n))
        estimates[start:start+size] = by_source[draws].mean(axis=1)
    return 100 * np.quantile(estimates, [0.025, 0.975])


def summarize(condition, scores, target="CoTraceNet"):
    hits = {method: (value > 0) == np.array([False, True]) for method, value in scores.items()}
    if target not in hits:
        raise ValueError(f"Missing detector for paired comparisons: {target}")
    metrics, contrasts = [], []
    for method, hit in hits.items():
        tn, tp = hit.sum(axis=0)
        metrics.append((condition, method, hit.size, int(hit.sum()), int(tp), int(tn),
                        float(100 * hit.mean()), auc(scores[method])))
        if method == target:
            continue
        changed = hits[target].astype(np.int8) - hit.astype(np.int8)
        by_source = changed.mean(axis=1)
        lo, hi = intervals(by_source)
        contrasts.append((condition, method, len(hit), int((changed > 0).sum()),
                          int((changed < 0).sum()), float(100 * by_source.mean()), float(lo), float(hi)))
    return metrics, contrasts


def parse_table(spec):
    pattern, *mappings = spec
    if not mappings:
        return pattern, FINAL_COLUMNS
    fields = {}
    for token in mappings:
        model, sep, col = token.partition("=")
        if not sep or not model or not col or model in fields:
            raise ValueError(f"Expected a unique METHOD=COLUMN, got {token!r}")
        fields[model] = col
    return pattern, fields


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="append", nargs="+", required=True, metavar="ARG",
                        help="Score CSV pattern with optional {condition}, followed by METHOD=COLUMN overrides. "
                             "Repeat for additional model score files. Default columns match the final scorer.")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, help="Ordered condition identifiers.")
    parser.add_argument("--target", default="CoTraceNet", help="Detector compared against each reference.")
    parser.add_argument("--out", required=True, type=Path, help="Directory for two aggregate CSVs.")
    args = parser.parse_args()
    tables = [parse_table(spec) for spec in args.table]
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("Condition identifiers must be unique.")
    if len(args.conditions) > 1 and any("{condition}" not in pattern for pattern, _ in tables):
        parser.error("Use {condition} in every score path for a multi-condition comparison.")
    cohort = None
    metric_rows, contrast_rows = [], []
    for condition in args.conditions:
        scores = {}
        for pattern, columns in tables:
            path = Path(pattern.replace("{condition}", condition))
            sources, new = read_table(path, columns)
            if cohort is None:
                cohort = sources
            if sources != cohort:
                raise ValueError(f"Source cohort does not match across models and conditions in {path}")
            if new.keys() & scores.keys():
                raise ValueError(f"Model supplied by multiple files: {new.keys() & scores.keys()}")
            scores.update(new)
        metrics, contrasts = summarize(condition, scores, args.target)
        metric_rows.extend(metrics)
        contrast_rows.extend(contrasts)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "metrics.csv",
              ("condition", "method", "images", "correct", "true_positive", "true_negative", "accuracy_pct", "auc"),
              metric_rows)
    write_csv(args.out / "paired.csv",
              ("condition", "reference", "sources", "fixed_errors", "new_errors", "delta_accuracy_pp",
               "ci95_low_pp", "ci95_high_pp"), contrast_rows)


if __name__ == "__main__":
    main()

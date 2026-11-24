"""Verify BOSSbase paired predictions and lock multi-layer architecture choices."""

import argparse
import json
from pathlib import Path

import numpy as np

from .hierarchy import digest
from ..binocular.evaluation import load_case, summarize
from .metrics import paired, write_csv

BASE_PARAMS = 16828
DOMAINS = ("WOW", "HILL")
FAMILIES = {
    "hierarchy": "hierarchy",
    "resmlp": "resmlp",
    "resmlp_shallow": "resmlp",
    "spatial": "spatial",
}
DEEP = ("resmlp", "spatial", "hierarchy")


def same(left, right):
    return np.array_equal(left > 0, right > 0) and np.allclose(
        left, right, atol=2e-5, rtol=2e-6
    )


def load(root, source, split, methods):
    logits, entries, file_hashes = {}, {"baseline": {"extra_params": 0}}, {}
    sources = None
    for family in sorted({FAMILIES[n] for n in methods}):
        selected = [n for n in methods if FAMILIES[n] == family]
        folder = root / family / source
        manifest = folder / "unified-models.json"
        preds = folder / ("unified-" + split + ".npz")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        ids, scores, _ = load_case(preds, ["baseline", *selected])
        if sources is None:
            sources, logits["baseline"] = ids, scores["baseline"]
        if not np.array_equal(sources, ids) or not same(
            logits["baseline"], scores["baseline"]
        ):
            raise ValueError(f"Source IDs or baseline disagree in {preds}")
        for n in selected:
            entry = data[n]
            h = entry["sha256"].lower()
            if len(h) != 64 or (set(h) - set("0123456789abcdef")):
                raise ValueError(f"Invalid checkpoint hash for {source}/{n}")
            if not 0 < entry["extra_params"] <= 4 * BASE_PARAMS:
                raise ValueError(f"Parameter budget exceeded for {source}/{n}")
            entries[n] = entry
            logits[n] = scores[n]
        for p in (manifest, preds):
            file_hashes[p.relative_to(root).as_posix()] = digest(p)
    if len(sources) != (1000 if split == "val" else 3000):
        raise ValueError("Unexpected number of source pairs")
    return sources, logits, entries, file_hashes


def lock(root, methods, seed, primary=None):
    result = dict(
        seed=seed,
        manifests={},
        correct={},
        selected={},
        files={},
        parent_lock_sha256=primary,
    )
    rows = []
    for source in DOMAINS:
        _, logits, entries, files = load(root, source, "val", methods)
        result["files"].update(files)
        counts = {n: summarize(scores)[0]["correct"] for n, scores in logits.items()}
        rank = sorted(
            counts,
            key=lambda n: (-counts[n], entries[n]["extra_params"], n != "baseline", n),
        )
        result["manifests"][source] = entries
        result["selected"][source] = rank[0]
        result["correct"][source] = counts
        for n in rank:
            rows.append(
                dict(
                    source=source,
                    variant=n,
                    correct=counts[n],
                    images=2000,
                    accuracy_pct=counts[n] / 20,
                    full_params=BASE_PARAMS + entries[n]["extra_params"],
                )
            )
    pooled = {n: sum(x[n] for x in result["correct"].values()) for n in counts}
    ranking = sorted(
        pooled,
        key=lambda n: (-pooled[n], entries[n]["extra_params"], n != "baseline", n),
    )
    result.update(
        pooled_correct=pooled,
        common_selected=ranking[0],
        repeat_candidate=next(n for n in ranking if n in DEEP),
    )
    return result, rows


def test(root, archive, methods, result):
    rows, shallow_rows, cover_rows = [], [], []
    known_ids = None
    for source in DOMAINS:
        prev_covers = {}
        for target in DOMAINS:
            ids, logits, entries, _ = load(root, source, "test-" + target, methods)
            if entries != result["manifests"][source]:
                raise ValueError("Checkpoint manifest changed after lock")
            stored_ids, old, _ = load_case(
                archive / source / (source + "-to-" + target + ".npz"), ["policy"]
            )
            if known_ids is None:
                known_ids = ids
            if (
                not np.array_equal(known_ids, ids)
                or not np.array_equal(stored_ids, ids)
                or not same(old["policy"], logits["baseline"])
            ):
                raise ValueError(
                    f"Archived baseline or held-out pair IDs differ for {source}/{target}"
                )
            hits = {n: (s > 0) == [0, 1] for n, s in logits.items()}
            for n, s in logits.items():
                if n in prev_covers:
                    err = float(np.max(np.abs(s[:, 0] - prev_covers[n])))
                    if not same(prev_covers[n], s[:, 0]):
                        raise ValueError(
                            f"Shared cover mismatch of {err}: {source}/{n}"
                        )
                    cover_rows.append(
                        dict(
                            source=source,
                            variant=n,
                            shared_covers=len(s),
                            max_abs_error=err,
                        )
                    )
                prev_covers[n] = s[:, 0]
                metrics, _ = summarize(s)
                rows.append(
                    dict(
                        seed=result["seed"],
                        source=source,
                        target=target,
                        variant=n,
                        params_with_anchor=BASE_PARAMS + entries[n]["extra_params"],
                        correct=metrics["correct"],
                        images=metrics["images"],
                        accuracy_pct=100 * metrics["accuracy"],
                        auc=metrics["auc"],
                        **paired(hits[n], hits["baseline"]),
                    )
                )
            if "resmlp_shallow" in hits and "resmlp" in hits:
                shallow_rows.append(
                    dict(
                        source=source,
                        target=target,
                        candidate="resmlp",
                        reference="resmlp_shallow",
                        **paired(hits["resmlp"], hits["resmlp_shallow"]),
                    )
                )
    write_csv(root / "test-results.csv", rows)
    write_csv(root / "cover-checks.csv", cover_rows)
    if shallow_rows:
        write_csv(root / "matched-depth.csv", shallow_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["lock", "test"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--parent-lock", type=Path, help="For the independent seed-29 repeat"
    )
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    parent = args.parent_lock
    methods = (
        [json.loads(parent.read_text())["repeat_candidate"]]
        if parent
        else list(FAMILIES)
    )
    path = args.root / "selection-lock.json"
    result, rows = lock(
        args.root, methods, 29 if parent else 73, digest(parent) if parent else None
    )
    if args.phase == "lock":
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        write_csv(args.root / "validation-results.csv", rows)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "selected",
                        "pooled_correct",
                        "common_selected",
                        "repeat_candidate",
                    )
                }
            )
        )
        print("LOCK", digest(path))
    else:
        if not args.archive or result != json.loads(path.read_text(encoding="utf-8")):
            raise ValueError(
                "Validation inputs or configuration differ from the recorded lock"
            )
        test(args.root, args.archive, methods, result)
        print("Archived baseline, test pairs and common cover predictions verified.")

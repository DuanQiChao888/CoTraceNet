"""Single-PGM deployment and CPU parity/timing for optional local reassessment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter_ns

import numpy as np
import torch
from cotracenet.data import load_image, read_splits

from cotracenet.binocular.predict import PolicyDetector
from .active_review import ReviewModel
from .patch_features import FIXED, describe_tile, select_tile


METHODS = (
    "anchor",
    "gate_selected",
    "gate_fixed",
    "uncertainty_selected",
    "uncertainty_fixed",
    "always_selected",
    "always_fixed",
)


class Detector:
    def __init__(self, binocular, previous, current, train):
        self.upstream = PolicyDetector(binocular, previous, train)
        self.folder = Path(current) / train
        self.models = {}

    def predict(self, path, method="gate_selected"):
        if method not in METHODS:
            raise ValueError("Unknown inference mode: " + method)
        s = self.upstream.predict(path)["score"]
        if method == "anchor":
            return {"score": s, "anchor": s, "query": False, "tile": -1}
        route, arm = method.split("_")
        if arm not in self.models:
            self.models[arm] = ReviewModel(self.folder, arm)
        model = self.models[arm]
        take = route == "always" or (
            model.should_query(s) if route == "gate" else model.uncertainty_query(s)
        )
        score, tile = s, -1
        if take:
            # Reuse the upstream path interface, and charge a second PGM decode only on acquisition.
            gray = load_image(Path(path))
            tile = select_tile(gray) if arm == "selected" else FIXED
            score = model.refine(s, describe_tile(gray, tile))
        return {"score": float(score), "anchor": float(s), "query": bool(take), "tile": int(tile)}


def summary(times):
    if not len(times):
        return {"n": 0}
    return {
        "n": len(times),
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.quantile(times, 0.9)),
    }


def profile_domain(args, train):
    model = Detector(args.binocular, args.previous, args.root, train)
    manifest = args.wow_manifest if train == "WOW" else args.hill_manifest
    pairs = read_splits(manifest, ("test",))["test"]
    # Choose sources before inspecting any score or decision; both classes always enter together.
    pairs = sorted(
        pairs, key=lambda row: hashlib.sha256(("active-timing73/" + row[0]).encode()).digest()
    )
    paths = {(key, label): path for key, images in pairs for label, path in enumerate(images)}
    run = [(s, i) for s, _ in pairs[: args.pairs] for i in (0, 1)]
    with np.load(
        args.root / train / (train + "-to-" + train + "-local.npz"), allow_pickle=False
    ) as f:
        expected = {k: f[k] for k in (*METHODS, "gate", "uncertainty", "selected_tile")}
        index = {(str(s), int(y)): i for i, (s, y) in enumerate(zip(f["source"], f["label"]))}
    if set(index) != set(paths):
        raise ValueError("Frozen score cache disagrees with test manifest")
    hits = {m: 0 for m in METHODS}
    maxerr = {m: 0.0 for m in METHODS}
    errors = dict(queries=0, predictions=0, tiles=0)
    samples = {m: {"all": [], "keep": [], "query": []} for m in METHODS}

    def verify(key, method, result):
        i = index[key]
        if method == "anchor":
            query = False
        elif method.startswith("always_"):
            query = True
        else:
            query = bool(expected[method.split("_")[0]][i])
        tile = (
            -1
            if not query
            else (FIXED if method.endswith("_fixed") else int(expected["selected_tile"][i]))
        )
        target = expected[method][i]
        errors["queries"] += int(query != result["query"])
        errors["predictions"] += int((target > 0) != (result["score"] > 0))
        errors["tiles"] += int(tile != result["tile"])
        err = abs(target - result["score"])
        if not np.isfinite(err) or err > 1e-8 or any(errors.values()):
            raise AssertionError((train, method, err, errors))
        maxerr[method] = max(float(err), maxerr[method])
        hits[method] += 1

    # Warm up all entry points on a source that is outside the timing sample.
    for m in METHODS:
        model.predict(paths[(pairs[-1][0], 0)], m)
    rng = random.Random(31810)
    paired_times = {m: {} for m in METHODS}
    for repeat in range(args.repeats):
        keys = list(run)
        rng.shuffle(keys)
        for key in keys:
            order = list(METHODS)
            rng.shuffle(order)
            for m in order:
                start = perf_counter_ns()
                result = model.predict(paths[key], m)
                ms = (perf_counter_ns() - start) / 1e6
                verify(key, m, result)
                paired_times[m][(repeat, key)] = ms
                samples[m]["all"].append(ms)
                samples[m]["query" if result["query"] else "keep"].append(ms)
    # Branch coverage independent of the timing sample. No supplementary records enter the latency estimate.
    for m in METHODS[1:]:
        criterion = m.split("_")[0]
        if criterion == "always":
            continue
        for answer in (False, True):
            candidates = [k for k in paths if bool(expected[criterion][index[k]]) == answer]
            if candidates and not samples[m]["query" if answer else "keep"]:
                key = candidates[0]
                verify(key, m, model.predict(paths[key], m))
    return {
        "test_images_per_repeat": len(run),
        "repeats": args.repeats,
        "parity_cases": hits,
        "mismatches": errors,
        "max_abs_logit_error": maxerr,
        "latency": {m: {k: summary(t) for k, t in rows.items()} for m, rows in samples.items()},
        "paired_mean_delta_ms_vs_anchor": {
            m: float(np.mean([v - paired_times["anchor"][k] for k, v in paired_times[m].items()]))
            for m in METHODS[1:]
        },
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("binocular", "previous", "root"):
        cli.add_argument("--" + name, type=Path, required=True)
    sub = cli.add_subparsers(dest="command", required=True)
    single = sub.add_parser("predict")
    single.add_argument("--train", choices=("WOW", "HILL"), required=True)
    single.add_argument("--method", choices=METHODS, default="gate_selected")
    single.add_argument("image", type=Path)
    bench = sub.add_parser("profile")
    bench.add_argument("--wow-manifest", type=Path, required=True)
    bench.add_argument("--hill-manifest", type=Path, required=True)
    bench.add_argument("--pairs", type=int, default=24)
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--cpu", type=int, required=True)
    bench.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.command == "predict":
        print(
            json.dumps(
                Detector(args.binocular, args.previous, args.root, args.train).predict(
                    args.image, args.method
                )
            )
        )
    else:
        os.sched_setaffinity(0, {args.cpu})
        result = {
            "timing_scope": "Warm weights, warm filesystem, one randomly interleaved full call from PGM path; per-query second PGM decode included.",
            "sample": "Deterministic source-hash sample, same cover/stego pairs across both domains; supplemental parity examples excluded from timing.",
            "threads": 1,
            "cpu_affinity": list(os.sched_getaffinity(0)),
            "domains": {},
        }
        for train in ("WOW", "HILL"):
            result["domains"][train] = profile_domain(args, train)
            print("Finished", train, flush=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

"""Benchmark file-to-score latency at batch size one with weights resident in memory.

Use the same paired test manifest and frozen models as the accuracy evaluation.
On Linux the process is pinned to one allowed core; platforms without affinity
support record that limitation. Torch and numerical-library thread counts are one.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

# Set these before importing numerical libraries or initializing worker pools.
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[key] = "1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotracenet import detector, lite, policy
from cotracenet.data import read_splits


def pin_core(requested):
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = os.sched_getaffinity(0)
        core = min(allowed) if requested is None else requested
        if core not in allowed:
            raise ValueError("The requested CPU is outside the process affinity mask")
        os.sched_setaffinity(0, {core})
        return core
    if requested is not None:
        raise ValueError("This platform does not support selecting a CPU core")
    return None


def primary_scorer(net, lock, params):
    gate = {k: params["cotracenet_" + k] for k in ("gw", "gb", "mu", "sd")}
    selected = lock["gate"]["cotracenet"]

    @torch.inference_mode()
    def score(path):
        image, third, context, ref, joint = detector.image_features(path)
        tensor = torch.from_numpy(image[None, None]).to(
            dtype=torch.float32, memory_format=torch.channels_last
        )
        logits = net(tensor)[0]
        cnn = ((logits[1] - logits[0]).item() - params["base_mean"]) / params["base_sd"]
        expert_scores = [
            features @ params[name + "_w"] + params[name + "_b"]
            for name, features in (("r", third), ("c", ref), ("e", joint), ("b", joint))
        ]
        actions = detector.four_heads(*expert_scores).reshape(1, 1, 4)
        anchor = actions @ params["strict_coef"] + params["intercept"]
        state = policy.descriptors(actions, anchor, context[None, None], np.array([[cnn]]))
        change, _ = policy.adjustment(state, gate)
        return policy.apply_final(anchor, change, selected["alpha"], selected["center"]).item()

    return score


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "backbone", "full-fit", "lite-fit", "out"):
        cli.add_argument("--" + arg, type=Path, required=True)
    cli.add_argument("--sources", type=int, default=12)
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--core", type=int, help="Linux CPU index; defaults to the first allowed core")
    args = cli.parse_args()
    if args.repeats < 1 or args.sources < 1 or args.out.exists():
        cli.error("Provide positive sample and repeat counts and a new output filename")

    core = pin_core(args.core)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pairs = read_splits(args.manifest, ("test",))["test"]
    if args.sources > len(pairs):
        cli.error("The manifest has fewer source pairs than requested")
    paths = [path for _, pair in pairs[: args.sources] for path in pair]
    net = detector.load_backbone(args.backbone, "cpu")
    full_lock, params = detector.load_model(args.full_fit, args.backbone)
    _, model = lite.load_model(args.lite_fit)

    def sparse(path):
        return float(lite.decision(lite.joint_features(path), model))

    scorers = [("full", primary_scorer(net, full_lock, params)), ("lite", sparse)]
    # Compute reference decisions outside the timed loop with the public batch scorer.
    features = detector.extract(paths, net, "cpu", 0)
    refs = {
        "full": detector.score_features(features, full_lock, params)["cotracenet"],
        "lite": lite.decision(features[-2], model),
    }
    for path in paths:
        path.read_bytes()
    for _, scorer in scorers:
        for path in paths[:2]:
            scorer(path)

    times = {name: [] for name, _ in scorers}
    decisions_agree = {name: np.ones(len(paths), dtype=bool) for name, _ in scorers}
    max_error = {name: 0.0 for name, _ in scorers}
    for rep in range(args.repeats):
        for i, path in enumerate(paths):
            order = scorers if (rep + i) % 2 == 0 else reversed(scorers)
            for name, scorer in order:
                start = time.perf_counter_ns()
                value = scorer(path)
                times[name].append((time.perf_counter_ns() - start) / 1e6)
                reference = refs[name][i]
                decisions_agree[name][i] &= (value > 0) == (reference > 0)
                max_error[name] = max(max_error[name], float(abs(value - reference)))
    quantiles = {
        name: dict(zip(("p25_ms", "median_ms", "p75_ms"), np.percentile(v, [25, 50, 75])))
        for name, v in times.items()
    }
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (
                s.partition(":")[2].strip()
                for s in cpuinfo.read_text().splitlines()
                if s.startswith("model name")
            ),
            cpu,
        )
    report = {
        "cpu": cpu,
        "pinned_core": core,
        "torch_threads": 1,
        "blas_threads": 1,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "images": len(paths),
        "source_pairs": args.sources,
        "repeats": args.repeats,
        "batch_size": 1,
        "timing": quantiles,
        "median_speedup": quantiles["full"]["median_ms"] / quantiles["lite"]["median_ms"],
        "classification_agreement": {k: int(v.sum()) for k, v in decisions_agree.items()},
        "max_abs_difference_vs_batch": max_error,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

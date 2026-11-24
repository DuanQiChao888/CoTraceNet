"""Apply a locked hierarchical blend to a single PGM using the public CoTraceNet package."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from cotracenet.lite import joint_features
from cotracenet.binocular.predict import PolicyDetector

from .hierarchy import Hierarchy, check_policy, digest, dump


class HierarchyDetector:
    def __init__(self, fitted, attention_root, policy_root, source):
        folder = Path(fitted) / source
        lock = json.loads((folder / "unified-models.json").read_text(encoding="utf-8"))[
            "hierarchy"
        ]
        if digest(folder / "weights.pt") != lock["sha256"]:
            raise ValueError("Hierarchical checkpoint fingerprint mismatch")
        state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        if digest(folder / "train-normalizer.npz") != state["normalizer_sha256"]:
            raise ValueError("Source normalization fingerprint mismatch")
        check_policy(state, Path(policy_root), source)
        self.net = Hierarchy().eval()
        self.net.load_state_dict(state["weights"])
        self.mix = float(state["alpha"])
        with np.load(folder / "train-normalizer.npz", allow_pickle=False) as f:
            self.mean, self.inv = f["mean"], f["inv_scale"]
        self.policy = PolicyDetector(attention_root, policy_root, source)

    @torch.inference_mode()
    def predict(self, pgm):
        raw = joint_features(Path(pgm))
        baseline = self.policy.predict_features(raw)["score"]
        z = np.clip((raw.astype(np.float32) - self.mean) * self.inv, -8.0, 8.0)
        standalone = float(self.net(torch.from_numpy(z.reshape(1, -1))).item())
        combined = (1 - self.mix) * baseline + self.mix * standalone
        return dict(
            baseline=baseline,
            standalone=standalone,
            score=combined,
            prediction="stego" if combined > 0 else "cover",
        )


def parity(args):
    seed_results = {}
    for source in ("WOW", "HILL"):
        detector = HierarchyDetector(
            args.fitted, args.attention_root, args.policy_root, source
        )
        for target in ("WOW", "HILL"):
            manifest = args.wow_manifest if target == "WOW" else args.hill_manifest
            data = manifest.parent
            with manifest.open(encoding="utf-8-sig") as f:
                pairs = sorted(
                    (row for row in csv.DictReader(f) if row["split"] == "test"),
                    key=lambda r: r["source"],
                )
            chosen = pairs[:4]
            with np.load(
                args.fitted / source / ("unified-test-" + target + ".npz")
            ) as z:
                ix = {
                    (s, int(y)): i
                    for i, (s, y) in enumerate(zip(z["source"], z["label"]))
                }
                max_err, base_err, same = 0.0, 0.0, 0
                for row in chosen:
                    for y, entry in enumerate(("cover", "stego")):
                        path = data / row[entry]
                        if digest(path) != row[entry + "_sha256"]:
                            raise ValueError("Held-out PGM fingerprint mismatch")
                        current = detector.predict(path)
                        j = ix[row["source"], y]
                        max_err = max(
                            max_err, abs(current["score"] - float(z["hierarchy"][j]))
                        )
                        base_err = max(
                            base_err, abs(current["baseline"] - float(z["baseline"][j]))
                        )
                        same += int((current["score"] > 0) == (z["hierarchy"][j] > 0))
            n = len(chosen) * 2
            if same != n or max_err > 5e-5 or base_err > 1e-8:
                raise ValueError((source, target, n, same, max_err, base_err))
            seed_results[source + "->" + target] = dict(
                images=n,
                cover=n // 2,
                stego=n // 2,
                decisions_equal=same,
                max_abs_score_error=max_err,
                max_abs_baseline_error=base_err,
            )
    dump(
        args.output,
        dict(
            selection="First four lexicographic held-out source pairs per target",
            protocol="One PGM per inference, compare with locked feature-batch logits",
            directions=seed_results,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fitted", type=Path, required=True)
    p.add_argument("--attention-root", type=Path, required=True)
    p.add_argument("--policy-root", type=Path, required=True)
    sub = p.add_subparsers(dest="action", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--source", choices=("WOW", "HILL"), required=True)
    prediction.add_argument("image", type=Path)
    check = sub.add_parser("parity")
    check.add_argument("--wow-manifest", type=Path, required=True)
    check.add_argument("--hill-manifest", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.action == "predict":
        detector = HierarchyDetector(
            args.fitted, args.attention_root, args.policy_root, args.source
        )
        print(json.dumps(detector.predict(args.image), indent=2))
    else:
        parity(args)

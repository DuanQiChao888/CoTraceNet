"""Train CoTraceNet, score paired test images, or predict one unlabeled image."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import classification_metrics, read_splits
from .detector import fit, load_backbone, load_model, predict, extract, score_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    training = sub.add_parser(
        "fit", help="Fit all residual experts and select a gate on validation pairs"
    )
    scoring = sub.add_parser("score", help="Evaluate a locked model on a paired manifest")
    single = sub.add_parser("predict", help="Classify one 8-bit 256×256 grayscale image")
    for cmd in (training, scoring, single):
        cmd.add_argument("--backbone", type=Path, required=True)
        cmd.add_argument("--workers", type=int, default=0 if cmd == single else 2)
        cmd.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for cmd in (training, scoring):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    for cmd in (scoring, single):
        cmd.add_argument("--fit", type=Path, required=True)
    single.add_argument("--image", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(73)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.action == "fit":
        result = fit(args.manifest, args.backbone, args.out, device, args.workers)
    elif args.action == "score":
        pairs = read_splits(args.manifest, ("test",))["test"]
        lock, params = load_model(args.fit, args.backbone)
        if {hashlib.sha256(s.encode()).hexdigest() for s, _ in pairs}.intersection(
            lock["source_ids"]
        ):
            raise ValueError("Test source IDs overlap training or validation")
        if args.out.exists():
            raise FileExistsError(f"Choose a new predictions CSV: {args.out}")
        paths = [path for _, pair in pairs for path in pair]
        features = extract(paths, load_backbone(args.backbone, device), device, args.workers)
        scores = score_features(features, lock, params)
        labels = np.tile([0, 1], len(pairs))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source", "label", *scores])
            for i, label in enumerate(labels):
                writer.writerow([pairs[i // 2][0], label, *[float(s[i]) for s in scores.values()]])
        result = classification_metrics(labels, scores["cotracenet"])
    else:
        scores = predict([args.image], args.fit, args.backbone, device, args.workers)
        s = float(scores["cotracenet"][0])
        result = {
            "score": s,
            "prediction": "stego" if s > 0 else "cover",
            "p_keep": float(scores["p_keep"][0]),
            "p_up": float(scores["p_up"][0]),
            "p_down": float(scores["p_down"][0]),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

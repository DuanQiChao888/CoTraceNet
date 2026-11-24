"""Fit, evaluate or apply the dual-view reward classifier."""

import argparse
import json
from pathlib import Path

import torch

from . import pipeline
from .predict import PolicyDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fit = commands.add_parser(
        "fit", help="Fit attention and policy on source training/validation pairs"
    )
    score = commands.add_parser(
        "score", help="Score untouched target test pairs with fixed source weights"
    )
    predict = commands.add_parser(
        "predict", help="Classify one 256 × 256 grayscale image"
    )
    for cmd in (fit, score, predict):
        cmd.add_argument(
            "--domain",
            choices=("WOW", "HILL"),
            required=True,
            help="Domain used for training",
        )
        cmd.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Run directory containing attention/ and policy/",
        )
    for cmd in (fit, score):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--workers", type=int, default=2)
        cmd.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    fit.add_argument(
        "--lite-fit",
        type=Path,
        required=True,
        help="Model fitted by cotracenet.iterative",
    )
    score.add_argument("--target", choices=("WOW", "HILL"), required=True)
    predict.add_argument("image", type=Path)
    args = parser.parse_args()
    if getattr(args, "workers", 0) < 0:
        parser.error("--workers must be nonnegative")
    torch.set_num_threads(2)
    if args.action == "fit":
        pipeline.fit(
            args.manifest,
            args.lite_fit,
            args.out,
            args.domain,
            args.device,
            args.workers,
        )
    elif args.action == "score":
        print(
            json.dumps(
                pipeline.score(
                    args.manifest,
                    args.out,
                    args.domain,
                    args.target,
                    args.device,
                    args.workers,
                ),
                indent=2,
            )
        )
    else:
        model = PolicyDetector(args.out / "attention", args.out / "policy", args.domain)
        print(json.dumps(model.predict(args.image), indent=2))


if __name__ == "__main__":
    main()

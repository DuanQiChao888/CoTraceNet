"""Run the paired, full-scale WOW/HILL local-reassessment ablation."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def run(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)], check=True)


def locked_heads(root):
    return {
        (domain, name): hashlib.sha256((root / domain / name).read_bytes()).digest()
        for domain in ("WOW", "HILL")
        for name in ("head-selected.npz", "head-fixed.npz", "local-lock.json")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, required=True, help="Output of binocular fit for both domains"
    )
    parser.add_argument("--wow-manifest", type=Path, required=True)
    parser.add_argument("--hill-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory for this ablation")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.out.exists():
        parser.error("Select a new output directory for this run")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifests = {"WOW": args.wow_manifest, "HILL": args.hill_manifest}
    policy = args.models / "policy"
    with TemporaryDirectory(dir=args.out.parent) as work:
        root = Path(work) / "review"
        for domain, manifest in manifests.items():
            run(
                "cotracenet.local_review.upstream",
                "--previous",
                policy,
                "--out",
                root,
                "--train",
                domain,
                "--stage",
                "train",
            )
            for split in ("train", "val"):
                run(
                    "cotracenet.local_review.patch_features",
                    "--manifest",
                    manifest,
                    "--previous-policy",
                    policy,
                    "--out",
                    root,
                    "--domain",
                    domain,
                    "--split",
                    split,
                    "--workers",
                    args.workers,
                )
            run("cotracenet.local_review.active_review", "fit", "--root", root, "--domain", domain)
        before = locked_heads(root)
        for source in manifests:
            for target, manifest in manifests.items():
                run(
                    "cotracenet.binocular",
                    "score",
                    "--domain",
                    source,
                    "--target",
                    target,
                    "--manifest",
                    manifest,
                    "--out",
                    args.models,
                    "--workers",
                    args.workers,
                    "--device",
                    args.device,
                )
            run(
                "cotracenet.local_review.upstream",
                "--previous",
                policy,
                "--out",
                root,
                "--train",
                source,
                "--stage",
                "test",
            )
        for target, manifest in manifests.items():
            run(
                "cotracenet.local_review.patch_features",
                "--manifest",
                manifest,
                "--previous-policy",
                policy,
                "--out",
                root,
                "--domain",
                target,
                "--split",
                "test",
                "--workers",
                args.workers,
            )
        if before != locked_heads(root):
            raise RuntimeError("Local weights changed during test preparation")
        for source in manifests:
            run(
                "cotracenet.local_review.active_review", "score", "--root", root, "--domain", source
            )
        run("cotracenet.local_review.active_review", "summary", "--root", root)
        root.rename(args.out)
    print(f"Accuracy, paired differences and query rates: {args.out / 'performance.csv'}")


if __name__ == "__main__":
    main()

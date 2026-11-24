"""Choose and describe one local 64x64 patch of a 256x256 grayscale image."""

import argparse
import csv
import hashlib
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def make_orbits():
    index = np.arange(125)
    a, b, c = index // 25, (index // 5) % 5, index % 5
    reverse = 25 * c + 5 * b + a
    canonical = np.minimum.reduce([index, reverse, 124 - index, 124 - reverse])
    _, words = np.unique(canonical, return_inverse=True)
    assert int(words.max()) == 38
    return words


ORBITS = make_orbits()
FIXED = 5


def require_image(gray):
    if gray.shape != (256, 256) or gray.dtype != np.uint8:
        raise ValueError("Expected a 256-by-256 array of uint8 grayscale pixels")


def double_midrank(scores):
    pairs = scores[:, None]
    # The doubled midrank equals 2*strictly-smaller + tied (including self).
    return (pairs > scores).sum(1) + (pairs >= scores).sum(1)


def select_tile(gray):
    """Pick maximum rank disagreement between texture and checkerboard activity."""
    require_image(gray)
    # One 2x2 neighborhood out of each 4x4 cell. Never cross patch boundaries.
    a = gray[1::4, 1::4].astype(np.int16)
    b = gray[1::4, 2::4].astype(np.int16)
    c = gray[2::4, 1::4].astype(np.int16)
    d = gray[2::4, 2::4].astype(np.int16)
    horizontal = b - a
    vertical = c - a
    checker = (d - c) - horizontal
    texture = np.abs(horizontal) + np.abs(vertical)
    freq = np.abs(checker)
    t = texture.reshape(4, 16, 4, 16).sum((1, 3), dtype=np.int32).ravel()
    f = freq.reshape(4, 16, 4, 16).sum((1, 3), dtype=np.int32).ravel()
    return int(np.abs(double_midrank(t) - double_midrank(f)).argmax())


def describe_tile(gray, index):
    """Square-root transform of four local symmetric 39-bin residual histograms."""
    require_image(gray)
    if not 0 <= index < 16:
        raise ValueError("Patch id must be between 0 and 15")
    row, col = divmod(int(index), 4)
    patch = gray[row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64].astype(np.int16)
    blocks = []
    for axis_view in (patch, patch.T):
        difference = np.diff(axis_view, axis=1)
        for field in (difference, np.diff(difference, axis=1)):
            q = np.clip(field, -2, 2) + 2
            word = 25 * q[:, :-2] + 5 * q[:, 1:-1] + q[:, 2:]
            count = np.bincount(ORBITS[word.ravel()], minlength=39)
            blocks.append(np.sqrt(count / word.size))
    return np.concatenate(blocks)


def random_tile(source):
    """Source-seeded allocation for the experimental random-location control only."""
    digest = hashlib.sha256(("random-patch-73|" + source).encode()).digest()
    return int(digest[0]) % 16


def read_image(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Image checksum differs from manifest for {path}")
    with Image.open(io.BytesIO(payload)) as img:
        if img.mode != "L":
            raise ValueError(f"Invalid image mode in {path}")
        gray = np.asarray(img).copy()
    require_image(gray)
    return gray


def extract_one(job):
    parent, row = job
    rid = random_tile(row["source"])
    output = []
    for role in ("cover", "stego"):
        gray = read_image(parent / row[role], row[role + "_sha256"])
        sid = select_tile(gray)
        vectors = {i: describe_tile(gray, i) for i in {FIXED, rid, sid}}
        output.append((sid, rid, vectors[FIXED], vectors[rid], vectors[sid]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Manifest from tools/prepare_fullscale.py"
    )
    parser.add_argument("--previous-policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--domain", choices=("WOW", "HILL"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    parent = args.manifest.parent
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        pairs = sorted(
            (r for r in csv.DictReader(f) if r["split"] == args.split),
            key=lambda r: r["source"],
        )
    assert len(pairs) == {"train": 6000, "val": 1000, "test": 3000}[args.split]
    assert len({r["source"] for r in pairs}) == len(pairs)
    source = np.repeat([r["source"] for r in pairs], 2)
    label = np.tile(np.array([0, 1], dtype="int8"), len(pairs))
    prior = {"train": "oof", "val": "val", "test": "test_" + args.domain}[args.split]
    with np.load(args.previous_policy / args.domain / (prior + ".npz"), allow_pickle=False) as gold:
        assert np.array_equal(source, gold["source"]) and np.array_equal(label, gold["label"])
        assert np.array_equal(
            np.repeat([r["source_sha256"] for r in pairs], 2), gold["source_sha256"]
        )
        assert np.array_equal(
            np.repeat([r["cover_sha256"] for r in pairs], 2), gold["cover_sha256"]
        )
    outdir = args.out / args.domain
    outdir.mkdir(exist_ok=True, parents=True)
    path = outdir / ("patch_" + args.split + ".npz")
    if path.exists():
        raise FileExistsError(path)
    n = len(source)
    features = {
        name: np.empty((n, 156), dtype=np.float64) for name in ("fixed", "random", "selected")
    }
    ids = {name: np.empty(n, dtype=np.int8) for name in ("random_tile", "selected_tile")}
    jobs = ((parent, row) for row in pairs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, examples in enumerate(pool.map(extract_one, jobs, chunksize=16)):
            for j, (sid, rid, fix, rnd, sel) in enumerate(examples, 2 * i):
                ids["random_tile"][j] = rid
                ids["selected_tile"][j] = sid
                features["fixed"][j], features["random"][j], features["selected"][j] = (
                    fix,
                    rnd,
                    sel,
                )
    for x in features.values():
        norms = (x.reshape(n, 4, 39) ** 2).sum(2)
        assert np.isfinite(x).all() and np.max(abs(norms - 1)) < 1e-12
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        source=source,
        label=label,
        fixed_tile=np.full(n, FIXED, dtype=np.int8),
        **ids,
        **features,
    )
    os.replace(tmp, path)
    print(
        args.domain,
        args.split,
        n,
        "images; selected_tile occupancy",
        np.bincount(ids["selected_tile"], minlength=16).tolist(),
        flush=True,
    )


if __name__ == "__main__":
    main()

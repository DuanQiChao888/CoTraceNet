"""Build one split of full-image residual grids; verify pair identity and file bytes."""

import argparse
import csv
import ctypes
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def create(manifest, domain, split, out, library, workers):
    destination = out / domain / (split + "-grids.npz")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.parent
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(
            (r for r in csv.DictReader(f) if r["split"] == split),
            key=lambda row: row["source"],
        )
    assert len(rows) == {"train": 6000, "val": 1000, "test": 3000}[split]
    ids = [r["source"] for r in rows]
    assert len(set(ids)) == len(ids) and all(r["split"] == split for r in rows)
    grid = ctypes.CDLL(str(library.resolve())).spatial_grid
    grid.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint16, ndim=3, flags="C_CONTIGUOUS"),
    ]
    grid.restype = None
    x = np.empty((len(rows) * 2, 144, 4, 4), np.uint16)

    def read_pair(item):
        i, row = item
        for label, key in enumerate(("cover", "stego")):
            blob = (data / row[key]).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == row[key + "_sha256"]
            with Image.open(io.BytesIO(blob)) as im:
                image = np.array(im, dtype=np.uint8)
            assert image.shape == (256, 256)
            grid(image, x[2 * i + label])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(read_pair, enumerate(rows)))
    groups = x.reshape(len(x), 8, 18, 4, 4)
    assert np.all(groups[:, :, :5].sum(axis=2) == 4096)
    assert np.all(groups[:, :, 5:].sum(axis=2) == 4096)
    tmp = destination.with_name(f".{split}-{os.getpid()}.npz")
    np.savez(
        tmp,
        source=np.repeat(np.asarray(ids), 2),
        label=np.tile([0, 1], len(ids)),
        counts=x,
    )
    os.replace(tmp, destination)
    print(domain, split, x.shape, "cached", flush=True)
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "out", "library"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--domain", choices=("WOW", "HILL"), required=True)
    p.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), required=True
    )
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    for name in args.splits:
        create(args.manifest, args.domain, name, args.out, args.library, args.workers)

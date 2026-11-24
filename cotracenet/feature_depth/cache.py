"""Cache the three-route descriptors for a BOSSbase manifest, one source pair at a time."""

import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..data import read_splits
from ..iterative import descriptors
from ..binocular.prepare import WIDTH
import hashlib

SPLIT_PAIRS = {"train": 6000, "val": 1000, "test": 3000}
IDENTITY = ("source", "source_sha256", "cover_sha256", "stego_sha256")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def manifest_rows(manifest, split):
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(
            (r for r in csv.DictReader(f) if r["split"] == split),
            key=lambda r: r["source"],
        )
    if len(rows) != SPLIT_PAIRS[split] or len({r["source"] for r in rows}) != len(rows):
        raise ValueError(
            f"Expected {SPLIT_PAIRS[split]} distinct source pairs for {split}"
        )
    if any(not row.get(k) for row in rows for k in IDENTITY):
        raise ValueError(
            "Use the fullscale paired manifest with source, cover and stego SHA-256 columns"
        )
    return rows


def read_matrix(root, domain, split):
    base = root / ("fullscale_" + domain.lower())
    folder = base / "features"
    manifest = base / "data" / "manifest.csv"
    matrix_path, rows_path = folder / (split + ".npy"), folder / (split + ".csv")
    receipt = json.loads((folder / (split + ".json")).read_text(encoding="utf-8"))
    for path, field in [
        (manifest, "manifest_sha256"),
        (rows_path, "ids_sha256"),
        (matrix_path, "npy_sha256"),
    ]:
        if digest(path) != receipt[field]:
            raise ValueError(f"Cache receipt mismatch for {path}")
    expected = manifest_rows(manifest, split)
    with rows_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(expected) or any(
        int(r["index"]) != i
        or r["split"] != split
        or any(r[k] != m[k] for k in IDENTITY)
        for i, (r, m) in enumerate(zip(rows, expected))
    ):
        raise ValueError("Cache rows disagree with the paired manifest")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(rows), 2, WIDTH) or matrix.dtype != np.float64:
        raise ValueError(
            "Expected double-precision three-route features for each cover/stego pair"
        )
    return (
        matrix,
        np.repeat([r["source"] for r in rows], 2),
        np.tile(np.array([0, 1], np.int8), len(rows)),
    )


def extract(root, domain, manifest, split, workers):
    base = root / ("fullscale_" + domain.lower())
    folder = base / "features"
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = base / "data" / "manifest.csv"
    snapshot.parent.mkdir(exist_ok=True)
    rows = manifest_rows(manifest, split)
    pairs = sorted(read_splits(manifest, (split,))[split])
    if workers < 0 or len(pairs) != len(rows):
        raise ValueError("Invalid worker count or pairs")
    manifest_hash = digest(manifest)
    with TemporaryDirectory(dir=folder) as temporary:
        tmp = Path(temporary)
        manifest_copy = tmp / "manifest.csv"
        manifest_copy.write_bytes(manifest.read_bytes())
        if digest(manifest_copy) != manifest_hash:
            raise ValueError("Manifest changed during extraction")
        # A hard link installs a complete, immutable manifest even across concurrent starts.
        try:
            os.link(manifest_copy, snapshot)
        except FileExistsError:
            if digest(snapshot) != manifest_hash:
                raise ValueError("Another manifest already owns this domain cache")
        if (folder / (split + ".json")).exists():
            read_matrix(root, domain, split)
            return
        matrix_path = tmp / (split + ".npy")
        index_path = tmp / (split + ".csv")
        matrix = np.lib.format.open_memmap(
            matrix_path, mode="w+", dtype=np.float64, shape=(len(pairs), 2, WIDTH)
        )
        with index_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "split", *IDENTITY])
            writer.writeheader()
            for i, ((name, paths), row) in enumerate(zip(pairs, rows)):
                if row["source"] != name or any(
                    digest(p) != row[k + "_sha256"]
                    for p, k in zip(paths, ("cover", "stego"))
                ):
                    raise ValueError(f"Image or index mismatch for {name}")
                writer.writerow(
                    dict(index=i, split=split, **{k: row[k] for k in IDENTITY})
                )
        count = 0
        for j, vector in enumerate(descriptors(pairs, workers)):
            if vector.shape != (WIDTH,) or not np.isfinite(vector).all():
                raise ValueError("Descriptor shape or values invalid")
            matrix[j // 2, j % 2] = vector
            count += 1
        if count != 2 * len(pairs):
            raise ValueError("Incomplete descriptor matrix")
        matrix.flush()
        shape = matrix.shape
        del matrix
        receipt = dict(
            manifest_sha256=manifest_hash,
            ids_sha256=digest(index_path),
            npy_sha256=digest(matrix_path),
            split=split,
            dtype="float64",
            shape=shape,
        )
        receipt_path = tmp / (split + ".json")
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        for p in (matrix_path, index_path, receipt_path):
            os.replace(p, folder / p.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--domain", choices=("WOW", "HILL"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=list(SPLIT_PAIRS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for split in args.splits:
        extract(args.root, args.domain, args.manifest, split, args.workers)

"""Rebuild the two full-size BOSSbase benchmarks from the official archive."""

import argparse
import csv
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from prepare_pairs import archive_members, historical_cohorts, make_source, sha


TARGETS = {"train": 6000, "val": 1000, "test": 3000}
PAIR_RECEIPTS = {
    "HILL_04": "729638638a3ffe59fde2e3f8a3c3d7ddd2aee0a80d1fc17f2924fbb975b35d9f",
    "WOW_04": "e09bda560a7cc624e669fcd37eadccca55277d18fd2d725c3313e229b96c7aaa",
}


def source_splits(members, cover_digests):
    original, fresh, seen = historical_cohorts(members)
    names = [f"{i}.pgm" for i in range(1, 10001)]
    eligible = [name for name in names if name not in seen]
    extension = np.random.default_rng(73640).choice(eligible, 480, replace=False)
    historical = seen.union(extension)
    split = {name: group for group, name in original + fresh}
    split.update((str(name), "train") for name in extension)
    rng = np.random.default_rng(73)
    unused = [name for name in names if name not in historical]
    split.update((str(name), "test") for name in rng.permutation(unused)[:2488])
    remaining = rng.permutation([name for name in names if name not in split])
    for i, name in enumerate(remaining):
        split[str(name)] = "val" if i < 952 else "train"

    # Four pairs have identical cover pixels under different archive filenames.
    # Preserve all files, move the duplicates together, then restore split sizes.
    groups = defaultdict(list)
    for name in names:
        groups[cover_digests[name]].append(name)
    duplicates = {name for names_in_group in groups.values() if len(names_in_group) > 1
                  for name in names_in_group}
    if len(groups) != 9996 or len(duplicates) != 8:
        raise ValueError("Unexpected set of duplicate BOSSbase cover images")
    for name in duplicates:
        split[name] = "train"
    counts = Counter(split.values())
    movable = np.random.default_rng(73).permutation(
        [name for name in names if name not in historical and name not in duplicates])
    for target, size in TARGETS.items():
        for name in movable:
            if counts[target] >= size:
                break
            old = split[name]
            if counts[old] > TARGETS[old]:
                counts[old] -= 1
                counts[target] += 1
                split[name] = target
    stamp = "".join(f"{name}\t{split[name]}\n" for name in names)
    if counts != TARGETS or sha(stamp.encode("ascii")) != "7fa21e5dc536c27698b1c05b2ff762cd7e03adc34e360304997362d629aa3c1b":
        raise ValueError("The reconstructed source split differs from the study")
    return split


def write_manifest(folder, records, splits, expected_receipt):
    ordered = sorted(records, key=lambda row: int(Path(row["source"]).stem))
    fields = ("source", "split", "cover_sha256", "stego_sha256")
    for row in ordered:
        row["split"] = splits[row["source"]]
    stamp = "".join("\t".join(row[k] for k in fields) + "\n" for row in ordered)
    if sha(stamp.encode("ascii")) != expected_receipt:
        raise ValueError(f"{folder.name}: image or split mismatch against the reference run")
    order = {key: i for i, key in enumerate(TARGETS)}
    with (folder / "manifest.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(ordered, key=lambda r: (order[r["split"]], r["source"])))


def prepare(archive_path, out, conditions, workers):
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    if workers < 1 or len(conditions) != len(set(conditions)):
        raise ValueError("Use positive workers and distinct conditions")
    with ZipFile(archive_path) as archive:
        members = archive_members(archive)
    historical_cohorts(members)  # Validate official archive order before processing.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "prepared"
        records = {condition: [] for condition in conditions}
        for condition in conditions:
            for role in ("cover", "stego"):
                (stage / condition / role).mkdir(parents=True)
        operation = partial(make_source, archive_path=archive_path.resolve(), output=stage,
                            conditions=conditions, fullscale=True)
        jobs = (("", name, member) for name, member in members.items())
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(operation, jobs, chunksize=8), 1):
                for condition, row in result:
                    records[condition].append(row)
                if i % 1000 == 0:
                    print(f"Prepared {i:,} / 10,000 sources", flush=True)
        covers = {r["source"]: r["cover_sha256"] for r in records[conditions[0]]}
        splits = source_splits(members, covers)
        for key, rows in records.items():
            write_manifest(stage / key, rows, splits, PAIR_RECEIPTS[key])
        stage.rename(out)
    print(f"Verified {', '.join(conditions)}: 6,000 train / 1,000 validation / 3,000 test pairs each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Local official BOSSbase 1.01 ZIP")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", choices=tuple(PAIR_RECEIPTS), default=list(PAIR_RECEIPTS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.archive, args.out, args.conditions, args.workers)

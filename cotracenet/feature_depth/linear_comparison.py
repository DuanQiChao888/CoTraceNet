"""Paired diagnostic: joint hierarchical model versus a separately fitted all-word linear classifier."""

import argparse
from pathlib import Path

import numpy as np

from .compare import DOMAINS, digest, load_case, paired, same, write_csv


def main(root):
    records = []
    for line in (root / "control-before-test.sha256").read_text().splitlines():
        saved, name = line.split(maxsplit=1)
        assert digest(root / name) == saved
    for source in DOMAINS:
        for target in DOMAINS:
            filename = "unified-test-" + target + ".npz"
            ids, deep, _ = load_case(
                root / "hierarchy" / source / filename, ["hierarchy", "baseline"]
            )
            ids2, shallow, _ = load_case(
                root / "hierarchy_control" / source / filename,
                ["hierarchy_linear", "baseline"],
            )
            assert (
                np.array_equal(ids, ids2)
                and len(ids) == 3000
                and same(deep["baseline"], shallow["baseline"])
            )
            linear_hits = (shallow["hierarchy_linear"] > 0) == [0, 1]
            deep_hits = (deep["hierarchy"] > 0) == [0, 1]
            records.append(
                dict(
                    source=source,
                    target=target,
                    images=linear_hits.size,
                    deep_correct=int(deep_hits.sum()),
                    linear_correct=int(linear_hits.sum()),
                    linear_pct=100 * linear_hits.mean(),
                    **paired(deep_hits, linear_hits)
                )
            )
    write_csv(root / "same-input-linear.csv", records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root)

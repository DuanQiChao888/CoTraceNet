"""Summarize locked model and path-removal diagnostics. Export aggregate counts only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def meta(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root):
    rows = []
    for source in ("WOW", "HILL"):
        linear_root = root / "hierarchy_control" / source
        for seed, sub in (
            (73, root / "hierarchy"),
            (29, root / "repeat29" / "hierarchy"),
        ):
            run = sub / source
            selection = meta(run / "unified-models.json")["hierarchy"]
            standalone = meta(run / "standalone-test.json")
            ablated = meta(run / "path-ablation-test.json")["targets"]
            for target in ("WOW", "HILL"):
                stats = standalone[target]
                masks = ablated[target]
                assert stats["standalone"] == masks["full"]["standalone_correct"]
                assert stats["combined"] == masks["full"]["mixture_correct"]
                row = dict(
                    seed=seed,
                    train=source,
                    test=target,
                    images=6000,
                    params=82461,
                    val_correct=selection["correct"],
                    epoch=selection["epoch"],
                    alpha=selection["alpha"],
                    baseline_correct=stats["baseline"],
                    full_correct=stats["combined"],
                    standalone_correct=stats["standalone"],
                    remove_deep_correct=masks["skip_only"]["mixture_correct"],
                    remove_word_skip_correct=masks["deep_only"]["mixture_correct"],
                    standalone_skip_only_correct=masks["skip_only"][
                        "standalone_correct"
                    ],
                    standalone_deep_only_correct=masks["deep_only"][
                        "standalone_correct"
                    ],
                    independent_word_control_correct="",
                    full_rescued_vs_control="",
                    full_harmed_vs_control="",
                )
                if seed == 73:
                    filename = "unified-test-" + target + ".npz"
                    with np.load(linear_root / filename) as control, np.load(
                        run / filename
                    ) as full:
                        assert np.array_equal(control["source"], full["source"])
                        y = full["label"]
                        assert np.array_equal(control["label"], y)
                        np.testing.assert_array_equal(
                            full["baseline"], control["baseline"]
                        )
                        ch = (control["hierarchy_linear"] > 0) == y
                        fh = (full["hierarchy"] > 0) == y
                        row.update(
                            independent_word_control_correct=int(ch.sum()),
                            full_rescued_vs_control=int((fh & ~ch).sum()),
                            full_harmed_vs_control=int((~fh & ch).sum()),
                        )
                rows.append(row)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = summarize(a.root)
    with a.output.open("w", encoding="utf-8-sig", newline="") as out:
        w = csv.DictWriter(out, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

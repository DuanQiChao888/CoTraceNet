"""Paired test accuracy differences grouped by original image."""

import csv
import numpy as np

BOOT_REPEATS = 10_000
BOOT_SEED = 7319


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired(candidate, reference):
    changes = candidate.astype(int) - reference.astype(int)
    rescued, harmed = int((changes == 1).sum()), int((changes == -1).sum())
    # The pair contributes -2, -1, 0, 1 or 2 additional correct images per draw.
    levels = np.arange(-2, 3)
    counts = np.bincount(changes.sum(axis=1) + 2, minlength=5)
    rng = np.random.default_rng(BOOT_SEED)
    samples = rng.multinomial(len(changes), counts / len(changes), size=BOOT_REPEATS)
    interval = np.quantile((samples @ levels) * 100 / changes.size, [0.025, 0.975])
    return dict(
        delta_pp=100 * (rescued - harmed) / changes.size,
        corrected=rescued,
        introduced=harmed,
        ci_low_pp=interval[0],
        ci_high_pp=interval[1],
    )

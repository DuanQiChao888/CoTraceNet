"""Five-region, 25,350-dimensional residual co-occurrences."""

import numpy as np
from .fourth import ORBIT


def uniform(field, size):
    p = np.pad(field, size // 2, mode="symmetric")
    integral = np.pad(p, ((1, 0), (1, 0))).cumsum(0, dtype=np.float64).cumsum(1)
    return (
        integral[size:, size:]
        + integral[:-size, :-size]
        - integral[size:, :-size]
        - integral[:-size, size:]
    ) / (size * size)


def histogram(x, mask, step):
    q = (np.sign(x) * np.minimum((np.abs(x) + step // 2) // step, 2) + 2).astype(np.int16)
    code = ((q[:, :-3] * 5 + q[:, 1:-2]) * 5 + q[:, 2:-1]) * 5 + q[:, 3:]
    return np.bincount(
        (ORBIT[code] + int(ORBIT.max() + 1) * mask[:, 1:-2]).ravel(),
        minlength=5 * int(ORBIT.max() + 1),
    ).reshape(5, -1)


def sections(image):
    p = np.pad(image.astype(np.float64), 1, mode="symmetric")
    axis = p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1]
    diag = p[:-2, :-2] + p[2:, :-2] + p[:-2, 2:] + p[2:, 2:]
    residual = np.abs(2 * axis - diag - 4 * image)
    cost = uniform(1 / np.maximum(uniform(residual, 3), 1.0e-8), 15).clip(1.0e-12, 1.0e10)
    # Coarse group is fixed by the training embedding family. It is estimated from
    # each submitted image, even for WOW/JPEG; no payload label or paired input is used.
    low, high = np.quantile(cost, [1 / 3, 2 / 3])
    fine = uniform(residual, 5)
    # Conditional medians isolate the relatively quieter and noisier patches *within*
    # each candidate high/medium-activity area. Background sites share a fifth bin.
    result = np.full(image.shape, 4, dtype=np.int16)
    for j, flag in enumerate((cost <= low, (cost > low) & (cost <= high))):
        if flag.any():
            result[flag] = j * 2 + (fine[flag] > np.median(fine[flag]))
    return result


def features(image):
    x = image.astype(np.int16)
    mask = sections(x)
    y = x[2:-2, 2:-2]
    t = mask[2:-2, 2:-2]

    def at(i, j):
        return x[2 + i : 254 + i, 2 + j : 254 + j]

    groups = []
    for dy, dx in ((0, 1), (1, 1)):
        yi, xi = dx, -dy
        for gy, gx in ((dy, dx), (yi, xi)):
            u, v = at(gy, gx), at(-gy, -gx)
            outputs = [(u - y, 1), (u + v - 2 * y, 2), (at(2 * gy, 2 * gx) - 3 * u + 3 * y - v, 3)]
            groups.append(outputs)
    axis = at(-1, 0) + at(1, 0) + at(0, 1) + at(0, -1)
    diag = at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1)
    blocks = []

    def joined(a, b):
        return (a + b) / np.maximum((a + b).sum(1, keepdims=True), 1)

    for off in (0, 2):
        for (r, g), (s, _) in zip(groups[off], groups[off + 1]):
            for step in (g, 2 * g):
                blocks.append(joined(histogram(r, t, step), histogram(s.T, t.T, step)))
                blocks.append(joined(histogram(s, t, step), histogram(r.T, t.T, step)))
    for r, g in ((axis - 4 * y, 4), (2 * axis - diag - 4 * y, 4), (diag - 4 * y, 4)):
        for step in (g, 2 * g):
            blocks.append(joined(histogram(r, t, step), histogram(r.T, t.T, step)))
    return np.concatenate([b.ravel() for b in blocks]).astype(np.float32)

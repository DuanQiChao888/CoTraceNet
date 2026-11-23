"""Exploratory texture-band pooling and square-symmetric 2D residual descriptors."""

import numpy as np
import torch


MODES = ("activity", "orientation", "both", "coarse")


def pooling_map(mode):
    """Return histogram-coordinate -> pooled-coordinate; independent of all photographs."""
    if mode not in MODES:
        raise ValueError(mode)
    tie_dir = mode in ("orientation", "both", "coarse")
    tie_bands = mode in ("activity", "both", "coarse")
    keys = []
    for block in range(24):
        # 3 orders x 3 quantizers x 2 directions, followed by 2x3 isotropic filters.
        key = (block // 2 if tie_dir else block) if block < 18 else block
        for orbit in range(39):
            keys.append(("triple", key, orbit))
    for block in range(16):
        key = (block // 2 if tie_dir else block) if block < 12 else block
        for band in range(3):
            for orbit in range(169):
                keys.append(("fourth", key, band, orbit))
    for block in range(30):
        # Axial and diagonal residuals each have 3 orders, 2 quantizers and 2 directions.
        if block < 24:
            key = block % 12 if mode == "coarse" else block
            if tie_dir:
                key //= 2
        else:
            key = block
        for band in range(5):
            band_key = (0, 0, 1, 1, 2)[band] if tie_bands else band
            for orbit in range(169):
                keys.append(("region", key, band_key, orbit))
    lookup = {}
    return np.array([lookup.setdefault(key, len(lookup)) for key in keys], dtype=np.int32)


def project(joint, mode, floor):
    """Input is the public detector's 34,398-D joint sqrt-frequency feature."""
    ix = pooling_map(mode)
    if joint.shape[-1] != len(ix):
        raise ValueError("Expected 34,398 joint features")
    flat = joint.reshape(-1, len(ix))
    if floor:
        # A smooth pseudocount dampens isolated rare bins before standardization.
        flat = np.sqrt(flat**2 + floor) - np.sqrt(floor)
    count = np.bincount(ix)
    result = np.stack([np.bincount(ix, weights=row, minlength=len(count)) / count for row in flat])
    return result.reshape(joint.shape[:-1] + (len(count),))


def fit_fisher(x, ridge):
    flat = x.flatten(0, 1)
    mu = flat.mean(0)
    s = flat.std(0, unbiased=False).clamp_min(1e-7)
    z = (x - mu) / s
    m = z.mean(1)
    diff = (z[:, 1] - z[:, 0]) / 2
    noise = torch.cat([m - m.mean(0), diff - diff.mean(0)]) / np.sqrt(len(x) - 1)
    delta = diff.mean(0) * 2
    cov = noise @ noise.T + ridge * torch.eye(len(noise), device=x.device, dtype=x.dtype)
    v = delta - noise.T @ torch.linalg.solve(cov, noise @ delta)
    mag = (z @ v).flatten().std(unbiased=False).clamp_min(1e-10)
    w = v / (s * mag)
    return w, -(mu @ w)


def fit_model(x, compact_ridge):
    full_w, full_b = fit_fisher(x, compact_ridge)
    dim = x.shape[-1]
    wb = torch.zeros_like(full_w)
    bb = torch.zeros_like(full_b)
    rng = np.random.default_rng(73)
    for _ in range(24):
        indices = torch.as_tensor(rng.choice(dim, 384, replace=False), device=x.device)
        w, b = fit_fisher(x.index_select(-1, indices), 0.15)
        wb.index_add_(0, indices, w / 24)
        bb += b / 24
    scale = (x @ wb + bb).flatten().std(unbiased=False).clamp_min(1e-10)
    return 0.5 * (full_w + wb / scale), 0.5 * (full_b + bb / scale)


def orbit_lookup():
    digits = (np.arange(625)[:, None] // np.array([125, 25, 5, 1])) % 5
    corners = digits.reshape(-1, 2, 2)
    symmetries = []
    for k in range(4):
        square = np.rot90(corners, k, axes=(1, 2))
        for perm in (square, square[:, :, ::-1]):
            code = perm.reshape(-1, 4) @ np.array([125, 25, 5, 1])
            symmetries.extend([code, 624 - code])
    return np.unique(np.min(symmetries, axis=0), return_inverse=True)[1]


ORBIT = orbit_lookup()
BINS = int(ORBIT.max()) + 1


def boxmean(image, size):
    pad = size // 2
    integral = np.pad(np.pad(image, pad, mode="symmetric"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[size:, size:]
        + integral[:-size, :-size]
        - integral[size:, :-size]
        - integral[:-size, size:]
    ) / (size * size)


def descriptor(im):
    im = np.asarray(im, dtype=np.int16)
    y = im[2:-2, 2:-2]

    def shift(dy, dx):
        return im[2 + dy : im.shape[0] - 2 + dy, 2 + dx : im.shape[1] - 2 + dx]

    h, v = shift(0, 1) + shift(0, -1) - 2 * y, shift(1, 0) + shift(-1, 0) - 2 * y
    corner = shift(1, 1) + y - shift(1, 0) - shift(0, 1)
    hp = 2 * (h + v) - (shift(1, 1) + shift(1, -1) + shift(-1, 1) + shift(-1, -1) - 4 * y)
    cost = boxmean(1.0 / np.maximum(boxmean(np.abs(hp).astype(float), 3), 1e-8), 15)
    breaks = np.quantile(cost, [0.2, 0.4, 0.65])
    zone = np.searchsorted(breaks, cost)[:-1, :-1]
    offset = BINS * zone

    def bins(a, b, c, d, step):
        def digit(field):
            return (2 + np.sign(field) * np.minimum(2, (abs(field) + step // 2) // step)).astype(
                np.int32
            )

        code = ((digit(a) * 5 + digit(b)) * 5 + digit(c)) * 5 + digit(d)
        return np.bincount((offset + ORBIT[code]).ravel(), minlength=4 * BINS).reshape(4, BINS)

    def square(r, step):
        return bins(r[:-1, :-1], r[:-1, 1:], r[1:, :-1], r[1:, 1:], step)

    def crop(x):
        return x[:-1, :-1]

    parts = []
    for step in (1, 2, 4):
        # Positive/negative flips, rotations and reflections use the same histogram entry.
        for counts in (
            square(corner, step),
            square(h + v, 2 * step),
            square(hp, 3 * step),
            square(h, step) + square(v, step),
            bins(
                crop(shift(-1, 0) - y),
                crop(shift(0, 1) - y),
                crop(shift(0, -1) - y),
                crop(shift(1, 0) - y),
                step,
            ),
        ):
            norm = counts / np.maximum(counts.sum(1, keepdims=True), 1)
            parts.append(np.sqrt(norm).ravel())
    return np.concatenate(parts).astype(np.float32)

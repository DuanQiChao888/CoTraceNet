"""Prepare per-image sensory evidence and reproduce the frozen Lite baseline.

The input descriptors already contain square roots of conditional histograms.
The first 12 and final four texture families match cost families 0:12 and
24:28. Texture bins are weak/mid/strong. Cost bins are low-a/low-b,
mid-a/mid-b and high. For comparison we combine each pair in probability
space, giving its subregions equal weight, then renormalize. The result is
ordered high/mid/low cost to align with increasing texture activity.

The train stage reads the source train and validation matrices only. Test
inference reloads the saved source-domain mean and coefficients; it neither
refits the selector nor reads target-domain training feature matrices.
"""

import os
from pathlib import Path
import tempfile

import numpy as np
import torch


WIDTH = 34_398
K = 16_384
T0, C0 = 936, 9_048
F = 169
MATCH = np.r_[0:12, 24:28]
PERM = np.random.default_rng(73).permutation(16)
KEYS = ("T", "C", "Ht", "Hc", "D", "Dperm")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, **arrays):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as tmp:
        name = tmp.name
    try:
        np.savez(name, **arrays)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def normalize_root(hist):
    length = torch.linalg.vector_norm(hist, dim=-1, keepdim=True)
    return hist / torch.where(length > 0, length, 1.0)


def hellinger(a, b):
    # An empty region is represented by a point mass in a distinct missing bin.
    # This gives distance one from an observed distribution, zero between two empties.
    missing_a = a.amax(dim=-1) == 0
    missing_b = b.amax(dim=-1) == 0
    d2 = ((a - b) ** 2).sum(dim=-1) + (missing_a ^ missing_b).to(a.dtype)
    return (d2 * 0.5).clamp(0, 1).sqrt()


def bin_statistics(x, weights, match, perm):
    t = x[:, T0:C0].reshape(-1, 16, 3, F)
    cost = x[:, C0:].reshape(-1, 30, 5, F).index_select(1, match)
    tw = weights[T0:C0].reshape(16, 3, F)
    cw = weights[C0:].reshape(30, 5, F).index_select(0, match)
    ts = (t * tw).sum(dim=(-1, -2))
    cs = (cost * cw).sum(dim=(-1, -2))
    # These are coarse conditional distributions with equal subregion weighting,
    # not pixel-count-weighted reconstructions of the original cost map.
    coarse = torch.stack(
        (
            cost[:, :, 4],
            torch.linalg.vector_norm(cost[:, :, 2:4], dim=2),
            torch.linalg.vector_norm(cost[:, :, :2], dim=2),
        ),
        dim=2,
    )
    t = normalize_root(t)
    c = normalize_root(coarse)
    return (
        ts,
        cs,
        hellinger(t[:, :, 0], t[:, :, 2]),
        hellinger(c[:, :, 0], c[:, :, 2]),
        hellinger(t, c).mean(dim=2),
        hellinger(t, c.index_select(1, perm)).mean(dim=2),
    )


def make_split(raw, ix, weight, bias, mean, chunk, device):
    index = torch.as_tensor(ix, device=device)
    wk = torch.as_tensor(weight, dtype=torch.float64, device=device)
    w = torch.zeros(WIDTH, dtype=torch.float64, device=device)
    w[index] = wk
    match = torch.as_tensor(MATCH, device=device)
    perm = torch.as_tensor(PERM, device=device)
    arrays = {}
    base = np.empty(len(raw), dtype=np.float64)
    stats = {k: np.empty((len(raw), 16), dtype=np.float64) for k in KEYS}
    total = torch.zeros_like(w) if mean is None else None
    for start in range(0, len(raw), chunk):
        end = min(start + chunk, len(raw))
        x = torch.tensor(raw[start:end], dtype=torch.float64, device=device)
        require(
            bool(torch.isfinite(x).all()) and bool((x >= 0).all()),
            "Invalid square-root histograms",
        )
        base[start:end] = (x.index_select(1, index) @ wk + bias).cpu().numpy()
        for key, value in zip(KEYS, bin_statistics(x, w, match, perm)):
            stats[key][start:end] = value.cpu().numpy()
        if total is not None:
            total += x.sum(dim=0)
    if mean is None:
        mean = (total / len(raw)).cpu().numpy()
    center = torch.as_tensor(mean, dtype=w.dtype, device=device) * w
    stats["T"] -= center[T0:C0].reshape(16, 3, F).sum(dim=(-1, -2)).cpu().numpy()
    stats["C"] -= (
        center[C0:]
        .reshape(30, 5, F)
        .sum(dim=(-1, -2))
        .index_select(0, match)
        .cpu()
        .numpy()
    )
    arrays.update(stats)
    arrays["base"] = base
    if total is not None:
        for key in ("T", "C"):
            np.testing.assert_allclose(arrays[key].mean(axis=0), 0.0, rtol=0, atol=1e-9)
    return arrays, mean

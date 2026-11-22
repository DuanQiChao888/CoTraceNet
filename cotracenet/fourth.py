"""Texture-stratified fourth-order, 8,112-dimensional residual co-occurrences."""

import numpy as np


def orbit_lookup():
    codes = np.arange(625)
    rev = sum(((codes // (5**k)) % 5) * 5 ** (3 - k) for k in range(4))
    _, inv = np.unique(np.minimum.reduce([codes, rev, 624 - codes, 624 - rev]), return_inverse=True)
    return inv


ORBIT = orbit_lookup()
N = int(ORBIT.max()) + 1


def local_texture(im):
    """Mean gradient energy over 7x7; smooth pooling makes the partition stable."""
    p = np.pad(im, 1, mode="reflect")
    grad = abs(p[1:-1, 2:] - p[1:-1, :-2]) + abs(p[2:, 1:-1] - p[:-2, 1:-1])
    p = np.pad(grad, 3, mode="reflect")
    ii = np.pad(p, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    return ii[7:, 7:] - ii[:-7, 7:] - ii[7:, :-7] + ii[:-7, :-7]


def counts(r, activity, step, cuts):
    x = (np.sign(r) * np.minimum((np.abs(r) + step // 2) // step, 2) + 2).astype(np.int16)
    codes = ((x[:, :-3] * 5 + x[:, 1:-2]) * 5 + x[:, 2:-1]) * 5 + x[:, 3:]
    t = activity[:, 1:-2]
    levels = (t > cuts[0]).astype(np.int16) + (t > cuts[1]).astype(np.int16)
    return np.bincount((codes + levels * 625).ravel(), minlength=1875).reshape(3, 625)


def symmetric_blocks(r1, t1, r2, t2, step, cuts):
    hist = counts(r1, t1, step, cuts) + counts(r2, t2, step, cuts)
    pooled = np.stack([np.bincount(ORBIT, weights=h, minlength=N) for h in hist])
    # Normalize inside each activity band; the occupancy of empty bands is retained as zero.
    return (pooled / np.maximum(1, pooled.sum(1, keepdims=True))).ravel()


def features(image):
    im = image.astype(np.int16)
    energy = local_texture(im)
    cuts = np.quantile(energy, [1 / 3, 2 / 3])
    blocks = []
    dx, dy = im, im
    for order, gain in [(1, 1), (2, 2), (3, 3)]:
        dx = np.diff(dx, axis=1)
        dy = np.diff(dy, axis=0)
        start = order // 2
        tx = energy[:, start : start + dx.shape[1]]
        ty = energy[start : start + dy.shape[0], :]
        for q in [1, 2]:
            blocks.append(symmetric_blocks(dx, tx, dy.T, ty.T, gain * q, cuts))
            blocks.append(symmetric_blocks(dx.T, tx.T, dy, ty, gain * q, cuts))
    c = im[1:-1, 1:-1]
    axis = im[1:-1, :-2] + im[1:-1, 2:] + im[:-2, 1:-1] + im[2:, 1:-1]
    diag = im[:-2, :-2] + im[:-2, 2:] + im[2:, :-2] + im[2:, 2:]
    t = energy[1:-1, 1:-1]
    for residual in [axis - 4 * c, 2 * axis - diag - 4 * c]:
        for q in [1, 2]:
            blocks.append(symmetric_blocks(residual, t, residual.T, t.T, 4 * q, cuts))
    return np.concatenate(blocks).astype(np.float32)

"""Direction-symmetric, 936-dimensional residual co-occurrences."""

import numpy as np

QUANTIZERS = (1, 2, 4)
CONTEXT_NAMES = (
    "mean_abs_d1",
    "mean_abs_d2",
    "mean_abs_d3",
    "mean_abs_laplacian",
    "mean_abs_square_highpass",
    "p90_abs_laplacian",
    "mean_abs_square_low_texture",
    "mean_abs_square_high_texture",
)
DESCRIPTOR_VERSION = "symmetric-triples-t2-q124-v1"


def _orbit_table():
    # T=2 gives 5^3 bins. Fold (a,b,c), (c,b,a) and their sign reversals.
    idx = np.arange(125)
    reverse = (idx % 5) * 25 + ((idx // 5) % 5) * 5 + (idx // 25)
    representative = np.minimum.reduce((idx, reverse, 124 - idx, 124 - reverse))
    _, inverse = np.unique(representative, return_inverse=True)
    return inverse


ORBITS = _orbit_table()
N_ORBITS = int(ORBITS.max()) + 1  # 39 equivalence classes
FEATURE_DIM = len(QUANTIZERS) * (3 * 2 + 2) * N_ORBITS


def _quantize(residual, step):
    # Round to nearest integer (ties away from zero), clip to [-2,2], then
    # shift to [0,4]. All supported filters fit in int16 for 8-bit inputs.
    magnitude = np.minimum((np.abs(residual) + step // 2) // step, 2)
    return (np.sign(residual) * magnitude + 2).astype(np.uint8)


def _triple_counts(field):
    # Entries in [0,4] produce base-5 codes in [0,124] without overflow.
    codes = field[:, :-2] * 25 + field[:, 1:-1] * 5 + field[:, 2:]
    return np.bincount(codes.ravel(), minlength=125)


def _merge(counts):
    folded = np.bincount(ORBITS, weights=counts, minlength=N_ORBITS)
    return (folded / folded.sum()).astype(np.float32)


def extract_features(image):
    """Return (936 normalized histogram features, 8 unscaled texture statistics)."""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 6:
        raise ValueError("Expected an 8-bit grayscale image at least 6 x 6 pixels")
    field = image.astype(np.int16)
    blocks, directional_means = [], []
    dx, dy = field, field
    for gain in (1, 2, 3):
        dx = np.diff(dx, axis=1)
        dy = np.diff(dy, axis=0)
        directional_means.append(
            (np.abs(dx).sum(dtype=np.int64) + np.abs(dy).sum(dtype=np.int64)) / (dx.size + dy.size)
        )
        for q in QUANTIZERS:
            x, y = _quantize(dx, q * gain), _quantize(dy, q * gain)
            blocks.append(_merge(_triple_counts(x) + _triple_counts(y.T)))
            blocks.append(_merge(_triple_counts(x.T) + _triple_counts(y)))

    center = field[1:-1, 1:-1]
    north, south = field[:-2, 1:-1], field[2:, 1:-1]
    east, west = field[1:-1, 2:], field[1:-1, :-2]
    axis_sum = north + south + east + west
    corner_sum = field[:-2, :-2] + field[:-2, 2:] + field[2:, :-2] + field[2:, 2:]
    laplacian = axis_sum - 4 * center
    square = 2 * axis_sum - corner_sum - 4 * center
    for residual in (laplacian, square):
        for q in QUANTIZERS:
            quantized = _quantize(residual, q * 4)
            blocks.append(_merge(_triple_counts(quantized) + _triple_counts(quantized.T)))

    texture = (
        np.abs(center - north)
        + np.abs(center - south)
        + np.abs(center - east)
        + np.abs(center - west)
    )
    low = texture <= np.median(texture)
    highpass = np.abs(square)
    low_mean = highpass[low].mean()
    high_mean = highpass[~low].mean() if np.any(~low) else low_mean
    lap_abs = np.abs(laplacian)
    context = np.asarray(
        (
            *directional_means,
            lap_abs.mean(),
            highpass.mean(),
            np.percentile(lap_abs, 90),
            low_mean,
            high_mean,
        ),
        dtype=np.float32,
    )
    return np.concatenate(blocks), context

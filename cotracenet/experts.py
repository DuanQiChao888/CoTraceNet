"""Regularized Fisher discriminants and reproducible subspace averaging."""

import numpy as np
import torch


def fisher(x, ridge):
    # Tikhonov Fisher with a system sized by the number of training samples.
    flat = x.flatten(0, 1)
    m = flat.mean(0)
    sd = flat.std(0, unbiased=False)
    sd = torch.where(sd > 1.0e-10, sd, 1.0)
    z = (x - m) / sd
    means = z.mean(0)
    within = (z - means).reshape(flat.shape) / np.sqrt(len(flat) - 2)
    delta = means[1] - means[0]
    matrix = within @ within.T + ridge * torch.eye(len(flat), device=x.device, dtype=flat.dtype)
    w = (delta - within.T @ torch.linalg.solve(matrix, within @ delta)) / ridge
    scores = (z @ w).ravel()
    # Collapse standardization into an equivalent affine transform.
    scale = scores.std(unbiased=False).clamp_min(1.0e-10)
    weight = w / (sd * scale)
    bias = -(m @ weight) - scores.mean() / scale
    return weight, bias


def paired_discriminant(x, ridge, eta):
    flat = x.flatten(0, 1)
    mean = flat.mean(0)
    scale = flat.std(0, unbiased=False).clamp_min(1.0e-7)
    z = (x - mean) / scale
    mid = z.mean(1)
    diff = (z[:, 1] - z[:, 0]) / 2
    # Learn a direction that changes within pairs and resists score drift across source photos.
    nuisance = torch.cat((mid - mid.mean(0), np.sqrt(eta) * (diff - diff.mean(0))), 0) / np.sqrt(
        len(x) - 1
    )
    delta = diff.mean(0) * 2
    dual = torch.linalg.solve(
        nuisance @ nuisance.T + ridge * torch.eye(len(nuisance), dtype=x.dtype, device=x.device),
        nuisance @ delta,
    )
    weight = (delta - nuisance.T @ dual) / ridge
    support = (z @ weight).flatten()
    normalizer = support.std(unbiased=False).clamp_min(1.0e-8)
    w = weight / (scale * normalizer)
    b = -(mean @ w) - support.mean() / normalizer
    return w, b


def fit_action(x, reg, dim):
    if dim == 0:
        return paired_discriminant(x, reg, 1)
    # Regularized random-subspace discriminants. Each subspace receives all 160
    # paired training photos. The normalized scores are averaged with equal weight.
    dims = x.shape[-1]
    w = torch.zeros(dims, device=x.device, dtype=x.dtype)
    b = torch.zeros((), device=x.device, dtype=x.dtype)
    rng = np.random.default_rng(73)
    groups = [rng.choice(dims, dim, replace=False) for _ in range(32)]
    for indices in groups:
        ix = torch.as_tensor(indices, device=x.device)
        ww, bb = paired_discriminant(x[:, :, ix], reg, 1)
        w[ix] += ww / len(groups)
        b += bb / len(groups)
    magnitude = (x @ w + b).std(unbiased=False).clamp_min(1.0e-8)
    return w / magnitude, b / magnitude

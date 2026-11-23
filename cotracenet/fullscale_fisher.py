"""The fixed 16,384-coordinate Fisher baseline used in the full BOSSbase study."""

import math

import numpy as np
import torch


def columns(x, indices, start, end, device):
    block = x[..., start:end] if indices is None else x.index_select(2, indices[start:end])
    return block.contiguous().to(device)


def project(x, indices, weight, device):
    weight = torch.as_tensor(weight, device=device)
    scores = []
    for block in x.split(256):
        if indices is not None:
            block = block.index_select(2, indices)
        scores.append(block.to(device) @ weight)
    return torch.cat(scores)


def head(x, indices, ridge, device):
    """Solve the paired Fisher system on the smaller of the sample and feature axes."""
    n = len(x)
    width = x.shape[-1] if indices is None else len(indices)
    opts = dict(device=device, dtype=torch.float64)
    nuisance = torch.empty((2 * n, width), **opts)
    mean, std, delta = (torch.empty(width, **opts) for _ in range(3))
    for start in range(0, width, 1024):
        end = min(start + 1024, width)
        sl = slice(start, end)
        block = columns(x, indices, start, end, device)
        mean[sl] = block.mean((0, 1))
        std[sl] = block.flatten(0, 1).std(0, unbiased=False)
        z = (block - mean[sl]) / std[sl].clamp_min(1e-7)
        midpoint = z.mean(1)
        difference = (z[:, 1] - z[:, 0]) / 2
        center = difference.mean(0)
        nuisance[:n, sl] = (midpoint - midpoint.mean(0)) / math.sqrt(n - 1)
        nuisance[n:, sl] = (difference - center) / math.sqrt(n - 1)
        delta[sl] = 2 * center

    dual = width > 2 * n
    gram = nuisance @ nuisance.T if dual else nuisance.T @ nuisance
    gram.diagonal().add_(ridge)
    factor = torch.linalg.cholesky(gram)
    del gram
    if dual:
        sol = torch.cholesky_solve((nuisance @ delta)[:, None], factor)[:, 0]
        direction = (delta - nuisance.T @ sol) / ridge
    else:
        direction = torch.cholesky_solve(delta[:, None], factor)[:, 0]
    del nuisance, factor

    scores = torch.zeros((n, 2), **opts)
    for start in range(0, width, 1024):
        end = min(start + 1024, width)
        sl = slice(start, end)
        block = columns(x, indices, start, end, device)
        scores += ((block - mean[sl]) / std[sl].clamp_min(1e-7)) @ direction[sl]
    normalizer = scores.std(unbiased=False).clamp_min(1e-8)
    weight = direction / std.clamp_min(1e-7) / normalizer
    bias = -(mean @ weight) - scores.mean() / normalizer
    return weight.cpu().numpy(), float(bias), std.cpu().numpy()


def mixture(x, indices, device):
    width = x.shape[-1] if indices is None else len(indices)
    dense, intercept, std = head(x, indices, 1.5, device)
    rng = np.random.default_rng(73)
    bank, bias = np.zeros(width), 0.0
    for _ in range(32):
        drawn = torch.from_numpy(rng.choice(width, 512, replace=False))
        cols = drawn if indices is None else indices[drawn]
        w, b, _ = head(x, cols, 0.12, device)
        bank[drawn.numpy()] += w / 32
        bias += b / 32
    norm = float((project(x, indices, bank, device) + bias).std(unbiased=False).clamp_min(1e-8))
    return 0.5 * dense + bank * (0.5 / norm), 0.5 * intercept + bias * (0.5 / norm), std


def fit(x, retained, device):
    if len(x) < 2:
        raise ValueError("The paired Fisher baseline needs at least two training pairs")
    teacher, _, std = mixture(x, None, device)
    importance = torch.as_tensor(np.abs(teacher * std), device=device)
    indices = torch.topk(importance, retained).indices.sort().values.cpu()
    student, bias, _ = mixture(x, indices, device)
    teacher = teacher[indices.numpy()]
    raw = project(x, indices, teacher, device)
    norm = float(raw.std(unbiased=False).clamp_min(1e-8))
    return (
        indices.numpy(),
        0.5 * student + teacher * (0.5 / norm),
        0.5 * bias - float(raw.mean()) * (0.5 / norm),
    )

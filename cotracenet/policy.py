"""Learn a bounded three-action correction around a validation-selected static anchor."""

import numpy as np
import torch
from torch.nn import functional as F
from .data import classification_metrics


def order(stats):
    return stats["accuracy"], stats["auc"]


def score_stats(scores):
    labels = np.broadcast_to(np.arange(2), scores.shape)
    return classification_metrics(labels.ravel(), scores.ravel())


def norm_stats(array):
    mean, stdev = array.mean(0), array.std(0)
    return mean, np.where(stdev > 1e-10, stdev, 1.0)


TAU = 0.55
AMPLITUDES = (0.0, 0.25, 0.5, 1.0)
HARM = (1.0, 2.0)
RIDGE = (0.02, 0.15)
BIASES = (-0.24, -0.12, 0.0, 0.12, 0.24)


def mixtures():
    for i in range(5):
        for j in range(5 - i):
            for k in range(5 - i - j):
                yield np.array([i, j, k, 4 - i - j - k], dtype=float) / 4


def static_candidates(actions):
    choices = []
    for w in mixtures():
        raw = actions @ w
        for b in BIASES:
            st = score_stats(raw + b)
            choices.append((order(st), w, b, raw + b))
    return sorted(choices, key=lambda x: x[0], reverse=True)


def descriptors(actions, anchor, ctx, cnn):
    c, f, l, b = actions.transpose(2, 0, 1)
    texture = np.log1p(ctx[..., [0, 3, 4, 6, 7]])
    # Distinguish consensus from a single outlier. CNN is capped to avoid allowing
    # its content-dependent score to directly override the protected anchor.
    signed = np.stack(
        (
            c,
            f,
            l,
            b,
            anchor,
            abs(anchor),
            b - l,
            l - c,
            b - c,
            f - c,
            actions.std(-1),
            np.tanh(cnn) * np.sign(c),
            texture[..., 3] - texture[..., 4],
        ),
        axis=-1,
    )
    return np.concatenate((signed, texture), axis=-1)


def action_banks(acts, ctx, cnn, ranked):
    # Use several predeclared training anchors. Training labels never choose a
    # policy for an individual deployment image; this augmentation only supplies
    # the policy with examples of calibrated alternative source detectors.
    anchors = [(w, 0.0) for w in np.eye(4)]
    anchors.extend((r[1], r[2]) for r in ranked[:4])
    z = []
    a = []
    for w, b in anchors:
        s = acts @ w + b
        a.append(s)
        z.append(descriptors(acts, s, ctx, cnn))
    return np.stack(a), np.stack(z)


def train_policy(anchors, raw, lam, harm, reward, device):
    def as_tensor(value):
        return torch.as_tensor(value, device=device, dtype=torch.float64)

    mu, sd = norm_stats(raw.reshape(-1, raw.shape[-1]))
    z = as_tensor(np.clip((raw - mu) / sd, -4, 4))
    q = as_tensor(anchors)
    y = torch.arange(2, device=q.device).double().expand_as(q)
    offsets = as_tensor(np.array([0.0, TAU, -TAU]))
    actions = q[..., None] + offsets
    ok = (actions > 0) == y[..., None]
    base = ok[..., :1]
    utility = (ok & (~base)).double() - harm * (base & (~ok)).double()
    w = torch.zeros((z.shape[-1], 3), dtype=q.dtype, device=q.device, requires_grad=True)
    b = torch.zeros(3, dtype=q.dtype, device=q.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.025)
    for _ in range(160):
        pi = (z @ w + b).softmax(-1)
        sc = q + (pi @ offsets)
        loss = (
            F.binary_cross_entropy_with_logits(sc, y)
            + 0.25 * F.softplus(-(sc[..., 1] - sc[..., 0])).mean()
            - reward * (pi * utility).sum(-1).mean()
            + lam * w.square().sum()
            + 0.01 * b.square().sum()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dict(gw=w.detach().cpu().numpy(), gb=b.detach().cpu().numpy(), mu=mu, sd=sd)


def adjustment(z, g):
    v = np.clip((z - g["mu"]) / g["sd"], -4, 4) @ g["gw"] + g["gb"]
    v = np.exp(v - v.max(-1, keepdims=True))
    v /= v.sum(-1, keepdims=True)
    return TAU * (v[..., 1] - v[..., 2]), v


def apply_final(q, d, alpha, mean):
    # Mean subtraction isolates the per-image adjustment from global recalibration.
    return q + alpha * (d - mean)

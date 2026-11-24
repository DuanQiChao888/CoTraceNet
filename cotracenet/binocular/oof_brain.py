"""Generate source-grouped out-of-fold scores and features for a final decision policy.

Training re-fits the feature selector, Lite readout and both stages of the brain
on the other two folds. Hyperparameters were fixed in the earlier attention
experiment. Validation and test apply the original frozen full-training model.
"""

import hashlib

import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F
from . import attention, prepare
from .grouping import source_groups


def folds_from_sources(rows):
    groups = source_groups(
        [r["source_sha256"] for r in rows],
        [r["cover_sha256"] for r in rows],
        range(len(rows)),
    )

    def priority(indexes):
        ident = min(rows[i]["source_sha256"] for i in indexes)
        return (
            -len(indexes),
            hashlib.sha256(("policy-oof-73|" + ident).encode()).hexdigest(),
        )

    counts = [0] * 3
    ids = np.full(len(rows), -1, dtype=np.int64)
    for group in sorted(groups, key=priority):
        fold = int(np.argmin(counts))
        ids[group] = fold
        counts[fold] += len(group)
    for col in ("source_sha256", "cover_sha256"):
        found = {}
        for f, r in zip(ids, rows):
            assert found.setdefault(r[col], f) == f
    if min(counts) < 2:
        raise ValueError(
            "Three source-grouped folds require at least two independent source pairs per fold"
        )
    return np.repeat(ids, 2)


def optimize_linear(z, labels, penalty, iterations):
    def objective(p):
        theta = torch.as_tensor(p, device=z.device, dtype=z.dtype)
        v, b = theta[:-1], theta[-1]
        scores = z @ v + b
        loss = F.binary_cross_entropy_with_logits(
            scores, labels
        ) + 0.5 * penalty * v.dot(v)
        err = (scores.sigmoid() - labels) / len(labels)
        return (
            float(loss),
            torch.cat((z.T @ err + penalty * v, err.sum().reshape(1))).cpu().numpy(),
        )

    r = minimize(
        objective,
        np.zeros(z.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": iterations,
            "maxcor": 15,
            "ftol": 1e-12,
            "gtol": 1e-7,
            "maxls": 35,
        },
    )
    if r.status not in (0, 1) or not np.isfinite(r.fun):
        raise RuntimeError(str(r.message))
    return r.x, r.nit


def logistic_model(raw, label, rows, device, ridge, steps):
    z = torch.as_tensor(raw[rows], device=device, dtype=torch.float64)
    y = torch.as_tensor(label[rows], device=device, dtype=torch.float64)
    var, mean = torch.var_mean(z, dim=0, unbiased=False)
    scale = var.sqrt().clamp_min(1e-7)
    z.sub_(mean).div_(scale)
    theta, selsteps = optimize_linear(z, y, 0.1, 200)
    keep = np.sort(np.argsort(-np.abs(theta[:-1]), kind="stable")[:16384])
    index = torch.as_tensor(keep, device=device)
    z = z.index_select(1, index)
    if steps:
        fitted, steps = optimize_linear(z, y, ridge, steps)
    else:
        fitted = np.zeros(z.shape[1] + 1)
    w = torch.as_tensor(fitted[:-1], device=device) / scale[index]
    b = float(fitted[-1] - mean[index].dot(w))
    return keep, w.cpu().numpy(), b, mean.cpu().numpy(), (selsteps, steps)


def fit_gate(data, kind, fixed, device):
    batch, stats = attention.build_inputs(data, kind, None, device)
    initial = attention.init(kind)
    limit = fixed["step"]
    penalty = fixed["penalty"]
    if limit == 0:
        return initial, stats, 0

    def objective(p):
        w = torch.tensor(p, device=device, dtype=torch.float64, requires_grad=True)
        scores, _ = attention.forward(batch, kind, w)
        loss = F.binary_cross_entropy_with_logits(
            scores, batch[-1]
        ) + 0.5 * penalty * w.dot(w)
        loss.backward()
        return float(loss.detach()), w.grad.detach().cpu().numpy()

    r = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": limit,
            "maxcor": 12,
            "maxls": 40,
            "ftol": 1e-12,
            "gtol": 1e-7,
        },
    )
    if r.status not in (0, 1) or not np.isfinite(r.fun):
        raise RuntimeError(str(r.message))
    return r.x, stats, r.nit


def gate_score(data, kind, theta, stats, device):
    batch, _ = attention.build_inputs(data, kind, stats, device)
    with torch.no_grad():
        return (
            attention.forward(
                batch, kind, torch.tensor(theta, device=device, dtype=torch.float64)
            )[0]
            .cpu()
            .numpy()
        )


def subset(d, ids):
    return {k: v[ids] for k, v in d.items()}


def train(raw, identity, selection, ridge, steps, device):
    """Re-fit the selector, linear head and both attention layers for each holdout."""
    manifest = [
        dict(source_sha256=src, cover_sha256=cover)
        for src, cover in zip(
            identity["source_sha256"][::2], identity["cover_sha256"][::2]
        )
    ]
    folds = folds_from_sources(manifest)
    packed = dict(identity, fold=folds)
    keys = ("base", *prepare.KEYS, "se", "brain")
    for fold in range(3):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        keep, weights, bias, mean, _ = logistic_model(
            raw, identity["label"], fit, device, ridge, steps
        )
        evidence, _ = prepare.make_split(raw, keep, weights, bias, mean, 128, device)
        evidence["label"] = identity["label"]
        for name, kind in (("se", "se"), ("brain", "binocular")):
            inputs = evidence if name == "se" else dict(evidence, base=evidence["se"])
            theta, scaler, _ = fit_gate(
                subset(inputs, fit), kind, selection[name], device
            )
            evidence[name] = gate_score(inputs, kind, theta, scaler, device)
        for name in keys:
            if name not in packed:
                packed[name] = np.empty_like(evidence[name])
            packed[name][held] = evidence[name][held]
        print(f"Fitted fold {fold+1}/3 ({len(held)//2} held-out pairs)", flush=True)
    return packed

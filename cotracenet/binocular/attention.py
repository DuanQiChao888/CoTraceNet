"""Fit and evaluate lightweight residual-family attention on prepared single-image evidence."""

import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
import torch
from torch.nn import functional as F


SEED = 73
FAMILIES = 16
PERM = np.random.default_rng(SEED).permutation(FAMILIES)
VARIANTS = ("fixed", "se", "self", "binocular", "shuffled")
SECOND_STAGE = {
    "brain_self": "self",
    "brain": "binocular",
    "brain_shuffled": "shuffled",
}
PENALTIES = (0.003, 0.03, 0.3)
STEPS = (0, 5, 15, 40, 100)
BOUND = 0.5


def load_split(path):
    with np.load(path, allow_pickle=False) as src:
        data = {k: src[k] for k in src.files}
    n = len(data["label"])
    for k in ("T", "C", "Ht", "Hc", "D", "Dperm"):
        assert data[k].shape == (n, FAMILIES) and np.isfinite(data[k]).all(), (
            k,
            data[k].shape,
        )
    for k in ("label", "base", "source"):
        data[k] = np.asarray(data[k]).reshape(n)
    assert np.isfinite(data["base"]).all()
    for y in (0, 1):
        ids = data["source"][data["label"] == y]
        assert len(np.unique(ids)) == len(ids) == n // 2
    assert set(data["source"][data["label"] == 0]) == set(
        data["source"][data["label"] == 1]
    )
    return data


def auc(y, s):
    pos = y == 1
    n = pos.sum()
    return float((rankdata(s)[pos].sum() - n * (n + 1) / 2) / (n * (~pos).sum()))


def scales(x):
    return x.mean(axis=0), x.std(axis=0).clip(1e-7)


def build_inputs(d, kind, stats, device):
    # The shuffled control changes the correspondence once, for every image and split.
    order = PERM if kind == "shuffled" else np.arange(FAMILIES)
    t = d["T"].astype(np.float64)
    c = d["C"][:, order].astype(np.float64)
    primitive = np.stack([t, c, d["Ht"], d["Hc"][:, order]], axis=-1)
    fitting = stats is None
    if fitting:
        pm, ps = scales(primitive)
        stats = {"pm": pm, "ps": ps}
    u, v, h, k = np.moveaxis((primitive - stats["pm"]) / stats["ps"], -1, 0)
    base = d["base"]
    if kind == "se":
        x = np.concatenate([u, v], axis=1)
    elif kind == "fixed":
        x = np.empty((len(t), FAMILIES, 0), dtype=t.dtype)
    else:
        if kind == "self":
            # Two independent eye-specific attention logits followed by two-way softmax.
            extra = [u * u, v * v, h * h, k * k]
        else:
            # Registered filter families can compare vote agreement and conditional word distributions.
            extra = [
                abs(u - v),
                np.tanh(u) * np.tanh(v),
                abs(h - k),
                d["Dperm"] if kind == "shuffled" else d["D"],
            ]
        x = np.stack([u, v, h, k, *extra], axis=-1)
        if fitting:
            stats["xm"], stats["xs"] = scales(x)
        x = (x - stats["xm"]) / stats["xs"]

    def tensor(a):
        return torch.as_tensor(
            np.ascontiguousarray(a), dtype=torch.float64, device=device
        )

    return tuple(tensor(a) for a in (base, t, c, x, d["label"])), stats


def init(kind):
    if kind == "se":
        # Pooled 32-channel branch votes -> 2 hidden units -> 32 independent channel weights.
        w1 = np.random.default_rng(SEED).normal(0, 1 / np.sqrt(32), (32, 2))
        return np.r_[w1.ravel(), np.zeros(2 + 2 * 32 + 32)]
    return np.zeros(FAMILIES if kind == "fixed" else FAMILIES * 9)


def forward(batch, kind, theta):
    base, t, c, x, _ = batch
    if kind == "se":
        hidden = torch.relu(x @ theta[:64].reshape(32, 2) + theta[64:66])
        logits = hidden @ theta[66:130].reshape(2, 32) + theta[130:]
        coeff = BOUND * torch.tanh(logits)
        change = (coeff * torch.cat((t, c), dim=1)).sum(dim=1)
    else:
        if kind == "fixed":
            logits = theta.reshape(1, FAMILIES)
        else:
            w = theta.reshape(FAMILIES, 9)
            logits = (x * w[None, :, :8]).sum(dim=2) + w[:, 8]
        # Opposite adjustments keep each pair's total multiplier equal to two.
        coeff = BOUND * torch.tanh(logits)
        change = (coeff * (t - c)).sum(dim=1)
    return base + change, coeff


def build_and_fit(train, valid, kind, device):
    tr, stats = build_inputs(train, kind, None, device)
    va, _ = build_inputs(valid, kind, stats, device)
    initial = init(kind)

    def to_tensor(p, grad=False):
        return torch.tensor(p, device=device, dtype=torch.float64, requires_grad=grad)

    with torch.no_grad():
        assert torch.equal(forward(va, kind, to_tensor(initial))[0], va[0])
    best_key, best, rows = None, None, []
    for penalty in PENALTIES:
        iteration, recorded = 0, set()

        def record(p):
            nonlocal best_key, best
            if iteration in recorded:
                return
            recorded.add(iteration)
            with torch.no_grad():
                pred = forward(va, kind, to_tensor(p))[0].cpu().numpy()
            a = auc(valid["label"], pred)
            row = {
                "step": iteration,
                "penalty": penalty,
                "val_auc": a,
                "val_accuracy": float(np.mean((pred > 0) == valid["label"])),
            }
            rows.append(row)
            key = (a, penalty, -iteration)
            if best_key is None or key > best_key:
                best_key, best = key, (p.copy(), row.copy())

        def objective(p):
            w = to_tensor(p, True)
            prediction, _ = forward(tr, kind, w)
            loss = F.binary_cross_entropy_with_logits(
                prediction, tr[-1]
            ) + 0.5 * penalty * w.dot(w)
            loss.backward()
            return float(loss.detach()), w.grad.detach().cpu().numpy()

        def callback(p):
            nonlocal iteration
            iteration += 1
            if iteration in STEPS:
                record(p)

        record(initial)
        result = minimize(
            objective,
            initial.copy(),
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": STEPS[-1],
                "maxcor": 12,
                "maxls": 40,
                "ftol": 1e-12,
                "gtol": 1e-7,
            },
        )
        if not np.isfinite(result.fun) or result.status not in (0, 1):
            raise ValueError((kind, penalty, result.message))
        assert result.nit == iteration
        record(result.x)
    return best[0], stats, best[1], rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(folder, device):
    tr = load_split(folder / "train.npz")
    va = load_split(folder / "val.npz")
    assert len(set(tr["source"]) & set(va["source"])) == 0
    out = folder / "models"
    out.mkdir(exist_ok=True)
    lock = {
        "seed": SEED,
        "threshold": 0,
        "base_weights": "frozen",
        "bound": BOUND,
        "penalties": list(PENALTIES),
        "checkpoint_steps": list(STEPS),
        "variants": {},
    }
    for kind in VARIANTS:
        w, stats, selected, rows = build_and_fit(tr, va, kind, device)
        path = out / (kind + ".npz")
        np.savez_compressed(path, theta=w, **stats)
        lock["variants"][kind] = {
            **selected,
            "coefficients": len(w),
            "sha256": digest(path),
            "validation_grid": rows,
        }
        print(kind, selected, flush=True)
    (out / "selection.json").write_text(json.dumps(lock, indent=2) + "\n")


def apply_gate(data, out, name, lock, device):
    path = out / (name + ".npz")
    assert digest(path) == lock["variants"][name]["sha256"]
    with np.load(path) as ck:
        w = torch.as_tensor(ck["theta"], dtype=torch.float64, device=device)
        stats = {k: ck[k] for k in ck.files if k != "theta"}
    kind = SECOND_STAGE.get(name, name)
    batch, _ = build_inputs(data, kind, stats, device)
    with torch.no_grad():
        prediction, coeff = forward(batch, kind, w)
    return prediction.cpu().numpy(), coeff


def fit_brain(folder, device):
    tr = load_split(folder / "train.npz")
    va = load_split(folder / "val.npz")
    out = folder / "models"
    lock = json.loads((out / "selection.json").read_text())
    anchor_hash = lock["variants"]["se"]["sha256"]
    for data in (tr, va):
        data["base"] = apply_gate(data, out, "se", lock, device)[0]
    for name, kind in SECOND_STAGE.items():
        w, stats, selected, rows = build_and_fit(tr, va, kind, device)
        path = out / (name + ".npz")
        np.savez_compressed(path, theta=w, **stats)
        lock["variants"][name] = {
            **selected,
            "coefficients": len(w) + lock["variants"]["se"]["coefficients"],
            "sha256": digest(path),
            "frozen_anchor_sha256": anchor_hash,
            "validation_grid": rows,
        }
        print(name, selected, flush=True)
    (out / "selection_brain.json").write_text(json.dumps(lock, indent=2) + "\n")

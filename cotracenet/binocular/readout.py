"""Full-feedback, single-step decision layer above a frozen two-stage binocular detector."""

import hashlib
import json

import numpy as np
from .grouping import source_groups
from .prepare import save
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from scipy.stats import rankdata


HARM = 1.5
BOUND = 2.0
REVIEW_RIDGE = (0.03, 0.12, 0.48)
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01)


def auc(y, s):
    y = np.asarray(y, dtype=bool)
    n = y.sum()
    return (rankdata(s)[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n))


def read(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    n = len(data["source"])
    for k in ("brain", "se", "base", "label"):
        data[k] = data[k].reshape(-1)
        assert len(data[k]) == n and np.isfinite(data[k]).all()
    for k in ("T", "C", "Ht", "Hc", "D"):
        assert data[k].shape == (n, 16) and np.isfinite(data[k]).all()
    assert set(data["label"]) == {0, 1}
    assert sorted(data["source"][data["label"] == 0]) == sorted(
        data["source"][data["label"] == 1]
    )
    return data


def subset(data, mask):
    return {k: v[mask] for k, v in data.items() if v.ndim >= 1 and len(v) == len(mask)}


def state(d):
    # Only statistics of this image, with all preceding classifiers' parameters frozen.
    t, c, ht, hc, dist = [d[k] for k in ("T", "C", "Ht", "Hc", "D")]
    s, se, raw = (d[k] for k in ("brain", "se", "base"))
    return np.column_stack(
        [
            s,
            abs(s),
            se - raw,
            s - se,
            t,
            c,
            ht,
            hc,
            dist,
            np.sign(t) * np.sign(c),
            ht - hc,
            t.sum(1),
            c.sum(1),
            t.std(1),
            c.std(1),
            dist.mean(1),
            dist.std(1),
            abs(t - c).mean(1),
        ]
    )


def scale_design(x, mu=None, sd=None):
    if mu is None:
        mu = x.mean(0)
        sd = x.std(0).clip(1e-6)
    z = np.clip((x - mu) / sd, -5.0, 5.0)
    return np.column_stack((z, np.ones(len(z)))), mu, sd


def solve(fun, start):
    result = minimize(
        fun,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 180, "ftol": 1e-12, "gtol": 1e-7, "maxls": 35},
    )
    if result.status not in (0, 1) or not np.isfinite(result.fun):
        raise RuntimeError(str(result.message))
    return result.x


def fit_reviewer(d, ridge):
    x, mu, sd = scale_design(state(d))
    base, y = d["brain"], d["label"]

    def fun(w):
        g = np.tanh(x @ w / BOUND)
        logits = base + BOUND * g
        loss = np.mean(np.logaddexp(0, logits) - y * logits)
        grad = x.T @ ((expit(logits) - y) * (1 - g * g)) / len(y)
        grad[:-1] += ridge * w[:-1]
        return loss + 0.5 * ridge * np.dot(w[:-1], w[:-1]), grad

    return {"rmu": mu, "rsd": sd, "rw": solve(fun, np.zeros(x.shape[1]))}


def reviewer(d, model):
    x, _, _ = scale_design(state(d), model["rmu"], model["rsd"])
    return d["brain"] + BOUND * np.tanh(x @ model["rw"] / BOUND)


def gate_features(d, answer):
    s, se, base = d["brain"], d["se"], d["base"]
    t, c, ht, hc, dist = [d[k] for k in ("T", "C", "Ht", "Hc", "D")]
    return np.column_stack(
        [
            s,
            abs(s),
            abs(answer),
            abs(answer) - abs(s),
            np.sign(answer),
            abs(se - base),
            abs(s - se),
            (np.sign(t) == np.sign(c)).mean(1),
            abs(t - c).mean(1),
            dist.mean(1),
            dist.std(1),
            ht.mean(1) - hc.mean(1),
        ]
    )


def crossfit_reviewer(d, ridge):
    # Within each outer fold the SAME backbone/selector has not seen either half.
    # Fit the review candidate on one half of that held-out fold; predict the other.
    fold = d["fold"].reshape(-1)
    key = d["source_sha256"] if "source_sha256" in d else d["source"]
    assert len(np.unique(fold)) == 3
    halves = np.array(
        [
            int(hashlib.sha256(("review73|" + str(v)).encode()).hexdigest()[:8], 16) % 2
            for v in key
        ]
    )
    covers = d.get("cover_sha256", key)
    for f in np.unique(fold):
        indices = np.flatnonzero(fold == f)
        groups = source_groups(key, covers, indices)
        if len(groups) < 2:
            raise ValueError("A holdout fold needs at least two distinct source groups")
        ordered = sorted(groups, key=lambda group: min(key[i] for i in group))
        for group in ordered:
            halves[group] = halves[min(group, key=lambda i: key[i])]
        if len(np.unique(halves[indices])) < 2:
            for number, group in enumerate(ordered):
                halves[group] = number % 2
    pred = np.full(len(fold), np.nan)
    details = []
    for f in np.unique(fold):
        for h in (0, 1):
            train = (fold == f) & (halves != h)
            held = (fold == f) & (halves == h)
            assert not set(key[train]) & set(key[held])
            assert not set(covers[train]) & set(covers[held])
            model = fit_reviewer(subset(d, train), ridge)
            pred[held] = reviewer(subset(d, held), model)
            details.append(
                {
                    "outer_fold": int(f),
                    "held_half": h,
                    "fit_images": int(train.sum()),
                    "held_images": int(held.sum()),
                }
            )
    assert np.isfinite(pred).all()
    return pred, details


def fit_gates(d, candidate):
    s, y = d["brain"], d["label"]
    disagree = (s > 0) != (candidate > 0)
    target = ((candidate[disagree] > 0) == y[disagree]).astype(float)
    assert len(target) >= 30 and 0 < sum(target) < len(target)
    x, mu, sd = scale_design(gate_features(d, candidate)[disagree])

    def sup(w):
        h = x @ w
        grad = x.T @ (expit(h) - target) / len(target)
        grad[:-1] += 0.08 * w[:-1]
        return np.mean(np.logaddexp(0, h) - target * h) + 0.04 * (w[:-1] @ w[:-1]), grad

    ws = solve(sup, np.zeros(x.shape[1]))
    reference = x @ ws
    advantage = np.where(target == 1.0, 1.0, -HARM)

    # Exact expectation of the two observable action rewards, plus an anchor
    # to the supervised policy. No simulated trajectories or online exploration.
    def actor(w):
        h = x @ w
        p = expit(h)
        kl = p * (h - reference) - np.logaddexp(0, h) + np.logaddexp(0, reference)
        reg = w - ws
        loss = (-p * advantage + 0.35 * kl).mean() + 0.06 * np.dot(reg, reg)
        grad = (
            x.T @ ((0.35 * (h - reference) - advantage) * p * (1 - p)) / len(target)
            + 0.12 * reg
        )
        return loss, grad

    wp = solve(actor, ws)
    return {"gmu": mu, "gsd": sd, "ws": ws, "wp": wp}, int(len(target))


def probabilities(d, review, model):
    z, _, _ = scale_design(gate_features(d, review), model["gmu"], model["gsd"])
    return expit(z @ model["ws"]), expit(z @ model["wp"])


def decide(s, answer, prob, cutoff):
    action = ((s > 0) != (answer > 0)) & (prob >= cutoff)
    return np.where(action, answer, s), action


def choose(d, answer, prob, cost):
    s, y = d["brain"], d["label"]
    baseline = (s > 0) == y
    order = []
    for threshold in THRESHOLDS:
        pred, take = decide(s, answer, prob, threshold)
        hit = (pred > 0) == y
        rescue = np.count_nonzero(hit & ~baseline)
        harm = np.count_nonzero(~hit & baseline)
        utility = float(rescue - cost * harm)
        order.append((utility, -int(take.sum()), threshold))
    best = max(order)
    return {"threshold": best[2], "reward": best[0], "changed": -best[1]}


def fit(folder):
    tr, va = read(folder / "oof.npz"), read(folder / "val.npz")
    assert not set(tr["source"]) & set(va["source"])
    ranked = []
    for ridge in REVIEW_RIDGE:
        model = fit_reviewer(tr, ridge)
        valreview = reviewer(va, model)
        ranked.append((auc(va["label"], valreview), ridge, model))
    _, ridge, model = max(ranked, key=lambda row: row[:2])
    trreview, halves = crossfit_reviewer(tr, ridge)
    disagree = (tr["brain"] > 0) != (trreview > 0)
    answers = (trreview[disagree] > 0) == tr["label"][disagree]
    n = len(answers)
    can_fit = n >= 30 and len(np.unique(answers)) == 2
    if can_fit:
        gate, n = fit_gates(tr, trreview)
    else:
        # With too few opposing predictions there is no supported policy update.
        width = gate_features(tr, trreview).shape[1]
        gate = dict(
            gmu=np.zeros(width),
            gsd=np.ones(width),
            ws=np.zeros(width + 1),
            wp=np.zeros(width + 1),
        )
    model.update(gate)
    valreview = reviewer(va, model)
    qs, qp = probabilities(va, valreview, model)
    selected = {
        "supervised": choose(va, valreview, qs, 1.0),
        "cost_supervised": choose(va, valreview, qs, HARM),
        "policy": choose(va, valreview, qp, HARM),
    }
    if not can_fit:
        for mode in selected:
            selected[mode] = {"threshold": 1.01, "reward": 0.0, "changed": 0}
    weightpath = folder / "readout-model.npz"
    np.savez_compressed(weightpath, **model)
    lock = {
        "review_ridge": ridge,
        "review_auc_grid": [[float(a), r] for a, r, _ in ranked],
        "cost_wrong_flip": HARM,
        "benefit_rescue": 1.0,
        "keep_reward": 0.0,
        "review_bound": BOUND,
        "training_disagreements": n,
        "inner_splits": halves,
        "selection": selected,
        "validation_disagreements": int(np.sum((va["brain"] > 0) != (valreview > 0))),
        "added_parameters": {"reviewer": len(model["rw"]), "actor": len(model["wp"])},
        "actor_fitted": can_fit,
        "model_sha256": hashlib.sha256(weightpath.read_bytes()).hexdigest(),
    }
    (folder / "readout-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in lock.items() if k not in ("inner_splits", "model_sha256")}
        ),
        flush=True,
    )


def score(folder, train, targets=("WOW", "HILL")):
    p = folder / "readout-model.npz"
    lock = json.loads((folder / "readout-lock.json").read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == lock["model_sha256"]
    with np.load(p) as ck:
        model = {k: ck[k] for k in ck.files}
    for target in targets:
        d = read(folder / ("test_" + target + ".npz"))
        review = reviewer(d, model)
        qs, qp = probabilities(d, review, model)
        b = d["brain"]
        out = {
            "source": d["source"],
            "label": d["label"],
            "brain": b,
            "always": review,
            "lite": d["base"],
        }
        for name, prob in (("supervised", qs), ("cost_supervised", qs), ("policy", qp)):
            out[name], take = decide(
                b, review, prob, lock["selection"][name]["threshold"]
            )
            out["@action:" + name] = take.astype(np.int8)
        for key in (
            "brain",
            "lite",
            "always",
            "supervised",
            "cost_supervised",
            "policy",
        ):
            out["@fit:" + key] = np.asarray(
                train + "-" + key + "-" + lock["model_sha256"]
            )
        save(folder / (train + "-to-" + target + ".npz"), **out)

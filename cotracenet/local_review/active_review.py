"""Acquire one local residual descriptor, optionally, after the frozen binocular policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


HARM = 1.5
QUERY_COST = 0.02
MAX_RATE = 0.15
BOUND = 1.5
RIDGES = (0.03, 0.15, 0.75)
QUANTILES = (0.05, 0.10, 0.20, 0.35, 0.55)
ARMS = ("selected", "fixed")


def design(s, f):
    return np.column_stack((f, s, np.abs(s)))


def standardize(x, mu, sd):
    return np.clip((x - mu) / sd, -5.0, 5.0)


def fit_head(scores, features, labels, ridge):
    raw = design(scores, features)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0).clip(1e-7)
    x = standardize(raw, mean, std)

    def objective(w):
        h = np.tanh((x @ w[:-1] + w[-1]) / BOUND)
        z = scores + BOUND * h
        ds = (expit(z) - labels) * (1 - h * h)
        grad = np.r_[x.T @ ds / len(ds) + ridge * w[:-1], ds.mean()]
        return np.mean(np.logaddexp(0, z) - labels * z) + 0.5 * ridge * (w[:-1] @ w[:-1]), grad

    res = minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 160, "ftol": 1e-12, "gtol": 1e-7, "maxls": 35},
    )
    if res.status not in (0, 1) or not np.isfinite(res.fun):
        raise ValueError(str(res.message))
    return {"w": res.x, "mu": mean, "sd": std}


def refine_many(s, f, head):
    x = standardize(design(s, f), head["mu"], head["sd"])
    return s + BOUND * np.tanh((x @ head["w"][:-1] + head["w"][-1]) / BOUND)


def bin_id(s, edges):
    return np.searchsorted(edges, np.abs(s), side="right") + (s > 0).astype(int) * 6


def actual(s, z, y, query):
    old = (s > 0) == y
    new = (z > 0) == y
    saved = int(np.sum((new & ~old) & query))
    wrong = int(np.sum((~new & old) & query))
    return {
        "queried": int(query.sum()),
        "rescued": saved,
        "harmed": wrong,
        "net_correct": saved - wrong,
        "reward": float(saved - HARM * wrong - QUERY_COST * query.sum()),
    }


def load_split(root, domain, which, target=None):
    anchor_name = which if which != "test" else "test_" + target
    pdomain = domain if which != "test" else target
    with np.load(root / domain / ("anchor_" + anchor_name + ".npz"), allow_pickle=False) as f:
        keys = ["source", "label", "anchor"]
        if which == "train":
            keys += ["fold", "policy_holdfold", "source_sha256"]
        d = {k: f[k].reshape(-1) for k in keys}
    with np.load(root / pdomain / ("patch_" + which + ".npz"), allow_pickle=False) as p:
        k1 = list(zip(d["source"].astype(str), d["label"].astype(int)))
        k2 = list(zip(p["source"].astype(str), p["label"].astype(int)))
        assert len(set(k1)) == len(set(k2)) == len(k1) == len(k2) and set(k1) == set(k2)
        indexes = None
        if k1 != k2:
            ix = {k: j for j, k in enumerate(k2)}
            indexes = [ix[k] for k in k1]
        for k in (*ARMS, "selected_tile"):
            v = p[k]
            d[k] = v if indexes is None else v[indexes]
    assert len(d["label"]) == {"train": 12000, "val": 2000, "test": 6000}[which]
    for k in ARMS:
        assert d[k].shape == (len(k1), 156) and np.isfinite(d[k]).all()
    assert np.isfinite(d["anchor"]).all()
    return d


class ReviewModel:
    def __init__(self, lock_folder, arm="selected"):
        folder = Path(lock_folder)
        self.cfg = json.loads((folder / "local-lock.json").read_text())
        name = folder / ("head-" + arm + ".npz")
        if hashlib.sha256(name.read_bytes()).hexdigest() != self.cfg["heads"][arm]:
            raise ValueError("Locked patch head has changed")
        with np.load(name, allow_pickle=False) as f:
            self.head = {k: f[k] for k in f.files}
        self.edges = np.array(self.cfg["edges"])
        self.values = np.array(self.cfg["values"])
        self.threshold = self.cfg["threshold"]
        self.uncertainty_threshold = self.cfg["uncertainty_threshold"]

    def should_query(self, score):
        group = int(np.searchsorted(self.edges, abs(score), side="right")) + 6 * int(score > 0)
        return self.values[group] >= self.threshold

    def uncertainty_query(self, score):
        return abs(score) <= self.uncertainty_threshold

    def refine(self, score, features):
        x = np.concatenate((np.asarray(features, dtype=float), [score, abs(score)]))
        z = standardize(x, self.head["mu"], self.head["sd"])
        return float(
            score + BOUND * np.tanh((z @ self.head["w"][:-1] + self.head["w"][-1]) / BOUND)
        )


def fit(root, domain):
    tr = load_split(root, domain, "train")
    va = load_split(root, domain, "val")
    sx, sy = tr["anchor"], va["anchor"]
    if set(tr["source"]) & set(va["source"]):
        raise ValueError("Train/val overlap")
    history = []
    for ridge in RIDGES:
        heads = {a: fit_head(sx, tr[a], tr["label"], ridge) for a in ARMS}
        vpred = {a: refine_many(sy, va[a], heads[a]) for a in ARMS}
        loss = np.mean([np.mean(np.logaddexp(0, vpred[a]) - va["label"] * vpred[a]) for a in ARMS])
        history.append((float(loss), -ridge, heads, vpred))
    _, neg, heads, vpred = min(history, key=lambda h: h[:2])
    ridge = -neg
    halves = np.array(
        [
            int(hashlib.sha256(("active73|" + str(k)).encode()).hexdigest()[:8], 16) % 2
            for k in tr["source_sha256"]
        ]
    )
    scores = np.full(len(sx), np.nan)
    joint, group_id = np.unique(
        np.column_stack((tr["fold"], tr["policy_holdfold"])), axis=0, return_inverse=True
    )
    for group in range(len(joint)):
        for h in (0, 1):
            a = (group_id == group) & (halves == h)
            b = (group_id == group) & (halves != h)
            assert not set(tr["source_sha256"][a]) & set(tr["source_sha256"][b])
            w = fit_head(sx[a], tr["selected"][a], tr["label"][a], ridge)
            scores[b] = refine_many(sx[b], tr["selected"][b], w)
    assert np.isfinite(scores).all()
    edges = np.quantile(abs(sx), QUANTILES)
    bid = bin_id(sx, edges)
    benefit = np.zeros(12)
    for i in range(12):
        m = bid == i
        r = actual(sx, scores, tr["label"], m)
        benefit[i] = (r["rescued"] - HARM * r["harmed"]) / (m.sum() + 50) - QUERY_COST
    gv = benefit[bin_id(sy, edges)]
    valrows = []
    for threshold in sorted(set(benefit.tolist() + [2.0])):
        q = gv >= threshold
        if q.mean() > MAX_RATE:
            continue
        valrows.append((actual(sy, vpred["selected"], va["label"], q), threshold))
    best, cut = max(valrows, key=lambda r: (r[0]["reward"], -r[0]["queried"], r[1]))
    unc = float(np.sort(np.abs(sy))[int(len(sy) * MAX_RATE) - 1])
    folder = root / domain
    sums = {}
    for name, head in heads.items():
        path = folder / ("head-" + name + ".npz")
        np.savez_compressed(path, **head)
        sums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = {
        "domain": domain,
        "threshold": cut,
        "uncertainty_threshold": unc,
        "edges": edges.tolist(),
        "values": benefit.tolist(),
        "heads": sums,
        "learned_head_coefficients": len(heads["selected"]["w"]),
        "validation_selected": best,
        "budget": MAX_RATE,
        "query_cost": QUERY_COST,
        "wrong_flip_cost": HARM,
        "validation_candidate_thresholds": valrows,
        "common_head_ridge": ridge,
        "validation_loss_by_ridge": [{"ridge": -h[1], "loss": h[0]} for h in history],
    }
    (folder / "local-lock.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(domain, "review ridge", ridge, "receipt", best, "uncertainty", unc, flush=True)


def score(root, domain):
    models = {a: ReviewModel(root / domain, a) for a in ARMS}
    for target in ("WOW", "HILL"):
        d = load_split(root, domain, "test", target)
        s = d["anchor"]
        m = models["selected"]
        gate = m.values[bin_id(s, m.edges)] >= m.threshold
        uncertainty = abs(s) <= m.uncertainty_threshold
        out = {
            "source": d["source"],
            "label": d["label"],
            "anchor": s,
            "gate": gate,
            "uncertainty": uncertainty,
            "selected_tile": d["selected_tile"],
        }
        for arm, model in models.items():
            post = refine_many(s, d[arm], model.head)
            out["always_" + arm] = post
            out["gate_" + arm] = np.where(gate, post, s)
            out["uncertainty_" + arm] = np.where(uncertainty, post, s)
        np.savez_compressed(root / domain / (domain + "-to-" + target + "-local.npz"), **out)


def summarize(root):
    from cotracenet.binocular.evaluation import summarize as metrics, contrast

    rows = []
    paired = []
    covers = {}
    checks = []
    for domain in ("WOW", "HILL"):
        for target in ("WOW", "HILL"):
            with np.load(
                root / domain / (domain + "-to-" + target + "-local.npz"), allow_pickle=False
            ) as f:
                order = np.lexsort((f["label"], f["source"].astype(str)))
                ids = f["source"][order].reshape(-1, 2)
                y = f["label"][order]
                assert np.array_equal(ids[:, 0], ids[:, 1]) and np.all(y.reshape(-1, 2) == [0, 1])
                s = f["anchor"][order]
                pairs = {"anchor": metrics(s.reshape(-1, 2))[1]}
                for key in (
                    "anchor",
                    "gate_fixed",
                    "gate_selected",
                    "uncertainty_fixed",
                    "uncertainty_selected",
                    "always_fixed",
                    "always_selected",
                ):
                    z = f[key][order]
                    if key == "anchor":
                        q = np.zeros(len(s), dtype=bool)
                    elif key.startswith("always"):
                        q = np.ones(len(s), dtype=bool)
                    else:
                        q = f[key.split("_")[0]][order]
                    r = actual(s, z, y, q)
                    stat, pairs[key] = metrics(z.reshape(-1, 2))
                    rows.append(
                        {
                            "train": domain,
                            "test": target,
                            "method": key,
                            "query_rate": q.mean(),
                            **r,
                            **stat,
                        }
                    )
                    cover = z[::2]
                    if (domain, key) in covers:
                        old_ids, old = covers[domain, key]
                        assert np.array_equal(old_ids, ids[:, 0])
                        diff = np.max(np.abs(old - cover))
                        assert diff < 1e-7
                        checks.append(
                            {
                                "train": domain,
                                "method": key,
                                "max_shared_cover_difference": float(diff),
                            }
                        )
                    else:
                        covers[domain, key] = (ids[:, 0], cover)
                for cand, ref in (
                    ("gate_selected", "anchor"),
                    ("gate_selected", "gate_fixed"),
                    ("uncertainty_selected", "anchor"),
                    ("uncertainty_selected", "uncertainty_fixed"),
                    ("always_selected", "anchor"),
                ):
                    paired.append(
                        {
                            "train": domain,
                            "test": target,
                            "candidate": cand,
                            "reference": ref,
                            **contrast(pairs[cand], pairs[ref], 10000, 7319),
                        }
                    )
    with (root / "performance.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "performance.json").write_text(
        json.dumps({"rows": rows, "paired": paired, "cover_checks": checks}, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "score", "summary"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--domain", choices=["WOW", "HILL"])
    a = p.parse_args()
    if a.stage == "summary":
        summarize(a.root)
    elif a.stage == "fit":
        fit(a.root, a.domain)
    else:
        score(a.root, a.domain)

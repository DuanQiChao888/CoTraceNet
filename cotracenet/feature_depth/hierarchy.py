"""Hierarchical histogram encoder. Fit/validation and locked test are separate entry points."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..binocular.prepare import WIDTH
from .cache import digest, read_matrix


CHECKPOINTS = (2, 4, 8, 12, 18, 24)
ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class Hierarchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_words = nn.Linear(39, 16)
        self.regional_words = nn.Linear(169, 16)
        self.texture_zones = nn.Linear(3 * 16, 16)
        self.cost_zones = nn.Linear(5 * 16, 16)
        self.routes = nn.ModuleList(
            [nn.Linear(families * 16, 22) for families in (24, 16, 30)]
        )
        self.fusion = nn.Linear(3 * 22, 16)
        self.decision = nn.Linear(16, 1)
        self.word_skip = nn.Linear(WIDTH, 1, bias=False)
        nn.init.zeros_(self.word_skip.weight)

    def forward(self, x):
        n = len(x)
        g = F.gelu(self.global_words(x[:, :936].reshape(n, 24, 39)))
        # One shared word encoder; retain all 48+150 conditional histograms separately.
        regional = F.gelu(self.regional_words(x[:, 936:].reshape(n, 198, 169)))
        t = F.gelu(self.texture_zones(regional[:, :48].reshape(n, 16, 3 * 16)))
        c = F.gelu(self.cost_zones(regional[:, 48:].reshape(n, 30, 5 * 16)))
        routes = [
            F.gelu(encode(data.reshape(n, -1)))
            for data, encode in zip((g, t, c), self.routes)
        ]
        merged = F.gelu(self.fusion(torch.cat(routes, dim=1)))
        return (self.decision(merged) + self.word_skip(x)).flatten()

    def macs(self):
        def linear(layer):
            return layer.in_features * layer.out_features

        return (
            24 * linear(self.global_words)
            + 198 * linear(self.regional_words)
            + 16 * linear(self.texture_zones)
            + 30 * linear(self.cost_zones)
            + sum(map(linear, self.routes))
            + linear(self.fusion)
            + linear(self.decision)
            + linear(self.word_skip)
        )


def matrix(root, domain, split, moments=None):
    raw, ids, y = read_matrix(root, domain, split)
    raw = raw.reshape(-1, WIDTH)
    if moments is None:
        assert split == "train"
        total, total2 = np.zeros(WIDTH), np.zeros(WIDTH)
        for start in range(0, len(raw), 128):
            z = raw[start : start + 128]
            total += z.sum(axis=0)
            total2 += np.einsum("ij,ij->j", z, z)
        mean = total / len(raw)
        scale = np.sqrt(np.maximum(total2 / len(raw) - mean**2, 1.0e-8))
        moments = dict(
            mean=mean.astype(np.float32), inv_scale=(1 / scale).astype(np.float32)
        )
    x = np.empty(raw.shape, np.float32)
    for start in range(0, len(raw), 128):
        block = x[start : start + 128]
        block[:] = raw[start : start + 128]
        block -= moments["mean"]
        block *= moments["inv_scale"]
        np.clip(block, -8.0, 8.0, out=block)
        assert np.isfinite(block).all()
    return torch.from_numpy(x), ids, y, moments


def reference(policy, train, split, target, ids, y):
    source = policy / train
    if split == "test":
        with np.load(source / (train + "-to-" + target + ".npz")) as data:
            assert np.array_equal(data["source"], ids) and np.array_equal(
                data["label"], y
            )
            return data["policy"].copy()
    from ..binocular import readout

    lock = json.loads((source / "readout-lock.json").read_text())
    p = source / "readout-model.npz"
    assert digest(p) == lock["model_sha256"]
    with np.load(p) as f:
        model = dict(f)
    d = readout.read(source / "val.npz")
    assert np.array_equal(d["source"], ids) and np.array_equal(d["label"], y)
    review = readout.reviewer(d, model)
    _, prob = readout.probabilities(d, review, model)
    s, _ = readout.decide(
        d["brain"], review, prob, lock["selection"]["policy"]["threshold"]
    )
    return s


@torch.inference_mode()
def score(model, x):
    model.eval()
    return torch.cat([model(part) for part in x.split(384)]).numpy().astype(np.float64)


def measures(score, y):
    return int(np.sum((score > 0) == y)), float(
        np.mean(np.logaddexp(0.0, score) - y * score)
    )


def fit(a, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "unified-models.json").exists():
        raise FileExistsError("Each fit needs a new directory")
    torch.manual_seed(a.seed)
    net = Hierarchy()
    params = sum(p.numel() for p in net.parameters())
    hp = dict(
        seed=a.seed,
        checkpoints=CHECKPOINTS,
        alphas=ALPHAS,
        n_pairs_per_batch=96,
        lr_hierarchy=0.001,
        lr_word_skip=0.00015,
        weight_decay=0.02,
        pair_loss=0.25,
        extra_params=params,
        dense_macs_per_image=net.macs(),
        select="maximize source val correct, then minimize BCE, then prefer smaller alpha, then earlier epoch",
    )
    assert params + 16828 <= 84140
    dump(out / "protocol.json", hp)
    x, tr_ids, _, moments = matrix(a.root, a.train, "train")
    v, ids, y, _ = matrix(a.root, a.train, "val", moments)
    assert set(ids).isdisjoint(tr_ids)
    base = reference(a.policy, a.train, "val", a.train, ids, y)
    np.savez(out / "train-normalizer.npz", **moments)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for name, p in net.named_parameters()
                    if not name.startswith("word_skip")
                ],
                "lr": hp["lr_hierarchy"],
            },
            {"params": net.word_skip.parameters(), "lr": hp["lr_word_skip"]},
        ],
        weight_decay=hp["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(CHECKPOINTS), eta_min=0.00003
    )
    order = torch.Generator().manual_seed(a.seed)
    best, rows = None, []
    train_pairs = x.reshape(-1, 2, WIDTH)
    started = time.monotonic()
    for epoch in range(1, max(CHECKPOINTS) + 1):
        net.train()
        for idx in torch.randperm(len(train_pairs), generator=order).split(
            hp["n_pairs_per_batch"]
        ):
            pair_scores = net(train_pairs[idx].reshape(-1, WIDTH)).reshape(-1, 2)
            cover, stego = pair_scores.unbind(dim=1)
            loss = 0.5 * (F.softplus(cover).mean() + F.softplus(-stego).mean())
            loss += hp["pair_loss"] * F.softplus(cover - stego).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
        sched.step()
        print(f"epoch {epoch} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        if epoch not in CHECKPOINTS:
            continue
        raw = score(net, v)
        for alpha in ALPHAS:
            scores = (1 - alpha) * base + alpha * raw
            n, bce = measures(scores, y)
            key = (n, -bce, -alpha, -epoch)
            rows.append(dict(epoch=epoch, alpha=alpha, correct=n, bce=bce))
            if best is None or key > best:
                best = key
                torch.save(
                    {
                        "weights": net.state_dict(),
                        "alpha": alpha,
                        "epoch": epoch,
                        "source": a.train,
                        "policy_lock_sha256": digest(
                            a.policy / a.train / "readout-lock.json"
                        ),
                        "normalizer_sha256": digest(out / "train-normalizer.npz"),
                    },
                    out / "weights.pt",
                )
                np.savez_compressed(
                    out / "unified-val.npz",
                    source=ids,
                    label=y,
                    baseline=base,
                    hierarchy=scores,
                )
                dump(
                    out / "chosen-validation.json",
                    dict(
                        correct=n,
                        epoch=epoch,
                        alpha=alpha,
                        standalone=measures(raw, y)[0],
                    ),
                )
        dump(out / "validation-grid.json", rows)
    chosen = json.loads((out / "chosen-validation.json").read_text())
    dump(
        out / "unified-models.json",
        {
            "hierarchy": dict(
                extra_params=params,
                sha256=digest(out / "weights.pt"),
                description="Shared histogram encoders with zone, family and route fusion plus word skip",
                **chosen,
            )
        },
    )


def check_policy(state, policy_root, source):
    # Initial research checkpoints predate this optional source-policy binding.
    if "policy_lock_sha256" in state:
        if (
            source != state["source"]
            or digest(policy_root / source / "readout-lock.json")
            != state["policy_lock_sha256"]
        ):
            raise ValueError(
                "The supplied policy differs from the one used to select this model"
            )


def test(a, out):
    lock = json.loads((out / "unified-models.json").read_text())["hierarchy"]
    assert digest(out / "weights.pt") == lock["sha256"]
    state = torch.load(out / "weights.pt", map_location="cpu", weights_only=True)
    check_policy(state, a.policy, a.train)
    assert digest(out / "train-normalizer.npz") == state["normalizer_sha256"]
    with np.load(out / "train-normalizer.npz") as f:
        moments = dict(f)
    net = Hierarchy()
    net.load_state_dict(state["weights"])
    alpha = state["alpha"]
    scores_meta = {}
    for target in ("WOW", "HILL"):
        x, ids, y, _ = matrix(a.root, target, "test", moments)
        base = reference(a.policy, a.train, "test", target, ids, y)
        raw = score(net, x)
        combined = (1 - alpha) * base + alpha * raw
        scores_meta[target] = dict(
            combined=measures(combined, y)[0],
            standalone=measures(raw, y)[0],
            baseline=measures(base, y)[0],
        )
        np.savez_compressed(
            out / ("unified-test-" + target + ".npz"),
            source=ids,
            label=y,
            baseline=base,
            hierarchy=combined,
        )
    dump(out / "standalone-test.json", scores_meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "test"])
    p.add_argument("--train", choices=["WOW", "HILL"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--threads", type=int, default=5)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    (fit if a.stage == "fit" else test)(a, a.out / a.train)

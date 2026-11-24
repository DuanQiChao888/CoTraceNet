"""Train the word-level affine skip alone on the same data, objective, epochs and blend grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import hierarchy as h


class LinearControl(nn.Linear):
    def __init__(self):
        super().__init__(h.WIDTH, 1, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return super().forward(x).flatten()


def fit(a, domain):
    domain.mkdir(parents=True, exist_ok=True)
    if (domain / "unified-models.json").exists():
        raise FileExistsError("Choose a new directory")
    torch.manual_seed(a.seed)
    model = LinearControl()
    h.dump(
        domain / "protocol.json",
        dict(
            seed=a.seed,
            epochs=h.CHECKPOINTS,
            alphas=h.ALPHAS,
            pair_bce_weight=0.25,
            lr=0.00015,
            weight_decay=0.02,
            batch_pairs=96,
        ),
    )
    x, train_ids, _, moments = h.matrix(a.root, a.train, "train")
    v, ids, y, _ = h.matrix(a.root, a.train, "val", moments)
    assert set(train_ids).isdisjoint(ids)
    base = h.reference(a.policy, a.train, "val", a.train, ids, y)
    np.savez(domain / "train-normalizer.npz", **moments)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00015, weight_decay=0.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(h.CHECKPOINTS), eta_min=0.00003
    )
    best, rows = None, []
    rng = torch.Generator().manual_seed(a.seed)
    pairs = x.reshape(-1, 2, h.WIDTH)
    for epoch in range(1, max(h.CHECKPOINTS) + 1):
        for idx in torch.randperm(len(pairs), generator=rng).split(96):
            cover, stego = (
                model(pairs[idx].reshape(-1, h.WIDTH)).reshape(-1, 2).unbind(dim=1)
            )
            loss = (
                0.5 * (F.softplus(cover).mean() + F.softplus(-stego).mean())
                + 0.25 * F.softplus(cover - stego).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        schedule.step()
        if epoch not in h.CHECKPOINTS:
            continue
        raw = h.score(model, v)
        for alpha in h.ALPHAS:
            signed = (1 - alpha) * base + alpha * raw
            correct, bce = h.measures(signed, y)
            key = (correct, -bce, -alpha, -epoch)
            rows.append(dict(alpha=alpha, epoch=epoch, correct=correct, bce=bce))
            if best is None or key > best:
                best = key
                chosen = dict(alpha=alpha, epoch=epoch)
                torch.save(
                    dict(
                        weights=model.state_dict(),
                        normalizer_sha256=h.digest(domain / "train-normalizer.npz"),
                        **chosen
                    ),
                    domain / "weights.pt",
                )
                np.savez_compressed(
                    domain / "unified-val.npz",
                    source=ids,
                    label=y,
                    baseline=base,
                    hierarchy_linear=signed,
                )
    h.dump(domain / "validation-grid.json", rows)
    h.dump(
        domain / "unified-models.json",
        dict(
            hierarchy_linear=dict(
                extra_params=h.WIDTH + 1,
                sha256=h.digest(domain / "weights.pt"),
                correct=best[0],
                description="Independently trained all-word affine control",
                **chosen
            )
        ),
    )


def test(a, domain):
    lock = json.loads((domain / "unified-models.json").read_text())["hierarchy_linear"]
    assert h.digest(domain / "weights.pt") == lock["sha256"]
    ck = torch.load(domain / "weights.pt", weights_only=True, map_location="cpu")
    assert h.digest(domain / "train-normalizer.npz") == ck["normalizer_sha256"]
    with np.load(domain / "train-normalizer.npz") as f:
        moments = dict(f)
    model = LinearControl()
    model.load_state_dict(ck["weights"])
    for target in ("WOW", "HILL"):
        x, ids, y, _ = h.matrix(a.root, target, "test", moments)
        raw = h.score(model, x)
        base = h.reference(a.policy, a.train, "test", target, ids, y)
        signed = (1 - ck["alpha"]) * base + ck["alpha"] * raw
        np.savez_compressed(
            domain / ("unified-test-" + target + ".npz"),
            source=ids,
            label=y,
            baseline=base,
            hierarchy_linear=signed,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "test"])
    p.add_argument("--train", choices=["WOW", "HILL"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--seed", type=int, default=73)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    (fit if a.stage == "fit" else test)(a, a.out / a.train)

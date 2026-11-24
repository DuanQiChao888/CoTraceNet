"""Train a compact, full-field spatial CNN on multiscale high-pass statistics."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hierarchy import reference


EPOCHS = (2, 4, 8, 12, 18, 24)
ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
PROTOCOL = {
    "epochs": EPOCHS,
    "branch_fractions": ALPHAS,
    "seed_default": 73,
    "batch_source_pairs": 128,
    "loss": "BCE + 0.20 paired logistic ranking",
    "learning_rate": 0.0015,
    "weight_decay": 0.001,
    "rule": "validation correct, validation log loss, smaller branch fraction, earlier epoch",
    "network": "144-channel 4x4 residual-count grid; stem 144->64; three separable 3x3/1x1 residual blocks; global mean/std -> 192 -> 64 -> 1",
}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return F.silu(x + self.layers(x))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(144, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU()
        )
        self.blocks = nn.Sequential(*[SpatialBlock() for _ in range(3)])
        self.head = nn.Sequential(
            nn.Linear(128, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.blocks(self.stem(x))
        var, mean = torch.var_mean(x, (2, 3), correction=0)
        return self.head(torch.cat([mean, (var + 1.0e-6).sqrt()], dim=1)).flatten()


def get_data(cache, domain, split):
    with np.load(cache / domain / (split + "-grids.npz"), allow_pickle=False) as a:
        source, label = a["source"], a["label"]
        x = torch.from_numpy(a["counts"].astype(np.float32)).div_(4096.0).sqrt_()
    assert np.array_equal(source[::2], source[1::2]) and np.array_equal(
        label.reshape(-1, 2), np.tile([0, 1], (len(x) // 2, 1))
    )
    return x, source, label


def get_baseline(args, source, target, split, ids, labels):
    return reference(args.policy, source, split, target, ids, labels)


def report(score, y):
    hit = (score > 0) == y
    return {
        "correct": int(hit.sum()),
        "count": int(y.size),
        "bce": float((np.logaddexp(0.0, score) - y * score).mean()),
    }


def unit(x, mu, sd):
    return ((x - mu) / sd).clamp(-6, 6).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def evaluate(net, x):
    net.eval()
    return torch.cat([net(b) for b in x.split(256)]).numpy().astype(np.float64)


def cost(net):
    # Dense MAC per complete image for convolution/linear operators.
    records = []

    def add(m, _, y):
        if isinstance(m, nn.Conv2d):
            records.append(
                y.numel() * m.in_channels // m.groups * math.prod(m.kernel_size)
            )
        else:
            records.append(y.numel() * m.in_features)

    hooks = [
        m.register_forward_hook(add)
        for m in net.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    evaluate(net, torch.zeros(1, 144, 4, 4))
    for h in hooks:
        h.remove()
    return {
        "trainable": sum(p.numel() for p in net.parameters()),
        "conv_linear_mac_per_image": sum(records),
        "conv_linear_flops_per_image": sum(records) * 2,
    }


def fit(args, out):
    torch.manual_seed(args.seed)
    x, train_ids, ytr = get_data(args.cache, args.source, "train")
    valid, ids, labels = get_data(args.cache, args.source, "val")
    assert len(x) == 12000 and len(valid) == 2000 and not set(train_ids) & set(ids)
    baseline = get_baseline(args, args.source, args.source, "val", ids, labels)
    sd, mu = torch.std_mean(x, (0, 2, 3), correction=0, keepdim=True)
    sd = sd.clamp_min(1e-5)
    x, valid = unit(x, mu, sd), unit(valid, mu, sd)
    y = torch.from_numpy(ytr.astype(np.float32))
    gen = torch.Generator().manual_seed(args.seed)
    net = Classifier().to(memory_format=torch.channels_last)
    stat = cost(net)
    assert stat["trainable"] + 16828 <= 84140
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=PROTOCOL["learning_rate"],
        weight_decay=PROTOCOL["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(EPOCHS), eta_min=0.00012
    )
    best = None
    trace = []
    start = time.monotonic()
    for epoch in range(1, max(EPOCHS) + 1):
        net.train()
        for pair in torch.randperm(len(x) // 2, generator=gen).split(
            PROTOCOL["batch_source_pairs"]
        ):
            index = (2 * pair[:, None] + torch.arange(2)).flatten()
            s = net(x[index].contiguous(memory_format=torch.channels_last))
            loss = (
                F.binary_cross_entropy_with_logits(s, y[index])
                + 0.20 * F.softplus(s[::2] - s[1::2]).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
        sched.step()
        if epoch in EPOCHS:
            branch = evaluate(net, valid)
            for alpha in ALPHAS:
                combined = (1 - alpha) * baseline + alpha * branch
                r = report(combined, labels)
                row = dict(epoch=epoch, alpha=alpha, **r)
                trace.append(row)
                key = (r["correct"], -r["bce"], -alpha, -epoch)
                if best is None or key > best:
                    best = key
                    weights = {
                        k: v.detach().clone() for k, v in net.state_dict().items()
                    }
                    locked = dict(
                        epoch=epoch,
                        alpha=alpha,
                        validation=r,
                        standalone=report(branch, labels),
                    )
                    payload = dict(
                        weights=weights,
                        mu=mu,
                        sd=sd,
                        epoch=epoch,
                        alpha=alpha,
                        seed=args.seed,
                    )
                    score_out = combined.copy()
            print(
                "epoch",
                epoch,
                "standalone",
                report(branch, labels),
                "best",
                locked,
                "seconds",
                round(time.monotonic() - start, 1),
                flush=True,
            )
    path = out / "spatial.pt"
    torch.save(payload, path)
    np.savez_compressed(
        out / "unified-val.npz",
        source=ids,
        label=labels,
        baseline=baseline,
        spatial=score_out,
    )
    dump(
        out / "unified-models.json",
        {
            "spatial": dict(
                extra_params=stat["trainable"],
                description=PROTOCOL["network"],
                sha256=sha(path),
                epoch=locked["epoch"],
                alpha=locked["alpha"],
            )
        },
    )
    dump(
        out / "spatial-fit.json",
        dict(
            seed=args.seed,
            weights_sha256=sha(path),
            metrics=stat,
            baseline=report(baseline, labels),
            selected=locked,
            valid_grid=trace,
            source_train_pairs=6000,
            val_pairs=1000,
        ),
    )


def test(args, out):
    entry = json.loads((out / "unified-models.json").read_text())["spatial"]
    assert sha(out / "spatial.pt") == entry["sha256"]
    model = torch.load(out / "spatial.pt", map_location="cpu", weights_only=True)
    net = Classifier().to(memory_format=torch.channels_last)
    net.load_state_dict(model["weights"])
    for target in ("WOW", "HILL"):
        x, ids, labels = get_data(args.cache, target, "test")
        assert len(x) == 6000
        base = get_baseline(args, args.source, target, "test", ids, labels)
        raw = evaluate(net, unit(x, model["mu"], model["sd"]))
        score = (1 - model["alpha"]) * base + model["alpha"] * raw
        np.savez_compressed(
            out / ("unified-test-" + target + ".npz"),
            source=ids,
            label=labels,
            baseline=base,
            spatial=score,
        )
        np.savez_compressed(
            out / ("standalone-test-" + target + ".npz"),
            source=ids,
            label=labels,
            standalone=raw,
        )
        dump(
            out / ("spatial-test-" + target + ".json"),
            dict(
                combined=report(score, labels),
                standalone=report(raw, labels),
                baseline=report(base, labels),
                weights_sha256=entry["sha256"],
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "test", "plan"))
    parser.add_argument("--source", choices=("WOW", "HILL"), required=True)
    for k in ("cache", "policy", "out"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out = args.out / args.source
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        dump(
            out / "spatial-protocol.json",
            dict(
                protocol=PROTOCOL,
                seed=args.seed,
                cost=cost(Classifier()),
                script_sha256=sha(Path(__file__)),
            ),
        )
    elif args.stage == "fit":
        assert (
            sha(Path(__file__))
            == json.loads((out / "spatial-protocol.json").read_text())["script_sha256"]
        )
        assert not (out / "unified-models.json").exists()
        fit(args, out)
    else:
        test(args, out)

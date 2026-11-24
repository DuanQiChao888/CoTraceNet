"""Three-branch classifiers of source-selected residual statistics, before vote pooling."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cache import read_matrix
from .hierarchy import reference


SLICES = ((0, 936), (936, 9048), (9048, 34398))
N_SELECTED = 512
EPOCHS = (1, 3, 6, 12, 20, 32, 48)
MIX = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("resmlp", "resmlp_shallow")
BASE_PARAMETERS = 16828


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )


def fit_preprocessing(raw):
    n, _, d = raw.shape
    diff = np.zeros(d)
    total = np.zeros(d)
    sq = np.zeros(d)
    for start in range(0, n, 32):
        pair = raw[start : start + 32]
        total += pair.sum(axis=(0, 1))
        diff += (pair[:, 1] - pair[:, 0]).sum(axis=0)
        sq += np.einsum("ijk,ijk->k", pair, pair)
    mean = total / (2 * n)
    std = np.maximum(sq / (2 * n) - mean**2, 0) ** 0.5
    score = abs(diff / n) / np.maximum(std, 1e-5)
    keep = []
    for a, b in SLICES:
        rank = np.argsort(-score[a:b], kind="stable")[:N_SELECTED]
        keep.extend((np.sort(rank) + a).tolist())
    ids = np.array(keep, dtype=np.int64)
    assert np.isfinite(score).all() and (std[ids] > 0).all()
    return {"indices": ids, "mu": mean[ids], "sd": std[ids].clip(1e-5)}


def encode(raw, pre):
    n = len(raw)
    out = np.empty((n, 2, 3 * N_SELECTED), dtype=np.float32)
    for start in range(0, n, 64):
        x = np.take(raw[start : start + 64], pre["indices"], axis=-1)
        out[start : start + 64] = ((x - pre["mu"]) / pre["sd"]).clip(-6.0, 6.0)
    assert np.isfinite(out).all()
    return torch.from_numpy(out)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(36, 16)
        self.expand = nn.Linear(16, 36)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        return x + self.expand(F.gelu(self.compress(x)))


class Detector(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.deep = variant == "resmlp"
        width = 36 if self.deep else 43
        branches = []
        for _ in SLICES:
            layers = [nn.Linear(N_SELECTED, width), nn.GELU()]
            if self.deep:
                layers += [Residual(), Residual()]
            branches.append(nn.Sequential(*layers))
        self.branches = nn.ModuleList(branches)
        self.fusion = (
            nn.Sequential(
                nn.Linear(108, 40), nn.GELU(), nn.Dropout(0.12), nn.Linear(40, 1)
            )
            if self.deep
            else nn.Linear(129, 1)
        )
        self.dropout = nn.Dropout(0.12)

    def forward(self, x):
        parts = x.split(N_SELECTED, -1)
        groups = [branch(part) for branch, part in zip(self.branches, parts)]
        return self.fusion(self.dropout(torch.cat(groups, dim=-1))).squeeze(-1)


def run_scores(net, x):
    net.eval()
    with torch.inference_mode():
        return (
            torch.cat([net(batch) for batch in x.split(128)])
            .reshape(-1)
            .numpy()
            .astype(np.float64)
        )


def mixed(base, model, fraction):
    return (1 - fraction) * base + fraction * model


def correct(s, y):
    return int(np.sum((s > 0) == y))


def fit_net(name, x, va, label, base, seed, pre, out):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = Detector(name)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.002)
    loss_label = torch.tensor([0.0, 1.0])
    best_rank, best_state = None, {}
    history = []
    for epoch in range(1, max(EPOCHS) + 1):
        model.train()
        opt.param_groups[0]["lr"] = (
            0.00012
            + 0.00088 * (1 + math.cos(math.pi * (epoch - 1) / (max(EPOCHS) - 1))) / 2
        )
        for ix in torch.randperm(len(x), generator=rng).split(128):
            logits = model(x[ix])
            bce = F.binary_cross_entropy_with_logits(
                logits, loss_label.expand_as(logits)
            )
            rank_loss = F.softplus(logits[:, 0] - logits[:, 1]).mean()
            opt.zero_grad(set_to_none=True)
            (bce + 0.2 * rank_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
        if epoch not in EPOCHS:
            continue
        pred = run_scores(model, va)
        for alpha in MIX:
            s = mixed(base, pred, alpha)
            n = correct(s, label)
            loss = float(np.mean(np.logaddexp(0.0, s) - label * s))
            history.append(
                {"epoch": epoch, "alpha": alpha, "correct": n, "logloss": loss}
            )
            rank = (n, -loss, -alpha, -epoch)
            if best_rank is None or rank > best_rank:
                best_rank, best_row = rank, history[-1]
                best_state = {
                    k: v.detach().cpu().numpy().copy()
                    for k, v in model.state_dict().items()
                }
                best_score, best_pure = s.copy(), correct(pred, label)
        print(name, "epoch", epoch, "best", best_row, flush=True)
    params = sum(p.numel() for p in model.parameters())
    macs = sum(
        m.in_features * m.out_features
        for m in model.modules()
        if isinstance(m, nn.Linear)
    )
    path = out / (name + ".npz")
    np.savez_compressed(
        path,
        **pre,
        **best_state,
        alpha=best_row["alpha"],
        epoch=best_row["epoch"],
        seed=seed
    )
    assert BASE_PARAMETERS + params <= 84140
    desc = (
        "three 512-to-36 branches, two 36-16-36 residual blocks per branch, 108-40-1 fusion"
        if name == "resmlp"
        else "three 512-to-43 single-layer branches and 129-1 fusion"
    )
    meta = dict(
        extra_params=params,
        description=desc,
        sha256=digest(path),
        extra_dense_macs_per_image=macs,
        extra_dense_flops_per_image=2 * macs,
    )
    return (
        meta,
        best_score,
        dict(
            selected=best_row,
            pure_correct_at_selected_epoch=best_pure,
            pure_peak_val=max(r["correct"] for r in history if r["alpha"] == 1),
            history=history,
        ),
    )


def fit(args, folder):
    tr, tid, _ = read_matrix(args.features, args.train, "train")
    va, vid, labels = read_matrix(args.features, args.train, "val")
    assert not set(tid) & set(vid)
    pre = fit_preprocessing(tr)
    x, v = encode(tr, pre), encode(va, pre)
    base = reference(args.archive, args.train, "val", args.train, vid, labels)
    scores = dict(source=vid, label=labels, baseline=base)
    summary, models = {}, {}
    for name in args.models:
        models[name], scores[name], summary[name] = fit_net(
            name, x, v, labels, base, args.seed, pre, folder
        )
    np.savez_compressed(folder / "unified-val.npz", **scores)
    write_json(
        folder / "fit-details.json",
        dict(variants=summary, seed=args.seed, baseline_correct=correct(base, labels)),
    )
    write_json(folder / "unified-models.json", models)


def test(args, folder):
    meta = json.loads((folder / "unified-models.json").read_text())
    target = args.target
    x, ids, labels = read_matrix(args.features, target, "test")
    base = reference(args.archive, args.train, "test", target, ids, labels)
    scores = dict(source=ids, label=labels, baseline=base)
    native = {}
    for name in meta:
        p = folder / (name + ".npz")
        assert digest(p) == meta[name]["sha256"]
        with np.load(p, allow_pickle=False) as state:
            ng = Detector(name)
            ng.load_state_dict({k: torch.from_numpy(state[k]) for k in ng.state_dict()})
            pure = run_scores(ng, encode(x, state))
            blend = mixed(base, pure, state["alpha"].item())
            native[name] = {
                "pure_correct": correct(pure, labels),
                "blended_correct": correct(blend, labels),
            }
            scores[name] = blend
    np.savez_compressed(folder / ("unified-test-" + target + ".npz"), **scores)
    write_json(folder / ("test-detail-" + target + ".json"), native)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("--train", choices=("WOW", "HILL"), required=True)
    parser.add_argument("--target", choices=("WOW", "HILL"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--models", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = args.out / args.train
    folder.mkdir(exist_ok=True, parents=True)
    if args.phase == "fit":
        if (folder / "unified-models.json").exists():
            raise FileExistsError(folder / "unified-models.json")
        write_json(
            folder / "protocol.json",
            dict(
                seed=args.seed,
                train=args.train,
                epochs=EPOCHS,
                mix=MIX,
                architecture=args.models,
                select="source validation correct, then logloss, smaller mixing weight, earlier epoch",
                selection_features="512 per branch, source-train paired mean shift divided by pooled single-image std",
                training="paired-source batches of 128 pairs, BCE plus 0.2 paired logistic ranking",
            ),
        )
        fit(args, folder)
    else:
        if not args.target:
            parser.error("test phase requires --target")
        test(args, folder)

"""Fit residual experts and score an image without requiring its cover or a label."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from . import fourth, policy, regions, triples
from .backbone import ShuffleBANet
from .data import load_image, read_splits, tuning_splits
from .experts import fisher, fit_action


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_features(path):
    im = load_image(path)
    three, ctx = triples.extract_features(im)
    four = fourth.features(im)
    ref = np.sqrt(np.maximum(np.concatenate((three, four)), 0)).astype(np.float64)
    channel = np.sqrt(regions.features(im).astype(np.float64))
    return im, three, ctx, ref, np.concatenate((ref, channel))


def load_backbone(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck["kind"] != "A" or ck["seed"] != 73:
        raise ValueError("Expected a seed-73 ShuffleBANet reference checkpoint")
    state = ck["model"]
    net = ShuffleBANet(state["encoder.0.srm_kernels"].detach().clone())
    net.load_state_dict(state)
    return net.eval().to(device, memory_format=torch.channels_last)


@torch.inference_mode()
def extract(paths, net, device, workers):
    paths = list(paths)
    if not paths or workers < 0:
        raise ValueError("Provide at least one image and a nonnegative worker count")
    pool = ProcessPoolExecutor(workers) if workers > 0 else None
    three, ctx, ref, joint, logits = [], [], [], [], []
    try:
        iterator = (
            pool.map(image_features, paths, chunksize=2) if pool else map(image_features, paths)
        )
        images = []
        for i, (im, t, a, z, j) in enumerate(iterator, 1):
            three.append(t)
            ctx.append(a)
            ref.append(z)
            joint.append(j)
            images.append(im)
            if len(images) == 8 or i == len(paths):
                # Keep the inference convolution shape constant for a single image and
                # a full batch. Some FP32 backends select different kernels by batch size.
                count = len(images)
                padded = images + [images[-1]] * (8 - count)
                batch = torch.from_numpy(np.stack(padded)[:, None]).to(
                    device, dtype=torch.float32, memory_format=torch.channels_last
                )
                score = net(batch)[:count]
                logits.extend((score[:, 1] - score[:, 0]).cpu().tolist())
                images.clear()
    finally:
        if pool:
            pool.shutdown()
    return tuple(np.asarray(v, dtype=np.float64) for v in (three, ctx, ref, joint, logits))


def four_heads(r, c, e, b):
    return np.stack((c, 0.25 * r + 0.75 * c, e, b), axis=-1)


def make_heads(train, evaluate):
    params = {}
    scores = []
    recipes = (
        ("r", 0, lambda x: fisher(x, 0.25)),
        ("c", 1, lambda x: fisher(x, 2.0)),
        ("e", 2, lambda x: fit_action(x, 1.5, 0)),
        ("b", 2, lambda x: fit_action(x, 0.12, 512)),
    )
    for name, idx, fn in recipes:
        w, b = fn(train[idx])
        scores.append((evaluate[idx] @ w + b).cpu().numpy())
        params[name + "_w"] = w.cpu().numpy()
        params[name + "_b"] = float(b)
    return four_heads(*scores), params


def fit(manifest, backbone, out, device, workers):
    groups = tuning_splits(manifest)
    n = len(groups["train"])
    if n < 4:
        raise ValueError("Four-fold grouped training requires at least four training pairs")
    sources = groups["train"] + groups["val"]
    paths = [path for _, pair in sources for path in pair]
    x, ctx, z, j, cnn = extract(paths, load_backbone(backbone, device), device, workers)
    features = [
        torch.as_tensor(array.reshape(len(sources), 2, -1), device=device) for array in (x, z, j)
    ]
    train = [v[:n] for v in features]
    val = [v[n:] for v in features]
    ctx = ctx.reshape(len(sources), 2, -1)
    cnn = cnn.reshape(-1, 2)
    base_mean, base_sd = policy.norm_stats(cnn[:n].ravel())
    cnn = (cnn - base_mean) / base_sd
    valid, params = make_heads(train, val)
    params.update(base_mean=base_mean, base_sd=base_sd)
    oof = np.empty((n, 2, 4))
    for fold in np.array_split(np.random.default_rng(73).permutation(n), 4):
        keep = np.ones(n, dtype=bool)
        keep[fold] = False
        oof[fold], _ = make_heads([v[keep] for v in train], [v[fold] for v in train])
    ranked = policy.static_candidates(oof)
    _, coeff, intercept, anchor = policy.static_candidates(valid)[0]
    train_anchors, train_state = policy.action_banks(oof, ctx[:n], cnn[:n], ranked)
    val_state = policy.descriptors(valid, anchor, ctx[n:], cnn[n:])
    params.update(strict_coef=coeff, intercept=intercept)
    chosen = {}
    for family, reward in (("cotracenet", 1.0), ("gate_ce", 0.0)):
        best = None
        for ridge in policy.RIDGE:
            for harm in policy.HARM if reward else (1.0,):
                g = policy.train_policy(train_anchors, train_state, ridge, harm, reward, device)
                correction, _ = policy.adjustment(val_state, g)
                center = float(correction.mean())
                for alpha in policy.AMPLITUDES:
                    score = policy.apply_final(anchor, correction, alpha, center)
                    metrics = policy.score_stats(score)
                    rank = policy.order(metrics)
                    if best is None or rank > best[0]:
                        best = (
                            rank,
                            g,
                            dict(
                                alpha=alpha,
                                harm=harm,
                                ridge=ridge,
                                center=center,
                                validation=metrics,
                            ),
                        )
        _, g, chosen[family] = best
        params.update({family + "_" + k: v for k, v in g.items()})
    if out.exists():
        raise FileExistsError(f"Choose a new model directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        np.savez_compressed(model / "weights.npz", **params)
        lock = {
            "model": "CoTraceNet",
            "version": 1,
            "seed": 73,
            "gate": chosen,
            "weights": digest(model / "weights.npz"),
            "backbone": digest(backbone),
            "training_pairs": n,
            "validation_pairs": len(groups["val"]),
            "source_ids": [hashlib.sha256(s.encode()).hexdigest() for s, _ in sources],
            "coeff": coeff.tolist(),
            "intercept": intercept,
        }
        (model / "selection.json").write_text(
            json.dumps(lock, indent=2, allow_nan=False), encoding="utf8"
        )
        model.rename(out)
    return chosen


def load_model(fit, backbone):
    lock = json.loads((fit / "selection.json").read_text(encoding="utf8"))
    if (
        lock.get("version") != 1
        or digest(fit / "weights.npz") != lock["weights"]
        or digest(backbone) != lock["backbone"]
    ):
        raise ValueError("Model lock, weights, or backbone do not match")
    with np.load(fit / "weights.npz", allow_pickle=False) as archive:
        params = dict(archive)
    return lock, params


def score_features(features, lock, params):
    x, ctx, z, j, raw = features
    matrices = dict(r=x, c=z, e=j, b=j)
    r, c, e, b = (
        matrix @ params[name + "_w"] + params[name + "_b"] for name, matrix in matrices.items()
    )
    acts = four_heads(r, c, e, b)
    anchor = acts @ params["strict_coef"] + params["intercept"]
    cnn = (raw - params["base_mean"]) / params["base_sd"]
    # The training descriptors use (source, class, head); a length-one sample
    # axis adapts independent images without consulting a companion image.
    state = policy.descriptors(acts[:, None, :], anchor[:, None], ctx[:, None, :], cnn[:, None])[
        :, 0, :
    ]
    result = dict(
        baseline=raw,
        texture=c,
        fixed_matched=acts[:, 1],
        directional=e,
        subspace=b,
        strict_fixed=anchor,
        mean_gate=anchor.copy(),
    )
    for family in ("cotracenet", "gate_ce"):
        g = {k: params[family + "_" + k] for k in ("gw", "gb", "mu", "sd")}
        delta, prob = policy.adjustment(state, g)
        cfg = lock["gate"][family]
        result[family] = policy.apply_final(anchor, delta, cfg["alpha"], cfg["center"])
        if family == "cotracenet":
            for i, k in enumerate(("p_keep", "p_up", "p_down")):
                result[k] = prob[:, i]
            result["score_change"] = result[family] - anchor
    return result


def predict(paths, fit_path, backbone, device, workers):
    lock, params = load_model(fit_path, backbone)
    features = extract(paths, load_backbone(backbone, device), device, workers)
    return score_features(features, lock, params)

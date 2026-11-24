"""Mask learned paths after fitting; re-use the full model's locked epoch and mixture."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import hierarchy as h


@torch.inference_mode()
def compare(a):
    domain = a.out / a.train
    lock = json.loads((domain / "unified-models.json").read_text())["hierarchy"]
    assert h.digest(domain / "weights.pt") == lock["sha256"]
    ck = torch.load(domain / "weights.pt", weights_only=True, map_location="cpu")
    assert h.digest(domain / "train-normalizer.npz") == ck["normalizer_sha256"]
    with np.load(domain / "train-normalizer.npz") as f:
        stats = dict(f)
    net = h.Hierarchy().eval()
    net.load_state_dict(ck["weights"])
    alpha = ck["alpha"]
    result = {}
    for target in (a.train,) if a.split == "val" else ("WOW", "HILL"):
        x, ids, y, _ = h.matrix(a.root, target, a.split, stats)
        key = "val" if a.split == "val" else "test-" + target
        with np.load(domain / ("unified-" + key + ".npz")) as archive:
            base, full_ref = archive["baseline"], archive["hierarchy"]
            assert np.array_equal(ids, archive["source"]) and np.array_equal(
                y, archive["label"]
            )
        full, skip, deep = [], [], []
        for rec in x.split(384):
            direct = net.word_skip(rec).flatten()
            signed = net(rec)
            skip.append(direct)
            deep.append(signed - direct)
            full.append(signed)
        diagnostics = {}
        for name, tensors in (("full", full), ("skip_only", skip), ("deep_only", deep)):
            raw = torch.cat(tensors).double().numpy()
            scores = (1 - alpha) * base + alpha * raw
            if name == "full":
                np.testing.assert_allclose(full_ref, scores, rtol=0, atol=1.0e-9)
            hits = (scores > 0) == y
            correct_ref = (full_ref > 0) == y
            diagnostics[name] = dict(
                standalone_correct=h.measures(raw, y)[0],
                mixture_correct=int(hits.sum()),
                full_rescued=int((correct_ref & ~hits).sum()),
                full_harmed=int((~correct_ref & hits).sum()),
            )
        result[target] = diagnostics
    h.dump(
        domain / ("path-ablation-" + a.split + ".json"),
        dict(alpha=alpha, epoch=ck["epoch"], targets=result),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train", choices=("WOW", "HILL"), required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(5)
    compare(args)

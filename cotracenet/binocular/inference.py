"""Single-image frozen dual-view attention on 256 × 256 gray images."""

import json
from pathlib import Path

import numpy as np
import torch
from cotracenet.lite import joint_features

from . import attention, prepare


class Detector:
    def __init__(self, fit):
        fit = Path(fit)
        with np.load(fit / "prepare_state.npz", allow_pickle=False) as state:
            self.indices = state["indices"]
            self.weights = state["weight"]
            self.bias = float(state["bias"].item())
            mean = state["mean"]
            self.domain = str(state["domain"].item())
            if not (
                np.array_equal(state["match"], prepare.MATCH)
                and np.array_equal(state["perm"], prepare.PERM)
            ):
                raise ValueError(
                    "Residual-family mapping disagrees with the saved model"
                )
        if mean.shape != (prepare.WIDTH,) or self.indices.shape != self.weights.shape:
            raise ValueError("Unexpected source model dimensions")
        dense = np.zeros(prepare.WIDTH, dtype=np.float64)
        dense[self.indices] = self.weights
        center = mean * dense
        self.texture_center = (
            center[prepare.T0 : prepare.C0].reshape(16, 3, prepare.F).sum(axis=(1, 2))
        )
        self.cost_center = (
            center[prepare.C0 :]
            .reshape(30, 5, prepare.F)
            .sum(axis=(1, 2))[prepare.MATCH]
        )
        self.dense_weights = torch.from_numpy(dense)
        self.match = torch.as_tensor(prepare.MATCH)
        self.perm = torch.as_tensor(prepare.PERM)
        out = fit / "models"
        lock = json.loads((out / "selection_brain.json").read_text())["variants"]
        if lock["brain"]["frozen_anchor_sha256"] != lock["se"]["sha256"]:
            raise ValueError("The binocular model and its SE anchor do not match")
        self.gates = {}
        for name in ("se", "brain"):
            path = out / (name + ".npz")
            if attention.digest(path) != lock[name]["sha256"]:
                raise ValueError(f"Unexpected weights for {name}")
            with np.load(path, allow_pickle=False) as ck:
                self.gates[name] = (
                    torch.from_numpy(ck["theta"]),
                    {k: ck[k] for k in ck.files if k != "theta"},
                )

    def base(self, x):
        return float(x[self.indices] @ self.weights + self.bias)

    def lite(self, path):
        return self.base(joint_features(path))

    @torch.inference_mode()
    def brain(self, path):
        image = joint_features(path)
        base = self.base(image)
        x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float64)).reshape(
            1, -1
        )
        parts = prepare.bin_statistics(x, self.dense_weights, self.match, self.perm)
        d = {key: value.numpy() for key, value in zip(prepare.KEYS, parts)}
        d["T"] -= self.texture_center
        d["C"] -= self.cost_center
        d["base"] = np.asarray([base])
        # The shared batch interface carries a label slot used only by training's loss.
        d["label"] = np.zeros(1, dtype=np.int8)
        for name, kind in (("se", "se"), ("brain", "binocular")):
            theta, stats = self.gates[name]
            batch, _ = attention.build_inputs(d, kind, stats, "cpu")
            score, _ = attention.forward(batch, kind, theta)
            d["base"] = score.numpy()
        return float(d["base"][0])

"""Apply a fitted reward classifier to one gray image."""

import json
from pathlib import Path

import numpy as np
import torch

from . import attention, inference, prepare, readout


def verified_lock(backbone_folder, policy_folder):
    lock = json.loads((policy_folder / "readout-lock.json").read_text())
    for key, path in (
        ("attention_sha256", backbone_folder / "models/selection_brain.json"),
        ("prepare_state_sha256", backbone_folder / "prepare_state.npz"),
    ):
        if key in lock and attention.digest(path) != lock[key]:
            raise ValueError(
                "The reward head belongs to a different attention or Lite model"
            )
    if attention.digest(policy_folder / "readout-model.npz") != lock["model_sha256"]:
        raise ValueError("Decision-head weights disagree with the locked configuration")
    return lock


class PolicyDetector:
    def __init__(self, previous, policy_root, train):
        previous = Path(previous).resolve()
        self.attention, self.inference, self.prepare = attention, inference, prepare
        self.backbone = inference.Detector(previous / train)
        if self.backbone.domain != train:
            raise ValueError("Feature extractor and selected training domain disagree")
        folder = Path(policy_root) / train
        self.lock = verified_lock(previous / train, folder)
        weights = folder / "readout-model.npz"
        with np.load(weights, allow_pickle=False) as ck:
            self.parameters = {k: ck[k] for k in ck.files}
        self.threshold = float(self.lock["selection"]["policy"]["threshold"])
        self.training = train

    @torch.inference_mode()
    def predict(self, path):
        return self.predict_features(self.inference.joint_features(Path(path)))

    @torch.inference_mode()
    def predict_features(self, features):
        """Score an already extracted 34,398-dimensional single-image descriptor."""
        b = self.backbone
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float64)).reshape(
            1, -1
        )
        parts = self.prepare.bin_statistics(x, b.dense_weights, b.match, b.perm)
        d = {k: v.numpy() for k, v in zip(self.prepare.KEYS, parts)}
        d["T"] -= b.texture_center
        d["C"] -= b.cost_center
        d["base"] = np.asarray([b.base(features)])
        # The archived build_inputs helper also packs a training-loss target.
        # forward() never reads this slot. No ground truth is passed to inference.
        for name, kind in (("se", "se"), ("brain", "binocular")):
            context = {
                **d,
                "base": d["base"] if name == "se" else d["se"],
                "label": np.zeros(1),
            }
            theta, stats = b.gates[name]
            batch, _ = self.attention.build_inputs(context, kind, stats, "cpu")
            d[name] = self.attention.forward(batch, kind, theta)[0].numpy()
        review = readout.reviewer(d, self.parameters)
        _, p = readout.probabilities(d, review, self.parameters)
        score, take = readout.decide(d["brain"], review, p, self.threshold)
        return {
            "trained_on": self.training,
            "brain": float(d["brain"][0]),
            "review": float(review[0]),
            "action_probability": float(p[0]),
            "action_threshold": self.threshold,
            "action": "review" if take[0] else "keep",
            "score": float(score[0]),
            "prediction": "stego" if score[0] > 0 else "cover",
        }

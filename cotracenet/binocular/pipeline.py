"""Train the two attention stages and reward head from a public paired manifest."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..data import classification_metrics, read_splits
from ..detector import digest
from ..iterative import descriptors
from ..lite import load_model, source_hash
from . import attention, oof_brain, prepare, readout
from .predict import verified_lock


def split_data(manifest, split, workers):
    """Keep both images together and retain content identities for fold isolation."""
    pairs = sorted(read_splits(manifest, (split,))[split])
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["source"]: r for r in csv.DictReader(stream) if r["split"] == split}
    identities = {field: [] for field in ("source", "source_sha256", "cover_sha256")}
    for name, paths in pairs:
        row = rows[name]
        for role, path in zip(("cover", "stego"), paths):
            file_hash = digest(path)
            if row.get(role + "_sha256") and row[role + "_sha256"] != file_hash:
                raise ValueError(f"Image differs from manifest: {path}")
            if role == "cover":
                identities["source"].append(name)
                identities["cover_sha256"].append(file_hash)
                identities["source_sha256"].append(
                    row.get("source_sha256") or file_hash
                )
    identity = {key: np.repeat(values, 2) for key, values in identities.items()}
    identity["label"] = np.tile(np.array([0, 1], dtype=np.int8), len(pairs))
    matrix = np.empty((len(pairs) * 2, prepare.WIDTH), dtype=np.float64)
    for i, vector in enumerate(descriptors(pairs, workers)):
        matrix[i] = vector
    return matrix, identity


def check_isolation(left, right):
    for key in ("source", "source_sha256", "cover_sha256"):
        if set(left[key]).intersection(right[key]):
            raise ValueError(f"Evaluation images overlap fitted images ({key})")


def frozen(evidence, model_dir, device, *, ablations=False):
    config = json.loads((model_dir / "models/selection_brain.json").read_text())
    variants = config["variants"] if ablations else ("se", "brain")
    out = dict(evidence, lite=evidence["base"])
    for name in variants:
        inputs = (
            dict(evidence, base=out["se"])
            if name in attention.SECOND_STAGE
            else evidence
        )
        out[name] = attention.apply_gate(
            inputs, model_dir / "models", name, config, device
        )[0]
    return out


def fit(manifest, lite_fit, out, domain, device, workers):
    manifest, lite_fit, out = map(Path, (manifest, lite_fit, out))
    lock, (indices, weight, bias) = load_model(lite_fit)
    if (
        lock.get("selector") != "logistic"
        or lock.get("selected") != "iterative_readout"
        or lock.get("manifest_sha256") != digest(manifest)
    ):
        raise ValueError(
            "Supply an iterative logistic Lite model trained on this manifest"
        )
    if len(indices) != prepare.K or np.any(np.diff(indices.astype(int)) <= 0):
        raise ValueError(
            "The Lite model must contain 16,384 distinct sorted coordinates"
        )
    attention_root, policy_root = out / "attention", out / "policy"
    for root in (attention_root, policy_root):
        root.mkdir(parents=True, exist_ok=True)
    if any((base / domain).exists() for base in (attention_root, policy_root)):
        raise FileExistsError(f"A fit already exists for {domain} in {out}")
    with TemporaryDirectory(dir=out) as tmp:
        work = Path(tmp)
        att = work / "attention"
        pol = work / "policy"
        att.mkdir()
        pol.mkdir()
        train, ti = split_data(manifest, "train", workers)
        val, vi = split_data(manifest, "val", workers)
        check_isolation(ti, vi)
        if set(lock["training_ids"]) != {
            source_hash(name) for name in ti["source"][::2]
        }:
            raise ValueError("Frozen Lite training IDs differ from the manifest")
        ix = indices.astype(np.int64)
        tr, mean = prepare.make_split(train, ix, weight, bias, None, 128, device)
        va, _ = prepare.make_split(val, ix, weight, bias, mean, 128, device)
        del val
        prepare.save(
            att / "prepare_state.npz",
            indices=ix,
            weight=weight,
            bias=bias,
            mean=mean,
            domain=domain,
            match=prepare.MATCH,
            perm=prepare.PERM,
            manifest_sha=lock["manifest_sha256"],
            model_sha=lock["model_sha256"],
        )
        prepare.save(att / "train.npz", **ti, **tr)
        prepare.save(att / "val.npz", **vi, **va)
        attention.fit(att, device)
        attention.fit_brain(att, device)
        selection = json.loads((att / "models/selection_brain.json").read_text())[
            "variants"
        ]
        oof = oof_brain.train(
            train,
            ti,
            selection,
            lock["readout_lambda"],
            lock["readout_iterations"],
            device,
        )
        prepare.save(pol / "oof.npz", **oof)
        val_scores = frozen(dict(vi, **va), att, device)
        prepare.save(pol / "val.npz", **val_scores, fold=np.full(len(vi["label"]), -1))
        readout.fit(pol)
        result = json.loads((pol / "readout-lock.json").read_text())
        result.update(
            attention_sha256=attention.digest(att / "models/selection_brain.json"),
            prepare_state_sha256=attention.digest(att / "prepare_state.npz"),
        )
        (pol / "readout-lock.json").write_text(json.dumps(result, indent=2) + "\n")
        # Publish each completed directory by rename. On contention an existing fit wins.
        att.rename(attention_root / domain)
        try:
            pol.rename(policy_root / domain)
        except BaseException:
            (attention_root / domain).rename(att)
            raise


def score(manifest, out, domain, target, device, workers):
    manifest, out = Path(manifest), Path(out)
    att = out / "attention" / domain
    pol = out / "policy" / domain
    verified_lock(att, pol)
    with np.load(att / "prepare_state.npz", allow_pickle=False) as state:
        if str(state["domain"]) != domain:
            raise ValueError(
                "Frozen attention model belongs to a different training domain"
            )
        same_manifest = str(state["manifest_sha"]) == digest(manifest)
        if (target == domain) != same_manifest:
            raise ValueError(
                "Target label and source/target manifest identity disagree"
            )
        data, identity = split_data(manifest, "test", workers)
        for split in ("train", "val"):
            with np.load(att / (split + ".npz"), allow_pickle=False) as fit_ids:
                check_isolation(fit_ids, identity)
        e, _ = prepare.make_split(
            data,
            state["indices"],
            state["weight"],
            float(state["bias"]),
            state["mean"],
            128,
            device,
        )
    result = frozen(dict(identity, **e), att, device, ablations=True)
    prepare.save(
        pol / ("test_" + target + ".npz"),
        **result,
        fold=np.full(len(identity["label"]), -1),
    )
    readout.score(pol, domain, (target,))
    ablation_keys = json.loads((att / "models/selection_brain.json").read_text())[
        "variants"
    ]
    prepare.save(
        att / (domain + "-to-" + target + ".npz"),
        **{key: result[key] for key in ("source", "label", "lite", *ablation_keys)},
    )
    with np.load(
        pol / (domain + "-to-" + target + ".npz"), allow_pickle=False
    ) as policy:
        return {
            key: classification_metrics(policy["label"], policy[key])
            for key in ("lite", "brain", "always", "supervised", "policy")
        }

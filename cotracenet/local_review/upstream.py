"""Freeze the previous binocular policy and export source-held-out scores for a new head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cotracenet.binocular import readout as ro


def groups(keys, salt, n):
    distinct = sorted(
        set(keys), key=lambda key: hashlib.sha256((salt + str(key)).encode()).digest()
    )
    lookup = {key: i % n for i, key in enumerate(distinct)}
    return np.array([lookup[key] for key in keys])


def assert_isolated(data, first, second):
    for field in ("source", "source_sha256", "cover_sha256"):
        if set(data[field][first]) & set(data[field][second]):
            raise ValueError("Source identity crosses a fitting boundary: " + field)


def training_anchor(d, ridge, threshold, ro):
    score = d["brain"].copy()
    assigned = np.full(len(score), -1, dtype=np.int8)
    audit = []
    for f in np.unique(d["fold"]):
        outer = d["fold"] == f
        assert_isolated(d, outer, ~outer)
        block = ro.subset(d, outer)
        keys = block["source_sha256"]
        shards = groups(keys, "anchor-policy-v1/", 5)
        all_indices = np.where(outer)[0]
        for s in np.unique(shards):
            held, fit = shards == s, shards != s
            assert_isolated(block, fit, held)
            ii = all_indices[held]
            assigned[ii] = s
            info = dict(
                outer_fold=int(f), policy_holdfold=int(s), fit=int(fit.sum()), held=int(held.sum())
            )
            audit.append(info)
            if threshold > 1:
                info["route"] = "locked policy keeps brain for every input"
                continue
            cal = ro.subset(block, fit)
            half = groups(cal["source_sha256"], "anchor-policy-review-inner-v1/", 2)
            calibration = np.empty(len(half))
            for h in (0, 1):
                a, b = half != h, half == h
                assert_isolated(cal, a, b)
                review = ro.fit_reviewer(ro.subset(cal, a), ridge)
                calibration[b] = ro.reviewer(ro.subset(cal, b), review)
            disagree = (cal["brain"] > 0) != (calibration > 0)
            answers = (calibration[disagree] > 0) == cal["label"][disagree]
            info["training_disagreements"] = int(len(answers))
            if len(answers) < 30 or len(np.unique(answers)) < 2:
                info["route"] = (
                    "keep brain; insufficient classes or original gate minimum of 30 disagreements"
                )
                continue
            gates, _ = ro.fit_gates(cal, calibration)
            review = ro.fit_reviewer(cal, ridge)
            probe = ro.subset(block, held)
            candidate = ro.reviewer(probe, review)
            _, prob = ro.probabilities(probe, candidate, gates)
            score[ii], take = ro.decide(probe["brain"], candidate, prob, threshold)
            info.update(route="source-held-out policy", changes=int(take.sum()))
    if np.any(assigned < 0) or not np.all(np.isfinite(score)):
        raise ValueError("Incomplete training predictions")
    return score, assigned, audit


def export(folder, tag, d, score, shards):
    keep = (
        "source",
        "label",
        "brain",
        "se",
        "base",
        "T",
        "C",
        "Ht",
        "Hc",
        "D",
        "fold",
        "source_sha256",
        "cover_sha256",
    )
    np.savez_compressed(
        folder / ("anchor_" + tag + ".npz"),
        **{k: d[k] for k in keep},
        anchor=score,
        policy_holdfold=shards
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--previous", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--train", choices=["WOW", "HILL"], required=True)
    cli.add_argument("--stage", choices=["train", "test"], required=True)
    args = cli.parse_args()
    before, folder = args.previous / args.train, args.out / args.train
    folder.mkdir(parents=True, exist_ok=True)
    lock = json.loads((before / "readout-lock.json").read_text())
    weights = before / "readout-model.npz"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != lock["model_sha256"]:
        raise ValueError("Readout weights differ from their locked digest")
    with np.load(weights, allow_pickle=False) as z:
        params = dict(z)
    selection = lock["selection"]["policy"]
    threshold = selection["threshold"]
    complete = dict(
        source=args.train,
        previous_model_sha256=lock["model_sha256"],
        locked_threshold=threshold,
        phase=args.stage,
    )
    if args.stage == "train":
        tr = ro.read(before / "oof.npz")
        if len(tr["source"]) != 12000:
            raise ValueError("Unexpected number of training images")
        scores, shards, audit = training_anchor(tr, lock["review_ridge"], threshold, ro)
        export(folder, "train", tr, scores, shards)
        complete["training"] = dict(
            shards=audit, rows=len(scores), changed=int(np.sum((scores > 0) != (tr["brain"] > 0)))
        )
        tags = ("val",)
    else:
        tags = ("test_WOW", "test_HILL")
    for tag in tags:
        d = ro.read(before / (tag + ".npz"))
        candidate = ro.reviewer(d, params)
        _, prob = ro.probabilities(d, candidate, params)
        scores, take = ro.decide(d["brain"], candidate, prob, threshold)
        base_hit, hit = (d["brain"] > 0) == d["label"], (scores > 0) == d["label"]
        fix, harm = int(np.sum(hit & ~base_hit)), int(np.sum(~hit & base_hit))
        record = dict(
            rows=len(hit), correct=int(hit.sum()), changes=int(take.sum()), rescued=fix, harmed=harm
        )
        if tag == "val":
            if (
                len(hit) != 2000
                or fix - ro.HARM * harm != selection["reward"]
                or record["changes"] != selection["changed"]
            ):
                raise ValueError("Validation replay disagrees with the previous policy lock")
        else:
            target = tag.removeprefix("test_")
            with np.load(before / (args.train + "-to-" + target + ".npz"), allow_pickle=False) as z:
                for key in ("source", "label"):
                    np.testing.assert_array_equal(d[key], z[key])
                np.testing.assert_array_equal(take, z["@action:policy"])
                np.testing.assert_allclose(scores, z["policy"], rtol=0, atol=1e-12)
                record["prior_max_abs_error"] = float(np.max(np.abs(scores - z["policy"])))
        export(folder, tag, d, scores, np.full(len(hit), -1, dtype=np.int8))
        complete[tag] = record
    (folder / ("anchor_" + args.stage + "_audit.json")).write_text(
        json.dumps(complete, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in complete.items() if k != "training"}, sort_keys=True))
    if "training" in complete:
        print("OOF rows and flips", complete["training"]["rows"], complete["training"]["changed"])


if __name__ == "__main__":
    main()

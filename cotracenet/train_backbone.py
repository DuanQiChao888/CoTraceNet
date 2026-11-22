"""Train the ShuffleBANet reconstruction used as CoTraceNet's frozen CNN reference."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import ShuffleBANet, shuffle_loss
from .data import ImagePairs, classification_metrics, tuning_splits
from .srm import load_srm_kernels


def epoch_pass(net, loader, device, optimizer=None, bf16=False):
    training = optimizer is not None
    net.train(training)
    labels, scores = [], []
    with torch.set_grad_enabled(training):
        for _, pairs in loader:
            x = pairs.flatten(0, 1).to(
                device, dtype=torch.float32, memory_format=torch.channels_last
            )
            y = torch.tensor([0, 1], device=device).repeat(len(pairs))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, torch.bfloat16, enabled=training and bf16):
                logits, left, right = net.forward_features(x)
            logits = logits.float()
            loss = shuffle_loss(logits, left.float(), right.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classification loss")
            if training:
                loss.backward()
                optimizer.step()
            labels.extend(y.tolist())
            scores.extend((logits[:, 1] - logits[:, 0]).detach().tolist())
    return classification_metrics(labels, scores)


def train(args):
    seed = 73
    device = torch.device(args.device)
    splits = tuning_splits(args.manifest)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = ShuffleBANet(load_srm_kernels(args.srm)).to(device, memory_format=torch.channels_last)
    torch.manual_seed(seed)
    loaders = {
        s: DataLoader(
            ImagePairs(rows),
            batch_size=4,
            shuffle=s == "train",
            generator=torch.Generator().manual_seed(seed + i),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        for i, (s, rows) in enumerate(splits.items())
    }
    optimizer = torch.optim.Adamax(
        net.parameters(), lr=0.001, betas=(0.9, 0.9), eps=1e-7, weight_decay=1e-4
    )
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch == max(2, int(0.75 * args.epochs) + 1):
            optimizer.param_groups[0]["lr"] = 0.0001
        train_metrics = epoch_pass(net, loaders["train"], device, optimizer, bf16)
        val_metrics = epoch_pass(net, loaders["val"], device)
        history.append({"epoch": epoch, "training": train_metrics, "validation": val_metrics})
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            tmp = args.out / "best.tmp"
            torch.save(
                {
                    "kind": "A",
                    "seed": seed,
                    "epoch": epoch,
                    "val_auc": best,
                    "model": net.state_dict(),
                },
                tmp,
            )
            tmp.replace(args.out / "best.pt")
        print(f"epoch {epoch}   validation AUC {val_metrics['auc']:.4f}", flush=True)
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "srm", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())

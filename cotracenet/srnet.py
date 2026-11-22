"""PyTorch reconstruction of the DDE Lab's grayscale SRNet, without selection-channel input.

Layer reference: SRNet/tflib/SRNet.py from the authors' SRNet.zip at
https://dde.binghamton.edu/download/feature_extractors/download/SRNet.zip
Archive SHA-256: 79b31cfa4177a9dee62639b5c9d2a9d154d82d75b5216878607360df62e1dfb3.
The course training schedule is fixed independently of the confirmation data.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import read_splits, tuning_splits, ImagePairs, classification_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(cin, cout, *, kernel=3, stride=1, activation=False):
    conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2)
    # Match the TF contrib truncated-normal FAN_IN initialization convention.
    std = math.sqrt(2.0 / (cin * kernel * kernel)) / 0.87962566
    nn.init.trunc_normal_(conv.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.constant_(conv.bias, 0.2)
    layers = [conv, nn.BatchNorm2d(cout, eps=0.001, momentum=0.1)]
    if activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, cin, cout, downsample=False):
        super().__init__()
        self.path = nn.Sequential(unit(cin, cout, activation=True), unit(cout, cout))
        self.skip = unit(cin, cout, kernel=1, stride=2) if downsample else None

    def forward(self, x):
        v = self.path(x)
        if self.skip is None:
            return x + v
        # TensorFlow SAME average pooling drops padded entries from the divisor.
        # For an even input it pads on the bottom/right, not the top/left.
        pad = (v.shape[-2] % 2, v.shape[-1] % 2)
        pooled = F.avg_pool2d(v, 3, stride=2, padding=pad, ceil_mode=True, count_include_pad=False)
        return self.skip(x) + pooled


class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(unit(1, 64, activation=True), unit(64, 16, activation=True))
        self.full_resolution = nn.Sequential(*(Residual(16, 16) for _ in range(5)))
        channels = [16, 16, 64, 128, 256]
        self.downsample = nn.Sequential(
            *(Residual(x, y, True) for x, y in zip(channels, channels[1:]))
        )
        self.final = nn.Sequential(unit(256, 512, activation=True), unit(512, 512))
        self.fc = nn.Linear(512, 2)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        x = self.downsample(self.full_resolution(self.stem(images)))
        return self.fc(self.final(x).mean((-1, -2)))


def loader(rows, train=False, seed=73, device="cuda", workers=2):
    return DataLoader(
        ImagePairs(rows),
        batch_size=4,
        shuffle=train,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=torch.device(device).type == "cuda",
        persistent_workers=workers > 0,
    )


def pass_epoch(model, data, optimizer=None):
    device = next(model.parameters()).device
    training = optimizer is not None
    model.train(training)
    pred = []
    with torch.set_grad_enabled(training):
        for sources, pair in data:
            x = pair.flatten(0, 1).to(
                device, dtype=torch.float32, memory_format=torch.channels_last
            )
            y = torch.tensor([0, 1], device=x.device).repeat(len(sources))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=training and device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                logits = model(x)
            logits = logits.float()
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite cross-entropy")
            if training:
                loss.backward()
                optimizer.step()
            for i, score in enumerate((logits[:, 1] - logits[:, 0]).detach().tolist()):
                pred.append((sources[i // 2], i % 2, score))
    metric = classification_metrics([r[1] for r in pred], [r[2] for r in pred])
    return metric, pred


def write_json(file, value):
    file.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_scores(file, pred):
    with file.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "label", "score"))
        writer.writerows(pred)


def train(opt):
    opt.out.mkdir(parents=True, exist_ok=False)
    splits = tuning_splits(opt.manifest)
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    model = SRNet().to(device=opt.device, memory_format=torch.channels_last)
    loaders = {
        s: loader(splits[s], s == "train", 73 + i, opt.device, opt.workers)
        for i, s in enumerate(splits)
    }
    opti = torch.optim.Adamax(
        model.parameters(), lr=0.001, betas=(0.9, 0.9), eps=1e-7, weight_decay=1e-4
    )
    cfg = dict(
        seed=73,
        epochs=opt.epochs,
        batch_pairs=4,
        train=len(splits["train"]),
        val=len(splits["val"]),
        optimizer="Adamax reference",
        lr=0.001,
        lr_epoch10=0.0001,
        betas=[0.9, 0.9],
        eps=1e-7,
        weight_decay=0.0001,
        loss="classification CE",
        augment="none",
        train_precision=(
            "bf16"
            if torch.device(opt.device).type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp32"
        ),
        eval_precision="fp32",
        tf32=False,
        selection="highest validation ROC AUC; earliest epoch on a tie",
        gpu=(
            torch.cuda.get_device_name(opt.device)
            if torch.device(opt.device).type == "cuda"
            else None
        ),
        torch=str(torch.__version__),
        parameter_count=sum(p.numel() for p in model.parameters()),
        source_sha256=sha(Path(__file__)),
        manifest_sha256=sha(opt.manifest),
    )
    write_json(opt.out / "protocol.json", cfg)
    print(json.dumps(cfg), flush=True)
    start = time.monotonic()
    best = -1
    with (opt.out / "history.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(("epoch", "lr", "train_auc", "val_auc", "val_accuracy", "seconds_elapsed"))
        for epoch in range(1, opt.epochs + 1):
            if epoch == 10:
                opti.param_groups[0]["lr"] = 0.0001
            tr, _ = pass_epoch(model, loaders["train"], opti)
            val, pred = pass_epoch(model, loaders["val"])
            data = (
                epoch,
                opti.param_groups[0]["lr"],
                tr["auc"],
                val["auc"],
                val["accuracy"],
                time.monotonic() - start,
            )
            writer.writerow(data)
            f.flush()
            print("epoch, lr, train AUC, val AUC, val acc, elapsed", data, flush=True)
            if val["auc"] > best:
                best = val["auc"]
                temp = opt.out / "best.tmp"
                torch.save(dict(model=model.state_dict(), epoch=epoch, seed=73, kind="SRNet"), temp)
                temp.replace(opt.out / "best.pt")
                write_scores(opt.out / "val_preds.csv", pred)
                best_epoch = epoch
    write_json(
        opt.out / "selection.json",
        dict(
            epoch=best_epoch,
            val_auc=best,
            checkpoint_sha256=sha(opt.out / "best.pt"),
            protocol_sha256=sha(opt.out / "protocol.json"),
            source_sha256=sha(Path(__file__)),
        ),
    )


def score(opt):
    lock = json.loads((opt.fit / "selection.json").read_text())
    if (
        sha(opt.fit / "best.pt") != lock["checkpoint_sha256"]
        or sha(Path(__file__)) != lock["source_sha256"]
    ):
        raise ValueError("Checkpoint/source mismatch with validation selection")
    examples = read_splits(opt.manifest, ("test",))["test"]
    model = SRNet().to(device=opt.device, memory_format=torch.channels_last)
    model.load_state_dict(
        torch.load(opt.fit / "best.pt", map_location="cpu", weights_only=True)["model"]
    )
    metrics, pred = pass_epoch(model, loader(examples, device=opt.device, workers=opt.workers))
    opt.out.mkdir(parents=True, exist_ok=False)
    write_scores(opt.out / "predictions.csv", pred)
    write_json(
        opt.out / "metrics.json",
        dict(
            metrics,
            source_pairs=len(examples),
            images=2 * len(examples),
            confirm_sha256=sha(opt.manifest),
            checkpoint_sha256=lock["checkpoint_sha256"],
            epoch=lock["epoch"],
        ),
    )
    print("CONFIRM", metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fit = sub.add_parser("train")
    evaluate = sub.add_parser("score")
    for cmd in (fit, evaluate):
        cmd.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        cmd.add_argument("--workers", type=int, default=2)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--fit", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=12)
    opt = parser.parse_args()
    if opt.workers < 0 or (opt.action == "train" and opt.epochs < 1):
        parser.error("workers must be nonnegative and epochs must be positive")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    (train if opt.action == "train" else score)(opt)


if __name__ == "__main__":
    main()

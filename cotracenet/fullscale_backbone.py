"""Train and score the independent ShuffleBANet in the full BOSSbase experiment."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .backbone import ShuffleBANet, shuffle_loss
from .data import ImagePairs, classification_metrics, read_splits
from .detector import digest, load_backbone
from .srm import load_srm_kernels


SEED = 73
ENDPOINTS = (20, 50, 100, 200)
RANKS = 3
LOCAL_PAIRS = 5


class TrainingGraph(nn.Module):
    """Route the two half-image descriptors through DDP for the contrastive term."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        return self.net.forward_features(images)


def configure():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def validated_pairs(manifest, splits):
    groups = read_splits(manifest, splits)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {"cover_sha256", "stego_sha256"}
        available = fields.intersection(reader.fieldnames or ())
        if available and available != fields:
            raise ValueError("Supply both image hashes, or omit both")
        if available:
            for row in reader:
                if row["split"] in groups:
                    for key in ("cover", "stego"):
                        if digest(manifest.parent / row[key]) != row[key + "_sha256"]:
                            raise ValueError(f"Image hash mismatch for {key} of {row['source']}")
    return {key: sorted(rows, key=lambda pair: pair[0]) for key, rows in groups.items()}


def cached_images(rows, device):
    cache = torch.empty((len(rows), 2, 1, 256, 256), dtype=torch.uint8)
    for i, (_, pair) in enumerate(ImagePairs(rows)):
        cache[i] = pair
    return cache.to(device)


def inputs(pairs, device):
    return pairs.flatten(0, 1).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)


@torch.inference_mode()
def pair_scores(net, pairs, device):
    logits = net(inputs(pairs, device))
    return (logits[:, 1] - logits[:, 0]).reshape(-1, 2)


def validation(net, shard, total, device):
    net.eval()
    scores = torch.cat([pair_scores(net, batch, device) for batch in shard.split(LOCAL_PAIRS)])
    padded = torch.zeros(((total + RANKS - 1) // RANKS, 2), device=device)
    padded[:len(scores)] = scores
    gathered = [torch.empty_like(padded) for _ in range(RANKS)]
    dist.all_gather(gathered, padded)
    joined = np.concatenate([part[:len(range(rank, total, RANKS))].cpu().numpy()
                             for rank, part in enumerate(gathered)])
    return classification_metrics(np.tile([0, 1], total), joined.ravel())


def adjust_schedule(epoch, auc, state, optimizer):
    if auc >= state["anchor"] + .001:
        state.update(anchor=auc, stale=0)
    else:
        state["stale"] += 1
    if not state["drop_epoch"] and ((epoch >= 20 and state["stale"] >= 8) or epoch >= 120):
        optimizer.param_groups[0]["lr"] = .0001
        state.update(drop_epoch=epoch, anchor=auc, stale=0)
    drop = state["drop_epoch"]
    return bool(drop and epoch - drop >= 12 and state["stale"] >= 12)


def write_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def distributed_train(args, rank, device):
    rows = validated_pairs(args.manifest, ("train", "val"))
    if (len(rows["train"]), len(rows["val"])) != (6000, 1000):
        raise ValueError("The full-dataset protocol expects 6,000 train and 1,000 validation pairs")
    torch.manual_seed(SEED)
    net = ShuffleBANet(load_srm_kernels(args.srm))
    initial = hashlib.sha256()
    for t in net.state_dict().values():
        initial.update(t.detach().contiguous().numpy().tobytes())
    config = {"seed": SEED, "parameters": sum(p.numel() for p in net.parameters()),
              "manifest_sha256": digest(args.manifest), "initial_network_sha256": initial.hexdigest(),
              "torch": str(torch.__version__), "selection": args.selection,
              "training_pairs": 6000, "validation_pairs": 1000, "local_pairs": LOCAL_PAIRS,
              "world": RANKS, "precision": "train BF16 / validation FP32",
              "optimizer": "Adamax, lr=0.001 to 0.0001, betas=(0.9,0.9), eps=1e-7, decay=1e-4"}
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=args.resume)
        if args.resume:
            if json.loads((args.out / "config.json").read_text("utf-8")) != config:
                raise ValueError("The resumed run configuration differs")
        else:
            write_json(args.out / "config.json", config)
    dist.barrier()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device=device, memory_format=torch.channels_last)
    graph = DistributedDataParallel(TrainingGraph(net), device_ids=[rank], broadcast_buffers=False)
    opt = torch.optim.Adamax(net.parameters(), lr=.001, betas=(.9, .9), eps=1e-7, weight_decay=1e-4)
    state = {"best": -1., "best_epoch": 0, "anchor": -1., "stale": 0, "drop_epoch": 0, "done": False}
    history, best = [], None
    resume_file = args.out / "resume.pt"
    if args.resume:
        saved = torch.load(resume_file, map_location="cpu", weights_only=True)
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        state, history, best = saved["state"], saved["history"], saved["best"]
    if not state["done"]:
        train = cached_images(rows["train"], device)
        val = cached_images(rows["val"][rank::RANKS], device)
        labels = torch.tensor([0, 1], device=device).repeat(LOCAL_PAIRS)
        for epoch in range(len(history) + 1, 201):
            lr = opt.param_groups[0]["lr"]
            net.train()
            torch.cuda.manual_seed(SEED + 10000 * epoch + rank)
            perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED + epoch))
            batches = perm.reshape(-1, RANKS, LOCAL_PAIRS)[:, rank].contiguous().to(device)
            loss_sum = torch.zeros((), device=device)
            for ids in batches:
                opt.zero_grad(set_to_none=True)
                batch = inputs(train.index_select(0, ids), device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, left, right = graph(batch)
                loss = shuffle_loss(logits.float(), left.float(), right.float(), labels)
                loss.backward()
                opt.step()
                loss_sum += loss.detach()
            dist.all_reduce(loss_sum)
            train_loss = float(loss_sum) / (len(batches) * RANKS)
            if not np.isfinite(train_loss):
                raise FloatingPointError("Nonfinite training loss")
            metrics = validation(net, val, len(rows["val"]), device)
            auc = metrics["auc"]
            candidate = args.selection == "every" or epoch in ENDPOINTS
            if candidate and auc > state["best"]:
                state.update(best=auc, best_epoch=epoch)
            stop = adjust_schedule(epoch, auc, state, opt)
            state["done"] = epoch == 200 or (args.selection == "every" and stop)
            history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                            "validation": metrics, "candidate": candidate})
            if rank == 0:
                if epoch == state["best_epoch"] or (args.selection == "fixed" and candidate):
                    ck = {"kind": "A", "seed": SEED, "epoch": epoch, "step": epoch * len(batches),
                          "val_auc": auc, "model": {k: t.cpu() for k, t in net.state_dict().items()}}
                    if epoch == state["best_epoch"]:
                        best = ck
                    if args.selection == "fixed":
                        write_checkpoint(args.out / f"epoch-{epoch:03d}.pt", ck)
                write_checkpoint(resume_file, {"model": net.state_dict(), "optimizer": opt.state_dict(),
                                               "state": state, "history": history, "best": best})
                print(f"Epoch {epoch}: val AUC {auc:.6f}", flush=True)
            if state["done"]:
                break
    if rank == 0:
        write_checkpoint(args.out / "best.pt", best)
        write_json(args.out / "history.json", history)
        write_json(args.out / "selection.json", {"epoch": state["best_epoch"], "val_auc": state["best"],
                   "epochs_run": len(history), "lr_drop_epoch": state["drop_epoch"],
                   "selection": args.selection, "best_sha256": digest(args.out / "best.pt")})
    dist.barrier()


def train(args):
    if int(os.getenv("WORLD_SIZE", "1")) != RANKS or not dist.is_nccl_available():
        raise ValueError("Launch on one host with torchrun --standalone --nproc-per-node=3 and NCCL")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("The full-dataset training protocol requires CUDA BF16")
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    try:
        if dist.get_rank() != rank:
            raise ValueError("Use three local ranks on one host")
        distributed_train(args, rank, device)
    finally:
        dist.destroy_process_group()


def score(args):
    device = torch.device(args.device)
    net = load_backbone(args.checkpoint, device)
    rows = validated_pairs(args.manifest, (args.split,))[args.split]
    batches = DataLoader(ImagePairs(rows), batch_size=4, shuffle=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=args.out.parent,
                            delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source", "label", "score"))
            for sources, pairs in batches:
                values = pair_scores(net, pairs, device).cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite classifier scores")
                for source, scores in zip(sources, values):
                    writer.writerows((source, i, float(v)) for i, v in enumerate(scores))
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    fit = actions.add_parser("train", help="Fit a CNN under the three-GPU fullscale protocol")
    fit.add_argument("--srm", type=Path, required=True)
    fit.add_argument("--selection", choices=("every", "fixed"), default="every",
                     help="Select by validation AUC every epoch or among 20, 50, 100, 200")
    fit.add_argument("--resume", action="store_true", help="Resume the last complete epoch")
    pred = actions.add_parser("score", help="Export single-image FP32 logit differences")
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--split", choices=("val", "test"), required=True)
    pred.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for action in (fit, pred):
        action.add_argument("--manifest", type=Path, required=True)
        action.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configure()
    if args.command == "train":
        train(args)
    else:
        score(args)


if __name__ == "__main__":
    main()

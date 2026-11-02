"""Train the supervised spread model (encoder + fusion + decoder from scratch) and evaluate on validation fires.

    python -m fireline.train.supervised --data DIR --stats STATS.json --cache-dir CACHE --output RUN_DIR [options]
    python -m fireline.train.supervised ... --overfit-batches 1 --epochs 200   # local sanity check

Writes ``config.json``, ``metrics.jsonl`` (one line per epoch), ``best.pt`` (highest validation AP) and ``last.pt``
into a new ``--output`` directory; refuses to reuse an existing one. Validation uses full images (padded to a
multiple of 32, padded pixels excluded) and the shared evaluation targets (at least 5 earlier days).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fireline.baselines.persistence import fire_paths
from fireline.data.cache import materialize_all
from fireline.data.inputs import Stats
from fireline.data.wind_features import WIND_FEATURES
from fireline.eval.fire_batches import iterate_fires, predict_fire
from fireline.eval.metrics import PixelMetrics
from fireline.models.spread import SpreadModel, spread_loss
from fireline.train.dataset import SpreadDataset, index_samples, worker_init


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: SpreadModel, loader, device: torch.device, amp: bool) -> dict:
    """Pooled pixel metrics plus a timing breakdown: waiting for data, model forward, metric updates (seconds)."""
    model.eval()
    metrics = PixelMetrics()
    timing = {"data_wait": 0.0, "model": 0.0, "metrics": 0.0}
    tick = time.perf_counter()
    for batch in loader:
        timing["data_wait"] += time.perf_counter() - tick
        tick = time.perf_counter()
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(batch["dynamic"], batch["static"])
        prob = torch.sigmoid(logits.float()).cpu().numpy()  # synchronises the device
        timing["model"] += time.perf_counter() - tick
        tick = time.perf_counter()
        metrics.update(prob, batch["target"].cpu().numpy(), batch["valid"].cpu().numpy())
        timing["metrics"] += time.perf_counter() - tick
        tick = time.perf_counter()
    return {**metrics.compute(), "timing_seconds": {k: round(v, 2) for k, v in timing.items()}}


CACHE_EXPANSION = 7.0  # uncompressed cache / compressed npz, measured 6.0 on the development sample, plus margin


def check_cache_disk(npzs: list[Path], cache_dir: Path) -> None:
    """Fail before caching if the cache disk cannot hold the uncompressed fires."""
    import shutil

    cache_dir.mkdir(parents=True, exist_ok=True)
    needed = CACHE_EXPANSION * sum(p.stat().st_size for p in npzs)
    free = shutil.disk_usage(cache_dir).free
    if free > 1e15:  # some container overlay filesystems report a meaningless huge value (seen on Modal)
        print(f"cache disk: free space not reported reliably; about {needed / 1e9:.1f} GB needed", flush=True)
        return
    print(f"cache disk: {free / 1e9:.1f} GB free, about {needed / 1e9:.1f} GB needed", flush=True)
    if free < needed:
        raise OSError(f"Not enough disk for the fire cache: {free / 1e9:.1f} GB free, {needed / 1e9:.1f} GB needed.")


def evaluate_fires(model: SpreadModel, fires, device: torch.device, amp: bool, batch_size: int) -> dict:
    """Pooled pixel metrics over full images; every day of a fire is encoded once (see ``predict_fire``)."""
    model.eval()
    metrics = PixelMetrics()
    timing = {"data_wait": 0.0, "model": 0.0, "metrics": 0.0}
    tick = time.perf_counter()
    for fire in fires:
        timing["data_wait"] += time.perf_counter() - tick
        tick = time.perf_counter()
        valid = fire.valid.numpy()
        for ts, prob in predict_fire(model, fire, device, amp, batch_size):
            prob = prob.cpu().numpy()  # synchronises the device
            timing["model"] += time.perf_counter() - tick
            tick = time.perf_counter()
            target = fire.fire_mask[[t + 1 for t in ts]].numpy()
            metrics.update(prob, target, np.broadcast_to(valid, target.shape))
            timing["metrics"] += time.perf_counter() - tick
            tick = time.perf_counter()
    return {**metrics.compute(), "timing_seconds": {k: round(v, 2) for k, v in timing.items()}}


def build_datasets(args: argparse.Namespace) -> tuple[SpreadDataset, list[Path], Stats]:
    stats = Stats.from_json(args.stats)
    if set(stats.years) != set(args.train_years):
        raise ValueError(f"Statistics are from {stats.years}, training years are {args.train_years}.")
    train_npz = fire_paths(args.data, args.train_years)[: args.max_train_fires or None]
    val_npz = fire_paths(args.data, args.val_years)[: args.max_val_fires or None]
    check_cache_disk(train_npz + val_npz, args.cache_dir)
    start = time.perf_counter()
    train_dirs = materialize_all(train_npz, args.cache_dir, workers=args.cache_workers)
    val_dirs = materialize_all(val_npz, args.cache_dir, workers=args.cache_workers)
    print(f"cache ready: {len(train_dirs)} train / {len(val_dirs)} val fires in {time.perf_counter() - start:.1f}s",
          flush=True)  # fmt: skip
    train = SpreadDataset(index_samples(train_dirs, args.history), stats, args.history, crop=args.crop, seed=args.seed,
                          wind=args.wind_features)  # fmt: skip
    return train, val_dirs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-years", type=int, nargs="+", default=[2018, 2019])
    parser.add_argument("--val-years", type=int, nargs="+", default=[2020])
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="bfloat16 autocast (CUDA).")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-train-fires", type=int, default=0, help="Development only.")
    parser.add_argument("--max-val-fires", type=int, default=0, help="Development only.")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--wind-features", action="store_true", help="Append fireline.data.wind_features per day.")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Sanity check: train on N fixed batches.")
    parser.add_argument("--init-encoder", type=Path, help="Encoder state dict to start from (e.g. JEPA encoder.pt).")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only fusion and decoder (needs --init-encoder or --random-encoder).",
    )
    parser.add_argument(
        "--random-encoder",
        action="store_true",
        help="Control: keep the randomly initialised (seeded) encoder and freeze it.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; refusing to overwrite.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    train, val_dirs, stats = build_datasets(args)
    n_val = len(index_samples(val_dirs, args.history))
    print(f"device {device}; {len(train)} training samples, {n_val} validation samples", flush=True)

    if args.overfit_batches:
        fixed = [train[i] for i in range(args.overfit_batches * args.batch_size)]  # fixed crops, fixed augmentation
        batches = [{k: torch.stack([s[k] for s in fixed[i : i + args.batch_size]]) for k in fixed[0]}
                   for i in range(0, len(fixed), args.batch_size)]  # fmt: skip
        train_loader: list | DataLoader = batches
    else:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  worker_init_fn=worker_init, pin_memory=device.type == "cuda", drop_last=True,
                                  persistent_workers=args.workers > 0)  # fmt: skip

    def val_fires():  # features built once per fire in background workers
        return iterate_fires(val_dirs, stats, args.history, args.workers, wind=args.wind_features)

    model = SpreadModel(history=args.history, extra_channels=len(WIND_FEATURES) if args.wind_features else 0)
    if args.freeze_encoder and (args.init_encoder is None) == (not args.random_encoder):
        raise ValueError("--freeze-encoder needs exactly one of --init-encoder and --random-encoder.")
    if args.random_encoder and not args.freeze_encoder:
        raise ValueError("--random-encoder is only meaningful with --freeze-encoder.")
    if args.init_encoder is not None:
        model.encoder.load_state_dict(torch.load(args.init_encoder, map_location="cpu"))
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)  # GroupNorm has no running statistics, so train/eval mode is irrelevant
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total, warmup = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))  # fmt: skip

    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(trainable_parameters=sum(p.numel() for p in trainable), parameters=n_params, train_samples=len(train), val_samples=n_val, device=str(device), gpu_name=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  validation_padding="reflect")  # fmt: skip
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"model parameters: {n_params:,}", flush=True)

    best_ap, amp = -1.0, args.amp and device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, data_time, losses, seen = time.perf_counter(), 0.0, [], 0
        tick = time.perf_counter()
        for batch in train_loader:
            data_time += time.perf_counter() - tick
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch["dynamic"], batch["static"])
            loss = spread_loss(logits.float(), batch["target"], batch["valid"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            losses.append(loss.item())
            seen += batch["target"].shape[0]
            tick = time.perf_counter()
        train_seconds = time.perf_counter() - start
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": schedule.get_last_lr()[0],
                  "train_seconds": round(train_seconds, 2), "data_wait_seconds": round(data_time, 2),
                  "samples_per_second": round(seen / train_seconds, 1)}  # fmt: skip
        if args.overfit_batches:
            record["overfit_batch"] = evaluate(model, [batches[0]], device, amp)  # type: ignore[arg-type]
        elif epoch % args.val_every == 0 or epoch == args.epochs:
            val_start = time.perf_counter()
            record["validation"] = evaluate_fires(model, val_fires(), device, amp, args.val_batch_size)
            record["val_seconds"] = round(time.perf_counter() - val_start, 2)
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "config": config}
        torch.save(checkpoint, args.output / "last.pt")
        ap = record.get("validation", record.get("overfit_batch", {})).get("ap")
        if ap is not None and ap > best_ap:
            best_ap = ap
            torch.save(checkpoint, args.output / "best.pt")
            record["best"] = True
        with (args.output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        summary = {k: v for k, v in record.items() if k not in ("validation", "overfit_batch")}
        result = record.get("validation") or record.get("overfit_batch")
        if result:
            summary.update(ap=result["ap"], f1=result["f1"], flagged_over_observed=result["flagged_over_observed"])
        print(json.dumps(summary), flush=True)
    print(f"best AP {best_ap:.4f}", flush=True)


if __name__ == "__main__":
    main()

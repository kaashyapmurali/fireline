"""JEPA pretraining on the training years, with the gate 2a/2b measurements on validation fires.

    python -m fireline.train.jepa --data DIR --stats STATS.json --cache-dir CACHE --output RUN_DIR [options]

Writes ``config.json``, ``metrics.jsonl`` (per epoch: losses; every ``--eval-every`` epochs: gate statistics),
``jepa.pt`` (full model, last epoch) and ``encoder.pt`` (context-encoder state dict for the frozen-encoder probe).
Refuses to reuse an existing ``--output``. There is no supervised loss, but fire detections are used: today's and
tomorrow's fire masks set the near-fire loss weights, and the crop sampling prefers fire. No checkpoint is selected on
validation.

Gate statistics on full validation images (padded cells excluded):
- 2a: effective rank of today's context latents (random sample of cells) and the smallest per-channel std.
- 2b: latent prediction error versus "tomorrow's target latent = today's target latent", on cells near fire.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fireline.baselines.persistence import fire_paths
from fireline.data.cache import materialize_all
from fireline.data.inputs import Stats
from fireline.eval.fire_batches import FireTensors, iterate_fires
from fireline.models.jepa import (
    FORECAST_CHANNELS,
    STRIDE,
    SpreadJEPA,
    effective_rank,
    momentum_schedule,
    near_fire_cells,
)
from fireline.train.dataset import SpreadDataset, index_samples, worker_init
from fireline.train.supervised import check_cache_disk, pick_device, to_device


@torch.no_grad()
def gate_statistics(model: SpreadJEPA, fires, device: torch.device, amp: bool, max_rank_cells: int = 20_000,
                    seed: int = 0) -> dict:  # fmt: skip
    model.eval()
    history = model.history
    sums = {"prediction_near": 0.0, "copy_near": 0.0, "prediction_all": 0.0, "copy_all": 0.0}
    counts = {"near": 0, "all": 0}
    cells: list[torch.Tensor] = []
    for fire in fires:
        fire: FireTensors
        if not fire.targets:
            continue
        dynamic, static = fire.dynamic.to(device), fire.static.to(device)
        first, last = min(fire.targets) - history + 1, max(fire.targets) + 1  # inputs and t + 1 targets
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            days = torch.cat([dynamic[first : last + 1], static.expand(last + 1 - first, -1, -1, -1)], dim=1)
            context = torch.cat([model.encoder(chunk)[-1] for chunk in days.split(8)]).float()
            target = torch.cat([model.target_encoder(chunk)[-1] for chunk in days.split(8)]).float()
        target = F.layer_norm(target.movedim(1, -1), target.shape[1:2]).movedim(-1, 1)
        valid_cells = F.avg_pool2d(fire.valid.float().to(device)[None, None], STRIDE, STRIDE)[0, 0] == 1
        fire_mask = fire.fire_mask.to(device)
        for t in fire.targets:
            i = t - first
            forecast = F.avg_pool2d(dynamic[t + 1, list(FORECAST_CHANNELS)][None], STRIDE, STRIDE)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                prediction = model.predict(context[i - history + 1 : i + 1][None], forecast, target[i][None]).float()[0]
            error = ((prediction - target[i + 1]) ** 2).mean(0)
            copy_error = ((target[i] - target[i + 1]) ** 2).mean(0)
            near = near_fire_cells(fire_mask[t][None], fire_mask[t + 1][None])[0] & valid_cells
            sums["prediction_all"] += float(error[valid_cells].sum())
            sums["copy_all"] += float(copy_error[valid_cells].sum())
            counts["all"] += int(valid_cells.sum())
            sums["prediction_near"] += float(error[near].sum())
            sums["copy_near"] += float(copy_error[near].sum())
            counts["near"] += int(near.sum())
            cells.append(context[i].movedim(0, -1)[valid_cells].cpu())
    sample = torch.cat(cells)
    generator = torch.Generator().manual_seed(seed)
    sample = sample[torch.randperm(len(sample), generator=generator)[:max_rank_cells]]
    near_ratio = sums["prediction_near"] / max(sums["copy_near"], 1e-12)
    return {
        "effective_rank": effective_rank(sample),
        "latent_dim": int(sample.shape[1]),
        "min_channel_std": float(sample.std(0).min()),
        "prediction_error_near_fire": sums["prediction_near"] / max(counts["near"], 1),
        "copy_error_near_fire": sums["copy_near"] / max(counts["near"], 1),
        "prediction_over_copy_near_fire": near_ratio,
        "prediction_error_all": sums["prediction_all"] / max(counts["all"], 1),
        "copy_error_all": sums["copy_all"] / max(counts["all"], 1),
        "near_fire_cells": counts["near"],
        "cells": counts["all"],
    }


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
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--near-fire-weight", type=float, default=10.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.04)
    parser.add_argument("--weighting", choices=["near_fire", "change"], default="near_fire")
    parser.add_argument("--target-mode", choices=["state", "change"], default="state")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-fires", type=int, default=0, help="Development only.")
    parser.add_argument("--max-val-fires", type=int, default=0, help="Development only.")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; refusing to overwrite.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    stats = Stats.from_json(args.stats)
    if set(stats.years) != set(args.train_years):
        raise ValueError(f"Statistics are from {stats.years}, training years are {args.train_years}.")
    train_npz = fire_paths(args.data, args.train_years)[: args.max_train_fires or None]
    val_npz = fire_paths(args.data, args.val_years)[: args.max_val_fires or None]
    check_cache_disk(train_npz + val_npz, args.cache_dir)
    start = time.perf_counter()
    train_dirs = materialize_all(train_npz, args.cache_dir, workers=args.cache_workers)
    val_dirs = materialize_all(val_npz, args.cache_dir, workers=args.cache_workers)
    print(
        f"cache ready: {len(train_dirs)} train / {len(val_dirs)} val fires in {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    train = SpreadDataset(index_samples(train_dirs, args.history), stats, args.history, crop=args.crop, seed=args.seed,
                          next_day=True)  # fmt: skip
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        worker_init_fn=worker_init, pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=args.workers > 0)  # fmt: skip

    model = SpreadJEPA(args.history, near_fire_weight=args.near_fire_weight, variance_weight=args.variance_weight,
                       covariance_weight=args.covariance_weight, weighting=args.weighting,
                       target_mode=args.target_mode).to(device)  # fmt: skip
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total, warmup = args.epochs * len(loader), int(args.warmup_epochs * len(loader))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))  # fmt: skip
    amp = args.amp and device.type == "cuda"

    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(train_samples=len(train), device=str(device), gpu_name=torch.cuda.get_device_name() if device.type == "cuda" else None, trainable_parameters=sum(p.numel() for p in trainable),
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters()))  # fmt: skip
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{len(train)} training samples; trainable parameters {config['trainable_parameters']:,}", flush=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, sums, n = time.perf_counter(), {}, 0
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, stats_step = model.loss(batch["dynamic"], batch["static"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            model.update_target(momentum_schedule(step, total))
            for k, v in {"loss": loss.item(), **stats_step}.items():
                if not (isinstance(v, float) and math.isnan(v)):
                    sums[k] = sums.get(k, 0.0) + v
            n += 1
        record = {"epoch": epoch, **{k: v / n for k, v in sums.items()}, "lr": schedule.get_last_lr()[0],
                  "momentum": momentum_schedule(step, total), "train_seconds": round(time.perf_counter() - start, 2)}  # fmt: skip
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_start = time.perf_counter()
            record["gates"] = gate_statistics(
                model, iterate_fires(val_dirs, stats, args.history, args.workers), device, amp
            )
            record["eval_seconds"] = round(time.perf_counter() - eval_start, 2)
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, args.output / "jepa.pt")
            torch.save(model.encoder.state_dict(), args.output / "encoder.pt")
        with (args.output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}), flush=True)


if __name__ == "__main__":
    main()

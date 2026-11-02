"""Fast full-image evaluation: features are built once per fire, then all of the fire's samples are scored in batches.

``SpreadDataset(crop=None)`` builds the 5-day features of every sample from scratch, so each day is processed about
five times and samples run at batch size 1. Here a worker builds one fire's standardised, reflection-padded features
for all days (``FireTensors``), the main process moves them to the device once and slices the ``history`` days of
every sample. Values are identical to ``SpreadDataset`` (standardisation and NaN filling are elementwise, so they
commute with reflection padding).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fireline.data.cache import open_cached
from fireline.data.inputs import Stats, raw_dynamic, raw_static, standardize
from fireline.data.samples import target_days
from fireline.data.schema import NO_FIRE
from fireline.data.wind_features import wind_features


@dataclass
class FireTensors:
    fire: str
    dynamic: torch.Tensor  # (days, C_dynamic, H_pad, W_pad), standardised
    static: torch.Tensor  # (C_static, H_pad, W_pad)
    fire_mask: torch.Tensor  # (days, H_pad, W_pad), zero-padded
    valid: torch.Tensor  # (H_pad, W_pad)
    targets: list[int]  # last input day t of every sample (target day t + 1)


def build_fire(fire_dir: Path, stats: Stats, history: int, min_history: int = 5, multiple: int = 32,
               wind: bool = False) -> FireTensors:  # fmt: skip
    record = open_cached(fire_dir)
    days, height, width = record.fire_hour.shape
    pad = ((0, -height % multiple), (0, -width % multiple))
    dynamic, static = standardize(raw_dynamic(record, slice(0, days)), raw_static(record), stats)
    if wind:  # computed on the unpadded image, like the training crops
        dynamic = np.concatenate([dynamic, wind_features(dynamic, static)], axis=1)
    dynamic = np.pad(dynamic, ((0, 0), (0, 0), *pad), mode="reflect")
    static = np.pad(static, ((0, 0), *pad), mode="reflect")
    fire_mask = np.pad(np.asarray(record.fire_hour) != NO_FIRE, ((0, 0), *pad))
    valid = np.zeros(fire_mask.shape[1:], dtype=bool)
    valid[:height, :width] = True
    return FireTensors(fire_dir.name, torch.from_numpy(dynamic), torch.from_numpy(static),
                       torch.from_numpy(fire_mask), torch.from_numpy(valid),
                       target_days(days, history, min_history))  # fmt: skip


class _Fires(Dataset):
    def __init__(self, fire_dirs: list[Path], stats: Stats, history: int, wind: bool = False) -> None:
        self.fire_dirs, self.stats, self.history, self.wind = fire_dirs, stats, history, wind

    def __len__(self) -> int:
        return len(self.fire_dirs)

    def __getitem__(self, index: int) -> FireTensors:
        return build_fire(self.fire_dirs[index], self.stats, self.history, wind=self.wind)


def iterate_fires(fire_dirs: list[Path], stats: Stats, history: int, workers: int = 4,
                  wind: bool = False) -> Iterator[FireTensors]:  # fmt: skip
    """Yield one fire's tensors at a time (built in ``workers`` background processes)."""
    yield from DataLoader(_Fires(fire_dirs, stats, history, wind), batch_size=None, num_workers=workers,
                          prefetch_factor=2 if workers else None)  # fmt: skip


@torch.no_grad()
def predict_fire(model, fire: FireTensors, device: torch.device, amp: bool = False, batch_size: int = 8,
                 day_chunk: int = 8) -> Iterator[tuple[list[int], torch.Tensor]]:  # fmt: skip
    """Yield (target days t, probabilities (B, H, W)) for a fire, encoding each day once (``SpreadModel``)."""
    from fireline.models.spread import FIRE_CHANNEL

    history = model.history
    dynamic, static = fire.dynamic.to(device), fire.static.to(device)
    needed = range(min(fire.targets) - history + 1, max(fire.targets) + 1) if fire.targets else range(0)
    per_scale: list[list[torch.Tensor]] = []
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        for start in range(needed.start, needed.stop, day_chunk):
            days = dynamic[start : min(start + day_chunk, needed.stop)]
            features = model.encoder(torch.cat([days, static.expand(len(days), -1, -1, -1)], dim=1))
            per_scale.append(features)
    encoded = [torch.cat([chunk[i] for chunk in per_scale]) for i in range(len(per_scale[0]))] if per_scale else []
    offset = needed.start
    for start in range(0, len(fire.targets), batch_size):
        ts = fire.targets[start : start + batch_size]
        windows = [torch.stack([f[t - history + 1 - offset : t + 1 - offset] for t in ts]) for f in encoded]
        last_fire = dynamic[ts, FIRE_CHANNEL : FIRE_CHANNEL + 1]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model.decode(windows, last_fire)
        yield ts, torch.sigmoid(logits.float())


def iterate_batches(
    fire_dirs: list[Path], stats: Stats, history: int, device: torch.device, batch_size: int = 16, workers: int = 4
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches with keys dynamic (B, T, C, H, W), static, target, valid, all on ``device``; one fire per batch."""
    loader = DataLoader(_Fires(fire_dirs, stats, history), batch_size=None, num_workers=workers,
                        prefetch_factor=2 if workers else None, persistent_workers=False)  # fmt: skip
    for fire in loader:
        dynamic = fire.dynamic.to(device, non_blocking=True)
        static, mask, valid = fire.static.to(device), fire.fire_mask.to(device), fire.valid.to(device)
        for start in range(0, len(fire.targets), batch_size):
            ts = fire.targets[start : start + batch_size]
            yield {
                "dynamic": torch.stack([dynamic[t - history + 1 : t + 1] for t in ts]),
                "static": static.expand(len(ts), -1, -1, -1),
                "target": mask[[t + 1 for t in ts]].float(),
                "valid": valid.expand(len(ts), -1, -1),
            }

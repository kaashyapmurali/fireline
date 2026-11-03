"""PyTorch datasets: fire-biased training crops with augmentation, and full-image evaluation samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fireline.analysis.far_fire import far_region
from fireline.data.cache import open_cached
from fireline.data.convert import FireRecord
from fireline.data.history_features import fire_history
from fireline.data.inputs import FULL, Stats, Window, raw_dynamic, raw_static, standardize
from fireline.data.samples import Sample, choose_crop, pad_to_multiple, random_transform, target_days
from fireline.data.schema import NO_FIRE
from fireline.data.wind_features import wind_features


@dataclass(frozen=True)
class SampleRef:
    fire: Path  # cache directory of the fire
    t: int  # last input day


def index_samples(fire_dirs: list[Path], history: int, min_history: int = 5) -> list[SampleRef]:
    refs = []
    for fire in fire_dirs:
        n_days = int(np.load(fire / "fire_hour.npy", mmap_mode="r").shape[0])
        refs += [SampleRef(fire, t) for t in target_days(n_days, history, min_history)]
    return refs


def far_fire_pixels(refs: list[SampleRef]) -> np.ndarray:
    """Number of tomorrow's fire pixels in the far region of today's fire, per sample (reads only two masks each)."""
    counts = np.zeros(len(refs), dtype=np.int64)
    masks: dict[Path, np.ndarray] = {}
    for i, ref in enumerate(refs):
        if ref.fire not in masks:
            masks[ref.fire] = np.load(ref.fire / "fire_hour.npy", mmap_mode="r")
        today, tomorrow = masks[ref.fire][ref.t] != NO_FIRE, masks[ref.fire][ref.t + 1] != NO_FIRE
        counts[i] = int((tomorrow & far_region(today)).sum())
    return counts


class SpreadDataset(Dataset):
    """Returns dict(dynamic (T, Cd, H, W), static (Cs, H, W), target (H, W), valid (H, W)) as float tensors.

    Training (``crop`` set): fire-biased random crop plus random flips/rotations. Evaluation (``crop`` None): the
    full image zero-padded to a multiple of 32 with a valid-pixel mask.
    """

    def __init__(self, refs: list[SampleRef], stats: Stats, history: int, crop: int | None, seed: int = 0,
                 next_day: bool = False, wind: bool = False, history_channels: bool = False,
                 far_crop_prob: float = 0.0) -> None:  # fmt: skip
        """``next_day``: append day t + 1 to ``dynamic`` (JEPA pretraining target); the target mask is unchanged.

        ``history_channels``: append ``fireline.data.history_features`` per input day. ``far_crop_prob``: with this
        probability, choose the training crop around tomorrow's far fire (more than 5 px from today's) when there is
        any, instead of around all of tomorrow's fire.
        """
        if history_channels and (wind or next_day):
            raise ValueError("history_channels cannot be combined with wind or next_day.")
        self.refs, self.stats, self.history, self.crop, self.next_day = refs, stats, history, crop, next_day
        self.wind = wind  # append fireline.data.wind_features after cropping/augmentation
        self.history_channels, self.far_crop_prob = history_channels, far_crop_prob
        self.seed = seed
        self._records: dict[Path, FireRecord] = {}
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.refs)

    def set_epoch(self, epoch: int, worker: int = 0) -> None:
        self._rng = np.random.default_rng([self.seed, epoch, worker])

    def _record(self, fire: Path) -> FireRecord:
        if fire not in self._records:
            self._records[fire] = open_cached(fire)  # memory-mapped; cheap to keep open
        return self._records[fire]

    def raw_sample(self, index: int, window: Window = FULL) -> Sample:
        ref = self.refs[index]
        record = self._record(ref.fire)
        days = slice(ref.t - self.history + 1, ref.t + 1 + int(self.next_day))
        target = (np.asarray(record.fire_hour[ref.t + 1][window]) != NO_FIRE).astype(np.uint8)
        dynamic = raw_dynamic(record, days, window)
        if self.history_channels:  # days 0..t of the fire, last `history` kept; appended after the standard channels
            past = np.asarray(record.fire_hour[: ref.t + 1][(slice(None), *window)]) != NO_FIRE
            dynamic = np.concatenate([dynamic, fire_history(past)[-self.history :]], axis=1)
        return Sample(dynamic, raw_static(record, window), target)

    def crop_window(self, index: int) -> Window:
        ref = self.refs[index]
        record = self._record(ref.fire)
        input_fire = np.asarray(record.fire_hour[ref.t - self.history + 1 : ref.t + 1]) != NO_FIRE
        target = np.asarray(record.fire_hour[ref.t + 1]) != NO_FIRE
        if self.far_crop_prob and self._rng.random() < self.far_crop_prob:
            far = target & far_region(input_fire[-1])
            if far.any():
                target = far
        return choose_crop(input_fire, target, self.crop, self._rng)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.crop is not None:
            sample = random_transform(self.raw_sample(index, self.crop_window(index)), self._rng)
            valid = np.ones(sample.target.shape, dtype=bool)
        else:
            sample, valid = pad_to_multiple(self.raw_sample(index), 32)
        dynamic, static = standardize(sample.dynamic, sample.static, self.stats)
        if self.wind:
            if self.crop is None:
                raise ValueError("Use fireline.eval.fire_batches for full-image evaluation with wind features.")
            dynamic = np.concatenate([dynamic, wind_features(dynamic, static)], axis=1)
        return {
            "dynamic": torch.from_numpy(dynamic),
            "static": torch.from_numpy(static),
            "target": torch.from_numpy(sample.target.astype(np.float32)),
            "valid": torch.from_numpy(valid),
        }


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset  # type: ignore[union-attr]
    dataset.set_epoch(int(torch.initial_seed() % (2**31)), worker_id)  # distinct, reproducible per epoch and worker

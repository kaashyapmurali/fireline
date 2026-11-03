"""Version 2 training options: fire-history channels and the far-fire focus (crop choice, loss weight, oversampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_model_dataset import _cached_fire

from fireline.analysis.far_fire import far_region, ring_map
from fireline.data.cache import open_cached
from fireline.data.history_features import CAP_DAYS, HISTORY_FEATURES, fire_history
from fireline.data.inputs import DYNAMIC_FEATURES, StatsAccumulator
from fireline.data.schema import NO_FIRE
from fireline.eval.fire_batches import build_fire
from fireline.models.spread import SpreadModel, far_mask, spread_loss
from fireline.train.dataset import SpreadDataset, far_fire_pixels, index_samples


def test_fire_history_values() -> None:
    fire = np.zeros((5, 1, 3), dtype=bool)
    fire[1, 0, 0] = True  # burns on day 1 only
    fire[1, 0, 1] = fire[3, 0, 1] = True  # burns on days 1 and 3
    out = fire_history(fire)
    assert out.shape == (5, len(HISTORY_FEATURES), 1, 3)
    np.testing.assert_array_equal(out[:, 0, 0, 0], [0, 1, 1, 1, 1])  # burned_before
    np.testing.assert_allclose(out[:, 1, 0, 0] * CAP_DAYS, [CAP_DAYS, 0, 1, 2, 3])  # days since fire
    np.testing.assert_allclose(out[:, 1, 0, 1] * CAP_DAYS, [CAP_DAYS, 0, 1, 0, 1])
    assert (out[:, 0, 0, 2] == 0).all() and (out[:, 1, 0, 2] == 1).all()  # never burns


def test_fire_history_uses_no_later_day() -> None:
    rng = np.random.default_rng(0)
    fire = rng.random((8, 6, 6)) > 0.8
    np.testing.assert_array_equal(fire_history(fire[:5]), fire_history(fire)[:5])


def test_far_definitions_agree() -> None:
    rng = np.random.default_rng(1)
    for density in (0.0, 0.01, 0.05):
        today = rng.random((40, 50)) < density
        np.testing.assert_array_equal(far_region(today), ring_map(today) >= 3)
        np.testing.assert_array_equal(far_mask(torch.from_numpy(today)[None])[0].numpy(), far_region(today))


def test_pixel_weight_in_loss() -> None:
    logits = torch.randn(2, 16, 16)
    target = (torch.rand(2, 16, 16) > 0.9).float()
    valid = torch.ones(2, 16, 16, dtype=torch.bool)
    base = spread_loss(logits, target, valid)
    torch.testing.assert_close(spread_loss(logits, target, valid, pixel_weight=torch.ones(2, 16, 16)), base)
    heavier = spread_loss(logits, target, valid, pixel_weight=1 + 4 * target)
    assert heavier > base


def _stats(fire: Path):
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire))
    return accumulator.result((2019,))


def test_history_channels_train_and_eval_paths_match(tmp_path: Path) -> None:
    fire = _cached_fire(tmp_path, days=10, h=40, w=36)
    stats = _stats(fire)
    refs = index_samples([fire], history=5)
    dataset = SpreadDataset(refs, stats, history=5, crop=None, history_channels=True)
    tensors = build_fire(fire, stats, 5, history_channels=True)
    assert tensors.dynamic.shape[1] == len(DYNAMIC_FEATURES) + len(HISTORY_FEATURES)
    for i, ref in enumerate(refs):
        item = dataset[i]["dynamic"]
        torch.testing.assert_close(item, tensors.dynamic[ref.t - 4 : ref.t + 1], rtol=1e-6, atol=1e-6)
    crop = SpreadDataset(refs, stats, history=5, crop=32, history_channels=True)[0]["dynamic"]
    assert crop.shape == (5, len(DYNAMIC_FEATURES) + len(HISTORY_FEATURES), 32, 32)
    model = SpreadModel(5, extra_channels=len(HISTORY_FEATURES))
    assert model(crop[None], dataset[0]["static"][None, ..., :32, :32]).shape == (1, 32, 32)


def test_far_crop_and_far_counts(tmp_path: Path) -> None:
    fire = _cached_fire(tmp_path, days=10, h=40, w=36)
    stats, refs = _stats(fire), index_samples([fire], history=5)
    masks = np.load(fire / "fire_hour.npy") != NO_FIRE
    expected = [int((masks[r.t + 1] & far_region(masks[r.t])).sum()) for r in refs]
    np.testing.assert_array_equal(far_fire_pixels(refs), expected)
    focused = SpreadDataset(refs, stats, history=5, crop=16, far_crop_prob=1.0)
    for i, ref in enumerate(refs):
        far = masks[ref.t + 1] & far_region(masks[ref.t])
        if far.any():  # the chosen crop holds some far fire whenever there is any
            window = focused.crop_window(i)
            assert far[window].any() or far.sum() == 0

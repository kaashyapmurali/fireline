from __future__ import annotations

import numpy as np
import torch
from test_model_dataset import _cached_fire

from fireline.data.cache import open_cached
from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES, StatsAccumulator
from fireline.data.samples import Sample, transform
from fireline.data.wind_features import WIND_FEATURES, shift_mask, wind_features
from fireline.models.spread import SpreadModel
from fireline.train.dataset import SpreadDataset, index_samples

FIRE, U, V = (DYNAMIC_FEATURES.index(n) for n in ("active_fire", "wind_u", "wind_v"))


def _arrays(h: int = 24, w: int = 24) -> tuple[np.ndarray, np.ndarray]:
    dynamic = np.zeros((2, len(DYNAMIC_FEATURES), h, w), dtype=np.float32)
    dynamic[:, FIRE, 10, 8] = 1.0
    dynamic[:, U] = 1.0  # wind blowing east
    static = np.zeros((len(STATIC_FEATURES), h, w), dtype=np.float32)
    static[STATIC_FEATURES.index("downslope_east")] = -1.0  # uphill is east
    return dynamic, static


def test_shift_mask_directions() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert shift_mask(mask, 1.0, 0.0, 2)[5, 7]  # east -> right
    assert shift_mask(mask, 0.0, 1.0, 2)[3, 5]  # north -> up
    assert not shift_mask(mask, 1.0, 0.0, 9).any()  # shifted out of the image
    assert not shift_mask(mask, 0.0, 0.0, 2).any()  # no wind


def test_wind_features_values() -> None:
    dynamic, static = _arrays()
    out = wind_features(dynamic, static)
    assert out.shape == (2, len(WIND_FEATURES), 24, 24)
    assert out[0, 0, 10, 8] == 0 and out[0, 0, 10, 12] == np.float32(4 / 20)
    assert abs(out[0, 0, 0, 0] - np.hypot(10, 8) / 20) < 1e-6
    dynamic[1, FIRE] = 0.0  # no fire on day 1: distance at the cap everywhere
    assert (wind_features(dynamic, static)[1, 0] == 1.0).all()
    assert out[0, 1, 10, 10] == 1 and out[0, 2, 10, 12] == 1 and out[0, 3, 10, 16] == 1  # downwind 2/4/8 px
    np.testing.assert_allclose(out[0, 4], 1.0)  # wind east, uphill east: aligned


def test_features_commute_with_rotation() -> None:
    dynamic, static = _arrays()
    sample = Sample(dynamic, static, np.zeros((24, 24), dtype=np.uint8))
    rotated = transform(sample, hflip=False, vflip=False, rotations=1)
    after = wind_features(rotated.dynamic, rotated.static)  # features of the rotated sample
    before = np.rot90(wind_features(dynamic, static), k=1, axes=(-2, -1))  # rotated features
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_dataset_and_model_with_wind(tmp_path) -> None:
    fire = _cached_fire(tmp_path, days=9, h=40, w=36)
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire))
    item = SpreadDataset(index_samples([fire], 5), accumulator.result((2019,)), 5, crop=32, wind=True)[0]
    assert item["dynamic"].shape == (5, len(DYNAMIC_FEATURES) + len(WIND_FEATURES), 32, 32)
    model = SpreadModel(5, extra_channels=len(WIND_FEATURES))
    assert model(item["dynamic"][None], item["static"][None]).shape == (1, 32, 32)


def test_batched_evaluation_with_wind_matches_fire_tensors(tmp_path) -> None:
    from fireline.eval.fire_batches import build_fire

    fire = _cached_fire(tmp_path, days=9, h=40, w=36)
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire))
    tensors = build_fire(fire, accumulator.result((2019,)), 5, wind=True)
    assert tensors.dynamic.shape[1] == len(DYNAMIC_FEATURES) + len(WIND_FEATURES)
    assert torch.isfinite(tensors.dynamic).all()

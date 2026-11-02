from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fireline.baselines.persistence import box_mean
from fireline.data.convert import FireRecord
from fireline.data.inputs import (
    DYNAMIC_CONTINUOUS,
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    Stats,
    StatsAccumulator,
    raw_dynamic,
    raw_static,
    standardize,
)
from fireline.data.samples import Sample, fire_biased_crop, pad_to_multiple, target_days, transform
from fireline.data.schema import DYNAMIC_CHANNELS, NO_FIRE, STATIC_CHANNELS

U, V = DYNAMIC_FEATURES.index("wind_u"), DYNAMIC_FEATURES.index("wind_v")
E_, N_ = STATIC_FEATURES.index("downslope_east"), STATIC_FEATURES.index("downslope_north")


def _record(days: int = 7, h: int = 8, w: int = 10, seed: int = 0) -> FireRecord:
    rng = np.random.default_rng(seed)
    dynamic = rng.uniform(1, 100, (days, len(DYNAMIC_CHANNELS), h, w)).astype(np.float16)
    dynamic[:, DYNAMIC_CHANNELS.index("wind_direction")] = 270.0  # wind from the west -> blows east
    dynamic[:, DYNAMIC_CHANNELS.index("wind_speed")] = 4.0
    static = np.zeros((len(STATIC_CHANNELS), h, w), dtype=np.float16)
    static[STATIC_CHANNELS.index("slope")] = 45.0
    static[STATIC_CHANNELS.index("aspect")] = 180.0  # south-facing
    static[STATIC_CHANNELS.index("landcover")] = 7.0
    fire_hour = np.full((days, h, w), NO_FIRE, dtype=np.uint8)
    fire_hour[:, 2, 3] = 12
    return FireRecord("fire_t", 2019, [f"2019-08-{d + 1:02d}" for d in range(days)], dynamic, static, fire_hour)


def test_raw_features() -> None:
    record = _record()
    dynamic = raw_dynamic(record, [0, 1])
    assert dynamic.shape == (2, len(DYNAMIC_FEATURES), 8, 10)
    np.testing.assert_allclose(dynamic[:, U], 4.0, atol=1e-5)
    np.testing.assert_allclose(dynamic[:, V], 0.0, atol=1e-5)
    assert dynamic[0, DYNAMIC_FEATURES.index("active_fire"), 2, 3] == 1
    static = raw_static(record)
    assert static.shape == (len(STATIC_FEATURES), 8, 10)
    np.testing.assert_allclose(static[N_], -1.0, atol=1e-3)  # south-facing 45 degrees: downslope points south
    assert (
        static[STATIC_FEATURES.index("landcover_7")].all() and static[STATIC_FEATURES.index("landcover_6")].sum() == 0
    )


def test_stats_and_standardize(tmp_path: Path) -> None:
    accumulator = StatsAccumulator()
    accumulator.add(_record(seed=1))
    accumulator.add(_record(seed=2))
    stats = accumulator.result((2018, 2019))
    k = DYNAMIC_CONTINUOUS
    assert stats.dynamic_mean[k.index("wind_u")] == 0 and stats.dynamic_std[k.index("wind_u")] == pytest.approx(
        np.sqrt(8.0), rel=1e-4)  # fmt: skip  # RMS over u and v: sqrt((16 + 0) / 2)
    stats.to_json(tmp_path / "stats.json")
    loaded = Stats.from_json(tmp_path / "stats.json")
    np.testing.assert_allclose(loaded.dynamic_mean, stats.dynamic_mean)
    record = _record(seed=3)
    dynamic, static = standardize(raw_dynamic(record, [0, 1, 2]), raw_static(record), loaded)
    assert np.isfinite(dynamic).all() and np.isfinite(static).all()
    i = k.index("temperature_min")
    assert abs(dynamic[:, i].mean()) < 0.3  # roughly centred


def test_target_days_share_one_evaluation_set() -> None:
    assert target_days(10, history=1) == target_days(10, history=5) == [4, 5, 6, 7, 8]
    assert target_days(10, history=1, min_history=1) == list(range(9))


def _sample() -> Sample:
    record = _record()
    return Sample(raw_dynamic(record, [3, 4]), raw_static(record), record.fire_mask[5].astype(np.uint8))


def test_rotation_turns_wind_and_image_together() -> None:
    sample = _sample()  # uniform wind blowing east; fire pixel at row 2, col 3 (image 8 x 10)
    rotated = transform(sample, hflip=False, vflip=False, rotations=1)  # 90 degrees counter-clockwise
    np.testing.assert_allclose(rotated.dynamic[:, U], 0.0, atol=1e-5)
    np.testing.assert_allclose(rotated.dynamic[:, V], 4.0, atol=1e-5)  # now blows north
    assert rotated.target.shape == (10, 8)
    assert rotated.target[10 - 1 - 3, 2] == sample.target[2, 3]
    flipped = transform(sample, hflip=False, vflip=True, rotations=0)
    np.testing.assert_allclose(flipped.static[N_], 1.0, atol=1e-3)  # south-facing becomes north-facing


def test_four_rotations_are_identity() -> None:
    sample = _sample()
    back = transform(transform(transform(transform(sample, True, False, 1), False, False, 1), False, False, 1),
                     False, False, 1)  # fmt: skip
    once = transform(sample, True, False, 0)
    np.testing.assert_allclose(back.dynamic, once.dynamic)
    np.testing.assert_array_equal(back.target, once.target)


def test_crop_prefers_target_fire_and_padding() -> None:
    sample = _sample()
    sample.target[:] = 0
    sample.target[7, 9] = 1
    crop = fire_biased_crop(sample, 4, np.random.default_rng(0), tries=50)
    assert crop.target.sum() == 1 and crop.dynamic.shape[-2:] == (4, 4)
    padded, valid = pad_to_multiple(sample, 32)
    assert padded.target.shape == (32, 32) and valid.sum() == 80 and padded.dynamic.shape[-2:] == (32, 32)
    assert padded.target[8:].sum() == 0 and padded.target[:, 10:].sum() == 0  # target padded with zeros
    np.testing.assert_array_equal(padded.dynamic[..., 8, :10], sample.dynamic[..., 6, :])  # reflected inputs
    zeros, _ = pad_to_multiple(sample, 32, mode="constant")
    assert zeros.dynamic[..., 8:, :].sum() == 0


def test_box_mean() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    result = box_mean(mask, 1)
    assert result[2, 2] == pytest.approx(1 / 9) and result[0, 0] == 0 and result[1, 1] == pytest.approx(1 / 9)
    assert box_mean(np.ones((4, 4), dtype=bool), 2).min() == pytest.approx(1.0)


def test_distance_profile_rings(tmp_path: Path) -> None:
    from fireline.baselines.persistence import distance_profile
    from fireline.data.convert import save_fire

    record = _record(days=7, h=40, w=40)
    record.fire_hour[:] = NO_FIRE
    record.fire_hour[5, 10, 10] = 12  # today's fire (t = 5)
    for r, c in [(10, 10), (11, 12), (10, 14), (10, 30)]:  # tomorrow: on fire, 2 px, 4 px, 20 px
        record.fire_hour[6, r, c] = 12
    record.fire_hour[4, 0, 0] = NO_FIRE
    save_fire(record, tmp_path / "2019" / "fire_t.npz")
    result = distance_profile([tmp_path / "2019" / "fire_t.npz"])
    rings = result["rings"]
    assert rings["no fire today"]["pixels"] == 40 * 40  # t = 4 has no fire
    assert [
        rings[n]["fire_tomorrow"] for n in ("on fire today", "1-2 px", "3-5 px", "6-10 px", "11-20 px", ">20 px")
    ] == [1, 1, 1, 0, 1, 0]
    assert rings["no fire today"]["fire_tomorrow"] == 1  # t = 4: no fire that day, one pixel on day 5
    assert result["share_beyond_5px"] == 2 / 5  # the 20 px pixel and the one with no fire the day before

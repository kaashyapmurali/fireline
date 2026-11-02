from __future__ import annotations

import numpy as np

from fireline.analysis.composite_leak import (
    EDGES,
    dilate,
    drop_auc,
    fire_histograms,
    first_burn_day,
    histogram_median,
    pixel_groups,
)
from fireline.data.convert import FireRecord
from fireline.data.schema import DYNAMIC_CHANNELS, NO_FIRE, STATIC_CHANNELS


def test_first_burn_day_and_dilate() -> None:
    mask = np.zeros((3, 4, 4), dtype=bool)
    mask[1, 0, 0] = mask[2, 0, 0] = mask[2, 3, 3] = True
    assert first_burn_day(mask).tolist() == [[1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 2]]
    single = np.zeros((5, 5), dtype=bool)
    single[2, 2] = True
    assert dilate(single, 1).sum() == 9 and dilate(single, 2).all()


def test_pixel_groups_partition_by_first_burn_day() -> None:
    first = np.array([0, 5, 6, 20, 22, -1, -1])
    near = np.array([True, True, True, True, True, True, False])
    groups = pixel_groups(first, update=5, near=near)
    assert groups["already_burned"].tolist() == [True, False, False, False, False, False, False]
    assert groups["burns_on_update_day"].tolist() == [False, True, False, False, False, False, False]
    assert groups["burns_in_window"].tolist() == [False, False, True, True, False, False, False]
    assert groups["burns_after_window"].tolist() == [False, False, False, False, True, False, False]
    assert groups["never_near"].tolist() == [False] * 5 + [True, False]
    assert groups["never_far"].tolist() == [False] * 6 + [True]
    assert sum(g.astype(int) for g in groups.values()).tolist() == [1] * 7


def test_drop_auc_and_median() -> None:
    low = np.histogram([-0.3] * 10, EDGES)[0]
    high = np.histogram([0.0] * 10, EDGES)[0]
    assert drop_auc(low, high) == 1.0 and drop_auc(high, low) == 0.0 and drop_auc(low, low) == 0.5
    assert abs(histogram_median(high)) < 0.005


def test_leaking_composite_is_detected() -> None:
    """Synthetic fire: the composite updates on day 8 and drops only where the fire arrives on day 10."""
    days, h, w = 20, 12, 12
    ndvi = np.full((days, h, w), 6000.0, dtype=np.float32)
    ndvi[8:] = 5900.0  # small scene-wide change at the update
    ndvi[8:, :, :6] = 3000.0  # big drop where the fire will be
    dynamic = np.zeros((days, len(DYNAMIC_CHANNELS), h, w), dtype=np.float16)
    dynamic[:, DYNAMIC_CHANNELS.index("ndvi")] = ndvi
    dynamic[:, DYNAMIC_CHANNELS.index("evi2")] = ndvi
    fire_hour = np.full((days, h, w), NO_FIRE, dtype=np.uint8)
    fire_hour[10:12, :, :6] = 12
    record = FireRecord("fire_x", 2019, [f"2019-08-{d + 1:02d}" for d in range(days)], dynamic,
                        np.zeros((len(STATIC_CHANNELS), h, w), dtype=np.float16), fire_hour)  # fmt: skip
    hist, summary = fire_histograms(record, near_px=3)
    assert summary["updates"] == 1
    assert summary["counts"]["burns_in_window"] == h * 6
    assert summary["median_ndvi_change"]["burns_in_window"] < -0.25
    assert abs(summary["median_ndvi_change"]["never_near"] + 0.01) < 0.005
    assert drop_auc(hist["ndvi"]["burns_in_window"], hist["ndvi"]["never_near"]) == 1.0

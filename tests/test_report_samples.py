"""The report's illustrative sample is picked by input coverage, never by its prediction score."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from fireline.data.convert import FireRecord
from fireline.data.schema import DYNAMIC_CHANNELS, NO_FIRE, STATIC_CHANNELS

spec = importlib.util.spec_from_file_location("report_samples", Path(__file__).parents[1] / "report/scripts/samples.py")
assert spec is not None and spec.loader is not None
samples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(samples)


def record() -> FireRecord:
    days = 16
    return FireRecord(
        "synthetic", 2021,
        [(date(2021, 8, 1) + timedelta(days=d)).isoformat() for d in range(days)],
        np.ones((days, len(DYNAMIC_CHANNELS), 4, 4), dtype=np.float16),
        np.ones((len(STATIC_CHANNELS), 4, 4), dtype=np.float16),
        np.full((days, 4, 4), NO_FIRE, dtype=np.uint8),
    )


def test_nearest_and_tie_break_are_independent_of_target_order() -> None:
    r = record()
    assert samples.select_model_day(r, [9, 7, 8], 8, [0, 4, 0, 4]) == 8
    assert samples.select_model_day(r, [9, 7], 8, [0, 4, 0, 4]) == 7


def test_distance_uses_calendar_days_not_array_indices() -> None:
    r = record()
    r.dates[9:] = [(date.fromisoformat(d) + timedelta(days=3)).isoformat() for d in r.dates[9:]]
    assert samples.select_model_day(r, [6, 9], 8, [0, 4, 0, 4]) == 6


@pytest.mark.parametrize("channel", samples.OPTICAL)
def test_every_optical_channel_is_required_on_all_five_days(channel: str) -> None:
    r = record()
    r.dynamic[4, DYNAMIC_CHANNELS.index(channel)] = np.nan
    # A missing oldest input rules out t=8; the next window t=9 excludes it.
    assert samples.select_model_day(r, [8, 9], 8, [0, 4, 0, 4]) == 9


def test_coverage_is_measured_in_crop_with_inclusive_half_threshold() -> None:
    r = record()
    k = DYNAMIC_CHANNELS.index("ndvi")
    r.dynamic[:, k, :2] = np.nan
    assert samples.select_model_day(r, [8], 8, [0, 4, 0, 4]) == 8
    with pytest.raises(ValueError, match="No timeline day"):
        samples.select_model_day(r, [8], 8, [0, 2, 0, 4])


@pytest.mark.parametrize("channel", ["energy_release_component", "temperature_max", "wind_speed", "wind_direction"])
def test_displayed_weather_must_be_present(channel: str) -> None:
    r = record()
    r.dynamic[8, DYNAMIC_CHANNELS.index(channel)] = np.nan
    assert samples.select_model_day(r, [8, 9], 8, [0, 4, 0, 4]) == 9


def test_missing_elevation_and_incomplete_windows_fail_closed() -> None:
    r = record()
    with pytest.raises(ValueError, match="No timeline day"):
        samples.select_model_day(r, [3, 15], 8, [0, 4, 0, 4])
    r.static[STATIC_CHANNELS.index("elevation")] = np.nan
    with pytest.raises(ValueError, match="elevation coverage"):
        samples.select_model_day(r, [8], 8, [0, 4, 0, 4])

from __future__ import annotations

import numpy as np
import pytest

from fireline.data.features import (
    composite_update_days,
    days_since_update,
    downslope_vector,
    normalized_difference,
    wind_axis_components,
    wind_uv_from_direction,
)


@pytest.mark.parametrize(
    ("direction_from", "expected"),
    [(0.0, (0.0, -10.0)), (90.0, (-10.0, 0.0)), (180.0, (0.0, 10.0)), (270.0, (10.0, 0.0))],
)
def test_wind_uv_from_direction(direction_from: float, expected: tuple[float, float]) -> None:
    u, v = wind_uv_from_direction(np.array(10.0), np.array(direction_from))
    np.testing.assert_allclose((u, v), expected, atol=1e-9)


def test_wind_axis_is_sign_free() -> None:
    speed = np.array(5.0)
    east_west = wind_axis_components(speed, np.array(0.0))
    north_south_a = wind_axis_components(speed, np.array(90.0))
    north_south_b = wind_axis_components(speed, np.array(-90.0))
    np.testing.assert_allclose(east_west, (5.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(north_south_a, (-5.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(north_south_a, north_south_b, atol=1e-9)


def test_downslope_vector() -> None:
    east, north = downslope_vector(np.array([45.0, 0.0]), np.array([180.0, 90.0]))
    np.testing.assert_allclose(east, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(north, [-1.0, 0.0], atol=1e-9)  # 45 deg south-facing slope points south


def test_normalized_difference_handles_bad_denominators() -> None:
    result = normalized_difference(np.array([3.0, 0.0, np.nan]), np.array([1.0, 0.0, 1.0]))
    assert result[0] == pytest.approx(0.5)
    assert np.isnan(result[1]) and np.isnan(result[2])


def test_composite_update_detection_and_age() -> None:
    base = np.full((6, 4, 4), 5000.0, dtype=np.float32)
    base[3:] = 6000.0  # composite changes on day 3
    base[1, 0, 0] = np.nan  # a missing pixel is not an update
    updated = composite_update_days(base)
    assert updated.tolist() == [True, False, False, True, False, False]
    dates = ["2019-08-01", "2019-08-02", "2019-08-03", "2019-08-04", "2019-08-06", "2019-08-07"]
    assert days_since_update(dates, updated).tolist() == [0, 1, 2, 0, 2, 3]

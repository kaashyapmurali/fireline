"""Report arrows preserve the meteorological FROM -> flow TO sign and north-up image orientation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fireline.data.features import wind_uv_from_direction

spec = importlib.util.spec_from_file_location("report_wind", Path(__file__).parents[1] / "report/scripts/wind.py")
assert spec is not None and spec.loader is not None
wind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wind)


@pytest.mark.parametrize("direction_from,expected", [(0, (0, 1)), (90, (-1, 0)), (180, (0, -1)), (270, (1, 0))])
def test_cardinal_winds_point_to_the_correct_image_direction(direction_from: float, expected: tuple[int, int]) -> None:
    speed = np.full((32, 32), 5.0)
    u, v = wind_uv_from_direction(speed, np.full_like(speed, direction_from))
    [(x0, y0, x1, y1)] = wind.wind_arrows(u, v)
    vector = np.array([x1 - x0, y1 - y0])
    np.testing.assert_allclose(vector / np.linalg.norm(vector), expected, atol=1e-6)


def test_sparse_arrows_have_equal_length_independent_of_speed() -> None:
    u = np.ones((96, 96))
    v = np.ones_like(u)
    v[:, 32:] = 10
    arrows = wind.wind_arrows(u, v)
    assert len(arrows) == 9  # one per 32 px (12 km), not per 375 m pixel
    lengths = [np.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in arrows]
    np.testing.assert_allclose(lengths, 32 * 0.65)


def test_calm_and_missing_winds_are_omitted() -> None:
    u, v = np.zeros((96, 96)), np.zeros((96, 96))
    u[16, 16] = np.nan
    u[16, 48] = 2
    v[16, 48] = np.nan
    u[48, 48] = np.inf
    assert wind.wind_arrows(u, v) == []


def test_edge_arrows_do_not_extend_outside_crop() -> None:
    arrows = wind.wind_arrows(np.ones((64, 50)), np.zeros((64, 50)))
    assert len(arrows) == 2  # right-edge samples are omitted, not clipped
    assert all(0 <= x < 50 and 0 <= y < 64 for arrow in arrows for x, y in (arrow[:2], arrow[2:]))


def test_invalid_shapes_or_spacing_fail() -> None:
    with pytest.raises(ValueError, match="matching 2D"):
        wind.wind_arrows(np.ones((3, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="spacing"):
        wind.wind_arrows(np.ones((3, 3)), np.ones((3, 3)), spacing=2)

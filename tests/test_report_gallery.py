"""Physical-unit ranges and categorical encoding do not depend on plotting libraries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("report_gallery", Path(__file__).parents[1] / "report/scripts/gallery.py")
assert spec is not None and spec.loader is not None
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


def test_observed_temperature_is_celsius_but_forecast_is_already_celsius() -> None:
    np.testing.assert_allclose(gallery.physical_values("temperature_min", np.array([273.15, 303.15])), [0, 30], atol=2e-5)
    np.testing.assert_allclose(gallery.physical_values("forecast_temperature", np.array([15, 30])), [15, 30])


@pytest.mark.parametrize("key", ["ndvi", "evi2", "viirs_i1", "viirs_i2", "viirs_m11"])
def test_optical_archive_scales_are_undone(key: str) -> None:
    np.testing.assert_allclose(gallery.physical_values(key, np.array([1000, 5000])), [0.1, 0.5])


def test_missing_constant_and_actual_ranges() -> None:
    missing = gallery.value_summary(np.array([np.nan, np.inf]))
    assert missing["missing"] and missing["min"] is None and missing["max"] is None
    uniform = gallery.value_summary(np.array([0, 0, np.nan]))
    assert uniform["uniform"] and not uniform["missing"] and uniform["min"] == uniform["max"] == 0
    values = np.arange(100, dtype=float)
    result = gallery.value_summary(values)
    assert result["min"] == 0 and result["max"] == 99 and result["coverage"] == 1
    assert result["stretch_min"] > result["min"] and result["stretch_max"] < result["max"]


def test_landcover_uses_only_present_fixed_class_colours() -> None:
    rgb, legend = gallery.landcover_rgb(np.array([[1, 10, np.nan], [1, 17, 0]]))
    assert [row["code"] for row in legend] == [1, 10, 17]
    np.testing.assert_array_equal(rgb[0, 0], rgb[1, 0])
    np.testing.assert_array_equal(rgb[0, 2], [235, 235, 235])
    np.testing.assert_array_equal(rgb[1, 2], [235, 235, 235])
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) == 4


def test_every_model_feature_is_represented() -> None:
    from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES

    keys = {row[0] for _, rows in gallery.GALLERY for row in rows}
    assert set(DYNAMIC_FEATURES) <= keys
    assert {k for k in STATIC_FEATURES if not k.startswith("landcover_")} <= keys
    assert "landcover" in keys and "false_colour" in keys and "forecast_wind_axis" not in keys

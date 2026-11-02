from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from fireline.data.convert import (
    convert_fire,
    fire_hours,
    load_fire,
    read_geotiff,
    save_fire,
)
from fireline.data.schema import (
    CHANNEL_INDEX,
    DYNAMIC_CHANNELS,
    N_SOURCE_CHANNELS,
    NO_FIRE,
    STATIC_CHANNELS,
)

H, W = 6, 5


def _tif_bytes(array: np.ndarray) -> bytes:
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=N_SOURCE_CHANNELS,
            dtype="float32",
            crs="EPSG:32610",
            transform=from_origin(500_000, 4_200_000, 375, 375),
        ) as dataset:
            dataset.write(array)
        return memfile.read()


def _day(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = rng.uniform(0, 3000, size=(N_SOURCE_CHANNELS, H, W)).astype(np.float32)
    for name in STATIC_CHANNELS:  # static layers identical across days
        array[CHANNEL_INDEX[name]] = np.arange(H * W, dtype=np.float32).reshape(H, W)
    fire = np.full((H, W), np.nan, dtype=np.float32)
    fire[1, 2] = 954.0
    fire[3, 4] = 2118.0
    fire[0, 0] = 30.0  # detection in UTC hour 0 must survive binarisation
    array[CHANNEL_INDEX["active_fire"]] = fire
    array[CHANNEL_INDEX["ndvi"], 2, 2] = np.nan  # natural missing value
    return array


def test_fire_hours_keeps_hour_zero() -> None:
    hours = fire_hours(np.array([[np.nan, 30.0], [954.0, 2359.0]], dtype=np.float32))
    assert hours.tolist() == [[NO_FIRE, 0], [9, 23]]


def test_fire_hours_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        fire_hours(np.array([2400.0], dtype=np.float32))


def test_read_geotiff_roundtrip() -> None:
    array = _day(0)
    decoded, geo = read_geotiff(_tif_bytes(array))
    np.testing.assert_array_equal(decoded, array)
    assert geo["crs"] == "EPSG:32610"


def test_convert_save_load_roundtrip(tmp_path: Path) -> None:
    days = {"2019-08-02": _day(2), "2019-08-01": _day(1)}  # unsorted on purpose
    tifs = [(f"2019/fire_7/{d}.tif", _tif_bytes(a)) for d, a in days.items()]
    record = convert_fire("fire_7", 2019, tifs, source="test")
    assert record.dates == ["2019-08-01", "2019-08-02"]
    assert record.dynamic.shape == (2, len(DYNAMIC_CHANNELS), H, W)
    assert record.static.shape == (len(STATIC_CHANNELS), H, W)
    assert record.meta["static_max_abs_change_across_days"]["elevation"] == 0.0

    path = tmp_path / "fire_7.npz"
    save_fire(record, path)
    loaded = load_fire(path)
    assert loaded.fire_id == "fire_7" and loaded.year == 2019 and loaded.dates == record.dates
    for i, d in enumerate(record.dates):
        source = days[d]
        for name in DYNAMIC_CHANNELS:
            np.testing.assert_allclose(loaded.channel(name)[i], source[CHANNEL_INDEX[name]], rtol=1e-3, atol=1e-3)
        assert int(loaded.fire_mask[i].sum()) == 3
    assert np.isnan(loaded.channel("ndvi")[0, 2, 2])
    np.testing.assert_allclose(loaded.channel("elevation"), days["2019-08-01"][CHANNEL_INDEX["elevation"]])


def test_convert_rejects_mixed_shapes() -> None:
    full = _tif_bytes(_day(0))
    cropped = _tif_bytes(_day(1)[:, :-1, :])
    with pytest.raises(ValueError, match="different shapes"):
        convert_fire("f", 2019, [("a/2019-08-01.tif", full), ("a/2019-08-02.tif", cropped)], source="t")

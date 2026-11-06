"""Satellite alignment: geometry only, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report" / "scripts"))

import satellite

TRANSFORM = [375.0, 0.0, 621375.0, 0.0, -375.0, 4509375.0]  # the Dixie Fire grid, UTM 10N
CROP = [0, 285, 0, 235]


def test_grid_is_north_up_and_covers_the_expected_area() -> None:
    lon, lat = satellite.crop_lonlat(TRANSFORM, "EPSG:32610", CROP)
    assert lon.shape == lat.shape == (570, 470)
    assert (np.diff(lat, axis=0) < 0).all()  # rows run north to south
    assert (np.diff(lon, axis=1) > 0).all()  # columns run west to east
    assert 39.7 < lat.min() < lat.max() < 40.8 and -121.6 < lon.min() < lon.max() < -120.5


def test_upscale_samples_the_same_extent_on_a_finer_grid() -> None:
    coarse = satellite.crop_lonlat(TRANSFORM, "EPSG:32610", CROP, upscale=1)
    fine = satellite.crop_lonlat(TRANSFORM, "EPSG:32610", CROP, upscale=2)
    assert fine[0].shape == (570, 470)
    # The mean of each 2x2 block of fine pixel centres is the coarse pixel centre.
    block = fine[1].reshape(285, 2, 235, 2).mean(axis=(1, 3))
    assert np.allclose(block, coarse[1], atol=1e-6)


def test_resample_recovers_a_known_gradient() -> None:
    """An image whose red channel encodes longitude and green encodes latitude must come back as the same values."""
    lon, lat = satellite.crop_lonlat(TRANSFORM, "EPSG:32610", CROP)
    bbox = satellite.bbox_for(lon, lat)
    width, height = satellite.image_size(bbox)
    la0, la1, lo0, lo1 = bbox
    cols = lo0 + (np.arange(width) + 0.5) / width * (lo1 - lo0)
    rows = la1 - (np.arange(height) + 0.5) / height * (la1 - la0)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    rgb[..., 0] = ((cols - lo0) / (lo1 - lo0) * 250)[None, :]
    rgb[..., 1] = ((rows - la0) / (la1 - la0) * 250)[:, None]
    out = satellite.resample(rgb, bbox, lon, lat).astype(float)
    assert np.abs(out[..., 0] - (lon - lo0) / (lo1 - lo0) * 250).max() < 1.5
    assert np.abs(out[..., 1] - (lat - la0) / (la1 - la0) * 250).max() < 1.5


def test_url_uses_wms_130_axis_order() -> None:
    url = satellite.gibs_url((39.0, 40.0, -121.0, -120.0), (400, 400), "2021-08-18")
    assert "BBOX=39.000000,-121.000000,40.000000,-120.000000" in url
    assert "TIME=2021-08-18" in url and satellite.GIBS_LAYER in url


def test_densest_window_finds_the_cluster() -> None:
    import imagery

    mask = np.zeros((50, 40), dtype=bool)
    mask[30:34, 5:9] = True  # the cluster
    mask[2, 35] = True  # a stray pixel
    r, c = imagery.densest_window(mask, 8, 8)
    assert mask[r : r + 8, c : c + 8].sum() == 16

"""True-colour satellite images for the report timeline and cover, aligned to the model's pixel grid.

Source: NASA GIBS (Global Imagery Browse Services), layer ``VIIRS_SNPP_CorrectedReflectance_TrueColor``, a daily
mosaic from the same VIIRS sensor family as the fire detections. Free, no key. Imagery courtesy of NASA EOSDIS GIBS /
Worldview. One image is fetched per timeline day for the day the forecast is about (``next_date``) over the timeline's
crop, then resampled onto that crop's UTM grid, so a pixel of the satellite image lines up with a pixel of the maps.

The network call (``fetch_gibs``) is kept apart from the geometry (``crop_lonlat``, ``resample``), which is unit tested
without network access. Raw responses are cached in ``cache`` so a rebuild does not fetch twice.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.warp import transform as warp_transform
from scipy.ndimage import map_coordinates

GIBS_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_DEGREES_PER_PIXEL = 0.0025  # about 250 m, finer than the 375 m model grid
PAD_DEGREES = 0.02
UPSCALE = 2  # output pixels per model pixel: sampled on a finer grid, not upscaled afterwards
CREDIT = "Imagery: NASA EOSDIS GIBS (VIIRS/Suomi NPP corrected reflectance, true colour)."
DATE_RULE = "the day the forecast is about (next_date), one daily mosaic, not the overpass of the fire detections"


def crop_lonlat(
    transform: list[float], crs: str, crop_box: list[int], upscale: int = UPSCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every output pixel centre, ``(rows*upscale, cols*upscale)`` each.

    ``transform`` is the affine ``[a, b, c, d, e, f]`` of the processed array (``x = a*col + b*row + c``) and
    ``crop_box`` is ``[row0, row1, col0, col1]`` into that array.
    """
    r0, r1, c0, c1 = crop_box
    a, b, c, d, e, f = transform
    rows = r0 + (np.arange((r1 - r0) * upscale) + 0.5) / upscale
    cols = c0 + (np.arange((c1 - c0) * upscale) + 0.5) / upscale
    col_grid, row_grid = np.meshgrid(cols, rows)
    x = a * col_grid + b * row_grid + c
    y = d * col_grid + e * row_grid + f
    lon, lat = warp_transform(crs, "EPSG:4326", x.ravel().tolist(), y.ravel().tolist())
    return np.asarray(lon).reshape(x.shape), np.asarray(lat).reshape(x.shape)


def bbox_for(lon: np.ndarray, lat: np.ndarray, pad: float = PAD_DEGREES) -> tuple[float, float, float, float]:
    """``(lat_min, lat_max, lon_min, lon_max)`` around the grid, padded so edge pixels can be interpolated."""
    return float(lat.min() - pad), float(lat.max() + pad), float(lon.min() - pad), float(lon.max() + pad)


def image_size(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    la0, la1, lo0, lo1 = bbox
    return round((lo1 - lo0) / GIBS_DEGREES_PER_PIXEL), round((la1 - la0) / GIBS_DEGREES_PER_PIXEL)


def gibs_url(bbox: tuple[float, float, float, float], size: tuple[int, int], date: str) -> str:
    """WMS 1.3.0 GetMap in EPSG:4326 (axis order latitude, longitude)."""
    la0, la1, lo0, lo1 = bbox
    return (
        f"{GIBS_WMS}?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={GIBS_LAYER}&STYLES=&CRS=EPSG:4326"
        f"&BBOX={la0:.6f},{lo0:.6f},{la1:.6f},{lo1:.6f}&WIDTH={size[0]}&HEIGHT={size[1]}&FORMAT=image/png&TIME={date}"
    )


def resample(rgb: np.ndarray, bbox: tuple[float, float, float, float], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear-sample a lat/lon image (rows run north to south over ``bbox``) at the given pixel centres."""
    la0, la1, lo0, lo1 = bbox
    height, width = rgb.shape[:2]
    # Pixel centres: column i sits at lo0 + (i + 0.5) / width * (lo1 - lo0), row j at la1 - (j + 0.5) / height * (la1 - la0).
    fx = (lon - lo0) / (lo1 - lo0) * width - 0.5
    fy = (la1 - lat) / (la1 - la0) * height - 0.5
    channels = [map_coordinates(rgb[..., k].astype(np.float32), [fy, fx], order=1, mode="nearest") for k in range(3)]
    return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)


def fetch_gibs(url: str, retries: int = 3, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fireline-report/1.0 (research prototype)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.URLError, TimeoutError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def write_satellite(days: list[dict], npz: Path, crop_box: list[int], out_dir: Path, cache: Path) -> dict:
    """Write ``satellite-<t>.jpg`` for each day and return the provenance block for ``site.json``."""
    meta = json.loads(str(np.load(npz)["meta"]))
    lon, lat = crop_lonlat(meta["transform"], meta["crs"], crop_box)
    bbox = bbox_for(lon, lat)
    size = image_size(bbox)
    cache_dir = cache / "satellite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        date = day["next_date"]
        raw = cache_dir / f"{date}-{size[0]}x{size[1]}.png"
        if not raw.exists():
            raw.write_bytes(fetch_gibs(gibs_url(bbox, size, date)))
        rgb = np.asarray(Image.open(io.BytesIO(raw.read_bytes())).convert("RGB"))
        if rgb.shape[:2] != (size[1], size[0]):
            raise ValueError(f"GIBS returned {rgb.shape[:2]} for {date}, expected {(size[1], size[0])}.")
        Image.fromarray(resample(rgb, bbox, lon, lat)).save(out_dir / f"satellite-{day['t']}.jpg", quality=86)
    return {
        "layer": GIBS_LAYER,
        "source": "https://earthdata.nasa.gov/gibs",
        "credit": CREDIT,
        "date_rule": DATE_RULE,
        "bbox_lat_lon": [round(v, 5) for v in bbox],
        "days": [d["t"] for d in days],
        "upscale": UPSCALE,
    }

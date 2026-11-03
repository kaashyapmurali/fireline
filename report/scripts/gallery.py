"""Physical-unit metadata and finite-value summaries for the report's input gallery."""

from __future__ import annotations

import numpy as np

# key, name, source, native resolution / temporal support, units, colormap
GALLERY = (
    ("Fire", (
        ("active_fire", "Active fire", "VIIRS VNP14IMG", "375 m · daily", "0/1", "YlOrRd"),
    )),
    ("Satellite", (
        ("viirs_i1", "Red (I1)", "VIIRS VNP09GA", "500 m · daily", "reflectance", "gray"),
        ("viirs_i2", "Near infrared (I2)", "VIIRS VNP09GA", "500 m · daily", "reflectance", "gray"),
        ("viirs_m11", "Shortwave infrared (M11)", "VIIRS VNP09GA", "1 km · daily", "reflectance", "gray"),
        ("false_colour", "False colour (M11, I2, I1 → RGB)", "VIIRS VNP09GA", "500 m / 1 km · daily", "reflectance", ""),
        ("ndvi", "NDVI", "VIIRS VNP13A1", "500 m · 16-day composite, updated every 8 days", "index", "YlGn"),
        ("evi2", "EVI2", "VIIRS VNP13A1", "500 m · 16-day composite, updated every 8 days", "index", "YlGn"),
    )),
    ("Observed weather", (
        ("temperature_min", "Minimum temperature", "GRIDMET", "4.6 km · daily", "°C", "inferno"),
        ("temperature_max", "Maximum temperature", "GRIDMET", "4.6 km · daily", "°C", "inferno"),
        ("specific_humidity", "Specific humidity", "GRIDMET", "4.6 km · daily", "kg/kg", "PuBu"),
        ("precipitation", "Precipitation", "GRIDMET", "4.6 km · daily", "mm/day", "PuBu"),
        ("energy_release_component", "Energy release component", "GRIDMET", "4.6 km · daily", "index", "YlOrRd"),
        ("pdsi", "Palmer drought severity (PDSI)", "GRIDMET drought", "4.6 km · 5-day index", "index", "BrBG"),
        ("wind_speed", "Wind speed + direction", "GRIDMET", "4.6 km · daily mean", "m/s", "PuBu"),
        ("wind_u", "Eastward wind (u)", "GRIDMET · derived", "4.6 km · daily mean", "m/s", "RdBu_r"),
        ("wind_v", "Northward wind (v)", "GRIDMET · derived", "4.6 km · daily mean", "m/s", "RdBu_r"),
    )),
    ("Forecast weather", (
        ("forecast_temperature", "Forecast temperature", "GFS 00 UTC", "~28 km · hours 1–24 mean", "°C", "inferno"),
        ("forecast_specific_humidity", "Forecast specific humidity", "GFS 00 UTC", "~28 km · hours 1–24 mean", "kg/kg", "PuBu"),
        ("forecast_precipitation", "Forecast precipitation", "GFS 00 UTC", "~28 km · archive total over hours 1–24", "mm", "PuBu"),
        ("forecast_wind_speed", "Forecast wind speed", "GFS 00 UTC", "~28 km · speed of hours 1–24 mean vector", "m/s", "PuBu"),
    )),
    ("Terrain and land", (
        ("elevation", "Elevation", "SRTMGL1 v003", "30 m · static", "m", "terrain"),
        ("slope", "Slope", "SRTM · derived", "30 m · static", "°", "magma"),
        ("aspect", "Aspect / downslope direction", "SRTM · derived", "30 m · static", "° clockwise from north", "twilight"),
        ("downslope_east", "Eastward downslope component", "SRTM · derived", "30 m · static", "unitless", "RdBu_r"),
        ("downslope_north", "Northward downslope component", "SRTM · derived", "30 m · static", "unitless", "RdBu_r"),
        ("landcover", "Land cover (IGBP)", "MODIS MCD12Q1.061", "500 m · annual, held static", "class ID", ""),
    )),
)

LANDCOVER = (
    ("Evergreen needleleaf forest", "#05450a"), ("Evergreen broadleaf forest", "#086a10"),
    ("Deciduous needleleaf forest", "#54a708"), ("Deciduous broadleaf forest", "#78d203"),
    ("Mixed forest", "#009900"), ("Closed shrubland", "#c6b044"), ("Open shrubland", "#dcd159"),
    ("Woody savanna", "#dade48"), ("Savanna", "#fbff13"), ("Grassland", "#b6ff05"),
    ("Permanent wetland", "#27ff87"), ("Cropland", "#c24f44"), ("Urban / built-up", "#a5a5a5"),
    ("Cropland / natural vegetation mosaic", "#ff6d4c"), ("Permanent snow / ice", "#69fff8"),
    ("Barren", "#f9ffa4"), ("Water", "#1c0dff"),
)


def physical_values(key: str, values: np.ndarray) -> np.ndarray:
    """Undo archive scales for display only. GFS temperatures are ALREADY Celsius, unlike GRIDMET."""
    values = values.astype(np.float32)
    if key in ("temperature_min", "temperature_max"):
        return values - 273.15
    if key in ("ndvi", "evi2", "viirs_i1", "viirs_i2", "viirs_m11"):
        return values / 1e4
    return values


def landcover_rgb(values: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """One fixed colour per present IGBP class; unknown and missing classes are grey, never interpolated."""
    rgb = np.full((*values.shape, 3), 235, dtype=np.uint8)
    legend = []
    for code, (label, colour) in enumerate(LANDCOVER, 1):
        present = values == code
        if present.any():
            rgb[present] = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
            legend.append({"code": code, "label": label, "colour": colour})
    return rgb, legend


def value_summary(values: np.ndarray) -> dict:
    """Actual min/max, percentile colour limits and availability; JSON-safe even when entirely missing."""
    valid = np.isfinite(values)
    finite = values[valid]
    result = {"missing": not bool(valid.any()), "coverage": float(valid.mean()), "uniform": False,
              "min": None, "max": None, "stretch_min": None, "stretch_max": None}
    if finite.size:
        low, high = np.percentile(finite, [2, 98])
        result.update(min=float(finite.min()), max=float(finite.max()), uniform=bool(finite.min() == finite.max()),
                      stretch_min=float(low), stretch_max=float(high))
    return result

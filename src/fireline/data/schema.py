"""Channel layout of the WildfireSpreadTS GeoTIFFs and the per-fire storage format.

Source semantics are taken from the dataset-creation code (github.com/SebastianGer/WildfireSpreadTSCreateDataset,
``DataPreparation/satellites/FirePred.py``).
"""

from __future__ import annotations

from typing import Final

ZENODO_WSTS_URL: Final = "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip?download=1"
ZENODO_WSTS_PLUS_URL: Final = "https://zenodo.org/records/17584629/files/WSTSPlus.zip?download=1"

SOURCE_CHANNELS: Final[tuple[str, ...]] = (
    "viirs_m11",  # 0  VNP09GA surface reflectance x 1e4, shortwave infrared (~2.25 um), daily median
    "viirs_i2",  # 1  near infrared (~0.86 um)
    "viirs_i1",  # 2  red (~0.64 um)
    "ndvi",  # 3  VNP13A1 16-day composite x 1e4, last composite STARTING within the previous 15 days, so it can
    #          include observations from up to ~15 days after this day (future leakage, see the README)
    "evi2",  # 4  same composite as ndvi
    "precipitation",  # 5  GRIDMET pr (mm/day)
    "wind_speed",  # 6  GRIDMET vs (m/s)
    "wind_direction",  # 7  GRIDMET th (degrees clockwise from north)
    "temperature_min",  # 8  GRIDMET tmmn (K)
    "temperature_max",  # 9  GRIDMET tmmx (K)
    "energy_release_component",  # 10 GRIDMET erc
    "specific_humidity",  # 11 GRIDMET sph (kg/kg)
    "slope",  # 12 SRTM-derived (degrees), static
    "aspect",  # 13 SRTM-derived downslope direction (degrees clockwise from north), static
    "elevation",  # 14 SRTM (m), static
    "pdsi",  # 15 Palmer drought severity index (5-day product)
    "landcover",  # 16 MODIS MCD12Q1 LC_Type1 class 1..17 for the fire year, static
    "forecast_precipitation",  # 17 GFS, forecast issued 00 UTC on this day; semantics change on 2019-11-07
    "forecast_wind_speed",  # 18 GFS 10 m wind speed from the mean u/v over forecast hours 01-24
    "forecast_wind_axis",  # 19 GFS wind axis in (-90, 90) degrees: the released data used atan instead of atan2, so
    #          the direction along the axis is lost; on one fire it behaves like a compass angle
    "forecast_temperature",  # 20 GFS 2 m temperature (deg C)
    "forecast_specific_humidity",  # 21 GFS 2 m specific humidity
    "active_fire",  # 22 VIIRS active-fire detection time hhmm (UTC), NaN where no detection
)
N_SOURCE_CHANNELS: Final = len(SOURCE_CHANNELS)
CHANNEL_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(SOURCE_CHANNELS)}

STATIC_CHANNELS: Final[tuple[str, ...]] = ("slope", "aspect", "elevation", "landcover")
FIRE_CHANNEL: Final = "active_fire"
DYNAMIC_CHANNELS: Final[tuple[str, ...]] = tuple(
    c for c in SOURCE_CHANNELS if c not in STATIC_CHANNELS and c != FIRE_CHANNEL
)

NO_FIRE: Final = 255
"""Sentinel in the stored ``fire_hour`` array for pixels without a detection (hours are 0..23)."""

FORMAT_VERSION: Final = 1

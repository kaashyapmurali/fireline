"""Derived input features computed at load time from raw ``FireRecord`` values.

Angles follow these conventions (verified against the dataset-creation code):

- GRIDMET ``wind_direction``: compass degrees clockwise from north, the direction the wind blows *from*
  (standard meteorological convention; the "from" part is assumed, not verified on data).
- GFS ``forecast_wind_axis``: an angle in (-90, 90). The released data used ``atan`` instead of ``atan2``, so only
  the axis of the forecast wind is known, not which way along it the wind blows. The creation code names it
  ``atan2(v, u)`` (math convention), but on the first converted fire it agrees with the GRIDMET compass angle modulo
  180 degrees, not the math angle; treat the convention as unverified until checked on more fires.
- ``aspect``: compass degrees clockwise from north of the downslope direction.

All vector features are returned as (east, north) components so flips and rotations can be applied consistently.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from fireline.data.convert import FireRecord


def wind_uv_from_direction(speed: np.ndarray, direction_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eastward (u) and northward (v) wind from speed and the compass direction the wind blows from."""
    theta = np.deg2rad(direction_from_deg)
    return -speed * np.sin(theta), -speed * np.cos(theta)


def wind_axis_components(speed: np.ndarray, axis_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign-free encoding of a wind known only up to its axis (angle modulo 180 degrees).

    Doubling the angle makes opposite directions identical: (speed * cos 2a, speed * sin 2a). With a math-convention
    angle (from east), east-west winds give (+speed, 0) and north-south winds (-speed, 0); with a compass angle the
    two cases swap.
    """
    doubled = np.deg2rad(2.0 * axis_deg)
    return speed * np.cos(doubled), speed * np.sin(doubled)


def downslope_vector(slope_deg: np.ndarray, aspect_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Terrain gradient pointing downhill, with length tan(slope); about (0, 0) on flat ground."""
    gradient = np.tan(np.deg2rad(slope_deg))
    theta = np.deg2rad(aspect_deg)
    return gradient * np.sin(theta), gradient * np.cos(theta)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where the denominator is not positive or an input is missing."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (a - b) / total
    return np.where(total > 0, result, np.nan).astype(np.float32)


def daily_ndvi(record: FireRecord) -> np.ndarray:
    """Daily NDVI-like index from VIIRS reflectance: (I2 - I1) / (I2 + I1); no compositing, no future data."""
    return normalized_difference(record.channel("viirs_i2"), record.channel("viirs_i1"))


def daily_nbr(record: FireRecord) -> np.ndarray:
    """Daily normalized burn ratio from VIIRS reflectance: (I2 - M11) / (I2 + M11)."""
    return normalized_difference(record.channel("viirs_i2"), record.channel("viirs_m11"))


def composite_update_days(values: np.ndarray, rel_tol: float = 1e-3, min_changed_fraction: float = 0.05) -> np.ndarray:
    """Boolean per day: True where a composite channel (``(days, H, W)``) changed from the previous day.

    A day counts as an update if at least ``min_changed_fraction`` of pixels finite on both days changed by more
    than ``rel_tol`` relative to the channel's typical magnitude. Day 0 is always marked as an update.
    """
    days = values.shape[0]
    updated = np.zeros(days, dtype=bool)
    updated[0] = True
    scale = float(np.nanmedian(np.abs(values))) or 1.0
    for d in range(1, days):
        both = np.isfinite(values[d]) & np.isfinite(values[d - 1])
        if not both.any():
            updated[d] = True
            continue
        changed = np.abs(values[d][both] - values[d - 1][both]) > rel_tol * scale
        updated[d] = changed.mean() >= min_changed_fraction
    return updated


def days_since_update(dates: list[str], updated: np.ndarray) -> np.ndarray:
    """Days elapsed since the most recent update day (0 on update days), using calendar dates."""
    ordinals = np.array([date.fromisoformat(d).toordinal() for d in dates])
    result = np.zeros(len(dates), dtype=np.float32)
    last = ordinals[0]
    for i, (ordinal, is_update) in enumerate(zip(ordinals, updated, strict=True)):
        if is_update:
            last = ordinal
        result[i] = ordinal - last
    return result

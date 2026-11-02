"""Model inputs built from a converted fire: feature definitions, normalisation statistics and sample building.

Per day (``DYNAMIC_FEATURES``): VIIRS reflectance, the WSTS composite NDVI/EVI2 (known to contain future information,
documented in the README; kept for comparability), GRIDMET weather with wind as u/v, PDSI, GFS forecasts (the axis-only forecast
wind direction is dropped: its convention is unverified and it cannot be rotated consistently under augmentation),
and the binary active-fire mask.

Per fire (``STATIC_FEATURES``): slope, elevation, the downslope vector (east, north) and one-hot land cover.

Continuous features are standardised with statistics from the training years; missing values become 0 (the mean)
after standardisation. Vector features are (east, north) pairs so flips and rotations can be applied exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fireline.data.convert import FireRecord
from fireline.data.features import downslope_vector, wind_uv_from_direction
from fireline.data.schema import DYNAMIC_CHANNELS, NO_FIRE, STATIC_CHANNELS

_DIRECT = (
    "viirs_m11", "viirs_i2", "viirs_i1", "ndvi", "evi2", "precipitation", "temperature_min", "temperature_max",
    "energy_release_component", "specific_humidity", "pdsi", "forecast_precipitation", "forecast_wind_speed",
    "forecast_temperature", "forecast_specific_humidity",
)  # fmt: skip
DYNAMIC_CONTINUOUS: tuple[str, ...] = (*_DIRECT, "wind_u", "wind_v")
DYNAMIC_FEATURES: tuple[str, ...] = (*DYNAMIC_CONTINUOUS, "active_fire")
STATIC_CONTINUOUS: tuple[str, ...] = ("slope", "elevation", "downslope_east", "downslope_north")
N_LANDCOVER = 17
STATIC_FEATURES: tuple[str, ...] = (*STATIC_CONTINUOUS, *(f"landcover_{i}" for i in range(1, N_LANDCOVER + 1)))
VECTOR_PAIRS: tuple[tuple[str, str], ...] = (("wind_u", "wind_v"),)
STATIC_VECTOR_PAIRS: tuple[tuple[str, str], ...] = (("downslope_east", "downslope_north"),)


Window = tuple[slice, slice]
FULL: Window = (slice(None), slice(None))


def _day_index(days: list[int] | np.ndarray | slice) -> slice | np.ndarray:
    """Contiguous day lists become a slice, so memory-mapped arrays read only the requested rows."""
    if isinstance(days, slice):
        return days
    days = np.asarray(days)
    if len(days) and np.all(np.diff(days) == 1):
        return slice(int(days[0]), int(days[-1]) + 1)
    return days


def raw_dynamic(record: FireRecord, days: list[int] | np.ndarray | slice, window: Window = FULL) -> np.ndarray:
    """Unstandardised dynamic features, float32 ``(len(days), len(DYNAMIC_FEATURES), h, w)`` inside ``window``.

    Slices days and window before converting, so it reads only that block from memory-mapped arrays.
    """
    index = _day_index(days)
    source = np.asarray(record.dynamic[index][..., window[0], window[1]], dtype=np.float32)
    out = np.empty((source.shape[0], len(DYNAMIC_FEATURES), *source.shape[-2:]), dtype=np.float32)
    for i, name in enumerate(_DIRECT):
        out[:, i] = source[:, DYNAMIC_CHANNELS.index(name)]
    speed = source[:, DYNAMIC_CHANNELS.index("wind_speed")]
    direction = source[:, DYNAMIC_CHANNELS.index("wind_direction")]
    u, v = wind_uv_from_direction(speed, direction)
    out[:, DYNAMIC_FEATURES.index("wind_u")] = u
    out[:, DYNAMIC_FEATURES.index("wind_v")] = v
    out[:, DYNAMIC_FEATURES.index("active_fire")] = (
        np.asarray(record.fire_hour[index][..., window[0], window[1]]) != NO_FIRE
    )
    return out


def raw_static(record: FireRecord, window: Window = FULL) -> np.ndarray:
    """Unstandardised static features, float32 ``(len(STATIC_FEATURES), h, w)``; land cover one-hot (0 if missing)."""
    static = np.asarray(record.static[:, window[0], window[1]], dtype=np.float32)
    slope, aspect = static[STATIC_CHANNELS.index("slope")], static[STATIC_CHANNELS.index("aspect")]
    east, north = downslope_vector(slope, aspect)
    landcover = static[STATIC_CHANNELS.index("landcover")]
    one_hot = np.zeros((N_LANDCOVER, *landcover.shape), dtype=np.float32)
    valid = np.isfinite(landcover) & (landcover >= 1) & (landcover <= N_LANDCOVER)
    rows, cols = np.nonzero(valid)
    one_hot[landcover[valid].astype(int) - 1, rows, cols] = 1.0
    elevation = static[STATIC_CHANNELS.index("elevation")]
    return np.concatenate([np.stack([slope, elevation, east, north]), one_hot]).astype(np.float32)


@dataclass(frozen=True)
class Stats:
    """Per-feature mean and standard deviation for the continuous features."""

    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    years: tuple[int, ...]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "years": list(self.years),
            "dynamic": {n: [float(m), float(s)] for n, m, s in zip(DYNAMIC_CONTINUOUS, self.dynamic_mean, self.dynamic_std, strict=True)},
            "static": {n: [float(m), float(s)] for n, m, s in zip(STATIC_CONTINUOUS, self.static_mean, self.static_std, strict=True)},
        }  # fmt: skip
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> Stats:
        payload = json.loads(path.read_text())
        dynamic = np.array([payload["dynamic"][n] for n in DYNAMIC_CONTINUOUS], dtype=np.float64)
        static = np.array([payload["static"][n] for n in STATIC_CONTINUOUS], dtype=np.float64)
        return cls(dynamic[:, 0], dynamic[:, 1], static[:, 0], static[:, 1], tuple(payload["years"]))


class StatsAccumulator:
    """Streaming sums over valid pixels; static layers are weighted by the fire's number of days."""

    def __init__(self) -> None:
        self.dynamic = np.zeros((3, len(DYNAMIC_CONTINUOUS)))  # count, sum, sum of squares
        self.static = np.zeros((3, len(STATIC_CONTINUOUS)))

    def add(self, record: FireRecord) -> None:
        days = record.dynamic.shape[0]
        for d in range(days):  # one day at a time keeps memory small
            values = raw_dynamic(record, [d])[0, : len(DYNAMIC_CONTINUOUS)].astype(np.float64)
            _accumulate(self.dynamic, values, weight=1)
        _accumulate(self.static, raw_static(record)[: len(STATIC_CONTINUOUS)].astype(np.float64), weight=days)

    def result(self, years: tuple[int, ...]) -> Stats:
        dynamic_mean, dynamic_std = _finalize(self.dynamic)
        static_mean, static_std = _finalize(self.static)
        _vector_scale(self.dynamic, dynamic_mean, dynamic_std, DYNAMIC_CONTINUOUS, VECTOR_PAIRS)
        _vector_scale(self.static, static_mean, static_std, STATIC_CONTINUOUS, STATIC_VECTOR_PAIRS)
        return Stats(dynamic_mean, dynamic_std, static_mean, static_std, years)


def _accumulate(totals: np.ndarray, values: np.ndarray, weight: int) -> None:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    totals[0] += weight * valid.sum(axis=(1, 2))
    totals[1] += weight * filled.sum(axis=(1, 2))
    totals[2] += weight * (filled**2).sum(axis=(1, 2))


def _finalize(totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = np.maximum(totals[0], 1)
    mean = totals[1] / count
    std = np.sqrt(np.maximum(totals[2] / count - mean**2, 0))
    return mean, np.where(std > 0, std, 1.0)


def _vector_scale(
    totals: np.ndarray, mean: np.ndarray, std: np.ndarray, names: tuple[str, ...], pairs: tuple[tuple[str, str], ...]
) -> None:
    """Vector components get mean 0 and one shared RMS scale, so rotations and flips commute with standardisation."""
    for a, b in pairs:
        i, j = names.index(a), names.index(b)
        mean_square = (totals[2, i] / max(totals[0, i], 1) + totals[2, j] / max(totals[0, j], 1)) / 2
        scale = float(np.sqrt(mean_square)) or 1.0
        mean[[i, j]] = 0.0
        std[[i, j]] = scale


def standardize(dynamic: np.ndarray, static: np.ndarray, stats: Stats) -> tuple[np.ndarray, np.ndarray]:
    """Standardise continuous features in place-safe copies; NaN -> 0 afterwards."""
    dynamic = dynamic.copy()
    static = static.copy()
    k, s = len(DYNAMIC_CONTINUOUS), len(STATIC_CONTINUOUS)
    dynamic[:, :k] = (dynamic[:, :k] - stats.dynamic_mean[None, :, None, None]) / stats.dynamic_std[None, :, None, None]
    static[:s] = (static[:s] - stats.static_mean[:, None, None]) / stats.static_std[:, None, None]
    return np.nan_to_num(dynamic, nan=0.0), np.nan_to_num(static, nan=0.0)

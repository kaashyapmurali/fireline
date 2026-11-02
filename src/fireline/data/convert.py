"""Convert one fire's daily GeoTIFFs into a compact per-fire ``.npz`` file and load it back.

Stored layout (raw source values, no normalisation; derived features are computed at load time):

- ``dynamic``: float16 ``(days, len(DYNAMIC_CHANNELS), H, W)``, NaN kept where the source is missing.
- ``static``: float16 ``(len(STATIC_CHANNELS), H, W)``, taken from the first day.
- ``fire_hour``: uint8 ``(days, H, W)``, UTC hour of the last detection that day, ``NO_FIRE`` otherwise.
- ``dates``: ISO dates, one per day, sorted.
- ``meta``: JSON string with provenance and conversion checks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.io import MemoryFile

from fireline.data.schema import (
    CHANNEL_INDEX,
    DYNAMIC_CHANNELS,
    FIRE_CHANNEL,
    FORMAT_VERSION,
    N_SOURCE_CHANNELS,
    NO_FIRE,
    STATIC_CHANNELS,
)

_DATE_FILE = re.compile(r"(\d{4}-\d{2}-\d{2})\.tif$")


@dataclass
class FireRecord:
    fire_id: str
    year: int
    dates: list[str]
    dynamic: np.ndarray
    static: np.ndarray
    fire_hour: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fire_mask(self) -> np.ndarray:
        """Binary active-fire mask ``(days, H, W)``."""
        return self.fire_hour != NO_FIRE

    def channel(self, name: str) -> np.ndarray:
        """Float32 ``(days, H, W)`` for a dynamic channel or ``(H, W)`` for a static one."""
        if name in DYNAMIC_CHANNELS:
            return self.dynamic[:, DYNAMIC_CHANNELS.index(name)].astype(np.float32)
        if name in STATIC_CHANNELS:
            return self.static[STATIC_CHANNELS.index(name)].astype(np.float32)
        raise KeyError(name)


def read_geotiff(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the float32 ``(23, H, W)`` array and georeferencing metadata of one daily GeoTIFF."""
    with MemoryFile(data) as memfile, memfile.open() as dataset:
        array = dataset.read().astype(np.float32, copy=False)
        geo = {
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": list(dataset.transform)[:6] if dataset.crs else None,
            "band_descriptions": list(dataset.descriptions),
        }
    if array.shape[0] != N_SOURCE_CHANNELS:
        raise ValueError(f"Expected {N_SOURCE_CHANNELS} bands, got {array.shape[0]}.")
    return array, geo


def fire_hours(active_fire: np.ndarray) -> np.ndarray:
    """Convert hhmm detection times (NaN = none) to uint8 UTC hours with ``NO_FIRE`` for no detection."""
    detected = np.isfinite(active_fire)
    hhmm = np.where(detected, active_fire, 0.0)
    if detected.any() and (hhmm[detected].min() < 0 or hhmm[detected].max() > 2359):
        raise ValueError("Active-fire values outside the hhmm range 0..2359.")
    hours = np.full(active_fire.shape, NO_FIRE, dtype=np.uint8)
    hours[detected] = (hhmm[detected] // 100).astype(np.uint8)
    return hours


def convert_fire(fire_id: str, year: int, daily_tifs: Iterable[tuple[str, bytes]], source: str) -> FireRecord:
    """Build a ``FireRecord`` from ``(member_name, tif_bytes)`` pairs for one fire."""
    named: list[tuple[str, bytes]] = []
    for name, data in daily_tifs:
        match = _DATE_FILE.search(name)
        if match is None:
            raise ValueError(f"Cannot parse a date from {name}.")
        named.append((match.group(1), data))
    if not named:
        raise ValueError(f"No images for fire {fire_id}.")
    named.sort(key=lambda item: item[0])
    dates = [date for date, _ in named]
    if len(set(dates)) != len(dates):
        raise ValueError(f"Fire {fire_id} has duplicate dates.")

    # Decode day by day into one preallocated array (peak memory: one float32 stack plus the source bytes).
    stack: np.ndarray | None = None
    geo: dict[str, Any] | None = None
    for i, (_, data) in enumerate(named):
        array, day_geo = read_geotiff(data)
        if stack is None:
            stack = np.empty((len(named), *array.shape), dtype=np.float32)
            geo = day_geo
        elif array.shape != stack.shape[1:]:
            raise ValueError(f"Fire {fire_id} has images of different shapes: {stack.shape[1:]} and {array.shape}.")
        stack[i] = array
    assert stack is not None
    dynamic_idx = [CHANNEL_INDEX[c] for c in DYNAMIC_CHANNELS]
    static_idx = [CHANNEL_INDEX[c] for c in STATIC_CHANNELS]
    static_all = stack[:, static_idx]
    with np.errstate(invalid="ignore"):
        static_drift = np.nanmax(np.abs(static_all - static_all[:1]), axis=(0, 2, 3))
    meta = {
        "format_version": FORMAT_VERSION,
        "source": source,
        "dynamic_channels": list(DYNAMIC_CHANNELS),
        "static_channels": list(STATIC_CHANNELS),
        "static_max_abs_change_across_days": dict(zip(STATIC_CHANNELS, map(float, static_drift), strict=True)),
        **(geo or {}),
    }
    return FireRecord(
        fire_id=fire_id,
        year=year,
        dates=dates,
        dynamic=stack[:, dynamic_idx].astype(np.float16),
        static=stack[0, static_idx].astype(np.float16),
        fire_hour=fire_hours(stack[:, CHANNEL_INDEX[FIRE_CHANNEL]]),
        meta=meta,
    )


def save_fire(record: FireRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**record.meta, "fire_id": record.fire_id, "year": record.year}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        dynamic=record.dynamic,
        static=record.static,
        fire_hour=record.fire_hour,
        dates=np.array(record.dates),
        meta=np.array(json.dumps(meta)),
    )
    tmp.replace(path)


def load_fire(path: Path) -> FireRecord:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return FireRecord(
            fire_id=meta["fire_id"],
            year=int(meta["year"]),
            dates=[str(d) for d in data["dates"]],
            dynamic=data["dynamic"],
            static=data["static"],
            fire_hour=data["fire_hour"],
            meta=meta,
        )

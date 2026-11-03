"""Deterministic imagery-only sample selection; never selects by prediction quality."""

from __future__ import annotations

from datetime import date

import numpy as np

from fireline.data.convert import FireRecord
from fireline.data.schema import DYNAMIC_CHANNELS

MIN_COVERAGE = 0.5
OPTICAL = ("ndvi", "evi2", "viirs_m11", "viirs_i2", "viirs_i1")
MODEL_DAY_RULE = (
    "timeline day nearest in calendar days to the filmstrip's median-F1 typical day; "
    "each of five input days must have at least 50% finite pixels in every NDVI, EVI2 and VIIRS reflectance channel "
    "inside the timeline crop; today's ERC, max temperature, wind speed/direction and elevation must also have "
    "at least 50% finite pixels each; ties go to the earlier date; fail if none qualifies. "
    "Finite coverage is not a cloud-quality guarantee. The threshold explainer stays on the typical day."
)


def select_model_day(record: FireRecord, targets: list[int], typical: int, crop_box: list[int]) -> int:
    """Nearest eligible five-day window, using only input availability, not its forecast scores."""
    y0, y1, x0, x1 = crop_box

    def covered(t: int, name: str) -> bool:
        values = record.dynamic[t, DYNAMIC_CHANNELS.index(name), y0:y1, x0:x1]
        return bool(np.isfinite(values).mean() >= MIN_COVERAGE)

    elevation = record.channel("elevation")[y0:y1, x0:x1]
    if np.isfinite(elevation).mean() < MIN_COVERAGE:
        raise ValueError("No model-view sample has sufficient elevation coverage in the timeline crop.")
    reference = date.fromisoformat(record.dates[typical])
    ordered = sorted(targets, key=lambda t: (abs((date.fromisoformat(record.dates[t]) - reference).days), record.dates[t]))
    for t in ordered:
        if t < 4 or t + 1 >= len(record.dates):
            continue
        if not all(covered(d, name) for d in range(t - 4, t + 1) for name in OPTICAL):
            continue
        if all(covered(t, name) for name in ("energy_release_component", "temperature_max", "wind_speed", "wind_direction")):
            return t
    raise ValueError("No timeline day has sufficient optical and weather coverage for the model view.")

"""Wind and distance features derived from each day's fire mask (computed after cropping/augmentation).

Per input day, appended after the standard dynamic features (``WIND_FEATURES``):

- ``fire_distance``: distance in pixels to that day's nearest fire pixel, capped at ``DISTANCE_CAP`` and scaled to
  [0, 1] (1 = at or beyond the cap, or no fire that day).
- ``fire_downwind_2/4/8``: that day's fire mask shifted 2, 4 and 8 px (0.75, 1.5, 3 km) in the direction the wind blows
  towards, using the image-mean wind of that day (GRIDMET resolves about 4.6 km, so the wind is nearly uniform within
  an image). Zero where the shift brings in pixels from outside the image.
- ``wind_upslope``: per-pixel dot product of the wind vector with the uphill direction (positive when wind and slope
  push the same way).

Inputs are the standardised arrays: vector features have zero mean and one shared scale per pair, so directions and
dot products keep their meaning, and flips/rotations applied before this step stay physically consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES

WIND_FEATURES: tuple[str, ...] = (
    "fire_distance",
    "fire_downwind_2",
    "fire_downwind_4",
    "fire_downwind_8",
    "wind_upslope",
)
DISTANCE_CAP = 20.0
SHIFTS = (2, 4, 8)
_FIRE, _U, _V = (DYNAMIC_FEATURES.index(n) for n in ("active_fire", "wind_u", "wind_v"))
_DOWN_E, _DOWN_N = STATIC_FEATURES.index("downslope_east"), STATIC_FEATURES.index("downslope_north")


def shift_mask(mask: np.ndarray, east: float, north: float, distance: int) -> np.ndarray:
    """Shift a 2-D mask ``distance`` pixels towards the (east, north) direction; row 0 is north."""
    length = float(np.hypot(east, north))
    if length < 1e-6:
        return np.zeros_like(mask)
    dx = round(distance * east / length)
    dy = round(-distance * north / length)  # north is up: negative row offset
    out = np.zeros_like(mask)
    h, w = mask.shape
    src = mask[max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)]
    out[max(dy, 0) : h - max(-dy, 0), max(dx, 0) : w - max(-dx, 0)] = src
    return out


def wind_features(dynamic: np.ndarray, static: np.ndarray) -> np.ndarray:
    """``dynamic`` (T, C, H, W) and ``static`` (Cs, H, W), standardised -> (T, len(WIND_FEATURES), H, W) float32."""
    days, _, height, width = dynamic.shape
    out = np.empty((days, len(WIND_FEATURES), height, width), dtype=np.float32)
    up_e, up_n = -static[_DOWN_E], -static[_DOWN_N]
    for d in range(days):
        fire = dynamic[d, _FIRE] > 0.5
        if fire.any():
            distance = distance_transform_edt(~fire)
            out[d, 0] = np.minimum(distance, DISTANCE_CAP) / DISTANCE_CAP
        else:
            out[d, 0] = 1.0
        u, v = dynamic[d, _U], dynamic[d, _V]
        mean_u, mean_v = float(u.mean()), float(v.mean())
        for k, step in enumerate(SHIFTS, start=1):
            out[d, k] = shift_mask(fire, mean_u, mean_v, step)
        out[d, 4] = u * up_e + v * up_n
    return out

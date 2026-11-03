"""Fire-history channels: what burned before the 5-day input window, computed from a fire's daily detections.

Per day d, from the binary active-fire masks of days 0..d of the fire (no later day is read):

- ``burned_before``: 1 where the pixel has been detected burning on any day up to d, else 0.
- ``days_since_fire``: days since the pixel was last detected burning (0 = burning on d), divided by ``CAP_DAYS`` and
  capped at 1; 1 where it never burned.

The input window shows only the last ``history`` days; these channels add the fire's longer past (fuel already burned,
pixels that burned and may flare up again). Both are spatially local, so they commute with flips and rotations.
"""

from __future__ import annotations

import numpy as np

HISTORY_FEATURES = ("burned_before", "days_since_fire")
CAP_DAYS = 30


def fire_history(fire: np.ndarray) -> np.ndarray:
    """``fire`` (D, H, W) binary masks of days 0..D-1 -> (D, 2, H, W) float32 channels for every day."""
    fire = np.asarray(fire).astype(bool)
    days = fire.shape[0]
    out = np.empty((days, len(HISTORY_FEATURES), *fire.shape[1:]), dtype=np.float32)
    burned = np.zeros(fire.shape[1:], dtype=bool)
    since = np.full(fire.shape[1:], CAP_DAYS, dtype=np.int32)
    for d in range(days):
        burned |= fire[d]
        since = np.where(fire[d], 0, np.minimum(since + 1, CAP_DAYS))
        since[~burned] = CAP_DAYS
        out[d, 0] = burned
        out[d, 1] = since / CAP_DAYS
    return out

"""Where does tomorrow's far-away fire come from? Classify fire pixels that appear far from today's detections.

For every evaluation target (day t -> t + 1, at least 5 earlier days) each fire pixel of day t + 1 gets a distance ring
(square/Chebyshev distance to today's nearest fire pixel; 1 px = 375 m, same rings as ``distance_profile``). Pixels
more than ``far_px`` from today's fire, or in an image with no fire today, are "far" and get explanations, which can
overlap:

- ``recent_fire``: the pixel (or a neighbour within ``memory_px``) burned on one of the previous ``memory_days`` days,
  so today's absence is plausibly a missed detection or a short pause (re-flare).
- ``unobserved_today``: today's VIIRS reflectance is missing at the pixel (no clear observation; clouds or gaps).
- ``downwind``: the direction from today's nearest fire pixel to the pixel lies within 60 degrees of the observed
  wind direction of day t (wind speed at least ``min_wind`` m/s), consistent with a wind-driven run or spotting.
- ``isolated``: none of the above (new ignition, or a cause these inputs cannot see).

A priority label (recent_fire > unobserved_today > downwind > isolated) gives a partition for the headline shares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from fireline.baselines.persistence import box_mean, fire_paths
from fireline.data.convert import load_fire
from fireline.data.inputs import DYNAMIC_FEATURES, raw_dynamic
from fireline.data.samples import target_days

RING_NAMES = ("on fire today", "1-2 px", "3-5 px", "6-10 px", "11-20 px", ">20 px", "no fire today")
RING_LIMITS = (0, 2, 5, 10, 20)
CATEGORIES = ("recent_fire", "unobserved_today", "downwind", "isolated")


FAR_PX = RING_LIMITS[2]  # "far": more than 5 px from today's fire (square distance), or no fire today


def far_region(today: np.ndarray) -> np.ndarray:
    """Pixels more than ``FAR_PX`` from today's fire (square distance); everything when there is no fire today.

    Equal to ``ring_map(today) >= 3``.
    """
    if not today.any():
        return np.ones(today.shape, dtype=bool)
    return ~(box_mean(today, FAR_PX) > 0)


def ring_map(today: np.ndarray) -> np.ndarray:
    """Ring index per pixel (0..6, see ``RING_NAMES``) from today's binary fire mask (square distance)."""
    rings = np.full(today.shape, 5, dtype=np.int8)
    if not today.any():
        return np.full(today.shape, 6, dtype=np.int8)
    assigned = np.zeros(today.shape, dtype=bool)
    for index, radius in enumerate(RING_LIMITS):
        within = today if radius == 0 else box_mean(today, radius) > 0
        ring = within & ~assigned
        rings[ring] = index
        assigned |= ring
    return rings


def explain_far(
    mask: np.ndarray, t: int, wind_u: np.ndarray, wind_v: np.ndarray, observed: np.ndarray, far: np.ndarray,
    memory_days: int = 4, memory_px: int = 1, min_wind: float = 1.0,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Boolean maps (H, W) of each explanation, restricted to ``far`` fire pixels of day t + 1."""
    today = mask[t]
    target = mask[t + 1] & far
    past = mask[max(t - memory_days, 0) : t].any(axis=0)
    recent = target & (box_mean(past, memory_px) > 0 if memory_px else past)
    unobserved = target & ~observed
    downwind = np.zeros_like(target)
    if today.any():
        _, (rows, cols) = distance_transform_edt(~today, return_indices=True)
        grid_r, grid_c = np.indices(today.shape)
        east, north = (grid_c - cols).astype(float), (rows - grid_r).astype(float)  # row 0 is north
        length = np.hypot(east, north)
        speed = np.hypot(wind_u, wind_v)
        cosine = (east * wind_u + north * wind_v) / np.maximum(length * speed, 1e-9)
        downwind = target & (speed >= min_wind) & (cosine >= 0.5) & (length > 0)
    isolated = target & ~recent & ~unobserved & ~downwind
    return {"recent_fire": recent, "unobserved_today": unobserved, "downwind": downwind, "isolated": isolated}


def diagnose(paths: list[Path], far_px: int = 5) -> dict:
    u_index, v_index = DYNAMIC_FEATURES.index("wind_u"), DYNAMIC_FEATURES.index("wind_v")
    i2_index = DYNAMIC_FEATURES.index("viirs_i2")
    ring_fire = np.zeros(len(RING_NAMES), dtype=np.int64)
    any_counts = dict.fromkeys(CATEGORIES, 0)
    priority_counts = dict.fromkeys(CATEGORIES, 0)
    by_ring = {name: dict.fromkeys(CATEGORIES, 0) for name in RING_NAMES[3:]}
    far_total = 0
    for i, path in enumerate(paths, 1):
        record = load_fire(path)
        mask = record.fire_mask
        for t in target_days(mask.shape[0], history=1):
            rings = ring_map(mask[t])
            tomorrow = mask[t + 1]
            ring_fire += np.bincount(rings[tomorrow], minlength=len(RING_NAMES))
            far = (rings >= 3) if far_px == 5 else (rings >= next(k for k, r in enumerate(RING_LIMITS) if r > far_px))
            if not (tomorrow & far).any():
                continue
            day = raw_dynamic(record, [t])[0]
            explained = explain_far(mask, t, day[u_index], day[v_index], np.isfinite(day[i2_index]), far)
            target = tomorrow & far
            far_total += int(target.sum())
            remaining = target.copy()
            for name in CATEGORIES:
                any_counts[name] += int(explained[name].sum())
                chosen = explained[name] & remaining
                priority_counts[name] += int(chosen.sum())
                remaining &= ~chosen
                for ring_index, ring_name in enumerate(RING_NAMES[3:], start=3):
                    by_ring[ring_name][name] += int((chosen & (rings == ring_index)).sum())
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] fires", flush=True)
    total = int(ring_fire.sum())
    return {
        "definition": __doc__,
        "far_px": far_px,
        "fire_pixels_tomorrow": total,
        "share_by_ring": {n: int(c) / max(total, 1) for n, c in zip(RING_NAMES, ring_fire, strict=True)},
        "far_fire_pixels": far_total,
        "far_share": far_total / max(total, 1),
        "far_priority_shares": {k: v / max(far_total, 1) for k, v in priority_counts.items()},
        "far_any_shares": {k: v / max(far_total, 1) for k, v in any_counts.items()},
        "far_priority_counts_by_ring": by_ring,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"{args.report} exists; refusing to overwrite.")
    paths = fire_paths(args.data, args.years)
    print(f"far-fire diagnosis: {len(paths)} fires from {args.years}", flush=True)
    report = {"years": args.years, **diagnose(paths)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "definition"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

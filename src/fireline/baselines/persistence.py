"""Training-free baselines and the normalisation-statistics job.

    python -m fireline.baselines.persistence evaluate --data DIR --years 2020 --report OUT.json
    python -m fireline.baselines.persistence stats --data DIR --years 2018 2019 --output stats.json
    python -m fireline.baselines.persistence distance --data DIR --years 2020 --report OUT.json

Baselines, both using only day t's active fire to predict day t + 1:
- ``persistence``: tomorrow's fire = today's fire (binary score).
- ``neighbourhood``: fraction of today's fire pixels within a (2r+1) square around each pixel (r = 2, about 1.9 km),
  a soft score that also ranks pixels next to the fire.
Targets follow ``fireline.data.samples.target_days`` (at least 5 earlier days), the same set every model uses.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fireline.data.convert import load_fire
from fireline.data.inputs import StatsAccumulator
from fireline.data.samples import target_days
from fireline.eval.metrics import PixelMetrics


def box_mean(mask: np.ndarray, radius: int) -> np.ndarray:
    """Mean of a 2-D array over a (2r+1)^2 window, edge pixels averaged over the in-image part of the window."""
    padded = np.pad(mask.astype(np.float64), radius)
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * radius + 1
    window_sum = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    ones = np.pad(np.ones(mask.shape), radius)
    count_integral = np.pad(ones.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = count_integral[k:, k:] - count_integral[:-k, k:] - count_integral[k:, :-k] + count_integral[:-k, :-k]
    return window_sum / counts


def fire_paths(data: Path, years: list[int]) -> list[Path]:
    return sorted(p for y in years for p in (data / str(y)).glob("fire_*.npz") if not p.name.endswith(".tmp.npz"))


def evaluate(paths: list[Path], radius: int = 2, min_history: int = 5) -> dict:
    metrics = {"persistence": PixelMetrics(), "neighbourhood": PixelMetrics()}
    samples = 0
    for i, path in enumerate(paths, 1):
        mask = load_fire(path).fire_mask
        for t in target_days(mask.shape[0], history=1, min_history=min_history):
            today, tomorrow = mask[t], mask[t + 1]
            metrics["persistence"].update(today.astype(np.float64), tomorrow)
            metrics["neighbourhood"].update(box_mean(today, radius), tomorrow)
            samples += 1
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] fires", flush=True)
    return {"fires": len(paths), "samples": samples, "radius": radius, "min_history": min_history,
            **{name: m.compute() for name, m in metrics.items()}}  # fmt: skip


RINGS = ((0, "on fire today"), (2, "1-2 px"), (5, "3-5 px"), (10, "6-10 px"), (20, "11-20 px"))


def distance_profile(paths: list[Path], min_history: int = 5) -> dict:
    """Where tomorrow's fire appears relative to today's: pixel and fire counts per square (Chebyshev) distance ring.

    Ring k contains pixels whose nearest fire pixel today is within the ring's distance in the max-norm (a square
    neighbourhood), beyond the previous ring; ">20 px" is everything farther, "no fire today" covers whole images
    without a detection on day t. 1 px = 375 m.
    """
    names = [name for _, name in RINGS] + [">20 px", "no fire today"]
    pixels = dict.fromkeys(names, 0)
    fire = dict.fromkeys(names, 0)
    for i, path in enumerate(paths, 1):
        mask = load_fire(path).fire_mask
        for t in target_days(mask.shape[0], history=1, min_history=min_history):
            today, tomorrow = mask[t], mask[t + 1]
            if not today.any():
                pixels["no fire today"] += today.size
                fire["no fire today"] += int(tomorrow.sum())
                continue
            assigned = np.zeros(today.shape, dtype=bool)
            for radius, name in RINGS:
                within = today if radius == 0 else box_mean(today, radius) > 0
                ring = within & ~assigned
                pixels[name] += int(ring.sum())
                fire[name] += int(tomorrow[ring].sum())
                assigned |= ring
            pixels[">20 px"] += int((~assigned).sum())
            fire[">20 px"] += int(tomorrow[~assigned].sum())
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] fires", flush=True)
    total_fire = sum(fire.values())
    return {
        "rings": {
            n: {
                "pixels": pixels[n],
                "fire_tomorrow": fire[n],
                "fire_rate": fire[n] / max(pixels[n], 1),
                "share_of_tomorrows_fire": fire[n] / max(total_fire, 1),
            }
            for n in names
        },
        "fire_tomorrow_total": total_fire,
        "share_beyond_5px": sum(fire[n] for n in names[3:]) / max(total_fire, 1),
        "distance": "Chebyshev (square neighbourhood), 1 px = 375 m",
    }


def compute_stats(paths: list[Path], years: list[int]) -> StatsAccumulator:
    accumulator = StatsAccumulator()
    for i, path in enumerate(paths, 1):
        accumulator.add(load_fire(path))
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] fires", flush=True)
    return accumulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--years", type=int, nargs="+", required=True)
    ev.add_argument("--radius", type=int, default=2)
    ev.add_argument("--report", type=Path, required=True)
    di = sub.add_parser("distance")
    di.add_argument("--data", type=Path, required=True)
    di.add_argument("--years", type=int, nargs="+", required=True)
    di.add_argument("--report", type=Path, required=True)
    st = sub.add_parser("stats")
    st.add_argument("--data", type=Path, required=True)
    st.add_argument("--years", type=int, nargs="+", required=True)
    st.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output if args.command == "stats" else args.report
    if target.exists():
        raise FileExistsError(f"{target} exists; refusing to overwrite.")
    start = time.perf_counter()
    paths = fire_paths(args.data, args.years)
    print(f"{args.command}: {len(paths)} fires from {args.years}", flush=True)
    if args.command in ("evaluate", "distance"):
        body = evaluate(paths, radius=args.radius) if args.command == "evaluate" else distance_profile(paths)
        result = {"years": args.years, **body}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    else:
        stats = compute_stats(paths, args.years).result(tuple(args.years))
        stats.to_json(target)
        print(target.read_text(), flush=True)
    print(f"done in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Does the WSTS composite NDVI/EVI2 channel contain information from after the prediction day?

WSTS stores, for day t, the last 16-day vegetation composite whose window STARTS within the previous 15 days, so
the window always reaches 8-15 days past t. If observations from inside that future window enter the
composite, vegetation should drop at a composite update in pixels that are going to burn during the window, before
they burn.

For every fire and every composite update day ``u`` (index into the fire's consecutive daily series), the change
``composite[u] - composite[u - 1]`` is collected for pixel groups defined by the pixel's first active-fire day ``b``:

- ``already_burned``: b < u (a drop here is legitimate: the fire happened before the update)
- ``burns_on_update_day``: b == u
- ``burns_in_window``: u < b <= u + 15 (a drop here, before burning, indicates future information)
- ``burns_after_window``: b > u + 15 (control: the window does not cover the burn)
- ``never_near``: never burns, within ``near_px`` pixels of the fire's eventual footprint (spatial control)
- ``never_far``: never burns, farther away

Changes are accumulated as histograms (bin width 0.005 in index units), from which medians and the AUC of
"larger drop" for ``burns_in_window`` against each control are computed. Run per year with a process pool.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fireline.data.convert import FireRecord, load_fire
from fireline.data.features import composite_update_days

GROUPS = ("already_burned", "burns_on_update_day", "burns_in_window", "burns_after_window", "never_near", "never_far")
CONTROLS = ("never_near", "never_far", "burns_after_window")
CHANNELS = ("ndvi", "evi2")
WINDOW_DAYS = 15  # a 16-day composite starting on day u covers u .. u + 15
BIN_WIDTH = 0.005
EDGES = np.arange(-2.0, 2.0 + BIN_WIDTH / 2, BIN_WIDTH)
SCALE = 1e4  # VNP13A1 values are stored x 1e4


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of a 2-D mask with a (2r+1)x(2r+1) square, using shifted ORs (no scipy)."""
    out = mask.copy()
    padded = np.pad(mask, radius)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def first_burn_day(fire_mask: np.ndarray) -> np.ndarray:
    """Index of each pixel's first active-fire day, -1 if it never burns."""
    burned = fire_mask.any(axis=0)
    return np.where(burned, fire_mask.argmax(axis=0), -1)


def pixel_groups(first: np.ndarray, update: int, near: np.ndarray) -> dict[str, np.ndarray]:
    never = first < 0
    return {
        "already_burned": (first >= 0) & (first < update),
        "burns_on_update_day": first == update,
        "burns_in_window": (first > update) & (first <= update + WINDOW_DAYS),
        "burns_after_window": first > update + WINDOW_DAYS,
        "never_near": never & near,
        "never_far": never & ~near,
    }


def empty_histograms() -> dict[str, dict[str, np.ndarray]]:
    return {c: {g: np.zeros(len(EDGES) - 1, dtype=np.int64) for g in GROUPS} for c in CHANNELS}


def fire_histograms(record: FireRecord, near_px: int) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Histograms of composite changes at update days for one fire, plus a small per-fire summary."""
    hist = empty_histograms()
    first = first_burn_day(record.fire_mask)
    near = dilate(first >= 0, near_px)
    updates: list[int] = []
    for channel in CHANNELS:
        values = record.channel(channel)
        days = [u for u in np.flatnonzero(composite_update_days(values)) if u > 0]
        if channel == "ndvi":
            updates = [int(u) for u in days]
        for u in days:
            delta = (values[u] - values[u - 1]) / SCALE
            valid = np.isfinite(delta)
            for group, mask in pixel_groups(first, int(u), near).items():
                selected = delta[mask & valid]
                if selected.size:
                    hist[channel][group] += np.histogram(np.clip(selected, EDGES[0], EDGES[-1]), EDGES)[0]
    summary = {
        "fire_id": record.fire_id,
        "year": record.year,
        "updates": len(updates),
        "counts": {g: int(hist["ndvi"][g].sum()) for g in GROUPS},
        "median_ndvi_change": {g: histogram_median(hist["ndvi"][g]) for g in GROUPS},
    }
    return hist, summary


def histogram_median(counts: np.ndarray) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    index = int(np.searchsorted(np.cumsum(counts), (total + 1) / 2))
    return float((EDGES[index] + EDGES[index + 1]) / 2)


def histogram_mean(counts: np.ndarray) -> float | None:
    total = int(counts.sum())
    return float((counts * (EDGES[:-1] + EDGES[1:]) / 2).sum() / total) if total else None


def drop_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """P(a positive pixel's change is lower than a negative pixel's), ties within a bin counted as one half."""
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if not n_pos or not n_neg:
        return None
    higher_negatives = n_neg - np.cumsum(negative)  # negatives in strictly higher bins
    wins = (positive * higher_negatives).sum() + 0.5 * (positive * negative).sum()
    return float(wins / (n_pos * n_neg))


def summarize(hist: dict[str, dict[str, np.ndarray]]) -> dict:
    out: dict = {}
    for channel in CHANNELS:
        groups = {
            g: {"pixels": int(h.sum()), "median_change": histogram_median(h), "mean_change": histogram_mean(h)}
            for g, h in hist[channel].items()
        }
        auc = {
            f"burns_in_window_vs_{c}": drop_auc(hist[channel]["burns_in_window"], hist[channel][c]) for c in CONTROLS
        }
        out[channel] = {"groups": groups, "auc_larger_drop": auc}
    return out


def _process(path: str, near_px: int) -> tuple[int, dict[str, dict[str, np.ndarray]], dict]:
    record = load_fire(Path(path))
    hist, summary = fire_histograms(record, near_px)
    return record.year, hist, summary


def analyze(paths: Iterable[Path], near_px: int = 3, workers: int = 1) -> dict:
    by_year: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    total = empty_histograms()
    fires: list[dict] = []
    paths = sorted(paths)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (year, hist, summary) in enumerate(pool.map(_process, map(str, paths), [near_px] * len(paths)), 1):
            year_hist = by_year.setdefault(year, empty_histograms())
            for channel in CHANNELS:
                for group in GROUPS:
                    year_hist[channel][group] += hist[channel][group]
                    total[channel][group] += hist[channel][group]
            fires.append(summary)
            if i % 50 == 0 or i == len(paths):
                print(f"[{i}/{len(paths)}] fires processed", flush=True)
    in_window_lower = [
        f
        for f in fires
        if f["median_ndvi_change"]["burns_in_window"] is not None
        and f["median_ndvi_change"]["never_near"] is not None
        and f["median_ndvi_change"]["burns_in_window"] < f["median_ndvi_change"]["never_near"]
    ]
    comparable = [
        f
        for f in fires
        if None not in (f["median_ndvi_change"]["burns_in_window"], f["median_ndvi_change"]["never_near"])
    ]
    return {
        "definition": __doc__,
        "near_px": near_px,
        "window_days": WINDOW_DAYS,
        "bin_width": BIN_WIDTH,
        "fires": len(fires),
        "overall": summarize(total),
        "by_year": {str(y): summarize(h) for y, h in sorted(by_year.items())},
        "fires_with_burns_in_window_and_near_control": len(comparable),
        "fires_where_burns_in_window_median_drop_exceeds_never_near": len(in_window_lower),
        "per_fire": fires,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Directory with <year>/<fire>.npz files.")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--near-px", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True, help="JSON report path (must not exist).")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"{args.report} exists; refusing to overwrite.")
    paths = [p for y in args.years for p in (args.data / str(y)).glob("fire_*.npz") if not p.name.endswith(".tmp.npz")]
    print(f"Analysing {len(paths)} fires with {args.workers} workers", flush=True)
    report = analyze(paths, near_px=args.near_px, workers=args.workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("fires", "overall")}, indent=2), flush=True)
    print(
        f"fires where in-window pixels drop more than never-burned nearby pixels: "
        f"{report['fires_where_burns_in_window_median_drop_exceeds_never_near']} / "
        f"{report['fires_with_burns_in_window_and_near_control']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

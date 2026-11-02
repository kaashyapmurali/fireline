"""Validate a directory of converted fires against the source zip listing and summarise it.

    uv run python -m fireline.data.validate --out datasets/processed/wsts --years 2019 --report validation.json

Checks per fire: loads, array shapes agree, dates sorted and unique and equal to the zip's dates for that fire,
fire hours in 0..23 or NO_FIRE, static channels constant across days. Checks per year: every fire in the zip is
present and no extra fires exist; no temporary files remain. Exits with an error if any check fails.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np

from fireline.data.convert import load_fire
from fireline.data.ingest import FireEntry, list_fires
from fireline.data.remote_zip import RemoteZip
from fireline.data.schema import DYNAMIC_CHANNELS, NO_FIRE, ZENODO_WSTS_URL

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})\.tif$")


def check_fire(path: Path, entry: FireEntry) -> tuple[list[str], dict]:
    """Return (problems, statistics) for one converted fire."""
    problems: list[str] = []
    record = load_fire(path)
    days, channels, height, width = record.dynamic.shape
    expected_dates = sorted(_DATE.search(m).group(1) for m in entry.members)  # type: ignore[union-attr]
    if record.dates != expected_dates:
        problems.append("dates differ from the zip listing")
    if channels != len(DYNAMIC_CHANNELS):
        problems.append(f"{channels} dynamic channels")
    if record.static.shape[1:] != (height, width) or record.fire_hour.shape != (days, height, width):
        problems.append("array shapes disagree")
    hours = np.unique(record.fire_hour)
    if np.any((hours > 23) & (hours != NO_FIRE)):
        problems.append("fire hour out of range")
    drift = {k: v for k, v in record.meta.get("static_max_abs_change_across_days", {}).items() if v not in (0.0,)}
    ordinals = [date.fromisoformat(d).toordinal() for d in record.dates]
    stats = {
        "days": days,
        "height": height,
        "width": width,
        "bytes": path.stat().st_size,
        "fire_pixels": int(record.fire_mask.sum()),
        "pixels": days * height * width,
        "fire_days": int(record.fire_mask.any(axis=(1, 2)).sum()),
        "date_gaps": int(sum(b - a > 1 for a, b in pairwise(ordinals))),
        "nan_fraction": [float(np.isnan(record.dynamic[:, c]).mean()) for c in range(channels)],
        "static_drift": drift,
        "georeferenced": record.meta.get("crs") is not None,
    }
    return problems, stats


def validate(out_dir: Path, fires: dict[tuple[int, str], FireEntry], years: list[int]) -> dict:
    problems: dict[str, list[str]] = defaultdict(list)
    per_year: dict[int, dict] = {}
    nan_sum = np.zeros(len(DYNAMIC_CHANNELS))
    nan_weight = 0
    leftovers = [str(p) for p in out_dir.rglob("*.tmp.npz")]
    if leftovers:
        problems["temporary files"] = leftovers
    for year in years:
        expected = {fire_id: e for (y, fire_id), e in fires.items() if y == year}
        found = {p.stem: p for p in (out_dir / str(year)).glob("fire_*.npz") if not p.name.endswith(".tmp.npz")}
        missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
        if missing:
            problems[f"{year} missing"] = missing
        if extra:
            problems[f"{year} extra"] = extra
        totals = defaultdict(int)
        drifted, gaps = [], 0
        for fire_id in sorted(set(expected) & set(found)):
            fire_problems, stats = check_fire(found[fire_id], expected[fire_id])
            if fire_problems:
                problems[f"{year}/{fire_id}"] = fire_problems
            for key in ("days", "bytes", "fire_pixels", "pixels", "fire_days"):
                totals[key] += stats[key]
            gaps += stats["date_gaps"] > 0
            if stats["static_drift"]:
                drifted.append(fire_id)
            nan_sum += np.array(stats["nan_fraction"]) * stats["pixels"]
            nan_weight += stats["pixels"]
        per_year[year] = {
            "fires_expected": len(expected),
            "fires_found": len(found),
            "images": totals["days"],
            "images_expected": sum(len(e.members) for e in expected.values()),
            "mb": round(totals["bytes"] / 1e6, 1),
            "fire_pixel_fraction": totals["fire_pixels"] / max(totals["pixels"], 1),
            "fire_days": totals["fire_days"],
            "fires_with_date_gaps": gaps,
            "fires_with_static_drift": drifted,
        }
        if per_year[year]["images"] != per_year[year]["images_expected"]:
            problems[f"{year} image count"] = [f"{per_year[year]['images']} != {per_year[year]['images_expected']}"]
    return {
        "ok": not problems,
        "problems": dict(problems),
        "years": per_year,
        "total_mb": round(sum(y["mb"] for y in per_year.values()), 1),
        "nan_fraction_by_channel": dict(
            zip(DYNAMIC_CHANNELS, (nan_sum / max(nan_weight, 1)).round(5).tolist(), strict=True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=ZENODO_WSTS_URL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True, help="JSON report path (must not exist).")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"{args.report} exists; refusing to overwrite.")
    summary = validate(args.out, list_fires(RemoteZip.from_url(args.url)), args.years)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "nan_fraction_by_channel"}, indent=2), flush=True)
    if not summary["ok"]:
        raise SystemExit("Validation failed; see the report.")


if __name__ == "__main__":
    main()

"""List fires in a WildfireSpreadTS zip and convert selected fires to per-fire ``.npz`` files.

Usage::

    uv run python -m fireline.data.ingest --year 2019 --list
    uv run python -m fireline.data.ingest --year 2019 --fire fire_23160475 --out datasets/processed/wsts
    uv run python -m fireline.data.ingest --year 2019 --first 16 --workers 8 --out OUT --report OUT/report.json

Existing output files are never overwritten: the command fails unless ``--skip-existing`` is given, in which case
fires whose ``.npz`` already exists are skipped (resumable full conversion).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from fireline.data.convert import convert_fire, save_fire
from fireline.data.remote_zip import RemoteZip
from fireline.data.schema import ZENODO_WSTS_URL

_MEMBER = re.compile(r"(?:^|/)(\d{4})/([^/]+)/(\d{4}-\d{2}-\d{2})\.tif$")


@dataclass(frozen=True)
class FireEntry:
    year: int
    fire_id: str
    members: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True)
class FireResult:
    year: int
    fire_id: str
    images: int
    source_bytes: int
    output_bytes: int
    download_seconds: float
    convert_seconds: float


def list_fires(archive: RemoteZip) -> dict[tuple[int, str], FireEntry]:
    """Group the archive's GeoTIFF members by (year, fire id)."""
    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name in archive.members:
        match = _MEMBER.search(name)
        if match:
            grouped[(int(match.group(1)), match.group(2))].append(name)
    return {
        key: FireEntry(key[0], key[1], tuple(sorted(names)), sum(archive.members[n].size for n in names))
        for key, names in sorted(grouped.items())
    }


def select_fires(fires: dict[str, FireEntry], fire_ids: list[str], first: int) -> list[FireEntry]:
    """Explicit ``fire_ids`` (in the given order) plus the first ``first`` remaining fires in sorted order."""
    unknown = [f for f in fire_ids if f not in fires]
    if unknown:
        raise KeyError(f"Unknown fires: {unknown}")
    chosen = [fires[f] for f in fire_ids]
    rest = [entry for fire_id, entry in sorted(fires.items()) if fire_id not in fire_ids]
    return chosen + rest[: max(first, 0)]


def output_path(out_dir: Path, entry: FireEntry) -> Path:
    return out_dir / str(entry.year) / f"{entry.fire_id}.npz"


def convert_entry(archive: RemoteZip, entry: FireEntry, out_dir: Path, source: str) -> FireResult:
    """Fetch (CRC-verified), convert and save one fire."""
    start = time.perf_counter()
    data = archive.read_many(list(entry.members))
    downloaded = time.perf_counter()
    record = convert_fire(entry.fire_id, entry.year, ((n, data[n]) for n in entry.members), source=source)
    path = output_path(out_dir, entry)
    save_fire(record, path)
    return FireResult(
        year=entry.year,
        fire_id=entry.fire_id,
        images=len(entry.members),
        source_bytes=entry.size_bytes,
        output_bytes=path.stat().st_size,
        download_seconds=round(downloaded - start, 3),
        convert_seconds=round(time.perf_counter() - downloaded, 3),
    )


def convert_many(
    archive: RemoteZip, entries: list[FireEntry], out_dir: Path, source: str, workers: int
) -> list[FireResult]:
    """Convert fires with ``workers`` concurrent downloads; raises on the first failure (no retries)."""
    results: list[FireResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_entry, archive, entry, out_dir, source): entry for entry in entries}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            mb_s = result.source_bytes / 1e6 / max(result.download_seconds, 1e-9)
            print(
                f"[{len(results)}/{len(entries)}] {result.year}/{result.fire_id}: {result.source_bytes / 1e6:.1f} MB "
                f"in {result.download_seconds:.1f}s ({mb_s:.2f} MB/s), convert {result.convert_seconds:.1f}s, "
                f"-> {result.output_bytes / 1e6:.2f} MB",
                flush=True,
            )
    return results


def summarize(results: list[FireResult], wall_seconds: float, workers: int) -> dict:
    source = sum(r.source_bytes for r in results)
    per_connection = [r.source_bytes / 1e6 / r.download_seconds for r in results if r.download_seconds > 0]
    return {
        "fires": len(results),
        "images": sum(r.images for r in results),
        "workers": workers,
        "wall_seconds": round(wall_seconds, 2),
        "source_mb": round(source / 1e6, 1),
        "output_mb": round(sum(r.output_bytes for r in results) / 1e6, 2),
        "aggregate_mb_per_s": round(source / 1e6 / wall_seconds, 3) if wall_seconds > 0 else None,
        "median_per_connection_mb_per_s": round(sorted(per_connection)[len(per_connection) // 2], 3)
        if per_connection
        else None,
        "output_mb_per_image": round(
            sum(r.output_bytes for r in results) / 1e6 / max(1, sum(r.images for r in results)), 3
        ),
        "results": [asdict(r) for r in sorted(results, key=lambda r: (r.year, r.fire_id))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=ZENODO_WSTS_URL)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--fire", action="append", default=[], help="Fire id; repeat for several.")
    parser.add_argument("--first", type=int, default=0, help="Also convert the first N other fires of the year.")
    parser.add_argument("--all", action="store_true", help="Convert every fire of the year.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent downloads (one fire per worker).")
    parser.add_argument("--list", action="store_true", help="List fires of the year with image counts and sizes.")
    parser.add_argument("--out", type=Path, default=Path("datasets/processed/wsts"))
    parser.add_argument("--report", type=Path, help="Write a JSON timing and size report here (must not exist).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip fires whose output already exists.")
    args = parser.parse_args()

    archive = RemoteZip.from_url(args.url)
    fires = {fire_id: entry for (year, fire_id), entry in list_fires(archive).items() if year == args.year}
    if args.list:
        for entry in fires.values():
            print(f"{entry.fire_id}\t{len(entry.members)}\t{entry.size_bytes / 1e6:.1f} MB")
        return
    entries = list(fires.values()) if args.all else select_fires(fires, args.fire, args.first)
    existing = [e for e in entries if output_path(args.out, e).exists()]
    if existing and not args.skip_existing:
        raise FileExistsError(f"{len(existing)} outputs exist (e.g. {output_path(args.out, existing[0])}); refusing.")
    if args.report is not None and args.report.exists():
        raise FileExistsError(f"{args.report} exists; refusing to overwrite.")
    todo = [e for e in entries if e not in existing]
    print(f"Converting {len(todo)} fires ({len(existing)} skipped) with {args.workers} workers", flush=True)
    start = time.perf_counter()
    results = convert_many(archive, todo, args.out, args.url, args.workers)
    summary = summarize(results, time.perf_counter() - start, args.workers)
    summary.update(year=args.year, url=args.url, skipped=len(existing))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

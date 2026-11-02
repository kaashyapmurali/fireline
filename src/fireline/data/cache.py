"""Uncompressed, memory-mapped copies of converted fires for fast random access during training.

``load_fire`` must decompress a whole fire to read any day. For training, each fire is decompressed once into
``cache_dir/<year>/<fire_id>/{dynamic,static,fire_hour}.npy`` and opened with ``mmap_mode="r"``, so a sample reads
only its own days and DataLoader workers share the operating system's page cache instead of copying arrays.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fireline.data.convert import FireRecord, load_fire

_ARRAYS = ("dynamic", "static", "fire_hour")


def cache_path(cache_dir: Path, npz: Path) -> Path:
    return cache_dir / npz.parent.name / npz.stem


def materialize(npz: Path, cache_dir: Path) -> Path:
    """Write the uncompressed arrays of one fire (idempotent; complete only once ``meta.json`` exists)."""
    target = cache_path(cache_dir, npz)
    if (target / "meta.json").exists():
        return target
    record = load_fire(npz)
    target.mkdir(parents=True, exist_ok=True)
    for name in _ARRAYS:
        np.save(target / f"{name}.npy", getattr(record, name))
    meta = {**record.meta, "dates": record.dates, "fire_id": record.fire_id, "year": record.year}
    (target / "meta.json").write_text(json.dumps(meta))  # written last: marks the cache entry complete
    return target


def materialize_all(npzs: list[Path], cache_dir: Path, workers: int = 4) -> list[Path]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(materialize, npzs, [cache_dir] * len(npzs)))


def open_cached(target: Path) -> FireRecord:
    meta = json.loads((target / "meta.json").read_text())
    arrays = {name: np.load(target / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
    return FireRecord(meta["fire_id"], int(meta["year"]), list(meta["dates"]), meta=meta, **arrays)

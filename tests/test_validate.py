from __future__ import annotations

from pathlib import Path

import numpy as np
from test_convert import _day, _tif_bytes

from fireline.data.convert import convert_fire, save_fire
from fireline.data.ingest import FireEntry
from fireline.data.validate import validate


def _write_fire(out: Path, fire_id: str, dates: list[str]) -> FireEntry:
    members = tuple(f"2019/{fire_id}/{d}.tif" for d in dates)
    record = convert_fire(fire_id, 2019, [(m, _tif_bytes(_day(i))) for i, m in enumerate(members)], source="t")
    save_fire(record, out / "2019" / f"{fire_id}.npz")
    return FireEntry(2019, fire_id, members, 1)


def test_valid_directory_passes(tmp_path: Path) -> None:
    entry = _write_fire(tmp_path, "fire_1", ["2019-08-01", "2019-08-02", "2019-08-04"])
    report = validate(tmp_path, {(2019, "fire_1"): entry}, [2019])
    assert report["ok"], report["problems"]
    year = report["years"][2019]
    assert year["images"] == 3 and year["fires_with_date_gaps"] == 1
    assert 0 < year["fire_pixel_fraction"] < 1
    assert np.isclose(report["nan_fraction_by_channel"]["ndvi"], 1 / 30, atol=1e-4)  # report rounds to 5 dp


def test_missing_extra_and_wrong_dates_are_reported(tmp_path: Path) -> None:
    entry = _write_fire(tmp_path, "fire_1", ["2019-08-01", "2019-08-02"])
    _write_fire(tmp_path, "fire_extra", ["2019-08-01"])
    (tmp_path / "2019" / "fire_9.tmp.npz").write_bytes(b"partial")
    fires = {
        (2019, "fire_1"): FireEntry(2019, "fire_1", (*entry.members, "2019/fire_1/2019-08-03.tif"), 1),
        (2019, "fire_2"): FireEntry(2019, "fire_2", ("2019/fire_2/2019-08-01.tif",), 1),
    }
    report = validate(tmp_path, fires, [2019])
    assert not report["ok"]
    assert report["problems"]["2019 missing"] == ["fire_2"]
    assert report["problems"]["2019 extra"] == ["fire_extra"]
    assert report["problems"]["2019/fire_1"] == ["dates differ from the zip listing"]
    assert "temporary files" in report["problems"]

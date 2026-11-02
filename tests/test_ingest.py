from __future__ import annotations

import pytest

from fireline.data.ingest import FireEntry, select_fires


def _fires(*ids: str) -> dict[str, FireEntry]:
    return {f: FireEntry(2019, f, (f"2019/{f}/2019-08-01.tif",), 1) for f in ids}


def test_select_explicit_then_first_remaining() -> None:
    fires = _fires("fire_3", "fire_1", "fire_2", "fire_4")
    chosen = select_fires(fires, ["fire_3"], first=2)
    assert [e.fire_id for e in chosen] == ["fire_3", "fire_1", "fire_2"]


def test_select_unknown_fire_raises() -> None:
    with pytest.raises(KeyError):
        select_fires(_fires("fire_1"), ["fire_9"], first=0)

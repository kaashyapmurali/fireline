from __future__ import annotations

from pathlib import Path

import pytest

from fireline.data.copy_tree import copy_verified, sha256


def test_copies_selected_dirs_and_verifies(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "processed/wsts/2019").mkdir(parents=True)
    (src / "smoke").mkdir()
    (src / "processed/wsts/2019/fire_1.npz").write_bytes(b"x" * 1000)
    (src / "smoke/junk.npz").write_bytes(b"junk")
    manifest = copy_verified(src, dst, ["processed"])
    assert [m["path"] for m in manifest] == ["processed/wsts/2019/fire_1.npz"]
    assert manifest[0]["sha256"] == sha256(src / "processed/wsts/2019/fire_1.npz")
    assert not (dst / "smoke").exists()
    with pytest.raises(FileExistsError):
        copy_verified(src, dst, ["processed"])

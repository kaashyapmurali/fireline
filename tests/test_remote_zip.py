from __future__ import annotations

import io
import zipfile
import zlib

import pytest

from fireline.data.remote_zip import RemoteZip


def _zip_bytes(members: dict[str, bytes], compression: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _fetcher(blob: bytes, calls: list[tuple[int, int]] | None = None):
    def fetch(start: int, end: int) -> bytes:
        if calls is not None:
            calls.append((start, end))
        return blob[start : end + 1]

    return fetch


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_lists_and_reads_members(compression: int) -> None:
    members = {"2019/fire_1/2019-08-01.tif": b"a" * 1000, "2019/fire_1/2019-08-02.tif": bytes(range(256)) * 10}
    blob = _zip_bytes(members, compression)
    archive = RemoteZip(_fetcher(blob), len(blob))
    assert set(archive.members) == set(members)
    for name, data in members.items():
        assert archive.read(name) == data
        assert archive.members[name].crc32 == zlib.crc32(data)


def test_reads_only_requested_ranges() -> None:
    members = {f"m{i}.bin": bytes([i]) * 50_000 for i in range(5)}
    blob = _zip_bytes(members, zipfile.ZIP_STORED)
    calls: list[tuple[int, int]] = []
    archive = RemoteZip(_fetcher(blob, calls), len(blob))
    calls.clear()
    archive.read("m3.bin")
    fetched = sum(end - start + 1 for start, end in calls)
    assert fetched < 50_000 + 2_000  # member plus local-header slack


def test_detects_corruption() -> None:
    members = {"x.bin": b"hello world" * 100}
    blob = bytearray(_zip_bytes(members, zipfile.ZIP_STORED))
    archive = RemoteZip(_fetcher(bytes(blob)), len(blob))
    offset = blob.find(b"hello world")
    blob[offset] ^= 0xFF
    corrupted = RemoteZip(_fetcher(bytes(blob)), len(blob))
    assert archive.members == corrupted.members
    with pytest.raises(ValueError, match="CRC"):
        corrupted.read("x.bin")


def test_read_many_merges_adjacent_members_into_one_request() -> None:
    members = {f"2019/fire_1/2019-08-0{i}.tif": bytes([i]) * 20_000 for i in range(1, 6)}
    blob = _zip_bytes(members, zipfile.ZIP_STORED)
    calls: list[tuple[int, int]] = []
    archive = RemoteZip(_fetcher(blob, calls), len(blob))
    calls.clear()
    assert archive.read_many(list(members)) == members
    assert len(calls) == 1


def test_read_many_splits_distant_members() -> None:
    members = {"a.bin": b"a" * 10, "big.bin": b"x" * 300_000, "b.bin": b"b" * 10}
    blob = _zip_bytes(members, zipfile.ZIP_STORED)
    calls: list[tuple[int, int]] = []
    archive = RemoteZip(_fetcher(blob, calls), len(blob))
    calls.clear()
    assert archive.read_many(["a.bin", "b.bin"], max_gap=1000) == {"a.bin": b"a" * 10, "b.bin": b"b" * 10}
    assert len(calls) == 2

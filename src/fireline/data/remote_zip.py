"""Read individual members of a large remote zip archive with HTTP range requests.

The WildfireSpreadTS archives on Zenodo are tens of gigabytes, but each GeoTIFF inside is only a few
megabytes. Reading the zip's central directory (at the end of the file) lists every member with its
offset, size and CRC-32, so single members can be fetched and verified without downloading the archive.
"""

from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass

RangeFetcher = Callable[[int, int], bytes]
"""Return bytes ``start..end`` (both inclusive) of the archive."""

_EOCD = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
_CENTRAL_HEADER = b"PK\x01\x02"
_LOCAL_HEADER = b"PK\x03\x04"
_STORED, _DEFLATED = 0, 8
_U32_MAX, _U16_MAX = 0xFFFFFFFF, 0xFFFF


@dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    header_offset: int


def http_range_fetcher(url: str, retries: int = 4, timeout: float = 120.0) -> RangeFetcher:
    """Range fetcher for ``url`` with simple exponential-backoff retries."""

    def fetch(start: int, end: int) -> bytes:
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError(f"Expected {end - start + 1} bytes, got {len(data)}.")
                return data
            except OSError, urllib.error.URLError:
                if attempt == retries:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    return fetch


def http_content_length(url: str, timeout: float = 60.0) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


class RemoteZip:
    """Member listing and verified member reads for a zip reachable through a range fetcher."""

    def __init__(self, fetch: RangeFetcher, total_size: int) -> None:
        self._fetch = fetch
        self.total_size = total_size
        self.members: dict[str, ZipMember] = {m.name: m for m in self._read_central_directory()}

    @classmethod
    def from_url(cls, url: str) -> RemoteZip:
        return cls(http_range_fetcher(url), http_content_length(url))

    def _read_central_directory(self) -> list[ZipMember]:
        tail_start = max(0, self.total_size - 70_000)  # EOCD comment is at most 64 KiB
        tail = self._fetch(tail_start, self.total_size - 1)
        eocd = tail.rfind(_EOCD)
        if eocd < 0:
            raise ValueError("End-of-central-directory record not found.")
        count, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10 : eocd + 20])
        if _U32_MAX in (cd_size, cd_offset) or count == _U16_MAX:
            locator = tail.rfind(_ZIP64_LOCATOR)
            if locator < 0:
                raise ValueError("ZIP64 archive without a ZIP64 locator.")
            record_offset = struct.unpack("<Q", tail[locator + 8 : locator + 16])[0]
            record = self._fetch(record_offset, record_offset + 55)
            count, cd_size, cd_offset = struct.unpack("<QQQ", record[32:56])
        directory = self._fetch(cd_offset, cd_offset + cd_size - 1)
        members = [_parse_central_header(directory, pos) for pos in _iter_headers(directory)]
        if len(members) != count:
            raise ValueError(f"Central directory lists {len(members)} entries, expected {count}.")
        return members

    def read(self, name: str) -> bytes:
        """Return the uncompressed bytes of member ``name`` after checking its CRC-32."""
        return self.read_many([name])[name]

    def read_many(self, names: list[str], max_gap: int = 1 << 20) -> dict[str, bytes]:
        """Read several members, merging members that are close together in the archive into one request.

        One fire's daily images are stored next to each other, so a whole fire usually takes a single request
        instead of two round trips per image.
        """
        members = sorted((self.members[n] for n in names), key=lambda m: m.header_offset)
        result: dict[str, bytes] = {}
        group: list[ZipMember] = []
        for member in members:
            if group and member.header_offset - _span_end(group[-1]) > max_gap:
                result.update(self._read_group(group))
                group = []
            group.append(member)
        if group:
            result.update(self._read_group(group))
        return result

    def _read_group(self, group: list[ZipMember]) -> dict[str, bytes]:
        start = group[0].header_offset
        end = min(_span_end(group[-1]), self.total_size - 1)
        buffer = self._fetch(start, end)
        out: dict[str, bytes] = {}
        for member in group:
            pos = member.header_offset - start
            if buffer[pos : pos + 4] != _LOCAL_HEADER:
                raise ValueError(f"Bad local header for {member.name}.")
            name_len, extra_len = struct.unpack("<HH", buffer[pos + 26 : pos + 30])
            data_start = pos + 30 + name_len + extra_len
            data = buffer[data_start : data_start + member.compressed_size]
            if len(data) != member.compressed_size:
                raise ValueError(f"Truncated data for {member.name}.")
            out[member.name] = _decode(member, data)
        return out


_LOCAL_HEADER_SLACK = 1024
"""Upper bound on local-header name + extra length beyond the central-directory values (ZIP64 extras)."""


def _span_end(member: ZipMember) -> int:
    """Last byte that may belong to ``member`` (local header, name, extra field and data)."""
    return member.header_offset + 30 + len(member.name.encode()) + _LOCAL_HEADER_SLACK + member.compressed_size


def _decode(member: ZipMember, data: bytes) -> bytes:
    if member.method == _DEFLATED:
        data = zlib.decompress(data, -15)
    elif member.method != _STORED:
        raise ValueError(f"Unsupported compression method {member.method} for {member.name}.")
    if len(data) != member.size or zlib.crc32(data) != member.crc32:
        raise ValueError(f"Size or CRC-32 mismatch for {member.name}.")
    return data


def _iter_headers(directory: bytes) -> list[int]:
    positions, pos = [], 0
    while pos < len(directory) and directory[pos : pos + 4] == _CENTRAL_HEADER:
        positions.append(pos)
        name_len, extra_len, comment_len = struct.unpack("<HHH", directory[pos + 28 : pos + 34])
        pos += 46 + name_len + extra_len + comment_len
    return positions


def _parse_central_header(directory: bytes, pos: int) -> ZipMember:
    method = struct.unpack("<H", directory[pos + 10 : pos + 12])[0]
    crc, compressed, size = struct.unpack("<III", directory[pos + 16 : pos + 28])
    name_len, extra_len = struct.unpack("<HH", directory[pos + 28 : pos + 32])
    offset = struct.unpack("<I", directory[pos + 42 : pos + 46])[0]
    name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8")
    extra = directory[pos + 46 + name_len : pos + 46 + name_len + extra_len]
    # ZIP64 extended information: only the fields set to 0xFFFFFFFF in the header are present, in order.
    k = 0
    while k + 4 <= len(extra):
        tag, field_len = struct.unpack("<HH", extra[k : k + 4])
        if tag == 0x0001:
            q = k + 4
            if size == _U32_MAX:
                size = struct.unpack("<Q", extra[q : q + 8])[0]
                q += 8
            if compressed == _U32_MAX:
                compressed = struct.unpack("<Q", extra[q : q + 8])[0]
                q += 8
            if offset == _U32_MAX:
                offset = struct.unpack("<Q", extra[q : q + 8])[0]
        k += 4 + field_len
    return ZipMember(name, method, crc, compressed, size, offset)

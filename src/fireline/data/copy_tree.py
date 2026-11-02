"""Copy directory trees between mounted volumes, verifying every file with SHA-256.

    python -m fireline.data.copy_tree --src /src --dst /data --dirs processed reports jobs --manifest MANIFEST.json

Refuses to overwrite any existing destination file. Writes a manifest of every copied file with size and hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(src_root: Path, dst_root: Path, dirs: list[str]) -> list[dict]:
    files = sorted(p for d in dirs for p in (src_root / d).rglob("*") if p.is_file())
    targets = [dst_root / p.relative_to(src_root) for p in files]
    existing = [t for t in targets if t.exists()]
    if existing:
        raise FileExistsError(f"{len(existing)} destination files exist (e.g. {existing[0]}); refusing.")
    manifest = []
    for i, (source, target) in enumerate(zip(files, targets, strict=True), 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hash, target_hash = sha256(source), sha256(target)
        if source_hash != target_hash or source.stat().st_size != target.stat().st_size:
            raise ValueError(f"Verification failed for {source}.")
        manifest.append(
            {"path": str(source.relative_to(src_root)), "bytes": target.stat().st_size, "sha256": target_hash}
        )
        if i % 100 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] copied and verified", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"{args.manifest} exists; refusing to overwrite.")
    manifest = copy_verified(args.src, args.dst, args.dirs)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"files": len(manifest), "bytes": sum(m["bytes"] for m in manifest),
                                         "entries": manifest}, indent=1) + "\n")  # fmt: skip
    print(f"Copied {len(manifest)} files, {sum(m['bytes'] for m in manifest) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()

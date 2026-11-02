"""Build the report bundle: numbers (site.json), images, and a manifest with SHA-256 per file.

    uv run --group notebook python report/scripts/build.py              # needs the test-year evaluation
    uv run --group notebook python report/scripts/build.py --draft      # validation-only draft before the test run

Run from the repository root. Output: ``report/bundle/`` (committed). The website syncs a pinned copy of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics

TEST_RINGS = Path("outputs/modal/test-2021-01/rings-2021.json")
DATA = Path("datasets/processed/wsts")
CHECKPOINT = Path("outputs/modal/unet-sup-02/checkpoints/best.pt")
STATS = Path("outputs/modal/baselines-01/stats-2018-2019.json")
CACHE = Path("/tmp/fireline-cache")

SCHEMA_VERSION = 1
BUNDLE = Path("report/bundle")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return None  # no commit yet: the manifest says so instead of inventing one


def write_manifest(bundle: Path, draft: bool) -> dict:
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "draft": draft,
        "source_commit": source_commit(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {str(p.relative_to(bundle)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true", help="Allow a bundle without the test-year results.")
    args = parser.parse_args()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    site = {"schema_version": SCHEMA_VERSION, "draft": args.draft, **metrics.collect(require_test=not args.draft)}
    if not args.draft:  # imagery shows a test-year fire, so it needs the test-year evaluation too
        import imagery  # torch is only needed here

        from fireline.data.inputs import Stats

        site["timeline"] = imagery.build(TEST_RINGS, DATA, 2021, CHECKPOINT, Stats.from_json(STATS), BUNDLE, CACHE)
        name = json.loads(Path("report/sources/timeline-fire.json").read_text())
        if name["fire"] != site["timeline"]["fire"]:
            raise ValueError(f"timeline-fire.json names {name['fire']} but the rule chose {site['timeline']['fire']}")
        site["timeline"]["name"], site["timeline"]["name_check"] = name["name"], name["check"]
    (BUNDLE / "site.json").write_text(json.dumps(site, indent=2) + "\n")
    manifest = write_manifest(BUNDLE, args.draft)
    print(f"bundle: {len(manifest['files'])} files, test results {'included' if site['test'] else 'missing (draft)'}")


if __name__ == "__main__":
    main()

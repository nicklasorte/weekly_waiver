"""Download nflverse source data and pin each file with a sha256.

nflverse revises history in place: stat corrections, play-by-play reclassification
and roster fixes all land weeks after the fact, and the release assets are
overwritten rather than versioned. A backtest run against the 2022 snap counts
downloaded in March is therefore not comparable to one run against the same URL
in October, even though nothing in this repo changed.

`data/raw/MANIFEST.json` records the url, sha256, byte count and UTC fetch time
of every file, so any downstream result can be tied to the exact bytes it was
computed from, and a silent upstream revision shows up as a hash change instead
of an unexplained metric drift.

Usage:
    python -m src.fetch                 # default seasons
    python -m src.fetch 2024 2025       # specific seasons
    FORCE=1 python -m src.fetch         # re-download files already on disk
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

DEFAULT_SEASONS = range(2022, 2027)

# Season-partitioned release assets: (release, filename template).
SEASON_ASSETS = [
    ("snap_counts", "snap_counts_{year}.csv"),
    ("stats_player", "stats_player_week_{year}.csv"),
    ("pbp", "play_by_play_{year}.csv.gz"),
]

CHUNK = 1 << 20
RETRIES = 4
USER_AGENT = "weekly-waiver/0.1 (+https://github.com/nicklasorte/weekly_waiver)"


def season_urls(year: int) -> list[tuple[str, str]]:
    """(filename, url) pairs for one season."""
    out = []
    for release, template in SEASON_ASSETS:
        name = template.format(year=year)
        out.append((name, f"{BASE}/{release}/{name}"))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"generated_utc": None, "files": {}}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError:
        print(f"  warning: {MANIFEST_PATH.name} is unreadable, starting a fresh one")
        return {"generated_utc": None, "files": {}}
    manifest.setdefault("files", {})
    return manifest


def data_revision(manifest: dict | None = None) -> str:
    """A stable fingerprint of the source data the repo is currently sitting on.

    Digests only (filename, sha256) pairs, not fetch timestamps -- re-downloading
    identical bytes must not change the revision, while an upstream stat
    correction must. This is what results get stamped with so a model or a
    backtest can be tied to the exact data it was produced from.
    """
    if manifest is None:
        manifest = load_manifest()
    payload = "\n".join(
        f"{name}:{record.get('sha256', '')}"
        for name, record in sorted(manifest.get("files", {}).items())
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_manifest(manifest: dict) -> None:
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["files"] = dict(sorted(manifest["files"].items()))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, replacing it only once the transfer completes.

    Raises the underlying HTTPError/URLError so the caller can distinguish a 404
    (asset not published yet) from a transport failure worth retrying.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        tmp = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with tempfile.NamedTemporaryFile(
                    dir=dest.parent, prefix=f".{dest.name}.", delete=False
                ) as handle:
                    tmp = Path(handle.name)
                    shutil.copyfileobj(response, handle, CHUNK)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            if tmp is not None and tmp.exists():
                tmp.unlink()
            if exc.code == 404 or attempt == RETRIES - 1:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp is not None and tmp.exists():
                tmp.unlink()
            if attempt == RETRIES - 1:
                raise
            last_error = exc
        wait = 2 ** attempt
        print(f"  retry in {wait}s after {type(last_error).__name__}: {last_error}")
        time.sleep(wait)


def fetch_one(name: str, url: str, manifest: dict, force: bool) -> str:
    """Fetch a single file. Returns 'downloaded', 'skipped' or 'missing'."""
    dest = RAW_DIR / name
    if dest.exists() and not force:
        record = manifest["files"].get(name)
        if record is None or record.get("bytes") != dest.stat().st_size:
            # On disk but unpinned (or changed underneath us): hash it now so the
            # manifest still describes the bytes we would actually read.
            manifest["files"][name] = {
                "url": url,
                "sha256": sha256_of(dest),
                "bytes": dest.stat().st_size,
                "fetched_utc": datetime.fromtimestamp(
                    dest.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
            print(f"  {name}: already present, hashed from disk")
        else:
            print(f"  {name}: skipped (already present)")
        return "skipped"

    try:
        download(url, dest)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"  {name}: not published yet ({url})")
            return "missing"
        print(f"  {name}: HTTP {exc.code} {exc.reason}")
        return "missing"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  {name}: download failed ({exc})")
        return "missing"

    previous = manifest["files"].get(name, {}).get("sha256")
    digest = sha256_of(dest)
    manifest["files"][name] = {
        "url": url,
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }
    note = ""
    if previous is not None and previous != digest:
        note = "  <-- upstream revised, prior results are not comparable"
    print(f"  {name}: {dest.stat().st_size:,} bytes  {digest[:16]}...{note}")
    return "downloaded"


def fetch(seasons, force: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    counts = {"downloaded": 0, "skipped": 0, "missing": 0}

    print("games.csv")
    counts[fetch_one("games.csv", GAMES_URL, manifest, force)] += 1

    for year in seasons:
        print(f"{year}")
        for name, url in season_urls(year):
            counts[fetch_one(name, url, manifest, force)] += 1

    write_manifest(manifest)
    print(
        f"\n{len(manifest['files'])} files in MANIFEST.json "
        f"({counts['downloaded']} downloaded, {counts['skipped']} skipped, "
        f"{counts['missing']} unavailable)"
    )
    return manifest


def main(argv: list[str]) -> int:
    seasons = [int(a) for a in argv] if argv else list(DEFAULT_SEASONS)
    force = os.environ.get("FORCE") == "1"
    if force:
        print("FORCE=1: re-downloading every file")
    fetch(seasons, force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

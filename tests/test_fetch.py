"""Tests that the tracked manifest holds nothing that moves on its own.

The scheduled job commits `data/raw/MANIFEST.json` and skips the commit when
nothing changed. That skip is only real if re-running against identical bytes
produces an identical file — one timestamp in there and the job pushes a commit
every week whether or not anything moved, which is how a heartbeat gets mistaken
for a record of change.

So the property under test is byte-level idempotence, not field-by-field
plausibility: write the same manifest twice and the tracked file must not move.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src import fetch
from src.weekly import SEEN_MANIFEST


def sample_manifest() -> dict:
    return {
        "files": {
            "games.csv": {
                "url": "https://example.invalid/games.csv",
                "sha256": "a" * 64,
                "bytes": 123,
                "fetched_utc": "2026-08-30T22:23:49.501490+00:00",
            },
            "snap_counts_2025.csv": {
                "url": "https://example.invalid/snap_counts_2025.csv",
                "sha256": "b" * 64,
                "bytes": 456,
                "fetched_utc": "2026-08-30T22:23:50.091115+00:00",
            },
        }
    }


class ManifestWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.manifest_path = root / "MANIFEST.json"
        self.log_path = root / "FETCH_LOG.json"
        for name, value in (
            ("MANIFEST_PATH", self.manifest_path),
            ("FETCH_LOG_PATH", self.log_path),
        ):
            original = getattr(fetch, name)
            setattr(fetch, name, value)
            self.addCleanup(setattr, fetch, name, original)
        self.addCleanup(self.tmp.cleanup)

    def test_no_timestamps_in_the_tracked_file(self):
        fetch.write_manifest(sample_manifest())
        written = json.loads(self.manifest_path.read_text())
        self.assertEqual(list(written), ["files"])
        for name, record in written["files"].items():
            with self.subTest(file=name):
                self.assertEqual(sorted(record), ["bytes", "sha256", "url"])

    def test_rewriting_unchanged_bytes_is_byte_identical(self):
        fetch.write_manifest(sample_manifest())
        first = self.manifest_path.read_bytes()

        # Second run, later clock, same digests: exactly the weekly case.
        later = sample_manifest()
        for record in later["files"].values():
            record["fetched_utc"] = "2026-09-06T06:00:01.000000+00:00"
        fetch.write_manifest(later)

        self.assertEqual(first, self.manifest_path.read_bytes())

    def test_a_changed_digest_still_moves_the_file(self):
        fetch.write_manifest(sample_manifest())
        first = self.manifest_path.read_bytes()

        revised = sample_manifest()
        revised["files"]["games.csv"]["sha256"] = "c" * 64
        fetch.write_manifest(revised)

        self.assertNotEqual(first, self.manifest_path.read_bytes())

    def test_timestamps_go_to_the_untracked_log(self):
        manifest = sample_manifest()
        fetch.write_manifest(manifest)
        log = json.loads(self.log_path.read_text())
        self.assertIn("generated_utc", log)
        self.assertEqual(
            log["files"],
            {n: r["fetched_utc"] for n, r in manifest["files"].items()},
        )

    def test_load_manifest_restores_fetch_times_from_the_log(self):
        # The timestamps are out of git, not thrown away: a round trip through
        # both files has to bring them back, or `make data` would forget when it
        # last pulled anything the moment the manifest was rewritten.
        original = sample_manifest()
        fetch.write_manifest(original)
        restored = fetch.load_manifest()
        self.assertEqual(restored["files"], original["files"])

    def test_load_manifest_survives_a_missing_log(self):
        fetch.write_manifest(sample_manifest())
        self.log_path.unlink()
        restored = fetch.load_manifest()
        self.assertNotIn("fetched_utc", restored["files"]["games.csv"])
        self.assertEqual(restored["files"]["games.csv"]["sha256"], "a" * 64)

    def test_revision_ignores_the_timestamps_either_way(self):
        with_stamps = fetch.data_revision(sample_manifest())
        bare = sample_manifest()
        for record in bare["files"].values():
            del record["fetched_utc"]
        self.assertEqual(with_stamps, fetch.data_revision(bare))


class SeenManifestTest(unittest.TestCase):
    def test_committed_seen_manifest_carries_no_timestamp(self):
        if not SEEN_MANIFEST.exists():
            self.skipTest("LAST_MANIFEST.json not present")
        seen = json.loads(SEEN_MANIFEST.read_text())
        self.assertEqual(sorted(seen), ["files", "revision"])


if __name__ == "__main__":
    unittest.main()

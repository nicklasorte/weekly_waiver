"""Tests for the fit-version guard on persisted model bundles.

The guard exists because the failure it prevents is silent. An estimator
unpickled through a scikit-learn it was not fit under still returns floats, and
a weekly table of slightly wrong numbers looks exactly like a weekly table of
right ones. So the interesting cases are not "does it load" but "does it refuse"
— on a drifted version, and on a bundle too old to carry the stamp at all.

The committed bundles are checked too: they are what the scheduled job actually
unpickles every Tuesday, and a pin that does not match them is the whole bug.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import joblib

from src.models import (
    ASSERTED_VERSION_KEYS,
    FIT_VERSION_KEYS,
    MODEL_DIR,
    POSITIONS,
    check_fit_versions,
    library_versions,
    load_bundle,
)


class FitVersionGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.running = library_versions()
        self.bundle = {"fit_versions": dict(self.running)}

    def test_matching_versions_pass(self):
        check_fit_versions("RB", self.bundle, self.running)

    def test_asserted_drift_raises(self):
        for key in ASSERTED_VERSION_KEYS:
            with self.subTest(library=key):
                bundle = {"fit_versions": dict(self.running) | {key: "0.0.0"}}
                with self.assertRaises(SystemExit) as caught:
                    check_fit_versions("RB", bundle, self.running)
                message = str(caught.exception)
                self.assertIn(key, message)
                self.assertIn("0.0.0", message)
                # The error has to say what to do, not just that it is unhappy.
                self.assertIn("make models", message)

    def test_unasserted_drift_is_allowed(self):
        # python and joblib are recorded for forensics but must not block a run:
        # the runner picks its own CPython patch level, and joblib only
        # serialises. Blocking on either would fail the job for a non-difference.
        unasserted = [k for k in FIT_VERSION_KEYS if k not in ASSERTED_VERSION_KEYS]
        self.assertEqual(sorted(unasserted), ["joblib", "python"])
        for key in unasserted:
            with self.subTest(library=key):
                bundle = {"fit_versions": dict(self.running) | {key: "0.0.0"}}
                check_fit_versions("RB", bundle, self.running)

    def test_unstamped_bundle_raises(self):
        # A bundle from before the stamp existed cannot be verified, so it is
        # refused rather than assumed fine.
        for bundle in ({}, {"fit_versions": {}}, {"fit_versions": None}):
            with self.subTest(bundle=bundle):
                with self.assertRaises(SystemExit):
                    check_fit_versions("RB", bundle, self.running)

    def test_missing_key_counts_as_drift(self):
        partial = {k: v for k, v in self.running.items() if k != "scipy"}
        with self.assertRaises(SystemExit) as caught:
            check_fit_versions("RB", {"fit_versions": partial}, self.running)
        self.assertIn("unrecorded", str(caught.exception))


class CommittedBundlesTest(unittest.TestCase):
    """The bundles in models/ are what the Tuesday job unpickles."""

    def test_every_position_is_stamped_and_loads(self):
        for position in POSITIONS:
            with self.subTest(position=position):
                path = MODEL_DIR / f"{position}.joblib"
                if not path.exists():
                    self.skipTest(f"{path.name} not present -- run `make models`")
                recorded = joblib.load(path).get("fit_versions")
                self.assertIsNotNone(recorded, "bundle carries no fit_versions")
                self.assertEqual(sorted(recorded), sorted(FIT_VERSION_KEYS))
                # load_bundle applies the guard; under the pins in
                # requirements.txt this must not raise.
                load_bundle(position)


if __name__ == "__main__":
    unittest.main()

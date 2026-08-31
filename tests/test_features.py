"""Tests for the one place the panel reaches across seasons.

`src/features.py` promises NO LOOKAHEAD, and every column in it is computed
inside a single (player, season) or (season, position, week) group -- except the
two empirical Bayes shares, whose beta priors are fitted on the pooled rate
distribution over every season in the build.

That distinction is load-bearing for the walk-forward replay in
`outputs/backtests/01_season_replay.py`: a 2023 replay may not see a prior
shaped by 2024 and 2025. So it is asserted here rather than believed:

- `test_only_eb_columns_cross_seasons` rebuilds the panel at two different
  scopes and demands the overlapping rows be bit-identical in every column but
  those two. If a future feature quietly starts pooling across seasons, this is
  what catches it -- it is a guard on the file's central claim, not on one
  function.
- the prior-mask tests pin the fix itself: a masked fit must ignore the excluded
  rows entirely, and must reproduce fitting on the kept rows alone.

Two source-data guards live here for the same reason -- both failure modes are
silent, and both were found by the full-history replay in
`outputs/backtests/02_walkforward_2014_2025.py`:

- `EmptySourceTest` pins `require_rows`. `snap_counts_2012.csv` returns HTTP 200
  and contains a header row and nothing else, so a fetch-succeeded check passes
  and the season contributes no rows to the panel without anything failing.
- `RelocationTest` pins the team-code normalisation. The three nflverse feeds
  disagree on team codes before 2020 -- stats says `LA` for the 2013 Rams while
  snaps and the schedule say `STL` -- which dropped three franchises from the
  merge and left their `fwd3` NaN.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import (
    RAW_DIR,
    RELOCATIONS,
    build,
    empirical_bayes_share,
    fit_beta_prior,
    load_schedule,
    normalize_teams,
    require_rows,
)

# The two columns that are allowed to depend on which seasons are in the build.
CROSS_SEASON_COLUMNS = {"eb_tgt_share", "eb_car_share"}


def _frame(seasons_rates: dict[int, list[tuple[int, int]]]) -> pd.DataFrame:
    """One-row-per-player panel: {season: [(successes, trials), ...]}."""
    rows = []
    for season, pairs in seasons_rates.items():
        for i, (successes, trials) in enumerate(pairs):
            rows.append(
                {
                    "player_id": f"{season}-{i}",
                    "season": season,
                    "week": 1,
                    "position": "WR",
                    "targets": successes,
                    "team_tgt": trials,
                }
            )
    return pd.DataFrame(rows)


class PriorMaskTest(unittest.TestCase):
    """The prior must be fitted on exactly the masked rows and no others."""

    def setUp(self) -> None:
        # Two seasons with deliberately different rate distributions, so a prior
        # fitted on one is not accidentally the prior fitted on both.
        self.panel = _frame(
            {
                2022: [(1, 10), (2, 10), (3, 10), (2, 10), (1, 10)],
                2023: [(8, 10), (9, 10), (7, 10), (8, 10), (9, 10)],
            }
        )

    def _priors(self, mask):
        """The alphas the shrinkage implies, recovered from the printed fit."""
        return empirical_bayes_share(
            self.panel, "targets", "team_tgt", "eb_tgt_share", mask
        )

    def test_mask_changes_the_prior(self):
        both = self._priors(None)
        early = self._priors(self.panel["season"] == 2022)
        self.assertFalse(
            np.allclose(both.to_numpy(), early.to_numpy()),
            "masking the prior to 2022 left every shrunk share unchanged, which "
            "means the mask is not reaching fit_beta_prior",
        )

    def test_mask_equals_fitting_on_the_subset(self):
        """A masked fit over both seasons == an unmasked fit over the kept rows."""
        masked = self._priors(self.panel["season"] == 2022)
        subset = self.panel[self.panel["season"] == 2022]
        alone = empirical_bayes_share(subset, "targets", "team_tgt", "eb_tgt_share")
        np.testing.assert_allclose(
            masked.loc[subset.index].to_numpy(), alone.to_numpy(), rtol=1e-12
        )

    def test_excluded_rows_do_not_move_the_prior(self):
        """Changing rows outside the mask must not change a single output value."""
        masked = self._priors(self.panel["season"] == 2022)
        moved = self.panel.copy()
        moved.loc[moved["season"] == 2023, "targets"] = 0
        masked_after = empirical_bayes_share(
            moved, "targets", "team_tgt", "eb_tgt_share", moved["season"] == 2022
        )
        kept = masked[self.panel["season"] == 2022]
        np.testing.assert_allclose(
            kept.to_numpy(),
            masked_after[moved["season"] == 2022].to_numpy(),
            rtol=1e-12,
        )

    def test_default_is_unchanged(self):
        """No mask must behave exactly as the shipped code did before the mask."""
        rates = pd.Series([0.1, 0.2, 0.3, 0.2, 0.1, 0.8, 0.9, 0.7, 0.8, 0.9])
        alpha, beta = fit_beta_prior(rates)
        result = self._priors(None)
        expected = (self.panel["targets"] + alpha) / (self.panel["team_tgt"] + alpha + beta)
        np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-12)


class CrossSeasonColumnTest(unittest.TestCase):
    """The panel's no-lookahead claim, checked by rebuilding at two scopes.

    Slow and data-dependent, so it skips when `make data` has not been run --
    but when the data is there this is the test that would actually catch a new
    feature reaching across seasons.
    """

    @classmethod
    def setUpClass(cls) -> None:
        needed = [
            RAW_DIR / name
            for name in (
                "games.csv",
                "stats_player_week_2022.csv",
                "snap_counts_2022.csv",
                "stats_player_week_2023.csv",
                "snap_counts_2023.csv",
            )
        ]
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            raise unittest.SkipTest(
                f"raw data not present ({', '.join(missing)}) -- run `make data`"
            )
        cls.narrow = build([2022])
        cls.wide = build([2022, 2023])

    def _aligned(self):
        key = ["player_id", "season", "week"]
        narrow = self.narrow.set_index(key).sort_index()
        wide = self.wide[self.wide["season"] == 2022].set_index(key).sort_index()
        self.assertTrue(narrow.index.equals(wide.index), "row sets diverged")
        return narrow, wide

    def test_only_eb_columns_cross_seasons(self):
        narrow, wide = self._aligned()
        drifted = set()
        for column in narrow.columns:
            a, b = narrow[column], wide[column]
            if a.dtype.kind in "fi":
                same = np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
            else:
                same = a.astype(str).equals(b.astype(str))
            if not same:
                drifted.add(column)
        self.assertLessEqual(
            drifted,
            CROSS_SEASON_COLUMNS,
            f"{sorted(drifted - CROSS_SEASON_COLUMNS)} changed when 2023 entered "
            "the build. Every column but the empirical Bayes shares must be "
            "computable from one season alone -- see the module docstring.",
        )

    def test_the_eb_columns_really_do_move(self):
        """Guards the guard: if they stopped moving, the test above is vacuous."""
        narrow, wide = self._aligned()
        for column in sorted(CROSS_SEASON_COLUMNS):
            self.assertFalse(
                np.allclose(narrow[column].to_numpy(), wide[column].to_numpy(),
                            equal_nan=True),
                f"{column} no longer depends on build scope; if the prior was "
                "made per-season, drop it from CROSS_SEASON_COLUMNS",
            )

    def test_prior_seasons_restricts_the_leak(self):
        """Building 2022-2023 with the prior pinned to 2022 reproduces 2022-alone."""
        pinned = build([2022, 2023], prior_seasons=[2022])
        key = ["player_id", "season", "week"]
        left = self.narrow.set_index(key).sort_index()
        right = pinned[pinned["season"] == 2022].set_index(key).sort_index()
        for column in sorted(CROSS_SEASON_COLUMNS):
            np.testing.assert_allclose(
                left[column].to_numpy(), right[column].to_numpy(),
                rtol=1e-10, equal_nan=True,
                err_msg=f"{column} differs from the 2022-only build even with "
                        "prior_seasons=[2022]",
            )


class EmptySourceTest(unittest.TestCase):
    """A source file that parses but carries no rows must fail, not vanish.

    This is the `snap_counts_2012.csv` failure mode: HTTP 200, 154 bytes, a
    header row and no data. Checking that the fetch succeeded passes it; only
    checking the parsed contents catches it.
    """

    def test_empty_frame_raises(self):
        with self.assertRaises(SystemExit) as caught:
            require_rows(pd.DataFrame(columns=["a", "b"]), RAW_DIR / "snap_counts_2012.csv")
        self.assertIn("zero data rows", str(caught.exception))

    def test_non_empty_frame_passes(self):
        require_rows(pd.DataFrame({"a": [1]}), RAW_DIR / "snap_counts_2013.csv")


class RelocationTest(unittest.TestCase):
    """Historical team codes must resolve to one convention across all feeds."""

    def test_relocated_codes_map_forward(self):
        mapped = normalize_teams(pd.Series(["STL", "SD", "OAK"]))
        self.assertEqual(mapped.tolist(), ["LA", "LAC", "LV"])

    def test_unaffected_codes_are_untouched(self):
        codes = ["KC", "NE", "LA", "LAC", "LV", "WAS", "JAX"]
        self.assertEqual(normalize_teams(pd.Series(codes)).tolist(), codes)

    def test_no_relocation_target_is_also_a_source(self):
        """The map must not chain: mapping twice has to be a no-op."""
        self.assertFalse(set(RELOCATIONS.values()) & set(RELOCATIONS))

    def test_schedule_uses_modern_codes(self):
        """A pre-2020 season's schedule must not carry a retired code.

        Without this the `(season, week, team)` lookup in `forward_three` misses
        for every relocated franchise and their fwd3 comes back NaN -- silently,
        because a missing key is indistinguishable from a bye.
        """
        if not (RAW_DIR / "games.csv").exists():
            self.skipTest("games.csv not fetched")
        played, _ = load_schedule()
        stale = {team for (_, _, team) in played if team in RELOCATIONS}
        self.assertEqual(
            stale, set(),
            "load_schedule returned retired team codes; forward_three will not "
            "match them against the panel, which uses modern franchise codes",
        )


if __name__ == "__main__":
    unittest.main()

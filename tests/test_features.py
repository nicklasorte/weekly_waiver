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

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import RAW_DIR, build, empirical_bayes_share, fit_beta_prior

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


if __name__ == "__main__":
    unittest.main()

"""Tests for how the report presents a bound the method cannot actually resolve.

`weekly.score_week` maps rank to points through a quantile function whose input
is clipped at PROJECTION_CLIP, so every `score_hi` at or above the upper clip
lands on one shared number: the position's 99th-percentile outcome. Printing
that as a player's ceiling is false precision, and it hits hardest at the top of
the table -- in 2025 Week 8 every candidate the report printed was saturated.

The cases below are the ones a naive check gets wrong: a bound that saturates
*without* reaching 1.0, and a table written before `score_hi` was a column.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.report import (
    REPLACEMENT_RANK,
    ceiling_saturated,
    projection,
    replacement_level,
    with_edge,
)
from src.weekly import PROJECTION_CLIP


def candidate(**overrides) -> pd.Series:
    row = {"proj_pts_lo": 1.5667, "proj_pts_hi": 14.8060, "score_hi": 0.8}
    row.update(overrides)
    return pd.Series(row)


class SaturatedCeilingTest(unittest.TestCase):
    def test_bound_at_the_clip_is_saturated(self):
        # Rashee Rice, 2025 wk8: score 0.83 + half-width 0.29 clips to 1.0.
        self.assertTrue(ceiling_saturated(candidate(score_hi=1.0)))

    def test_bound_past_the_clip_but_under_one_is_saturated(self):
        # Darnell Mooney, same week, score_hi 0.9932: never reaches 1.0, but the
        # quantile input is clipped at 0.99 all the same, so he prints the same
        # 14.8 as Rice. Testing `== 1.0` would call this one honest.
        self.assertTrue(ceiling_saturated(candidate(score_hi=0.9932)))

    def test_bound_below_the_clip_is_that_players_own(self):
        self.assertFalse(ceiling_saturated(candidate(score_hi=0.98)))
        self.assertFalse(ceiling_saturated(candidate(score_hi=PROJECTION_CLIP[1] - 1e-9)))

    def test_table_without_score_hi_still_renders(self):
        # Weekly tables written before this column existed fall back to the
        # plain range rather than raising.
        self.assertFalse(ceiling_saturated(pd.Series({"proj_pts_lo": 1.5, "proj_pts_hi": 14.8})))
        self.assertFalse(ceiling_saturated(candidate(score_hi=np.nan)))


class ProjectionTextTest(unittest.TestCase):
    def test_saturated_prints_a_floor_and_no_ceiling(self):
        text = projection(candidate(score_hi=1.0))
        self.assertEqual(text, "1.6+ pts/wk (upside unbounded)")
        self.assertNotIn("14.8", text)

    def test_unsaturated_prints_the_range(self):
        self.assertEqual(
            projection(candidate(score_hi=0.85)), "1.6–14.8 pts/wk"
        )


if __name__ == "__main__":
    unittest.main()


class ReplacementLevelTest(unittest.TestCase):
    """The one definition of replacement level, which the headline now rests on.

    `REPLACEMENT_RANK` was untested before PAR became the outcome metric. It is
    now read by the weekly table's tiering, by the ledger's scoring and by the
    walk-forward re-score, so its exact index convention is load-bearing in
    three places and is pinned here.
    """

    def test_the_rank_is_zero_indexed_so_qb_2_is_the_third_best(self):
        # "Replacement is the next player you would actually take instead --
        # the third-best available QB." Off by one and every quarterback claim
        # is scored against the wrong player.
        values = pd.Series([30.0, 20.0, 10.0, 5.0, 1.0])
        self.assertEqual(REPLACEMENT_RANK["QB"], 2)
        self.assertAlmostEqual(replacement_level(values, "QB"), 10.0)

    def test_each_position_reads_its_own_rank(self):
        values = pd.Series(range(20, 0, -1), dtype=float)
        for position, rank in REPLACEMENT_RANK.items():
            self.assertAlmostEqual(
                replacement_level(values, position), float(20 - rank),
                msg=f"{position} did not read rank {rank}",
            )

    def test_row_order_does_not_matter(self):
        ordered = pd.Series([9.0, 7.0, 5.0, 3.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            replacement_level(ordered, "WR"),
            replacement_level(ordered.sample(frac=1.0, random_state=0), "WR"),
        )

    def test_an_unknown_position_falls_back_rather_than_raising(self):
        self.assertAlmostEqual(
            replacement_level(pd.Series(range(10, 0, -1), dtype=float), "K"), 5.0
        )

    def test_nan_values_are_dropped_not_ranked(self):
        # A player whose window has not resolved has no outcome to contribute.
        # Sorting NaN into the order would move the baseline by a non-number.
        with_nan = pd.Series([30.0, np.nan, 20.0, 10.0, np.nan, 5.0])
        self.assertAlmostEqual(with_nan.pipe(replacement_level, "QB"), 10.0)

    def test_a_pool_shorter_than_the_rank_clamps_to_its_worst(self):
        # Where this function lives -- tiering a candidate table -- the clamp is
        # a safety filter: the worst player gets an edge of exactly zero and
        # `assign_tiers` drops him. The outcome-side caller refuses to clamp
        # instead, for the opposite reason, and says so where it does it.
        self.assertAlmostEqual(replacement_level(pd.Series([5.0, 1.0]), "WR"), 1.0)

    def test_an_empty_pool_is_nan_rather_than_zero(self):
        self.assertTrue(np.isnan(replacement_level(pd.Series([], dtype=float), "WR")))


class WithEdgeTest(unittest.TestCase):
    def table(self):
        rows = []
        for position, points in (("QB", [30, 25, 20, 15]), ("WR", [9, 8, 7, 6, 5, 4, 3])):
            for i, value in enumerate(points):
                rows.append({"player_display_name": f"{position}{i}",
                             "position": position, "proj_pts": float(value)})
        return pd.DataFrame(rows)

    def test_edge_is_measured_against_the_position_not_the_pool(self):
        # The correction the report is wrong without: a 20-point quarterback is
        # replacement level and a 9-point receiver is the best on the wire.
        edged = with_edge(self.table()).set_index("player_display_name")["edge"]
        self.assertAlmostEqual(edged["QB2"], 0.0)     # QB rank 2 == the baseline
        self.assertAlmostEqual(edged["QB0"], 10.0)
        self.assertAlmostEqual(edged["WR0"], 6.0)     # WR rank 6 == 3.0

    def test_exactly_rank_many_players_clear_the_tier_filter(self):
        # `assign_tiers` keeps `edge > 0`, and the baseline is the rank-indexed
        # player, so the eligible count at a position is its rank by
        # construction. This is what makes the filter a filter rather than a
        # threshold that happens to bind.
        edged = with_edge(self.table())
        eligible = edged[edged["edge"] > 0]["position"].value_counts()
        self.assertEqual(eligible["QB"], REPLACEMENT_RANK["QB"])
        self.assertEqual(eligible["WR"], REPLACEMENT_RANK["WR"])

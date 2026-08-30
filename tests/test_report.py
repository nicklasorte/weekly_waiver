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

from src.report import ceiling_saturated, projection
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

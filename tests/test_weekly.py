"""Tests for schedule-derived week resolution.

The whole point of current_week() is that it does not do calendar arithmetic.
These cases are the ones arithmetic gets wrong: seasons that kick off on
different dates, a bye-heavy week with only 13 games, the five-day gap before
Week 18, and the September window where a new season has started but has not
finished a week yet.

Fixture is real schedule data (2025 plus the tail of 2024) committed to
tests/fixtures/, so the tests do not depend on anything being downloaded.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.weekly import current_season_week, current_week

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "games_sample.csv"


class WeekResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        games = pd.read_csv(FIXTURE, low_memory=False)
        cls.games = games[games["game_type"] == "REG"].copy()

    def week(self, year: int, month: int, day: int, season: int = 2025):
        return current_week(season, date(year, month, day), self.games)

    def test_tuesday_after_week_one(self):
        # 2025 Week 1 ran Sep 4-8. Tuesday the 9th: Week 1 is in the books.
        self.assertEqual(self.week(2025, 9, 9), 1)

    def test_bye_heavy_week_eight(self):
        # Week 8 is the bye-heaviest week of 2025 -- 13 games, six teams idle.
        # A count-the-games heuristic trips here; reading the schedule does not.
        self.assertEqual(self.week(2025, 10, 28), 8)

    def test_monday_of_week_eight_is_not_yet_complete(self):
        # Week 8's last game is Monday Oct 27. On that Monday the week is still
        # live, so the most recent completed week is 7.
        self.assertEqual(self.week(2025, 10, 27), 7)

    def test_thanksgiving_week(self):
        # Week 12 opens on Thanksgiving Thursday and closes Monday Nov 24.
        self.assertEqual(self.week(2025, 11, 25), 12)

    def test_week_eighteen_gap(self):
        # Week 17 ends Dec 29; Week 18 does not start until Jan 3, a five-day
        # gap that rolling a 7-day counter straight through gets wrong.
        self.assertEqual(self.week(2025, 12, 30), 17)
        self.assertEqual(self.week(2026, 1, 2), 17)
        self.assertEqual(self.week(2026, 1, 6), 18)

    def test_before_the_season_starts(self):
        self.assertIsNone(self.week(2025, 8, 20))

    def test_after_the_season_ends(self):
        self.assertEqual(self.week(2026, 6, 1), 18)

    def test_one_date_means_different_things_to_different_seasons(self):
        # Jan 6 2025: the 2024 season has just finished Week 18, and the 2025
        # season has not played a snap. One calendar date, two answers -- which
        # is precisely what date arithmetic cannot represent.
        self.assertEqual(self.week(2025, 1, 6, season=2024), 18)
        self.assertIsNone(self.week(2025, 1, 6, season=2025))


class SeasonWeekResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        games = pd.read_csv(FIXTURE, low_memory=False)
        cls.games = games[games["game_type"] == "REG"].copy()

    def resolve(self, year: int, month: int, day: int):
        return current_season_week(date(year, month, day), self.games)

    def test_picks_the_live_season(self):
        self.assertEqual(self.resolve(2025, 10, 28), (2025, 8))

    def test_september_gap_falls_back_to_last_season(self):
        # 2025 has kicked off but has not finished Week 1. The answer is last
        # season's finale, not a Week 0 in the new one.
        self.assertEqual(self.resolve(2025, 9, 6), (2024, 18))

    def test_offseason_points_at_the_last_completed_week(self):
        self.assertEqual(self.resolve(2026, 8, 30), (2025, 18))

"""Tests for the three-arm comparison.

Written against synthetic weeks rather than the real ledger, because the real
ledger has one week in it and every case that matters here needs several. The
cases below are the ones where a plausible implementation is quietly wrong in a
direction that flatters the repo:

- the naive arm picking with hindsight, by skipping players whose fwd3 has not
  resolved;
- an arm's pick being dropped as unscoreable because the player sat out the week
  he was recommended in, which hits the news-driven arm hardest;
- a contaminated week being kept for two arms and dropped for the third, so the
  means end up over different week sets;
- the pre-registered rule drifting at its boundaries.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ledger import (
    NAIVE_MARGIN,
    REPO_MARGIN,
    TIE_BAND,
    arm_rows,
    arm_summary,
    bootstrap_ci,
    forward_three,
    grade_arms,
    naive_picks,
    order_status,
    outcome,
    paired_differences,
    replacement_levels,
    usable_weeks,
    verdict,
    week_means,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "games_sample.csv"

# 2025 weeks 8-11 are all played weeks for both teams in the fixture, so a
# week-8 window spans weeks 9, 10 and 11 with no bye to reason about.
def panel_row(player, week, pts, fwd3=np.nan, position="WR", team="SEA", on_wire=True):
    return {
        "season": 2025,
        "week": week,
        "player_display_name": player,
        "position": position,
        "team": team,
        "pts": pts,
        "fwd3": fwd3,
        "on_wire": on_wire,
        "snap": 0.5,
    }


def claim_row(week, arm, player, rank, logged_at="", contaminated="", position="WR"):
    return {
        "season": 2025,
        "week": week,
        "tier": "burn",
        "action": "ADD",
        "player": player,
        "position": position,
        "dropped": "",
        "rationale": "",
        "arm": arm,
        "rank_within_arm": rank,
        "logged_at": logged_at,
        "contaminated": contaminated,
    }


class NaivePicksTest(unittest.TestCase):
    def test_takes_the_top_three_by_last_weeks_points(self):
        panel = pd.DataFrame([
            panel_row("A", 8, pts=20.0, fwd3=1.0),
            panel_row("B", 8, pts=18.0, fwd3=2.0),
            panel_row("C", 8, pts=16.0, fwd3=3.0),
            panel_row("D", 8, pts=14.0, fwd3=4.0),
        ])
        picks = naive_picks(panel, 2025, 8)
        self.assertEqual(list(picks["player_display_name"]), ["A", "B", "C"])
        self.assertEqual(list(picks["rank_within_arm"]), [1, 2, 3])

    def test_selection_never_looks_at_fwd3(self):
        # The highest scorer has no resolved outcome. Filtering him out to keep
        # the arm's average clean would hand it foresight the other arms do not
        # have, so he is still picked -- and dropped later, as an unscoreable
        # pick, which is the honest cost.
        panel = pd.DataFrame([
            panel_row("Unresolved", 8, pts=30.0, fwd3=np.nan),
            panel_row("B", 8, pts=18.0, fwd3=2.0),
            panel_row("C", 8, pts=16.0, fwd3=3.0),
        ])
        self.assertEqual(
            list(naive_picks(panel, 2025, 8)["player_display_name"]),
            ["Unresolved", "B", "C"],
        )

    def test_ties_break_on_name_not_row_order(self):
        panel = pd.DataFrame([
            panel_row("Zeta", 8, pts=10.0, fwd3=1.0),
            panel_row("Alpha", 8, pts=10.0, fwd3=1.0),
        ])
        self.assertEqual(
            list(naive_picks(panel, 2025, 8)["player_display_name"]), ["Alpha", "Zeta"]
        )

    def test_only_the_wire_pool_is_eligible(self):
        panel = pd.DataFrame([
            panel_row("Rostered", 8, pts=99.0, fwd3=1.0, on_wire=False),
            panel_row("Free", 8, pts=5.0, fwd3=1.0),
        ])
        self.assertEqual(
            list(naive_picks(panel, 2025, 8)["player_display_name"]), ["Free"]
        )


class OutcomeTest(unittest.TestCase):
    """A pick who did not play the week he was recommended still has an outcome."""

    def setUp(self):
        # No week-8 row: he was inactive, which is often exactly why he was
        # available. Weeks 9 and 11 he played; week 10 his team played and he
        # did not, which scores 0 under the fwd3 convention.
        self.panel = pd.DataFrame([
            panel_row("Returning", 7, pts=3.0),
            panel_row("Returning", 9, pts=12.0),
            panel_row("Returning", 11, pts=6.0),
        ])

    def test_reconstructed_from_the_surrounding_weeks(self):
        got = forward_three(self.panel, 2025, 8, "Returning", games_path=FIXTURE)
        self.assertAlmostEqual(got, (12.0 + 0.0 + 6.0) / 3)

    def test_outcome_falls_back_to_reconstruction_and_flags_the_pool(self):
        got, in_pool = outcome(self.panel, 2025, 8, "Returning", games_path=FIXTURE)
        self.assertAlmostEqual(got, 6.0)
        self.assertFalse(in_pool)  # no week-8 row, so not in that week's pool

    def test_a_name_the_panel_has_never_seen_is_unscoreable(self):
        got, _ = outcome(self.panel, 2025, 8, "Nobody", games_path=FIXTURE)
        self.assertIsNone(got)

    def test_the_panel_row_wins_when_there_is_one(self):
        panel = pd.concat([self.panel, pd.DataFrame([
            panel_row("Returning", 8, pts=1.0, fwd3=99.0)
        ])], ignore_index=True)
        got, in_pool = outcome(panel, 2025, 8, "Returning", games_path=FIXTURE)
        self.assertEqual(got, 99.0)
        self.assertTrue(in_pool)


class OrderStatusTest(unittest.TestCase):
    def status(self, rows) -> str:
        frame = pd.DataFrame(rows)
        frame["arm"] = frame["arm"].str.lower()
        return order_status(frame).iloc[0]["order_status"]

    def test_prompt_stamped_first_is_clean(self):
        self.assertEqual(self.status([
            claim_row(8, "prompt", "A", 1, logged_at="2025-10-28T09:00:00+00:00"),
            claim_row(8, "repo", "B", 1, logged_at="2025-10-28T11:00:00+00:00"),
        ]), "clean")

    def test_repo_stamped_first_is_unverified_not_contaminated(self):
        # CI writes the repo rows at 06:00 UTC Tuesday, hours before a human is
        # awake to write down a prompt pick. Calling this contaminated would
        # throw out most of a clean season.
        self.assertEqual(self.status([
            claim_row(8, "prompt", "A", 1, logged_at="2025-10-28T09:00:00+00:00"),
            claim_row(8, "repo", "B", 1, logged_at="2025-10-28T06:00:00+00:00"),
        ]), "unverified")

    def test_a_missing_timestamp_is_unverified_not_clean(self):
        self.assertEqual(self.status([
            claim_row(8, "prompt", "A", 1, logged_at=""),
            claim_row(8, "repo", "B", 1, logged_at="2025-10-28T06:00:00+00:00"),
        ]), "unverified")

    def test_one_arm_alone_has_no_order_to_break(self):
        self.assertEqual(self.status([
            claim_row(8, "repo", "B", 1, logged_at=""),
        ]), "clean")

    def test_a_marked_row_contaminates_the_week(self):
        self.assertEqual(self.status([
            claim_row(8, "prompt", "A", 1, logged_at="2025-10-28T09:00:00+00:00",
                      contaminated="true"),
            claim_row(8, "repo", "B", 1, logged_at="2025-10-28T11:00:00+00:00"),
        ]), "contaminated")


class ArmRowsTest(unittest.TestCase):
    def test_rows_without_an_arm_or_a_counted_rank_are_left_out(self):
        claims = pd.DataFrame([
            claim_row(8, "repo", "Counted", 1),
            claim_row(8, "repo", "PastDepth", 4),
            claim_row(8, "repo", "Unranked", np.nan),
            claim_row(8, "", "PreComparison", 1),
            claim_row(8, "naive", "NotLoggable", 1),
        ])
        claims["rank_within_arm"] = pd.to_numeric(claims["rank_within_arm"])
        self.assertEqual(list(arm_rows(claims)["player"]), ["Counted"])


class ContaminationTest(unittest.TestCase):
    def weekly(self):
        return pd.DataFrame([
            {"season": 2025, "week": 8, "arm": a, "order_status": "clean",
             "n": 3, "mean_fwd3": 10.0, "week_ceiling": 20.0}
            for a in ("naive", "prompt", "repo")
        ] + [
            {"season": 2025, "week": 9, "arm": a, "order_status": "contaminated",
             "n": 3, "mean_fwd3": 30.0, "week_ceiling": 40.0}
            for a in ("naive", "prompt", "repo")
        ])

    def test_a_contaminated_week_drops_for_every_arm(self):
        usable = usable_weeks(self.weekly(), strict_order=False)
        self.assertEqual(sorted(usable["week"].unique()), [8])
        # All three arms lose it, so no arm's mean is computed over a week set
        # the others do not share.
        self.assertEqual(len(usable), 3)

    def test_strict_order_also_drops_unverified(self):
        weekly = self.weekly()
        weekly.loc[weekly["week"] == 9, "order_status"] = "unverified"
        self.assertEqual(len(usable_weeks(weekly, strict_order=False)), 6)
        self.assertEqual(len(usable_weeks(weekly, strict_order=True)), 3)


class PairedTest(unittest.TestCase):
    def weekly(self):
        """PAR and raw points deliberately disagree in every row.

        The headline pairs on PAR. If it ever silently reverted to raw points
        the arithmetic below would still look reasonable, so the fixture makes
        the two answers different and the assertions name which one is right.
        """
        rows = []
        for week, (prompt, repo) in {8: (5.0, 9.0), 9: (7.0, 8.0), 10: (6.0, 6.0)}.items():
            rows.append({"season": 2025, "week": week, "arm": "prompt",
                         "order_status": "clean", "n": 3, "mean_par": prompt,
                         "mean_fwd3": prompt + 100.0, "week_ceiling": 20.0})
            rows.append({"season": 2025, "week": week, "arm": "repo",
                         "order_status": "clean", "n": 3, "mean_par": repo,
                         "mean_fwd3": repo + 200.0, "week_ceiling": 20.0})
        # Week 11 has a repo arm only: nothing to pair it against.
        rows.append({"season": 2025, "week": 11, "arm": "repo", "order_status": "clean",
                     "n": 3, "mean_par": 99.0, "mean_fwd3": 99.0,
                     "week_ceiling": 20.0})
        return pd.DataFrame(rows)

    def test_only_weeks_both_arms_covered_are_paired(self):
        pairs = paired_differences(self.weekly(), "repo", "prompt")
        self.assertEqual(list(pairs["week"]), [8, 9, 10])
        self.assertEqual(list(pairs["diff"]), [4.0, 1.0, 0.0])

    def test_pairing_reads_par_not_raw_points(self):
        # Raw points would put every difference at +100. PAR is the headline.
        par = paired_differences(self.weekly(), "repo", "prompt")
        raw = paired_differences(self.weekly(), "repo", "prompt", value="mean_fwd3")
        self.assertEqual(list(par["diff"]), [4.0, 1.0, 0.0])
        self.assertEqual(list(raw["diff"]), [104.0, 101.0, 100.0])

    def test_a_week_with_no_par_is_dropped_rather_than_paired_as_nan(self):
        # A position with nobody resolved on the wire leaves PAR undefined. That
        # week must leave the comparison, not enter it as a NaN difference that
        # silently shortens the interval.
        weekly = self.weekly()
        weekly.loc[(weekly["week"] == 9) & (weekly["arm"] == "repo"),
                   "mean_par"] = np.nan
        pairs = paired_differences(weekly, "repo", "prompt")
        self.assertEqual(list(pairs["week"]), [8, 10])

    def test_an_unpaired_week_cannot_move_the_paired_mean(self):
        # Week 11's repo arm scored 99 and must not leak into the comparison.
        pairs = paired_differences(self.weekly(), "repo", "prompt")
        self.assertAlmostEqual(pairs["diff"].mean(), 5.0 / 3)

    def test_bootstrap_interval_brackets_the_mean(self):
        diffs = np.array([4.0, 1.0, 0.0])
        lo, hi = bootstrap_ci(diffs, reps=2000)
        self.assertLess(lo, diffs.mean())
        self.assertGreater(hi, diffs.mean())

    def test_bootstrap_declines_a_single_observation(self):
        lo, hi = bootstrap_ci(np.array([4.0]))
        self.assertTrue(np.isnan(lo) and np.isnan(hi))


class SummaryTest(unittest.TestCase):
    def test_head_to_head_is_strict_and_naive_has_no_record(self):
        weekly = pd.DataFrame([
            # Week 8: repo ties naive. A tie is not a win.
            {"season": 2025, "week": 8, "arm": "naive", "order_status": "clean",
             "n": 3, "mean_par": 1.0, "mean_fwd3": 10.0, "week_ceiling": 20.0},
            {"season": 2025, "week": 8, "arm": "repo", "order_status": "clean",
             "n": 3, "mean_par": 1.0, "mean_fwd3": 10.0, "week_ceiling": 20.0},
            # Week 9: repo wins on PAR. It loses on raw points, which is the
            # case the metric exists for -- an arm can capture more fantasy
            # points and less value, by claiming a position it cannot field.
            {"season": 2025, "week": 9, "arm": "naive", "order_status": "clean",
             "n": 3, "mean_par": 1.0, "mean_fwd3": 18.0, "week_ceiling": 20.0},
            {"season": 2025, "week": 9, "arm": "repo", "order_status": "clean",
             "n": 3, "mean_par": 3.0, "mean_fwd3": 12.0, "week_ceiling": 20.0},
        ])
        summary = arm_summary(weekly).set_index("arm")
        self.assertAlmostEqual(summary.loc["repo", "beat_naive_share"], 0.5)
        self.assertTrue(np.isnan(summary.loc["naive", "beat_naive_share"]))
        self.assertAlmostEqual(summary.loc["repo", "ceiling_share"], (10 / 20 + 12 / 20) / 2)
        self.assertEqual(summary.loc["repo", "weeks"], 2)
        self.assertEqual(summary.loc["repo", "n"], 6)


class VerdictTest(unittest.TestCase):
    """The pre-registered rule, at its boundaries. Any edit here is an edit to
    the rule and should be argued for, not slipped in."""

    def test_neither_arm_clears_naive(self):
        self.assertEqual(verdict(0.2, 1.4, 1.2), "decoration")

    def test_decoration_is_checked_before_the_tie_band(self):
        # Both arms are far behind naive AND within the tie band of each other.
        # "The whole thing is decoration" is the finding, not "keep the prompt".
        self.assertEqual(verdict(-3.0, -2.5, 0.5), "decoration")

    def test_tie_band_when_an_arm_has_cleared_naive(self):
        self.assertEqual(verdict(2.0, 2.5, 0.5), "repo-adds-nothing")

    def test_the_tie_band_is_symmetric(self):
        # Prompt ahead by less than the band is still a tie, not a prompt win.
        self.assertEqual(verdict(3.0, 2.5, -0.5), "repo-adds-nothing")

    def test_repo_earns_its_keep(self):
        self.assertEqual(verdict(2.0, 4.0, REPO_MARGIN), "repo-earns-its-keep")

    def test_the_gap_between_the_thresholds_is_inconclusive(self):
        # Repo ahead by more than the tie band but less than the repo margin.
        gap = (TIE_BAND + REPO_MARGIN) / 2
        self.assertEqual(verdict(2.0, 2.0 + gap, gap), "inconclusive")

    def test_prompt_well_ahead_is_inconclusive_not_a_repo_result(self):
        self.assertEqual(verdict(5.0, 2.0, -3.0), "inconclusive")

    def test_naive_margin_boundary_is_inclusive(self):
        self.assertNotEqual(verdict(NAIVE_MARGIN, 0.0, 0.5), "decoration")
        self.assertEqual(verdict(NAIVE_MARGIN - 0.01, 0.0, 0.5), "decoration")


class GradeArmsTest(unittest.TestCase):
    def test_the_naive_arm_is_derived_and_the_logged_arms_are_read(self):
        panel = pd.DataFrame([
            panel_row("Hot", 8, pts=25.0, fwd3=4.0),
            panel_row("Warm", 8, pts=20.0, fwd3=5.0),
            panel_row("Cool", 8, pts=15.0, fwd3=6.0),
            panel_row("Quiet", 8, pts=1.0, fwd3=12.0),
        ])
        claims = pd.DataFrame([claim_row(8, "prompt", "Quiet", 1)])
        claims["rank_within_arm"] = pd.to_numeric(claims["rank_within_arm"])
        graded, problems = grade_arms(claims, panel)

        self.assertEqual(problems, [])
        naive = graded[graded["arm"] == "naive"]
        self.assertEqual(list(naive["player"]), ["Hot", "Warm", "Cool"])
        # Chasing last week's box score loses this week; that is the whole point
        # of running the benchmark rather than assuming it.
        weekly = week_means(graded).set_index("arm")
        self.assertAlmostEqual(weekly.loc["naive", "mean_fwd3"], 5.0)
        self.assertAlmostEqual(weekly.loc["prompt", "mean_fwd3"], 12.0)
        self.assertAlmostEqual(weekly.loc["prompt", "week_ceiling"], 12.0)

    def test_an_unscoreable_pick_is_reported_not_silently_dropped(self):
        panel = pd.DataFrame([panel_row("Hot", 8, pts=25.0, fwd3=4.0)])
        claims = pd.DataFrame([claim_row(8, "prompt", "Misspelt Name", 1)])
        claims["rank_within_arm"] = pd.to_numeric(claims["rank_within_arm"])
        graded, problems = grade_arms(claims, panel)
        self.assertEqual(len(problems), 1)
        self.assertIn("Misspelt Name", problems[0])
        self.assertTrue(graded[graded["arm"] == "prompt"].empty)


if __name__ == "__main__":
    unittest.main()


class ReplacementLevelsTest(unittest.TestCase):
    """The baseline PAR is measured against, and the properties it has to have.

    Each of these is a way the metric could quietly flatter one arm, and each
    would leave a table that looked fine.
    """

    def pool(self):
        """A wire pool shaped like a real one: a few starters and a long tail.

        The tail is the point. Most available quarterbacks are backups who do
        not play and score zero, which is what pulls the *mean* down to a level
        no startable quarterback is anywhere near. Testing against a pool of six
        evenly-spaced quarterbacks would make the metric look like it does
        nothing.
        """
        rows = []
        # Ten quarterbacks: three who play, seven who do not. Rank 2 -> 26.0.
        qb = [30.0, 28.0, 26.0] + [0.0] * 7
        for i, fwd3 in enumerate(qb):
            rows.append(panel_row(f"QB{i}", 8, pts=float(len(qb) - i), fwd3=fwd3,
                                  position="QB"))
        # Eight receivers, a flatter distribution. WR rank 6 is the seventh, 1.0.
        for i, fwd3 in enumerate([14.0, 12.0, 9.0, 7.0, 5.0, 3.0, 1.0, 0.0]):
            rows.append(panel_row(f"WR{i}", 8, pts=float(8 - i), fwd3=fwd3))
        return pd.DataFrame(rows)

    def test_the_level_is_the_position_rank_of_realised_outcomes(self):
        levels = replacement_levels(self.pool())
        self.assertAlmostEqual(levels["QB"], 26.0)
        self.assertAlmostEqual(levels["WR"], 1.0)

    def test_it_sits_well_above_the_pool_mean_which_is_the_whole_point(self):
        # The objection `with_edge` was written against: the median wire
        # quarterback is a backup who will not play, so the mean makes every
        # starter look like a huge edge. If these ever converge the metric has
        # stopped doing anything.
        pool = self.pool()
        levels = replacement_levels(pool)
        mean = pool[pool["position"] == "QB"]["fwd3"].mean()
        self.assertGreater(levels["QB"], mean)

    def test_unresolved_players_do_not_drag_the_level_down(self):
        # Counting a player whose window has not closed as a zero would lower
        # the baseline and inflate every arm's PAR at once.
        pool = pd.concat([self.pool(), pd.DataFrame([
            panel_row("Pending", 8, pts=9.0, fwd3=np.nan, position="QB"),
        ])], ignore_index=True)
        self.assertAlmostEqual(replacement_levels(pool)["QB"], 26.0)

    def test_the_level_does_not_depend_on_who_was_picked(self):
        # Arm-neutrality. The baseline is a function of the pool and the
        # position; nothing about an arm's ranking may reach it.
        pool = self.pool()
        shuffled = pool.sample(frac=1.0, random_state=7).reset_index(drop=True)
        self.assertEqual(replacement_levels(pool), replacement_levels(shuffled))

    def test_a_position_too_thin_to_have_a_replacement_is_absent(self):
        thin = pd.DataFrame([panel_row("Only", 8, pts=1.0, fwd3=5.0, position="TE")])
        self.assertNotIn("TE", replacement_levels(thin))


class ParScoringTest(unittest.TestCase):
    """PAR on graded rows: what it changes, and what it must not."""

    def panel(self):
        # Same shape as the pool above: streamable quarterbacks bunched at the
        # top, receivers spread out. QB replacement 26.0, WR replacement 1.0.
        rows = []
        qb = [30.0, 28.0, 26.0] + [0.0] * 7
        for i, fwd3 in enumerate(qb):
            rows.append(panel_row(f"QB{i}", 8, pts=float(30 - i), fwd3=fwd3,
                                  position="QB"))
        for i, fwd3 in enumerate([14.0, 12.0, 9.0, 7.0, 5.0, 3.0, 1.0]):
            rows.append(panel_row(f"WR{i}", 8, pts=float(10 - i), fwd3=fwd3))
        return pd.DataFrame(rows)

    def graded(self):
        claims = pd.DataFrame([
            claim_row(8, "repo", "WR0", 1, position="WR"),
            claim_row(8, "repo", "QB0", 2, position="QB"),
        ])
        claims["rank_within_arm"] = pd.to_numeric(claims["rank_within_arm"])
        graded, _ = grade_arms(claims, self.panel())
        return graded.set_index(["arm", "player"])

    def test_raw_points_are_kept_alongside_par(self):
        # The training target, and the thing a reader wants to see. Losing it
        # to the metric change would make the ledger less readable, not more.
        rows = self.graded()
        self.assertAlmostEqual(rows.loc[("repo", "QB0"), "actual_fwd3"], 30.0)
        self.assertAlmostEqual(rows.loc[("repo", "WR0"), "actual_fwd3"], 14.0)

    def test_par_is_the_outcome_less_the_position_baseline(self):
        rows = self.graded()
        # QB rank 2 -> third-best of [30, 28, 26, 0 x 7] = 26.0
        self.assertAlmostEqual(rows.loc[("repo", "QB0"), "repl_fwd3"], 26.0)
        self.assertAlmostEqual(rows.loc[("repo", "QB0"), "par"], 4.0)
        # WR rank 6 -> seventh-best of the seven = 1.0
        self.assertAlmostEqual(rows.loc[("repo", "WR0"), "repl_fwd3"], 1.0)
        self.assertAlmostEqual(rows.loc[("repo", "WR0"), "par"], 13.0)

    def test_par_reorders_picks_that_raw_points_ranks_the_other_way(self):
        # The reason the metric changed. On raw points the quarterback wins by
        # 16; on PAR the receiver wins by 9, because the quarterback you would
        # have streamed instead was nearly as good and the receiver's
        # replacement was not.
        rows = self.graded()
        self.assertGreater(rows.loc[("repo", "QB0"), "actual_fwd3"],
                           rows.loc[("repo", "WR0"), "actual_fwd3"])
        self.assertGreater(rows.loc[("repo", "WR0"), "par"],
                           rows.loc[("repo", "QB0"), "par"])

    def test_within_a_position_par_shifts_every_pick_by_the_same_amount(self):
        # Which is why the within-position ablation is invariant to the metric:
        # a constant per position cancels out of any same-position comparison.
        rows = self.graded().reset_index()
        for position, group in rows.groupby("position"):
            shifts = (group["actual_fwd3"] - group["par"]).round(9).unique()
            self.assertEqual(len(shifts), 1, msg=f"{position} shifted unevenly")

    def test_week_means_carry_both_metrics(self):
        claims = pd.DataFrame([claim_row(8, "repo", "WR0", 1, position="WR")])
        claims["rank_within_arm"] = pd.to_numeric(claims["rank_within_arm"])
        graded, _ = grade_arms(claims, self.panel())
        weekly = week_means(graded).set_index("arm")
        self.assertAlmostEqual(weekly.loc["repo", "mean_fwd3"], 14.0)
        self.assertAlmostEqual(weekly.loc["repo", "mean_par"], 13.0)


class HeadToHeadNaNTest(unittest.TestCase):
    def test_a_week_with_no_par_is_not_scored_as_a_loss(self):
        # `np.nan > x` is False. Counting an unpriceable week as a week the arm
        # lost would understate every contender against the benchmark, quietly
        # and in one direction.
        weekly = pd.DataFrame([
            {"season": 2025, "week": 8, "arm": "naive", "order_status": "clean",
             "n": 3, "mean_par": 1.0, "mean_fwd3": 10.0, "week_ceiling": 20.0},
            {"season": 2025, "week": 8, "arm": "repo", "order_status": "clean",
             "n": 3, "mean_par": 5.0, "mean_fwd3": 12.0, "week_ceiling": 20.0},
            {"season": 2025, "week": 9, "arm": "naive", "order_status": "clean",
             "n": 3, "mean_par": 1.0, "mean_fwd3": 10.0, "week_ceiling": 20.0},
            {"season": 2025, "week": 9, "arm": "repo", "order_status": "clean",
             "n": 3, "mean_par": np.nan, "mean_fwd3": 12.0, "week_ceiling": 20.0},
        ])
        summary = arm_summary(weekly).set_index("arm")
        # One priceable week, which repo won. Not two, of which it won one.
        self.assertAlmostEqual(summary.loc["repo", "beat_naive_share"], 1.0)

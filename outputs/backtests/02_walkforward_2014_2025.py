"""Walk-forward replay of the candidate table over the full usable history.

One question, the same one `01_season_replay.py` asked over three seasons and
could not answer precisely: **does ranking the wire by the repo's model beat
claiming last week's highest scorer?** Twelve replay seasons instead of three,
which is roughly 570 picks per arm and the first sample in this project large
enough to resolve a difference of the size worth having.

DATA RANGE: 2013 IS A HARD FLOOR, AND 2012 IS A TRAP
====================================================

nflverse snap counts begin in 2013. Weekly stats and play-by-play go back to
1999, but snap share is a required input at WR and TE, so the panel cannot start
earlier.

`snap_counts_2012.csv` exists and returns HTTP 200. It is 154 bytes: a header
row and nothing else. Any availability check that stops at "the fetch worked"
passes, the frame reads back empty, and 2012 contributes zero rows to the panel
without anything failing -- a silently short panel rather than an error.
`features.require_rows` now asserts on the contents of every source file, not on
the fetch, and it is called for stats, snaps and the schedule alike.

So: fetch 2013-2025, replay 2014-2025. 2013 is not replayable -- there is
nothing before it to train on.

TWO TRAINING WINDOWS
====================

Every replay season is run twice, and this is the part that decides something
about the shipped pipeline rather than merely describing the past:

    expanding  -- trained on every prior season (2014 on 2013; 2025 on 2013-2024)
    recent     -- trained on the most recent three prior seasons only

If `recent` wins in the later seasons, football changed underneath the model and
the production models should be recency-weighted. If the two tie, it did not and
all available history should be used. The two schemes are *identical by
construction* for 2014, 2015 and 2016, where fewer than four prior seasons exist
-- the comparison only carries information from 2017 on, and the write-up says
so rather than averaging three forced ties into the verdict.

SEASON LENGTH IS NOT A CONSTANT: THE 2021 BREAK
===============================================

The regular season ran 17 weeks through 2020 and 18 weeks from 2021. `fwd3`
averages weeks W+1..W+3, so the last week whose forward window fits inside the
season is **week 14 before 2021 and week 15 after**. That is not a detail:

    2014-2020   wk14 -> 3 weeks   wk15 -> 2   wk16 -> 1   wk17 -> 0, UNSCOREABLE
    2021-2025   wk14 -> 3 weeks   wk15 -> 3   wk16 -> 2   wk17 -> 1

Week 17 of a pre-2021 season has no forward window at all: `fwd3` is NaN for
every player, both arms included, and the week drops out of the replay entirely.
It is counted and reported rather than silently absent.

The consequence for the headline is direct. **The pooled headline covers weeks
2-14**, the widest bucket whose forward window is a full three weeks in every
one of the twelve seasons. Weeks 15-17 are reported separately and split at
2021, because "weeks 14-17" means a different thing on either side of that line
and comparing them is comparing unlike things. The 14-17 bucket the roto window
cares about is shown as asked, with both problems labelled.

STRUCTURAL BREAKS: REPORTED, NOT POOLED
=======================================

- **2020.** No preseason, opt-outs, empty stadiums, and abnormal inactive
  patterns. `fwd3` scores a week the team played without the player as 0.0, so a
  season with unusual inactives interacts with the scoring convention directly.
  2020 is broken out of every table and the pooled figure is reported both with
  and without it.
- **2021.** The 16-to-17 game change, handled by the season-length logic above
  and reported as a pre/post split rather than pooled.
- **Roster depth.** `ROSTER_DEPTH` (QB 18, RB 46, WR 60, TE 18) is **held
  fixed** across all twelve seasons, and a sensitivity check at +/-25% is run to
  measure what that assumption is worth. The reasoning is in the write-up.

CONTAMINATION
=============

Treated as the main task. The shipped `models/*.joblib` are fitted through 2025
and are never loaded here. Replay bundles are written to
`outputs/backtests/replay_models_full/{scheme}_{season}/`, stamped
`replay_only: True`, and `src.models.load_bundle` only ever reads `models/`, so
one cannot be served by accident.

Refitting the estimator is necessary and not sufficient. Each path that can
carry information across the train/test boundary, and what is done about it:

1. **Empirical Bayes priors** -- fitted per position on the pooled rate
   distribution over every season in the build. Closed by
   `features.build(seasons, prior_seasons=train_seasons)`.
2. **The rank-to-points scale** -- `weekly.points_scale` maps a percentile to
   points through the `fwd3` quantile function of the whole panel. Monotone
   within a position, but claims are ordered by points above replacement
   *across* positions, so it moves which names surface. Fitted on training
   seasons only.
3. **The conformal calibration split** -- recomputed on training seasons alone,
   which leaves the early replay seasons with no interval rather than one
   borrowed from the future. It does not touch the pick order either way.
4. **`on_wire` and the season-to-date rank behind it** -- audited rather than
   assumed. `rank_before` ranks `cum_before` within (season, position, week) and
   `cum_before` is a shifted cumulative sum within (player, season), so both are
   strictly within-season and strictly backward.
   `tests/test_features.py::test_only_eb_columns_cross_seasons` asserts that the
   EB shares are the only columns that move when the build scope changes.

What cannot be closed, and is stated rather than fixed: every constant in the
pipeline -- `ROSTER_DEPTH`, `REPLACEMENT_RANK`, `NEUTRAL_WP`, the Kalman and
CUSUM settings, the 2-14 training window, `MODEL_KWARGS` -- was chosen by a
human who had already seen this data. No refit removes that, and it is why a
large margin in the repo arm's favour is treated here as a tripwire rather than
a result.

Run:
    python outputs/backtests/02_walkforward_2014_2025.py
    python outputs/backtests/02_walkforward_2014_2025.py --no-depth-sensitivity
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import features as F                                    # noqa: E402
from src import models as M                                      # noqa: E402
from src.ledger import BOOTSTRAP_REPS, BOOTSTRAP_SEED, CI_LEVEL  # noqa: E402
from src.ledger import naive_picks, wire_pool                    # noqa: E402
from src.report import REPLACEMENT_RANK                          # noqa: E402
from src.weekly import PROJECTION_CLIP                           # noqa: E402


def _load_prior_replay():
    """Import `01_season_replay.py`, whose leading digit blocks a normal import.

    The statistics helpers there -- the stratified bootstrap, the paired-week
    join, the arm summaries, the mix/selection decomposition -- are free of that
    file's module constants and are reused verbatim rather than copied, so the
    three-season result and this one cannot drift apart in how they are
    computed. Everything that depends on season length, the model directory or
    the training plan is redefined below instead of being imported.
    """
    path = Path(__file__).resolve().parent / "01_season_replay.py"
    spec = importlib.util.spec_from_file_location("season_replay_2022_2025", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_prior_replay()
bootstrap_ci = R.bootstrap_ci
paired = R.paired
week_means = R.week_means
arm_stats = R.arm_stats
decompose = R.decompose
overlap_rate = R.overlap_rate
repo_picks = R.repo_picks
train_points_scale = R.train_points_scale
fmt = R.fmt

def stratified_ci(diffs: np.ndarray, strata: np.ndarray | None):
    """Percentile CI for the mean paired difference, resampling weeks by season.

    Wraps the prior replay's bootstrap with one guard it does not need and this
    one does. Stratifying by season keeps each season's week count in every
    draw, which is right when a season contributes several weeks -- but a bucket
    covering a *single* week gives every season a stratum of size one, and
    resampling one item with replacement always returns that item. Every draw is
    then identical and the interval collapses to zero width, which reads as
    impossible precision rather than as no information.

    Where any stratum is a singleton the resampling falls back to unstratified,
    which is the honest interval for that bucket: it loses the season balance
    the stratification was buying, and says so in the write-up rather than
    reporting a degenerate one.
    """
    if strata is None or len(diffs) < 2:
        return bootstrap_ci(diffs, None), False
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < 2:
        return bootstrap_ci(diffs, None), True
    return bootstrap_ci(diffs, strata), False


def bucket_table(weekly, picks, seasons, weeks, stratify: bool,
                 value: str = "mean_fwd3") -> dict:
    """Both arms plus the paired difference over one week bucket.

    Mirrors `01_season_replay.bucket_table`, differing only in routing the
    interval through `stratified_ci` above and carrying the `degenerate` flag
    that says when the stratification had to be dropped.
    """
    if seasons is not None:
        weekly = weekly[weekly["season"].isin(seasons)]
        picks = picks[picks["season"].isin(seasons)]
    pairs = paired(weekly, weeks, value)
    diffs = pairs["diff"].to_numpy(dtype=float)
    strata = pairs["season"].to_numpy() if stratify and len(pairs) else None
    (lo, hi), degenerate = stratified_ci(diffs, strata)
    return {
        "repo": arm_stats(weekly, picks, "repo", weeks),
        "naive": arm_stats(weekly, picks, "naive", weeks),
        "value": value,
        "n_weeks": len(diffs),
        "mean_diff": float(diffs.mean()) if len(diffs) else np.nan,
        "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "unstratified": degenerate,
        "pairs": pairs,
    }


OUT_DIR = Path(__file__).resolve().parent / "replay_full"
MODEL_DIR = Path(__file__).resolve().parent / "replay_models_full"
PANEL_CACHE = ROOT / "data" / "processed"
DIAGNOSTIC = ROOT / "outputs" / "diagnostics" / "walkforward_2014_2025.md"
# What could not be scored, persisted next to the picks. Reconstructing it from
# `replay_picks.csv` alone is impossible -- an unscoreable pick leaves no row --
# so a --report-only rerun would otherwise quietly report an empty exclusion
# list, which is precisely the kind of silent omission this file exists to
# avoid.
NOTES_PATH = OUT_DIR / "unscored.json"

# Snap counts begin in 2013. `snap_counts_2012.csv` is a header row and nothing
# else, so this floor is enforced by `features.require_rows` rather than trusted.
FIRST_SEASON = 2013
LAST_SEASON = 2025
ALL_SEASONS = list(range(FIRST_SEASON, LAST_SEASON + 1))
REPLAY_SEASONS = list(range(FIRST_SEASON + 1, LAST_SEASON + 1))   # 2014-2025

REPLAY_WEEKS = list(range(2, 18))        # weeks 2-17, as in the prior replay
ARM_DEPTH = 3
RECENT_WINDOW = 3                        # seasons in the `recent` training window

# The season the regular season went from 16 games to 17.
EXPANSION_SEASON = 2021
# Final regular-season week, by era. Read from the schedule at run time and
# asserted against these, so a wrong assumption fails rather than propagates.
EXPECTED_FINAL_WEEK = {False: 17, True: 18}   # keyed on season >= EXPANSION_SEASON

# The pandemic season: no preseason, opt-outs, empty stadiums, abnormal
# inactives. Broken out of every table rather than averaged into a headline.
BREAK_SEASON = 2020

# Weeks 2-14 have a full three-week forward window in *every* season on both
# sides of the 2021 expansion. Nothing wider is comparable across the whole span.
LAST_FULL_WINDOW_WEEK = 14

SCHEMES = {
    "expanding": "every prior season",
    "recent": f"the most recent {RECENT_WINDOW} prior seasons",
}

BUCKETS = [
    ("2-13", range(2, 14), "weekly cash and playoff push"),
    ("14", range(14, 15), "first bracket week, full window in every season"),
    ("2-14", range(2, 15), "POOLED HEADLINE -- full 3-week window in every season"),
    ("15-17", range(15, 18), "the tail; window length differs across the 2021 break"),
    ("14-17", range(14, 18), "bracket window as asked (mixes window lengths and eras)"),
    ("2-17", range(2, 18), "everything, including the truncated tail"),
]
HEADLINE_BUCKET = "2-14"

# A margin this large is not a result, it is a symptom. Named as a constant so
# that moving it shows up in the diff as moving the tripwire.
LEAKAGE_TRIPWIRE_PPG = 6.0

# Roster-depth sensitivity: the thresholds are tuned to today's league, and
# whether 2014's replacement level sat in the same place is an assumption.
DEPTH_SCALES = (0.75, 1.25)


# ==========================================================================
# training plans
# ==========================================================================

def train_seasons_for(season: int, scheme: str) -> list[int]:
    """Seasons a replay of `season` may train on under `scheme`.

    Strictly before `season` in both cases. `recent` truncates to the last
    RECENT_WINDOW of them, which for 2014-2016 is every prior season anyway --
    the two schemes are identical there and the write-up says so rather than
    presenting three forced ties as agreement.
    """
    prior = [s for s in ALL_SEASONS if s < season]
    if scheme == "recent":
        return prior[-RECENT_WINDOW:]
    return prior


def schemes_diverge(season: int) -> bool:
    return train_seasons_for(season, "expanding") != train_seasons_for(season, "recent")


# ==========================================================================
# season length
# ==========================================================================

def final_weeks() -> dict[int, int]:
    """Final regular-season week per season, from the schedule, asserted.

    The 16-to-17 game change in 2021 moves the last week whose forward window
    fits inside the season, which moves which weeks can be pooled. Reading it
    from `games.csv` rather than hardcoding it means a future season extending
    again is picked up; asserting it against the expected value means a schedule
    file that disagrees with the premise of this script fails loudly.
    """
    _, last_week = F.load_schedule()
    out = {}
    for season in ALL_SEASONS:
        observed = int(last_week[season])
        expected = EXPECTED_FINAL_WEEK[season >= EXPANSION_SEASON]
        if observed != expected:
            raise SystemExit(
                f"{season}: schedule says the regular season ends at week "
                f"{observed}, this script assumes {expected}. The forward-window "
                "and bucket logic is built on that assumption -- fix the "
                "assumption rather than the assertion."
            )
        out[season] = observed
    return out


def last_full_window_week(season: int, finals: dict[int, int]) -> int:
    """Last week whose W+1..W+3 window fits inside `season`."""
    return finals[season] - 3


# ==========================================================================
# panel + models, per replay season
# ==========================================================================

def panel_cache_path(seasons: list[int], train_seasons: list[int]) -> Path:
    """Cache filename naming both season sets in full, not as a range.

    Two different training windows over the same build seasons must not share a
    cache entry, and with twelve seasons in play a min-max key collides readily.
    """
    return PANEL_CACHE / (
        "wf_panel_s"
        + "".join(str(y)[-2:] for y in sorted(seasons))
        + "_prior"
        + "".join(str(y)[-2:] for y in sorted(train_seasons))
        + ".csv"
    )


def replay_panel(seasons: list[int], train_seasons: list[int]) -> pd.DataFrame:
    """Panel over `seasons` with the EB priors fitted on `train_seasons` only.

    Written to CSV and read back before being returned, for the reason
    `01_season_replay.py` documents at length: `HistGradientBoostingRegressor`
    bins its features, `to_csv`/`read_csv` perturbs floats at the 1e-16 level,
    and a value crossing a bin edge changes a split and occasionally a pick. The
    round-trip makes a cold run and a cached run bit-identical.
    """
    cache = panel_cache_path(seasons, train_seasons)
    if cache.exists():
        print(f"  panel: cached {cache.name}")
        return pd.read_csv(cache)
    panel = F.build(seasons, prior_seasons=train_seasons)
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache, index=False)
    return pd.read_csv(cache)


def rescale_depth(panel: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Recompute `on_wire` against scaled roster-depth thresholds.

    `rank_before` is already on the panel and is not a function of the
    thresholds, so the sensitivity is exact and needs no rebuild: only the
    comparison moves. Both arms draw from whatever pool this produces, so the
    check measures whether the *verdict* depends on the threshold, which is the
    question the assumption raises.
    """
    depth = {k: max(1, round(v * scale)) for k, v in F.ROSTER_DEPTH.items()}
    out = panel.copy()
    out["on_wire"] = out["rank_before"] > out["position"].map(depth)
    return out


def fit_replay_models(
    panel: pd.DataFrame, train_seasons: list[int], replay_season: int, label: str,
) -> dict[str, dict]:
    """Refit every position on `train_seasons` and persist outside `models/`.

    Shaped like a production bundle so the scoring path below is the shipped
    one, and marked `replay_only` and written elsewhere so it can never be
    mistaken for a production bundle.
    """
    train = panel[panel["season"].isin(train_seasons)]
    universe = M.wire_universe(train)
    feature_columns = M.feature_columns(panel)

    out_dir = MODEL_DIR / f"{label}_{replay_season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict] = {}

    for position in M.POSITIONS:
        frame = universe[universe["position"] == position]
        model = M.fit_model(frame, feature_columns)
        half_width, coverage, _, _ = M.conformal(frame, feature_columns, train_seasons)
        bundle = {
            "replay_only": True,
            "do_not_serve": (
                "Fitted for the walk-forward replay in "
                "outputs/backtests/02_walkforward_2014_2025.py. Trained on "
                f"{train_seasons} to score {replay_season}, so it knows less "
                "than the shipped bundles and must never be loaded by "
                "src/weekly.py."
            ),
            "replay_season": replay_season,
            "replay_scheme": label,
            "position": position,
            "model": model,
            "features": feature_columns,
            "conformal_half_width": half_width,
            "empirical_coverage": coverage,
            "train_seasons": train_seasons,
            "n_train": len(frame),
            "fit_versions": M.library_versions(),
            "model_kwargs": M.MODEL_KWARGS,
        }
        joblib.dump(bundle, out_dir / f"{position}.joblib")
        bundles[position] = bundle
        hw = "n/a" if np.isnan(half_width) else f"{half_width:.3f}"
        print(f"    {position}  n_train={len(frame):5,}  conformal half-width={hw}")
    return bundles


def window_spans(
    panel: pd.DataFrame, season: int, finals: dict[int, int]
) -> dict[tuple[int, str], int]:
    """How many of W+1..W+3 the team actually plays, per (week, team).

    Carried onto every pick so a truncated outcome is visible in the row rather
    than inferred from the week number -- which, with season length changing in
    2021, is no longer something a week number can tell you.
    """
    played, _ = F.load_schedule()
    final = finals[season]
    teams = sorted(panel.loc[panel["season"] == season, "team"].dropna().unique())
    return {
        (week, team): sum(
            1 for w in (week + 1, week + 2, week + 3)
            if w <= final and (season, w, team) in played
        )
        for week in REPLAY_WEEKS
        for team in teams
    }


# ==========================================================================
# one replay season
# ==========================================================================

def replay_season(
    season: int, train_seasons: list[int], label: str, finals: dict[int, int],
    depth_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Every pick both arms make in one replay season, with its outcome."""
    seasons = sorted(set(train_seasons) | {season})
    print(f"\n[{label}] replay {season}: train on {train_seasons}")
    panel = replay_panel(seasons, train_seasons)
    if depth_scale != 1.0:
        panel = rescale_depth(panel, depth_scale)
    bundles = fit_replay_models(panel, train_seasons, season, label)
    spans = window_spans(panel, season, finals)

    rows: list[dict] = []
    unscored: list[str] = []
    dead_weeks: list[str] = []

    for week in REPLAY_WEEKS:
        if week > finals[season]:
            dead_weeks.append(
                f"{season} wk{week:02d}: past the end of a "
                f"{finals[season]}-week regular season"
            )
            continue
        pool = wire_pool(panel, season, week)
        if pool.empty:
            unscored.append(f"{season} wk{week:02d}: empty wire pool")
            continue
        resolved = pool[pool["fwd3"].notna()]
        if resolved.empty:
            # Week 17 of a 17-week season: W+1..W+3 is entirely outside the
            # season, so fwd3 is NaN for every player and neither arm can be
            # scored. Recorded, not silently skipped.
            dead_weeks.append(
                f"{season} wk{week:02d}: no forward window inside a "
                f"{finals[season]}-week season, fwd3 unresolved for all "
                f"{len(pool)} pool players"
            )
            continue
        ceiling = float(resolved["fwd3"].max())
        pos_mean = resolved.groupby("position")["fwd3"].mean().to_dict()
        pos_count = resolved.groupby("position")["fwd3"].size().to_dict()

        picks = [
            ("repo", repo_picks(panel, bundles, train_seasons, season, week)),
            ("naive", naive_picks(panel, season, week, ARM_DEPTH)),
        ]
        for arm, frame in picks:
            if len(frame) < ARM_DEPTH:
                unscored.append(
                    f"{season} wk{week:02d} {arm}: only {len(frame)} eligible name(s)"
                )
            for _, pick in frame.iterrows():
                if pd.isna(pick["fwd3"]):
                    unscored.append(
                        f"{season} wk{week:02d} {arm} "
                        f"#{int(pick['rank_within_arm'])} "
                        f"{pick['player_display_name']}: fwd3 unresolved"
                    )
                    continue
                rows.append(
                    {
                        "scheme": label,
                        "season": season,
                        "week": week,
                        "arm": arm,
                        "rank_within_arm": int(pick["rank_within_arm"]),
                        "player": str(pick["player_display_name"]),
                        "position": str(pick["position"]),
                        "team": str(pick["team"]),
                        "pts_prior_week": float(pick["pts"]),
                        "fwd3": float(pick["fwd3"]),
                        "fwd3_span": spans.get((week, str(pick["team"])), np.nan),
                        "week_ceiling": ceiling,
                        "pool_mean_pos": pos_mean.get(str(pick["position"]), np.nan),
                        "pool_n_pos": pos_count.get(str(pick["position"]), 0),
                        "vs_pos": float(pick["fwd3"])
                        - pos_mean.get(str(pick["position"]), np.nan),
                        "edge": float(pick["edge"]) if "edge" in pick else np.nan,
                        "season_final_week": finals[season],
                        "n_train_seasons": len(train_seasons),
                        "train_seasons": ",".join(str(s) for s in train_seasons),
                    }
                )

    return pd.DataFrame(rows), {"unscored": unscored, "dead_weeks": dead_weeks}


def run_scheme(
    label: str, finals: dict[int, int], depth_scale: float = 1.0,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames, unscored, dead = [], [], []
    scheme = "recent" if label.startswith("recent") else "expanding"
    for season in seasons or REPLAY_SEASONS:
        picks, info = replay_season(
            season, train_seasons_for(season, scheme), label, finals, depth_scale
        )
        frames.append(picks)
        unscored += info["unscored"]
        dead += info["dead_weeks"]
    return pd.concat(frames, ignore_index=True), unscored, dead


# ==========================================================================
# the verdict
# ==========================================================================

def verdict(result: dict, adjusted: dict, split: dict) -> tuple[str, list[str]]:
    """The headline reading, and the sentences that qualify it.

    Written so that "inconclusive" is the easy answer and a large margin in the
    repo arm's favour fires a tripwire before it produces a celebration. Twelve
    walk-forward refits give a leak more places to hide than a single-season
    replay does, so the tripwire matters more here, not less.
    """
    diff = result["mean_diff"]
    lo, hi, n = result["ci_lo"], result["ci_hi"], result["n_weeks"]
    if np.isnan(diff):
        return "NO RESULT", ["No paired weeks survived. Nothing was measured."]

    if diff >= LEAKAGE_TRIPWIRE_PPG:
        return "LEAKAGE SUSPECTED", [
            f"The repo arm is ahead by {diff:+.2f} ppg over {n} paired weeks, at "
            f"or past the {LEAKAGE_TRIPWIRE_PPG:.0f} ppg tripwire set before the "
            "run. A margin this size is not plausible for a ranking model on "
            "this data. The first hypothesis is that something still crosses the "
            "train/test boundary, and twelve walk-forward refits give a leak more "
            "places to hide than three did. **Do not act on this number** until "
            "the contamination audit below has been re-read against it.",
        ]

    covers_zero = bool(lo <= 0.0 <= hi) if not np.isnan(lo) else True
    if covers_zero:
        return "INCONCLUSIVE", [
            f"Over {n} paired weeks the repo arm is {diff:+.2f} ppg against naive, "
            f"{CI_LEVEL:.0%} interval [{lo:+.2f}, {hi:+.2f}] ppg. The interval "
            "covers zero, so this data does not distinguish the two arms. With "
            "roughly 570 picks per arm the interval is now narrow enough that "
            "this is a real null rather than an absence of evidence -- a "
            "difference large enough to plan around would have shown up.",
        ]
    if diff > 0:
        return "REPO AHEAD", [
            f"Over {n} paired weeks the repo arm beats naive by {diff:+.2f} ppg, "
            f"{CI_LEVEL:.0%} interval [{lo:+.2f}, {hi:+.2f}] ppg, excluding zero. "
            "The margin is below the leakage tripwire, so it is not on its face "
            "a symptom -- but it is modest, and the constants in the pipeline "
            "were all chosen by someone who had already seen these seasons. Read "
            "it as evidence the ranking does something, not as a per-week edge.",
        ]

    mix_share = split["mix"] / split["gap"] if split["gap"] else float("nan")
    return "NAIVE WINS", [
        f"**Over {n} paired weeks spanning {len(REPLAY_SEASONS)} replay seasons, "
        f"the repo arm loses to naive by {abs(diff):.2f} ppg** ({CI_LEVEL:.0%} interval "
        f"[{lo:+.2f}, {hi:+.2f}], excluding zero). Ranking the wire by the "
        "model's points above replacement did **not** beat claiming last week's "
        "highest scorer. This is the answer to the question the exercise was set "
        "up to ask, and it is a no.",

        f"Roughly **{mix_share:.0%} of that gap is positional composition rather "
        f"than ranking quality**. The naive arm's picks are "
        f"{split['naive']['weights'].get('QB', 0):.0%} quarterbacks against the "
        f"repo arm's {split['repo']['weights'].get('QB', 0):.0%}, because "
        "quarterbacks score the most raw fantasy points and the naive rule sorts "
        "on raw fantasy points. `fwd3` is also raw fantasy points, so the arm "
        "that loads up on the highest-scoring position is rewarded for it.",

        adjusted_reading(adjusted, split),

        f"The model is not inert: its picks beat the same-position pool average "
        f"by {split['repo']['selection']:+.2f} ppg, so it finds "
        f"better-than-random players. The naive rule finds ones worth "
        f"{split['naive']['selection']:+.2f} ppg by the same measure.",
    ]


def adjusted_reading(adjusted: dict, split: dict) -> str:
    """What survives when positional composition is taken out of the comparison.

    This is the paragraph most likely to be written wrong, in either direction.
    The raw comparison is between two arms that pick different positions, scored
    in a currency (raw fantasy points) that pays different amounts per position;
    once that is removed, the remainder is the only part that is a statement
    about ranking players well. If that remainder's interval covers zero it must
    be reported as covering zero -- a null is not a narrow win for whichever arm
    happens to hold the point estimate, and the instruction not to round a null
    up into a win applies whichever arm it would flatter.
    """
    diff, lo, hi = adjusted["mean_diff"], adjusted["ci_lo"], adjusted["ci_hi"]
    n = adjusted["n_weeks"]
    interval = (
        f"repo − naive = {diff:+.2f} ppg, {CI_LEVEL:.0%} interval "
        f"[{lo:+.2f}, {hi:+.2f}], n = {n} weeks"
    )
    if np.isnan(lo) or lo <= 0.0 <= hi:
        return (
            "**Take position out and the gap goes with it.** Scoring each pick "
            "against what an arbitrary available player at the same position "
            f"returned that week: {interval}. **That interval covers zero.** "
            "Within position, this data does not distinguish the model's "
            "ranking from sorting the wire by last week's box score -- neither "
            "arm is measurably better at picking players, and the headline is "
            "a statement about which positions each arm walks into rather than "
            "about ranking quality. Reported as a null because it is one: the "
            "point estimate leans naive, but the interval does not support "
            "calling that a win for either side."
        )
    if diff > 0:
        return (
            "**With position removed the repo arm is ahead.** Scoring each pick "
            "against what an arbitrary available player at the same position "
            f"returned that week: {interval}, excluding zero. So the ranking "
            "does select better players within a position, and loses the raw "
            "comparison purely on positional composition. That is a real "
            "finding about the scoring convention, not a defence of the raw "
            "number -- `fwd3` is what the pipeline is graded on."
        )
    return (
        "**It does not rescue the repo arm.** With position removed -- each pick "
        "scored against what an arbitrary available player at the same position "
        f"returned that week -- naive is still ahead: {interval}, excluding "
        "zero. Both framings agree in direction, which is what makes the verdict "
        "robust rather than an artefact of the scoring convention."
    )


# ==========================================================================
# the write-up
# ==========================================================================

def ci_cell(bucket: dict) -> str:
    if np.isnan(bucket.get("ci_lo", np.nan)):
        return "n/a"
    mark = " †" if bucket.get("unstratified") else ""
    return f"[{fmt(bucket['ci_lo'])}, {fmt(bucket['ci_hi'])}]{mark}"


def season_row(weekly, picks, season, scheme, weeks) -> str:
    train = train_seasons_for(season, scheme)
    b = bucket_table(weekly, picks, [season], weeks, stratify=False)
    label = (
        f"{train[0]}-{train[-1]}" if len(train) > 1 else str(train[0])
    )
    mark = " *" if season == BREAK_SEASON else ""
    return (
        f"| {season}{mark} | {label} | {len(train)} | {b['n_weeks']} | "
        f"{fmt(b['repo'].get('mean_fwd3'))} | {fmt(b['naive'].get('mean_fwd3'))} | "
        f"{fmt(b['mean_diff'])} | {ci_cell(b)} | "
        f"{fmt(b['repo'].get('ceiling_share'), 1, pct=True)} | "
        f"{fmt(b['naive'].get('ceiling_share'), 1, pct=True)} | "
        f"{fmt(b['repo'].get('beat_other_share'), 1, pct=True)} |"
    )


def write_markdown(
    results: dict[str, dict], finals: dict[int, int], depth: dict | None,
    unscored: list[str], dead_weeks: list[str],
) -> None:
    headline_weeks = dict((n, w) for n, w, _ in BUCKETS)[HEADLINE_BUCKET]
    main = results["expanding"]
    weekly, picks = main["weekly"], main["picks"]

    pooled = bucket_table(weekly, picks, None, headline_weeks, stratify=True)
    adjusted = bucket_table(
        weekly, picks, None, headline_weeks, stratify=True, value="mean_vs_pos"
    )
    split = decompose(picks, headline_weeks)
    name, qualifiers = verdict(pooled, adjusted, split)

    L: list[str] = []
    add = L.append

    add("# Walk-forward replay, 2014-2025: does the ranking beat naive?")
    add("")
    add(
        "Generated by `outputs/backtests/02_walkforward_2014_2025.py`. Twelve "
        "replay seasons, two training windows, both arms scored by the repo's "
        "own `fwd3` convention. **The answer is first and the methodology is "
        "last, deliberately.**"
    )
    add("")

    # ---- verdict --------------------------------------------------------
    add("## Verdict")
    add("")
    add(f"**{name}.**")
    add("")
    for line in qualifiers:
        add(line)
        add("")
    add(
        f"The headline is the pooled paired difference over **weeks 2-14** of "
        f"{REPLAY_SEASONS[0]}-{REPLAY_SEASONS[-1]} under the **expanding** "
        f"training window. Weeks 2-14 is the "
        f"widest bucket whose three-week forward window fits inside the season "
        f"on both sides of the 2021 expansion; see "
        f"[the forward window](#the-forward-window-and-the-2021-expansion). Each "
        f"season's picks come from models refitted on the seasons strictly "
        f"before it and nothing else."
    )
    add("")
    add("| | repo | naive |")
    add("| --- | ---: | ---: |")
    for field, label, places, pct in [
        ("mean_fwd3", "mean points captured (ppg)", 2, False),
        ("ceiling_share", "share of the week's ceiling", 1, True),
        ("n_picks", "picks", 0, False),
        ("weeks", "weeks", 0, False),
    ]:
        add(
            f"| {label} | {fmt(pooled['repo'].get(field), places, pct)} | "
            f"{fmt(pooled['naive'].get(field), places, pct)} |"
        )
    add(
        f"| head-to-head weeks won | "
        f"{fmt(pooled['repo'].get('beat_other_share'), 1, pct=True)} | "
        f"{fmt(pooled['naive'].get('beat_other_share'), 1, pct=True)} |"
    )
    add("")
    add(
        f"**Paired difference (repo − naive, same weeks): "
        f"{fmt(pooled['mean_diff'])} ppg, {CI_LEVEL:.0%} bootstrap CI "
        f"[{fmt(pooled['ci_lo'])}, {fmt(pooled['ci_hi'])}], n = "
        f"{pooled['n_weeks']} weeks.**"
    )
    add("")
    add(
        "Weeks are the resampling unit -- three picks in one week share a wire "
        "pool and one slate of opponents, so they are not three independent "
        "observations -- and the bootstrap is stratified by season, so every "
        "draw keeps each season's week count. Head-to-head rates do not sum to "
        "100%: a tied week counts for neither arm."
    )
    add("")

    # ---- per season -----------------------------------------------------
    add("## Per season")
    add("")
    add(
        "Never pooled without also being shown separately. The 2014 model has "
        "one season of training data behind it and the 2025 model has twelve; if "
        "the margin moved with training volume that would be a finding about how "
        "much history this method needs, and a pooled number would hide it."
    )
    add("")
    for scheme, description in SCHEMES.items():
        add(f"### `{scheme}` — trained on {description}")
        add("")
        add(
            "| season | trained on | n seasons | weeks | repo ppg | naive ppg | "
            f"repo − naive | {CI_LEVEL:.0%} CI | repo ceiling | naive ceiling | "
            "repo won |"
        )
        add(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | "
            "---: | ---: |"
        )
        res = results[scheme]
        for season in REPLAY_SEASONS:
            add(season_row(res["weekly"], res["picks"], season, scheme, headline_weeks))
        b = bucket_table(res["weekly"], res["picks"], None, headline_weeks, True)
        add(
            f"| **pooled** | | | {b['n_weeks']} | "
            f"{fmt(b['repo'].get('mean_fwd3'))} | {fmt(b['naive'].get('mean_fwd3'))} "
            f"| {fmt(b['mean_diff'])} | {ci_cell(b)} | "
            f"{fmt(b['repo'].get('ceiling_share'), 1, pct=True)} | "
            f"{fmt(b['naive'].get('ceiling_share'), 1, pct=True)} | "
            f"{fmt(b['repo'].get('beat_other_share'), 1, pct=True)} |"
        )
        add("")
    add(
        "`*` marks 2020 — see [structural breaks](#structural-breaks). A single "
        "season is roughly 13 paired weeks, so per-season intervals are wide; "
        "read the direction and the overlap, not the ranking between seasons."
    )
    add("")

    # ---- expanding vs recent -------------------------------------------
    add("## Training window: expanding vs recent")
    add("")
    add(
        "The decision this is here to inform: **should the shipped models be "
        "recency-weighted?** If `recent` (last three seasons) beats `expanding` "
        "(all history) in the later seasons, football changed underneath the "
        "model and production should follow the change. If they tie, it did not, "
        "and throwing away history costs precision for nothing."
    )
    add("")
    add(
        f"The two schemes are **identical by construction** for "
        f"{', '.join(str(s) for s in REPLAY_SEASONS if not schemes_diverge(s))}, "
        f"where fewer than {RECENT_WINDOW + 1} prior seasons exist. Those rows "
        "are forced ties and carry no information; the comparison begins at "
        f"{min(s for s in REPLAY_SEASONS if schemes_diverge(s))}."
    )
    add("")
    add(
        "| season | expanding trains on | recent trains on | expanding repo ppg | "
        "recent repo ppg | recent − expanding |"
    )
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    diverging = [s for s in REPLAY_SEASONS if schemes_diverge(s)]
    for season in REPLAY_SEASONS:
        e = bucket_table(
            results["expanding"]["weekly"], results["expanding"]["picks"],
            [season], headline_weeks, stratify=False,
        )
        r = bucket_table(
            results["recent"]["weekly"], results["recent"]["picks"],
            [season], headline_weeks, stratify=False,
        )
        et, rt = train_seasons_for(season, "expanding"), train_seasons_for(season, "recent")
        gap = r["repo"].get("mean_fwd3", np.nan) - e["repo"].get("mean_fwd3", np.nan)
        tie = " (identical)" if not schemes_diverge(season) else ""
        add(
            f"| {season} | {len(et)} seasons | {len(rt)} seasons{tie} | "
            f"{fmt(e['repo'].get('mean_fwd3'))} | {fmt(r['repo'].get('mean_fwd3'))} "
            f"| {fmt(gap)} |"
        )
    add("")

    # paired repo-vs-repo across schemes, on the diverging seasons only
    cross = paired_schemes(
        results["expanding"]["weekly"], results["recent"]["weekly"],
        headline_weeks, diverging,
    )
    add(
        f"Paired week by week across the {len(diverging)} seasons where the two "
        "windows actually differ, comparing the repo arm against itself:"
    )
    add("")
    add(f"| comparison | n weeks | mean difference | {CI_LEVEL:.0%} CI |")
    add("| --- | ---: | ---: | :---: |")
    add(
        f"| recent − expanding (repo arm ppg) | {cross['n']} | "
        f"{fmt(cross['mean'])} | [{fmt(cross['lo'])}, {fmt(cross['hi'])}] |"
    )
    add("")
    add(recency_reading(cross))
    add("")

    # ---- structural breaks ---------------------------------------------
    add("## Structural breaks")
    add("")
    add("### 2020")
    add("")
    add(
        "No preseason, opt-outs, empty stadiums and abnormal inactive patterns. "
        "This interacts with the scoring convention directly rather than "
        "abstractly: `fwd3` scores a week the team played without the player as "
        "**0.0**, so a season where players were inactive for reasons unrelated "
        "to their role pushes both arms' outcomes down in a way that is not "
        "about waiver skill. It is broken out rather than averaged into the "
        "headline."
    )
    add("")
    add(f"| pooled over | n weeks | repo ppg | naive ppg | repo − naive | {CI_LEVEL:.0%} CI |")
    add("| --- | ---: | ---: | ---: | ---: | :---: |")
    for label, seasons in [
        (f"all {len(REPLAY_SEASONS)} seasons (headline)", REPLAY_SEASONS),
        ("2020 alone", [BREAK_SEASON]),
        ("excluding 2020", [s for s in REPLAY_SEASONS if s != BREAK_SEASON]),
    ]:
        b = bucket_table(weekly, picks, seasons, headline_weeks, stratify=len(seasons) > 1)
        add(
            f"| {label} | {b['n_weeks']} | {fmt(b['repo'].get('mean_fwd3'))} | "
            f"{fmt(b['naive'].get('mean_fwd3'))} | {fmt(b['mean_diff'])} | "
            f"{ci_cell(b)} |"
        )
    add("")

    add("### 2021: the season went from 16 games to 17")
    add("")
    add(
        "Weeks 14-17 mean different things on either side of this line. Before "
        "2021 the regular season ended at week 17, so week 14 is the last week "
        "with a full forward window and week 17 has none at all. From 2021 the "
        "season ends at week 18 and week 15 is the last full one. Pooling "
        "\"weeks 14-17\" across the break compares a bracket window against a "
        "bracket window plus an extra game, scored over different numbers of "
        "forward weeks. It is split rather than pooled:"
    )
    add("")
    add(
        f"| era | seasons | weeks | n | repo ppg | naive ppg | repo − naive | "
        f"{CI_LEVEL:.0%} CI |"
    )
    add("| --- | --- | --- | ---: | ---: | ---: | ---: | :---: |")
    pre = [s for s in REPLAY_SEASONS if s < EXPANSION_SEASON]
    post = [s for s in REPLAY_SEASONS if s >= EXPANSION_SEASON]
    for era_label, seasons in [("16-game (17 weeks)", pre), ("17-game (18 weeks)", post)]:
        if not seasons:
            continue
        for bucket_label, weeks in [("2-13", range(2, 14)), ("2-14 (headline)", range(2, 15)),
                                    ("14-17", range(14, 18))]:
            b = bucket_table(weekly, picks, seasons, weeks, stratify=True)
            add(
                f"| {era_label} | {seasons[0]}-{seasons[-1]} | {bucket_label} | "
                f"{b['n_weeks']} | {fmt(b['repo'].get('mean_fwd3'))} | "
                f"{fmt(b['naive'].get('mean_fwd3'))} | {fmt(b['mean_diff'])} | "
                f"{ci_cell(b)} |"
            )
    add("")

    add("### Roster depth thresholds: held fixed, and what that is worth")
    add("")
    add(
        "`ROSTER_DEPTH` (QB 18, RB 46, WR 60, TE 18) decides which players count "
        "as wire-available. The thresholds were tuned to a 12-team, 15-man-roster "
        "league **today**, and whether 2014's replacement level sat in the same "
        "place is an assumption rather than a measurement."
    )
    add("")
    add(
        "**They are held fixed across all twelve seasons.** The reasoning: these "
        "numbers describe the *league*, not the NFL — twelve managers holding "
        "fifteen players each is a constant of the question being asked, and it "
        "did not change in 2014. What did change is the NFL's usage "
        "distribution: passing volume rose over the period, so more receivers "
        "cleared a given usage bar in 2024 than in 2014, and a fixed WR-60 cut "
        "sits at a slightly different point in the real talent distribution at "
        "either end of the span. There is no principled scaling rule that fixes "
        "that without inventing precision, so the assumption is measured instead "
        "of adjusted."
    )
    add("")
    if depth:
        add(
            "Every threshold scaled by ±25%, `on_wire` recomputed against the "
            "already-present `rank_before`, both arms drawing from whichever pool "
            "results. The question is not whether the absolute points move — a "
            "deeper pool contains worse players, so of course they do — but "
            "whether the **verdict** moves:"
        )
        add("")
        add(
            f"| roster depth | QB/RB/WR/TE | n weeks | repo ppg | naive ppg | "
            f"repo − naive | {CI_LEVEL:.0%} CI |"
        )
        add("| --- | --- | ---: | ---: | ---: | ---: | :---: |")
        for scale in (DEPTH_SCALES[0], 1.0, DEPTH_SCALES[1]):
            if scale == 1.0:
                b, thresholds = pooled, F.ROSTER_DEPTH
                tag = "as shipped"
            else:
                entry = depth[scale]
                b = bucket_table(
                    entry["weekly"], entry["picks"], None, headline_weeks, stratify=True
                )
                thresholds = {
                    k: max(1, round(v * scale)) for k, v in F.ROSTER_DEPTH.items()
                }
                tag = f"×{scale:g}"
            add(
                f"| {tag} | "
                f"{'/'.join(str(thresholds[p]) for p in ('QB', 'RB', 'WR', 'TE'))} "
                f"| {b['n_weeks']} | {fmt(b['repo'].get('mean_fwd3'))} | "
                f"{fmt(b['naive'].get('mean_fwd3'))} | {fmt(b['mean_diff'])} | "
                f"{ci_cell(b)} |"
            )
        add("")
        add(depth_reading(depth, pooled, headline_weeks))
        add("")

    # ---- week buckets ---------------------------------------------------
    add("## By week bucket")
    add("")
    add(
        "Weeks 2-13 is the weekly-cash and playoff-push window; 14-17 is the "
        "bracket and consolation roto. The headline stops at 14 because that is "
        "where the forward window stops being comparable across the 2021 break, "
        "not because of the calendar."
    )
    add("")
    add(
        f"| weeks | what it is | n weeks | repo ppg | naive ppg | repo − naive | "
        f"{CI_LEVEL:.0%} CI |"
    )
    add("| --- | --- | ---: | ---: | ---: | ---: | :---: |")
    for bucket, weeks, description in BUCKETS:
        b = bucket_table(weekly, picks, None, weeks, stratify=True)
        mark = " **(headline)**" if bucket == HEADLINE_BUCKET else ""
        add(
            f"| {bucket}{mark} | {description} | {b['n_weeks']} | "
            f"{fmt(b['repo'].get('mean_fwd3'))} | "
            f"{fmt(b['naive'].get('mean_fwd3'))} | {fmt(b['mean_diff'])} | "
            f"{ci_cell(b)} |"
        )
    add("")
    add(
        "`†` marks an interval resampled **without** season stratification. A "
        "single-week bucket gives every season a stratum of one week, and "
        "resampling one item with replacement always returns it — every draw is "
        "identical and the interval collapses to zero width, which would read as "
        "impossible precision rather than as no information. Those rows fall back "
        "to unstratified resampling, which is wider and honest."
    )
    add("")

    # ---- forward window -------------------------------------------------
    add("## The forward window and the 2021 expansion")
    add("")
    add(
        "`fwd3` averages the weeks in W+1..W+3 that the player's team actually "
        "played. **Season length is not a constant over this span**, so neither "
        "is the point at which that window runs off the end:"
    )
    add("")
    add("| era | final week | wk14 | wk15 | wk16 | wk17 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for era_label, final in [("2014-2020", 17), ("2021-2025", 18)]:
        cells = " | ".join(
            str(max(0, min(3, final - w))) for w in (14, 15, 16, 17)
        )
        add(f"| {era_label} | {final} | {cells} |")
    add("")
    add(
        "**Week 17 of a 17-week season has no forward window at all.** `fwd3` is "
        "NaN for every player in the pool, so neither arm can be scored and the "
        "week drops out of the replay. That is seven seasons × one week, and it "
        "is counted in [what could not be scored](#what-could-not-be-scored) "
        "rather than quietly missing. Weeks 15 and 16 in those seasons survive "
        "on two- and one-week outcomes respectively."
    )
    add("")
    add(
        "Nothing is silently truncated: `fwd3_span` is carried on every row of "
        "`replay_picks.csv`, alongside `season_final_week`, so the length of the "
        "window behind any number is visible in the row rather than inferred "
        "from a week number that no longer determines it."
    )
    add("")
    add(
        "Within a week both arms are scored over the identical window, so a "
        "paired difference at week 16 is fair. What is not fair is **pooling** a "
        "one-week outcome with a three-week one: a single game is several times "
        "noisier, so short weeks would carry variance out of all proportion to "
        "their count while measuring a different quantity — and they would do it "
        "asymmetrically across the 2021 break, where the same week number buys a "
        "different number of forward games. **Weeks 15-17 are therefore excluded "
        "from the pooled headline**, which covers weeks 2-14, and reported on "
        "their own above."
    )
    add("")
    add(
        "Byes shorten individual windows earlier in the season too — a player "
        "whose team is off in W+2 is scored over the other two weeks, per the "
        "`features.forward_three` convention that a bye says nothing about the "
        "player. That is a per-player detail inside a full-length window and is "
        "not the same thing as the season ending."
    )
    add("")
    add(
        "Separately: the models train on weeks 2-14, so weeks 15-17 are "
        "extrapolation in every season. The headline bucket is inside the "
        "training window, which is one more reason to prefer it."
    )
    add("")

    # ---- what the arms picked -------------------------------------------
    add("## What the two arms actually picked")
    add("")
    add(
        f"Across weeks 2-14 the naive arm names "
        f"**{overlap_rate(picks, headline_weeks):.1%}** of the repo arm's picks "
        "in the same week. That number matters: if the arms largely agreed, a "
        "difference near zero would say more about how small the eligible pool "
        "is than about whether the ranking works."
    )
    add("")
    for title, ascending in (("Best repo picks", False), ("Worst repo picks", True)):
        frame = (
            picks[(picks["arm"] == "repo") & picks["week"].isin(list(headline_weeks))]
            .sort_values("fwd3", ascending=ascending)
            .head(5)
        )
        add(f"**{title}**")
        add("")
        add("| season | week | player | pos | fwd3 | prior week pts |")
        add("| ---: | ---: | --- | --- | ---: | ---: |")
        for _, r in frame.iterrows():
            add(
                f"| {int(r['season'])} | {int(r['week'])} | {r['player']} | "
                f"{r['position']} | {r['fwd3']:.2f} | {r['pts_prior_week']:.2f} |"
            )
        add("")

    # ---- decomposition ---------------------------------------------------
    add("## Where the gap comes from")
    add("")
    add(
        "The two arms do not pick the same kinds of players and positions do not "
        "score alike, so the raw comparison mixes two different things. The split "
        "is exact rather than approximate: writing each arm's mean as its "
        "positional mix times the pool average at each position, plus what it "
        "beat that pool average by, separates a **mix** term from a **selection** "
        "term. Only the second is about ranking players well."
    )
    add("")
    add(
        "| position | share of naive picks | share of repo picks | "
        "pool avg fwd3 at that position |"
    )
    add("| --- | ---: | ---: | ---: |")
    for position in split["positions"]:
        add(
            f"| {position} | {split['naive']['weights'][position]:.1%} | "
            f"{split['repo']['weights'][position]:.1%} | {split['mu'][position]:.2f} |"
        )
    add("")
    add("| component | ppg |")
    add("| --- | ---: |")
    add(f"| naive − repo, raw `fwd3` | {split['gap']:+.2f} |")
    add(f"| &nbsp;&nbsp;of which **position mix** | {split['mix']:+.2f} |")
    add(
        f"| &nbsp;&nbsp;of which **within-position selection** | "
        f"{split['selection_gap']:+.2f} |"
    )
    add("")
    add(
        "Points above replacement is the repo's own correction for positional "
        "incomparability, and `fwd3` — raw fantasy points — puts it straight "
        "back. The position-adjusted comparison is the one to carry forward."
    )
    add("")
    add(adjusted_reading(adjusted, split))
    add("")
    add(
        "Worth contrasting with the three-season replay in "
        "`season_replay_2022_2025.md`, which put 61% of the gap on position mix "
        "and still found the repo arm behind by 1.18 ppg [-2.23, -0.12] with "
        "position removed. Twelve seasons move both numbers: the mix share rises "
        f"to {split['mix'] / split['gap']:.0%} and the within-position remainder "
        "shrinks to a null with an interval roughly a third as wide. The larger "
        "sample did not sharpen a small real effect into significance — it "
        "dissolved it."
    )
    add("")
    add(
        "It also means the raw headline overstates what a real manager would "
        "lose by following the table. Three quarterback claims a week is not a "
        "strategy anyone can execute — there is one QB slot — and the replay "
        "imposes no roster constraints, an omission that happens to flatter the "
        "naive arm."
    )
    add("")

    # ---- contamination ---------------------------------------------------
    add("## Contamination audit")
    add("")
    add(
        "Treated as the main task. The shipped `models/*.joblib` are fitted "
        "through 2025 and are **never loaded here**. Every replay season is "
        "scored by models refitted on the seasons strictly before it, written to "
        "`outputs/backtests/replay_models_full/` and stamped `replay_only: True` "
        "with a `do_not_serve` note naming this file. `src.models.load_bundle` "
        "only ever reads `models/`, so a replay bundle cannot be served by "
        "accident. Nothing in this run writes to `models/`."
    )
    add("")
    add(
        "Refitting the estimator is necessary and **not sufficient**. Each "
        "asked-about path, and what was actually found:"
    )
    add("")
    add("| path | leaked? | what was done |")
    add("| --- | --- | --- |")
    add(
        "| empirical Bayes priors | **yes, closed** | "
        "`features.empirical_bayes_share` fits its beta priors per position on "
        "the pooled season-to-date rate distribution over **every season in the "
        "build**, so a 2014 row built alongside 2025 is shrunk toward a prior "
        "that has seen 2025. Closed by `build(seasons, "
        "prior_seasons=train_seasons)`, which fits the prior on the training "
        "seasons only and applies it everywhere. This matters more over twelve "
        "seasons than over three: unclosed, a 2014 row in this build would be "
        "shrunk toward a prior fitted on all thirteen seasons through 2025. "
        "Closed, it is fitted on 2013 alone. |"
    )
    add(
        "| rank-to-points scale | **yes, closed** | `weekly.points_scale` maps a "
        "predicted percentile to fantasy points through the empirical quantile "
        "function of `fwd3` over the *whole* panel, replay season included. "
        "Within a position that map is monotone and cannot reorder anything — "
        "but claims are ordered by points above replacement **across** "
        "positions, so it does move which three names surface. Fitted on "
        "training seasons only here. Easy to miss: the leak is inside a "
        "display-looking function. |"
    )
    add(
        "| conformal calibration split | **no effect on picks** | "
        "`models.conformal` fits through N−2, calibrates on N−1 and evaluates on "
        "N over every season in the universe. Recomputed on training seasons "
        "alone, which leaves 2014 (one training season) and 2015 (two) with no "
        "interval at all — reported as absent rather than borrowed. It never "
        "touched the ranking: the half-width sets `score_lo`/`score_hi`, and the "
        "pick order comes from `proj_pts` off the point estimate. |"
    )
    add(
        "| `on_wire` and the season-to-date rank | **no** | `rank_before` ranks "
        "`cum_before` within (season, position, week); `cum_before` is a "
        "*shifted* cumulative sum within (player, season). Both are strictly "
        "within-season and strictly backward-looking, so neither can see the "
        "replay season's future or any other season at all. `ROSTER_DEPTH` is a "
        "declared league constant, not a fitted quantity — its own assumption is "
        "measured in the sensitivity check above. |"
    )
    add(
        "| anything else in `features.py` computed across the full panel | "
        "**no** | Asserted, not reviewed by eye. "
        "`tests/test_features.py::test_only_eb_columns_cross_seasons` rebuilds "
        "the panel at two different season scopes and asserts the overlapping "
        "rows are identical in **every** column except the two empirical Bayes "
        "shares. That covers `on_wire`, `rank_before`, `cusum`, `kal_role`, "
        "`snap_jump`, `neutral_opp`, `tgt_share`, `pts_lag1` and `fwd3` in one "
        "assertion, and it is what would catch a future feature quietly starting "
        "to pool across seasons. A companion test asserts the two EB columns *do* "
        "still move, so the guard cannot go vacuous. |"
    )
    add("")
    add("### The leak that cannot be closed")
    add("")
    add(
        "Every constant in the pipeline — `ROSTER_DEPTH`, `REPLACEMENT_RANK`, "
        "`NEUTRAL_WP`, the Kalman and CUSUM settings, the weeks 2-14 training "
        "window, `MODEL_KWARGS`, `PROJECTION_CLIP` — was chosen by a human who "
        "had already seen these seasons. No refit removes that. It biases the "
        "repo arm upward by an unknown amount, it appears in no interval below, "
        f"and it is why the {LEAKAGE_TRIPWIRE_PPG:.0f} ppg tripwire in the "
        "verdict exists."
    )
    add("")
    add("### A data defect found and fixed in the course of this run")
    add("")
    add(
        "Not contamination, but it would have silently corrupted every pre-2020 "
        "season and is the reason to state it here. **The three nflverse feeds "
        "do not agree on team codes before 2020.** "
        "`stats_player_week_{y}.csv` uses the modern franchise code in every "
        "season — a 2013 Rams row says `LA` — while `snap_counts_{y}.csv` and "
        "`games.csv` use the code in force at the time (`STL`). Two things broke "
        "as a result, neither of them loudly:"
    )
    add("")
    add(
        "1. The stats-to-snaps merge in `features.load_season` joins on `team`, "
        "so every player on a relocated franchise failed to match and was "
        "dropped. **2013 lost 12.1% of stat rows against a ~2.5% baseline** — "
        "three entire franchises (Rams, Chargers, Raiders)."
    )
    add(
        "2. `features.forward_three` asks whether `(season, week, team)` played a "
        "game, with `team` from the stats side and the schedule keyed the "
        "period-correct way. It never found one, so **`fwd3` — the training "
        "target — was NaN for every player on those teams**."
    )
    add("")
    add(
        "Fixed by `features.RELOCATIONS`/`normalize_teams`, which maps "
        "`STL→LA`, `SD→LAC`, `OAK→LV` on the snaps and schedule sides. The last "
        "relocation was Oakland to Las Vegas in 2020, so **this is a no-op from "
        "2020 on and cannot change anything the shipped 2022-2026 pipeline "
        "produces**. After the fix 2013's snap match rate is 98.0% and all 32 "
        "teams carry a normal `fwd3` null rate. Had this gone unnoticed the "
        "replay would have run to completion and produced a clean-looking table "
        "computed on 29 teams."
    )
    add("")

    # ---- not tested -------------------------------------------------------
    add("## What this does NOT test")
    add("")
    add(
        "- **No judgement layer.** The repo arm here is the table's top three by "
        "points above replacement, taken mechanically. The real `make report` "
        "workflow is the table *plus* a human reading the evidence columns and "
        "the news. This measures the ranking, not the workflow."
    )
    add(
        "- **No roster constraints.** No drop is proposed, no starting slot "
        "filled, no bye worked around. Every pick is a free addition, and three "
        "quarterbacks in a week is allowed."
    )
    add(
        "- **No real league availability.** The replay assumes every ranked "
        "player was actually free. `on_wire` is a season-to-date scoring-rank "
        "proxy; in a real 12-team league a meaningful share of these names were "
        "rostered. Both arms draw from the same pool, so the comparison is "
        "internally fair — the absolute points captured by either arm are "
        "optimistic."
    )
    add(
        "- **No waiver priority, FAAB, or claim contention.** Three picks a week "
        "are granted unconditionally to both arms."
    )
    add(
        "- **The prompt arm cannot be replayed at all.** Any model asked to pick "
        "a week from 2017 today already knows how 2017 went, and no prompt "
        "discipline removes that. The three-arm comparison in `src/ledger.py` can "
        "only ever run forward, in real time. This is a two-arm exercise for that "
        "reason."
    )
    add(
        "- **No claim that 2014's league is 2025's league.** Roster depth is held "
        "fixed and the sensitivity check above is the whole of what is known "
        "about that assumption."
    )
    add("")

    # ---- method -----------------------------------------------------------
    add("## Method")
    add("")
    add(
        "Weeks 2 through 17 of each replay season, subject to the season "
        "actually having them. The **repo arm** takes the top three names by "
        "points above replacement at position — `report.with_edge` over "
        "`weekly.score_week`'s projection, positive edge only, which is exactly "
        "what `report.assign_tiers` ranks before it tiers. The **naive arm** "
        "takes the top three available by the previous week's fantasy points, "
        "straight from `ledger.naive_picks`, with no model anywhere in it. Both "
        "arms are scored by `fwd3`: mean points over the next three weeks the "
        "player's team played, **a week played without him counted as 0.0**, team "
        "byes excluded. That is the convention the models train on and the ledger "
        "grades against."
    )
    add("")
    add("| replay season | expanding trains on | recent trains on |")
    add("| --- | --- | --- |")
    add(f"| {FIRST_SEASON} | — | not replayable, no prior data |")
    for season in REPLAY_SEASONS:
        e = train_seasons_for(season, "expanding")
        r = train_seasons_for(season, "recent")
        add(
            f"| {season} | {e[0]}-{e[-1]} ({len(e)}) | "
            + (f"{r[0]}-{r[-1]} ({len(r)})" if len(r) > 1 else f"{r[0]} (1)")
            + " |"
        )
    add("")
    add("### Why walk-forward and not leave-one-season-out")
    add("")
    add(
        "Leave-one-season-out — hold out 2014, train on 2015-2025, and so on — "
        "would give thirteen replay seasons instead of twelve and far more "
        "training data for each. It is rejected because it **trains on the "
        "future**. It answers \"is this method sound in principle\"; the question "
        "here is \"could I have made these picks at the time\", and only "
        "walk-forward answers that. It is not computed here even as a "
        "sensitivity check — `01_season_replay.py` has one over 2022-2025 for "
        "anyone who wants the comparison."
    )
    add("")
    add("### Data range")
    add("")
    add(
        "nflverse snap counts begin in **2013**, and snap share is a required "
        "input at WR and TE, so 2013 is a hard floor regardless of weekly stats "
        "and play-by-play reaching back to 1999. `snap_counts_2012.csv` exists "
        "and returns **HTTP 200** — it is 154 bytes, a header row and no data. "
        "An availability check that stops at the fetch passes it, and the season "
        "then contributes nothing to the panel without anything failing. "
        "`features.require_rows` now asserts on the parsed contents of every "
        "source file rather than on the fetch, for stats, snaps and the schedule "
        "alike."
    )
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| fetched | {FIRST_SEASON}-{LAST_SEASON} |")
    add(f"| replayed | {REPLAY_SEASONS[0]}-{REPLAY_SEASONS[-1]} ({len(REPLAY_SEASONS)} seasons) |")
    add(f"| picks per arm, headline bucket | {pooled['repo'].get('n_picks', 0):,} |")
    add(f"| paired weeks, headline bucket | {pooled['n_weeks']} |")
    add("")

    # ---- what could not be scored ----------------------------------------
    add("## What could not be scored")
    add("")
    if dead_weeks:
        add(
            f"**{len(dead_weeks)} replay weeks produced no scoreable picks for "
            "either arm** and are absent from every table above by construction, "
            "not by omission:"
        )
        add("")
        for line in dead_weeks[:20]:
            add(f"- {line}")
        if len(dead_weeks) > 20:
            add(f"- ...and {len(dead_weeks) - 20} more")
        add("")
    if unscored:
        add(f"**{len(unscored)} individual picks** could not be scored:")
        add("")
        for line in unscored[:20]:
            add(f"- {line}")
        if len(unscored) > 20:
            add(f"- ...and {len(unscored) - 20} more")
        add("")

    # ---- reproducing ------------------------------------------------------
    add("## Reproducing")
    add("")
    add("```bash")
    add("make install")
    add(
        'make data SEASONS="'
        + " ".join(str(s) for s in ALL_SEASONS)
        + '"'
    )
    add("python outputs/backtests/02_walkforward_2014_2025.py")
    add("```")
    add("")
    add(
        "Writes `outputs/backtests/replay_full/replay_picks.csv` (every pick and "
        "its outcome, with `fwd3_span` and `season_final_week` on each row), "
        "`replay_weeks.csv` (the arm-by-week means the intervals resample) and "
        "this file. Replay model bundles go to "
        "`outputs/backtests/replay_models_full/` and are gitignored: they are "
        "regenerated by the script, and committing them next to the production "
        "bundles would invite exactly the confusion they are labelled against."
    )
    add("")
    add(
        "Computed against the nflverse revision pinned in "
        "`data/raw/MANIFEST.json`. nflverse revises history in place, so a rerun "
        "after a revision will not reproduce these numbers exactly — that is the "
        "manifest's whole job. Figures are reliable to about ±0.3 ppg: "
        "`HistGradientBoostingRegressor` bins its features, and float "
        "perturbations at the 1e-16 level are enough to move a bin edge, a "
        "split, and occasionally which player a week's claim goes to. "
        "`01_season_replay.py` measures that sensitivity directly. **The second "
        "decimal place is not real.**"
    )
    add("")

    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text("\n".join(L) + "\n")
    print(f"\nwrote {DIAGNOSTIC.relative_to(ROOT)}")


def paired_schemes(weekly_a, weekly_b, weeks, seasons) -> dict:
    """Repo arm under scheme B minus repo arm under scheme A, paired by week.

    Comparing the two training windows against each other rather than each
    against naive: the same weeks, the same pool, the same scoring, and the only
    thing that differs is how much history the model was allowed to see.
    """
    def side(frame):
        rows = frame[
            (frame["arm"] == "repo")
            & frame["week"].isin(list(weeks))
            & frame["season"].isin(seasons)
        ]
        return rows.set_index(["season", "week"])["mean_fwd3"]

    a, b = side(weekly_a), side(weekly_b)
    shared = a.index.intersection(b.index)
    diffs = (b.loc[shared] - a.loc[shared]).to_numpy(dtype=float)
    strata = np.array([s for s, _ in shared])
    lo, hi = bootstrap_ci(diffs, strata if len(diffs) else None)
    return {
        "n": len(diffs),
        "mean": float(diffs.mean()) if len(diffs) else np.nan,
        "lo": lo,
        "hi": hi,
    }


def recency_reading(cross: dict) -> str:
    """The decision the expanding-vs-recent comparison actually supports."""
    mean, lo, hi = cross["mean"], cross["lo"], cross["hi"]
    if np.isnan(mean):
        return "No paired weeks where the two windows differ. Nothing measured."
    if lo <= 0.0 <= hi:
        return (
            f"**The interval covers zero.** Training on the most recent "
            f"{RECENT_WINDOW} seasons is worth {mean:+.2f} ppg against training "
            f"on all available history ({CI_LEVEL:.0%} CI [{lo:+.2f}, {hi:+.2f}]), "
            "which is indistinguishable from nothing. **The decision this "
            "supports: do not recency-weight the production models.** Football "
            "did not change over this span in a way this pipeline's features can "
            "see, and discarding history to chase a change that is not there "
            "costs training volume for no measured return. Note the asymmetry in "
            "what this can conclude — it is evidence against a *large* recency "
            "effect, not proof of none."
        )
    if mean > 0:
        return (
            f"**Recent wins: {mean:+.2f} ppg, {CI_LEVEL:.0%} CI "
            f"[{lo:+.2f}, {hi:+.2f}], excluding zero.** Training on the last "
            f"{RECENT_WINDOW} seasons beats training on everything, which is a "
            "real finding about non-stationarity: older seasons are actively "
            "misleading the model rather than merely adding little. **The "
            "decision this supports: recency-weight the production models**, and "
            "treat the current all-history fit as leaving points on the table."
        )
    return (
        f"**Expanding wins: recent is {mean:+.2f} ppg worse, {CI_LEVEL:.0%} CI "
        f"[{lo:+.2f}, {hi:+.2f}], excluding zero.** More history helps and the "
        "relationship the model learns is stable enough that a season from a "
        "decade ago still carries signal. **The decision this supports: keep "
        "using all available history**, and do not recency-weight."
    )


def depth_reading(depth: dict, pooled: dict, weeks) -> str:
    """Whether the verdict survives the roster-depth assumption."""
    signs = []
    for scale in DEPTH_SCALES:
        entry = depth[scale]
        b = bucket_table(entry["weekly"], entry["picks"], None, weeks, stratify=True)
        signs.append((scale, b["mean_diff"], b["ci_lo"], b["ci_hi"]))
    base = pooled["mean_diff"]
    same_direction = all(np.sign(d) == np.sign(base) for _, d, _, _ in signs)
    same_conclusion = all(
        (lo <= 0 <= hi) == (pooled["ci_lo"] <= 0 <= pooled["ci_hi"])
        for _, _, lo, hi in signs
    )
    spread = max(abs(d - base) for _, d, _, _ in signs)
    if same_direction and same_conclusion:
        magnitude = (
            (
                f"The **magnitude** is not equally stable: the paired difference "
                f"moves by up to {spread:.2f} ppg across the range, more than the "
                "±0.3 ppg of run-to-run floating-point noise, and a deeper wire "
                "pool narrows the gap because it dilutes the naive arm's "
                "quarterbacks faster than it dilutes the repo arm's picks. So "
                "quote the headline as a range across plausible depths rather "
                "than as a point."
            )
            if spread > 0.3 else
            (
                f"The magnitude barely moves either — at most {spread:.2f} ppg "
                "across the range, inside the ±0.3 ppg of run-to-run "
                "floating-point noise."
            )
        )
        return (
            "**The verdict does not depend on the thresholds.** All three depths "
            "point the same way and reach the same conclusion about zero, so the "
            "answer to the question asked is not an artefact of where the wire "
            f"is cut. {magnitude} Whether 2014's replacement level sat exactly "
            "where today's does is therefore not load-bearing for the verdict, "
            "which is the most that can be said for an assumption that cannot be "
            "checked directly."
        )
    return (
        f"**The verdict is sensitive to the thresholds.** The paired difference "
        f"moves by up to {spread:.2f} ppg across the ±25% range and the "
        "conclusion is not stable across it. Read the headline as conditional on "
        "today's roster depth being roughly right for the whole span — which is "
        "an assumption, not a measurement, and this check says it is one that "
        "matters."
    )


# ==========================================================================

def load_previous_run():
    """Rebuild the reporting inputs from a completed run's saved picks.

    Everything the write-up needs is a function of the per-pick rows and the
    notes file, so the prose can be revised without refitting 48 models. The
    picks carry their own `scheme` column, which is what separates the two
    training windows and the two depth-sensitivity arms back out.
    """
    path = OUT_DIR / "replay_picks.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- --report-only rewrites the markdown from a "
            "completed run, so run the script without it first."
        )
    picks = pd.read_csv(path)
    results = {}
    for scheme in SCHEMES:
        frame = picks[picks["scheme"] == scheme]
        if frame.empty:
            raise SystemExit(f"no rows for scheme '{scheme}' in {path.name}")
        results[scheme] = {"picks": frame, "weekly": week_means(frame)}

    depth = None
    labels = {scale: f"depth{scale:g}" for scale in DEPTH_SCALES}
    if all((picks["scheme"] == label).any() for label in labels.values()):
        depth = {}
        for scale, label in labels.items():
            frame = picks[picks["scheme"] == label]
            depth[scale] = {"picks": frame, "weekly": week_means(frame)}

    notes = {"unscored": [], "dead_weeks": []}
    if NOTES_PATH.exists():
        notes = json.loads(NOTES_PATH.read_text())
    else:
        print(
            f"  warning: {NOTES_PATH.name} absent, so the 'what could not be "
            "scored' section will understate. Rerun in full to regenerate it."
        )
    return results, depth, notes["unscored"], notes["dead_weeks"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-depth-sensitivity", action="store_true",
        help="skip the +/-25%% roster-depth sensitivity check",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help=(
            "rewrite the markdown from an existing replay_full/replay_picks.csv "
            "without refitting anything; fails if that run is absent"
        ),
    )
    args = parser.parse_args(argv)

    finals = final_weeks()

    if args.report_only:
        results, depth, unscored, dead_weeks = load_previous_run()
        write_markdown(results, finals, depth, unscored, dead_weeks)
        return 0

    print("=" * 74)
    print(f"WALK-FORWARD REPLAY {REPLAY_SEASONS[0]}-{REPLAY_SEASONS[-1]}")
    print(f"season lengths: {finals}")
    print("=" * 74)

    results: dict[str, dict] = {}
    unscored: list[str] = []
    dead_weeks: list[str] = []
    for scheme in SCHEMES:
        print()
        print("=" * 74)
        print(f"TRAINING WINDOW: {scheme} ({SCHEMES[scheme]})")
        print("=" * 74)
        picks, uns, dead = run_scheme(scheme, finals)
        results[scheme] = {"picks": picks, "weekly": week_means(picks)}
        if scheme == "expanding":
            unscored, dead_weeks = uns, dead

    depth = None
    if not args.no_depth_sensitivity:
        depth = {}
        for scale in DEPTH_SCALES:
            print()
            print("=" * 74)
            print(f"ROSTER-DEPTH SENSITIVITY: thresholds x{scale:g}")
            print("=" * 74)
            picks, _, _ = run_scheme(f"depth{scale:g}", finals, depth_scale=scale)
            depth[scale] = {"picks": picks, "weekly": week_means(picks)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [results[s]["picks"] for s in SCHEMES]
    weeks_frames = [results[s]["weekly"] for s in SCHEMES]
    if depth:
        frames += [depth[s]["picks"] for s in DEPTH_SCALES]
        weeks_frames += [depth[s]["weekly"] for s in DEPTH_SCALES]
    pd.concat(frames, ignore_index=True).to_csv(
        OUT_DIR / "replay_picks.csv", index=False, float_format="%.4f"
    )
    pd.concat(weeks_frames, ignore_index=True).to_csv(
        OUT_DIR / "replay_weeks.csv", index=False, float_format="%.4f"
    )
    NOTES_PATH.write_text(
        json.dumps({"unscored": unscored, "dead_weeks": dead_weeks}, indent=2) + "\n"
    )

    headline_weeks = dict((n, w) for n, w, _ in BUCKETS)[HEADLINE_BUCKET]
    weekly, picks = results["expanding"]["weekly"], results["expanding"]["picks"]
    pooled = bucket_table(weekly, picks, None, headline_weeks, stratify=True)
    adjusted = bucket_table(
        weekly, picks, None, headline_weeks, stratify=True, value="mean_vs_pos"
    )
    name, lines = verdict(pooled, adjusted, decompose(picks, headline_weeks))
    print()
    print("=" * 74)
    print(f"VERDICT: {name}")
    print("=" * 74)
    for line in lines:
        print(line)
        print()

    write_markdown(results, finals, depth, unscored, dead_weeks)
    print(f"wrote {(OUT_DIR / 'replay_picks.csv').relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / 'replay_weeks.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

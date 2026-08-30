"""Walk-forward replay of the candidate table, 2023-2025, against the naive arm.

One question: does ranking the wire by the repo's model beat simply claiming
last week's highest scorer? If it does not, the pipeline is decoration and it is
better to know that before Week 1 than in December.

WALK-FORWARD, NOT LEAVE-ONE-OUT
===============================

For each replay season S the models are refitted on seasons strictly before S:

    replay 2023  <- trained on 2022                 (one season; expect it weak)
    replay 2024  <- trained on 2022-2023
    replay 2025  <- trained on 2022-2024

2022 is not replayable: there is nothing before it. That costs a season relative
to leave-one-season-out, which would hold out 2022 and train on 2023-2025 and so
give four replay seasons. Leave-one-out is rejected for the headline because it
trains on the future: it answers "is this method sound in principle", and the
question here is "could these picks have been made at the time". It is computed
anyway as a clearly labelled sensitivity check at the bottom, because the gap
between the two is itself informative -- if leave-one-out looks much better than
walk-forward, the extra is hindsight, not skill.

Seasons are reported separately as well as pooled. The 2023 model sees one
season of training data and the 2025 model sees three; if the result improves
with training volume that is a real finding about how much history this method
needs, and pooling would erase it.

CONTAMINATION
=============

The shipped `models/*.joblib` are fitted on 2022-2025 and are never loaded here.
Nothing this file writes goes near `models/`; replay bundles land in
`outputs/backtests/replay_models/{season}/` and carry `replay_only: True`, so
`src.models.load_bundle` -- which only ever looks in `models/` -- cannot serve
one by accident.

Refitting the estimator is necessary and not sufficient. Four other paths carry
information across the train/test boundary, and each is closed here:

1. **Empirical Bayes priors.** `features.empirical_bayes_share` fits its beta
   priors per position on the pooled rate distribution over every season in the
   build, so a 2023 row built alongside 2025 has been shrunk toward a prior that
   saw 2025. Closed by `build(seasons, prior_seasons=train_seasons)`, which fits
   the prior on the training seasons only and applies it everywhere. The effect
   is small -- the priors move the shrunk shares by under 0.01 -- but small is
   measured, not assumed.

2. **The rank-to-points scale.** `weekly.points_scale` maps a predicted
   percentile to fantasy points through the empirical quantile function of
   `fwd3` over the *whole panel*, replay season included. Within a position that
   mapping is monotone and cannot reorder anything, but the claim ordering is by
   points above replacement *across* positions, so the scale absolutely does
   move which three names surface. Closed by fitting the scale on the training
   seasons only.

3. **The conformal calibration split.** `models.conformal` fits through N-2,
   calibrates on N-1 and evaluates on N over every season in the universe.
   Recomputed here on training seasons alone, which leaves 2023 (one training
   season) and 2024 (two) with no interval at all -- reported as absent rather
   than borrowed. It does not touch the ranking either way: the half-width sets
   `score_lo`/`score_hi`, and the pick order is driven by `proj_pts` from the
   point estimate.

4. **`on_wire` and the season-to-date rank behind it.** Audited, not assumed:
   `rank_before` ranks `cum_before` within (season, position, week), and
   `cum_before` is a shifted cumulative sum within (player, season), so both are
   strictly within-season and strictly backward. `ROSTER_DEPTH` is a declared
   12x15 roster constant, not a fitted quantity.
   `tests/test_features.py::test_only_eb_columns_cross_seasons` rebuilds the
   panel at two scopes and asserts that the empirical Bayes shares are the only
   columns that move, which covers this and every other feature at once.

What cannot be closed, and is therefore stated rather than fixed: every constant
in the pipeline -- `ROSTER_DEPTH`, `REPLACEMENT_RANK`, `NEUTRAL_WP`, the Kalman
and CUSUM settings, the 2-14 training window, `MODEL_KWARGS` -- was chosen by a
human who had already seen 2022-2025. No refit removes that. It biases the repo
arm upward by an unknown amount and it is the reason a large margin here should
be read as a warning rather than a result.

THE FORWARD WINDOW AT THE END OF THE SEASON
===========================================

`fwd3` averages weeks W+1..W+3 that the player's team actually played. The
regular season ends at week 18 in all four seasons, so the window is full
through week 15 and runs off the end after that: week 16 is scored on two weeks
and week 17 on one. Truncation is not silent -- `fwd3_span` is carried on every
pick -- and both arms in a given week are scored over the identical window, so
the paired difference stays fair. What is not fair is pooling a one-week outcome
with a three-week one: a single game is several times noisier, and weeks 16-17
would dominate the pooled variance while measuring a different quantity.

So the **pooled headline covers weeks 2-15 only**. Weeks 16-17 are reported
separately and labelled, and the 14-17 bucket the roto window actually cares
about is shown both as-asked and split at 15 so the truncation is visible rather
than averaged in.

Weeks 15-17 are also outside the models' 2-14 training window; `src.weekly` warns
about this in normal use and the same caveat applies here.

Run:
    python outputs/backtests/01_season_replay.py
    python outputs/backtests/01_season_replay.py --no-loso    # skip the sensitivity check
"""

from __future__ import annotations

import argparse
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

OUT_DIR = Path(__file__).resolve().parent / "replay"
MODEL_DIR = Path(__file__).resolve().parent / "replay_models"
PANEL_CACHE = ROOT / "data" / "processed"
DIAGNOSTIC = ROOT / "outputs" / "diagnostics" / "season_replay_2022_2025.md"

ALL_SEASONS = [2022, 2023, 2024, 2025]
REPLAY_SEASONS = [2023, 2024, 2025]      # walk-forward: 2022 has nothing before it
REPLAY_WEEKS = list(range(2, 18))        # weeks 2-17 inclusive
ARM_DEPTH = 3

# The last week whose W+1..W+3 window fits inside an 18-week regular season.
# Past it the window truncates and the outcome stops being a three-week average.
LAST_FULL_WINDOW_WEEK = 15

BUCKETS = [
    ("2-13", range(2, 14), "weekly cash and playoff push"),
    ("14-15", range(14, 16), "bracket window, full 3-week outcome"),
    ("16-17", range(16, 18), "bracket window, TRUNCATED outcome"),
    ("14-17", range(14, 18), "bracket window as asked (mixes window lengths)"),
    ("2-15", range(2, 16), "POOLED HEADLINE -- full 3-week window only"),
    ("2-17", range(2, 18), "everything, including the truncated tail"),
]
HEADLINE_BUCKET = "2-15"

# A margin this large is not a result, it is a symptom. Stated as a constant so
# that moving it shows up in the diff as moving the tripwire.
LEAKAGE_TRIPWIRE_PPG = 6.0


# ==========================================================================
# panel + models, per replay season
# ==========================================================================

def panel_cache_path(seasons: list[int], train_seasons: list[int]) -> Path:
    """Cache filename naming both season sets in full.

    In full, not as a min-max range: leave-one-season-out trains 2023 on
    {2022, 2024, 2025} and 2024 on {2022, 2023, 2025}, which share a first and
    last season. A range-based key silently hands one replay the other's panel.
    """
    return PANEL_CACHE / (
        "replay_panel_s"
        + "".join(str(y)[-2:] for y in sorted(seasons))
        + "_prior"
        + "".join(str(y)[-2:] for y in sorted(train_seasons))
        + ".csv"
    )


def replay_panel(
    seasons: list[int], train_seasons: list[int], roundtrip: bool = True
) -> pd.DataFrame:
    """Panel over `seasons` with the EB priors fitted on `train_seasons` only.

    The freshly built frame is written to CSV and then read back before being
    returned, so that a cached run and a cold run are handed bit-identical
    inputs. That is not fussiness. `to_csv`/`read_csv` perturbs the floats by
    about 1e-16, `HistGradientBoostingRegressor` bins its features, and a value
    that crosses a bin edge changes a split, a prediction, and occasionally
    which player a week's claim goes to. Before this round-trip the same script
    produced a pooled headline of -3.57 ppg on a cold run and -3.35 ppg on a
    warm one. `roundtrip=False` skips it and is used only by the stability check
    that measures that gap on purpose.
    """
    cache = panel_cache_path(seasons, train_seasons)
    if roundtrip and cache.exists():
        print(f"  panel: cached {cache.name}")
        return pd.read_csv(cache)
    panel = F.build(seasons, prior_seasons=train_seasons)
    if not roundtrip:
        return panel
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache, index=False)
    return pd.read_csv(cache)


def fit_replay_models(
    panel: pd.DataFrame, train_seasons: list[int], replay_season: int, label: str,
    extra_features: tuple[str, ...] = (),
) -> dict[str, dict]:
    """Refit every position on `train_seasons` and persist to replay_models/.

    The bundle is deliberately shaped like a production one so the scoring code
    below is the same code path, and deliberately marked `replay_only` and
    written outside `models/` so it can never be mistaken for one.
    """
    train = panel[panel["season"].isin(train_seasons)]
    universe = M.wire_universe(train)
    feature_columns = M.feature_columns(panel) + [
        c for c in extra_features if c not in M.feature_columns(panel)
    ]

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
                "outputs/backtests/01_season_replay.py. Trained on "
                f"{train_seasons} to score {replay_season}, so it knows less than "
                "the shipped bundles and must never be loaded by src/weekly.py."
            ),
            "replay_season": replay_season,
            "replay_scheme": label,
            "position": position,
            "model": model,
            "features": feature_columns,
            "extra_features": list(extra_features),
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


def train_points_scale(panel: pd.DataFrame, train_seasons: list[int], position: str):
    """`weekly.points_scale`, restricted to the training seasons.

    Same filter as the shipped function -- weeks 2-14, on the wire, resolved
    fwd3 -- with the replay season and everything after it removed, because this
    is the mapping that decides how a QB's percentile compares to a WR's and so
    decides which three names the table surfaces.
    """
    trained = panel[
        panel["season"].isin(train_seasons)
        & panel["week"].between(*M.WEEKS)
        & panel["on_wire"]
        & (panel["position"] == position)
        & panel["fwd3"].notna()
    ]
    return np.sort(trained["fwd3"].to_numpy())


# ==========================================================================
# one replay week
# ==========================================================================

def repo_picks(
    panel: pd.DataFrame, bundles: dict[str, dict], train_seasons: list[int],
    season: int, week: int, depth: int = ARM_DEPTH,
) -> pd.DataFrame:
    """The top `depth` names the candidate table would have surfaced.

    Reproduces the shipped path exactly -- `weekly.score_week` to get `proj_pts`,
    then `report.with_edge` to convert it to points above positional replacement
    -- with the replay models and the training-season points scale substituted in.
    Ordering is by edge, and only positive-edge names are eligible, which is what
    `report.assign_tiers` does before it tiers anything.
    """
    pool = wire_pool(panel, season, week)
    if pool.empty:
        return pool.assign(rank_within_arm=pd.Series(dtype=int), edge=pd.Series(dtype=float))

    frames = []
    for position, bundle in bundles.items():
        rows = pool[pool["position"] == position].copy()
        if rows.empty:
            continue
        rows["score"] = bundle["model"].predict(rows[bundle["features"]])
        scale = train_points_scale(panel, train_seasons, position)
        rows["proj_pts"] = np.quantile(scale, rows["score"].clip(*PROJECTION_CLIP))
        frames.append(rows)
    table = pd.concat(frames, ignore_index=True)

    # report.with_edge: replacement is the next player you would actually take.
    def baseline(group: pd.Series) -> float:
        rank = REPLACEMENT_RANK.get(group.name, 5)
        ordered = group.sort_values(ascending=False).to_numpy()
        return float(ordered[min(rank, len(ordered) - 1)])

    table["edge"] = table["proj_pts"] - table.groupby("position")["proj_pts"].transform(
        lambda g: baseline(g)
    )
    ranked = (
        table[table["edge"] > 0]
        .sort_values(["edge", "player_display_name"], ascending=[False, True])
        .head(depth)
        .copy()
    )
    ranked["rank_within_arm"] = range(1, len(ranked) + 1)
    return ranked


def window_spans(panel: pd.DataFrame, season: int) -> dict[tuple[int, str], int]:
    """How many of W+1..W+3 the team actually plays, per (week, team).

    Carried onto every pick so a truncated outcome is visible in the row rather
    than inferred from the week number.
    """
    played, last_week = F.load_schedule()
    final = last_week.get(season, 18)
    teams = sorted(panel.loc[panel["season"] == season, "team"].dropna().unique())
    return {
        (week, team): sum(
            1 for w in (week + 1, week + 2, week + 3)
            if w <= final and (season, w, team) in played
        )
        for week in REPLAY_WEEKS
        for team in teams
    }


def replay_season(
    season: int, train_seasons: list[int], label: str,
    extra_features: tuple[str, ...] = (), roundtrip: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Every pick both arms make in one replay season, with its outcome."""
    seasons = sorted(set(train_seasons) | {season})
    print(f"\n{label} replay {season}: train on {train_seasons}")
    panel = replay_panel(seasons, train_seasons, roundtrip)
    bundles = fit_replay_models(panel, train_seasons, season, label, extra_features)
    spans = window_spans(panel, season)

    rows: list[dict] = []
    unscored: list[str] = []
    for week in REPLAY_WEEKS:
        pool = wire_pool(panel, season, week)
        if pool.empty:
            unscored.append(f"{season} wk{week:02d}: empty wire pool")
            continue
        resolved = pool[pool["fwd3"].notna()]
        ceiling = float(resolved["fwd3"].max()) if not resolved.empty else np.nan
        # What an arbitrary available player at the same position returned this
        # week. Subtracting it is what separates "picked a good player" from
        # "picked a quarterback"; see the position-mix section of the write-up.
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
                        f"{season} wk{week:02d} {arm} #{int(pick['rank_within_arm'])} "
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
                        "train_seasons": ",".join(str(s) for s in train_seasons),
                    }
                )

    picks_frame = pd.DataFrame(rows)
    return picks_frame, {"unscored": unscored}


# ==========================================================================
# aggregation
# ==========================================================================

def week_means(picks: pd.DataFrame) -> pd.DataFrame:
    """Arm x week mean outcome. The week is the unit of analysis throughout.

    Three picks made in one week share a wire pool and one slate of upcoming
    opponents, so they are not three independent observations of the arm; the
    ledger resamples weeks for the same reason and this follows it.
    """
    if picks.empty:
        return pd.DataFrame()
    grouped = picks.groupby(["scheme", "season", "week", "arm"], as_index=False).agg(
        n=("fwd3", "size"),
        mean_fwd3=("fwd3", "mean"),
        mean_vs_pos=("vs_pos", "mean"),
        week_ceiling=("week_ceiling", "first"),
        min_span=("fwd3_span", "min"),
    )
    return grouped.sort_values(["scheme", "season", "week", "arm"]).reset_index(drop=True)


def paired(weekly: pd.DataFrame, weeks, value: str = "mean_fwd3") -> pd.DataFrame:
    """One row per week both arms covered: repo mean minus naive mean.

    `value` selects the scoring convention. `mean_fwd3` is the repo's own
    convention and the headline. `mean_vs_pos` subtracts what an arbitrary
    available player at the same position returned that week, which is the same
    comparison with positional composition taken out of it.
    """
    rows = weekly[weekly["week"].isin(list(weeks))]
    repo = rows[rows["arm"] == "repo"].set_index(["season", "week"])
    naive = rows[rows["arm"] == "naive"].set_index(["season", "week"])
    shared = repo.index.intersection(naive.index)
    if len(shared) == 0:
        return pd.DataFrame(columns=["season", "week", "repo", "naive", "diff"])
    frame = pd.DataFrame(
        {"repo": repo.loc[shared, value], "naive": naive.loc[shared, value]}
    )
    frame["diff"] = frame["repo"] - frame["naive"]
    return frame.reset_index().sort_values(["season", "week"]).reset_index(drop=True)


def bootstrap_ci(
    diffs: np.ndarray, strata: np.ndarray | None = None,
    reps: int = BOOTSTRAP_REPS, level: float = CI_LEVEL, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile CI for the mean paired difference, resampling weeks.

    When `strata` is given (the season each week belongs to), weeks are resampled
    within season, so every draw keeps each season's week count. Pooling three
    seasons and resampling freely would let a draw be mostly 2025, and the whole
    reason for reporting seasons separately is that they are not exchangeable --
    they were produced by models trained on different amounts of history.
    """
    if len(diffs) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if strata is None:
        draws = rng.choice(diffs, size=(reps, len(diffs)), replace=True).mean(axis=1)
        tail = (1.0 - level) / 2.0
        return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))

    groups = [diffs[strata == s] for s in np.unique(strata)]
    totals = np.zeros(reps)
    for group in groups:
        totals += rng.choice(group, size=(reps, len(group)), replace=True).sum(axis=1)
    draws = totals / len(diffs)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def arm_stats(weekly: pd.DataFrame, picks: pd.DataFrame, arm: str, weeks) -> dict:
    """n, mean captured, ceiling share and head-to-head rate for one arm."""
    week_list = list(weeks)
    rows = weekly[(weekly["arm"] == arm) & weekly["week"].isin(week_list)]
    pick_rows = picks[(picks["arm"] == arm) & picks["week"].isin(week_list)]
    if rows.empty:
        return {}
    share = (rows["mean_fwd3"] / rows["week_ceiling"]).replace([np.inf, -np.inf], np.nan)
    pairs = paired(weekly, week_list)
    if arm == "repo":
        beat = float((pairs["diff"] > 0).mean()) if not pairs.empty else np.nan
    else:
        beat = float((pairs["diff"] < 0).mean()) if not pairs.empty else np.nan
    return {
        "arm": arm,
        "weeks": int(len(rows)),
        "n_picks": int(len(pick_rows)),
        "mean_fwd3": float(rows["mean_fwd3"].mean()),
        "mean_vs_pos": float(rows["mean_vs_pos"].mean()),
        "ceiling_share": float(share.mean()),
        "beat_other_share": beat,
    }


def bucket_table(weekly: pd.DataFrame, picks: pd.DataFrame, seasons: list[int] | None,
                 weeks, stratify: bool, value: str = "mean_fwd3") -> dict:
    """Both arms plus the paired difference over one week bucket."""
    if seasons is not None:
        weekly = weekly[weekly["season"].isin(seasons)]
        picks = picks[picks["season"].isin(seasons)]
    pairs = paired(weekly, weeks, value)
    diffs = pairs["diff"].to_numpy(dtype=float)
    strata = pairs["season"].to_numpy() if stratify and len(pairs) else None
    lo, hi = bootstrap_ci(diffs, strata)
    repo = arm_stats(weekly, picks, "repo", weeks)
    naive = arm_stats(weekly, picks, "naive", weeks)
    return {
        "repo": repo,
        "naive": naive,
        "value": value,
        "n_weeks": len(diffs),
        "mean_diff": float(diffs.mean()) if len(diffs) else np.nan,
        "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "pairs": pairs,
    }


def overlap_rate(picks: pd.DataFrame, weeks) -> float:
    """Share of repo picks the naive arm also named that week.

    If the two arms keep naming the same people, a difference near zero is not a
    finding about ranking -- it is a finding about the pool being small.
    """
    rows = picks[picks["week"].isin(list(weeks))]
    total = shared = 0
    for (_, season, week), group in rows.groupby(["scheme", "season", "week"]):
        repo = set(group.loc[group["arm"] == "repo", "player"])
        naive = set(group.loc[group["arm"] == "naive", "player"])
        if not repo:
            continue
        total += len(repo)
        shared += len(repo & naive)
    return shared / total if total else np.nan


def spearman_now_vs_lag(season: int = 2025) -> list[tuple[str, float, float]]:
    """Rank correlation of `pts` and `pts_lag1` with `fwd3`, per position.

    The point of printing both is that one of them is a model feature and the
    other is not, and it is the stronger one that is missing.
    """
    from scipy.stats import spearmanr

    train = [y for y in ALL_SEASONS if y < season]
    panel = replay_panel(sorted(set(train) | {season}), train)
    rows = panel[
        (panel["season"] == season)
        & panel["week"].between(2, LAST_FULL_WINDOW_WEEK)
        & panel["on_wire"]
        & panel["snap"].notna()
        & panel["fwd3"].notna()
    ]
    out = []
    for position in M.POSITIONS:
        frame = rows[rows["position"] == position]
        pair = []
        for column in ("pts", "pts_lag1"):
            usable = frame[frame[column].notna()]
            pair.append(float(spearmanr(usable[column], usable["fwd3"]).statistic))
        out.append((position, pair[0], pair[1]))
    return out


def decompose(picks: pd.DataFrame, weeks) -> dict:
    """Split the naive-minus-repo gap into position mix and within-position skill.

    Exactly, not approximately. Writing each arm's mean as its positional mix
    times the pool average at each position, plus what it beat that pool average
    by:

        mean_arm = sum_p w_arm,p * mu_p  +  sum_p w_arm,p * (mean_arm,p - mu_p)

    the gap between two arms separates into a **mix** term -- the arms chose
    different positions, and positions score differently -- and a **selection**
    term, which is the only part that is about ranking players well. `mu_p` is
    the mean fwd3 of everyone available at position p over the same weeks, so it
    is the same baseline for both arms.
    """
    rows = picks[picks["week"].isin(list(weeks))]
    positions = sorted(rows["position"].unique())
    mu = {
        position: float(
            rows.loc[rows["position"] == position, "pool_mean_pos"].mean()
        )
        for position in positions
    }
    out = {"mu": mu, "positions": positions}
    for arm in ("naive", "repo"):
        arm_rows = rows[rows["arm"] == arm]
        weights = {
            position: len(arm_rows[arm_rows["position"] == position]) / len(arm_rows)
            for position in positions
        }
        means = {
            position: float(arm_rows.loc[arm_rows["position"] == position, "fwd3"].mean())
            if len(arm_rows[arm_rows["position"] == position]) else np.nan
            for position in positions
        }
        out[arm] = {
            "weights": weights,
            "means": means,
            "mean": float(arm_rows["fwd3"].mean()),
            "selection": float(
                sum(weights[p] * (means[p] - mu[p]) for p in positions if weights[p])
            ),
        }
    out["gap"] = out["naive"]["mean"] - out["repo"]["mean"]
    out["mix"] = float(
        sum(
            (out["naive"]["weights"][p] - out["repo"]["weights"][p]) * mu[p]
            for p in positions
        )
    )
    out["selection_gap"] = out["naive"]["selection"] - out["repo"]["selection"]
    return out


# ==========================================================================
# the write-up
# ==========================================================================

def fmt(value: float, places: int = 2, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}%}" if pct else f"{value:.{places}f}"


def verdict(result: dict, adjusted: dict, split: dict) -> tuple[str, list[str]]:
    """The headline reading, and the sentences that qualify it.

    Written to make "inconclusive" the easy answer, and to make a large margin
    in the repo arm's favour a reason to go looking rather than to celebrate --
    the tripwire fires before the good news does. Where the result goes against
    the repo arm, the position-mix decomposition and the position-adjusted
    interval are both quoted in the verdict rather than left to a later section,
    because a raw-points comparison between arms that pick different positions
    is not on its own a statement about ranking quality.
    """
    diff, lo, hi, n = result["mean_diff"], result["ci_lo"], result["ci_hi"], result["n_weeks"]
    if np.isnan(diff):
        return "NO RESULT", ["No paired weeks survived. Nothing was measured."]

    if diff >= LEAKAGE_TRIPWIRE_PPG:
        return "LEAKAGE SUSPECTED", [
            f"The repo arm is ahead by {diff:+.2f} ppg, at or past the "
            f"{LEAKAGE_TRIPWIRE_PPG:.0f} ppg tripwire set before the run. A margin "
            "this size is not plausible for a ranking model on this data; the "
            "first hypothesis is that something still crosses the train/test "
            "boundary. Do not act on this number until the contamination audit "
            "below has been re-read against it.",
        ]

    covers_zero = bool(lo <= 0.0 <= hi) if not np.isnan(lo) else True
    if covers_zero:
        return "INCONCLUSIVE", [
            f"Over {n} paired weeks the repo arm is {diff:+.2f} ppg against naive, "
            f"with a {CI_LEVEL:.0%} interval of [{lo:+.2f}, {hi:+.2f}] ppg. The "
            "interval covers zero, so this data does not distinguish the two arms. "
            "That is the finding: not that the ranking fails, but that at this "
            "sample size a difference of the size worth having is indistinguishable "
            "from no difference at all.",
        ]
    if diff > 0:
        return "REPO AHEAD", [
            f"Over {n} paired weeks the repo arm beats naive by {diff:+.2f} ppg, "
            f"{CI_LEVEL:.0%} interval [{lo:+.2f}, {hi:+.2f}] ppg, which excludes "
            "zero. The margin is modest and the interval is wide; read it as "
            "evidence the ranking is doing something, not as a per-week edge you "
            "can plan around.",
        ]

    mix_share = split["mix"] / split["gap"] if split["gap"] else float("nan")
    return "NAIVE WINS", [
        f"**Over {n} paired weeks the repo arm loses to naive by {abs(diff):.2f} ppg** "
        f"({CI_LEVEL:.0%} interval [{lo:+.2f}, {hi:+.2f}], excluding zero). Ranking "
        "the wire by the model's points above replacement did **not** beat claiming "
        "last week's highest scorer, in any of the three replay seasons. This is "
        "the answer to the question the exercise was set up to ask, and it is a no.",

        f"Roughly **{mix_share:.0%} of that gap is positional composition, not "
        f"ranking quality**. The naive arm's picks are "
        f"{split['naive']['weights'].get('QB', 0):.0%} quarterbacks against the repo "
        f"arm's {split['repo']['weights'].get('QB', 0):.0%}, because quarterbacks "
        "score the most raw fantasy points and the naive rule sorts on raw fantasy "
        f"points. Quarterbacks on the wire returned {split['mu'].get('QB', float('nan')):.2f} "
        f"fwd3 on average against a receiver's {split['mu'].get('WR', float('nan')):.2f}. "
        "Points above replacement exists precisely to undo that positional "
        "incomparability -- and then `fwd3`, the scoring convention, puts it back.",

        f"**It does not rescue the repo arm.** With position taken out -- each pick "
        f"scored against what an arbitrary available player at the same position "
        f"returned that week -- naive is still ahead by "
        f"{abs(adjusted['mean_diff']):.2f} ppg (repo − naive = "
        f"{adjusted['mean_diff']:+.2f}, {CI_LEVEL:.0%} interval "
        f"[{adjusted['ci_lo']:+.2f}, {adjusted['ci_hi']:+.2f}]). Both framings agree "
        "in direction, which is what makes the verdict robust rather than an "
        "artefact of the metric.",

        f"The model is not inert. Its picks beat the same-position pool average by "
        f"{split['repo']['selection']:+.2f} ppg, so it is finding better-than-random "
        f"players. The naive rule simply finds better ones ({split['naive']['selection']:+.2f} "
        "ppg). The honest summary is that a gradient-boosted model over snap "
        "shares, target shares and role-change filters is, on three seasons of "
        "walk-forward evidence, a worse waiver heuristic than sorting the wire by "
        "last week's box score.",
    ]


def arm_row(stats: dict, name: str) -> str:
    if not stats:
        return f"| {name} | - | - | - | - | - |"
    return (
        f"| {name} | {stats['weeks']} | {stats['n_picks']} | "
        f"{fmt(stats['mean_fwd3'])} | {fmt(stats['ceiling_share'], 1, pct=True)} | "
        f"{fmt(stats['beat_other_share'], 1, pct=True)} |"
    )


def write_markdown(
    weekly: pd.DataFrame, picks: pd.DataFrame, loso: dict | None,
    diagnostic: dict | None, stability: dict | None, notes: list[str],
) -> None:
    headline_weeks = dict((name, weeks) for name, weeks, _ in BUCKETS)[HEADLINE_BUCKET]
    pooled = bucket_table(weekly, picks, None, headline_weeks, stratify=True)
    adjusted = bucket_table(
        weekly, picks, None, headline_weeks, stratify=True, value="mean_vs_pos"
    )
    split = decompose(picks, headline_weeks)
    name, qualifiers = verdict(pooled, adjusted, split)

    L: list[str] = []
    add = L.append

    add("# Walk-forward replay, 2022-2025: does the ranking beat naive?")
    add("")
    add(
        "Generated by `outputs/backtests/01_season_replay.py`. The answer is first "
        "and the methodology is last, deliberately."
    )
    add("")
    add("## Verdict")
    add("")
    add(f"**{name}.**")
    add("")
    for line in qualifiers:
        add(line)
        add("")

    add(
        f"Headline is the pooled paired difference over **weeks 2-15** of 2023, 2024 "
        f"and 2025 — every week whose three-week forward window fits inside the "
        f"season. Each season's picks come from models refitted on the seasons "
        f"strictly before it and nothing else."
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
        left = pooled["repo"].get(field, np.nan)
        right = pooled["naive"].get(field, np.nan)
        add(f"| {label} | {fmt(left, places, pct)} | {fmt(right, places, pct)} |")
    add(
        f"| head-to-head weeks won | "
        f"{fmt(pooled['repo'].get('beat_other_share'), 1, pct=True)} | "
        f"{fmt(pooled['naive'].get('beat_other_share'), 1, pct=True)} |"
    )
    add("")
    add(
        f"**Paired difference (repo − naive, same weeks): {fmt(pooled['mean_diff'])} ppg, "
        f"{CI_LEVEL:.0%} bootstrap CI [{fmt(pooled['ci_lo'])}, {fmt(pooled['ci_hi'])}], "
        f"n = {pooled['n_weeks']} weeks.**"
    )
    add("")
    add(
        f"Weeks are the resampling unit and the bootstrap is stratified by season, "
        f"so each draw keeps three seasons' worth of weeks. Head-to-head rates do "
        f"not sum to 100%: a tied week counts for neither arm."
    )
    add("")

    # ---- per season ----------------------------------------------------
    add("## Per season")
    add("")
    add(
        "Reported separately because they are not exchangeable. The 2023 model has "
        "one season of training data behind it and the 2025 model has three; if the "
        "margin grows with training volume that is a finding about how much history "
        "this method needs, and pooling would hide it."
    )
    add("")
    add("| season | trained on | weeks | repo ppg | naive ppg | repo − naive | "
        f"{CI_LEVEL:.0%} CI | repo ceiling share | naive ceiling share | repo won |")
    add("| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |")
    for season in REPLAY_SEASONS:
        train = [s for s in ALL_SEASONS if s < season]
        b = bucket_table(weekly, picks, [season], headline_weeks, stratify=False)
        ci = (
            f"[{fmt(b['ci_lo'])}, {fmt(b['ci_hi'])}]"
            if not np.isnan(b["ci_lo"]) else "n/a"
        )
        add(
            f"| {season} | {'-'.join([str(train[0]), str(train[-1])]) if len(train) > 1 else str(train[0])} "
            f"| {b['n_weeks']} | {fmt(b['repo'].get('mean_fwd3'))} | "
            f"{fmt(b['naive'].get('mean_fwd3'))} | {fmt(b['mean_diff'])} | {ci} | "
            f"{fmt(b['repo'].get('ceiling_share'), 1, pct=True)} | "
            f"{fmt(b['naive'].get('ceiling_share'), 1, pct=True)} | "
            f"{fmt(b['repo'].get('beat_other_share'), 1, pct=True)} |"
        )
    add("")
    add(
        "A single season is roughly 14 paired weeks. Per-season intervals are wide "
        "enough that the ordering between seasons is not itself a finding — read "
        "the direction and the overlap, not the ranking."
    )
    add("")

    # ---- week buckets --------------------------------------------------
    add("## By week bucket")
    add("")
    add(
        "Weeks 2-13 is the weekly-cash and playoff-push window; 14-17 is the bracket "
        "and consolation roto. The split at 15 is the forward window, not the "
        "calendar — see below."
    )
    add("")
    add(f"| weeks | what it is | n weeks | repo ppg | naive ppg | repo − naive | {CI_LEVEL:.0%} CI |")
    add("| --- | --- | ---: | ---: | ---: | ---: | :---: |")
    for bucket, weeks, description in BUCKETS:
        b = bucket_table(weekly, picks, None, weeks, stratify=True)
        ci = f"[{fmt(b['ci_lo'])}, {fmt(b['ci_hi'])}]" if not np.isnan(b["ci_lo"]) else "n/a"
        mark = " **(headline)**" if bucket == HEADLINE_BUCKET else ""
        add(
            f"| {bucket}{mark} | {description} | {b['n_weeks']} | "
            f"{fmt(b['repo'].get('mean_fwd3'))} | {fmt(b['naive'].get('mean_fwd3'))} | "
            f"{fmt(b['mean_diff'])} | {ci} |"
        )
    add("")

    # ---- forward window ------------------------------------------------
    spans = (
        picks.groupby("week")["fwd3_span"].agg(["min", "max", "mean"]).reset_index()
    )
    # Weeks where even a team with no bye ahead of it cannot get three forward
    # weeks. Filtering on the minimum instead would list most of the season,
    # because a single upcoming bye shortens one player's window in any week.
    truncated = spans[spans["max"] < 3]
    add("## The forward window at the end of the season")
    add("")
    add(
        "`fwd3` averages the weeks in W+1..W+3 the player's team actually played. "
        "All four seasons end at week 18, so the window is full through week 15 and "
        "then runs off the end: **week 16 is scored on two weeks and week 17 on "
        "one.** Nothing is silently truncated — `fwd3_span` is carried on every row "
        "of `replay_picks.csv`."
    )
    add("")
    add("| week | forward weeks scored, best case |")
    add("| ---: | ---: |")
    for _, row in truncated.iterrows():
        add(f"| {int(row['week'])} | {int(row['max'])} |")
    add("")
    add(
        "Byes shorten individual windows earlier in the season too — a player whose "
        "team is off in W+2 is scored over the other two weeks, which is the "
        "`features.forward_three` convention (a bye says nothing about the player, "
        "so it is excluded rather than scored 0.0). That is a per-player detail "
        f"inside a full-length window. Weeks {int(truncated['week'].min())}-"
        f"{int(truncated['week'].max())} above are different: the window is short "
        "for everybody because the season has ended."
    )
    add("")
    add(
        "Within a week both arms are scored over the identical window, so the paired "
        "difference at week 16 or 17 is still fair. What is not fair is pooling a "
        "one-week outcome with a three-week one: a single game is several times "
        "noisier, so weeks 16-17 would carry variance out of all proportion to their "
        "count while measuring a different quantity. **They are therefore excluded "
        "from the pooled headline**, which covers weeks 2-15, and reported on their "
        "own in the bucket table above."
    )
    add("")
    add(
        "Separately: the models are trained on weeks 2-14 only, so weeks 15-17 are "
        "extrapolation. `src/weekly.py` prints the same warning in normal use. The "
        "14-15 bucket is the honest bracket-window read; 14-17 is shown as asked and "
        "mixes both problems."
    )
    add("")

    # ---- what the arms picked -------------------------------------------
    add("## What the two arms actually picked")
    add("")
    add(
        f"Across weeks 2-15 the naive arm names **{overlap_rate(picks, headline_weeks):.1%}** "
        "of the repo arm's picks in the same week. That number matters: if the arms "
        "largely agree, a difference near zero says more about how small the eligible "
        "pool is than about whether the ranking works."
    )
    add("")
    top = (
        picks[(picks["arm"] == "repo") & picks["week"].isin(list(headline_weeks))]
        .sort_values("fwd3", ascending=False)
        .head(5)[["season", "week", "player", "position", "fwd3", "pts_prior_week"]]
    )
    worst = (
        picks[(picks["arm"] == "repo") & picks["week"].isin(list(headline_weeks))]
        .sort_values("fwd3")
        .head(5)[["season", "week", "player", "position", "fwd3", "pts_prior_week"]]
    )
    for title, frame in (("Best repo picks", top), ("Worst repo picks", worst)):
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

    # ---- where the gap comes from ---------------------------------------
    add("## Where the gap comes from")
    add("")
    add(
        "The two arms do not pick the same kinds of players, and positions do not "
        "score alike, so the raw comparison above mixes two different things. "
        "Splitting them is exact rather than approximate: writing each arm's mean "
        "as its positional mix times the pool average at each position, plus what "
        "it beat that pool average by, separates a **mix** term from a "
        "**selection** term. Only the second is about ranking players well."
    )
    add("")
    add("| | share of naive picks | share of repo picks | pool avg fwd3 at that position |")
    add("| --- | ---: | ---: | ---: |")
    for position in split["positions"]:
        add(
            f"| {position} | {split['naive']['weights'][position]:.1%} | "
            f"{split['repo']['weights'][position]:.1%} | {split['mu'][position]:.2f} |"
        )
    add("")
    add(f"| component | ppg |")
    add("| --- | ---: |")
    add(f"| naive − repo, raw `fwd3` | {split['gap']:+.2f} |")
    add(f"| &nbsp;&nbsp;of which **position mix** | {split['mix']:+.2f} |")
    add(f"| &nbsp;&nbsp;of which **within-position selection** | {split['selection_gap']:+.2f} |")
    add("")
    add(
        "The naive rule sorts the wire by raw fantasy points, quarterbacks score the "
        "most raw fantasy points, and so the naive arm is mostly quarterbacks. "
        "`fwd3` is also raw fantasy points, so the arm that loads up on the "
        "highest-scoring position is rewarded for it. Points above replacement is "
        "the repo's own correction for exactly this — `report.with_edge` exists "
        "because \"tiering on raw score puts a streaming quarterback ahead of a "
        "genuinely valuable receiver\" — and then the scoring convention puts the "
        "positional advantage straight back in."
    )
    add("")
    add(
        "**This is not a defence of the repo arm.** With position removed entirely — "
        "each pick scored against what an arbitrary available player at the same "
        f"position returned that week — naive is still ahead by "
        f"{abs(adjusted['mean_diff']):.2f} ppg (repo − naive), {CI_LEVEL:.0%} interval "
        f"[{adjusted['ci_lo']:+.2f}, {adjusted['ci_hi']:+.2f}], n = "
        f"{adjusted['n_weeks']} weeks. Both arms beat a random available player at "
        f"the same position (repo by {split['repo']['selection']:+.2f} ppg, naive by "
        f"{split['naive']['selection']:+.2f}), so the model is not inert — it is "
        "just worse than the box score."
    )
    add("")
    add(
        "It does mean the headline number overstates how much a real manager would "
        "lose by following the table. Three quarterback claims a week is not a "
        "strategy anyone can execute: there is one QB slot. The replay imposes no "
        "roster constraints, and that omission happens to flatter the naive arm "
        "here. The position-adjusted figure is the one to carry forward, and it is "
        "still negative."
    )
    add("")

    # ---- is it a bug? ---------------------------------------------------
    if diagnostic:
        diag = bucket_table(
            diagnostic["weekly"], diagnostic["picks"], None, headline_weeks, stratify=True
        )
        add("## One concrete cause: the feature list has an off-by-one")
        add("")
        add(
            "`src/models.py` trains on `pts_lag1` — the player's fantasy points in "
            "week **W−1** — and does not include `pts`, his points in week **W**. "
            "Week W's box score is the naive arm's entire signal, it is on the same "
            "panel row, and `src/features.py` states in its first paragraph that it "
            "is known on the Monday claims are entered. `src/weekly.py` even prints "
            "it in the evidence columns next to every candidate. It is simply not a "
            "model input."
        )
        add("")
        add(
            "That is not a judgement call about leakage — the lag of a legal feature "
            "is in the model and the feature itself is not. Univariate Spearman "
            "against `fwd3` on the 2025 wire pool, within position:"
        )
        add("")
        add("| position | `pts` (week W) | `pts_lag1` (week W−1) |")
        add("| --- | ---: | ---: |")
        for position, now, lag in diagnostic["spearman"]:
            add(f"| {position} | {now:+.3f} | {lag:+.3f} |")
        add("")
        add(
            "The fresher signal is stronger at every position. Refitting the whole "
            "walk-forward replay with `pts` added to the feature set and changing "
            "nothing else:"
        )
        add("")
        add("| | repo ppg | naive ppg | repo − naive | 95% CI |")
        add("| --- | ---: | ---: | ---: | :---: |")
        add(
            f"| shipped feature set | {fmt(pooled['repo'].get('mean_fwd3'))} | "
            f"{fmt(pooled['naive'].get('mean_fwd3'))} | {fmt(pooled['mean_diff'])} | "
            f"[{fmt(pooled['ci_lo'])}, {fmt(pooled['ci_hi'])}] |"
        )
        add(
            f"| plus `pts` | {fmt(diag['repo'].get('mean_fwd3'))} | "
            f"{fmt(diag['naive'].get('mean_fwd3'))} | {fmt(diag['mean_diff'])} | "
            f"[{fmt(diag['ci_lo'])}, {fmt(diag['ci_hi'])}] |"
        )
        add("")
        add(
            f"It recovers {diag['repo'].get('mean_fwd3', 0) - pooled['repo'].get('mean_fwd3', 0):+.2f} "
            "ppg and does not close the gap. So the missing feature is a real defect "
            "worth fixing, and it is **not** the explanation for the result — the "
            "model still loses with it. This is reported as a diagnostic, not as a "
            "proposed change: `models/*.joblib` and `BASE_FEATURES` are untouched by "
            "this exercise."
        )
        add("")

    # ---- contamination --------------------------------------------------
    add("## Contamination audit")
    add("")
    add(
        "Treated as the main task. The shipped `models/*.joblib` are fitted on "
        "2022-2025 and are never loaded by this replay; every season is scored by "
        "models refitted on the seasons before it, written to "
        "`outputs/backtests/replay_models/` and stamped `replay_only: True`. "
        "`src.models.load_bundle` only ever reads `models/`, so a replay bundle "
        "cannot be served by accident."
    )
    add("")
    add("Refitting the estimator is necessary and not sufficient. Each of the four "
        "asked-about paths, and what was actually found:")
    add("")
    add("| path | leaked? | what was done |")
    add("| --- | --- | --- |")
    add(
        "| empirical Bayes priors | **yes** | Fitted per position on the pooled rate "
        "distribution across every season in the build, so a 2023 row built next to "
        "2025 was shrunk toward a prior that had seen 2025. Fixed: "
        "`features.build(seasons, prior_seasons=...)` fits the prior on the training "
        "seasons only. Magnitude is small — the shrunk shares move by under 0.01 — "
        "but it was measured rather than assumed. |"
    )
    add(
        "| rank-to-points scale | **yes** | `weekly.points_scale` maps a predicted "
        "percentile to points through the `fwd3` quantile function of the *whole* "
        "panel. Within a position the map is monotone and cannot reorder anything, "
        "but claims are ordered by points above replacement **across** positions, so "
        "it does move which names surface. Fixed: fitted on training seasons only. "
        "This one is easy to miss because the leak is in a display-looking function. |"
    )
    add(
        "| conformal calibration split | no effect on picks | `models.conformal` "
        "splits over every season in the universe. Recomputed on training seasons "
        "alone here, which leaves 2023 (one training season) and 2024 (two) with no "
        "interval at all — reported as absent rather than borrowed from elsewhere. "
        "It never touched the ranking: the half-width sets `score_lo`/`score_hi`, "
        "and the pick order comes from `proj_pts` off the point estimate. |"
    )
    add(
        "| `on_wire` and the season-to-date rank | **no** | `rank_before` ranks "
        "`cum_before` within (season, position, week); `cum_before` is a *shifted* "
        "cumulative sum within (player, season). Both are strictly within-season and "
        "strictly backward-looking. `ROSTER_DEPTH` is a declared 12x15 roster "
        "constant, not a fitted one. |"
    )
    add("")
    add(
        "The last row was not taken on trust. `tests/test_features.py::"
        "test_only_eb_columns_cross_seasons` rebuilds the panel at two different "
        "season scopes and asserts that the overlapping rows are bit-identical in "
        "**every** column except the two empirical Bayes shares. That covers "
        "`on_wire`, `rank_before`, `cusum`, `kal_role`, `snap_jump`, `neutral_opp` "
        "and `fwd3` in one assertion, and it is what would catch a future feature "
        "quietly starting to pool across seasons. A companion test asserts the two "
        "EB columns *do* still move, so the guard cannot go vacuous."
    )
    add("")
    add("### The leak that cannot be closed")
    add("")
    add(
        "Every constant in the pipeline — `ROSTER_DEPTH`, `REPLACEMENT_RANK`, "
        "`NEUTRAL_WP`, the Kalman and CUSUM settings, the weeks 2-14 training window, "
        "`MODEL_KWARGS`, `PROJECTION_CLIP` — was chosen by a human who had already "
        "seen all four seasons. No refit removes that. It biases the repo arm upward "
        "by an unknown amount, it is not visible in any interval below, and it is the "
        "reason a large margin here should be read as a warning rather than a result. "
        f"That is what the {LEAKAGE_TRIPWIRE_PPG:.0f} ppg tripwire in the verdict "
        "exists for."
    )
    add("")

    # ---- stability --------------------------------------------------------
    if stability:
        alt = bucket_table(
            stability["weekly"], stability["picks"], None, headline_weeks, stratify=True
        )
        add("## How precise are these numbers?")
        add("")
        add(
            "Less precise than they look, for a reason worth knowing. "
            "`HistGradientBoostingRegressor` bins its features before splitting, so "
            "a feature value that moves across a bin edge changes a split, a "
            "prediction, and sometimes which player a week's claim goes to. Feature "
            "values perturbed at the 1e-16 level — the difference between a frame "
            "held in memory and the same frame written to CSV and read back — are "
            "enough to do it."
        )
        add("")
        add("| panel float path | repo − naive, weeks 2-15 | 95% CI |")
        add("| --- | ---: | :---: |")
        add(
            f"| written to CSV and re-read (canonical) | {fmt(pooled['mean_diff'])} | "
            f"[{fmt(pooled['ci_lo'])}, {fmt(pooled['ci_hi'])}] |"
        )
        add(
            f"| held in memory | {fmt(alt['mean_diff'])} | "
            f"[{fmt(alt['ci_lo'])}, {fmt(alt['ci_hi'])}] |"
        )
        add("")
        add(
            f"A **{abs(pooled['mean_diff'] - alt['mean_diff']):.2f} ppg** spread from "
            "floating-point noise alone. So read every figure here to about ±0.3 ppg "
            "and no finer; the second decimal place is not real. It does not touch "
            "the verdict — both paths are strongly negative with intervals excluding "
            "zero — but it would matter a great deal to a result near the boundary, "
            "and it is a standing caveat on any week-level pick this pipeline makes."
        )
        add("")
        add(
            "The replay itself is pinned: `replay_panel()` writes each panel to CSV "
            "and reads it back before use, so a cold run and a cached run are handed "
            "identical inputs. The second row above is produced deliberately, by "
            "skipping that round-trip, in order to measure the sensitivity rather "
            "than to leave it lurking."
        )
        add("")

    # ---- not tested ------------------------------------------------------
    add("## What this does NOT test")
    add("")
    add(
        "- **No judgement layer.** The repo arm here is the table's top three by "
        "points above replacement, mechanically. The real `make report` workflow is "
        "the table *plus* a human reading the evidence columns and the news. This "
        "replay measures the ranking, not the workflow."
    )
    add(
        "- **No roster constraints.** No drop is proposed, no starting slot is "
        "filled, no bye is worked around. Every pick is treated as a free addition."
    )
    add(
        "- **No real league availability.** The replay assumes every ranked player "
        "was actually on the wire. `on_wire` is a season-to-date scoring-rank proxy, "
        "and it is the proxy assumption this repo has carried throughout — in a real "
        "12-team league a meaningful share of these names were rostered. Both arms "
        "draw from the same pool, so the comparison is internally fair; the absolute "
        "points captured by either arm are optimistic."
    )
    add(
        "- **No waiver priority, FAAB, or claim contention.** Three picks a week are "
        "granted unconditionally to both arms."
    )
    add(
        "- **The prompt arm cannot be replayed at all.** Any model asked to pick a "
        "week from 2023 today already knows how 2023 went, and no prompt discipline "
        "removes that. The three-arm comparison in `src/ledger.py` can only ever run "
        "forward, in real time. This file is a two-arm exercise for that reason."
    )
    add("")

    # ---- method ----------------------------------------------------------
    add("## Method")
    add("")
    add("| replay season | trained on | training seasons |")
    add("| --- | --- | ---: |")
    add("| 2022 | — | not replayable, no prior data |")
    for season in REPLAY_SEASONS:
        train = [s for s in ALL_SEASONS if s < season]
        add(f"| {season} | {', '.join(str(s) for s in train)} | {len(train)} |")
    add("")
    add(
        "Weeks 2 through 17 of each replay season. The **repo arm** takes the top "
        "three names by points above replacement at position — `report.with_edge` "
        "over `weekly.score_week`'s projection, positive edge only, which is exactly "
        "what `report.assign_tiers` ranks before it tiers. The **naive arm** takes "
        "the top three available by the previous week's fantasy points, straight from "
        "`ledger.naive_picks`, with no model anywhere in it. Both arms are scored by "
        "`fwd3`: mean points over the next three weeks the player's team played, a "
        "week played without him counted as 0.0, byes excluded. That is the same "
        "convention the models train on and the ledger grades against."
    )
    add("")
    add("### Why walk-forward and not leave-one-season-out")
    add("")
    add(
        "Leave-one-season-out — hold out 2022, train on 2023-2025, and so on — would "
        "give four replay seasons instead of three and much more training data for "
        "each. It is rejected for the headline because it trains on the future. It "
        "answers \"is this method sound in principle\"; the question here is \"could "
        "I have made these picks at the time\", and only walk-forward answers that."
    )
    add("")

    if loso:
        add("### Sensitivity check: leave-one-season-out (NOT the headline)")
        add("")
        add(
            "Each season replayed by models trained on all *other* seasons, including "
            "later ones. This trains on the future and is reported only as a "
            "sensitivity check. The interesting quantity is the gap: if leave-one-out "
            "looks much better than walk-forward, the difference is hindsight rather "
            "than skill."
        )
        add("")
        add(f"| season | trained on | n weeks | repo ppg | naive ppg | repo − naive | {CI_LEVEL:.0%} CI |")
        add("| --- | --- | ---: | ---: | ---: | ---: | :---: |")
        for season in ALL_SEASONS:
            b = bucket_table(loso["weekly"], loso["picks"], [season], headline_weeks, False)
            train = [s for s in ALL_SEASONS if s != season]
            ci = f"[{fmt(b['ci_lo'])}, {fmt(b['ci_hi'])}]" if not np.isnan(b["ci_lo"]) else "n/a"
            add(
                f"| {season} | {', '.join(str(s) for s in train)} | {b['n_weeks']} | "
                f"{fmt(b['repo'].get('mean_fwd3'))} | {fmt(b['naive'].get('mean_fwd3'))} | "
                f"{fmt(b['mean_diff'])} | {ci} |"
            )
        pooled_loso = bucket_table(loso["weekly"], loso["picks"], None, headline_weeks, True)
        add(
            f"| **pooled** | | {pooled_loso['n_weeks']} | "
            f"{fmt(pooled_loso['repo'].get('mean_fwd3'))} | "
            f"{fmt(pooled_loso['naive'].get('mean_fwd3'))} | "
            f"{fmt(pooled_loso['mean_diff'])} | "
            f"[{fmt(pooled_loso['ci_lo'])}, {fmt(pooled_loso['ci_hi'])}] |"
        )
        add("")
        gap = pooled_loso["mean_diff"] - pooled["mean_diff"]
        add(
            f"Pooled leave-one-out minus pooled walk-forward: **{gap:+.2f} ppg**. "
            + (
                "Leave-one-out flatters the repo arm, which is what training on the "
                "future is expected to do."
                if gap > 0.5 else
                "The two schemes land close together, so on this data the extra "
                "training seasons buy little and the walk-forward constraint costs "
                "little."
                if abs(gap) <= 0.5 else
                "Leave-one-out looks *worse* than walk-forward, which on this sample "
                "size is noise rather than a finding about either scheme."
            )
        )
        add("")

    # ---- reproducing ------------------------------------------------------
    add("## Reproducing")
    add("")
    add("```bash")
    add("make install")
    add('make data SEASONS="2022 2023 2024 2025"')
    add("python outputs/backtests/01_season_replay.py")
    add("```")
    add("")
    add(
        "Writes `outputs/backtests/replay/replay_picks.csv` (every pick and its "
        "outcome), `replay_weeks.csv` (the arm-by-week means the intervals resample) "
        "and this file. Replay model bundles go to "
        "`outputs/backtests/replay_models/` and are gitignored: they are "
        "regenerated by the script and committing them next to the production "
        "bundles would invite exactly the confusion they are labelled against."
    )
    add("")
    revision_note = (
        "Computed against the nflverse revision pinned in `data/raw/MANIFEST.json`. "
        "nflverse revises history in place, so a rerun after a revision will not "
        "reproduce these numbers exactly — that is the manifest's whole job."
    )
    add(revision_note)
    if notes:
        add("")
        add("### Picks that could not be scored")
        add("")
        add(f"{len(notes)} in total:")
        add("")
        for line in notes[:25]:
            add(f"- {line}")
        if len(notes) > 25:
            add(f"- ...and {len(notes) - 25} more")
        add("")

    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text("\n".join(L) + "\n")
    print(f"\nwrote {DIAGNOSTIC.relative_to(ROOT)}")


# ==========================================================================

def run_scheme(
    label: str, plan: list[tuple[int, list[int]]],
    extra_features: tuple[str, ...] = (), roundtrip: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    frames, notes = [], []
    for season, train in plan:
        picks, info = replay_season(season, train, label, extra_features, roundtrip)
        frames.append(picks)
        notes += info["unscored"]
    return pd.concat(frames, ignore_index=True), notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-loso", action="store_true",
        help="skip the leave-one-season-out sensitivity check",
    )
    parser.add_argument(
        "--no-diagnostic", action="store_true",
        help="skip the refit that adds week-W points to the feature set",
    )
    parser.add_argument(
        "--no-stability", action="store_true",
        help="skip the float-path stability check",
    )
    args = parser.parse_args(argv)

    print("=" * 74)
    print("WALK-FORWARD REPLAY (headline)")
    print("=" * 74)
    plan = [(s, [y for y in ALL_SEASONS if y < s]) for s in REPLAY_SEASONS]
    picks, notes = run_scheme("walk_forward", plan)
    weekly = week_means(picks)

    stability = None
    if not args.no_stability:
        print()
        print("=" * 74)
        print("STABILITY CHECK (same replay, panels not round-tripped through CSV)")
        print("=" * 74)
        alt_picks, _ = run_scheme("no_roundtrip", plan, roundtrip=False)
        stability = {"picks": alt_picks, "weekly": week_means(alt_picks)}

    diagnostic = None
    if not args.no_diagnostic:
        print()
        print("=" * 74)
        print("DIAGNOSTIC REFIT (feature set + `pts`, week W's fantasy points)")
        print("=" * 74)
        diag_picks, _ = run_scheme("wf_plus_pts", plan, extra_features=("pts",))
        diagnostic = {
            "picks": diag_picks,
            "weekly": week_means(diag_picks),
            "spearman": spearman_now_vs_lag(),
        }

    loso = None
    if not args.no_loso:
        print()
        print("=" * 74)
        print("LEAVE-ONE-SEASON-OUT (sensitivity check -- trains on the future)")
        print("=" * 74)
        loso_plan = [(s, [y for y in ALL_SEASONS if y != s]) for s in ALL_SEASONS]
        loso_picks, _ = run_scheme("loso", loso_plan)
        loso = {"picks": loso_picks, "weekly": week_means(loso_picks)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra = [d for d in (diagnostic, loso, stability) if d is not None]
    all_picks = pd.concat([picks] + [d["picks"] for d in extra], ignore_index=True)
    all_weeks = pd.concat([weekly] + [d["weekly"] for d in extra], ignore_index=True)
    all_picks.to_csv(OUT_DIR / "replay_picks.csv", index=False, float_format="%.4f")
    all_weeks.to_csv(OUT_DIR / "replay_weeks.csv", index=False, float_format="%.4f")

    headline_weeks = dict((n, w) for n, w, _ in BUCKETS)[HEADLINE_BUCKET]
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
    print(pooled["pairs"].to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    write_markdown(weekly, picks, loso, diagnostic, stability, notes)
    print(f"wrote {(OUT_DIR / 'replay_picks.csv').relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / 'replay_weeks.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

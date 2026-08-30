"""Build the player-week panel that every later step reads.

NO LOOKAHEAD. Every feature on a row for season S week W is computable from data
that exists on the Monday after week W's games -- the morning waiver claims are
entered. Concretely: usage features use weeks 1..W inclusive (week W's box score
is known Monday), the availability proxy uses weeks 1..W-1 (it asks whether the
player was rosterable going *into* the week), and the only forward-looking column
is `fwd3`, the training target, which is naturally missing for the current week.

Nothing here may be recomputed with hindsight. If a feature ever needs a value
from week W+1 or later, it does not belong in this file.

Two documented judgement calls, both stated here rather than buried:

1. The empirical Bayes beta priors are fit per position on the pooled
   season-to-date rate distribution across all seasons in the build. That is a
   position-level constant, not a player-level one, so it cannot leak a specific
   player's future -- but it is technically fit on data that includes seasons
   later than a given row. Refitting the prior per week costs a lot of
   complexity for a prior that barely moves; the choice is called out here so it
   is a decision rather than an oversight.

   `eb_tgt_share` and `eb_car_share` are the ONLY two columns this file produces
   that depend on which other seasons are in the build. Every other column is
   computed within a single (player, season) or (season, position, week) group,
   so a row for season S is bit-identical whether the build covers S alone or S
   through 2025 -- `tests/test_features.py::test_only_eb_columns_cross_seasons`
   asserts exactly that rather than leaving it to be believed. For live weekly
   scoring the shared prior is contemporaneous and harmless. For a walk-forward
   replay of a past season it is a genuine leak, so `build(seasons,
   prior_seasons=...)` restricts the rows the prior is fitted on; see
   `outputs/backtests/01_season_replay.py`.

2. `snap_jump` and the CUSUM reference window look back over prior *appearances*,
   not prior week numbers: a bye produces no snap row, and "the last two weeks he
   played" is the meaningful comparison for a role change. `fwd3` instead aligns
   on week *numbers*, since "the next three weeks" is calendar-forward. A week in
   that span where the player's team played but he did not appear scores 0.0 --
   claiming someone who then sits is worth nothing, and dropping those weeks
   would quietly grade every pickup on his good games only. Team byes are
   excluded from the average rather than scored 0, since a bye says nothing
   about the player. `fwd3_played`, the games-played-only alternative, is kept
   alongside it because the choice between them moves measured input importance
   materially -- see outputs/backtests/results_input_importance.txt.

Usage:
    python -m src.features                 # default seasons
    python -m src.features 2022 2023       # specific seasons
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "panel.csv"

DEFAULT_SEASONS = range(2022, 2027)
POSITIONS = ["QB", "RB", "WR", "TE"]

# 12 teams x 15-man rosters, split by how deep each position is actually rostered.
# A player ranked below this in season-to-date scoring is treated as wire-available.
ROSTER_DEPTH = {"QB": 18, "RB": 46, "WR": 60, "TE": 18}

# Neutral script: plays where the possession team's pre-snap win probability
# is in this range. Outside it, one side is already playing from ahead or
# behind, and volume stops describing the player and starts describing the
# scoreboard -- garbage-time targets and clock-killing carries both count the
# same as meaningful ones unless this filter excludes them.
NEUTRAL_WP = (0.20, 0.80)

# Kalman local-level filter on the snap series.
KALMAN_Q = 0.010
KALMAN_R = 0.020

# Upward CUSUM on standardized snap share.
CUSUM_K = 0.5
CUSUM_H = 3.0
CUSUM_WARMUP = 3
CUSUM_SD_FLOOR = 0.05


def join_key(names: pd.Series) -> pd.Series:
    """Lowercased name with periods and apostrophes stripped."""
    return (
        names.astype("string")
        .str.lower()
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )


def load_season(year: int) -> pd.DataFrame | None:
    """Weekly stats joined to snap share for one season, or None if not fetched."""
    stats_path = RAW_DIR / f"stats_player_week_{year}.csv"
    snaps_path = RAW_DIR / f"snap_counts_{year}.csv"
    if not stats_path.exists() or not snaps_path.exists():
        print(f"{year}: source files missing, skipping")
        return None

    stats = pd.read_csv(stats_path, low_memory=False)
    stats = stats[
        (stats["season_type"] == "REG") & (stats["position"].isin(POSITIONS))
    ].copy()

    snaps = pd.read_csv(snaps_path, low_memory=False)
    snaps = snaps[snaps["game_type"] == "REG"].copy()

    stats["season"] = year
    snaps["season"] = year
    stats["join_key"] = join_key(stats["player_display_name"])
    snaps["join_key"] = join_key(snaps["player"])

    # One snap row per player-team-week: a player can appear twice in a game file,
    # and max() keeps the real offensive workload rather than diluting it.
    snaps = snaps.groupby(
        ["join_key", "team", "week", "season"], as_index=False
    )["offense_pct"].max()
    snaps = snaps.rename(columns={"offense_pct": "snap"})

    merged = stats.merge(snaps, on=["join_key", "team", "week", "season"], how="left")
    matched = merged["snap"].notna()
    print(
        f"{year}: {len(merged):,} stat rows, {int(matched.sum()):,} matched to snaps "
        f"({matched.mean():.1%}), {int((~matched).sum()):,} dropped"
    )
    return merged[matched].copy()


def league_points(frame: pd.DataFrame) -> pd.Series:
    """Half PPR, full PPR for tight ends."""
    return pd.Series(
        np.where(
            frame["position"] == "TE",
            frame["fantasy_points_ppr"],
            (frame["fantasy_points"] + frame["fantasy_points_ppr"]) / 2,
        ),
        index=frame.index,
    )


def fit_beta_prior(rates: pd.Series) -> tuple[float, float]:
    """Method-of-moments beta prior. Falls back to uniform when overdispersed."""
    rates = rates.dropna()
    rates = rates[(rates >= 0) & (rates <= 1)]
    if len(rates) < 2:
        return 1.0, 1.0
    mean = float(rates.mean())
    var = float(rates.var(ddof=1))
    if var <= 0 or var >= mean * (1 - mean):
        return 1.0, 1.0
    c = mean * (1 - mean) / var - 1
    if c <= 0:
        return 1.0, 1.0
    return mean * c, (1 - mean) * c


def empirical_bayes_share(
    panel: pd.DataFrame, successes: str, trials: str, out: str,
    prior_mask: pd.Series | None = None,
) -> pd.Series:
    """Shrink each player's season-to-date share toward a per-position beta prior.

    Season-to-date is cumulative through the current week inclusive -- week W's
    box score is known on the Monday the claim is made.

    `prior_mask` selects the rows the beta prior is *fitted* on; the shrinkage is
    applied to every row either way. Default (None) fits on every row in the
    build, which is the documented judgement call at the top of this file: the
    prior is a position-level constant, so it cannot leak a specific player's
    future, but it is technically fitted on rows from seasons later than the row
    it is applied to. For live weekly scoring that is contemporaneous and
    harmless. For a walk-forward replay of a past season it is not -- a 2023
    replay must not see a prior shaped by 2024 and 2025 -- so
    `outputs/backtests/01_season_replay.py` passes a mask restricted to that
    replay's training seasons. See the module docstring of that file.
    """
    group = panel.groupby(["player_id", "season"], sort=False)
    cum_successes = group[successes].cumsum()
    cum_trials = group[trials].cumsum()

    raw_rate = np.where(cum_trials > 0, cum_successes / cum_trials.replace(0, np.nan), np.nan)
    raw_rate = pd.Series(raw_rate, index=panel.index)

    result = pd.Series(np.nan, index=panel.index, dtype=float)
    for position, idx in panel.groupby("position", sort=False).groups.items():
        fit_on = raw_rate.loc[idx]
        if prior_mask is not None:
            fit_on = fit_on[prior_mask.loc[idx].to_numpy()]
        alpha, beta = fit_beta_prior(fit_on)
        result.loc[idx] = (cum_successes.loc[idx] + alpha) / (
            cum_trials.loc[idx] + alpha + beta
        )
        print(
            f"  {out} prior {position}: alpha={alpha:.3f} beta={beta:.3f} "
            f"(fitted on {len(fit_on.dropna()):,} rows)"
        )
    return result


def kalman_local_level(series: np.ndarray) -> np.ndarray:
    """1-D local-level filter, initialized at the first observation."""
    out = np.empty(len(series), dtype=float)
    if len(series) == 0:
        return out
    x = float(series[0])
    p = KALMAN_R
    out[0] = x
    for i in range(1, len(series)):
        p += KALMAN_Q
        k = p / (p + KALMAN_R)
        x = x + k * (float(series[i]) - x)
        p = (1 - k) * p
        out[i] = x
    return out


def upward_cusum(series: np.ndarray) -> np.ndarray:
    """Upward CUSUM against the player's own prior mean and sd, reset on signal."""
    n = len(series)
    out = np.zeros(n, dtype=float)
    s = 0.0
    for i in range(n):
        if i < CUSUM_WARMUP:
            continue
        prior = series[:i]
        mean = float(np.mean(prior))
        sd = max(float(np.std(prior, ddof=1)) if len(prior) > 1 else 0.0, CUSUM_SD_FLOOR)
        z = (float(series[i]) - mean) / sd
        s = max(0.0, s + z - CUSUM_K)
        out[i] = s
        if s > CUSUM_H:
            s = 0.0
    return out


PBP_COLUMNS = [
    "season", "week", "season_type", "wp", "pass_attempt", "rush_attempt",
    "receiver_player_id", "rusher_player_id",
]


def load_neutral_opp(seasons) -> tuple[pd.DataFrame, list[int]]:
    """Targets plus carries on neutral-script plays, per player-week.

    Read straight from play_by_play_{year}.csv.gz -- `wp` is the possession
    team's pre-snap win probability, so it is known before the play and
    carries no lookahead. A season whose pbp file has not been fetched is
    skipped and reported, not silently zeroed, so the caller can tell "no
    neutral volume" from "not covered by this build".
    """
    frames = []
    covered: list[int] = []
    for year in seasons:
        path = RAW_DIR / f"play_by_play_{year}.csv.gz"
        if not path.exists():
            print(f"{year}: play_by_play missing, neutral_opp not computed for this season")
            continue

        pbp = pd.read_csv(path, low_memory=False, usecols=PBP_COLUMNS)
        pbp = pbp[
            (pbp["season_type"] == "REG") & pbp["wp"].between(*NEUTRAL_WP)
        ]

        targets = (
            pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()]
            .groupby(["week", "receiver_player_id"])
            .size()
        )
        targets.index = targets.index.set_names(["week", "player_id"])

        carries = (
            pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()]
            .groupby(["week", "rusher_player_id"])
            .size()
        )
        carries.index = carries.index.set_names(["week", "player_id"])

        combined = pd.concat(
            [targets.rename("t"), carries.rename("c")], axis=1
        ).fillna(0.0)
        combined["neutral_opp"] = combined["t"] + combined["c"]
        combined = combined.reset_index()
        combined["season"] = year
        frames.append(combined[["player_id", "season", "week", "neutral_opp"]])
        covered.append(year)

    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "week", "neutral_opp"]), covered
    return pd.concat(frames, ignore_index=True), covered


def load_schedule(
    games_path: Path | str | None = None,
) -> tuple[set[tuple[int, int, str]], dict[int, int]]:
    """(season, week, team) triples that had a regular-season game, and each
    season's final regular-season week. Used to tell a bye from a healthy scratch.

    `games_path` defaults to the downloaded schedule; tests point it at the
    committed fixture so they do not depend on `make data` having run.
    """
    games = pd.read_csv(games_path or (RAW_DIR / "games.csv"), low_memory=False)
    games = games[games["game_type"] == "REG"]
    played = set(
        zip(games["season"], games["week"], games["away_team"])
    ) | set(zip(games["season"], games["week"], games["home_team"]))
    last_week = games.groupby("season")["week"].max().to_dict()
    return played, last_week


def forward_three(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean points over weeks W+1..W+3, under both defensible definitions.

    `fwd3` (the training target) scores a week his team played but he did not
    appear in as 0.0 -- that is what the claim actually returned, and it is
    defined across the whole wire universe.

    `fwd3_played` averages only the games he actually played. It grades the
    player rather than the claim, and is undefined for someone who never plays
    again, so it silently drops the busts that matter most.

    Both are kept because they are not interchangeable: the choice moves
    measured input importance materially (see outputs/backtests). A team bye is
    excluded from both, since a bye says nothing about the player.
    """
    played, last_week = load_schedule()
    claimed = pd.Series(np.nan, index=panel.index, dtype=float)
    playing = pd.Series(np.nan, index=panel.index, dtype=float)
    for (_, season), group in panel.groupby(["player_id", "season"], sort=False):
        by_week = dict(zip(group["week"], group["pts"]))
        final = last_week.get(season, 18)
        for idx, week, team in zip(group.index, group["week"], group["team"]):
            span = [
                w
                for w in (week + 1, week + 2, week + 3)
                if w <= final and (season, w, team) in played
            ]
            if span:
                claimed.loc[idx] = float(np.mean([by_week.get(w, 0.0) for w in span]))
            appeared = [by_week[w] for w in span if w in by_week]
            if appeared:
                playing.loc[idx] = float(np.mean(appeared))
    return claimed, playing


def build(seasons, prior_seasons=None) -> pd.DataFrame:
    """The panel over `seasons`.

    `prior_seasons`, when given, restricts the seasons the empirical Bayes beta
    priors are *fitted* on -- the one quantity in this file computed across the
    whole build rather than per season. Default None fits them on everything,
    which is the shipped behaviour. Every other column here is computed within a
    single (player, season) or (season, position, week) group and is therefore
    bit-identical whether or not later seasons are in the build; that is
    asserted in tests/test_features.py rather than asserted in prose.
    """
    frames = [f for f in (load_season(y) for y in seasons) if f is not None]
    if not frames:
        raise SystemExit("no seasons available -- run `make data` first")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    panel["pts"] = league_points(panel)

    # --- team context -------------------------------------------------------
    team_totals = panel.groupby(["season", "team", "week"])[["targets", "carries"]].transform("sum")
    panel["team_tgt"] = team_totals["targets"]
    panel["team_car"] = team_totals["carries"]
    panel["tgt_share"] = panel["targets"] / panel["team_tgt"].replace(0, np.nan)
    panel["carry_share"] = panel["carries"] / panel["team_car"].replace(0, np.nan)
    panel["wopr_opp"] = panel["carries"] + 2.5 * panel["targets"]

    by_player = panel.groupby(["player_id", "season"], sort=False)

    # --- role trajectory ----------------------------------------------------
    prior_two = by_player["snap"].transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).mean()
    )
    panel["snap_jump"] = panel["snap"] - prior_two

    if prior_seasons is None:
        prior_mask = None
        print("empirical bayes priors (fitted on every season in the build):")
    else:
        prior_mask = panel["season"].isin(list(prior_seasons))
        print(f"empirical bayes priors (fitted on {sorted(prior_seasons)} only):")
    panel["eb_tgt_share"] = empirical_bayes_share(
        panel, "targets", "team_tgt", "eb_tgt_share", prior_mask
    )
    panel["eb_car_share"] = empirical_bayes_share(
        panel, "carries", "team_car", "eb_car_share", prior_mask
    )

    panel["kal_role"] = by_player["snap"].transform(
        lambda s: kalman_local_level(s.to_numpy())
    )
    panel["cusum"] = by_player["snap"].transform(lambda s: upward_cusum(s.to_numpy()))

    panel["pts_lag1"] = by_player["pts"].shift(1)

    # --- target -------------------------------------------------------------
    panel["fwd3"], panel["fwd3_played"] = forward_three(panel)

    # --- neutral-script opportunity -------------------------------------
    neutral, covered_seasons = load_neutral_opp(seasons)
    if covered_seasons:
        panel = panel.merge(neutral, on=["player_id", "season", "week"], how="left")
        covered = panel["season"].isin(covered_seasons)
        # A row in a covered season with no merge match had zero neutral-script
        # plays that week -- a real zero. A row in an uncovered season is
        # genuinely unknown and stays NaN rather than being reported as zero.
        panel.loc[covered, "neutral_opp"] = panel.loc[covered, "neutral_opp"].fillna(0.0)
    else:
        panel["neutral_opp"] = np.nan

    # --- waiver availability proxy -----------------------------------------
    # Strictly *before* this week: it asks whether the player was rosterable
    # going into the week, which is what determines whether he sat on the wire.
    panel["cum_before"] = by_player["pts"].transform(lambda s: s.shift(1).cumsum().fillna(0.0))
    panel["rank_before"] = panel.groupby(["season", "position", "week"])["cum_before"].rank(
        ascending=False, method="min"
    )
    panel["on_wire"] = panel["rank_before"] > panel["position"].map(ROSTER_DEPTH)

    keep = [
        "player_id", "player_display_name", "position", "team", "season", "week",
        "snap", "snap_jump", "targets", "carries", "receptions", "air_yards_share",
        "team_tgt", "team_car", "tgt_share", "carry_share", "wopr_opp",
        "eb_tgt_share", "eb_car_share", "kal_role", "cusum",
        "pts", "pts_lag1", "fwd3", "fwd3_played", "neutral_opp",
        "cum_before", "rank_before", "on_wire",
    ]
    return panel[keep]


def main(argv: list[str]) -> int:
    seasons = [int(a) for a in argv] if argv else list(DEFAULT_SEASONS)
    panel = build(seasons)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)

    print(f"\nwrote {PANEL_PATH.relative_to(ROOT)}")
    print(f"rows: {len(panel):,}")
    print(f"seasons: {sorted(panel['season'].unique().tolist())}")
    print(f"positions: {panel['position'].value_counts().to_dict()}")
    print(f"on_wire rows: {int(panel['on_wire'].sum()):,}")
    print(f"rows with fwd3: {int(panel['fwd3'].notna().sum()):,}")
    print(f"rows with fwd3_played: {int(panel['fwd3_played'].notna().sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

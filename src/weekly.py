"""Score the waiver-wire pool for one season and week.

Writes outputs/weekly/{season}/wk{NN}.csv: every wire-eligible player ranked by
model score, with the raw usage numbers that produced it sitting in the same
row. A bare score is not actionable -- if a name surfaces, the reason it
surfaced has to be visible next to it, or there is no way to tell a real role
change from a model artifact.

This deliberately does NOT filter by availability in any particular league. It
narrows a few thousand player-weeks to a few dozen names; confirming who is
actually free is a separate manual step, and baking a roster into this file
would make the output un-auditable against history.

The current week is derived from the schedule in data/raw/games.csv, never from
calendar arithmetic. Week boundaries move for byes, international kickoffs and
the Week 18 gap; "days since some Thursday in September" is wrong several times
a season and wrong silently.

Run:
    python -m src.weekly --season 2025 --week 8
    python -m src.weekly --current
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch import data_revision, load_manifest
from src.models import (
    MODEL_DIR,
    POSITIONS,
    WEEKS,
    load_bundle,
    load_panel,
    wire_universe,
)

ROOT = Path(__file__).resolve().parents[1]
GAMES_PATH = ROOT / "data" / "raw" / "games.csv"
WEEKLY_DIR = ROOT / "outputs" / "weekly"
SEEN_MANIFEST = WEEKLY_DIR / "LAST_MANIFEST.json"

# Shown alongside the score so every name carries its own evidence.
EVIDENCE = [
    "snap",
    "snap_jump",
    "targets",
    "tgt_share",
    "eb_tgt_share",
    "carries",
    "carry_share",
    "eb_car_share",
    "receptions",
    "air_yards_share",
    "wopr_opp",
    "kal_role",
    "cusum",
    "pts",
    "pts_lag1",
    "cum_before",
    "rank_before",
]

SEASON_IN_FILENAME = re.compile(r"(20\d{2})")

# Clip on the rank scale before mapping to points; see points_scale().
PROJECTION_CLIP = (0.01, 0.99)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------

def load_games(games_path: Path | str = GAMES_PATH) -> pd.DataFrame:
    path = Path(games_path)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `make data` first")
    games = pd.read_csv(path, low_memory=False)
    return games[games["game_type"] == "REG"].copy()


def current_week(
    season: int, as_of: date | None = None, games: pd.DataFrame | None = None
) -> int | None:
    """The latest regular-season week of `season` that is fully in the books.

    Read from the schedule, not from the calendar. A week counts as complete
    once its last scheduled game day is strictly in the past, so a Tuesday
    morning run sees the Monday night game that just finished. Returns None if
    the season has not yet completed a week.
    """
    as_of = as_of or date.today()
    games = load_games() if games is None else games
    rows = games[games["season"] == season]
    if rows.empty:
        return None
    finished = rows.groupby("week")["gameday"].max()
    complete = finished[finished < as_of.isoformat()]
    return int(complete.index.max()) if len(complete) else None


def current_season_week(
    as_of: date | None = None, games: pd.DataFrame | None = None
) -> tuple[int, int]:
    """The (season, week) a Tuesday run should be scoring.

    Walks seasons newest-first and takes the first with a completed week, so
    the September gap between a new season kicking off and its first Monday
    night resolving still points at last season's Week 18 instead of crashing
    or inventing a Week 0.
    """
    as_of = as_of or date.today()
    games = load_games() if games is None else games
    for season in sorted(games["season"].unique(), reverse=True):
        week = current_week(int(season), as_of, games)
        if week is not None:
            return int(season), week
    raise SystemExit("no completed regular-season week found in games.csv")


# --------------------------------------------------------------------------
# upstream revision check
# --------------------------------------------------------------------------

def check_revision(season: int) -> bool:
    """Warn loudly if nflverse revised a *prior* season since the last run.

    Warns, never fails. A revision does not make today's table wrong, but it
    does mean anything cached from before it -- backtests, the model card, last
    week's numbers -- is no longer comparable, and that is worth noticing at the
    moment it happens rather than a month later.
    """
    manifest = load_manifest()
    current = {name: rec.get("sha256") for name, rec in manifest.get("files", {}).items()}
    if not SEEN_MANIFEST.exists():
        return False

    previous = json.loads(SEEN_MANIFEST.read_text()).get("files", {})
    changed = []
    for name, digest in current.items():
        if name not in previous or previous[name] == digest:
            continue
        match = SEASON_IN_FILENAME.search(name)
        # games.csv carries no season and legitimately changes every week.
        if match and int(match.group(1)) < season:
            changed.append(name)

    if changed:
        print("!" * 74)
        print("!! nflverse REVISED HISTORY since the last run:")
        for name in sorted(changed):
            print(f"!!   {name}")
        print("!! Prior-season inputs changed. Backtests, the model card and any")
        print("!! cached weekly tables were computed against different bytes and")
        print("!! are NOT comparable. Re-run `make panel models` to bring them")
        print("!! back in line. Continuing anyway -- this week's table is fine.")
        print("!" * 74)
    return bool(changed)


def record_revision() -> None:
    manifest = load_manifest()
    SEEN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SEEN_MANIFEST.write_text(
        json.dumps(
            {
                "revision": data_revision(manifest),
                "recorded": date.today().isoformat(),
                "files": {
                    name: rec.get("sha256")
                    for name, rec in sorted(manifest.get("files", {}).items())
                },
            },
            indent=2,
        )
        + "\n"
    )


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def points_scale(panel: pd.DataFrame, position: str) -> np.ndarray:
    """The position's historical fwd3 distribution, for turning a rank into points.

    The models rank; a range in fantasy points is what is actually readable. The
    mapping is the empirical quantile function of fwd3 over the training
    universe, so a predicted 0.9 reads as "what the 90th percentile wire player
    at this position went on to average".

    The mapping input is clipped to [0.01, 0.99]. Un-clipped, a conformal upper
    bound that saturates at rank 1.0 maps to the single best fwd3 ever recorded
    at the position, and the range reads "1 to 32 points" -- the record, not a
    projection. The clip keeps the top of the range at a genuinely attainable
    outcome. It does not make the interval narrower or more certain; it only
    stops the tail from being reported as if it were a forecast.
    """
    trained = panel[
        panel["week"].between(*WEEKS)
        & panel["on_wire"]
        & (panel["position"] == position)
        & panel["fwd3"].notna()
    ]
    return np.sort(trained["fwd3"].to_numpy())


def score_week(season: int, week: int) -> pd.DataFrame:
    panel = load_panel()
    pool = panel[
        (panel["season"] == season)
        & (panel["week"] == week)
        & panel["on_wire"]
        & panel["snap"].notna()
    ].copy()
    if pool.empty:
        raise SystemExit(
            f"no wire-eligible rows for {season} week {week} -- "
            "check the panel covers that week"
        )

    if not WEEKS[0] <= week <= WEEKS[1]:
        print(
            f"note: week {week} is outside the {WEEKS[0]}-{WEEKS[1]} training "
            "window; scores are extrapolated"
        )

    frames = []
    for position in POSITIONS:
        rows = pool[pool["position"] == position]
        if rows.empty:
            continue
        bundle = load_bundle(position)
        features = bundle["features"]
        missing = [f for f in features if f not in rows.columns]
        if missing:
            raise SystemExit(
                f"panel is missing features the {position} model expects: {missing} "
                "-- rebuild with `make panel models`"
            )

        rows = rows.copy()
        half = bundle["conformal_half_width"]
        rows["score"] = bundle["model"].predict(rows[features])
        rows["score_lo"] = (rows["score"] - half).clip(0, 1)
        rows["score_hi"] = (rows["score"] + half).clip(0, 1)

        scale = points_scale(panel, position)
        for column, source in (
            ("proj_pts", "score"),
            ("proj_pts_lo", "score_lo"),
            ("proj_pts_hi", "score_hi"),
        ):
            rows[column] = np.quantile(scale, rows[source].clip(*PROJECTION_CLIP))

        rows["conformal_half_width"] = half
        rows["model_coverage"] = bundle["empirical_coverage"]
        rows["pos_rank"] = rows["score"].rank(ascending=False, method="min").astype(int)
        frames.append(rows)

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    table.insert(0, "overall_rank", np.arange(1, len(table) + 1))

    columns = [
        "overall_rank", "pos_rank", "player_display_name", "position", "team",
        "season", "week", "score", "score_lo", "score_hi",
        "proj_pts", "proj_pts_lo", "proj_pts_hi",
        *EVIDENCE,
        "conformal_half_width", "model_coverage",
    ]
    return table[[c for c in columns if c in table.columns]]


def write_week(season: int, week: int) -> Path:
    table = score_week(season, week)
    out_dir = WEEKLY_DIR / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"wk{week:02d}.csv"
    table.to_csv(path, index=False, float_format="%.4f")

    print(f"\n{season} week {week}: {len(table)} wire candidates -> "
          f"{path.relative_to(ROOT)}")
    for position in POSITIONS:
        rows = table[table["position"] == position].head(5)
        if rows.empty:
            continue
        print(f"\ntop 5 {position}")
        view = rows[[
            "pos_rank", "player_display_name", "team", "score",
            "proj_pts_lo", "proj_pts_hi", "snap", "targets", "tgt_share",
            "carries", "carry_share", "pts",
        ]]
        print(view.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--current",
        action="store_true",
        help="derive season and week from the schedule in games.csv",
    )
    args = parser.parse_args(argv)

    if args.current or args.season is None or args.week is None:
        season, week = current_season_week()
        if args.season is not None:
            season = args.season
        if args.week is not None:
            week = args.week
        print(f"resolved from schedule: {season} week {week}")
    else:
        season, week = args.season, args.week

    check_revision(season)
    write_week(season, week)
    record_revision()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Grade past recommendations against what actually happened.

This is the only thing in the repo that tests the WORKFLOW rather than the
metrics. Every other output measures whether a feature predicts fwd3; this asks
whether following the reports beat not following them. If a season of
recommendations does not clear the naive benchmark, the apparatus is decoration,
and the point of writing this down is to find that out rather than assume
otherwise.

Two benchmarks per claim, both computed from the same wire pool the
recommendation came from:

- **hot hand** -- whoever simply scored the most points the week before. This is
  what most managers actually do, it costs no model, and it is the bar to clear.
  Beating fwd3 on average is not interesting if chasing last week's box score
  does the same thing.
- **ceiling** -- the best fwd3 anyone on the wire went on to post. Nobody hits
  this; it is the denominator for how much of the available value was captured.

Both are reported same-position and pool-wide. Same-position is the honest
counterfactual -- claims fill a slot -- but the pool-wide number is what you
would have got with perfect foresight and no positional constraint, so both are
shown rather than quietly picking one.

"Beat" is strict. When the recommendation and the hot-hand pick are the same
player -- which happens, and did in the first graded week -- the model added
nothing that week and is scored as not having beaten it.

Claims whose three-week window has not resolved yet are skipped, not scored.

Run:
    python -m src.ledger
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"
CLAIMS_PATH = ROOT / "outputs" / "ledger" / "claims.csv"
GRADES_PATH = ROOT / "outputs" / "ledger" / "grades.csv"

# Only these tiers are real claims; the watch list is not a recommendation.
GRADED_TIERS = ["burn", "fallback"]


def wire_pool(panel: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return panel[
        (panel["season"] == season)
        & (panel["week"] == week)
        & panel["on_wire"]
        & panel["snap"].notna()
    ]


def benchmarks(pool: pd.DataFrame, position: str) -> dict:
    """Ceiling and hot-hand outcomes, same-position and pool-wide."""
    result: dict[str, float | str] = {}
    for scope, rows in (("pos", pool[pool["position"] == position]), ("all", pool)):
        graded = rows[rows["fwd3"].notna()]
        if graded.empty:
            result[f"ceiling_{scope}"] = np.nan
            result[f"hot_hand_{scope}"] = np.nan
            result[f"hot_hand_name_{scope}"] = ""
            continue
        result[f"ceiling_{scope}"] = float(graded["fwd3"].max())
        hot = graded.loc[graded["pts"].idxmax()]
        result[f"hot_hand_{scope}"] = float(hot["fwd3"])
        result[f"hot_hand_name_{scope}"] = str(hot["player_display_name"])
    return result


def grade() -> pd.DataFrame:
    if not CLAIMS_PATH.exists():
        raise SystemExit(
            f"{CLAIMS_PATH} not found -- generate a report first "
            "(`python -m src.report --season ... --week ...`)"
        )
    if not PANEL_PATH.exists():
        raise SystemExit(f"{PANEL_PATH} not found -- run `make panel` first")

    claims = pd.read_csv(CLAIMS_PATH)
    claims = claims[claims["tier"].isin(GRADED_TIERS)]
    panel = pd.read_csv(PANEL_PATH)

    rows = []
    pending = 0
    for _, claim in claims.iterrows():
        season, week = int(claim["season"]), int(claim["week"])
        pool = wire_pool(panel, season, week)
        actual = pool[pool["player_display_name"] == claim["player"]]
        if actual.empty or pd.isna(actual.iloc[0]["fwd3"]):
            pending += 1
            continue

        got = float(actual.iloc[0]["fwd3"])
        marks = benchmarks(pool, claim["position"])
        rows.append(
            {
                "season": season,
                "week": week,
                "tier": claim["tier"],
                "player": claim["player"],
                "position": claim["position"],
                "actual_fwd3": got,
                **marks,
                "beat_hot_hand_pos": got > marks["hot_hand_pos"],
                "beat_hot_hand_all": got > marks["hot_hand_all"],
                "ceiling_pct_pos": got / marks["ceiling_pos"]
                if marks["ceiling_pos"]
                else np.nan,
                "ceiling_pct_all": got / marks["ceiling_all"]
                if marks["ceiling_all"]
                else np.nan,
            }
        )

    if pending:
        print(f"{pending} claim(s) not yet resolved -- skipped, not scored")
    return pd.DataFrame(rows)


def report(grades: pd.DataFrame) -> None:
    if grades.empty:
        print("nothing to grade yet: no claim has three weeks of forward data")
        return

    print("=" * 74)
    print("CLAIM LEDGER")
    print("=" * 74)
    view = grades[[
        "season", "week", "tier", "player", "position", "actual_fwd3",
        "hot_hand_pos", "ceiling_pos", "beat_hot_hand_pos", "ceiling_pct_pos",
    ]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    print()
    print("-" * 74)
    for scope, name in (("pos", "same position"), ("all", "whole wire pool")):
        beat = grades[f"beat_hot_hand_{scope}"].mean()
        captured = grades[f"ceiling_pct_{scope}"].mean()
        print(
            f"vs {name:16s}  beat the hot hand {beat:6.1%} of claims   "
            f"captured {captured:6.1%} of the ceiling"
        )

    weekly = grades[grades["tier"] == "burn"]
    if not weekly.empty:
        top = weekly.groupby(["season", "week"]).first()
        print(
            f"\ntop claim only, per week ({len(top)} week(s)):  "
            f"beat the hot hand {top['beat_hot_hand_pos'].mean():.1%} of weeks   "
            f"captured {top['ceiling_pct_pos'].mean():.1%} of the ceiling"
        )

    print("-" * 74)
    if len(grades) < 20:
        print(
            f"NOTE: {len(grades)} graded claim(s). Far too few to conclude anything."
        )
        print("A full season is roughly 26 claims; judge the workflow then, not now.")


def main(argv: list[str] | None = None) -> int:
    grades = grade()
    if not grades.empty:
        GRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        grades.to_csv(GRADES_PATH, index=False, float_format="%.4f")
    report(grades)
    if not grades.empty:
        print(f"\nwrote {GRADES_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

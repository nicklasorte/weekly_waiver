"""Log a comparison-arm pick to the ledger in one command.

The three-arm comparison only works if a pick is written down *before* its
outcome is known, and the cost of writing it down decides whether that actually
happens. Hand-editing a CSV on a phone on a Tuesday morning does not happen; a
line you can type into a terminal does.

    make log-claim ARM=prompt PLAYERS="Rashee Rice, Ty Johnson, Cade Otton"

Order is rank order: first name is rank 1. Season and week are resolved from the
schedule the same way `make weekly` resolves them -- never from calendar
arithmetic, which is wrong several times a season and wrong silently. Positions
are looked up in the panel, or given inline as `Name:WR` when the panel is not
built or the name is ambiguous.

The `naive` arm is not loggable here. It is derived from the panel by
`src.ledger.naive_picks`, and a hand-entered version of it could drift from the
definition it is supposed to be a fixed benchmark for.

Contamination: pass `CONTAMINATED=1` when logging a `prompt` pick made after the
candidate table was opened. It marks the week and drops it from the paired
analysis, which is the honest outcome and a much better one than a clean-looking
row that is not. The command warns on its own when a prompt pick is being logged
for a week whose candidate table already exists on disk, because at that point
the file cannot tell the two cases apart.

Run:
    python -m src.log_claim --arm prompt --players "Rashee Rice, Ty Johnson"
    python -m src.log_claim --arm repo --players "Cade Otton:TE" --season 2025 --week 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "outputs" / "ledger" / "claims.csv"
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"

CLAIM_COLUMNS = [
    "season", "week", "tier", "action", "player", "position", "dropped",
    "rationale", "arm", "rank_within_arm", "logged_at", "contaminated",
]

LOGGABLE_ARMS = ["prompt", "repo"]
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


def now_stamp() -> str:
    """UTC, seconds, ISO 8601. The ordering audit is only as good as this."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_players(raw: str) -> list[tuple[str, str | None]]:
    """`"A, B:WR, C"` -> [("A", None), ("B", "WR"), ("C", None)], in rank order."""
    picks = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, _, position = chunk.partition(":")
            position = position.strip().upper()
            if position not in VALID_POSITIONS:
                raise SystemExit(
                    f"'{position}' is not a position this repo models "
                    f"({', '.join(sorted(VALID_POSITIONS))})"
                )
            picks.append((name.strip(), position))
        else:
            picks.append((chunk, None))
    if not picks:
        raise SystemExit("no players given")
    return picks


def lookup_positions(
    picks: list[tuple[str, str | None]], season: int, week: int
) -> list[tuple[str, str]]:
    """Fill in missing positions from the panel; fail loudly rather than guess.

    A wrong position silently corrupts the same-position benchmarks, so an
    unresolvable name stops the whole command instead of logging some of the
    picks and leaving the arm short by one.
    """
    missing = [name for name, position in picks if position is None]
    resolved: dict[str, str] = {}
    if missing:
        if not PANEL_PATH.exists():
            raise SystemExit(
                f"{PANEL_PATH.relative_to(ROOT)} not found, so positions cannot be "
                "looked up -- run `make panel`, or give them inline as "
                f"'{missing[0]}:WR'"
            )
        panel = pd.read_csv(
            PANEL_PATH, usecols=["season", "week", "player_display_name", "position"]
        )
        # Season-wide, not week W. A player who sat out week W has no week-W row,
        # and he is exactly the kind of name an arm recommends -- the position is
        # the same either way, so refusing to look it up would only make the
        # command annoying in the case it most needs to work.
        rows = panel[panel["season"] == season]
        for name in missing:
            hits = rows[rows["player_display_name"] == name]["position"].unique()
            if len(hits) == 1:
                resolved[name] = str(hits[0])
            elif len(hits) == 0:
                raise SystemExit(
                    f"'{name}' has no {season} row in the panel at all. Check the "
                    f"spelling against the panel, or give the position inline as "
                    f"'{name}:WR' to log it anyway -- but a name the panel does not "
                    "know cannot be graded either."
                )
            else:
                raise SystemExit(
                    f"'{name}' is ambiguous in {season} ({', '.join(hits)}) "
                    f"-- give the position inline as '{name}:{hits[0]}'"
                )
    return [(name, position or resolved[name]) for name, position in picks]


def resolve_week(season: int | None, week: int | None) -> tuple[int, int]:
    if season is not None and week is not None:
        return season, week
    from src.weekly import current_season_week, current_week, load_games

    games = load_games()
    if season is None and week is None:
        return current_season_week(games=games)
    if week is None:
        resolved = current_week(season, games=games)
        if resolved is None:
            raise SystemExit(f"{season} has no completed regular-season week yet")
        return season, resolved
    raise SystemExit("--week without --season is ambiguous; give both or neither")


def candidate_table_exists(season: int, week: int) -> bool:
    from src.weekly import WEEKLY_DIR

    return (WEEKLY_DIR / str(season) / f"wk{week:02d}.csv").exists()


def existing_keys() -> set[tuple[int, int, str, str]]:
    """(season, week, arm, player) already in the ledger.

    Keyed on the arm too: the same player being the prompt arm's pick and the
    repo arm's pick in one week is a real and interesting outcome, not a
    duplicate row to swallow.
    """
    if not CLAIMS_PATH.exists():
        return set()
    previous = pd.read_csv(CLAIMS_PATH)
    if "arm" not in previous.columns:
        previous["arm"] = ""
    return {
        (int(s), int(w), str(a or "").strip().lower(), str(p))
        for s, w, a, p in zip(
            previous["season"], previous["week"], previous["arm"], previous["player"]
        )
    }


def append(rows: list[dict]) -> int:
    CLAIMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_keys()
    fresh = [
        r for r in rows
        if (r["season"], r["week"], r["arm"], r["player"]) not in seen
    ]
    write_header = not CLAIMS_PATH.exists()
    with CLAIMS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLAIM_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(fresh)
    return len(fresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=LOGGABLE_ARMS)
    parser.add_argument(
        "--players",
        required=True,
        help='comma-separated, in rank order: "Rashee Rice, Ty Johnson:RB"',
    )
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--why", default="", help="one line of rationale, optional")
    parser.add_argument(
        "--contaminated",
        action="store_true",
        help="this prompt pick was made after the candidate table was opened",
    )
    args = parser.parse_args(argv)

    season, week = resolve_week(args.season, args.week)
    picks = lookup_positions(parse_players(args.players), season, week)

    stamp = now_stamp()
    rows = [
        {
            "season": season,
            "week": week,
            # Every logged arm pick is a claim, and the top three are all
            # "would have burned it" -- the tier column stays for the older
            # grader, which keys off it.
            "tier": "burn",
            "action": "ADD",
            "player": name,
            "position": position,
            "dropped": "",
            "rationale": args.why,
            "arm": args.arm,
            "rank_within_arm": rank,
            "logged_at": stamp,
            "contaminated": "true" if args.contaminated else "",
        }
        for rank, (name, position) in enumerate(picks, start=1)
    ]

    added = append(rows)
    for row in rows:
        print(f"  #{row['rank_within_arm']} {row['player']} ({row['position']})")
    print(
        f"logged {added} of {len(rows)} {args.arm} pick(s) for {season} week {week} "
        f"at {stamp}"
    )
    if added < len(rows):
        print(f"{len(rows) - added} already in the ledger for this arm and week; skipped")

    if args.contaminated:
        print(
            "marked contaminated -- this week is excluded from the paired "
            "prompt-vs-repo comparison"
        )
    elif args.arm == "prompt" and candidate_table_exists(season, week):
        print(
            f"\nNOTE: outputs/weekly/{season}/wk{week:02d}.csv already exists, so the "
            "ledger cannot show this pick preceded it.\nIf you had already looked at "
            "the candidate table, re-log with CONTAMINATED=1 -- an excluded week is "
            "worth more than a clean-looking one that is not."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

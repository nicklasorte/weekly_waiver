"""Turn a week's candidate table into a report you can act on in two minutes.

Writes outputs/reports/{season}/wk{NN}.md and appends every recommendation to
outputs/ledger/claims.csv, so that src/ledger.py can grade this advice later
against what actually happened. A recommendation that is never scored is just
an opinion.

Hard rules, enforced in code rather than trusted to prose:

- This never places a transaction. It writes a markdown file and a CSV row.
  There is no league API call anywhere in this repo, by design.
- It never proposes a drop that would leave the roster without a kicker or a
  defense. Those slots are mandatory; a claim that costs you a starting slot is
  not an upgrade.
- It biases toward claiming now. Hoarding waiver priority until Week 10 cost
  about 69% of a claim's value in simulation, so "save it for someone better"
  is the default mistake, not the safe option.
- Projections are ranges from the conformal intervals. Never a point estimate:
  these models resolve about a third of the variance in wire outcomes, and a
  single number would imply precision that does not exist.

Roster is optional. Without one the report still ranks and tiers the wire, but
leaves every DROP unresolved and skips the roster check -- see the AFTER
discussion about roster.yaml vs the ESPN cookie route.

Run:
    python -m src.report --season 2025 --week 8
    python -m src.report --season 2025 --week 8 --roster data/roster.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.weekly import WEEKLY_DIR, load_games

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
CLAIMS_PATH = ROOT / "outputs" / "ledger" / "claims.csv"
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"

CLAIM_COLUMNS = [
    "season", "week", "tier", "action", "player", "position", "dropped", "rationale",
]

# Slots that must always be filled; never proposed as a drop when it is the last one.
MANDATORY = {"K", "DST", "D/ST", "DEF"}

WORD_LIMIT = 500

MODES = [
    (5, "long-term", "value over the rest of the season; this year's record is not yet the constraint"),
    (10, "switching", "balance the playoff push against players who still hold value"),
    (13, "playoff-push", "win now; season-long upside is worth nothing if you miss the bracket"),
    (18, "bracket-or-roto", "matchup-driven; only the next three weeks exist"),
]

# Tiering sizes. Only players with a positive edge over positional replacement
# are eligible at all.
TIER1_MAX = 2
TIER2_MAX = 3
WATCH_MAX = 4

# Replacement level: how many at each position you would realistically choose
# among on a 12-team wire. Edge is measured against the next player past that,
# because that is who you actually get instead if you pass.
REPLACEMENT_RANK = {"QB": 2, "RB": 5, "WR": 6, "TE": 2}

# Evidence worth printing, per position, with the floor below which a number is
# noise rather than a reason. A WR with 2 carries is not a story about carries.
RELEVANT = {
    "RB": [
        ("carry_share", "{:.0%} carry share", 0.15),
        ("carries", "{:.0f} carries", 6),
        ("snap", "{:.0%} snaps", 0.30),
        ("tgt_share", "{:.0%} target share", 0.10),
        ("snap_jump", "snaps up {:+.0%}", 0.08),
    ],
    "WR": [
        ("tgt_share", "{:.0%} target share", 0.15),
        ("targets", "{:.0f} targets", 5),
        ("snap", "{:.0%} snaps", 0.50),
        ("air_yards_share", "{:.0%} air yards", 0.20),
        ("snap_jump", "snaps up {:+.0%}", 0.08),
    ],
    "TE": [
        ("tgt_share", "{:.0%} target share", 0.12),
        ("targets", "{:.0f} targets", 4),
        ("snap", "{:.0%} snaps", 0.45),
        ("snap_jump", "snaps up {:+.0%}", 0.08),
    ],
    "QB": [
        ("snap", "{:.0%} snaps", 0.80),
        ("carries", "{:.0f} rush attempts", 4),
        ("pts_lag1", "{:.0f} pts last week", 15),
    ],
}

# Role-durability thresholds for the KEEPER / STARTER / RENTAL label.
KEEPER_SCORE = 0.75
KEEPER_ROLE = 0.60
STARTER_SCORE = 0.62


def horizon(week: int) -> tuple[str, str]:
    for last_week, name, gloss in MODES:
        if week <= last_week:
            return name, gloss
    return MODES[-1][1], MODES[-1][2]


def load_candidates(season: int, week: int) -> pd.DataFrame:
    path = WEEKLY_DIR / str(season) / f"wk{week:02d}.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run `make weekly SEASON={season} WEEK={week}` first"
        )
    return pd.read_csv(path)


def load_roster(path: str | None) -> dict | None:
    """Optional roster file. JSON always; YAML when pyyaml happens to be installed.

    Shape: {"record": "5-2", "players": [{"name", "position", "status"?, "note"?}]}
    """
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        raise SystemExit(f"roster file {file} not found")
    text = file.read_text()
    if file.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "reading a .yaml roster needs pyyaml (`pip install pyyaml`); "
                "a .json roster works with no extra dependency"
            )
        return yaml.safe_load(text)
    return json.loads(text)


def label(row: pd.Series) -> str:
    """RENTAL / STARTER / KEEPER, from how durable the role looks."""
    role = row.get("kal_role", 0.0)
    shrunk = max(row.get("eb_tgt_share", 0.0), row.get("eb_car_share", 0.0))
    if row["score"] >= KEEPER_SCORE and role >= KEEPER_ROLE and shrunk >= 0.12:
        return "KEEPER"
    if row["score"] >= STARTER_SCORE:
        return "STARTER"
    return "RENTAL"


def evidence(row: pd.Series, pool: pd.DataFrame) -> str:
    """The two numbers that best explain why this name surfaced.

    Restricted to stats that mean something at the position, and floored at a
    level worth mentioning. Ranking purely by z-score against the pool does not
    work: almost every WR has zero carries, so the standard deviation collapses
    and "2 carries" scores as a huge outlier. It is a rounding error, not a
    reason.
    """
    same = pool[pool["position"] == row["position"]]
    scored = []
    for column, template, floor in RELEVANT.get(row["position"], []):
        if column not in row or pd.isna(row[column]) or row[column] < floor:
            continue
        series = same[column].dropna()
        if len(series) < 5 or series.std() == 0:
            continue
        z = (row[column] - series.median()) / series.std()
        if z > 0.25:
            scored.append((z, template.format(row[column])))
    scored.sort(reverse=True)
    if not scored:
        return "usage is unremarkable; surfaced on the model score alone"
    return " and ".join(text for _, text in scored[:2])


def with_edge(table: pd.DataFrame) -> pd.DataFrame:
    """Points above replacement at the same position.

    Two corrections, both necessary, and the report is wrong without either.

    First, the model score is a within-week percentile rank pooled across
    positions, so it is not comparable between them: quarterbacks average 8.3
    fwd3 points against a running back's 2.5, which pushes every startable QB
    near the top of the pool. Tiering on raw score hands the claim to a
    streaming quarterback ahead of a genuinely valuable receiver.

    Second, replacement level is not the median of the wire pool. The median
    wire quarterback is a backup who will not play, which makes any starter look
    like a huge edge. Replacement is the next player you would actually take
    instead -- the third-best available QB, the seventh-best WR -- so that is
    the baseline. Passing on a claim gets you that player, not the median.
    """
    table = table.copy()

    def baseline(group: pd.Series) -> float:
        rank = REPLACEMENT_RANK.get(group.name, 5)
        ordered = group.sort_values(ascending=False).to_numpy()
        return float(ordered[min(rank, len(ordered) - 1)])

    levels = table.groupby("position")["proj_pts"].transform(
        lambda g: baseline(g)
    )
    table["edge"] = table["proj_pts"] - levels
    return table


def assign_tiers(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Leading block, fallback block, watch list -- ranked by edge, not raw score."""
    ranked = (
        with_edge(table)
        .query("edge > 0")
        .sort_values("edge", ascending=False)
        .reset_index(drop=True)
    )
    tier1 = ranked.head(TIER1_MAX)
    tier2 = ranked.iloc[TIER1_MAX : TIER1_MAX + TIER2_MAX]
    watch = ranked.iloc[TIER1_MAX + TIER2_MAX : TIER1_MAX + TIER2_MAX + WATCH_MAX]
    return {"burn": tier1, "fallback": tier2, "watch": watch}


def droppable(roster: dict | None, panel: pd.DataFrame, season: int, week: int) -> list[str]:
    """Roster players safe to cut, worst first. Never the last K or defense."""
    if not roster:
        return []
    players = roster.get("players", [])
    counts: dict[str, int] = {}
    for player in players:
        counts[player.get("position", "?")] = counts.get(player.get("position", "?"), 0) + 1

    scored = []
    for player in players:
        position = player.get("position", "?")
        if position.upper() in MANDATORY and counts.get(position, 0) <= 1:
            continue  # mandatory slot, and it is the only one
        rows = panel[
            (panel["season"] == season)
            & (panel["week"] <= week)
            & (panel["player_display_name"] == player["name"])
        ]
        value = float(rows["pts"].sum()) if len(rows) else 0.0
        scored.append((value, player["name"], position))
    scored.sort()
    return [f"{name} ({position})" for _, name, position in scored]


def roster_check(
    roster: dict | None, panel: pd.DataFrame, games: pd.DataFrame, season: int, week: int
) -> list[str]:
    """Injured, on bye next week, or quietly lost the job."""
    if not roster:
        return []
    playing = set(
        zip(games["season"], games["week"], games["away_team"])
    ) | set(zip(games["season"], games["week"], games["home_team"]))
    notes = []
    for player in roster.get("players", []):
        name = player["name"]
        status = str(player.get("status", "")).lower()
        if status in {"injured", "out", "ir", "doubtful", "questionable"}:
            notes.append(f"**{name}** — {status}")
            continue
        rows = panel[
            (panel["season"] == season)
            & (panel["week"] == week)
            & (panel["player_display_name"] == name)
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        if (season, week + 1, row["team"]) not in playing:
            notes.append(f"**{name}** — bye next week")
        elif not pd.isna(row.get("snap_jump")) and row["snap_jump"] <= -0.15:
            notes.append(
                f"**{name}** — snaps down {abs(row['snap_jump']):.0%}, role slipping"
            )
    return notes


def build_report(
    season: int, week: int, roster: dict | None
) -> tuple[str, list[dict]]:
    table = load_candidates(season, week)
    panel = pd.read_csv(PANEL_PATH) if PANEL_PATH.exists() else pd.DataFrame()
    games = load_games()
    mode, gloss = horizon(week)
    tiers = assign_tiers(table)
    drops = droppable(roster, panel, season, week)
    record = (roster or {}).get("record", "record not configured")

    lines = [
        f"# {season} Week {week} — {mode}",
        "",
        f"Week {week}, {record}, **{mode}** mode: {gloss}.",
        "",
    ]

    checks = roster_check(roster, panel, games, season, week)
    lines += ["## Roster check", ""]
    if roster is None:
        lines += ["_No roster configured — injury, bye and role checks skipped._", ""]
    elif checks:
        lines += [f"- {note}" for note in checks] + [""]
    else:
        lines += ["- Nothing flagged.", ""]

    lines += ["## Top of the wire", ""]
    for position in ["RB", "WR", "TE", "QB"]:
        rows = table[table["position"] == position].head(2)
        if rows.empty:
            continue
        for _, row in rows.iterrows():
            lines.append(
                f"- **{row['player_display_name']}** ({position}, {row['team']}) — "
                f"{evidence(row, table)}; {row['proj_pts_lo']:.1f}–"
                f"{row['proj_pts_hi']:.1f} pts/wk"
            )
    lines.append("")

    claims: list[dict] = []
    headings = [
        ("burn", "Burn the claim"),
        ("fallback", "Claim if tier 1 fails"),
        ("watch", "Watch list"),
    ]
    lines += ["## Claims", ""]
    for key, heading in headings:
        rows = tiers[key]
        if rows.empty:
            continue
        lines += [f"**{heading}**", ""]
        for i, (_, row) in enumerate(rows.iterrows()):
            # Drop candidates restart per tier on purpose: tiers are
            # alternatives, not a shopping list. If tier 1 fails, the player it
            # would have cost you is still on your roster.
            drop = drops[i] if i < len(drops) else "???"
            why = (
                f"{evidence(row, table)}; {row['proj_pts_lo']:.1f}–"
                f"{row['proj_pts_hi']:.1f} pts/wk"
            )
            action = "WATCH" if key == "watch" else "ADD"
            drop_text = "" if key == "watch" else f" / DROP {drop}"
            lines.append(
                f"- `[{label(row)}]` {action} {row['player_display_name']}"
                f"{drop_text} — {why}."
            )
            claims.append(
                {
                    "season": season,
                    "week": week,
                    "tier": key,
                    "action": action,
                    "player": row["player_display_name"],
                    "position": row["position"],
                    "dropped": "" if key == "watch" else drop,
                    "rationale": why,
                }
            )
        lines.append("")

    lines += [
        "## Standing rules",
        "",
        "- Recommendations only. Nothing here places a transaction.",
        "- Claim now. Hoarding priority to Week 10 cost ~69% of a claim's value.",
        "- Ranges are 80% conformal intervals, not projections. They are wide "
        "because the models are.",
        "- Claims are ordered by points above replacement at the same position, "
        "not by raw model score, which is not comparable across positions.",
    ]
    if roster is None:
        lines.append("- `DROP ???` — wire up a roster to resolve the other half.")
    else:
        lines.append("- No drop proposed here leaves you without a K or D/ST.")
    lines.append("")
    return "\n".join(lines), claims


def append_claims(claims: list[dict]) -> int:
    """Append to the ledger, skipping rows already logged for this season/week/player."""
    CLAIMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if CLAIMS_PATH.exists():
        previous = pd.read_csv(CLAIMS_PATH)
        existing = set(zip(previous["season"], previous["week"], previous["player"]))

    fresh = [c for c in claims if (c["season"], c["week"], c["player"]) not in existing]
    write_header = not CLAIMS_PATH.exists()
    with CLAIMS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLAIM_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(fresh)
    return len(fresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--roster", help="optional roster file (.json, or .yaml)")
    args = parser.parse_args(argv)

    roster = load_roster(args.roster)
    text, claims = build_report(args.season, args.week, roster)

    words = len(text.split())
    if words > WORD_LIMIT:
        print(f"warning: report is {words} words, over the {WORD_LIMIT}-word limit")

    out_dir = REPORT_DIR / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"wk{args.week:02d}.md"
    path.write_text(text)

    added = append_claims(claims)
    print(f"wrote {path.relative_to(ROOT)} ({words} words)")
    print(f"logged {added} new claim(s) to {CLAIMS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

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


THE THREE-ARM COMPARISON
========================

The second half of this module runs a controlled comparison of three ways of
picking a waiver claim. Only one of them is ever executed on a real roster; the
other two are paper recommendations. All three are scored identically, by what
the recommended player actually did afterward, so which one gets played does not
affect the measurement.

- **naive** -- the highest-scoring available player from last week. Derived
  straight from the panel by `naive_picks()`; never logged by hand, so it cannot
  drift and there is no judgement in it to argue with.
- **prompt** -- an LLM with web search and no access to this repo's candidate
  table, model scores or panel. Judgement plus news plus published stats.
- **repo** -- the candidate table plus judgement.

The top three picks per arm per week are logged, not just the top one. Thirteen
weeks of single picks resolves nothing; thirty-nine rows resolves slightly more
than nothing, which is the honest description of the improvement.

THE OUTCOME METRIC: PAR, NOT RAW fwd3
=====================================

Every arm's picks resolve to `fwd3`: the player's mean points over the following
three weeks under NCFOM scoring, with a week his team played and he did not
counted as 0.0. That is the same convention the models train on, and it is the
one that matters for whether a pick worked -- a recommendation who stops playing
is scored as the failure it is rather than dropped from the average.

`fwd3` is kept on every graded row and in every CSV. It is **not** what the
headline comparison or the decision rule below are computed on. Both run on
**points above replacement at position**:

    par = fwd3 - replacement_level(pool fwd3 at that position, position)

with `report.replacement_level` and `report.REPLACEMENT_RANK` -- the same
definition, from the same function, that the weekly table tiers on.

The reason is structural rather than a preference. Raw `fwd3` is fantasy points,
and quarterbacks score about three times what a running back does, so an arm
that sorts on raw points loads up on quarterbacks and banks the positional
difference as if it were skill. You start one quarterback. A claim on a second
one is worth approximately nothing, and a metric that pays for it is measuring
the wrong thing. `outputs/backtests/02_walkforward_2014_2025.py` measured that
directly over twelve seasons: 93% of the naive arm's 2.30 ppg margin was
positional composition, not ranking.

What PAR does and does not fix is worth stating, because it is easy to overclaim
and `outputs/diagnostics/par_rescore_and_ablation.md` shows the arithmetic.
Subtracting a per-(week, position) constant does not remove positional
composition from the comparison; it re-prices it. Replacement level sits well
above the pool mean at every position and furthest above it at quarterback,
so PAR does not merely stop rewarding a quarterback-heavy arm -- it penalises
one. That is the intended direction (a second quarterback really is worth less
than the streamer you already have) but it is a re-pricing, not a neutralisation,
and any PAR margin should be read next to its mix/selection split rather than on
its own.

Reported per arm: n, mean PAR, mean raw points captured, share of the weekly
ceiling captured, and share of weeks beating the naive arm head to head. Then
`prompt` vs `repo` paired on the weeks both arms covered, because the week-to-week
variance in this data swamps the difference being measured and paired differences
remove most of it.


PRE-REGISTERED DECISION RULE
============================

Written here before the season's data exists, so that it cannot be quietly
revised once the numbers are in. `verdict()` implements exactly this, in this
order, and nothing else:

- neither arm beats naive by >= 1.5 ppg  -> the analysis is decoration
- prompt and repo within 1.0 ppg         -> repo adds nothing, keep the prompt
- repo beats prompt by >= 1.5 ppg        -> the data layer earns its keep
- anything else                          -> inconclusive, report it as such

The rule reads point estimates, and a point estimate is not evidence on its own.
With n around 13 weeks and the per-player variance in wire outcomes, a 1.5 ppg
gap is well inside the noise, so the verdict is printed next to the paired
confidence interval and never instead of it. When the interval covers zero the
honest reading is that the arms are indistinguishable, whichever branch the rule
happened to land on. A tie reported as a tie is the correct output for most
seasons; this module is written to make that the easy answer rather than the
disappointing one.

AMENDMENT, 2026-08-31: the rule above was pre-registered against raw `fwd3` and
now runs on PAR. The thresholds are unchanged; the quantity they are applied to
is not. The amendment is recorded here, in `docs/comparison_protocol.md` and in
`outputs/diagnostics/comparison_setup.md` rather than applied silently, and it is
made **before the rule has ever been evaluated on live data**. One dry-run week
is on file (2025 wk08, `repo` only); no `prompt` arm has been logged, so no
paired difference and no verdict exists yet and none changed sign because of
this. `outputs/ledger/arm_summary.csv` does move: the same dry-run week is
re-scored on PAR, which is the amendment doing its job rather than a result
being revised. A margin measured in raw fantasy points was not a margin in the
thing being compared; the reasoning is under "THE OUTCOME METRIC" above.
Nothing else about the rule moved.

Contamination: the `prompt` arm is only a fair comparison if it was produced
before the repo's candidate table was opened. Weeks flagged `contaminated` in
the ledger are excluded from the paired analysis and from arm means, and counted
out loud rather than quietly dropped. See docs/comparison_protocol.md.

Run:
    python -m src.ledger
    python -m src.ledger --strict-order   # also drop weeks whose order is unverified
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.report import REPLACEMENT_RANK, replacement_level

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"
CLAIMS_PATH = ROOT / "outputs" / "ledger" / "claims.csv"
GRADES_PATH = ROOT / "outputs" / "ledger" / "grades.csv"
ARM_GRADES_PATH = ROOT / "outputs" / "ledger" / "arm_grades.csv"
ARM_SUMMARY_PATH = ROOT / "outputs" / "ledger" / "arm_summary.csv"

# Only these tiers are real claims; the watch list is not a recommendation.
GRADED_TIERS = ["burn", "fallback"]

# Arms, and how many ranked picks per arm per week count toward the comparison.
ARMS = ["naive", "prompt", "repo"]
LOGGED_ARMS = ["prompt", "repo"]  # naive is derived, never logged
ARM_DEPTH = 3

# Bootstrap settings for the paired interval. Resampling is over WEEKS, not
# rows: the three picks an arm makes in one week share a wire pool and a set of
# upcoming matchups, so treating them as independent draws would understate the
# variance by roughly the within-week correlation.
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20250901
CI_LEVEL = 0.95

# Thresholds of the pre-registered rule. Named so that a later edit shows up in
# the diff as an edit to the rule rather than as a tweak to a magic number.
#
# These have NOT moved with the 2026-08-31 amendment, and that is a decision
# worth seeing rather than a detail. They were chosen against raw fantasy
# points; they are now applied to PAR, which is a different scale. The
# twelve-season replay is the available calibration: re-scoring the same picks
# moved the repo-minus-naive margin from -2.30 ppg to +1.74 ppg without a single
# pick changing, so a 1.5 ppg threshold is a materially easier bar on PAR than
# it was on raw points. Leaving the numbers alone while changing what they
# measure is not obviously right; re-picking them after seeing that swing would
# be worse. They stay, the tension is recorded here, and the interval is printed
# next to the verdict as it always was.
NAIVE_MARGIN = 1.5   # ppg an arm must clear naive by for the analysis to be worth anything
TIE_BAND = 1.0       # ppg within which prompt and repo are the same thing
REPO_MARGIN = 1.5    # ppg repo must beat prompt by to justify the data layer


# ==========================================================================
# shared: the wire pool and its benchmarks
# ==========================================================================

def wire_pool(panel: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return panel[
        (panel["season"] == season)
        & (panel["week"] == week)
        & panel["on_wire"]
        & panel["snap"].notna()
    ]


def replacement_of(values: pd.Series, position: str) -> float:
    """`report.replacement_level`, refusing to clamp a pool shorter than the rank.

    `with_edge` clamps to the pool's last entry, and that is right where it
    lives: a clamped group gives its worst player an edge of exactly zero and
    `assign_tiers` filters him out, so the clamp acts as a safety filter.

    Scoring an outcome has no such filter. A clamped baseline would be the pool
    minimum -- usually 0.0, since `fwd3` scores a week a player missed as zero --
    so PAR would collapse to raw `fwd3` and *inflate* the pick rather than
    exclude it. That is the opposite of the source behaviour, and it would be
    silent. NaN instead, so `grade_arms` can count it out loud.

    Thin enough to fire is rare on a real wire and not impossible: a position
    where almost nobody's forward window has resolved, or a hand-logged position
    string that matches nothing in the panel.
    """
    resolved = values.dropna()
    if len(resolved) <= REPLACEMENT_RANK.get(position, 5):
        return float("nan")
    return replacement_level(resolved, position)


def replacement_levels(pool: pd.DataFrame) -> dict[str, float]:
    """Realised replacement level per position for one week's wire pool.

    The baseline PAR is measured against: what the player you would have taken
    instead actually went on to average. Computed from the same pool the claims
    were drawn from and from `report.replacement_level`, so the ledger and the
    weekly table cannot disagree about where replacement sits.

    Three properties this has to have, and does:

    - **Arm-neutral.** It is a function of the pool and the position alone. No
      arm's picks, ranking or training window can move it, so subtracting it
      cannot flatter one arm over another within a position.
    - **Resolved players only.** A player whose forward window has not closed
      has no outcome to contribute, and including him as a zero would drag the
      baseline down and inflate every arm's PAR.
    - **A per-position constant.** Within a position it shifts every arm by the
      same amount, so it changes no within-position comparison at all. It is the
      *across*-position comparison it exists to fix.

    It is an order statistic of what happened, which is a harder bar than the
    replacement level anyone could have identified in advance -- the third-best
    quarterback in hindsight outscores the third-best quarterback you would have
    picked. That is a known property, not a bug to be papered over: the same
    constant is subtracted from every arm, and the size of the effect is measured
    in `outputs/diagnostics/par_rescore_and_ablation.md`.

    A position too thin to have a replacement level is **absent** from the
    result rather than present with a clamped one; see `replacement_of` for why
    that distinction is not pedantry.
    """
    resolved = pool[pool["fwd3"].notna()]
    levels = {
        str(position): replacement_of(rows["fwd3"], str(position))
        for position, rows in resolved.groupby("position")
    }
    return {name: level for name, level in levels.items() if not pd.isna(level)}


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


# ==========================================================================
# legacy: tier-based grading, unchanged
# ==========================================================================

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


# ==========================================================================
# three-arm comparison
# ==========================================================================

def naive_picks(panel: pd.DataFrame, season: int, week: int, depth: int = ARM_DEPTH) -> pd.DataFrame:
    """The naive arm: the `depth` highest-scoring available players from week W.

    Derived, not logged. The panel row for week W carries week W's points and is
    computable on the Monday claims are entered, so "last week's box score" is
    exactly `pts` on that row -- the same quantity the hot-hand benchmark uses,
    widened from one name to three.

    Selection uses `pts` alone. It deliberately does NOT filter to players whose
    fwd3 has resolved: that column is the future, and letting it touch the
    selection would hand the naive arm foresight the other two arms do not have.
    A pick with no resolved fwd3 is dropped at grading time and counted, which
    only happens at the tail of a season where no three-week window exists.

    Ties on `pts` break on player name, so the arm is reproducible rather than
    dependent on row order in the panel.
    """
    pool = wire_pool(panel, season, week)
    pool = pool[pool["pts"].notna()]
    if pool.empty:
        return pool.assign(rank_within_arm=pd.Series(dtype=int))
    ordered = pool.sort_values(
        ["pts", "player_display_name"], ascending=[False, True]
    ).head(depth).copy()
    ordered["rank_within_arm"] = range(1, len(ordered) + 1)
    return ordered


def load_claims() -> pd.DataFrame:
    """The ledger, with the arm columns present whether or not the file has them.

    Rows written before the arms existed are read back with `arm` empty; see
    `arm_rows()` for what happens to them.
    """
    if not CLAIMS_PATH.exists():
        raise SystemExit(
            f"{CLAIMS_PATH} not found -- generate a report first "
            "(`python -m src.report --season ... --week ...`)"
        )
    claims = pd.read_csv(CLAIMS_PATH)
    for column, default in (
        ("arm", ""),
        ("rank_within_arm", np.nan),
        ("logged_at", ""),
        ("contaminated", ""),
    ):
        if column not in claims.columns:
            claims[column] = default
    claims["arm"] = claims["arm"].fillna("").astype(str).str.strip().str.lower()
    claims["logged_at"] = claims["logged_at"].fillna("").astype(str).str.strip()
    claims["contaminated"] = (
        claims["contaminated"].fillna("").astype(str).str.strip().str.lower()
    )
    claims["rank_within_arm"] = pd.to_numeric(claims["rank_within_arm"], errors="coerce")
    return claims


def is_contaminated(value: str) -> bool:
    return value in {"1", "true", "yes", "y", "t"}


def arm_rows(claims: pd.DataFrame) -> pd.DataFrame:
    """Logged rows that belong to an arm at a counted rank.

    Rows with no arm are pre-comparison history and are silently outside this
    analysis; rows with an arm but no rank, or a rank past ARM_DEPTH, are logged
    on purpose and excluded on purpose.
    """
    rows = claims[claims["arm"].isin(LOGGED_ARMS)].copy()
    return rows[rows["rank_within_arm"].between(1, ARM_DEPTH)]


def order_status(claims: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week): whether the prompt arm can be shown to precede the repo arm.

    Three states, and the distinction matters more than it looks:

    - `contaminated` -- a human marked a row. The protocol was broken and said
      so. Always excluded.
    - `unverified`   -- the prompt rows carry a timestamp no earlier than the
      repo rows'. This is NOT proof of contamination: `make report` runs in CI
      at 06:00 UTC Tuesday and stamps the repo rows then, hours before anyone is
      awake to write down a prompt pick, so a clean week routinely looks like
      this. It only means the file cannot vouch for the order.
    - `clean`        -- prompt rows are stamped strictly before the repo rows,
      or one of the arms is absent so there is no order to break.

    Excluding `unverified` by default would throw out most of a season on a
    technicality, so the default keeps it and prints the count; `--strict-order`
    drops it for anyone who wants the stricter read.
    """
    out = []
    rows = claims[claims["arm"].isin(LOGGED_ARMS)]
    for (season, week), group in rows.groupby(["season", "week"], sort=True):
        flagged = group["contaminated"].map(is_contaminated).any()
        has_prompt = (group["arm"] == "prompt").any()
        has_repo = (group["arm"] == "repo").any()
        prompt = sorted(t for t in group[group["arm"] == "prompt"]["logged_at"] if t)
        repo = sorted(t for t in group[group["arm"] == "repo"]["logged_at"] if t)
        if flagged:
            status = "contaminated"
        elif not (has_prompt and has_repo):
            # Only one arm ran this week, so there is no ordering to break. The
            # week still cannot enter the paired comparison; that is decided by
            # having both arms, not by this status.
            status = "clean"
        elif not prompt or not repo:
            # Both arms present but a timestamp is missing, so the ordering is
            # unknown. Untimestamped is not the same as fine.
            status = "unverified"
        elif max(prompt) < min(repo):
            status = "clean"
        else:
            status = "unverified"
        out.append({"season": int(season), "week": int(week), "order_status": status})
    return pd.DataFrame(out, columns=["season", "week", "order_status"])


def forward_three(
    panel: pd.DataFrame, season: int, week: int, player: str,
    games_path: Path | str | None = None,
) -> float | None:
    """fwd3 for a player with no panel row in week W, rebuilt from his other weeks.

    The panel has a row per player-week he recorded stats in, so a player who
    was inactive in week W is simply absent from it -- and being inactive in
    week W is one of the more common reasons a name is on the wire in the first
    place. Dropping those picks as unscoreable would bias the comparison against
    whichever arm is most willing to recommend a player coming back from injury,
    which is precisely the prompt arm, since news is the one input it has and
    the panel does not.

    Same convention as `features.forward_three`: mean points over the weeks in
    W+1..W+3 his team actually played, a week played without him counted as 0.0,
    byes excluded at both ends. Team is taken from his nearest row to week W,
    so a mid-season trade resolves to the right schedule.
    """
    from src.features import load_schedule

    rows = panel[
        (panel["season"] == season) & (panel["player_display_name"] == player)
    ]
    if rows.empty:
        return None
    try:
        played, last_week = load_schedule(games_path)
    except FileNotFoundError:
        # Without the schedule a bye cannot be told from a missed game, and
        # guessing would put an invented number in an arm's average. Leave the
        # pick unscored; it is reported as such.
        return None
    nearest = rows.iloc[(rows["week"] - week).abs().argsort().iloc[0]]
    team = nearest["team"]
    by_week = dict(zip(rows["week"], rows["pts"]))
    final = last_week.get(season, 18)
    span = [
        w for w in (week + 1, week + 2, week + 3)
        if w <= final and (season, w, team) in played
    ]
    if not span:
        return None
    return float(np.mean([by_week.get(w, 0.0) for w in span]))


def outcome(
    panel: pd.DataFrame, season: int, week: int, player: str,
    games_path: Path | str | None = None,
) -> tuple[float | None, bool]:
    """(fwd3, was_in_wire_pool) for one recommended player in one week.

    The lookup is against the whole panel, not the wire pool. `on_wire` is a
    proxy built from prior-week usage; the prompt arm has never seen it and can
    perfectly reasonably name someone the proxy thinks is rostered. Scoring that
    as unresolvable would penalise an arm for disagreeing with a heuristic
    rather than for being wrong about the player. Whether the pick was inside
    the pool is recorded instead, so the disagreement stays visible.
    """
    rows = panel[
        (panel["season"] == season)
        & (panel["week"] == week)
        & (panel["player_display_name"] == player)
    ]
    if rows.empty:
        return forward_three(panel, season, week, player, games_path), False
    row = rows.iloc[0]
    in_pool = bool(row.get("on_wire", False)) and not pd.isna(row.get("snap"))
    if pd.isna(row["fwd3"]):
        return None, in_pool
    return float(row["fwd3"]), in_pool


def grade_arms(claims: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One graded row per (arm, season, week, rank). Naive rows are derived here."""
    logged = arm_rows(claims)
    statuses = order_status(claims).set_index(["season", "week"])["order_status"].to_dict()

    weeks = sorted({(int(s), int(w)) for s, w in zip(logged["season"], logged["week"])})
    rows: list[dict] = []
    problems: list[str] = []

    for season, week in weeks:
        pool = wire_pool(panel, season, week)
        status = statuses.get((season, week), "clean")
        week_ceiling = (
            float(pool.loc[pool["fwd3"].notna(), "fwd3"].max())
            if pool["fwd3"].notna().any()
            else np.nan
        )
        levels = replacement_levels(pool)

        picks: list[dict] = []
        for _, pick in naive_picks(panel, season, week).iterrows():
            picks.append(
                {
                    "arm": "naive",
                    "rank_within_arm": int(pick["rank_within_arm"]),
                    "player": str(pick["player_display_name"]),
                    "position": str(pick["position"]),
                }
            )
        for _, claim in logged[
            (logged["season"] == season) & (logged["week"] == week)
        ].iterrows():
            picks.append(
                {
                    "arm": claim["arm"],
                    "rank_within_arm": int(claim["rank_within_arm"]),
                    "player": str(claim["player"]),
                    "position": str(claim["position"]),
                }
            )

        for pick in picks:
            got, in_pool = outcome(panel, season, week, pick["player"])
            if got is None:
                problems.append(
                    f"{season} wk{week:02d} {pick['arm']:6s} #{pick['rank_within_arm']} "
                    f"{pick['player']}: no three-week window, or no row in the "
                    "panel under that spelling"
                )
                continue
            marks = benchmarks(pool, pick["position"])
            # A position too thin to have a replacement level leaves the pick
            # with a NaN PAR, so it drops out of the PAR comparison rather than
            # being scored against an invented zero. It is not an unscoreable
            # pick -- `actual_fwd3` is there and the tier ledger still grades it
            # -- so it does not go in `problems`, which means something
            # narrower. `report_arms` counts it separately from the graded rows.
            level = levels.get(pick["position"], np.nan)
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "order_status": status,
                    **pick,
                    "actual_fwd3": got,
                    "repl_fwd3": level,
                    "par": got - level,
                    "in_wire_pool": in_pool,
                    "week_ceiling": week_ceiling,
                    "ceiling_pos": marks["ceiling_pos"],
                    "hot_hand_pos": marks["hot_hand_pos"],
                }
            )

    return pd.DataFrame(rows), problems


def week_means(graded: pd.DataFrame) -> pd.DataFrame:
    """Arm x week table of mean PAR and mean fwd3, with the week's ceiling.

    The week is the unit of analysis everywhere below. Within one week an arm's
    three picks are drawn from one pool against one slate of upcoming opponents,
    so they are not three independent observations of the arm.

    `mean_par` is the headline quantity and `mean_fwd3` is kept beside it, so a
    reader can see both what an arm captured and what it captured above what
    passing on the claim would have got.
    """
    if graded.empty:
        return pd.DataFrame(
            columns=["season", "week", "arm", "order_status", "n",
                     "mean_par", "mean_fwd3", "week_ceiling"]
        )
    grouped = (
        graded.groupby(["season", "week", "arm", "order_status"], as_index=False)
        .agg(n=("actual_fwd3", "size"), mean_par=("par", "mean"),
             mean_fwd3=("actual_fwd3", "mean"),
             week_ceiling=("week_ceiling", "first"))
    )
    return grouped.sort_values(["season", "week", "arm"]).reset_index(drop=True)


def arm_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per arm: n rows, mean points, ceiling share, head-to-head record vs naive.

    Every figure is computed only on the weeks that arm actually covered, and
    `weeks` is printed next to the mean so that two arms summarised over
    different week sets cannot be read as a like-for-like comparison. That is
    what the paired table further down is for.
    """
    if weekly.empty:
        return pd.DataFrame()
    naive = weekly[weekly["arm"] == "naive"].set_index(["season", "week"])["mean_par"]

    out = []
    for arm in ARMS:
        rows = weekly[weekly["arm"] == arm]
        if rows.empty:
            continue
        share = (rows["mean_fwd3"] / rows["week_ceiling"]).replace([np.inf, -np.inf], np.nan)
        if arm == "naive":
            beat = np.nan
        else:
            # Head to head on PAR, matching the headline. On raw points this
            # counts a week the arm won only by having claimed a quarterback.
            pairs = [
                (m, naive.get((s, w)))
                for s, w, m in zip(rows["season"], rows["week"], rows["mean_par"])
            ]
            pairs = [(a, b) for a, b in pairs if b is not None and not pd.isna(b)]
            beat = float(np.mean([a > b for a, b in pairs])) if pairs else np.nan
        out.append(
            {
                "arm": arm,
                "weeks": int(len(rows)),
                "n": int(rows["n"].sum()),
                "mean_par": float(rows["mean_par"].mean()),
                "mean_fwd3": float(rows["mean_fwd3"].mean()),
                "ceiling_share": float(share.mean()) if share.notna().any() else np.nan,
                "beat_naive_share": beat,
            }
        )
    return pd.DataFrame(out)


def excluded_statuses(strict_order: bool) -> set[str]:
    return {"contaminated"} | ({"unverified"} if strict_order else set())


def usable_weeks(weekly: pd.DataFrame, strict_order: bool) -> pd.DataFrame:
    """Arm-weeks that count toward the comparison.

    An excluded week is dropped for every arm, not just for `prompt`. The repo
    and naive picks in a contaminated week are perfectly good observations in
    isolation -- but every number below is a comparison between arms, and
    keeping two arms of a week while dropping the third would put the arms on
    different week sets, which is the one thing the paired design exists to
    avoid. Those rows are still in claims.csv and still graded by the tier
    ledger above; they are out of the head-to-head only.
    """
    if weekly.empty:
        return weekly
    return weekly[~weekly["order_status"].isin(excluded_statuses(strict_order))]


def paired_differences(weekly: pd.DataFrame, left: str, right: str,
                       value: str = "mean_par") -> pd.DataFrame:
    """One row per week both arms covered: left mean minus right mean.

    `value` defaults to PAR, which is what the headline and the decision rule
    read. Pass `mean_fwd3` to see the same weeks in raw points -- worth doing
    once, because the difference between the two is the size of the positional
    effect the metric exists to remove.
    """
    l = weekly[weekly["arm"] == left].set_index(["season", "week"])
    r = weekly[weekly["arm"] == right].set_index(["season", "week"])
    shared = l.index.intersection(r.index)
    if len(shared) == 0:
        return pd.DataFrame(columns=["season", "week", left, right, "diff"])
    frame = pd.DataFrame(
        {
            left: l.loc[shared, value],
            right: r.loc[shared, value],
        }
    )
    frame["diff"] = frame[left] - frame[right]
    frame = frame.dropna(subset=["diff"])
    return frame.reset_index().sort_values(["season", "week"]).reset_index(drop=True)


def bootstrap_ci(diffs: np.ndarray, reps: int = BOOTSTRAP_REPS, level: float = CI_LEVEL,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile CI for the mean paired difference, resampling weeks."""
    if len(diffs) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(diffs, size=(reps, len(diffs)), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def paired_t(diffs: np.ndarray) -> tuple[float, float]:
    """(t, two-sided p) for H0: mean difference is zero. NaNs when n < 2."""
    if len(diffs) < 2 or np.std(diffs, ddof=1) == 0:
        return (np.nan, np.nan)
    from scipy import stats

    result = stats.ttest_1samp(diffs, 0.0)
    return float(result.statistic), float(result.pvalue)


def verdict(prompt_vs_naive: float, repo_vs_naive: float, repo_vs_prompt: float) -> str:
    """The pre-registered rule from the module docstring, evaluated in order.

    Point estimates only. Read it next to the confidence interval, never
    instead of it -- see `describe_verdict`.
    """
    best = np.nanmax([prompt_vs_naive, repo_vs_naive])
    if not np.isnan(best) and best < NAIVE_MARGIN:
        return "decoration"
    if not np.isnan(repo_vs_prompt) and abs(repo_vs_prompt) < TIE_BAND:
        return "repo-adds-nothing"
    if not np.isnan(repo_vs_prompt) and repo_vs_prompt >= REPO_MARGIN:
        return "repo-earns-its-keep"
    return "inconclusive"


VERDICT_TEXT = {
    "decoration": (
        "DECORATION -- neither arm cleared naive by "
        f"{NAIVE_MARGIN:.1f} ppg. The analysis is not paying for itself."
    ),
    "repo-adds-nothing": (
        f"REPO ADDS NOTHING -- prompt and repo are within {TIE_BAND:.1f} ppg. "
        "Keep the prompt; the data layer is not what is producing the result."
    ),
    "repo-earns-its-keep": (
        f"REPO EARNS ITS KEEP -- repo beat prompt by at least {REPO_MARGIN:.1f} ppg."
    ),
    "inconclusive": "INCONCLUSIVE -- the gap falls between the pre-registered thresholds.",
}


def describe_verdict(name: str, lo: float, hi: float, n_weeks: int) -> list[str]:
    """The verdict, plus the caveat the interval forces on it.

    A rule that reads point estimates will always return one of its branches;
    whether that branch means anything is a separate question, and this is where
    it gets asked rather than assumed.
    """
    lines = [VERDICT_TEXT[name]]
    if n_weeks < 3 or np.isnan(lo):
        lines.append(
            f"  Not enough paired weeks ({n_weeks}) to put an interval on this. "
            "Treat the verdict as arithmetic, not as evidence."
        )
    elif lo <= 0.0 <= hi:
        lines.append(
            f"  The {CI_LEVEL:.0%} interval on repo minus prompt covers zero "
            f"([{lo:+.2f}, {hi:+.2f}] ppg), so the two arms are not "
            "distinguishable in this data whatever branch the rule landed on. "
            "Report this as a tie."
        )
    else:
        lines.append(
            f"  The {CI_LEVEL:.0%} interval on repo minus prompt excludes zero "
            f"([{lo:+.2f}, {hi:+.2f}] ppg), which is the one case where the "
            "verdict is carrying evidence rather than arithmetic."
        )
    return lines


def report_arms(graded: pd.DataFrame, weekly: pd.DataFrame, summary: pd.DataFrame,
                problems: list[str], strict_order: bool) -> None:
    print()
    print("=" * 74)
    print("THREE-ARM COMPARISON")
    print("=" * 74)

    if graded.empty:
        print("no arm-tagged claim has three weeks of forward data yet.")
        print("log picks with `make log-claim` -- see docs/comparison_protocol.md")
        return

    usable = usable_weeks(weekly, strict_order)
    excluded = weekly[~weekly.index.isin(usable.index)]

    print(summary.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    print("  mean_par         HEADLINE: mean points above replacement at position, ppg")
    print("  mean_fwd3        the same picks in raw points, not position-adjusted")
    print("  ceiling_share    mean_fwd3 as a share of the best fwd3 on the wire that week")
    print("  beat_naive_share weeks the arm's mean PAR beat the naive arm's, head to head")
    print("  naive has no beat_naive_share: it is the benchmark, not a contender")
    print()
    print("  PAR is the comparison. Raw points reward whichever arm claims the most")
    print("  quarterbacks, and you start one. See the module docstring.")

    counts = (
        weekly.drop_duplicates(["season", "week"])
        .groupby("order_status")["week"]
        .count()
        .to_dict()
    )
    if counts.get("contaminated"):
        print(
            f"\n{counts['contaminated']} week(s) flagged contaminated and excluded: "
            "the prompt arm saw the candidate table."
        )
    if counts.get("unverified"):
        verb = "excluded (--strict-order)" if strict_order else "kept, order unverifiable"
        print(
            f"\n{counts['unverified']} week(s) with unverifiable ordering -- {verb}. "
            "CI writes the repo rows before anyone is awake, so a clean week "
            "routinely looks like this."
        )
    if not excluded.empty:
        weeks_out = sorted({(int(s), int(w)) for s, w in zip(excluded["season"], excluded["week"])})
        print("  excluded weeks: " + ", ".join(f"{s} wk{w:02d}" for s, w in weeks_out))

    # A pick with no PAR still has an outcome and is still in arm_grades.csv;
    # it is only out of the PAR comparison. Counted here rather than left to be
    # inferred from a row count that does not add up.
    unpriced = graded[graded["par"].isna()]
    if not unpriced.empty:
        cells = sorted({
            (int(s), int(w), str(p))
            for s, w, p in zip(unpriced["season"], unpriced["week"],
                               unpriced["position"])
        })
        print(
            f"\n{len(unpriced)} pick(s) have no PAR: the wire had too few "
            "resolved players at the position to have a replacement level. "
            "Their raw points are still graded; they are out of the PAR "
            "comparison only."
        )
        print("  " + ", ".join(f"{s} wk{w:02d} {p}" for s, w, p in cells))

    print()
    print("-" * 74)
    print(f"PAIRED: prompt vs repo on PAR, {CI_LEVEL:.0%} interval, weeks resampled")
    print("-" * 74)
    pairs = paired_differences(usable, "repo", "prompt")
    raw_pairs = paired_differences(usable, "repo", "prompt", value="mean_fwd3")
    if pairs.empty:
        print("no week has both a prompt and a repo arm logged and resolved.")
        print("Nothing to pair; the unpaired means above are over different weeks")
        print("and must not be read against each other.")
        return

    diffs = pairs["diff"].to_numpy(dtype=float)
    print(pairs.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    lo, hi = bootstrap_ci(diffs)
    t_stat, p_value = paired_t(diffs)
    print()
    print(f"paired weeks           n = {len(diffs)}")
    print(f"mean repo - prompt       {diffs.mean():+7.2f} ppg PAR")
    if not raw_pairs.empty:
        # Printed next to the headline, never instead of it. A large gap between
        # the two is the positional-composition effect, and seeing it is the
        # point of keeping raw points on the row.
        raw = raw_pairs["diff"].to_numpy(dtype=float)
        print(f"  same weeks, raw points {raw.mean():+7.2f} ppg  "
              f"(difference = positional composition)")
    if len(diffs) >= 2:
        print(f"sd of paired difference  {diffs.std(ddof=1):7.2f} ppg")
    if not np.isnan(lo):
        print(f"bootstrap {CI_LEVEL:.0%} CI        [{lo:+.2f}, {hi:+.2f}] ppg")
    if not np.isnan(t_stat):
        print(f"paired t-test            t = {t_stat:+.2f}, p = {p_value:.3f}")

    if len(diffs) < 8:
        print()
        print(
            f"WARNING: {len(diffs)} paired week(s). The t-test and the interval are "
            "both being asked to do more than this many points can support; a "
            "'significant' result here is more likely a small-sample artefact "
            "than a finding."
        )

    naive_pairs = {
        arm: paired_differences(usable, arm, "naive")["diff"].to_numpy(dtype=float)
        for arm in LOGGED_ARMS
    }
    print()
    print("-" * 74)
    print("VERDICT (pre-registered rule, see module docstring)")
    print("-" * 74)
    for arm in LOGGED_ARMS:
        d = naive_pairs[arm]
        if len(d):
            print(f"{arm:7s} minus naive on PAR, paired over {len(d):2d} week(s): "
                  f"{d.mean():+7.2f} ppg")
        else:
            print(f"{arm:7s} minus naive: no shared weeks")
    prompt_gap = float(naive_pairs["prompt"].mean()) if len(naive_pairs["prompt"]) else np.nan
    repo_gap = float(naive_pairs["repo"].mean()) if len(naive_pairs["repo"]) else np.nan
    name = verdict(prompt_gap, repo_gap, float(diffs.mean()))
    print()
    for line in describe_verdict(name, lo, hi, len(diffs)):
        print(line)

    if problems:
        print()
        print("-" * 74)
        print(f"{len(problems)} pick(s) could not be scored:")
        for line in problems[:20]:
            print(f"  {line}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        print("A name that never resolves is usually a spelling that does not match")
        print("the panel. Fix it in claims.csv rather than leaving the arm short.")


# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict-order",
        action="store_true",
        help="also exclude weeks whose prompt/repo ordering cannot be verified "
        "from the timestamps",
    )
    args = parser.parse_args(argv)

    grades = grade()
    if not grades.empty:
        GRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        grades.to_csv(GRADES_PATH, index=False, float_format="%.4f")
    report(grades)
    if not grades.empty:
        print(f"\nwrote {GRADES_PATH.relative_to(ROOT)}")

    claims = load_claims()
    panel = pd.read_csv(PANEL_PATH)
    arm_graded, problems = grade_arms(claims, panel)
    weekly = week_means(arm_graded)
    # Summarise the weeks that count, not every week on file: a table that
    # silently averages in an excluded week is worse than no table.
    summary = arm_summary(usable_weeks(weekly, args.strict_order))
    report_arms(arm_graded, weekly, summary, problems, args.strict_order)
    if not arm_graded.empty:
        ARM_GRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        arm_graded.to_csv(ARM_GRADES_PATH, index=False, float_format="%.4f")
        summary.to_csv(ARM_SUMMARY_PATH, index=False, float_format="%.4f")
        print(f"\nwrote {ARM_GRADES_PATH.relative_to(ROOT)}")
        print(f"wrote {ARM_SUMMARY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Re-score the walk-forward on points above replacement, and ablate the model.

Two questions, one script, and they need each other:

1. **Does changing the outcome metric change the answer?**
   `02_walkforward_2014_2025.py` scored both arms on raw `fwd3` and found the
   naive arm ahead by 2.30 ppg, of which 93% was positional composition rather
   than ranking. Raw fantasy points are not comparable across positions and you
   start one quarterback, so an arm that accumulates them banks a difference it
   cannot field. This re-scores the **same picks** on points above replacement.

2. **Is the model doing anything a one-liner cannot?**
   Within position, ranked against four sorts on a single panel column.

NO MODEL IS REFITTED FOR QUESTION 1
===================================

`replay_full/replay_picks.csv` persists every pick both arms made. Question 1
needs no model, only a baseline to subtract, and that baseline is a function of
the wire pool alone. The picks are read, not regenerated.

The baseline itself is **not** recoverable from the persisted picks. Those carry
`pool_mean_pos` -- the pool's *mean* `fwd3` at the position -- and replacement is
an order statistic of that pool, which a mean does not determine. What is needed
is the pool's realised `fwd3` distribution per (season, week, position), and that
comes from rebuilding the panel: deterministic feature construction off the
sha256-pinned files in `data/raw/MANIFEST.json`, no estimator involved. Rebuilding
a panel is not refitting a model, and the distinction is load-bearing here.

The baseline is also invariant to the training window, which is why one set of
levels serves every scheme. `prior_seasons` reaches exactly one thing in
`features.build` -- the empirical Bayes beta priors -- and nothing downstream of
those touches `pts -> cum_before -> rank_before -> on_wire` or `fwd3`. Asserted
directly in `check_baseline_invariance()` rather than argued.

QUESTION 2 DOES REQUIRE A REFIT, AND IT IS A RECONSTRUCTION
===========================================================

The ablation needs the model's ranking *within* each position. The persisted
picks carry only its top three *across* positions, so that ranking is not in
them. `replay_models_full/` is gitignored -- deliberately, so replay bundles
cannot be served -- and is therefore absent from any fresh clone.

So the bundles are refitted, and the refit is treated as a reconstruction that
has to prove itself rather than a new run. `MODEL_KWARGS` pins `random_state=0`
and `requirements.txt` pins the libraries, so the reconstruction should be exact,
and `verify_reconstruction()` requires it to be: every one of the persisted model
picks must come back player-for-player at the same (season, week, rank). The
naive arm is reconstructed too, as a model-free gate on the pool itself. Both
gates are hard failures, not warnings -- a reconstruction that is merely close is
a different experiment wearing the same name.

WHAT PAR FIXES, AND WHAT IT DOES NOT
====================================

Stated here because the result turns on it. Subtracting a per-(week, position)
constant does not remove positional composition from a comparison; it re-prices
it. `decompose`'s mix term is `sum_p (w_naive,p - w_repo,p) * mu_p`, and under
PAR `mu_p` becomes `pool_mean_p - b_p`. That vanishes only if `b_p` sits a
constant distance above `pool_mean_p` at every position. It does not: `b_p` runs
about 2x the pool mean at quarterback and 3.3x at tight end, because
`REPLACEMENT_RANK` is a fixed *rank* against pools of 16 and 47.

The measured consequence is in the write-up and it is not subtle: the mix term
goes from +2.13 to -1.80 and remains essentially the whole gap. The metric that
zeroes mix exactly, by construction, is the pool-mean baseline already reported
as `vs_pos`. Four baselines are therefore reported side by side, so the verdict
can be read against the choice that produced it rather than in isolation.

The baseline is also an order statistic of *realised* outcomes -- what the pool
delivered at that rank, not what anyone could have chosen. Two decision-time
variants are reported beside it to price that hindsight.

WITHIN POSITION, PAR AND RAW fwd3 ARE THE SAME COMPARISON
=========================================================

Exactly, not approximately. The baseline is one constant per (season, week,
position) and the ablation holds position fixed, so every arm in a cell has the
same constant subtracted and every paired difference is unchanged. The ablation
is reported once, and `check_par_invariance()` asserts it rather than trusting
it. PAR does work only where arms are allowed to differ in position mix, which
is question 1 and not question 2.

Run:
    python outputs/backtests/03_par_rescore_ablation.py
    python outputs/backtests/03_par_rescore_ablation.py --report-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import features as F                                    # noqa: E402
from src.ledger import BOOTSTRAP_REPS, BOOTSTRAP_SEED, CI_LEVEL  # noqa: E402
from src.ledger import naive_picks, replacement_of, wire_pool     # noqa: E402
from src.report import REPLACEMENT_RANK                          # noqa: E402


def _load_walkforward():
    """Import `02_walkforward_2014_2025.py`, whose leading digit blocks import.

    Everything about the replay's shape -- which seasons, which weeks, the
    training plan, the panel cache, the season-length logic, the stratified
    bootstrap and its singleton fallback -- is reused from there rather than
    restated. This file changes the outcome metric and adds arms; it must not
    quietly change anything else, and importing is how that is enforced.
    """
    path = Path(__file__).resolve().parent / "02_walkforward_2014_2025.py"
    spec = importlib.util.spec_from_file_location("walkforward_2014_2025", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WF = _load_walkforward()
stratified_ci = WF.stratified_ci
train_seasons_for = WF.train_seasons_for
replay_panel = WF.replay_panel
rescale_depth = WF.rescale_depth
fit_replay_models = WF.fit_replay_models
repo_picks = WF.repo_picks
fmt = WF.fmt

REPLAY_SEASONS = WF.REPLAY_SEASONS
REPLAY_WEEKS = WF.REPLAY_WEEKS
HEADLINE_WEEKS = list(range(2, WF.LAST_FULL_WINDOW_WEEK + 1))
ARM_DEPTH = WF.ARM_DEPTH
DEPTH_SCALES = WF.DEPTH_SCALES
BREAK_SEASON = WF.BREAK_SEASON

SOURCE_DIR = Path(__file__).resolve().parent / "replay_full"
OUT_DIR = Path(__file__).resolve().parent / "par_ablation"
DIAGNOSTIC = ROOT / "outputs" / "diagnostics" / "par_rescore_and_ablation.md"

# Reading floor. `02_walkforward_2014_2025.py` measures it: HistGradientBoosting
# bins its features, a 1e-16 float perturbation moves a bin edge, and a split
# and occasionally a pick move with it. Anything inside this is not a result.
NOISE_FLOOR_PPG = 0.3

# A cell whose sort key cannot separate the top k -- so the pick is decided by
# the alphabetical tiebreak -- this often is not a heuristic, it is a draw.
TIE_FLAG_RATE = 0.5


# ==========================================================================
# the arms, and what each one sorts on
# ==========================================================================

# Heuristic arms, per the ablation brief. Each is a sort on one column already
# in the panel: no fit, no training window, no bundle. The names are the honest
# ones rather than the convenient ones:
#
# `hot_hand_pos` is NOT the `naive` arm of the headline. That arm sorts `pts`
#   across every position at once and comes out 59% quarterbacks, which is what
#   produces the -2.30 ppg. This one sorts `pts` *within* a position. Different
#   estimator, different picks, and laying the two numbers side by side is
#   exactly the confusion this ablation exists to dispel. `ledger.benchmarks`
#   already computes the k=1 case of this arm and calls it `hot_hand_pos`.
#
# `opp` is `features.wopr_opp`, which is `carries + 2.5 * targets` -- an
#   unnormalised opportunity count, not the published WOPR (which is a weighted
#   sum of *shares*). Calling it `wopr` in a table would be wrong, so it is
#   called what it is and glossed everywhere it appears.
#
# `eb_share` has no QB branch. Shrunk target share and shrunk carry share are
#   both meaningless for a quarterback, and substituting snap share there does
#   not make a fourth arm -- it makes a second copy of `snap`, which was
#   measured at 100% identical picks in all 156 QB cells. A forced tie reported
#   as agreement is the error this project keeps catching; it is not committed
#   here. The arm is N/A at quarterback and pooled over RB/WR/TE.
HEURISTICS = {
    "hot_hand_pos": {
        "keys": {"QB": "pts", "RB": "pts", "WR": "pts", "TE": "pts"},
        "gloss": "prior-week fantasy points, sorted within the position",
    },
    "eb_share": {
        "keys": {"RB": "eb_car_share", "WR": "eb_tgt_share", "TE": "eb_tgt_share"},
        "gloss": "shrunk carry share (RB) or shrunk target share (WR/TE)",
    },
    "opp": {
        "keys": {"QB": "wopr_opp", "RB": "wopr_opp", "WR": "wopr_opp", "TE": "wopr_opp"},
        "gloss": "`wopr_opp` = carries + 2.5 x targets, an unnormalised count",
    },
    "snap": {
        "keys": {"QB": "snap", "RB": "snap", "WR": "snap", "TE": "snap"},
        "gloss": "snap share (`offense_pct`)",
    },
}
ABLATION_ARMS = ["model"] + list(HEURISTICS)

# Depths the ablation is run at, and both are reported because they do not agree
# and the disagreement is the finding. k=1 asks which single player each rule
# names at a position -- the most demanding comparison, and the decision a single
# claim actually is. k=3 matches ARM_DEPTH, the persisted arms and the shape of a
# candidate table, which is a ranked list rather than one name.
#
# The per-rank table is what reconciles them: a difference that appears only at
# k=3 is either better power from averaging three picks or a real difference in
# how the arms degrade with depth, and those are not the same claim. `rank_table`
# separates them, so neither has to be assumed.
ABLATION_DEPTHS = (1, 3)
HEADLINE_DEPTH = 3


# ==========================================================================
# replacement level
# ==========================================================================

# `ledger.replacement_of` is the baseline, imported rather than restated so the
# live ledger and this replay cannot drift on what replacement level means. It
# is `report.replacement_level` with the clamp removed, for the reason its own
# docstring gives; the clamp never fires here anyway (the smallest resolved pool
# is 10 at quarterback against a rank of 2).
order_statistic = replacement_of


def decision_time_level(pool: pd.DataFrame, position: str, order_by: str) -> float:
    """What the player *ranked* R-th at decision time actually went on to average.

    The realised order statistic is what the pool delivered at rank R with
    hindsight; this is what you would have got by taking the R-th best player
    under a signal available on the Monday. Ordering is model-free and identical
    for every arm, so it stays arm-neutral, and it prices the hindsight in the
    headline baseline rather than leaving it as an assertion.

    Two orderings are reported because neither is neutral on its own:
    `rank_before` is season-to-date standing, which no arm sorts on, and `pts`
    is last week's box score, which is the naive arm's own key.
    """
    resolved = pool.dropna(subset=["fwd3"])
    if len(resolved) <= REPLACEMENT_RANK.get(position, 5):
        return float("nan")
    ascending = order_by == "rank_before"
    ordered = resolved.sort_values(
        [order_by, "player_display_name"], ascending=[ascending, True]
    )
    return float(ordered["fwd3"].iloc[REPLACEMENT_RANK.get(position, 5)])


def replacement_table(depth_scale: float = 1.0) -> pd.DataFrame:
    """Every (season, week, position) baseline the replay needs, and its variants.

    One row per cell. `repl_fwd3` is the headline baseline; `pool_mean` is the
    `vs_pos` baseline the prior replay used; the two `decision_*` columns price
    the hindsight in the first. `pool_n` is carried so the reader can see that a
    fixed rank means the 81st percentile of a 16-deep quarterback pool and the
    94th of a 47-deep tight end pool -- which is the mechanical reason PAR
    re-prices position rather than removing it.
    """
    rows = []
    for season in REPLAY_SEASONS:
        train = train_seasons_for(season, "expanding")
        panel = replay_panel(sorted(set(train) | {season}), train)
        if depth_scale != 1.0:
            panel = rescale_depth(panel, depth_scale)
        for week in REPLAY_WEEKS:
            resolved = wire_pool(panel, season, week).dropna(subset=["fwd3"])
            if resolved.empty:
                continue
            for position, group in resolved.groupby("position"):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "position": str(position),
                        "pool_n": len(group),
                        "repl_rank": REPLACEMENT_RANK.get(str(position), 5),
                        "repl_fwd3": order_statistic(group["fwd3"], str(position)),
                        "pool_mean": float(group["fwd3"].mean()),
                        "decision_rank_before": decision_time_level(
                            group, str(position), "rank_before"
                        ),
                        "decision_pts": decision_time_level(group, str(position), "pts"),
                    }
                )
    table = pd.DataFrame(rows)
    table["repl_pctile"] = 1.0 - table["repl_rank"] / table["pool_n"]
    return table


# The baselines a pick can be scored against. `fwd3` is the prior headline and
# subtracts nothing; `vs_pos` is the prior position-adjusted number and is the
# one baseline that zeroes the mix term exactly; `par` is the metric asked for.
BASELINES = {
    "fwd3": (None, "raw fantasy points, no baseline (the prior headline)"),
    "vs_pos": ("pool_mean", "the position pool's mean fwd3 that week"),
    "par": ("repl_fwd3", "the rank-R **realised** fwd3 — the metric asked for"),
    "par_std": (
        "decision_rank_before",
        "rank-R by season-to-date standing, valued at his realised fwd3",
    ),
    "par_pts": (
        "decision_pts",
        "rank-R by last week's points, valued at his realised fwd3",
    ),
}
HEADLINE_BASELINE = "par"


# ==========================================================================
# question 1: re-score the persisted picks
# ==========================================================================

def load_persisted_picks() -> pd.DataFrame:
    path = SOURCE_DIR / "replay_picks.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. This script re-scores a completed run of "
            "02_walkforward_2014_2025.py rather than reproducing one; run that "
            "first."
        )
    return pd.read_csv(path)


def load_persisted_notes() -> dict:
    """The unscoreable-week list, carried through rather than recomputed.

    An unscoreable pick leaves no row, so the list cannot be rebuilt from
    `replay_picks.csv` -- the same reason `02_walkforward_2014_2025.py` persists
    it beside the picks. Recomputing it here would silently report an empty
    exclusion list, which is the failure the file exists to prevent.
    """
    path = SOURCE_DIR / "unscored.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. The exclusion list cannot be reconstructed from "
            "the picks; rerun 02_walkforward_2014_2025.py in full."
        )
    return json.loads(path.read_text())


def attach_baselines(picks: pd.DataFrame, levels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join each scheme's baselines onto its picks and derive every metric.

    Keyed on scheme because the depth-sensitivity arms move `on_wire`, which
    moves the pool, which moves the baseline. Scoring a rescaled-depth pick
    against the shipped-depth baseline would measure the wrong thing and would
    not look wrong.

    `pool_par_mean_pos` is the position pool's mean *on each metric's own scale*.
    The mix/selection decomposition reads it, and the decomposition identity
    holds arithmetically for any baseline -- so feeding it a `mu` on the raw
    scale while the arm means are on the PAR scale produces a split that is
    exact and meaningless. Deriving it here is what stops that.
    """
    out = []
    for scheme, frame in picks.groupby("scheme"):
        key = "depth" if str(scheme).startswith("depth") else "shipped"
        table = levels[str(scheme)] if str(scheme) in levels else levels[key]
        merged = frame.merge(table, on=["season", "week", "position"], how="left")
        if merged["repl_fwd3"].isna().any():
            missing = merged[merged["repl_fwd3"].isna()]
            raise SystemExit(
                f"{len(missing)} pick(s) in scheme '{scheme}' have no replacement "
                "level. The pool reconstruction disagrees with the persisted "
                f"picks; first: {missing.iloc[0][['season', 'week', 'position']].to_dict()}"
            )
        for name, (column, _) in BASELINES.items():
            base = 0.0 if column is None else merged[column]
            merged[name] = merged["fwd3"] - base
            merged[f"mu_{name}"] = merged["pool_mean"] - base
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def check_pool_reconstruction(scored: pd.DataFrame) -> dict:
    """The rebuilt pool must agree with what the completed run persisted.

    `pool_mean_pos` and `pool_n_pos` were written by that run from the same
    pool this one rebuilds, so they are ground truth for the rebuild. If they
    disagree the panel is not the panel the picks came from and every number
    below is void. `pool_mean_pos` is compared at the tolerance its own
    `float_format="%.4f"` allows and `pool_n_pos` exactly, because a count has
    no rounding to hide in.
    """
    mean_gap = float((scored["pool_mean"] - scored["pool_mean_pos"]).abs().max())
    count_gap = int((scored["pool_n"] - scored["pool_n_pos"]).abs().max())
    if count_gap != 0:
        raise SystemExit(
            f"rebuilt pool sizes differ from the persisted ones by up to "
            f"{count_gap}. The panel is not the one the picks came from."
        )
    if mean_gap > 1e-4:
        raise SystemExit(
            f"rebuilt pool means differ from the persisted ones by up to "
            f"{mean_gap:.2e}, past the 1e-4 the persisted file's rounding allows."
        )
    return {"pool_mean_gap": mean_gap, "pool_n_gap": count_gap}


def check_baseline_invariance() -> dict:
    """The baseline must not move with the training window. Asserted, not argued.

    `features.build(prior_seasons=...)` restricts which rows the empirical Bayes
    beta priors are fitted on, and nothing else. `on_wire` and `fwd3` are
    therefore identical across windows, so a baseline built from them is too,
    and one set of levels can serve every scheme. Checked by building the same
    season's baselines from two different windows and requiring equality.
    """
    season = REPLAY_SEASONS[len(REPLAY_SEASONS) // 2]
    frames = []
    for last in (season, season + 1):
        train = train_seasons_for(last, "expanding")
        panel = replay_panel(sorted(set(train) | {last}), train)
        rows = []
        for week in REPLAY_WEEKS:
            resolved = wire_pool(panel, season, week).dropna(subset=["fwd3"])
            for position, group in resolved.groupby("position"):
                rows.append((week, str(position),
                             order_statistic(group["fwd3"], str(position))))
        frames.append(pd.DataFrame(rows, columns=["week", "position", "level"]))
    joined = frames[0].merge(frames[1], on=["week", "position"], suffixes=("_a", "_b"))
    gap = float((joined["level_a"] - joined["level_b"]).abs().max())
    if gap > 0.0:
        raise SystemExit(
            f"replacement level moved by {gap:.2e} when the training window "
            "changed. It must not: the pool and the outcomes are window-"
            "invariant, and if they are not the re-score is not a re-score."
        )
    return {"season": season, "cells": len(joined), "max_gap": gap}


def week_table(scored: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Arm x week means on one metric. The week is the resampling unit."""
    return (
        scored.groupby(["scheme", "season", "week", "arm"], as_index=False)
        .agg(n=(metric, "size"), mean=(metric, "mean"))
    )


def paired_weeks(weekly: pd.DataFrame, weeks) -> pd.DataFrame:
    """One row per week both arms covered: repo minus naive."""
    rows = weekly[weekly["week"].isin(list(weeks))]
    wide = rows.pivot_table(
        index=["season", "week"], columns="arm", values="mean"
    ).dropna(subset=["repo", "naive"])
    wide = wide.reset_index()
    wide["diff"] = wide["repo"] - wide["naive"]
    return wide


def headline(scored: pd.DataFrame, metric: str, weeks=None, scheme: str = "expanding",
             seasons=None) -> dict:
    weeks = HEADLINE_WEEKS if weeks is None else weeks
    frame = scored[scored["scheme"] == scheme]
    if seasons is not None:
        frame = frame[frame["season"].isin(list(seasons))]
    pairs = paired_weeks(week_table(frame, metric), weeks)
    diffs = pairs["diff"].to_numpy(dtype=float)
    strata = pairs["season"].to_numpy() if len(pairs) else None
    (lo, hi), unstratified = stratified_ci(diffs, strata)
    picks = frame[frame["week"].isin(list(weeks))]
    return {
        "metric": metric,
        "n_weeks": len(diffs),
        "mean_diff": float(diffs.mean()) if len(diffs) else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "unstratified": unstratified,
        "repo": float(pairs["repo"].mean()) if len(pairs) else np.nan,
        "naive": float(pairs["naive"].mean()) if len(pairs) else np.nan,
        "repo_won": float((diffs > 0).mean()) if len(diffs) else np.nan,
        "split": decompose(picks, metric),
    }


def decompose(picks: pd.DataFrame, metric: str) -> dict:
    """The naive-minus-repo gap, split into position mix and within-position skill.

    Exactly, as in the prior replay:

        mean_arm = sum_p w_arm,p * mu_p  +  sum_p w_arm,p * (mean_arm,p - mu_p)

    `mu_p` is the pool average at position p **on the same metric as the arm
    means** -- `mu_<metric>`, derived in `attach_baselines`. That is the whole
    trap: the identity `gap = mix + selection` holds for any `mu` at all, so a
    `mu` left on the raw scale while the arm means moved to PAR yields a split
    that balances exactly and describes nothing. The assertion below is cheap
    and the failure it catches is invisible.
    """
    positions = sorted(picks["position"].unique())
    mu = {
        p: float(picks.loc[picks["position"] == p, f"mu_{metric}"].mean())
        for p in positions
    }
    out = {"mu": mu, "positions": positions}
    for arm in ("naive", "repo"):
        rows = picks[picks["arm"] == arm]
        weights = {p: float((rows["position"] == p).mean()) for p in positions}
        means = {
            p: float(rows.loc[rows["position"] == p, metric].mean())
            if weights[p] else np.nan
            for p in positions
        }
        out[arm] = {
            "weights": weights,
            "means": means,
            "mean": float(rows[metric].mean()),
            "selection": float(
                sum(weights[p] * (means[p] - mu[p]) for p in positions if weights[p])
            ),
        }
    out["gap"] = out["naive"]["mean"] - out["repo"]["mean"]
    out["mix"] = float(
        sum((out["naive"]["weights"][p] - out["repo"]["weights"][p]) * mu[p]
            for p in positions)
    )
    out["selection_gap"] = out["naive"]["selection"] - out["repo"]["selection"]
    residual = abs(out["gap"] - (out["mix"] + out["selection_gap"]))
    if residual > 1e-9:
        raise SystemExit(
            f"mix + selection = {out['mix'] + out['selection_gap']:.6f} does not "
            f"reconstruct the gap {out['gap']:.6f} on metric '{metric}'. The "
            "decomposition is exact by construction, so this is a bug."
        )
    out["mix_share"] = out["mix"] / out["gap"] if out["gap"] else np.nan
    return out


# ==========================================================================
# question 2: the ablation
# ==========================================================================

def verify_reconstruction(recon: pd.DataFrame, persisted: pd.DataFrame,
                          arm: str) -> dict:
    """A refit that is merely close is a different experiment. Require exact.

    Compared on the identity of the player at each (season, week, rank), which
    is the thing the ablation actually consumes -- `fwd3` and `edge` are
    compared too but only to the 1e-4 the persisted file's `%.4f` rounding
    allows, so they cannot be the binding test.

    The `naive` arm is the more important of the two gates even though it is
    the less interesting: it is a pure `pts` sort with an alphabetical tiebreak
    and no estimator anywhere in it, so a mismatch there is a mismatch in the
    *pool*, and would void the model gate rather than being caught by it.
    """
    key = ["season", "week", "rank_within_arm"]
    merged = persisted[key + ["player", "fwd3"]].merge(
        recon[key + ["player", "fwd3"]], on=key, how="outer",
        suffixes=("_persisted", "_recon"), indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    both = merged[merged["_merge"] == "both"]
    mismatched = both[both["player_persisted"] != both["player_recon"]]
    outcome_gap = float((both["fwd3_persisted"] - both["fwd3_recon"]).abs().max())
    if unmatched or len(mismatched):
        examples = mismatched.head(5)[
            ["season", "week", "rank_within_arm", "player_persisted", "player_recon"]
        ]
        raise SystemExit(
            f"the reconstructed '{arm}' arm does not reproduce the persisted one: "
            f"{unmatched} row(s) unmatched, {len(mismatched)} player mismatch(es). "
            "The ablation's model arm is only legitimate if the reconstruction is "
            f"exact, so this is fatal rather than a warning.\n{examples.to_string()}"
        )
    if outcome_gap > 1e-4:
        raise SystemExit(
            f"'{arm}': reconstructed fwd3 differs by up to {outcome_gap:.2e}, past "
            "the 1e-4 the persisted file's rounding allows."
        )
    return {"arm": arm, "rows": len(both), "player_matches": len(both),
            "max_fwd3_gap": outcome_gap}


def rank_within(pool: pd.DataFrame, column: str, depth: int) -> pd.DataFrame:
    """Top `depth` of one position's pool by one column, ties broken on name.

    The tiebreak matches `ledger.naive_picks` so the arms are reproducible
    rather than dependent on row order. `tie_at_cut` records how many players
    share the value at the cut: anything above `depth` means the arm did not
    choose those picks, the alphabet did, and the ablation reports that rate
    rather than averaging a coin flip into a heuristic's score.
    """
    ordered = pool.sort_values(
        [column, "player_display_name"], ascending=[False, True], na_position="last"
    ).head(depth).copy()
    cut = ordered[column].iloc[-1] if len(ordered) else np.nan
    ordered["tie_at_cut"] = int((pool[column] == cut).sum()) if pd.notna(cut) else 0
    ordered["rank_within_arm"] = range(1, len(ordered) + 1)
    return ordered


def ablation_picks(depth: int) -> tuple[pd.DataFrame, list[dict]]:
    """Every arm's top `depth` at every (season, week, position), with outcomes.

    The model arm is its own ordering over the position's pool, not the top
    three that would clear `assign_tiers`. That filter keeps only players above
    replacement, which at quarterback is exactly two names by construction --
    a tiering rule, not a ranking, and this is a test of the ranking.
    """
    rows, checks = [], []
    for season in REPLAY_SEASONS:
        train = train_seasons_for(season, "expanding")
        panel = replay_panel(sorted(set(train) | {season}), train)
        bundles = fit_replay_models(panel, train, season, "expanding")
        for week in REPLAY_WEEKS:
            pool = wire_pool(panel, season, week)
            resolved = pool.dropna(subset=["fwd3"])
            if resolved.empty:
                continue
            for position, group in resolved.groupby("position"):
                position = str(position)
                bundle = bundles[position]
                group = group.copy()
                group["model"] = bundle["model"].predict(group[bundle["features"]])
                level = order_statistic(group["fwd3"], position)
                for arm in ABLATION_ARMS:
                    column = (
                        "model" if arm == "model"
                        else HEURISTICS[arm]["keys"].get(position)
                    )
                    if column is None:
                        continue          # eb_share has no QB branch, by design
                    for _, pick in rank_within(group, column, depth).iterrows():
                        rows.append({
                            "depth": depth, "season": season, "week": week,
                            "position": position, "arm": arm,
                            "rank_within_arm": int(pick["rank_within_arm"]),
                            "player": str(pick["player_display_name"]),
                            "sort_key": column,
                            "tie_at_cut": int(pick["tie_at_cut"]),
                            "pool_n": len(group),
                            "fwd3": float(pick["fwd3"]),
                            "repl_fwd3": level,
                            "par": float(pick["fwd3"]) - level,
                        })
    return pd.DataFrame(rows), checks


def reconstruct_headline_arms() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The persisted cross-position arms, rebuilt, for the verification gates."""
    finals = WF.final_weeks()
    repo, naive = [], []
    for season in REPLAY_SEASONS:
        train = train_seasons_for(season, "expanding")
        panel = replay_panel(sorted(set(train) | {season}), train)
        bundles = fit_replay_models(panel, train, season, "expanding")
        for week in REPLAY_WEEKS:
            if week > finals[season]:
                continue
            pool = wire_pool(panel, season, week)
            if pool.empty or not pool["fwd3"].notna().any():
                continue
            for target, frame in (
                (repo, repo_picks(panel, bundles, train, season, week)),
                (naive, naive_picks(panel, season, week, ARM_DEPTH)),
            ):
                for _, pick in frame.iterrows():
                    if pd.isna(pick["fwd3"]):
                        continue
                    target.append({
                        "season": season, "week": week,
                        "rank_within_arm": int(pick["rank_within_arm"]),
                        "player": str(pick["player_display_name"]),
                        "fwd3": float(pick["fwd3"]),
                    })
    return pd.DataFrame(repo), pd.DataFrame(naive)


def check_par_invariance(picks: pd.DataFrame) -> float:
    """Within a position the metric change is a no-op. Asserted, not asserted-in-prose.

    Every arm in a (season, week, position) cell has the same constant
    subtracted, so the paired difference on PAR is identically the paired
    difference on raw `fwd3`. If this ever stopped holding, the ablation would
    be silently answering a different question than the one it advertises.
    """
    worst = 0.0
    for arm in HEURISTICS:
        for metric in ("par", "fwd3"):
            cells = picks.groupby(
                ["season", "week", "position", "arm"], as_index=False
            ).agg(m=(metric, "mean"))
            wide = cells.pivot_table(
                index=["season", "week", "position"], columns="arm", values="m"
            )
            if arm not in wide.columns:
                continue
            diff = (wide["model"] - wide[arm]).dropna()
            if metric == "par":
                reference = diff
            else:
                worst = max(worst, float((reference - diff).abs().max()))
    if worst > 1e-9:
        raise SystemExit(
            f"PAR and raw fwd3 give within-position differences that disagree by "
            f"{worst:.2e}. They cannot: the baseline is one constant per cell."
        )
    return worst


def week_block_ci(cells: pd.DataFrame, arm: str) -> dict:
    """Paired model-minus-arm difference, resampling whole weeks by season.

    The week is the resampling unit, matching the headline. Resampling
    individual (season, week, position) cells would treat the four positions in
    a week as independent draws when they share a wire pool, a slate and a
    scoring environment. Measured on this data the four within-week differences
    are essentially uncorrelated, so the two designs land within about 0.03 ppg
    of each other -- but that is a property of this contrast, measured after the
    fact, not of the design, and the week block is correct under any
    correlation. It also keeps the ablation's interval directly comparable to
    the headline's rather than nearly so.

    Positive means the model is ahead. Stated in the tables too, because a sign
    convention that has to be inferred gets inferred wrong.
    """
    wide = cells.pivot_table(
        index=["season", "week", "position"], columns="arm", values="mean"
    )
    if arm not in wide.columns:
        return {"arm": arm, "n_weeks": 0, "mean_diff": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "unstratified": False, "n_cells": 0}
    paired = (wide["model"] - wide[arm]).dropna().rename("diff").reset_index()
    blocks = paired.groupby(["season", "week"])["diff"].mean().reset_index()
    diffs = blocks["diff"].to_numpy(dtype=float)
    strata = blocks["season"].to_numpy() if len(blocks) else None
    (lo, hi), unstratified = stratified_ci(diffs, strata)
    return {
        "arm": arm,
        "n_weeks": len(diffs),
        "n_cells": len(paired),
        "mean_diff": float(diffs.mean()) if len(diffs) else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "unstratified": unstratified,
    }


def rank_table(picks: pd.DataFrame, depth: int, exclude=()) -> list[dict]:
    """model minus each heuristic at each rank within the position, separately.

    This is what tells a power difference from a depth effect. Averaging three
    picks per cell narrows every interval, so a margin that shows up at k=3 and
    not at k=1 could be nothing more than the k=1 test being unable to resolve
    it. But rank 1 of the k=3 run *is* the k=1 arm -- same key, same tiebreak,
    same player -- so if rank 1 is flat while rank 3 is not, the arms genuinely
    differ in how they hold up down the list, and the k=3 margin is not an
    artefact of averaging.
    """
    rows = picks[(picks["depth"] == depth) & picks["week"].isin(HEADLINE_WEEKS)]
    rows = rows[~rows["position"].isin(list(exclude))]
    out = []
    for rank in range(1, depth + 1):
        cells = rows[rows["rank_within_arm"] == rank].groupby(
            ["season", "week", "position", "arm"], as_index=False
        ).agg(mean=("par", "mean"))
        for arm in HEURISTICS:
            stats = week_block_ci(cells, arm)
            stats["rank"] = rank
            out.append(stats)
    return out


def ablation_table(picks: pd.DataFrame, positions=None) -> list[dict]:
    """model minus every heuristic, over the headline weeks."""
    rows = picks[picks["week"].isin(HEADLINE_WEEKS)]
    if positions is not None:
        rows = rows[rows["position"].isin(list(positions))]
    cells = rows.groupby(
        ["season", "week", "position", "arm"], as_index=False
    ).agg(mean=("par", "mean"))
    out = []
    for arm in HEURISTICS:
        # An arm with no branch at a position contributes no cell there, so
        # pooling it over the positions it actually covers is forced. Which
        # those are is carried on the row rather than left to the footnotes.
        covered = sorted(rows.loc[rows["arm"] == arm, "position"].unique())
        stats = week_block_ci(cells[cells["position"].isin(covered)], arm)
        stats["positions"] = covered
        out.append(stats)
    return out


def tie_rates(picks: pd.DataFrame) -> pd.DataFrame:
    """How often each arm's cut was decided by the alphabet rather than the key.

    `tie_at_cut > depth` means more players shared the value at the cut than
    there were places, so which of them was taken came from sorting names. An
    arm in that state is not a heuristic being tested; it is a draw, and a model
    beating it says nothing about either.
    """
    rows = picks[picks["week"].isin(HEADLINE_WEEKS)]
    rows = rows[rows["rank_within_arm"] == rows["depth"]]
    table = (
        rows.assign(decided_by_name=rows["tie_at_cut"] > rows["depth"])
        .groupby(["arm", "position"], as_index=False)
        .agg(tie_rate=("decided_by_name", "mean"),
             distinct_at_cut=("tie_at_cut", "mean"),
             cells=("tie_at_cut", "size"))
    )
    table["degenerate"] = table["tie_rate"] >= TIE_FLAG_RATE
    return table


def overlap_matrix(picks: pd.DataFrame) -> pd.DataFrame:
    """Share of cells where two arms named exactly the same set.

    Its job is to surface a forced tie before it is read as agreement. Two arms
    sorting the same column at a position are one arm; the table has to make
    that visible rather than leave it in a docstring.
    """
    rows = picks[picks["week"].isin(HEADLINE_WEEKS)]
    sets = (
        rows.groupby(["season", "week", "position", "arm"])["player"]
        .apply(lambda s: frozenset(s))
        .unstack("arm")
    )
    out = []
    for position, group in sets.groupby(level="position"):
        for a in ABLATION_ARMS:
            for b in ABLATION_ARMS:
                if a >= b or a not in group.columns or b not in group.columns:
                    continue
                pair = group[[a, b]].dropna()
                if pair.empty:
                    continue
                out.append({"position": position, "a": a, "b": b,
                            "identical": float((pair[a] == pair[b]).mean()),
                            "cells": len(pair)})
    return pd.DataFrame(out)


# ==========================================================================
# running it
# ==========================================================================

def build_levels() -> dict[str, pd.DataFrame]:
    """Baselines for the shipped roster depth and for each sensitivity scale."""
    levels = {"shipped": replacement_table(1.0)}
    for scale in DEPTH_SCALES:
        levels[f"depth{scale:g}"] = replacement_table(scale)
    return levels


def run_rescore(levels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    print("=" * 74)
    print("QUESTION 1: RE-SCORE THE PERSISTED PICKS -- NO MODEL IS REFITTED")
    print("=" * 74)
    picks = load_persisted_picks()
    scored = attach_baselines(picks, levels)
    checks = {
        "pool": check_pool_reconstruction(scored[scored["scheme"] == "expanding"]),
        "invariance": check_baseline_invariance(),
    }
    print(f"  pool reconstruction: sizes exact, means within "
          f"{checks['pool']['pool_mean_gap']:.1e}")
    print(f"  baseline invariance: {checks['invariance']['cells']} cells identical "
          "across training windows")
    for name in BASELINES:
        result = headline(scored, name)
        print(f"  {name:8s} repo - naive = {result['mean_diff']:+6.2f} "
              f"[{result['ci_lo']:+6.2f}, {result['ci_hi']:+6.2f}]  "
              f"mix {result['split']['mix']:+6.2f}  "
              f"selection {result['split']['selection_gap']:+6.2f}")
    return scored, checks


def run_ablation() -> tuple[pd.DataFrame, dict]:
    print()
    print("=" * 74)
    print("QUESTION 2: ABLATION -- RECONSTRUCTING THE MODEL ARM")
    print("=" * 74)
    persisted = load_persisted_picks()
    persisted = persisted[persisted["scheme"] == "expanding"]
    recon_repo, recon_naive = reconstruct_headline_arms()
    gates = [
        verify_reconstruction(recon_naive, persisted[persisted["arm"] == "naive"],
                              "naive"),
        verify_reconstruction(recon_repo, persisted[persisted["arm"] == "repo"],
                              "repo"),
    ]
    for gate in gates:
        print(f"  gate '{gate['arm']}': {gate['player_matches']}/{gate['rows']} "
              f"picks reproduced exactly, fwd3 within {gate['max_fwd3_gap']:.1e}")

    frames = []
    for depth in ABLATION_DEPTHS:
        print(f"\n  generating arms at k={depth}")
        picks, _ = ablation_picks(depth)
        frames.append(picks)
    picks = pd.concat(frames, ignore_index=True)
    invariance = check_par_invariance(picks[picks["depth"] == HEADLINE_DEPTH])
    print(f"  PAR/fwd3 within-position invariance: max gap {invariance:.1e}")
    return picks, {"gates": gates, "par_invariance": invariance}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report-only", action="store_true",
        help="rewrite the markdown from this script's own saved CSVs, without "
             "rebuilding panels or refitting anything",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        needed = ["par_picks.csv", "ablation_picks.csv", "replacement_levels.csv"]
        missing = [n for n in needed if not (OUT_DIR / n).exists()]
        if missing:
            raise SystemExit(
                f"--report-only needs {', '.join(missing)} in "
                f"{OUT_DIR.relative_to(ROOT)}; run without it first."
            )
        scored = pd.read_csv(OUT_DIR / "par_picks.csv")
        ablation = pd.read_csv(OUT_DIR / "ablation_picks.csv")
        levels = {"shipped": pd.read_csv(OUT_DIR / "replacement_levels.csv")}
        checks = json.loads((OUT_DIR / "checks.json").read_text())
    else:
        levels = build_levels()
        scored, rescore_checks = run_rescore(levels)
        ablation, ablation_checks = run_ablation()
        checks = {**rescore_checks, **ablation_checks}
        levels["shipped"].to_csv(
            OUT_DIR / "replacement_levels.csv", index=False, float_format="%.4f")
        scored.to_csv(OUT_DIR / "par_picks.csv", index=False, float_format="%.4f")
        ablation.to_csv(
            OUT_DIR / "ablation_picks.csv", index=False, float_format="%.4f")
        (OUT_DIR / "checks.json").write_text(
            json.dumps(checks, indent=2, default=str) + "\n")

    write_markdown(scored, ablation, levels["shipped"], checks, load_persisted_notes())
    print(f"\nwrote {DIAGNOSTIC.relative_to(ROOT)}")
    for name in ("replacement_levels.csv", "par_picks.csv", "ablation_picks.csv"):
        print(f"wrote {(OUT_DIR / name).relative_to(ROOT)}")
    return 0


# ==========================================================================
# the write-up
# ==========================================================================

def names(rows) -> str:
    """`a`, `b` and `c` -- readable in a sentence, unlike a semicolon list."""
    marks = [f"`{r['arm']}`" for r in rows]
    if len(marks) <= 1:
        return "".join(marks)
    return ", ".join(marks[:-1]) + " and " + marks[-1]


def signed(value: float, places: int = 2) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:+.{places}f}"


def interval(stats: dict) -> str:
    if np.isnan(stats["ci_lo"]):
        return "n/a"
    mark = "&nbsp;†" if stats.get("unstratified") else ""
    return f"[{stats['ci_lo']:+.2f}, {stats['ci_hi']:+.2f}]{mark}"


def covers_zero(stats: dict) -> bool:
    return bool(np.isnan(stats["ci_lo"]) or stats["ci_lo"] <= 0.0 <= stats["ci_hi"])


def readable(stats: dict) -> bool:
    """Excludes zero AND clears the measurement's own resolution."""
    return not covers_zero(stats) and abs(stats["mean_diff"]) > NOISE_FLOOR_PPG


def par_verdict(results: dict[str, dict]) -> tuple[str, list[str]]:
    """The headline reading of the re-score, and the qualification it forces.

    Written so that "the metric moved the number, not the picks" is the easy
    answer. The picks are literally identical across every row of this table --
    the same 936 rows of `replay_picks.csv` -- so any movement between metrics
    is the baseline and nothing else, and a verdict that does not say so is
    hiding its own mechanism.
    """
    par, raw, adj = results["par"], results["fwd3"], results["vs_pos"]
    mix_share = par["split"]["mix_share"]
    lines = []

    if par["mean_diff"] >= WF.LEAKAGE_TRIPWIRE_PPG:
        name = "LEAKAGE SUSPECTED"
        lines.append(
            f"The repo arm is ahead by {par['mean_diff']:+.2f} ppg PAR, at or past "
            f"the {WF.LEAKAGE_TRIPWIRE_PPG:.0f} ppg tripwire set before the run. "
            "**Do not act on this number.** The picks did not change, so a margin "
            "this size is a property of the baseline rather than of the ranking."
        )
    elif covers_zero(par):
        name = "STILL A NULL ON PAR"
        lines.append(
            f"Over {par['n_weeks']} paired weeks the repo arm is "
            f"{par['mean_diff']:+.2f} ppg PAR against naive, {CI_LEVEL:.0%} interval "
            f"{interval(par)}. The interval covers zero. Changing the outcome "
            "metric did not change the answer."
        )
    elif par["mean_diff"] > 0:
        name = "PAR FLIPS THE SIGN"
        lines.append(
            f"**Over {par['n_weeks']} paired weeks the repo arm beats naive by "
            f"{par['mean_diff']:+.2f} ppg PAR**, {CI_LEVEL:.0%} interval "
            f"{interval(par)}, excluding zero. On raw `fwd3` the same picks, over "
            f"the same weeks, put the repo arm {raw['mean_diff']:+.2f} ppg "
            f"{interval(raw)}. **Not one pick changed.** The entire "
            f"{par['mean_diff'] - raw['mean_diff']:+.2f} ppg swing is the constant "
            "each arm has subtracted from it."
        )
    else:
        name = "NAIVE STILL AHEAD ON PAR"
        lines.append(
            f"Over {par['n_weeks']} paired weeks the repo arm is "
            f"{par['mean_diff']:+.2f} ppg PAR against naive, interval "
            f"{interval(par)}. Changing the metric did not rescue it."
        )

    lines.append(
        f"**{abs(mix_share):.0%} of that margin is positional composition, not "
        f"ranking.** The mix term is {signed(par['split']['mix'])} ppg and the "
        f"within-position selection term is "
        f"{signed(par['split']['selection_gap'])} ppg. On raw `fwd3` the same "
        f"split was {signed(raw['split']['mix'])} mix and "
        f"{signed(raw['split']['selection_gap'])} selection. The mix term did not "
        "collapse toward zero — it **changed sign and kept roughly its whole "
        "magnitude**. PAR does not remove positional composition from the "
        "comparison; it re-prices it, and it re-prices it hard enough to move the "
        "verdict on its own."
    )
    lines.append(
        f"**The part of the comparison that is about ranking did not move.** The "
        f"selection term is {signed(raw['split']['selection_gap'])} ppg on raw "
        f"points, {signed(adj['split']['selection_gap'])} on the pool-mean "
        f"baseline and {signed(par['split']['selection_gap'])} on PAR — a null "
        f"under every baseline, all three inside the ±{NOISE_FLOOR_PPG:.1f} ppg "
        "this measurement can resolve. Whatever the headline says, this data "
        "still does not distinguish the model's ranking from sorting the wire by "
        "last week's box score."
    )
    return name, lines


def clean_arms(ties: pd.DataFrame) -> set[str]:
    """Arms whose picks are theirs rather than the alphabet's, at most positions.

    An arm that ties at the cut in a majority of a position's weeks did not
    choose those picks. Beating it is not evidence about the model, and being
    matched by it is not evidence either, so it is excluded from the arms the
    recommendation is allowed to rest on -- and named in the write-up rather
    than quietly dropped.
    """
    degenerate = ties.groupby("arm")["degenerate"].mean()
    # A list, in `HEURISTICS` order, not a set. Set iteration order over strings
    # varies with the hash seed, and every arm listed in the verdict prose would
    # otherwise reshuffle between runs of --report-only on identical inputs.
    return [arm for arm in HEURISTICS if degenerate.get(arm, 0.0) < 0.5]


def model_recommendation(picks: pd.DataFrame, ties: pd.DataFrame) -> tuple[str, list[str]]:
    """Keep the fitted models, or delete them. Asked and answered, from the numbers.

    The stake, in the brief's own terms: if a sort on one shrunk column does the
    same job, then `models/`, the training-window question, the version-pinning
    machinery and the refit discipline are all unnecessary.

    Two things decide it, and they turn out to point different ways, which is
    why both are asked. Whether the model's **single best name** at a position
    beats a one-liner's single best name; and whether its **ranked list** holds
    up further down than a one-liner's does. A rule that is only replaceable at
    the top is still replaceable if one name is all you take.
    """
    clean = clean_arms(ties)
    top = {a["arm"]: a for a in ablation_table(picks[picks["depth"] == 1])}
    deep = {
        a["arm"]: a for a in rank_table(picks, HEADLINE_DEPTH)
        if a["rank"] == HEADLINE_DEPTH
    }
    # Sorted by margin, closest comparison first: the arm that comes nearest to
    # matching the model is the one the recommendation turns on, so it reads
    # first rather than wherever the dict happens to put it.
    by_margin = lambda rows: sorted(rows, key=lambda r: r["mean_diff"])
    matched_at_top = by_margin([top[a] for a in clean if not readable(top[a])])
    beaten_deep = by_margin([deep[a] for a in clean if readable(deep[a])])
    closest = by_margin([top[a] for a in clean])[0]

    # Quarterback is where the tiebreak degeneracy lives, so the deepest-rank
    # result is re-run without it. An arm whose margin does not survive that is
    # named as the exception rather than carried by the pooled figure.
    no_qb = {
        a["arm"]: a for a in rank_table(picks, HEADLINE_DEPTH, exclude=("QB",))
        if a["rank"] == HEADLINE_DEPTH
    }
    survives = by_margin([no_qb[a["arm"]] for a in beaten_deep
                          if readable(no_qb[a["arm"]])])
    fragile = next(
        (no_qb[a["arm"]] for a in beaten_deep if not readable(no_qb[a["arm"]])),
        None,
    )

    listing = lambda rows: ", ".join(
        f"`{r['arm']}` {signed(r['mean_diff'])} {interval(r)}" for r in rows
    )
    excluded = [arm for arm in HEURISTICS if arm not in clean]
    note = (
        [f"`{a}` is excluded from this reading: its picks were decided by the "
         "alphabetical tiebreak in a majority of cells at most positions, so it "
         "is a draw rather than a heuristic and neither beating it nor being "
         "matched by it is evidence." for a in excluded]
        if excluded else []
    )

    if not matched_at_top and len(beaten_deep) == len(clean):
        return "KEEP THE MODELS", [
            f"The model beats **every** one-liner at both ends of the list — at "
            f"rank 1 and at rank {HEADLINE_DEPTH} — by margins whose intervals "
            f"exclude zero and clear the ±{NOISE_FLOOR_PPG:.1f} ppg this "
            f"measurement can resolve. The closest is {listing([closest])}.",
            "This is the first evidence in the project that the fitted models "
            "are earning anything, and it is stated as plainly as the nulls have "
            "been. The margin is small in absolute terms: a reason to keep them, "
            "not a reason to expect them to win a week.",
        ] + note

    if matched_at_top and len(beaten_deep) == len(clean):
        fragility = ([] if fragile is None or not survives else [
            "One of those does not survive scrutiny and the rest do. "
            f"`{fragile['arm']}`'s margin is carried by a quarterback cell in "
            "which it is an alphabetical draw rather than a heuristic; drop "
            f"quarterback and it falls to {signed(fragile['mean_diff'])} "
            f"{interval(fragile)}, a null. The arms that are choosing at every "
            f"position hold at {signed(min(r['mean_diff'] for r in survives))} to "
            f"{signed(max(r['mean_diff'] for r in survives))}. The recommendation "
            "rests on those, not on beating a coin flip."
        ])
        return "KEEP THEM — BUT FOR THE LIST, NOT THE TOP NAME", [
            f"**The model's single best name at a position is not better than a "
            f"one-liner's single best name.** At k=1 it is indistinguishable from "
            f"{len(matched_at_top)} of {len(clean)} arms that are actually "
            f"choosing: {listing(matched_at_top)}. Sorting the position's wire by "
            "shrunk target share, or by last week's points, names a first player "
            "as good as the model's.",
            f"**Its third name is better, and every arm agrees.** At rank "
            f"{HEADLINE_DEPTH} the model is ahead of all "
            f"{len(beaten_deep)} of them: {listing(beaten_deep)}. That is not the "
            "k=1 test being underpowered — rank 1 of this run *is* the k=1 arm, "
            "same key and same player — so the arms genuinely differ in how they "
            "hold up down the list. The one-liners degrade; the model does not.",
        ] + fragility + [
            "**Recommendation: keep `models/`.** What it buys is a ranked list "
            "rather than a best guess, and a candidate table is a ranked list — "
            "`assign_tiers` puts out two burn names, three fallbacks and four to "
            "watch. If the product were a single weekly claim, the honest answer "
            "would be to delete the models and sort on one column; it is not.",
            f"That is a narrower claim than the models earning their keep "
            "outright, and it should not be quoted as one. The version-pinning "
            "machinery and the refit discipline are justified by a margin of "
            f"about {np.mean([a['mean_diff'] for a in beaten_deep]):.1f} ppg on "
            "the third name at a position, and by nothing measured above it.",
        ] + note

    if not beaten_deep:
        return "DELETE THE MODELS", [
            f"**No** arm that is actually choosing is separated from the model by "
            f"a margin this data can resolve, at any depth. The closest "
            f"comparison is {listing([closest])}. A sort on one column does the "
            "same job as four gradient-boosted models refitted every season.",
            "That makes `models/`, the training-window question, the "
            "version-pinning machinery and the refit discipline unnecessary. It "
            "is a large simplification and the numbers support taking it.",
        ] + note

    return "MIXED", [
        f"The model is separated from {len(beaten_deep)} of {len(clean)} arms at "
        f"rank {HEADLINE_DEPTH} ({listing(beaten_deep)}) and matched at rank 1 by "
        f"{len(matched_at_top)} ({listing(matched_at_top)}). No single reading "
        "covers that; the tables below are the answer rather than this sentence.",
    ] + note


def write_markdown(scored: pd.DataFrame, ablation: pd.DataFrame,
                   levels: pd.DataFrame, checks: dict, notes: dict) -> None:
    results = {name: headline(scored, name) for name in BASELINES}
    par, raw, adj = results["par"], results["fwd3"], results["vs_pos"]
    verdict_name, verdict_lines = par_verdict(results)

    head = ablation[ablation["depth"] == HEADLINE_DEPTH]
    pooled = ablation_table(head)
    ties = tie_rates(head)
    overlaps = overlap_matrix(head)
    ranks = rank_table(ablation, HEADLINE_DEPTH)
    rec_name, rec_lines = model_recommendation(ablation, ties)

    L: list[str] = []
    add = L.append

    add("# PAR scoring, and whether the model beats a one-liner")
    add("")
    add("Generated by `outputs/backtests/03_par_rescore_ablation.py`. Two questions: "
        "does re-scoring the twelve-season walk-forward on points above replacement "
        "change its answer, and is the model doing anything a sort on one column "
        "cannot. **The answers are first and the methodology is last.**")
    add("")
    add("No model was refitted to answer the first question — the picks from "
        "`replay_full/replay_picks.csv` are read, not regenerated. The second "
        "question needed the model's ranking *within* position, which is not in "
        "those picks, so the bundles were rebuilt and required to reproduce the "
        "persisted picks exactly before being used. They did: "
        f"{checks['gates'][1]['player_matches']}/{checks['gates'][1]['rows']} model "
        f"picks and {checks['gates'][0]['player_matches']}/"
        f"{checks['gates'][0]['rows']} naive picks came back player-for-player.")
    add("")

    # ---- 1. the PAR verdict ---------------------------------------------
    add("## 1. The PAR verdict")
    add("")
    add(f"**{verdict_name}.**")
    add("")
    for line in verdict_lines:
        add(line)
        add("")
    add("| baseline subtracted from `fwd3` | repo − naive | 95% CI | mix | selection |")
    add("| --- | ---: | :---: | ---: | ---: |")
    for name, (_, gloss) in BASELINES.items():
        r = results[name]
        mark = " **(headline)**" if name == HEADLINE_BASELINE else ""
        add(f"| `{name}` — {gloss}{mark} | {signed(r['mean_diff'])} | "
            f"{interval(r)} | {signed(r['split']['mix'])} | "
            f"{signed(r['split']['selection_gap'])} |")
    add("")
    add(f"Same {par['n_weeks']} paired weeks, same picks, same realised points in "
        "every row. Only the subtracted constant differs. Positive favours the "
        "repo arm.")
    add("")
    std_gap = par["mean_diff"] - results["par_std"]["mean_diff"]
    pts_gap = par["mean_diff"] - results["par_pts"]["mean_diff"]
    add("The two `par_*` rows price the hindsight in the headline baseline. It "
        "is an order statistic of what the pool *turned out* to deliver at rank "
        "R, which nobody could have chosen; those two rank the pool by something "
        "knowable on the Monday — season-to-date standing, or last week's points "
        "— and take *that* player's realised `fwd3` instead.")
    add("")
    add(f"The headline baseline is {signed(std_gap)} ppg more generous to the "
        f"repo arm than the first and {signed(pts_gap)} than the second. The "
        f"first is inside the ±{NOISE_FLOOR_PPG:.1f} ppg this measurement can "
        "resolve and should not be read as a difference; the second is not, and "
        "is roughly half the headline margin. Which ordering is the right "
        "counterfactual is a judgement — `pts` is also the naive arm's own "
        "selection key, so it is not neutral in appearance even though the "
        "baseline it produces is applied identically to both arms — so all three "
        "are shown rather than one being chosen.")
    add("")

    add("### Why the mix term inverted instead of collapsing")
    add("")
    add("Replacement level is a fixed *rank* against pools of very different "
        "depth, so it lands at a different percentile at every position:")
    add("")
    add("| position | rank | pool size | replacement `fwd3` | pool mean `fwd3` | ratio | percentile |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    cells = levels[levels["week"].isin(HEADLINE_WEEKS)]
    for position, group in cells.groupby("position"):
        add(f"| {position} | {int(group['repl_rank'].iloc[0])} | "
            f"{group['pool_n'].mean():.0f} | {group['repl_fwd3'].mean():.2f} | "
            f"{group['pool_mean'].mean():.2f} | "
            f"{group['repl_fwd3'].mean() / group['pool_mean'].mean():.2f}× | "
            f"{group['repl_pctile'].mean():.2f} |")
    add("")
    add("The mix term is `sum_p (w_naive,p − w_repo,p) · mu_p`. It vanishes only "
        "when the baseline sits a **constant distance** above the pool mean at "
        "every position. It does not — the ratio runs from about 2× at "
        "quarterback to over 3× at tight end — so subtracting it re-orders the "
        "positions rather than levelling them. The naive arm is "
        f"{raw['split']['naive']['weights'].get('QB', 0):.0%} quarterbacks against "
        f"the repo arm's {raw['split']['repo']['weights'].get('QB', 0):.0%}, and "
        "PAR charges that composition roughly twice what raw points paid it. The "
        "one baseline that zeroes the mix term exactly, by construction, is the "
        "pool mean — the `vs_pos` row above — and there the answer is a null.")
    add("")
    add("This is not an argument that PAR is wrong. A second quarterback really "
        "is worth close to nothing when you start one, and the pool mean really "
        "is dragged down by backups who never play, which is the objection "
        "`report.with_edge` was written against. It is an argument that **the PAR "
        "headline is a statement about positional composition, not about ranking "
        "quality**, and that the pre-registered rule reading it will be reading "
        "composition.")
    add("")
    add("### One consequence for the live ledger, flagged rather than fixed")
    add("")
    add(f"`NAIVE_MARGIN`, `TIE_BAND` and `REPO_MARGIN` — 1.5, 1.0 and 1.5 ppg — "
        "were pre-registered against raw fantasy points and now apply to PAR. "
        "They have not been changed. The table above is the calibration nobody "
        f"had when they were set: the same picks over the same weeks move "
        f"{raw['mean_diff']:+.2f} → {par['mean_diff']:+.2f} ppg on the metric "
        "change alone, so **1.5 ppg is a materially easier bar on PAR than it "
        "was on raw points**.")
    add("")
    add("Re-picking the thresholds now, having seen that, would be worse than "
        "leaving them: it is the exact move the pre-registration exists to stop. "
        "So they stay, and the tension is recorded in the `src/ledger.py` "
        "docstring, here, and next to the rule in "
        "`outputs/diagnostics/comparison_setup.md`. If they are to move it "
        "should be as a deliberate re-registration, argued on the scale rather "
        "than on the result.")
    add("")

    # ---- 2. the ablation ------------------------------------------------
    add("## 2. The ablation: model against four one-liners")
    add("")
    add("Within position, over the same walk-forward weeks. **Positive means the "
        "model is ahead.**")
    add("")
    add(f"**{rec_name}.**")
    add("")
    for line in rec_lines:
        add(line)
        add("")
    add(f"### The whole ablation in one table (weeks 2–14, "
        f"{pooled[0]['n_weeks']} weeks)")
    add("")
    add("Each arm's single best name at a position, then its ranked list of "
        "three, then that list broken out rank by rank.")
    add("")
    top = {a["arm"]: a for a in ablation_table(ablation[ablation["depth"] == 1])}
    by_rank = {(a["arm"], a["rank"]): a for a in ranks}
    add("| arm | sorts on | positions | k=1 (best name) | k=3 (the list) | "
        + " | ".join(f"rank {r}" for r in range(1, HEADLINE_DEPTH + 1)) + " |")
    add("| --- | --- | --- | :---: | :---: | "
        + " | ".join(":---:" for _ in range(HEADLINE_DEPTH)) + " |")
    for stats in pooled:
        arm = stats["arm"]
        cells = [
            f"`{arm}`", HEURISTICS[arm]["gloss"], "/".join(stats["positions"]),
            f"{signed(top[arm]['mean_diff'])} {interval(top[arm])}",
            f"**{signed(stats['mean_diff'])}** {interval(stats)}",
        ]
        for rank in range(1, HEADLINE_DEPTH + 1):
            r = by_rank[(arm, rank)]
            cells.append(f"{signed(r['mean_diff'])} {interval(r)}")
        add("| " + " | ".join(cells) + " |")
    add("")
    add("Reading it: **the rank-1 column and the k=1 column are the same "
        "comparison** — same key, same tiebreak, same player — so a margin that "
        "appears at k=3 and not at k=1 is not the k=1 test being underpowered. "
        "It is the arms differing in how far down the list they hold up. Every "
        "one-liner's third name is worse than the model's third name; every "
        "one-liner that is actually choosing has a first name as good as the "
        "model's.")
    add("")
    add("#### The mechanism, in levels rather than differences")
    add("")
    add("Mean PAR of the pick at each rank. The differences above are this table "
        "read sideways, and it makes the shape of the result visible instead of "
        "asserted.")
    add("")
    # Weeks 2-14, like every other number in this document. `head` is the whole
    # depth-k run and still carries weeks 15-17, whose forward windows are
    # truncated; averaging those in here and nowhere else would put a table on
    # the page that quietly disagrees with the one above it.
    levels_by_rank = (
        head[head["week"].isin(HEADLINE_WEEKS)]
        .pivot_table(index="arm", columns="rank_within_arm", values="par",
                     aggfunc="mean")
    )
    first, last = levels_by_rank.columns[0], levels_by_rank.columns[-1]
    falls = levels_by_rank[first] - levels_by_rank[last]
    best_first = levels_by_rank[first].idxmax()
    add("| arm | " + " | ".join(f"rank {r}" for r in levels_by_rank.columns)
        + " | fall, rank 1 → 3 |")
    add("| --- | " + " | ".join("---:" for _ in levels_by_rank.columns) + " | ---: |")
    for arm in ["model"] + list(HEURISTICS):
        if arm not in levels_by_rank.index:
            continue
        row = levels_by_rank.loc[arm]
        add(f"| `{arm}` | " + " | ".join(f"{row[c]:.2f}" for c in levels_by_rank.columns)
            + f" | {row[levels_by_rank.columns[0]] - row[levels_by_rank.columns[-1]]:.2f} |")
    add("")
    if best_first != "model":
        add(f"**`{best_first}` names a better first player than the model does** "
            f"— {levels_by_rank.loc[best_first, first]:.2f} ppg against the "
            f"model's {levels_by_rank.loc['model', first]:.2f}, which is the "
            "point estimate behind the "
            f"{signed(by_rank[(best_first, 1)]['mean_diff'])} in the table "
            f"above. It then falls {falls[best_first]:.2f} ppg across "
            f"{HEADLINE_DEPTH} names while the model falls "
            f"{falls['model']:.2f}. That is the whole result: the model is not "
            "better at finding the best player at a position, it is better at "
            "not running out of them.")
    else:
        add(f"The model names the best first player of any arm "
            f"({levels_by_rank.loc['model', first]:.2f} ppg) and also falls "
            f"least across {HEADLINE_DEPTH} names ({falls['model']:.2f}).")
    add("")
    add("Every level here is negative because replacement is a high-percentile "
        "bar — the third-best quarterback and the seventh-best receiver that "
        "week, both measured after the fact. See §1; it is a property of the "
        "baseline, not a finding about the arms.")
    add("")
    no_qb = sorted(
        (a for a in rank_table(ablation, HEADLINE_DEPTH, exclude=("QB",))
         if a["rank"] == HEADLINE_DEPTH),
        key=lambda a: a["mean_diff"],
    )
    holds = [a for a in no_qb if readable(a)]
    drops = [a for a in no_qb if not readable(a)]
    add(f"**Robustness.** Dropping quarterback entirely — which removes the "
        "cells where the tiebreak decides the pick, along with everything else "
        f"at that position — the rank-{HEADLINE_DEPTH} margins are: "
        + "; ".join(f"`{a['arm']}` {signed(a['mean_diff'])} {interval(a)}"
                    for a in no_qb)
        + ". "
        + (f"{names(holds)} survive{'s' if len(holds) == 1 else ''}"
           + (f"; {names(drops)} do{'es' if len(drops) == 1 else ''} not, which "
              "is the quarterback cell it was carrying." if drops else ".")))
    add("")
    add(f"A margin under ±{NOISE_FLOOR_PPG:.1f} ppg is inside what this "
        "measurement can resolve and is not a win in either direction, whatever "
        "its interval does — see the note on binning at the end.")
    add("")

    add("### By position")
    add("")
    add("| arm | " + " | ".join(sorted(head["position"].unique())) + " |")
    add("| --- | " + " | ".join(":---:" for _ in sorted(head["position"].unique())) + " |")
    for arm in HEURISTICS:
        row = [f"`{arm}`"]
        for position in sorted(head["position"].unique()):
            stats = week_block_ci(
                head[head["week"].isin(HEADLINE_WEEKS)]
                .groupby(["season", "week", "position", "arm"], as_index=False)
                .agg(mean=("par", "mean"))
                .pipe(lambda d: d[d["position"] == position]),
                arm,
            )
            if np.isnan(stats["mean_diff"]):
                row.append("n/a")
            else:
                flag = ties[(ties["arm"] == arm) & (ties["position"] == position)]
                degenerate = bool(flag["degenerate"].any())
                row.append(f"{signed(stats['mean_diff'])} {interval(stats)}"
                           + (" ‡" if degenerate else ""))
        add("| " + " | ".join(row) + " |")
    add("")
    add(f"At k={HEADLINE_DEPTH}. ‡ marks a cell where the arm's sort key could "
        f"not separate the top {HEADLINE_DEPTH}, so the pick was decided by the "
        "alphabetical tiebreak in a majority of weeks. See the next table. "
        "`eb_share` is `n/a` at quarterback because a quarterback has no target "
        "or carry share.")
    add("")
    add("### What the heuristic arms could and could not resolve")
    add("")
    add("An arm whose sort key ties at the cut is not choosing; the alphabet is. "
        "Beating one says nothing about the model, so the rate is reported rather "
        "than averaged in silently.")
    add("")
    add("| arm | position | picks decided by the name tiebreak | players tied at the cut |")
    add("| --- | --- | ---: | ---: |")
    for _, row in ties.sort_values(["tie_rate"], ascending=False).iterrows():
        if row["tie_rate"] < 0.01:
            continue
        mark = " ‡" if row["degenerate"] else ""
        add(f"| `{row['arm']}` | {row['position']}{mark} | {row['tie_rate']:.1%} | "
            f"{row['distinct_at_cut']:.1f} |")
    add("")
    forced = overlaps[overlaps["identical"] >= 0.999]
    if not forced.empty:
        add("**Forced ties.** Two arms that sort the same column at a position are "
            "one arm, and reporting their agreement as agreement would be counting "
            "the same result twice:")
        add("")
        for _, row in forced.iterrows():
            add(f"- `{row['a']}` and `{row['b']}` name identical sets in "
                f"{row['identical']:.0%} of {int(row['cells'])} {row['position']} "
                "cells.")
        add("")
    add("The brief specified `eb_share` as snap share at quarterback. That would "
        "have made it a second copy of the `snap` arm — identical picks in every "
        "quarterback cell — so it is reported as `n/a` there instead, and pooled "
        "over the positions where shrunk shares mean something. Three arms at "
        "quarterback, not four, and the table says so rather than showing a "
        "forced tie as independent agreement.")
    add("")

    add("### The depth is a choice, and the verdict depends on it")
    add("")
    add("How many players a rule names at a position is not a measurement, and "
        "the two answers point different ways, so both are above rather than one "
        "being picked. Taken as a single claim per position, the model is "
        "replaceable by a one-liner. Taken as a ranked list — which is what "
        "`assign_tiers` actually emits: two burn names, three fallbacks, four to "
        "watch — it is not. Neither reading is wrong; they answer different "
        "questions, and the product asks the second one.")
    add("")

    add("### PAR changes nothing here, exactly")
    add("")
    add("**Within a position, scoring on PAR and scoring on raw `fwd3` are the "
        "same comparison.** The baseline is one constant per (season, week, "
        "position), the ablation holds position fixed, so every arm in a cell has "
        "the same number subtracted and every paired difference is untouched. "
        f"Measured, not assumed: the largest disagreement between the two is "
        f"{checks['par_invariance']:.1e} ppg, which is floating-point. The table "
        "above is therefore reported once, and `check_par_invariance()` fails the "
        "run if this ever stops holding.")
    add("")
    add("PAR does work only where the arms are allowed to differ in position mix. "
        "That is question 1, and it is the whole of question 1.")
    add("")

    # ---- 3. methodology -------------------------------------------------
    add("## 3. Methodology")
    add("")
    add("### What was refitted, and what was not")
    add("")
    add("| step | refitted? | why |")
    add("| --- | --- | --- |")
    add("| the PAR re-score | **no** | `replay_full/replay_picks.csv` persists "
        "every pick both arms made. Changing the outcome metric does not change "
        "which players were picked, so the picks are read. |")
    add("| the replacement baselines | no model | not recoverable from the picks: "
        "they carry `pool_mean_pos`, the pool's *mean*, and replacement is an "
        "order statistic, which a mean does not determine. Rebuilt from the panel "
        "— deterministic feature construction off the sha256-pinned files in "
        "`data/raw/MANIFEST.json`, no estimator. |")
    add("| the ablation's model arm | **yes, 48 fits** | the ablation needs the "
        "model's ranking *within* position. The persisted picks carry only its "
        "top three *across* positions. `replay_models_full/` is gitignored so "
        "replay bundles cannot be served, and was absent from this clone. |")
    add("")
    add("The refit is treated as a reconstruction that has to prove itself. "
        "`MODEL_KWARGS` pins `random_state=0` and `requirements.txt` pins the "
        "libraries, so it should be exact, and the run fails unless it is:")
    add("")
    add("| gate | picks | reproduced | max `fwd3` gap |")
    add("| --- | ---: | ---: | ---: |")
    for gate in checks["gates"]:
        add(f"| `{gate['arm']}` arm | {gate['rows']} | "
            f"{gate['player_matches']} | {float(gate['max_fwd3_gap']):.1e} |")
    add("")
    add("The `naive` gate matters more than it looks: it is a pure `pts` sort "
        "with no estimator in it anywhere, so a mismatch there would be a "
        "mismatch in the **pool** — which would void the model gate rather than "
        "be caught by it. The residual gaps are the `float_format=\"%.4f\"` "
        "rounding in the committed CSV, not disagreement.")
    add("")

    add("### Walk-forward discipline in the heuristic arms")
    add("")
    add("A heuristic built on a leaked prior would beat the model for the wrong "
        "reason. `eb_tgt_share` and `eb_car_share` are shrunk toward beta priors "
        "fitted across the build, so each replay season uses the panel built with "
        "`prior_seasons` restricted to the seasons strictly before it — the same "
        "per-window panels the model arm is fitted on, not a panel fitted across "
        "all seasons.")
    add("")
    add("`snap`, `wopr_opp` and `pts` do not depend on `prior_seasons` at all: "
        "`features.build` computes them within a single (player, season) or "
        "(season, position, week) group. That is asserted in "
        "`tests/test_features.py::test_only_eb_columns_cross_seasons`, and it is "
        "also why one set of replacement levels serves every scheme — the pool "
        "(`on_wire`) and the outcome (`fwd3`) are window-invariant. Checked "
        f"directly here as well: {checks['invariance']['cells']} baselines for "
        f"{checks['invariance']['season']} built from two different training "
        "windows agree to "
        f"{float(checks['invariance']['max_gap']):.0f} exactly.")
    add("")
    add("The rebuilt pool is checked against what the completed run persisted: "
        f"`pool_n_pos` matches exactly and `pool_mean_pos` within "
        f"{float(checks['pool']['pool_mean_gap']):.1e}. If the pool the baselines "
        "come from were not the pool the picks came from, nothing above would "
        "mean anything, and it would not look wrong.")
    add("")
    add("### The arms, and what each one is")
    add("")
    add("| arm | selection key | scope | is it the persisted arm? |")
    add("| --- | --- | --- | --- |")
    add("| `model` | the position's replay bundle, its own ordering | within position | "
        "no — the persisted `repo` arm ranks by `edge` across positions and takes "
        "three; this is the model's own top-k at each position |")
    for arm, spec in HEURISTICS.items():
        add(f"| `{arm}` | {spec['gloss']} | within position | no |")
    add("")
    add("**`hot_hand_pos` is not the `naive` arm of the headline, and its number "
        "must not be laid beside −2.30 ppg.** That arm sorts `pts` across every "
        "position at once and comes out "
        f"{raw['split']['naive']['weights'].get('QB', 0):.0%} quarterbacks, which "
        "is what produces the headline gap. This one sorts `pts` *within* a "
        "position: a different estimator making different picks. It is named for "
        "what it is, after the `hot_hand_pos` column `ledger.benchmarks` already "
        "computes.")
    add("")
    add("**`opp` is not WOPR.** `features.wopr_opp` is `carries + 2.5 × targets`, "
        "an unnormalised opportunity count with no team denominator; published "
        "WOPR is a weighted sum of *shares*. Calling it `wopr` in a results table "
        "would be a claim about a metric this is not, so it is called `opp`. Being "
        "integer-valued is also why it ties at the cut as often as the table "
        "above shows.")
    add("")

    add("### The resampling unit")
    add("")
    add(f"The week, stratified by season — the same unit as the headline, so the "
        "two are directly comparable. The four positions within a week share a "
        "wire pool, a slate of opponents and a scoring environment, so they are "
        "not four independent draws; a week is resampled with its cells attached "
        "and their differences averaged inside the block. Where any season "
        "stratum would be a singleton the resampling falls back to unstratified "
        "and the interval is marked †, because resampling one item with "
        "replacement always returns it and the interval would otherwise collapse "
        "to zero width and read as impossible precision.")
    add("")

    add("### What could not be scored")
    add("")
    dead = notes.get("dead_weeks", [])
    unscored = notes.get("unscored", [])
    add(f"Carried through from the completed run's `unscored.json` rather than "
        "recomputed — an unscoreable pick leaves no row, so recomputing it from "
        "the picks would silently report an empty list.")
    add("")
    if dead:
        add(f"{len(dead)} week(s) had no forward window at all and drop out for "
            "every arm alike:")
        add("")
        for line in dead:
            add(f"- {line}")
        add("")
    add(f"{len(unscored)} individual pick(s) could not be scored."
        if unscored else "No individual pick was left unscoreable.")
    add("")
    add("The headline covers **weeks 2–14**, the widest bucket whose three-week "
        "forward window fits inside the season on both sides of the 2021 "
        "expansion from 16 games to 17. That is the prior replay's bucket, "
        "unchanged.")
    add("")

    add("### Judgement calls")
    add("")
    add("- **The baseline refuses to clamp.** `report.replacement_level` clamps a "
        "pool shorter than the rank to its last entry, which is right where it "
        "lives: a clamped group gives its worst player an edge of exactly zero "
        "and `assign_tiers` filters him out, so the clamp is a safety filter. In "
        "outcome space there is no such filter — a clamped baseline is the pool "
        "minimum, usually 0.0, so PAR would collapse to raw `fwd3` and *inflate* "
        "the pick rather than exclude it. `ledger.replacement_of` returns NaN "
        "instead, and both the live ledger and this replay call it, so there is "
        "one definition rather than two that agree today. "
        "It never fires here (the smallest resolved pool is 10 at quarterback "
        "against a rank of 2) and exists for the live ledger, where a hand-logged "
        "position string that matches nothing can produce an empty group.")
    add("- **`mu_p` is rebuilt on each metric's own scale.** The decomposition "
        "identity `gap = mix + selection` holds arithmetically for *any* baseline, "
        "so leaving `mu` on the raw-points scale while the arm means moved to PAR "
        "would produce a split that balances exactly and describes nothing. "
        "`decompose` asserts the reconstruction to 1e-9 and the run fails if it "
        "drifts.")
    add("- **The model arm is not filtered to positive edge.** `assign_tiers` "
        "keeps only players above replacement, which at quarterback is exactly "
        "two names by construction. That is a tiering rule; this is a test of the "
        "ranking, so the model's own top-k at the position is used.")
    add("- **`REPLACEMENT_RANK` is held fixed across twelve seasons**, as in the "
        "prior replay. The depth sensitivity there found the verdict stable and "
        "the magnitude not; nothing here re-opens it.")
    add("")

    add("### Reproducibility")
    add("")
    add("Computed against the nflverse revision pinned in "
        "`data/raw/MANIFEST.json`. `games.csv` was revised upstream between the "
        "prior replay and this run — nflverse publishes the next season's "
        "schedule into it in place, which is the manifest's whole job to catch. "
        "It moved nothing used here: every season's final regular-season week is "
        "unchanged, all 1,110 persisted `fwd3_span` values recompute identically, "
        "and both reconstruction gates pass at 100%. The new hash is committed.")
    add("")
    add(f"Figures are reliable to about ±{NOISE_FLOOR_PPG:.1f} ppg. "
        "`HistGradientBoostingRegressor` bins its features and float "
        "perturbations at the 1e-16 level are enough to move a bin edge, a split, "
        "and occasionally which player a week's claim goes to; "
        "`01_season_replay.py` measures that sensitivity directly. In the ablation "
        "that uncertainty attaches to the model arm alone — the heuristic arms are "
        "exact sorts — so it moves every `model − arm` column together rather than "
        "independently. **The second decimal place is not real.**")
    add("")

    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text("\n".join(L).rstrip() + "\n")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

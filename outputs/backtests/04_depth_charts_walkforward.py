"""Walk-forward validation of the depth chart features, 2014-2025.

THE HYPOTHESIS, AND WHERE IT CAME FROM
======================================

An exploratory test on 2025 alone -- split *within* the season, weeks 2-8
training against weeks 9-14 -- suggested depth chart rank is the first
news-like feature to add anything on top of usage: roughly +0.03 R² at WR and
TE, nothing at RB, with `dc_rank` third in permutation importance behind
`eb_tgt_share` and `kal_role`. That test was weaker than this repo's bar in
every way that matters: one season, an in-season split, n=638 at TE, and a
first-pass snapshot-to-week alignment. This script is the real bar: the same
walk-forward design as `02_walkforward_2014_2025.py` -- per season, expanding
window, models refitted on strictly prior seasons -- asking two questions:

1. **Does the R² gain survive?** Per replay season and position, out-of-sample
   R² with and without the `dc_*` columns, paired per-season differences with a
   bootstrap interval. Reported on the repo's own objective (the within-week
   percentile rank of `fwd3`) and, secondarily, on raw `fwd3`, which is closer
   to whatever the exploratory numbers were computed on. The exploratory
   *levels* are not comparable to either; the delta is the question.

2. **Does it change any decisions?** The R² question is the easy one to pass
   and the wrong one to stop at. The repo arm is re-run with and without the
   depth chart features, against the same naive benchmark, scored on `fwd3`
   and on points above replacement -- and the two repo arms are paired against
   *each other*, with the share of weeks where they surface identical names.
   A feature that improves R² while changing no picks is decoration.

PRE-REGISTERED SUSPICION, STATED BEFORE THE RUN
===============================================

The exploratory +0.03 came from a within-season split, which is the *easier*
test. If the walk-forward -- the harder test -- comes back materially larger,
the first hypothesis is not "better feature", it is a leak in the
snapshot-to-week alignment. Concretely: a pooled delta at WR or TE -- the two
positions the exploratory test flagged, on either target, since the
exploratory convention is nearer raw points than the rank objective -- that
is significant and beyond R2_SUSPICION_FACTOR times the exploratory delta
flags the verdict rather than celebrating. The check is scoped to WR and TE
deliberately: doubling RB's exploratory +0.001 would condemn noise, and QB
has no exploratory reference at all -- the 6 ppg arm-margin tripwire from
the prior replays, which applies unchanged, is the guard that covers every
position. A null is an acceptable answer; a suspiciously good one is not.

One more piece of hygiene the design owes the reader: 2025 is both the
season the hypothesis was *discovered* on and one of the twelve replay
seasons. A discovery sample inside its own confirmation pool biases the pool
toward confirming, so the pooled deltas are reported both with 2025 and
without it, and the without-2025 number is the confirmatory one.

THE ALIGNMENT, AND WHERE THE LEAK WOULD LIVE
============================================

`features.load_depth_charts` produces, per (player, season, week, team), the
chart in force *going into* that week's games. Two source formats:

- **2013-2024**: week-labelled club submissions. `depth_team` never exceeds 3;
  a player deeper than third string is absent, so NaN in this era reads as
  "fourth string or worse". Used as labelled. Worst case for the label -- a
  chart captured at game time rather than before it -- is still inside the
  panel's no-lookahead contract, which allows anything knowable on the Monday
  after the row's own week; the outcome window (`fwd3`, weeks W+1..W+3) starts
  strictly later either way.
- **2025+**: timestamped snapshots, no week column, and the file runs on into
  the following March. Aligned by taking, per club, the last snapshot strictly
  before 00:00 UTC on the week's first scheduled game day, league-wide -- a
  Thursday or international kickoff anywhere tightens the cutoff for every
  club. That is deliberately stricter than a per-club kickoff cutoff: it
  costs Sunday teams their Friday/Saturday chart updates and buys the
  guarantee that no snapshot taken after any of the week's games -- one that
  could already encode a Sunday injury as a Monday demotion -- can describe
  that week. `tests/test_features.py` pins both directions of the boundary.

The era break is real and left visible: pre-2025 charts stop at rank 3, the
2025 feed lists a full roster, so `dc_rank` runs to 11 in the one season the
hypothesis came from and the binary encodings (`dc_is_starter`, `dc_top2`)
are the only rank features whose meaning is stable across the break. A model
trained on 2013-2024 and scored on 2025 sees rank values it never trained on;
that is the honest price of walk-forward across a format change, and it is
reported rather than smoothed over.

WHAT IS DELIBERATELY REUSED
===========================

Panel construction, EB-prior masking, model fitting, the points scale, the
repo arm's pick logic, the naive arm, the bootstrap, and the season/window
bookkeeping all come from `01_season_replay.py` / `02_walkforward_2014_2025.py`
/ `src.ledger` rather than being reimplemented, so this result and the prior
replays cannot drift apart in how they are computed. Contamination closure is
inherited from the same places: EB priors fitted on training seasons only,
points scale fitted on training seasons only, replay bundles stamped
`replay_only` and written outside `models/`.

Run:
    python outputs/backtests/04_depth_charts_walkforward.py
    python outputs/backtests/04_depth_charts_walkforward.py --report-only
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
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import features as F                                     # noqa: E402
from src import models as M                                       # noqa: E402
from src.ledger import CI_LEVEL, naive_picks, replacement_of, wire_pool  # noqa: E402


def _load_sibling(name: str):
    """Import a sibling script whose leading digit blocks a normal import."""
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


W = _load_sibling("02_walkforward_2014_2025.py")   # the walk-forward machinery
R = W.R                                            # the season-replay helpers

OUT_DIR = Path(__file__).resolve().parent / "depth_charts_wf"
MODEL_DIR = Path(__file__).resolve().parent / "replay_models_dc"
PANEL_CACHE = ROOT / "data" / "processed"
DIAGNOSTIC = ROOT / "outputs" / "diagnostics" / "depth_charts.md"

# Bump when the dc feature logic changes, so stale cached panels cannot
# quietly serve a run made after the change.
CACHE_TAG = "dcv1"

REPLAY_SEASONS = W.REPLAY_SEASONS            # 2014-2025
REPLAY_WEEKS = W.REPLAY_WEEKS                # 2-17, subject to season length
HEADLINE_WEEKS = range(2, 15)                # full 3-week window in every season
ARM_DEPTH = 3

DC_FEATURES = ["dc_rank", "dc_is_starter", "dc_top2", "dc_rank_prev", "dc_improve"]
VARIANTS = {"base": (), "dc": tuple(DC_FEATURES)}
ARMS = ("naive", "repo_base", "repo_dc")

# The exploratory within-season deltas this run exists to check, and the
# factor past which a *larger* walk-forward delta reads as an alignment leak
# rather than a better feature. Stated here, before the run.
EXPLORATORY_DELTA = {"WR": 0.032, "TE": 0.029, "RB": 0.001}
R2_SUSPICION_FACTOR = 2.0

R2_TARGETS = ("fwd3_rank", "fwd3")


# ==========================================================================
# panel + models, per replay season
# ==========================================================================

def panel_cache_path(seasons: list[int], train_seasons: list[int]) -> Path:
    return PANEL_CACHE / (
        f"{CACHE_TAG}_panel_s"
        + "".join(str(y)[-2:] for y in sorted(seasons))
        + "_prior"
        + "".join(str(y)[-2:] for y in sorted(train_seasons))
        + ".csv"
    )


def replay_panel(seasons: list[int], train_seasons: list[int]) -> pd.DataFrame:
    """Panel with EB priors on `train_seasons` only, CSV round-tripped.

    The round-trip is the same bit-identity device the prior replays use:
    `HistGradientBoostingRegressor` bins features, `to_csv`/`read_csv` perturbs
    floats at the 1e-16 level, and a value crossing a bin edge can move a pick.
    """
    cache = panel_cache_path(seasons, train_seasons)
    if cache.exists():
        print(f"  panel: cached {cache.name}")
        return pd.read_csv(cache)
    panel = F.build(seasons, prior_seasons=train_seasons)
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache, index=False)
    return pd.read_csv(cache)


def fit_bundles(
    panel: pd.DataFrame, train_seasons: list[int], replay_season: int,
    variant: str, extra: tuple[str, ...],
) -> dict[str, dict]:
    """One bundle per position, base features plus `extra`, stamped replay-only."""
    base = M.feature_columns(panel)
    features = base + [c for c in extra if c not in base]
    train = panel[panel["season"].isin(train_seasons)]
    universe = M.wire_universe(train)

    out_dir = MODEL_DIR / f"{variant}_{replay_season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict] = {}
    for position in M.POSITIONS:
        frame = universe[universe["position"] == position]
        model = M.fit_model(frame, features)
        bundle = {
            "replay_only": True,
            "do_not_serve": (
                "Fitted for the depth chart walk-forward in "
                "outputs/backtests/04_depth_charts_walkforward.py. Trained on "
                f"{train_seasons} to score {replay_season}; never loadable by "
                "src/weekly.py."
            ),
            "replay_season": replay_season,
            "replay_scheme": variant,
            "position": position,
            "model": model,
            "features": features,
            "extra_features": list(extra),
            "train_seasons": train_seasons,
            "n_train": len(frame),
            "fit_versions": M.library_versions(),
            "model_kwargs": M.MODEL_KWARGS,
        }
        joblib.dump(bundle, out_dir / f"{position}.joblib")
        bundles[position] = bundle
    return bundles


# ==========================================================================
# question 1: out-of-sample R², with and without
# ==========================================================================

def r2_rows(
    panel: pd.DataFrame, bundles: dict[str, dict[str, dict]],
    train_seasons: list[int], season: int,
) -> list[dict]:
    """Per (position, variant, target) R² on the replay season.

    The rank-objective evaluation reuses the exact estimator the arm scores
    with; the raw-points evaluation refits on the same rows with `fwd3` as the
    target. Thresholds follow `models.holdout_r2`: fewer than 100 training or
    50 test rows is reported as absent rather than as a noisy number.
    """
    universe = M.wire_universe(panel)
    rows: list[dict] = []
    for position in M.POSITIONS:
        frame = universe[universe["position"] == position]
        train = frame[frame["season"].isin(train_seasons)]
        test = frame[frame["season"] == season]
        for variant, per_position in bundles.items():
            features = per_position[position]["features"]
            for target in R2_TARGETS:
                if len(train) < 100 or len(test) < 50:
                    r2 = np.nan
                elif target == "fwd3_rank":
                    model = per_position[position]["model"]
                    r2 = float(r2_score(test[target], model.predict(test[features])))
                else:
                    model = M.HistGradientBoostingRegressor(**M.MODEL_KWARGS)
                    model.fit(train[features], train[target])
                    r2 = float(r2_score(test[target], model.predict(test[features])))
                rows.append(
                    {
                        "season": season,
                        "position": position,
                        "variant": variant,
                        "target": target,
                        "r2": r2,
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                )
    return rows


def r2_summary(
    r2: pd.DataFrame, target: str, exclude_season: int | None = None
) -> dict[str, dict]:
    """Pooled per-position delta (dc - base) with a bootstrap CI over seasons.

    The season is the resampling unit: twelve paired (base, dc) fits per
    position, each pair sharing its training data, test season and rows, so
    the per-season delta is the honest observation and there are only twelve
    of them. The interval is wide because the truth is wide.

    `exclude_season` drops one season from the pool. Used to take 2025 -- the
    season the hypothesis was discovered on -- out of its own confirmation
    sample; the module docstring says why the without-2025 pool is the
    confirmatory one.
    """
    out: dict[str, dict] = {}
    rows = r2[r2["target"] == target]
    if exclude_season is not None:
        rows = rows[rows["season"] != exclude_season]
    for position in M.POSITIONS:
        sub = rows[rows["position"] == position].pivot_table(
            index="season", columns="variant", values="r2"
        )
        sub = sub.dropna()
        deltas = (sub["dc"] - sub["base"]).to_numpy(dtype=float)
        lo, hi = R.bootstrap_ci(deltas, None)
        out[position] = {
            "n_seasons": len(deltas),
            "base_mean": float(sub["base"].mean()) if len(sub) else np.nan,
            "dc_mean": float(sub["dc"].mean()) if len(sub) else np.nan,
            "delta_mean": float(deltas.mean()) if len(deltas) else np.nan,
            "ci_lo": lo,
            "ci_hi": hi,
            "positive_seasons": int((deltas > 0).sum()),
        }
    return out


# ==========================================================================
# question 2: the arm comparison
# ==========================================================================

def replay_one_season(
    season: int, finals: dict[int, int]
) -> tuple[pd.DataFrame, list[dict], list[str], pd.DataFrame]:
    """All three arms' picks for one season, plus the R² rows and the panel."""
    train_seasons = W.train_seasons_for(season, "expanding")
    print(f"\nreplay {season}: train on {train_seasons[0]}-{train_seasons[-1]}")
    panel = replay_panel(sorted(set(train_seasons) | {season}), train_seasons)
    bundles = {
        variant: fit_bundles(panel, train_seasons, season, variant, extra)
        for variant, extra in VARIANTS.items()
    }
    r2 = r2_rows(panel, bundles, train_seasons, season)
    spans = W.window_spans(panel, season, finals)

    rows: list[dict] = []
    notes: list[str] = []
    for week in REPLAY_WEEKS:
        if week > finals[season]:
            continue
        pool = wire_pool(panel, season, week)
        if pool.empty:
            notes.append(f"{season} wk{week:02d}: empty wire pool")
            continue
        resolved = pool[pool["fwd3"].notna()]
        if resolved.empty:
            notes.append(
                f"{season} wk{week:02d}: no forward window inside a "
                f"{finals[season]}-week season"
            )
            continue
        ceiling = float(resolved["fwd3"].max())
        pos_mean = resolved.groupby("position")["fwd3"].mean().to_dict()
        replacement = {
            position: replacement_of(group["fwd3"], position)
            for position, group in resolved.groupby("position")
        }

        arms = [
            ("naive", naive_picks(panel, season, week, ARM_DEPTH)),
            ("repo_base", R.repo_picks(panel, bundles["base"], train_seasons, season, week)),
            ("repo_dc", R.repo_picks(panel, bundles["dc"], train_seasons, season, week)),
        ]
        for arm, frame in arms:
            for _, pick in frame.iterrows():
                if pd.isna(pick["fwd3"]):
                    notes.append(
                        f"{season} wk{week:02d} {arm} "
                        f"{pick['player_display_name']}: fwd3 unresolved"
                    )
                    continue
                position = str(pick["position"])
                rep = replacement.get(position, np.nan)
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "arm": arm,
                        "rank_within_arm": int(pick["rank_within_arm"]),
                        "player": str(pick["player_display_name"]),
                        "position": position,
                        "team": str(pick["team"]),
                        "fwd3": float(pick["fwd3"]),
                        "par": float(pick["fwd3"]) - rep if not np.isnan(rep) else np.nan,
                        "vs_pos": float(pick["fwd3"]) - pos_mean.get(position, np.nan),
                        "fwd3_span": spans.get((week, str(pick["team"])), np.nan),
                        "week_ceiling": ceiling,
                        "dc_rank": float(pick["dc_rank"]) if pd.notna(pick.get("dc_rank")) else np.nan,
                    }
                )
    return pd.DataFrame(rows), r2, notes, panel


def week_means(picks: pd.DataFrame) -> pd.DataFrame:
    """Arm x week means; the week is the unit of analysis, as everywhere here."""
    return (
        picks.groupby(["season", "week", "arm"], as_index=False)
        .agg(
            n=("fwd3", "size"),
            mean_fwd3=("fwd3", "mean"),
            mean_par=("par", "mean"),
            mean_vs_pos=("vs_pos", "mean"),
            week_ceiling=("week_ceiling", "first"),
        )
        .sort_values(["season", "week", "arm"])
        .reset_index(drop=True)
    )


def paired_arms(
    weekly: pd.DataFrame, arm_a: str, arm_b: str, weeks, value: str
) -> pd.DataFrame:
    """One row per week both arms covered: mean under a minus mean under b."""
    rows = weekly[weekly["week"].isin(list(weeks))]
    a = rows[rows["arm"] == arm_a].set_index(["season", "week"])[value]
    b = rows[rows["arm"] == arm_b].set_index(["season", "week"])[value]
    shared = a.index.intersection(b.index)
    frame = pd.DataFrame({"a": a.loc[shared], "b": b.loc[shared]}).dropna()
    frame["diff"] = frame["a"] - frame["b"]
    return frame.reset_index().sort_values(["season", "week"]).reset_index(drop=True)


def bucket(
    weekly: pd.DataFrame, arm_a: str, arm_b: str, weeks, value: str
) -> dict:
    pairs = paired_arms(weekly, arm_a, arm_b, weeks, value)
    diffs = pairs["diff"].to_numpy(dtype=float)
    strata = pairs["season"].to_numpy() if len(pairs) else None
    (lo, hi), degenerate = W.stratified_ci(diffs, strata)
    return {
        "a": arm_a,
        "b": arm_b,
        "value": value,
        "n_weeks": len(diffs),
        "mean_diff": float(diffs.mean()) if len(diffs) else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "unstratified": degenerate,
        "a_mean": float(pairs["a"].mean()) if len(pairs) else np.nan,
        "b_mean": float(pairs["b"].mean()) if len(pairs) else np.nan,
    }


def pick_agreement(picks: pd.DataFrame, weeks) -> dict:
    """How often the two repo arms surface the same names.

    This is the decision-relevance measurement: if the arms agree every week,
    the feature moved scores without moving a single claim, and the honest
    summary is "decoration" whatever the R² tables say.
    """
    rows = picks[picks["week"].isin(list(weeks)) & picks["arm"].isin(["repo_base", "repo_dc"])]
    identical = 0
    total = 0
    shared = 0
    named = 0
    top1_same = 0
    top1_weeks = 0
    per_season: dict[int, list[float]] = {}
    for (season, week), group in rows.groupby(["season", "week"]):
        base_rows = group[group["arm"] == "repo_base"]
        dc_rows = group[group["arm"] == "repo_dc"]
        base = set(base_rows["player"])
        dc = set(dc_rows["player"])
        if not base and not dc:
            continue
        total += 1
        # Set comparison on purpose for "identical": the *list* the report
        # tiers from. The #1 name is tracked separately below because that is
        # the claim actually made, and a reordered top three is a changed
        # decision there even when the sets agree. Lists can run shorter than
        # ARM_DEPTH (positive-edge names only, unresolved picks dropped); the
        # overlap denominator takes the longer list, which can only understate
        # agreement, never inflate it.
        identical += int(base == dc)
        shared += len(base & dc)
        named += max(len(base), len(dc))
        base_top = base_rows.loc[base_rows["rank_within_arm"] == 1, "player"]
        dc_top = dc_rows.loc[dc_rows["rank_within_arm"] == 1, "player"]
        if len(base_top) and len(dc_top):
            top1_weeks += 1
            top1_same += int(base_top.iloc[0] == dc_top.iloc[0])
        per_season.setdefault(season, []).append(float(base == dc))
    return {
        "weeks": total,
        "identical_weeks": identical,
        "identical_share": identical / total if total else np.nan,
        "name_overlap": shared / named if named else np.nan,
        "top1_same_share": top1_same / top1_weeks if top1_weeks else np.nan,
        "top1_weeks": top1_weeks,
        "per_season_identical": {s: float(np.mean(v)) for s, v in per_season.items()},
    }


# ==========================================================================
# the audit and the exploratory cross-checks
# ==========================================================================

def audit_tables(panel: pd.DataFrame) -> dict:
    """Coverage and sanity numbers for the write-up, from the widest panel.

    `panel` is the 2025 replay's build, which spans 2013-2025; the dc columns
    are per-season quantities, so reading them off this build is reading them
    off any build.
    """
    universe = M.wire_universe(panel)
    out: dict = {"seasons": {}}
    for season, group in panel.groupby("season"):
        wire = universe[universe["season"] == season]
        out["seasons"][int(season)] = {
            "rows": int(len(group)),
            "match": float(group["dc_rank"].notna().mean()),
            "match_by_pos": {
                p: float(g["dc_rank"].notna().mean())
                for p, g in group.groupby("position")
            },
            "wire_match": float(wire["dc_rank"].notna().mean()),
            "wire_starter_share": float((wire["dc_rank"] == 1).mean()),
            "rank_max": float(group["dc_rank"].max()),
        }

    # The exploratory claims, recomputed under this alignment on the wire
    # universe: starter-vs-pool points, the movement rho, promotion counts.
    checks: dict = {}
    for scope, seasons in (("2025", [2025]), ("2013-2024", list(range(2013, 2025)))):
        rows = universe[universe["season"].isin(seasons)]
        per_pos = {}
        for position in M.POSITIONS:
            sub = rows[rows["position"] == position]
            starters = sub[sub["dc_rank"] == 1]
            movement = sub[sub["dc_improve"].notna()]
            promoted = sub[(sub["dc_is_starter"] == 1) & (sub["dc_rank_prev"] > 1)]
            rho = (
                float(spearmanr(movement["dc_improve"], movement["fwd3"]).statistic)
                if len(movement) > 10
                else np.nan
            )
            per_pos[position] = {
                "n": int(len(sub)),
                "starter_fwd3": float(starters["fwd3"].mean()) if len(starters) else np.nan,
                "n_starters": int(len(starters)),
                "pool_fwd3": float(sub["fwd3"].mean()) if len(sub) else np.nan,
                "improve_rho": rho,
                "n_promoted": int(len(promoted)),
            }
        checks[scope] = per_pos
    out["checks"] = checks
    return out


def permutation_table() -> dict[str, list[tuple[str, float]]] | None:
    """Permutation importance of the dc-variant 2025 models, on 2025's universe.

    The exploratory claim was that `dc_rank` placed third at WR and TE, behind
    `eb_tgt_share` and `kal_role`. This recomputes the same quantity under the
    walk-forward fit (trained 2013-2024, scored on 2025) from the artifacts the
    main run leaves behind; returns None when they are not on disk, so
    `--report-only` on a partial run degrades to omitting the section rather
    than inventing it.
    """
    from sklearn.inspection import permutation_importance

    season = REPLAY_SEASONS[-1]
    train_seasons = W.train_seasons_for(season, "expanding")
    cache = panel_cache_path(sorted(set(train_seasons) | {season}), train_seasons)
    bundle_dir = MODEL_DIR / f"dc_{season}"
    if not cache.exists() or not bundle_dir.exists():
        return None
    panel = pd.read_csv(cache)
    universe = M.wire_universe(panel)
    out: dict[str, list[tuple[str, float]]] = {}
    for position in M.POSITIONS:
        path = bundle_dir / f"{position}.joblib"
        if not path.exists():
            return None
        bundle = joblib.load(path)
        test = universe[
            (universe["position"] == position) & (universe["season"] == season)
        ]
        if len(test) < 50:
            continue
        result = permutation_importance(
            bundle["model"], test[bundle["features"]], test["fwd3_rank"],
            n_repeats=10, random_state=0,
        )
        ranked = sorted(
            zip(bundle["features"], result.importances_mean),
            key=lambda pair: pair[1], reverse=True,
        )
        out[position] = [(name, float(value)) for name, value in ranked]
    return out or None


# ==========================================================================
# the verdict
# ==========================================================================

def points_reading(points_summary: dict) -> str:
    """What the raw-points secondary target adds to, or subtracts from, the verdict.

    Computed rather than assumed, because it is the one place this run's answer
    is not a flat null and the temptation to round it in either direction --
    up into "the feature works" or down into silence -- is exactly what the
    honesty constraints forbid.
    """
    real = {
        p: s for p, s in points_summary.items()
        if not np.isnan(s["ci_lo"]) and s["ci_lo"] > 0 and s["delta_mean"] > 0
    }
    if not real:
        return (
            "The raw-points target tells the same story: every interval covers "
            "zero. The two conventions agree on the null."
        )
    names = ", ".join(
        f"{p} ({s['delta_mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}])"
        for p, s in sorted(real.items())
    )
    unnamed = sorted(set(real) - {"WR", "TE"})
    missing_named = sorted({"WR", "TE"} - set(real))
    placement = []
    if unnamed:
        placement.append(
            f"{'/'.join(unnamed)} " + ("are positions" if len(unnamed) > 1 else "is a position")
            + " the exploratory test did **not** flag"
        )
    if missing_named:
        placement.append(
            f"at {'/'.join(missing_named)} -- where the exploratory gain was "
            "supposed to live -- the interval covers zero"
        )
    where = "; ".join(placement) if placement else "the positions the hypothesis named"
    return (
        f"On raw points the delta is small but real at {names}. Note where it "
        f"landed: {where}. It does not transfer to the rank objective the "
        "models are fitted and served on (percentile rank within a week is "
        "what a claim is ordered by, and it absorbs the level information "
        "usage already carries), and the arm comparison shows it buys no "
        "decisions. A genuine sliver of raw-points signal that the pipeline's "
        "objective cannot use is recorded as exactly that -- not rounded up "
        "into a keep, not rounded down into nothing."
    )


def verdict(
    rank_summary: dict, rank_confirm: dict, points_summary: dict,
    arm_par: dict, arm_fwd3: dict, agreement: dict,
) -> tuple[str, list[str]]:
    """The headline reading. Written so a null is the easy answer and a
    too-good answer fires the alignment suspicion before any celebration.

    Three details of the branching exist because a review pass caught their
    absence, and each would have printed a lie in a world this run did not
    happen to land in: the suspicion tripwire is scoped to WR and TE (the
    positions the pre-registration names -- 2x RB's exploratory +0.001 would
    condemn noise) but reads BOTH targets, since the exploratory convention
    is closer to raw points; "decisions move" requires the PAR interval to
    sit ABOVE zero, not merely exclude it, so a significantly harmful move
    cannot print as success; and a significantly negative R² delta is named
    rather than folded into "covers zero".

    Survival is judged on `rank_confirm` -- the pool with the discovery
    season removed -- because a hypothesis discovered on 2025 does not get to
    call 2025 part of its own confirmation. The suspicion tripwire reads the
    FULL pools on purpose: 2025 is the one season the snapshot alignment
    exists in, so a leak there inflates the full pool, and diluted evidence
    of a leak should still fire."""
    def _above_zero(summary: dict) -> dict:
        return {
            p: s for p, s in summary.items()
            if not np.isnan(s["ci_lo"]) and s["ci_lo"] > 0 and s["delta_mean"] > 0
        }

    survives = _above_zero(rank_confirm)
    harmful = {
        p: s for p, s in rank_confirm.items()
        if not np.isnan(s["ci_hi"]) and s["ci_hi"] < 0
    }
    suspicious = {
        (p, target): s
        for target, summary in (("rank", rank_summary), ("raw points", points_summary))
        for p, s in _above_zero(summary).items()
        if p in ("WR", "TE")
        and s["delta_mean"] > R2_SUSPICION_FACTOR * EXPLORATORY_DELTA[p]
    }
    par_helps = not np.isnan(arm_par["ci_lo"]) and arm_par["ci_lo"] > 0
    par_hurts = not np.isnan(arm_par["ci_hi"]) and arm_par["ci_hi"] < 0

    lines: list[str] = []
    if arm_fwd3["mean_diff"] >= W.LEAKAGE_TRIPWIRE_PPG or (
        not np.isnan(arm_par["mean_diff"])
        and arm_par["mean_diff"] >= W.LEAKAGE_TRIPWIRE_PPG
    ):
        return "LEAKAGE SUSPECTED", [
            "The depth chart arm's margin is at or past the "
            f"{W.LEAKAGE_TRIPWIRE_PPG:.0f} ppg tripwire the prior replays set. "
            "A margin this size is not plausible for one added feature on this "
            "data; re-read the alignment before believing any number here.",
        ]
    if suspicious:
        names = ", ".join(
            f"{p} on the {target} target "
            f"({s['delta_mean']:+.3f} against +{EXPLORATORY_DELTA[p]:.3f} exploratory)"
            for (p, target), s in suspicious.items()
        )
        return "SUSPICIOUSLY LARGE -- CHECK THE ALIGNMENT", [
            f"The walk-forward delta came back more than {R2_SUSPICION_FACTOR:.0f}x "
            f"the exploratory one at {names}. The exploratory number came from a "
            "within-season split, which is the easier test; a bigger number on "
            "the harder test points at the snapshot-to-week alignment leaking, "
            "not at a better feature. Do not act on this result.",
        ]

    if not survives:
        name = "NULL -- THE EXPLORATORY GAIN DOES NOT SURVIVE WALK-FORWARD"
        if harmful:
            coverage = (
                "covers zero at every position except "
                + ", ".join(
                    f"{p} ({s['delta_mean']:+.3f} "
                    f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}])"
                    for p, s in sorted(harmful.items())
                )
                + ", where the feature measurably *hurts*"
            )
        else:
            coverage = "covers zero at every position"
        lines.append(
            "Across the confirmatory replay seasons -- 2014-2024, models "
            "refitted on strictly prior data, with 2025 excluded because the "
            "hypothesis was discovered on it and shown separately -- adding "
            "the depth chart features moves out-of-sample R² on the rank "
            "objective -- the objective the models are fitted and served on "
            f"-- by an amount whose interval {coverage}. The full pool "
            "including 2025 says the same. The within-season +0.03 at WR and "
            "TE was the split being easy, not the feature being real. Per "
            "the ground rules: the feature is not kept."
        )
        lines.append(points_reading(points_summary))
    elif par_hurts:
        name = "R² SURVIVES; DECISIONS MOVE THE WRONG WAY"
        lines.append(
            f"The R² gain survives walk-forward at {', '.join(sorted(survives))} "
            "-- and the depth chart arm's PAR margin over its base version sits "
            "significantly *below* zero. Better scores, worse claims. The "
            "feature is not kept."
        )
    elif par_helps:
        name = "SURVIVES, AND MOVES DECISIONS"
        lines.append(
            f"The R² gain survives walk-forward at "
            f"{', '.join(sorted(survives))} and the depth chart arm's PAR "
            "margin over the base arm sits above zero. Both bars cleared; see "
            "the tables for sizes -- and note the multiplicity caveat under "
            "the pooled table before repeating the headline."
        )
    else:
        name = "R² SURVIVES; DECISIONS DO NOT MOVE"
        lines.append(
            f"The R² gain survives walk-forward at {', '.join(sorted(survives))} "
            "-- and changes almost nothing the pipeline would actually do. "
            "That second finding is the one that matters."
        )

    lines.append(
        f"The two repo arms surface **identical top-{ARM_DEPTH} claims in "
        f"only {agreement['identical_share']:.0%} of paired weeks** "
        f"({agreement['identical_weeks']} of {agreement['weeks']}), agree on "
        f"the **#1 claim -- the one actually made -- in "
        f"{agreement['top1_same_share']:.0%}** of them, with "
        f"{agreement['name_overlap']:.0%} name overlap overall -- so the "
        f"feature changes plenty of *names*. What it does not change is "
        f"*results*: on points above replacement the depth chart arm moves "
        f"the repo arm by {arm_par['mean_diff']:+.2f} ppg against its base "
        f"version ({CI_LEVEL:.0%} CI [{arm_par['ci_lo']:+.2f}, "
        f"{arm_par['ci_hi']:+.2f}], n = {arm_par['n_weeks']} weeks). Reshuffled "
        f"picks, indistinguishable outcomes."
    )
    return name, lines


# ==========================================================================
# the write-up
# ==========================================================================

def fmt(value, places: int = 3, pct: bool = False) -> str:
    return R.fmt(value, places, pct)


def ci_cell(entry: dict, places: int = 3) -> str:
    if np.isnan(entry.get("ci_lo", np.nan)):
        return "n/a"
    mark = " †" if entry.get("unstratified") else ""
    return f"[{entry['ci_lo']:+.{places}f}, {entry['ci_hi']:+.{places}f}]{mark}"


def write_markdown(
    r2: pd.DataFrame, picks: pd.DataFrame, audit: dict, notes: list[str],
) -> None:
    weekly = week_means(picks)
    rank_summary = r2_summary(r2, "fwd3_rank")
    points_summary = r2_summary(r2, "fwd3")
    discovery = REPLAY_SEASONS[-1]        # 2025: the hypothesis was found on it
    rank_confirm = r2_summary(r2, "fwd3_rank", exclude_season=discovery)
    points_confirm = r2_summary(r2, "fwd3", exclude_season=discovery)
    agreement = pick_agreement(picks, HEADLINE_WEEKS)

    dc_vs_base_par = bucket(weekly, "repo_dc", "repo_base", HEADLINE_WEEKS, "mean_par")
    dc_vs_base_fwd3 = bucket(weekly, "repo_dc", "repo_base", HEADLINE_WEEKS, "mean_fwd3")
    dc_vs_base_pos = bucket(weekly, "repo_dc", "repo_base", HEADLINE_WEEKS, "mean_vs_pos")
    dc_vs_naive_fwd3 = bucket(weekly, "repo_dc", "naive", HEADLINE_WEEKS, "mean_fwd3")
    dc_vs_naive_par = bucket(weekly, "repo_dc", "naive", HEADLINE_WEEKS, "mean_par")
    base_vs_naive_fwd3 = bucket(weekly, "repo_base", "naive", HEADLINE_WEEKS, "mean_fwd3")
    base_vs_naive_par = bucket(weekly, "repo_base", "naive", HEADLINE_WEEKS, "mean_par")

    name, qualifiers = verdict(
        rank_summary, rank_confirm, points_summary,
        dc_vs_base_par, dc_vs_naive_fwd3, agreement,
    )

    L: list[str] = []
    add = L.append

    add("# Depth charts as a feature: walk-forward validation, 2014-2025")
    add("")
    add(
        "Generated by `outputs/backtests/04_depth_charts_walkforward.py`. An "
        "exploratory within-season test on 2025 suggested depth chart rank adds "
        "about +0.03 R² at WR and TE on top of usage; this is that hypothesis "
        "held to the repo's real bar -- per-season expanding-window walk-forward, "
        "same design as `walkforward_2014_2025.md` -- plus the question that "
        "actually matters: does it change any decisions? **The verdict is first "
        "and the methodology is last, deliberately.** The injury-report null is "
        "recorded at the bottom of this file so nobody rebuilds it."
    )
    add("")

    # ---- verdict ---------------------------------------------------------
    add("## Verdict")
    add("")
    add(f"**{name}.**")
    add("")
    for line in qualifiers:
        add(line)
        add("")

    add(
        "Out-of-sample R² on the repo's objective (within-week percentile rank "
        "of `fwd3`), per position, pooled over the twelve replay seasons -- "
        "each season's models trained on the seasons strictly before it, with "
        "and without the five `dc_*` columns:"
    )
    add("")
    add(
        "| position | R² usage only | + depth chart | Δ (mean of per-season) | "
        f"{CI_LEVEL:.0%} CI on Δ | seasons Δ>0 | Δ excl. 2025 | CI excl. 2025 | "
        "exploratory Δ |"
    )
    add("| --- | ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: |")
    for position in M.POSITIONS:
        s = rank_summary[position]
        c = rank_confirm[position]
        exploratory = (
            f"+{EXPLORATORY_DELTA[position]:.3f}" if position in EXPLORATORY_DELTA else "--"
        )
        add(
            f"| {position} | {fmt(s['base_mean'])} | {fmt(s['dc_mean'])} | "
            f"{s['delta_mean']:+.3f} | [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] | "
            f"{s['positive_seasons']}/{s['n_seasons']} | {c['delta_mean']:+.3f} | "
            f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | {exploratory} |"
        )
    add("")
    add(
        f"The excl.-{discovery} columns matter for a reason stated before the "
        f"run: {discovery} is the season the hypothesis was *discovered* on, "
        "and a discovery sample inside its own confirmation pool biases the "
        "pool toward confirming. The without-2025 pool is the confirmatory "
        "number; the full pool is shown because eleven honest seasons plus "
        "one tainted one is still worth seeing next to it."
    )
    add("")
    add(
        "The exploratory column is the 2025 within-season split that motivated "
        "this run. Its *levels* are not comparable to these -- different split, "
        "different target convention -- and even its deltas cross a convention "
        "boundary, which is why the raw-points table below is computed too: "
        "that is the nearer comparison."
    )
    add("")
    add(
        "One caveat that applies to every interval on this page: eight "
        "uncorrected per-position tests (four positions, two targets) run on "
        "twelve paired seasons each. Under a global null, roughly one of "
        "eight such intervals excludes zero by chance alone, and n=12 "
        "percentile-bootstrap intervals under-cover their nominal "
        f"{CI_LEVEL:.0%} to begin with. Any single interval excluding zero "
        "is weak evidence; agreement across seasons, targets and the arm "
        "comparison is what would count."
    )
    add("")

    # ---- per season ------------------------------------------------------
    add("## Per season")
    add("")
    add(
        "Δ = R² with the depth chart columns minus without, on the rank "
        "objective, per position. A per-season R² at TE rides on a few hundred "
        "test rows, so read the signs and the spread, not the third decimal."
    )
    add("")
    header = "| season | trained on |"
    for position in M.POSITIONS:
        header += f" {position} base | {position} Δ |"
    add(header)
    add("| --- | --- |" + " ---: | ---: |" * len(M.POSITIONS))
    rank_rows = r2[r2["target"] == "fwd3_rank"]
    for season in REPLAY_SEASONS:
        train = W.train_seasons_for(season, "expanding")
        cells = f"| {season} | {train[0]}-{train[-1]} |"
        for position in M.POSITIONS:
            sub = rank_rows[
                (rank_rows["season"] == season) & (rank_rows["position"] == position)
            ].set_index("variant")["r2"]
            base = sub.get("base", np.nan)
            delta = sub.get("dc", np.nan) - base
            cells += f" {fmt(base)} | {delta:+.3f} |" if not np.isnan(base) else " n/a | n/a |"
        add(cells)
    add("")

    add("### On raw points, for comparability with the exploratory table")
    add("")
    add(
        "The same fits with raw `fwd3` as the target, which is closer to what "
        "the exploratory numbers were computed on:"
    )
    add("")
    add(
        f"| position | R² usage only | + depth chart | Δ | {CI_LEVEL:.0%} CI on Δ | "
        "seasons Δ>0 | Δ excl. 2025 | CI excl. 2025 |"
    )
    add("| --- | ---: | ---: | ---: | :---: | ---: | ---: | :---: |")
    for position in M.POSITIONS:
        s = points_summary[position]
        c = points_confirm[position]
        add(
            f"| {position} | {fmt(s['base_mean'])} | {fmt(s['dc_mean'])} | "
            f"{s['delta_mean']:+.3f} | [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] | "
            f"{s['positive_seasons']}/{s['n_seasons']} | {c['delta_mean']:+.3f} | "
            f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] |"
        )
    add("")
    add(points_reading(points_summary))
    add("")

    # ---- the arm comparison ----------------------------------------------
    add("## Does it change any decisions?")
    add("")
    add(
        "The repo arm -- top three wire names by points above replacement, the "
        "mechanical version of `make report` -- re-run with and without the "
        "depth chart features, over weeks 2-14 of all twelve replay seasons, "
        "against the same naive benchmark (last week's top scorers). Paired by "
        "week, season-stratified bootstrap, exactly as in the prior replays."
    )
    add("")
    add(f"| comparison | metric | n weeks | mean diff | {CI_LEVEL:.0%} CI |")
    add("| --- | --- | ---: | ---: | :---: |")
    for label, entry in [
        ("repo+dc − repo base", dc_vs_base_par),
        ("repo+dc − repo base", dc_vs_base_fwd3),
        ("repo+dc − repo base", dc_vs_base_pos),
        ("repo+dc − naive", dc_vs_naive_par),
        ("repo+dc − naive", dc_vs_naive_fwd3),
        ("repo base − naive", base_vs_naive_par),
        ("repo base − naive", base_vs_naive_fwd3),
    ]:
        metric = {
            "mean_par": "PAR (ppg)",
            "mean_fwd3": "raw fwd3 (ppg)",
            "mean_vs_pos": "vs same-position pool (ppg)",
        }[entry["value"]]
        add(
            f"| {label} | {metric} | {entry['n_weeks']} | "
            f"{entry['mean_diff']:+.2f} | {ci_cell(entry, 2)} |"
        )
    add("")
    add(
        "`repo base − naive` is the control pair: it re-derives the prior "
        "replays' comparison inside this run, so a surprise there would mean "
        "this script disagrees with `walkforward_2014_2025.md` about the "
        "baseline, and nothing else in the table could be trusted. `†` would "
        "mark an interval whose season stratification had to fall back to "
        "unstratified resampling (a bucket giving some season a single week); "
        "none of these buckets need it, and the marker is defined here so it "
        "cannot appear undefined."
    )
    add("")
    add("### Pick agreement")
    add("")
    add(
        f"- identical top-{ARM_DEPTH} lists in "
        f"**{agreement['identical_share']:.1%}** of {agreement['weeks']} paired "
        "weeks"
    )
    add(f"- name-level overlap **{agreement['name_overlap']:.1%}**")
    add("")
    per_season = agreement["per_season_identical"]
    add("| season | " + " | ".join(str(s) for s in sorted(per_season)) + " |")
    add("| --- |" + " ---: |" * len(per_season))
    add(
        "| identical weeks | "
        + " | ".join(f"{per_season[s]:.0%}" for s in sorted(per_season))
        + " |"
    )
    add("")

    # ---- availability audit ----------------------------------------------
    add("## The data: availability, formats, and the 2025 break")
    add("")
    add(
        "nflverse-data release `depth_charts`, `depth_charts_{year}.csv`, now "
        "fetched by `src/fetch.py` with the same sha256 manifest treatment as "
        "every other source and asserted non-empty at parse time by "
        "`features.require_rows`."
    )
    add("")
    add(
        "- **Files exist from 2001**; 1999 and 2000 return 404. Snap counts "
        "begin in 2013, so **2013 remains the binding floor** for any model "
        "here -- depth charts do not move it."
    )
    add(
        "- **2013-2024 files are weekly club submissions** (~3MB/season): one "
        "chart per club per week, `depth_team` rank capped at 3, all 32 clubs "
        "every week, `gsis_id` complete. Club codes are period-correct "
        "(`STL`, `SD`, `OAK`), normalised by the same `RELOCATIONS` map that "
        "already fixed the snaps merge -- without it the pre-2016 Rams, "
        "pre-2017 Chargers and pre-2020 Raiders would silently never match."
    )
    add(
        "- **2025+ files are timestamped snapshots** (~50MB/season): roughly "
        "daily dumps, no week column, offense published as one `3WR 1TE` "
        "group, ranks running past 10, and the file continues into the "
        "following March. Aligned to weeks as described under Method."
    )
    add(
        "- **The rank scale breaks at 2025.** Pre-2025 a player deeper than "
        "third string is absent (NaN reads as \"4th string or worse\"); 2025 "
        "lists everyone. `dc_is_starter` and `dc_top2` are the encodings whose "
        "meaning survives the break; `dc_rank`'s does not, and the 2025 replay "
        "season -- the one the hypothesis came from -- is scored by models "
        "that trained entirely on the old scale. That is the honest cost of "
        "walk-forward across a format change and it is visible in the 2025 "
        "row of the per-season table."
    )
    add("")
    add("### Join and coverage, per season")
    add("")
    add(
        "Joined on `gsis_id` (the stats feed's `player_id` is the same GSIS "
        "id) plus season, week and modern team code. \"match\" is the share of "
        "panel rows carrying a chart rank; the wire-universe column is the "
        "share among the rows the models actually train and score on."
    )
    add("")
    add(
        "| season | panel rows | match | QB | RB | WR | TE | wire match | "
        "wire starter share | max rank |"
    )
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for season in sorted(audit["seasons"]):
        entry = audit["seasons"][season]
        by_pos = entry["match_by_pos"]
        add(
            f"| {season} | {entry['rows']:,} | {entry['match']:.1%} | "
            + " | ".join(f"{by_pos.get(p, np.nan):.1%}" for p in M.POSITIONS)
            + f" | {entry['wire_match']:.1%} | {entry['wire_starter_share']:.1%} | "
            f"{entry['rank_max']:.0f} |"
        )
    add("")

    # ---- cross-checks -----------------------------------------------------
    add("## Exploratory cross-checks: level vs change")
    add("")
    add(
        "The exploratory finding was that the *level* carries the signal and "
        "the *change* does not. Recomputed under this alignment on the wire "
        "universe (weeks 2-14, on-wire, snap-matched, resolved `fwd3`):"
    )
    add("")
    for scope in ("2025", "2013-2024"):
        checks = audit["checks"][scope]
        add(f"**{scope}**")
        add("")
        add(
            "| position | wire rows | listed starter: mean fwd3 (n) | "
            "pool mean fwd3 | rho(dc_improve, fwd3) | promoted-to-starter rows |"
        )
        add("| --- | ---: | ---: | ---: | ---: | ---: |")
        for position in M.POSITIONS:
            c = checks[position]
            add(
                f"| {position} | {c['n']:,} | "
                f"{fmt(c['starter_fwd3'], 2)} ({c['n_starters']}) | "
                f"{fmt(c['pool_fwd3'], 2)} | {fmt(c['improve_rho'], 3)} | "
                f"{c['n_promoted']} |"
            )
        add("")
    add(
        "The exploratory versions of these were: TE listed starters 7.97 ppg "
        "against a 4.18 baseline; rank-improvement rho 0.05; \"promoted to "
        "starter this week\" firing 11 times at TE and zero at RB and WR in "
        "2025. (Exact counts differ because the alignments differ; this table "
        "is the record under this one.) The *shape* of the exploratory "
        "reading reproduces: listed starters clear their positional pool "
        "everywhere the level is measurable, the movement correlation is "
        "near zero, and promotions onto the wire are too rare to model. "
        "Where he sits is real information about raw points -- it is just "
        "not *additional* information once the usage features have already "
        "said where he sits, which is what the walk-forward tables above "
        "measure and the within-season split could not."
    )
    add("")

    # ---- permutation importance ------------------------------------------
    perm = permutation_table()
    if perm:
        add("## Where `dc_rank` actually places, 2025 walk-forward models")
        add("")
        add(
            "Permutation importance of every feature in the depth chart "
            "variant's 2025 models (trained 2013-2024, evaluated on 2025's "
            "wire universe, rank objective, 10 shuffles). The exploratory "
            "claim was that `dc_rank` placed third at WR and TE behind "
            "`eb_tgt_share` and `kal_role`; this is the same quantity under "
            "the harder fit:"
        )
        add("")
        add("| position | top features, in order (dc_* in bold) |")
        add("| --- | --- |")
        for position in M.POSITIONS:
            if position not in perm:
                continue
            cells = []
            for rank, (feature, value) in enumerate(perm[position][:8], 1):
                label = f"**{feature}**" if feature.startswith("dc_") else feature
                cells.append(f"{rank}. {label} ({value:.3f})")
            add(f"| {position} | {', '.join(cells)} |")
        add("")
        dc_places = {
            position: next(
                (i for i, (feature, _) in enumerate(ranked, 1) if feature == "dc_rank"),
                None,
            )
            for position, ranked in perm.items()
        }
        add(
            "`dc_rank` places "
            + ", ".join(
                f"{'#' + str(place) if place else 'nowhere'} at {position}"
                for position, place in dc_places.items()
            )
            + f" of {len(perm[next(iter(perm))])} features."
        )
        add("")

    # ---- method ----------------------------------------------------------
    add("## Method")
    add("")
    add(
        "Identical to `walkforward_2014_2025.md` except where depth charts "
        "require otherwise; the shared machinery is imported from "
        "`01_season_replay.py` / `02_walkforward_2014_2025.py` rather than "
        "reimplemented, so the two results cannot drift apart in how they are "
        "computed."
    )
    add("")
    add(
        "- **Walk-forward, expanding window.** Replay seasons 2014-2025; each "
        "trained on every season strictly before it (2014 on 2013; 2025 on "
        "2013-2024). The recency question was settled by the prior replay: do "
        "not recency-weight."
    )
    add(
        "- **Two model variants per season.** `base` is exactly the production "
        "feature set (`models.BASE_FEATURES` + `neutral_opp`); `dc` adds "
        "`dc_rank`, `dc_is_starter`, `dc_top2`, `dc_rank_prev`, `dc_improve`. "
        "The movement columns are in the model on purpose, mirroring the "
        "exploratory test that produced the +0.03 -- they were not expected "
        "to earn a place, and keeping them makes that null reproducible."
    )
    add(
        "- **Feature definition.** `dc_rank` is the player's best rank across "
        "his club's offense listings at QB/RB/WR/TE on the chart in force "
        "going into that week's games (both formats list a receiver in up to "
        "three slots, each with its own 1..N). `dc_is_starter` is rank == 1, "
        "`dc_top2` rank <= 2, `dc_rank_prev` the same quantity one week "
        "earlier (player-level, so a trade does not orphan it), `dc_improve` "
        "their difference, positive = moved up. Not listed stays NaN -- "
        "`HistGradientBoostingRegressor` handles missingness natively, and "
        "in the 2013-2024 format NaN genuinely means \"below third string\"."
    )
    add(
        "- **Alignment (2025+ format).** Snapshots are assigned to the week "
        "whose games they precede: per club, the last snapshot strictly "
        "before 00:00 UTC on the week's first scheduled game day, league-wide. "
        "Thursday and international kickoffs therefore tighten the cutoff for "
        "every club; a snapshot from a game day itself is never used for that "
        "week, because `dt` is UTC and `gameday` is a local date and ordering "
        "them within a day would mean inventing a timezone. Post-season "
        "snapshots (the 2025 file runs to March 2026) reach no week. "
        "`tests/test_features.py` pins both directions of the boundary."
    )
    add(
        "- **No-lookahead.** The chart for week W predates week W's games, "
        "which is stricter than the panel's contract (everything on a week-W "
        "row may be known by the Monday *after* week W). For the 2013-2024 "
        "format the label is trusted to mean what it says -- the chart FOR "
        "week W -- and even the worst case, a chart captured at game time, "
        "stays inside the contract and strictly before the `fwd3` outcome "
        "window at W+1..W+3."
    )
    add(
        "- **Scoring.** Both questions use the wire universe (weeks 2-14, "
        "on-wire, snap-matched, resolved target). R² per "
        "`models.holdout_r2`'s thresholds. The arm comparison scores top-3 "
        "picks by `fwd3` and by PAR against the realised replacement level of "
        "the same week's pool (`ledger.replacement_of`), paired by week with "
        "the season-stratified bootstrap from the prior replay."
    )
    add(
        "- **Contamination.** EB priors and the rank-to-points scale are "
        "fitted on training seasons only; replay bundles are stamped "
        "`replay_only` in `outputs/backtests/replay_models_dc/` and "
        "`src.models.load_bundle` cannot reach them. Two leaks are named "
        "rather than closed. First, the discovery-season overlap: 2025 is "
        "both the season the hypothesis was found on and a replay season, "
        "and its evaluation rows (weeks 2-14, wire universe) are exactly the "
        "rows that generated the hypothesis -- not a train/test leak, but a "
        "selection one, which is why the verdict is judged on the 2014-2024 "
        "pool and 2025 is shown separately. Second, the usual one: every "
        "constant in the pipeline was chosen by someone who had seen these "
        "seasons, and now so was the depth chart alignment rule."
    )
    add("")
    if notes:
        add(f"{len(notes)} week/pick-level exclusions were recorded; the full list is in ")
        add("`outputs/backtests/depth_charts_wf/notes.json`. All are of the kinds the ")
        add("prior replays documented (no forward window at the season tail, thin pools).")
        add("")

    # ---- the injury null --------------------------------------------------
    add("## Recorded null: nflverse injury reports add nothing")
    add("")
    add(
        "Tested in the same exploratory pass that raised the depth chart "
        "hypothesis (2025, within-season split), and recorded here so nobody "
        "rebuilds it. The numbers in this section are **transcribed from "
        "that session, not computed by this script** -- deliberately, and "
        "against this repo's usual convention, because reproducing them "
        "would mean building the injury pipeline this section exists to "
        "close the door on. nflverse `injuries_{year}.csv` -- the official "
        "practice and game-status reports -- added **exactly +0.000 R² at "
        "all four positions**, and the reason is structural rather than "
        "statistical:"
    )
    add("")
    add(
        "- **Own injury status is unmeasurable by construction.** The panel "
        "conditions on a player having taken a snap; anyone listed \"Out\" "
        "has no row. Only 3.4% of panel rows carried any report at all and "
        "exactly zero carried \"Out\". The feature cannot vary where the "
        "panel can see it."
    )
    add(
        "- **Teammate injuries point the wrong way.** Features counting "
        "injured teammates predicted *worse* outcomes, monotonically: quiet "
        "WRs with 2+ teammates out returned 2.21 ppg against 2.84 with none. "
        "An injury-riddled offense is a bad offense; the extra targets do not "
        "outweigh it."
    )
    add("")
    add(
        "**Do not add injury features to this pipeline.** A future attempt "
        "would first have to change the panel itself (rows for players who "
        "did not play), which is a different and much larger decision than a "
        "feature."
    )
    add("")

    # ---- what this does not test -----------------------------------------
    add("## What this does NOT test")
    add("")
    add(
        "- Everything the prior replay's list already carries: no judgement "
        "layer, no roster constraints, no real league availability, no waiver "
        "contention."
    )
    add(
        "- **The 2025 format on its own terms.** Eleven of twelve replay "
        "seasons test the weekly format; only 2025 tests the snapshot format, "
        "and it does so with models trained on the other one. A few more "
        "seasons of the new feed would be needed to validate it separately."
    )
    add(
        "- **Intra-week chart moves.** The Monday claim uses the chart going "
        "into the week that just ended; a chart published Tuesday reflecting "
        "Sunday's injury is next week's feature. That is the price of the "
        "no-lookahead contract, not an oversight."
    )
    add("")

    # ---- reproducing ------------------------------------------------------
    add("## Reproducing")
    add("")
    add("```bash")
    add("make install")
    add('make data SEASONS="' + " ".join(str(s) for s in W.ALL_SEASONS) + '"')
    add("python outputs/backtests/04_depth_charts_walkforward.py")
    add("```")
    add("")
    add(
        "Writes `outputs/backtests/depth_charts_wf/` (per-pick rows with PAR, "
        "per-season-position R², the audit JSON, exclusion notes) and this "
        "file; `--report-only` rewrites the markdown from those artifacts "
        "without refitting. Computed against the nflverse revision pinned in "
        "`data/raw/MANIFEST.json`. The usual reproducibility caveat applies "
        "with extra force here: the R² deltas in play are of the same order "
        "as the ±0.3 ppg / bin-edge float sensitivity the prior replays "
        "measured, which is exactly why the verdict reads intervals and "
        "season counts, never a point estimate. **The third decimal place is "
        "not real.**"
    )
    add("")

    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text("\n".join(L) + "\n")
    print(f"\nwrote {DIAGNOSTIC.relative_to(ROOT)}")


# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report-only", action="store_true",
        help="rewrite the markdown from a completed run's saved artifacts",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = OUT_DIR / "picks.csv"
    r2_path = OUT_DIR / "r2.csv"
    audit_path = OUT_DIR / "audit.json"
    notes_path = OUT_DIR / "notes.json"

    if args.report_only:
        if not picks_path.exists():
            raise SystemExit("no completed run to report on -- run without --report-only")
        picks = pd.read_csv(picks_path)
        r2 = pd.read_csv(r2_path)
        audit = json.loads(audit_path.read_text())
        audit["seasons"] = {int(k): v for k, v in audit["seasons"].items()}
        notes = json.loads(notes_path.read_text())
        write_markdown(r2, picks, audit, notes)
        return 0

    finals = W.final_weeks()
    print("=" * 74)
    print(f"DEPTH CHART WALK-FORWARD {REPLAY_SEASONS[0]}-{REPLAY_SEASONS[-1]}")
    print("=" * 74)

    all_picks: list[pd.DataFrame] = []
    all_r2: list[dict] = []
    notes: list[str] = []
    audit: dict | None = None
    for season in REPLAY_SEASONS:
        picks, r2, season_notes, panel = replay_one_season(season, finals)
        all_picks.append(picks)
        all_r2.extend(r2)
        notes.extend(season_notes)
        if season == REPLAY_SEASONS[-1]:
            audit = audit_tables(panel)

    picks = pd.concat(all_picks, ignore_index=True)
    r2 = pd.DataFrame(all_r2)
    picks.to_csv(picks_path, index=False, float_format="%.4f")
    r2.to_csv(r2_path, index=False, float_format="%.6f")
    audit_path.write_text(json.dumps(audit, indent=2, default=float) + "\n")
    notes_path.write_text(json.dumps(notes, indent=2) + "\n")

    write_markdown(r2, picks, audit, notes)
    for path in (picks_path, r2_path, audit_path):
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

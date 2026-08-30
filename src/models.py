"""Fit and persist the per-position waiver ranking models.

Learning to rank, not regression. The models predict the *within-week percentile
rank* of fwd3 rather than fwd3 itself, because only the top one or two claims
ever get made: getting Nacua above the other twelve names on the wire is the
whole job, and being wrong about whether he scores 14.2 or 11.6 costs nothing.
Predicting the rank also sidesteps the fact that a good week in December is
worth fewer raw points than a good week in October.

Each position gets its own model, fit on the same wire universe the backtest
uses (weeks 2-14, on_wire, snap and fwd3 present), over every complete season.

Alongside each model, a split-conformal half-width: fit on seasons through N-2,
calibrate on N-1, take the 80th percentile of absolute calibration residuals.
Empirical coverage is then measured on season N, which the conformal pipeline
never saw. This is what turns a score into an honest range in the weekly report;
a point estimate would imply a precision these models do not have.

Run:
    python -m src.models
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from src.fetch import data_revision

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"
MODEL_DIR = ROOT / "models"
MODEL_CARD = MODEL_DIR / "MODEL_CARD.md"

POSITIONS = ["RB", "WR", "TE", "QB"]
WEEKS = (2, 14)

# Always present in the panel.
BASE_FEATURES = [
    "snap",
    "targets",
    "tgt_share",
    "carries",
    "carry_share",
    "wopr_opp",
    "receptions",
    "air_yards_share",
    "eb_tgt_share",
    "eb_car_share",
    "kal_role",
    "cusum",
    "pts_lag1",
]

# Included when the panel carries them. neutral_opp can be NaN for a season
# whose play-by-play file was not fetched; HistGradientBoostingRegressor
# handles missing values natively, so that is not a reason to exclude it.
OPTIONAL_FEATURES = ["neutral_opp"]

MODEL_KWARGS = dict(max_depth=3, max_iter=250, learning_rate=0.05, random_state=0)

TARGET = "fwd3"
CONFORMAL_LEVEL = 0.80


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return BASE_FEATURES + [c for c in OPTIONAL_FEATURES if c in panel.columns]


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise SystemExit(f"{PANEL_PATH} not found -- run `make panel` first")
    return pd.read_csv(PANEL_PATH)


def wire_universe(panel: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Weeks 2-14, on the wire, with a snap match and a usable target."""
    universe = panel[
        panel["week"].between(*WEEKS)
        & panel["on_wire"]
        & panel["snap"].notna()
        & panel[target].notna()
    ].copy()
    # Rank within week across the whole wire pool, per the ranking objective.
    universe["fwd3_rank"] = universe.groupby(["season", "week"])[target].rank(pct=True)
    return universe


def fit_model(frame: pd.DataFrame, features: list[str]) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(**MODEL_KWARGS)
    model.fit(frame[features], frame["fwd3_rank"])
    return model


def conformal(
    frame: pd.DataFrame,
    features: list[str],
    seasons: list[int],
    label: str = "fwd3_rank",
) -> tuple[float, float, int, int]:
    """Split-conformal half-width and its empirical coverage on the held-out season.

    Fit through N-2, calibrate on N-1, evaluate on N. Returns
    (half_width, coverage, n_calibration, n_test). NaN when there are too few
    seasons to do the split honestly.
    """
    if len(seasons) < 3:
        return float("nan"), float("nan"), 0, 0
    latest = seasons[-1]
    fit_seasons = [s for s in seasons if s <= latest - 2]
    calibrate_season = latest - 1

    train = frame[frame["season"].isin(fit_seasons)]
    calibrate = frame[frame["season"] == calibrate_season]
    test = frame[frame["season"] == latest]
    if len(train) < 100 or len(calibrate) < 50 or len(test) < 50:
        return float("nan"), float("nan"), len(calibrate), len(test)

    model = HistGradientBoostingRegressor(**MODEL_KWARGS)
    model.fit(train[features], train[label])
    residuals = np.abs(calibrate[label] - model.predict(calibrate[features]))
    half_width = float(np.quantile(residuals, CONFORMAL_LEVEL))

    error = np.abs(test[label] - model.predict(test[features]))
    coverage = float((error <= half_width).mean())
    return half_width, coverage, len(calibrate), len(test)


def holdout_r2(frame: pd.DataFrame, features: list[str], seasons: list[int]) -> float:
    """R² on the last complete season, training on everything before it."""
    if len(seasons) < 2:
        return float("nan")
    latest = seasons[-1]
    train = frame[frame["season"] < latest]
    test = frame[frame["season"] == latest]
    if len(train) < 100 or len(test) < 50:
        return float("nan")
    model = fit_model(train, features)
    return float(r2_score(test["fwd3_rank"], model.predict(test[features])))


def train_all() -> dict:
    panel = load_panel()
    features = feature_columns(panel)
    universe = wire_universe(panel)
    seasons = sorted(universe["season"].unique().tolist())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    revision = data_revision()
    summary: dict[str, dict] = {}

    print(f"seasons: {seasons}")
    print(f"features ({len(features)}): {', '.join(features)}")
    missing = [c for c in OPTIONAL_FEATURES if c not in panel.columns]
    if missing:
        print(f"not in panel, skipped: {', '.join(missing)}")
    print()

    for position in POSITIONS:
        frame = universe[universe["position"] == position]
        model = fit_model(frame, features)
        half_width, coverage, n_cal, n_test = conformal(frame, features, seasons)
        # Same split against raw points, purely to show what the ranking
        # objective buys in interval quality. Not persisted, not served.
        points_hw, points_coverage, _, _ = conformal(frame, features, seasons, TARGET)
        r2 = holdout_r2(frame, features, seasons)

        bundle = {
            "position": position,
            "model": model,
            "features": features,
            "target": f"within-week percentile rank of {TARGET}",
            "conformal_half_width": half_width,
            "conformal_level": CONFORMAL_LEVEL,
            "empirical_coverage": coverage,
            "points_coverage_reference": points_coverage,
            "holdout_r2": r2,
            "train_seasons": seasons,
            "n_train": len(frame),
            "trained_on": date.today().isoformat(),
            "data_revision": revision,
            "model_kwargs": MODEL_KWARGS,
        }
        joblib.dump(bundle, MODEL_DIR / f"{position}.joblib")

        flag = ""
        if not np.isnan(coverage) and abs(coverage - CONFORMAL_LEVEL) > 0.03:
            flag = "   <-- off target"
        print(
            f"{position}  n={len(frame):5,}  R2={r2:6.3f}  "
            f"half-width={half_width:.3f}  coverage={coverage:.3f}"
            f"  (cal n={n_cal:,}, test n={n_test:,}){flag}"
        )
        summary[position] = bundle

    write_model_card(summary, features, seasons, revision)
    print(f"\nwrote {MODEL_CARD.relative_to(ROOT)} and {len(summary)} model files")
    return summary


def write_model_card(
    summary: dict, features: list[str], seasons: list[int], revision: str
) -> None:
    manifest_path = ROOT / "data" / "raw" / "MANIFEST.json"
    n_files = len(json.loads(manifest_path.read_text()).get("files", {})) if manifest_path.exists() else 0

    lines = [
        "# Model card",
        "",
        "Regenerated by `python -m src.models` on every retrain. Do not edit by hand.",
        "",
        f"- **Trained** {date.today().isoformat()}",
        f"- **Data revision** `{revision}`",
        f"  (sha256 over the {n_files} (filename, sha256) pairs in `data/raw/MANIFEST.json`;",
        "  changes when nflverse revises history, not when identical bytes are re-fetched)",
        f"- **Training window** {seasons[0]}-{seasons[-1]}, regular season weeks "
        f"{WEEKS[0]}-{WEEKS[1]}",
        "- **Universe** players on the waiver wire (season-to-date scoring rank below",
        "  12x15 roster depth) with a snap-count match and a defined target",
        "",
        "## Objective",
        "",
        f"Each model predicts the **within-week percentile rank of `{TARGET}`**",
        "(`groupby(['season','week']).fwd3.rank(pct=True)`), not the raw points.",
        "Only the top one or two claims ever get made, so ordering the wire correctly",
        "is the entire job; the magnitude of a projection is not.",
        "",
        f"`{TARGET}` scores a week the player's team played but he did not appear in as",
        "0.0 -- the claim returned nothing that week. `fwd3_played`, the games-played-only",
        "alternative, is in the panel and is not used here; the choice materially moves",
        "measured input importance (see `outputs/backtests/results_input_importance.txt`).",
        "",
        "## Features",
        "",
    ]
    lines += [f"{i}. `{f}`" for i, f in enumerate(features, 1)]
    neutral_line = (
        "Neutral-script opportunity (`neutral_opp`) is in this fit."
        if "neutral_opp" in features
        else "Neutral-script opportunity is named in `src/models.py` but not built into "
        "the panel, so it is absent from this fit."
    )
    lines += [
        "",
        f"All strictly backward-looking as of the Monday claims are entered. {neutral_line}",
        "",
        "## Per-position results",
        "",
        f"Model: `HistGradientBoostingRegressor({', '.join(f'{k}={v}' for k, v in MODEL_KWARGS.items())})`",
        "",
        f"Out-of-sample R² trains on {seasons[0]}-{seasons[-1] - 1} and tests on {seasons[-1]}.",
        f"The conformal half-width is the {int(CONFORMAL_LEVEL * 100)}th percentile of absolute residuals on",
        f"season {seasons[-2]} for a model fit through {seasons[-3]}; coverage is then measured on",
        f"{seasons[-1]}, which that pipeline never saw.",
        "",
        "| position | n | out-of-sample R² | conformal half-width | empirical coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for position in POSITIONS:
        bundle = summary[position]
        note = ""
        coverage = bundle["empirical_coverage"]
        if not np.isnan(coverage) and coverage < CONFORMAL_LEVEL - 0.03:
            note = f" ⚠️ under-covers"
        elif not np.isnan(coverage) and coverage > CONFORMAL_LEVEL + 0.03:
            note = f" ⚠️ over-covers"
        lines.append(
            f"| {position} | {bundle['n_train']:,} | {bundle['holdout_r2']:.3f} | "
            f"{bundle['conformal_half_width']:.3f} | {coverage:.3f}{note} |"
        )

    lines += [
        "",
        f"Target coverage is {CONFORMAL_LEVEL:.2f}. Anything outside ±0.03 is flagged above",
        "rather than quietly reported, because an interval that does not cover at its stated",
        "rate is worse than no interval: it invites confidence it has not earned.",
        "",
        "### What the ranking objective buys",
        "",
        "The same split-conformal procedure run against raw `fwd3` points instead of the",
        "rank, for reference only — these intervals are not fitted, persisted or served:",
        "",
        "| position | coverage on rank (shipped) | coverage on raw points |",
        "| --- | ---: | ---: |",
    ]
    for position in POSITIONS:
        bundle = summary[position]
        reference = bundle.get("points_coverage_reference", float("nan"))
        lines.append(
            f"| {position} | {bundle['empirical_coverage']:.3f} | {reference:.3f} |"
        )
    lines += [
        "",
        "QB is the position where this matters. Raw fantasy points at QB have a heavy",
        "right tail — a starter who also runs blows past any symmetric interval — so",
        "point-scale intervals under-cover there while the skill positions hold near",
        "0.80. Percentile rank is bounded, which flattens that tail and brings QB back",
        "into line. The improvement is a property of the objective, not a better model.",
        "",
        "## Files",
        "",
        "`models/{position}.joblib` — a dict holding the fitted estimator, the feature list",
        "it expects, the conformal half-width and coverage, and the data revision above.",
        "The half-width travels with the model so a score can never be served without the",
        "range that belongs to it.",
        "",
    ]
    MODEL_CARD.write_text("\n".join(lines))


def load_bundle(position: str) -> dict:
    path = MODEL_DIR / f"{position}.joblib"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `make models` first")
    return joblib.load(path)


def main(argv: list[str] | None = None) -> int:
    train_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

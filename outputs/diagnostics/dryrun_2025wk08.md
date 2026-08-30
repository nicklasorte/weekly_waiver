# Dry run: 2025 Week 8

**VERDICT: PASS** — all six commands completed successfully.

## Environment

- System python3: 3.11.15 (not used directly)
- Bootstrap interpreter (`python3.12`): 3.12.3
- `.venv` python (created by `make install`): Python 3.12.3
- Venv state: **fresh** — this is a new ephemeral container clone, so no `.venv`
  existed at session start despite the prior session having built one. It was
  created by `make install` per the Makefile's normal behavior (not a fix).

## Step-by-step results

### 1. `make install` — PASS

```
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
Requirement already satisfied: pip in ./.venv/lib/python3.12/site-packages (24.0)
Collecting pip
  Downloading pip-26.2.1-py3-none-any.whl.metadata (4.6 kB)
Downloading pip-26.2.1-py3-none-any.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 39.8 MB/s eta 0:00:00
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 24.0
    Uninstalling pip-24.0:
      Successfully uninstalled pip-24.0
Successfully installed pip-26.2.1
.venv/bin/python -m pip install -r requirements.txt
Collecting pandas>=2.0 (from -r requirements.txt (line 1))
  Downloading pandas-3.0.5-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (79 kB)
Collecting numpy>=1.24 (from -r requirements.txt (line 2))
  Downloading numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (6.6 kB)
Collecting scikit-learn>=1.3 (from -r requirements.txt (line 3))
  Downloading scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (11 kB)
Collecting scipy>=1.10 (from -r requirements.txt (line 4))
  Downloading scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting joblib (from -r requirements.txt (line 5))
  Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting python-dateutil>=2.8.2 (from pandas>=2.0->-r requirements.txt (line 1))
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting narwhals>=2.0.1 (from scikit-learn>=1.3->-r requirements.txt (line 3))
  Downloading narwhals-2.25.0-py3-none-any.whl.metadata (15 kB)
Collecting threadpoolctl>=3.5.0 (from scikit-learn>=1.3->-r requirements.txt (line 3))
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas>=2.0->-r requirements.txt (line 1))
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Downloading pandas-3.0.5-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (11.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.0/11.0 MB 43.0 MB/s  0:00:00
Downloading numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.7/16.7 MB 44.6 MB/s  0:00:00
Downloading scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (9.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.1/9.1 MB 54.2 MB/s  0:00:00
Downloading scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (35.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.3/35.3 MB 42.4 MB/s  0:00:00
Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
Downloading narwhals-2.25.0-py3-none-any.whl (467 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Installing collected packages: threadpoolctl, six, numpy, narwhals, joblib, scipy, python-dateutil, scikit-learn, pandas

Successfully installed joblib-1.5.3 narwhals-2.25.0 numpy-2.5.2 pandas-3.0.5 python-dateutil-2.9.0.post0 scikit-learn-1.9.0 scipy-1.18.1 six-1.17.0 threadpoolctl-3.6.0
```

Exit code: 0

### 2. `make data` — PASS

```
.venv/bin/python -m src.fetch 2022 2023 2024 2025 2026
games.csv
  games.csv: 2,177,202 bytes  9a423169f9f0c17a...  <-- upstream revised, prior results are not comparable
2022
  snap_counts_2022.csv: 2,379,719 bytes  0018a4833fbf0f82...
  stats_player_week_2022.csv: 8,408,729 bytes  ad426c3fe5bf1cc3...
  play_by_play_2022.csv.gz: 19,093,961 bytes  0c69a71eb3949895...
2023
  snap_counts_2023.csv: 2,394,875 bytes  303b61aa5c33ffda...
  stats_player_week_2023.csv: 8,332,874 bytes  f19cb71a5de0dce7...
  play_by_play_2023.csv.gz: 19,169,807 bytes  4649804ee0f0a40b...
2024
  snap_counts_2024.csv: 2,402,841 bytes  a2aa58efe093f8aa...
  stats_player_week_2024.csv: 8,470,040 bytes  3ddc45a84f759aa3...
  play_by_play_2024.csv.gz: 19,362,351 bytes  23370d5d10f8104d...
2025
  snap_counts_2025.csv: 2,401,193 bytes  80b02a6e511aa202...
  stats_player_week_2025.csv: 8,656,387 bytes  e5e0615b3d96a3ea...
  play_by_play_2025.csv.gz: 19,105,296 bytes  2f135887790a013f...
2026
  snap_counts_2026.csv: not published yet (https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2026.csv)
  stats_player_week_2026.csv: not published yet (https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.csv)
  play_by_play_2026.csv.gz: not published yet (https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.csv.gz)

13 files in MANIFEST.json (13 downloaded, 0 skipped, 3 unavailable)
```

Exit code: 0. The 3 "unavailable" files are 2026 season data (not yet published upstream) — expected, since 2026 is a future season relative to the data source.

### 3. `make panel` — PASS

```
.venv/bin/python -m src.features 2022 2023 2024 2025 2026
2022: 5,808 stat rows, 5,631 matched to snaps (97.0%), 177 dropped
2023: 5,801 stat rows, 5,600 matched to snaps (96.5%), 201 dropped
2024: 5,864 stat rows, 5,664 matched to snaps (96.6%), 200 dropped
2025: 6,037 stat rows, 5,720 matched to snaps (94.7%), 317 dropped
2026: source files missing, skipping
empirical bayes priors:
  eb_tgt_share prior QB: alpha=0.041 beta=87.624
  eb_tgt_share prior TE: alpha=1.623 beta=14.946
  eb_tgt_share prior WR: alpha=1.584 beta=10.223
  eb_tgt_share prior RB: alpha=1.289 beta=18.021
  eb_car_share prior QB: alpha=1.706 beta=10.757
  eb_car_share prior TE: alpha=0.021 beta=7.233
  eb_car_share prior WR: alpha=0.222 beta=32.924
  eb_car_share prior RB: alpha=0.868 beta=2.158

wrote data/processed/panel.csv
rows: 22,615
seasons: [2022, 2023, 2024, 2025]
positions: {'WR': 9349, 'RB': 5915, 'TE': 4781, 'QB': 2570}
on_wire rows: 11,622
rows with fwd3: 21,293
rows with fwd3_played: 19,498
```

Exit code: 0

### 4. `make models` — PASS

```
.venv/bin/python -m src.models
seasons: [2022, 2023, 2024, 2025]
features (13): snap, targets, tgt_share, carries, carry_share, wopr_opp, receptions, air_yards_share, eb_tgt_share, eb_car_share, kal_role, cusum, pts_lag1
not in panel, skipped: neutral_opp

RB  n=1,788  R2= 0.259  half-width=0.297  coverage=0.793  (cal n=462, test n=435)
WR  n=3,495  R2= 0.325  half-width=0.291  coverage=0.806  (cal n=859, test n=890)
TE  n=2,454  R2= 0.382  half-width=0.290  coverage=0.827  (cal n=620, test n=653)
QB  n=  862  R2= 0.435  half-width=0.337  coverage=0.825  (cal n=217, test n=223)

wrote models/MODEL_CARD.md and 4 model files
```

Exit code: 0

### 5. `make weekly SEASON=2025 WEEK=8` — PASS

```
.venv/bin/python -m src.weekly --season 2025 --week 8

2025 week 8: 134 wire candidates -> outputs/weekly/2025/wk08.csv

top 5 RB
 pos_rank player_display_name team  score  proj_pts_lo  proj_pts_hi   snap  targets  tgt_share  carries  carry_share    pts
        1          Ty Johnson  BUF  0.796        0.967       16.804  0.320        0      0.000        4        0.114  0.600
        2         Sean Tucker   TB  0.706        0.533       16.804  0.340        0      0.000       12        0.414 10.200
        3      Ameer Abdullah  IND  0.568        0.050        5.600  0.190        1      0.053        2        0.111  1.300
        4         Tank Bigsby  PHI  0.531        0.000        4.892  0.270        0      0.000        9        0.273 10.400
        5     Jaret Patterson  LAC  0.524        0.000        4.669  0.210        0      0.000       11        0.256  3.000

top 5 WR
 pos_rank player_display_name team  score  proj_pts_lo  proj_pts_hi   snap  targets  tgt_share  carries  carry_share    pts
        1         Rashee Rice   KC  0.835        1.600       14.806  0.860        9      0.281        2        0.067 21.000
        2    Christian Watson   GB  0.702        0.767       14.806  0.560        4      0.114        0        0.000 10.500
        3      Darnell Mooney  ATL  0.702        0.767       14.806  0.900        4      0.148        0        0.000  1.600
        4        Jahan Dotson  PHI  0.656        0.533        9.471  0.710        2      0.100        0        0.000 10.500
        5        Roman Wilson  PIT  0.637        0.433        8.445  0.460        5      0.172        0        0.000 15.400

top 5 TE
 pos_rank player_display_name team  score  proj_pts_lo  proj_pts_hi   snap  targets  tgt_share  carries  carry_share    pts
        1          Cade Otton   TB  0.886        2.733       16.463  0.970        5      0.208        0        0.000  8.000
        2      Pat Freiermuth  PIT  0.859        2.500       16.463  0.430        4      0.138        0        0.000  4.300
        3         Jonnu Smith  PIT  0.819        2.150       16.463  0.600        3      0.103        0        0.000  3.700
        4           Noah Fant  CIN  0.812        2.133       16.463  0.470        3      0.091        0        0.000  7.100
        5         Evan Engram  DEN  0.796        1.967       16.463  0.430        4      0.154        0        0.000  7.600

top 5 QB
 pos_rank player_display_name team  score  proj_pts_lo  proj_pts_hi   snap  targets  tgt_share  carries  carry_share    pts
        1            Cam Ward  TEN  0.927       10.507       25.798  1.000        0      0.000        2        0.080 12.760
        2      Dillon Gabriel  CLE  0.926       10.487       25.798  1.000        0      0.000        1        0.067 11.540
        3          Joe Flacco  CIN  0.878        9.149       25.798  1.000        0      0.000        2        0.087 24.320
        4        Carson Wentz  MIN  0.767        5.090       25.798  0.880        0      0.000        0        0.000  7.760
        5        Tyler Shough   NO  0.764        5.017       25.798  0.540        0      0.000        3        0.200  4.320
```

Exit code: 0

### 6. `make report SEASON=2025 WEEK=8` — PASS

```
.venv/bin/python -m src.report --season 2025 --week 8 
wrote outputs/reports/2025/wk08.md (358 words)
logged 0 new claim(s) to outputs/ledger/claims.csv
```

Exit code: 0

## Key checks

- **MANIFEST.json file count after `make data`: 13** (13 downloaded, 0 skipped,
  3 unavailable — the 3 unavailable are 2026 season files not yet published
  upstream, listed separately and not counted in the 13).
- **panel.csv row count after `make panel`: 22,615 data rows** (22,616 lines
  including header). **Season list: [2022, 2023, 2024, 2025]** (2026 skipped —
  source files missing, as expected since 2026 data isn't published yet).
- **Full contents of `models/MODEL_CARD.md`:**

```markdown
# Model card

Regenerated by `python -m src.models` on every retrain. Do not edit by hand.

- **Trained** 2026-08-30
- **Data revision** `5e81b1c10fc997e3e8319ed2ba8e1c610990f78a3a56adb0467c15ee91ce40b0`
  (sha256 over the 13 (filename, sha256) pairs in `data/raw/MANIFEST.json`;
  changes when nflverse revises history, not when identical bytes are re-fetched)
- **Training window** 2022-2025, regular season weeks 2-14
- **Universe** players on the waiver wire (season-to-date scoring rank below
  12x15 roster depth) with a snap-count match and a defined target

## Objective

Each model predicts the **within-week percentile rank of `fwd3`**
(`groupby(['season','week']).fwd3.rank(pct=True)`), not the raw points.
Only the top one or two claims ever get made, so ordering the wire correctly
is the entire job; the magnitude of a projection is not.

`fwd3` scores a week the player's team played but he did not appear in as
0.0 -- the claim returned nothing that week. `fwd3_played`, the games-played-only
alternative, is in the panel and is not used here; the choice materially moves
measured input importance (see `outputs/backtests/results_input_importance.txt`).

## Features

1. `snap`
2. `targets`
3. `tgt_share`
4. `carries`
5. `carry_share`
6. `wopr_opp`
7. `receptions`
8. `air_yards_share`
9. `eb_tgt_share`
10. `eb_car_share`
11. `kal_role`
12. `cusum`
13. `pts_lag1`

All strictly backward-looking as of the Monday claims are entered. Neutral-script
opportunity is named in `src/models.py` but not yet built into the panel, so it is
absent from this fit.

## Per-position results

Model: `HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, random_state=0)`

Out-of-sample R² trains on 2022-2024 and tests on 2025.
The conformal half-width is the 80th percentile of absolute residuals on
season 2024 for a model fit through 2023; coverage is then measured on
2025, which that pipeline never saw.

| position | n | out-of-sample R² | conformal half-width | empirical coverage |
| --- | ---: | ---: | ---: | ---: |
| RB | 1,788 | 0.259 | 0.297 | 0.793 |
| WR | 3,495 | 0.325 | 0.291 | 0.806 |
| TE | 2,454 | 0.382 | 0.290 | 0.827 |
| QB | 862 | 0.435 | 0.337 | 0.825 |

Target coverage is 0.80. Anything outside ±0.03 is flagged above
rather than quietly reported, because an interval that does not cover at its stated
rate is worse than no interval: it invites confidence it has not earned.

### What the ranking objective buys

The same split-conformal procedure run against raw `fwd3` points instead of the
rank, for reference only — these intervals are not fitted, persisted or served:

| position | coverage on rank (shipped) | coverage on raw points |
| --- | ---: | ---: |
| RB | 0.793 | 0.805 |
| WR | 0.806 | 0.790 |
| TE | 0.827 | 0.786 |
| QB | 0.825 | 0.762 |

QB is the position where this matters. Raw fantasy points at QB have a heavy
right tail — a starter who also runs blows past any symmetric interval — so
point-scale intervals under-cover there while the skill positions hold near
0.80. Percentile rank is bounded, which flattens that tail and brings QB back
into line. The improvement is a property of the objective, not a better model.

## Files

`models/{position}.joblib` — a dict holding the fitted estimator, the feature list
it expects, the conformal half-width and coverage, and the data revision above.
The half-width travels with the model so a score can never be served without the
range that belongs to it.
```

- **Top 5 rows for RB/WR/TE/QB from `outputs/weekly/2025/wk08.csv`:** see the
  `make weekly` output above (section 5) — identical top-5 tables for all four
  positions.

- **Does Rashee Rice appear near the top of the WR board? YES.**
  Rashee Rice is **WR pos_rank 1** (overall_rank 6 across all positions), with
  score 0.8346, 9 targets, 28.12% target share, and 21.0 fantasy points the
  prior week. Full CSV row:

  ```
  overall_rank,pos_rank,player_display_name,position,team,season,week,score,score_lo,score_hi,proj_pts,proj_pts_lo,proj_pts_hi,snap,snap_jump,targets,tgt_share,eb_tgt_share,carries,carry_share,eb_car_share,receptions,air_yards_share,wopr_opp,kal_role,cusum,pts,pts_lag1,cum_before,rank_before,conformal_half_width,model_coverage
  6,1,Rashee Rice,WR,KC,2025,8,0.8346,0.5439,1.0000,5.4667,1.6000,14.8060,0.8600,0.4500,9,0.2812,0.2646,2,0.0667,0.0220,9,0.1689,24.5000,0.6800,0.0000,21.0000,19.7000,19.7000,63.0000,0.2908,0.8056
  ```

## Full generated report — `outputs/reports/2025/wk08.md`

```markdown
# 2025 Week 8 — switching

Week 8, record not configured, **switching** mode: balance the playoff push against players who still hold value.

## Roster check

_No roster configured — injury, bye and role checks skipped._

## Top of the wire

- **Ty Johnson** (RB, BUF) — 32% snaps; 1.0–16.8 pts/wk
- **Sean Tucker** (RB, TB) — 41% carry share and 12 carries; 0.5–16.8 pts/wk
- **Rashee Rice** (WR, KC) — 9 targets and 28% target share; 1.6–14.8 pts/wk
- **Christian Watson** (WR, GB) — 38% air yards and 56% snaps; 0.8–14.8 pts/wk
- **Cade Otton** (TE, TB) — 21% target share and 5 targets; 2.7–16.5 pts/wk
- **Pat Freiermuth** (TE, PIT) — 4 targets and 14% target share; 2.5–16.5 pts/wk
- **Cam Ward** (QB, TEN) — 100% snaps; 10.5–25.8 pts/wk
- **Dillon Gabriel** (QB, CLE) — 100% snaps; 10.5–25.8 pts/wk

## Claims

**Burn the claim**

- `[KEEPER]` ADD Rashee Rice / DROP ??? — 9 targets and 28% target share; 1.6–14.8 pts/wk.
- `[STARTER]` ADD Ty Johnson / DROP ??? — 32% snaps; 1.0–16.8 pts/wk.

**Claim if tier 1 fails**

- `[KEEPER]` ADD Cade Otton / DROP ??? — 21% target share and 5 targets; 2.7–16.5 pts/wk.
- `[STARTER]` ADD Cam Ward / DROP ??? — 100% snaps; 10.5–25.8 pts/wk.
- `[STARTER]` ADD Dillon Gabriel / DROP ??? — 100% snaps; 10.5–25.8 pts/wk.

**Watch list**

- `[STARTER]` WATCH Sean Tucker — 41% carry share and 12 carries; 0.5–16.8 pts/wk.
- `[STARTER]` WATCH Christian Watson — 38% air yards and 56% snaps; 0.8–14.8 pts/wk.
- `[STARTER]` WATCH Darnell Mooney — 90% snaps and 30% air yards; 0.8–14.8 pts/wk.
- `[STARTER]` WATCH Pat Freiermuth — 4 targets and 14% target share; 2.5–16.5 pts/wk.

## Standing rules

- Recommendations only. Nothing here places a transaction.
- Claim now. Hoarding priority to Week 10 cost ~69% of a claim's value.
- Ranges are 80% conformal intervals, not projections. They are wide because the models are.
- Claims are ordered by points above replacement at the same position, not by raw model score, which is not comparable across positions.
- `DROP ???` — wire up a roster to resolve the other half.
```

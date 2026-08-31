# Repo state audit

Read-only audit of `main` at `b4acae2` (Merge pull request #18), taken
2026-08-31. Nothing in this session changed code, fixed a bug or opened a PR;
the only file added is this one. The pipeline was run end to end and the
regenerated artifacts were then restored to their committed bytes, so the
working tree is as it was.

## The ten-second version

| # | question | answer | |
| --- | --- | --- | --- |
| 1 | Does `src/ledger.py` grade on PAR or raw `fwd3`? | **PAR.** `fwd3` is kept on every row but the headline, the head-to-head and the pre-registered rule all read `mean_par`. | **YES (PAR)** |
| 2 | Did the heuristic ablation run? | **Yes.** Model vs `eb_share` / `opp` / `snap` / `hot_hand_pos`, within position, on PAR, 156 weeks. Verdict: keep `models/`, narrowly, and only because the product is a *list*. | **YES** |
| 3 | Three-arm comparison infrastructure? | `claims.csv` has `arm`, `rank_within_arm`, `logged_at` (and `contaminated`); `docs/comparison_protocol.md` exists with the contamination rule and thresholds; `make log-claim` exists. One factual error in the protocol doc — see §3. | **YES** (doc bug) |
| 4 | Roster wiring? | Only `data/roster.example.yaml`. `data/roster.yaml` does not exist and is gitignored by design. With `ROSTER=data/roster.example.yaml` drops resolve; with no roster it prints `DROP ???`. | **PARTIAL** |
| 5 | What's on main? | 58 commits, 18 PRs all merged, 0 open PRs, 11 stale-but-merged branches. Full listing in §5. | — |
| 6 | Does it still run? | Every step passes. 105 tests OK. Rashee Rice is **WR #1** on the 2025 Week 8 board, and the regenerated `wk08.csv` and `wk08.md` are **byte-identical** to what is committed. | **YES** |
| 7 | Known gaps | 9 items. **One is blocking before Week 1** (no `data/roster.yaml`); the rest are not. | see §7 |

**Nothing found in this audit blocks Week 1 except wiring up your own
`data/roster.yaml`.** The scoring question you were most worried about — item 1
— is already resolved the way you wanted it.

---

## 1. Ledger scoring: PAR, not raw `fwd3`

**It grades on points above replacement at position.** This is not merely
documented; it is what the code computes.

The module docstring, `src/ledger.py:52-68`:

```
THE OUTCOME METRIC: PAR, NOT RAW fwd3
=====================================
...
`fwd3` is kept on every graded row and in every CSV. It is **not** what the
headline comparison or the decision rule below are computed on. Both run on
**points above replacement at position**:

    par = fwd3 - replacement_level(pool fwd3 at that position, position)

with `report.replacement_level` and `report.REPLACEMENT_RANK` -- the same
definition, from the same function, that the weekly table tiers on.
```

The code behind it:

- `src/ledger.py:641` — every graded pick carries `"par": got - level`.
- `src/ledger.py:670` — the week/arm rollup aggregates `mean_par=("par", "mean")`
  and keeps `mean_fwd3` beside it for reference only.
- `src/ledger.py:687` — the head-to-head benchmark series is
  `weekly[weekly["arm"] == "naive"] ... ["mean_par"]`, i.e. beating naive is
  decided on PAR.
- `src/ledger.py:746-747` — `def paired_differences(..., value: str = "mean_par")`;
  the paired `prompt` vs `repo` difference the verdict reads defaults to PAR.
- `src/ledger.py:698-700` — comment at the head-to-head: *"Head to head on PAR,
  matching the headline. On raw points this counts a week the arm won only by
  having claimed a quarterback."*

The quarterback-stockpiling failure you describe is exactly the one the module
was rewritten to close, and the rewrite is on `main` (PR #16, commit `dcb5f4c`
"Score on points above replacement, and ask whether the model beats a sort").

Confirmed live: `make ledger` on the one logged week prints

```
  arm  weeks  n  mean_par  mean_fwd3  ceiling_share  beat_naive_share
naive      1  3     1.982     10.660          0.526               NaN
 repo      1  3     4.794     10.522          0.519             1.000
```

Note the two columns disagree in direction — on raw points naive is ahead
(10.66 vs 10.52), on PAR repo is ahead (+4.79 vs +1.98) — on the same three
picks. That is the metric change doing its job on a single week.

### The one thing worth knowing about it

`NAIVE_MARGIN`, `TIE_BAND`, `REPO_MARGIN` (1.5 / 1.0 / 1.5 ppg) were
pre-registered against raw `fwd3` and now apply to PAR. They were deliberately
**not** re-picked. Over the 156-week replay the same picks move −2.30 → +1.74 ppg
on the metric change alone, so **1.5 ppg is a materially easier bar on PAR than
it was on raw points**. The tension is recorded in three places
(`src/ledger.py` docstring "AMENDMENT, 2026-08-31",
`docs/comparison_protocol.md`, `outputs/diagnostics/par_rescore_and_ablation.md`
§1) rather than silently patched. Not blocking — but it is the one number in the
comparison that is knowingly mis-calibrated, and you may want to re-register it
deliberately before Week 1 rather than after seeing a result.

---

## 2. The heuristic ablation: it ran

`outputs/backtests/03_par_rescore_ablation.py` →
`outputs/diagnostics/par_rescore_and_ablation.md` §2. Within position, scored on
PAR, over the same 156 walk-forward weeks (2014–2025, weeks 2–14). Arms are
exactly the ones you specified: `eb_share` (eb_car_share for RB, eb_tgt_share
for WR/TE), `opp` (= `wopr_opp`), `snap`, and `hot_hand_pos` (prior-week points
sorted *within* position — the naive arm restricted to the same pool).

**Positive means the model is ahead.**

| arm | sorts on | positions | k=1 | k=3 | k=5 |
| --- | --- | --- | :---: | :---: | :---: |
| `hot_hand_pos` | prior-week fantasy points, within position | QB/RB/TE/WR | +0.15 [−0.45, +0.73] | **+0.40** [+0.13, +0.67] | +0.42 [+0.22, +0.62] |
| `eb_share` | shrunk carry share (RB) / target share (WR/TE) | RB/TE/WR | −0.03 [−0.57, +0.50] | **+0.36** [+0.11, +0.60] | +0.30 [+0.13, +0.47] |
| `opp` | `wopr_opp` = carries + 2.5 × targets | QB/RB/TE/WR | +0.09 [−0.43, +0.60] | **+0.64** [+0.40, +0.88] | +0.61 [+0.43, +0.78] |
| `snap` | snap share (`offense_pct`) | QB/RB/TE/WR | +1.10 [+0.53, +1.67] | **+0.85** [+0.59, +1.11] | +0.67 [+0.51, +0.84] |

Rank by rank, the margin is one column — rank 3:

| arm | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| `hot_hand_pos` | +0.15 | +0.14 | **+0.93** [+0.40, +1.46] | +0.41 | +0.47 |
| `eb_share` | −0.03 | +0.17 | **+0.93** [+0.43, +1.44] | +0.19 | +0.25 |
| `opp` | +0.09 | **+0.60** | **+1.23** [+0.68, +1.82] | **+0.75** | +0.38 |
| `snap` | **+1.10** | **+0.64** | **+0.82** [+0.29, +1.35] | +0.50 | +0.31 |

### The conclusion, as written

> **KEEP THEM, NARROWLY — FOR THE LIST, NOT THE TOP NAME.**

Unpacked, because the hedges are the content:

- **At k=1 the model is not better than a one-liner.** Against 3 of the 4 arms
  it is indistinguishable (`eb_share` −0.03, `opp` +0.09, `hot_hand_pos` +0.15,
  all intervals ~1.1 ppg wide and covering zero). `eb_share` names as good a
  first player as the model does — −0.16 vs −0.19 mean PAR on the pool it covers.
- **At rank 3 the model is ahead of all four**, and no single season carries it
  (leave-one-season-out swings less than the margin).
- **By rank 5 it is gone** — 4 of 4 intervals cover zero.
- **Read the count as 3, not 4.** `snap`'s margin is carried by a QB cell where
  its key ties at the cut in ~every week, so the pick is the alphabet's. Drop QB
  and `snap` falls to +0.41 [−0.15, +0.96], a null.
- **The four are not four replications** — same model arm, same 156 weeks,
  correlated difference series. And each arm's deficit sits at a different
  position, which is weaker corroboration than four arms losing together.
- **PAR vs raw `fwd3` changes nothing here, exactly.** Within a position the
  baseline is one constant per (season, week, position), so every paired
  difference is untouched; the largest disagreement measured is 5.3e−15 ppg, and
  `check_par_invariance()` fails the run if that ever stops holding.

### So: is `models/` deletable?

**On this evidence, no — but the margin is thin enough that the question is
live.** The advantage exists only over a ranked list of ~3 names, which is what
`assign_tiers` actually emits (two burn, three fallback, four watch), so it is
the right quantity. It does not exist for the single best name, and it does not
grow with depth. The write-up says so itself: *"Taken as a single claim per
position, the model is replaceable by a one-liner. Taken as a ranked list … it
is not."* If you ever move to a one-name-per-week workflow, the models,
the joblib bundles, the version-pinning assertions and the retrain discipline
all become deletable on this measurement.

Figures are reliable to about ±0.3 ppg (`HistGradientBoostingRegressor` bin-edge
sensitivity). The second decimal place is not real.

---

## 3. Three-arm comparison infrastructure

**All three pieces exist.**

**`outputs/ledger/claims.csv`** — schema (`src/report.py:62`, `src/log_claim.py:46`):

```
season,week,tier,action,player,position,dropped,rationale,arm,rank_within_arm,logged_at,contaminated
```

`arm`, `rank_within_arm` and `logged_at` are all present, plus `contaminated`.
Arms are `["naive", "prompt", "repo"]` (`src/ledger.py:163`), with
`LOGGED_ARMS = ["prompt", "repo"]` — naive is derived from the pool, never
logged. `ARM_DEPTH = 3`.

Current contents: **9 rows, all `arm=repo`, all 2025 wk08**, stamped
`2026-08-30T19:30:44+00:00`, `contaminated` blank. Five carry a
`rank_within_arm` (1–5); the four `watch` rows carry none, by design.
**No `prompt` arm has ever been logged**, so there is no paired difference and
no verdict yet.

**`docs/comparison_protocol.md`** — exists, 8 sections. Contamination rule:

> **The `prompt` arm must be produced and logged BEFORE the repo's candidate
> table is opened, in a separate session.**

with the explicit forbidden-reads list (`outputs/weekly/…csv`,
`outputs/reports/…md`, `data/processed/panel.csv`, model scores, tiering, and a
prior week's report if you are about to name something from it), the
`CONTAMINATED=1` escape hatch, and the three order states
(clean / unverified / contaminated). Pre-registered thresholds are stated at
lines 157–159 and are the same 1.5 / 1.0 / 1.5 ppg the code uses, with the
PAR amendment noted at line 163.

**`make log-claim`** — exists, `src/log_claim.py` (255 lines), wired in the
Makefile:

```
make log-claim ARM=prompt PLAYERS="Rashee Rice, Ty Johnson"
```

with `SEASON`/`WEEK` defaulting to the schedule, positions defaulting to the
panel, inline position override (`"Ty Johnson:RB"`), `WHY=`, and
`CONTAMINATED=1`. `ARM` and `PLAYERS` are checked in the Makefile rather than
argparse so the error names the make variable you typed.

### One error in the protocol doc — recorded, not fixed

`docs/comparison_protocol.md:125` says:

> `.github/workflows/weekly.yml` runs `make report` at 06:00 UTC on Tuesday,
> hours before anyone is awake to write down a prompt pick

**It does not.** The workflow runs `make install`, `make test`, `make data`,
`make panel`, `make weekly` and nothing else, and its commit step *hard-fails*
if anything outside `outputs/weekly/**` and `data/raw/MANIFEST.json` was
touched. `README.md:136-145` documents this correctly (*"never runs:
`make models`, `make report`, `make ledger`"*). So the protocol doc's
justification for treating `unverified` as the normal case is wrong: there is no
CI-generated repo arm. The repo arm is hand-run, which means in practice you
control both timestamps and `unverified` should be *rare*, not routine. Not
blocking — but it means `make ledger STRICT_ORDER=1` is a cheaper check than the
doc implies, and it also means **nothing logs the repo arm for you each week**.

---

## 4. Roster wiring

**`data/roster.yaml` does not exist.** Only `data/roster.example.yaml` is
present, and `data/roster.yaml` is listed in `.gitignore:16` — deliberately, so
your roster and record never get committed. `src/report.py:60` sets
`DEFAULT_ROSTER_PATH = ROOT / "data" / "roster.yaml"` and picks it up
automatically when present.

Both runs, on this clone:

**`make report SEASON=2025 WEEK=8`** (no roster) — succeeds, 444 words:

```
- `[KEEPER]` ADD Rashee Rice / DROP ??? — 9 targets and 28% target share; 1.6+ pts/wk (upside unbounded).
- `[STARTER]` ADD Ty Johnson / DROP ??? — 32% snaps; 0.8+ pts/wk (upside unbounded).
...
- `DROP ???` — wire up a roster to resolve the other half.
```

plus `_No roster configured — injury, bye and role checks skipped._`

**`make report SEASON=2025 WEEK=8 ROSTER=data/roster.example.yaml`** — succeeds,
462 words, **drops resolve**:

```
- `[KEEPER]` ADD Rashee Rice / DROP Samaje Perine (RB) — 9 targets and 28% target share; 1.6+ pts/wk (upside unbounded).
- `[STARTER]` ADD Ty Johnson / DROP Tez Johnson (WR) — 32% snaps; 0.8+ pts/wk (upside unbounded).
- `[KEEPER]` ADD Cade Otton / DROP Samaje Perine (RB) — 21% target share and 5 targets; 2.8+ pts/wk (upside unbounded).
- `[STARTER]` ADD Sean Tucker / DROP Tez Johnson (WR) — 41% carry share and 12 carries; 0.6+ pts/wk (upside unbounded).
- `[STARTER]` ADD Cam Ward / DROP Gunnar Helm (TE) — 100% snaps; 11.1+ pts/wk (upside unbounded).
```

and the roster check fires (Goedert bye next week, Bijan Robinson snaps down
21%, Tez Johnson questionable). **This run reproduced the committed
`outputs/reports/2025/wk08.md` byte-for-byte**, which also tells you the report
on `main` was generated against the *example* roster, not a real one.

So: the machinery works end to end; the only missing piece is your actual team.
`DROP ???` is not a bug, it is the no-roster path.

Note (documented, not a defect): drop candidates restart per tier
(`src/report.py:424-427`) — burn #1 and fallback #1 both propose the same drop,
because the tiers are alternatives rather than a shopping list. If you ever
claim two names in one week, the second drop suggestion needs your own judgement.

**This is the one blocking item before Week 1.** Copy
`data/roster.example.yaml` to `data/roster.yaml` and fill in your team; names
must match `player_display_name` exactly or they silently drop out of both the
roster check and the drop ranking.

---

## 5. What's actually on main

`main` = `b4acae2`, **58 commits**.

### Last 10 commit subjects

```
b4acae2 Merge pull request #18 from nicklasorte/claude/depth-charts-validation-f1e6e2
ee2c67f Depth chart walk-forward: the exploratory gain is a null, and the verdict logic earns it
b2e0cfe Diagnostics: depth charts do not survive walk-forward, and injuries never could
d7818ef Merge pull request #17 from nicklasorte/claude/depth-charts-validation-f1e6e2
1e0fd3b Fetch depth charts, build dc_* features, and stage their walk-forward validation
4f92fd5 Merge pull request #16 from nicklasorte/claude/par-scoring-model-ablation-8c2oey
3e32ffe Diagnostics: correct the PAR write-up after an adversarial recomputation
b39a545 Take the verification pass: a sign error, an incommensurable comparison, and a depth claim one more rank contradicts
5a3cd83 Diagnostics: PAR scoring, and whether the model beats a one-liner
3e6e2ec Scope the levels table to weeks 2-14 like everything else on the page
```

### Files

**`src/`** (7 modules, 3,598 lines)

| file | lines | what |
| --- | ---: | --- |
| `src/fetch.py` | 278 | hash-pinned nflverse download, `MANIFEST.json`, `data_revision()` |
| `src/features.py` | 711 | panel build: usage, shrunk shares, `kal_role`, `cusum`, `neutral_opp`, `dc_*`, `fwd3` |
| `src/models.py` | 427 | per-position `HistGradientBoostingRegressor`, conformal half-widths, bundle + card |
| `src/weekly.py` | 341 | wire pool, scoring, projection intervals, `outputs/weekly/{season}/wk{NN}.csv` |
| `src/report.py` | 553 | tiering on PAR, roster check, <500-word markdown report, claim append |
| `src/ledger.py` | 1,033 | tier ledger + three-arm comparison, PAR scoring, verdict |
| `src/log_claim.py` | 255 | `make log-claim` — hand-log an arm's ranked picks |

**`outputs/diagnostics/`** (8 files)

| file | what |
| --- | --- |
| `dryrun_2025wk08.md` | first end-to-end dry run |
| `followup_2025wk08.md` | QB R² inversion, `proj_pts_hi` saturation, roster wiring |
| `actions_first_run.md` | the saturation fix, and getting the workflow green on a runner |
| `comparison_setup.md` | three-arm schema, backfill, scoring, pre-registered rule |
| `season_replay_2022_2025.md` | 4-season replay: the ranking loses to naive |
| `walkforward_2014_2025.md` | 12-season walk-forward; 93% of naive's margin is position mix |
| `par_rescore_and_ablation.md` | PAR re-score + the heuristic ablation (§1–2 above) |
| `depth_charts.md` | depth charts are a null on walk-forward; injury reports too |
| `repo_state.md` | this file |

**`outputs/backtests/`** — 5 scripts (`00_input_importance.py`,
`01_season_replay.py`, `02_walkforward_2014_2025.py`,
`03_par_rescore_ablation.py`, `04_depth_charts_walkforward.py`) plus persisted
results: `replay/`, `replay_full/` (+`unscored.json`), `par_ablation/`
(4 files incl. `checks.json`, `replacement_levels.csv`), `depth_charts_wf/`
(4 files incl. `audit.json`), and `results_input_importance.txt`.

**`models/`** — `MODEL_CARD.md`, `QB.joblib`, `RB.joblib`, `TE.joblib`,
`WR.joblib`.

**`docs/`** — `comparison_protocol.md` only.

### Merged PRs — all 18, none open

| # | one-liner |
| ---: | --- |
| 1 | Scaffold repo; hash-pinned nflverse fetch + `MANIFEST.json` |
| 2 | Build the backward-looking player-week panel |
| 3 | Reproduce the input-importance backtest |
| 4 | Fit and persist the per-position ranking models |
| 5 | Weekly candidate table, schedule-derived week resolution, CI |
| 6 | Report writer and claim ledger |
| 7 | Fix `make install` on managed Python; refresh pipeline artifacts |
| 8 | Build `neutral_opp` from pbp; wire up hand-maintained `roster.yaml` |
| 9 | Print a saturated projection ceiling as unbounded, not as a number |
| 10 | Three-arm comparison: naive vs prompt vs repo, pre-registered decision rule |
| 11 | Build the venv on the runner; fence what the weekly job may commit |
| 12 | Pin the environment to the fitted models; stop the weekly noise commit |
| 13 | Diagnostics: the run on main, and what its commit contains |
| 14 | Walk-forward replay 2022–2025: the ranking loses to the naive benchmark |
| 15 | Walk-forward replay 2014–2025: naive wins the headline, but 93% is position mix |
| 16 | **Score on points above replacement, and ask whether the model beats a sort** |
| 17 | Depth charts as a structured news feature: fetch, `dc_*` columns, validation |
| 18 | Depth chart walk-forward: results, and the verdict hardening a review demanded |

**0 open pull requests.**

### Branches

12 on the remote. **Every one is an ancestor of `main`** — nothing unmerged is
sitting anywhere. The 11 non-`main` branches are spent PR branches that were
never deleted:

```
claude/weekly-waiver-scaffold-99g1mw       (PRs 1-6)
claude/pipeline-dry-run-zqatie             (PR 7)
claude/qb-diagnostics-roster-62d7dv        (PR 8)
claude/proj-pts-hi-actions-run-9s3p7v      (PR 9)
claude/three-arm-comparison-ledger-sro8mz  (PR 10)
claude/weekly-workflow-repair-mqf0jt       (PR 11)
claude/workflow-deps-noise-commit-m6x9aq   (PRs 12-13)
claude/walk-forward-backtest-2022-2025-acjqcf (PR 14)
claude/walkforward-backtest-2014-2025-yfkijg  (PR 15)
claude/par-scoring-model-ablation-8c2oey   (PR 16)
claude/depth-charts-validation-f1e6e2      (PRs 17-18)
```

Safe to delete whenever you feel like tidying; none carries unique work.

---

## 6. Does it still run

Full pipeline plus tests, on a clean clone of `main`, Python 3.12.3.

| step | result | notes |
| --- | --- | --- |
| `make install` | **PASS** | venv built, `requirements.txt` pins installed |
| `make data` | **PASS** | 54 files in manifest; 18 downloaded, 3 unavailable (2026 assets not published yet — reported, not an error) |
| `make panel` | **PASS** | 21,293 rows with `fwd3`, 19,498 with `fwd3_played` |
| `make models` | **PASS** | 4 bundles + `MODEL_CARD.md`; R² RB 0.262 / WR 0.325 / TE 0.382 (⚠ over-covers, 0.830) / QB 0.428 |
| `make weekly SEASON=2025 WEEK=8` | **PASS** | 134 wire candidates → `outputs/weekly/2025/wk08.csv` |
| `make test` | **PASS** | **Ran 105 tests in 21.8s — OK**, 0 failures, 0 errors |
| `make report SEASON=2025 WEEK=8` | **PASS** | (extra) 444 words no-roster / 462 with example roster |
| `make ledger` | **PASS** | (extra) grades the one logged week, prints "nothing to pair" |

No step failed, so there is no error text to report.

### Rashee Rice sanity check — confirmed

2025 Week 8, top 5 WR:

```
 pos_rank player_display_name team  score  proj_pts_lo  proj_pts_hi   snap  targets  tgt_share  carries  carry_share    pts
        1         Rashee Rice   KC  0.828        1.567       14.806  0.860        9      0.281        2        0.067 21.000
        2    Christian Watson   GB  0.710        0.800       14.806  0.560        4      0.114        0        0.000 10.500
        3      Darnell Mooney  ATL  0.704        0.800       14.806  0.900        4      0.148        0        0.000  1.600
        4        Jahan Dotson  PHI  0.638        0.467        8.433  0.710        2      0.100        0        0.000 10.500
        5       Tyler Johnson  NYJ  0.626        0.400        7.833  0.780        5      0.172        0        0.000 13.900
```

**WR #1, score 0.828**, and he is the `burn` claim in the report.

### Reproducibility, unasked but worth having

After the full rebuild, `git status` showed **`outputs/weekly/2025/wk08.csv`
unchanged** and **`outputs/reports/2025/wk08.md` unchanged** against `main`.
A from-scratch refit reproduces the shipped Week 8 table and report bit for bit,
and `make ledger` reproduced `arm_grades.csv` / `arm_summary.csv` / `grades.csv`
identically too. Only four things moved, all of them expected:

- `models/*.joblib` — different bytes, identical numbers (every R² and
  half-width in the card is unchanged).
- `models/MODEL_CARD.md` — trained-date and data-revision lines only.
- `data/raw/MANIFEST.json` — nflverse republished `games.csv` (+3 bytes) and
  `depth_charts_2026.csv` now exists. This churn is exactly what the manifest is
  for.
- `outputs/weekly/LAST_MANIFEST.json` — records the above.

All four were restored to their committed bytes; this audit changed nothing.

---

## 7. Known gaps

Everything documented as a deliberate deferral or known issue, plus what this
audit turned up. Flag is **blocking before Week 1** or **not**.

| # | gap | flag |
| ---: | --- | --- |
| 1 | **No `data/roster.yaml`.** Only the example file exists (real file is gitignored by design). Without it every claim prints `DROP ???` and injury/bye/role checks are skipped. | **BLOCKING** |
| 2 | **PAR thresholds are knowingly mis-calibrated.** 1.5 / 1.0 / 1.5 ppg were pre-registered on raw points and now apply to PAR, where the same 156 weeks move −2.30 → +1.74 on the metric change alone. Deliberately left, so re-picking them post-hoc is not an option — but a *deliberate* re-registration now, before any data, still is. | not blocking (decide now, not in week 12) |
| 3 | **`pts` is not a model feature, `pts_lag1` is.** Week-W points are in the panel, printed next to every candidate, and out-correlate their own lag with `fwd3` at every position — but `BASE_FEATURES` carries only the lag. Adding it in replay recovers ~0.5 ppg and does not close the naive gap. Left because fixing it means a deliberate retrain and re-card. | not blocking |
| 4 | **`proj_pts_hi` saturation — FIXED, contrary to what you may remember.** It was documented-only after PR #8; PR #9 shipped the presentation fix. Verified live: `src/report.py:183-212` (`ceiling_saturated`), and the regenerated Week 8 report prints `1.6+ pts/wk (upside unbounded)`. The clip, the intervals and `outputs/weekly/*.csv` are unchanged. | resolved |
| 5 | **CPython patch level is not pinned.** `requirements.txt` pins the libraries; the workflow asks `setup-python` for `3.12` and takes whatever patch the runner holds. Bundles record the patch but the load check does not assert it (a runner cannot honour a pin it cannot satisfy). | not blocking |
| 6 | **`git push` in the workflow has no retry or rebase.** `concurrency` stops the job racing itself, not a human pushing to `main` during its ~45s window. Fails loudly — but *after* doing all the work. | not blocking |
| 7 | **`actions/checkout@v4` and `actions/setup-python@v5` run on Node 20**, which the runner force-upgrades to Node 24 with a deprecation warning. Harmless today. | not blocking |
| 8 | **`make report` and `make ledger` are still excluded from the workflow — confirmed.** The job runs install/test/data/panel/weekly and its commit step hard-fails on any write outside `outputs/weekly/**` and `data/raw/MANIFEST.json`. Deliberate (automating the report would hand a scheduled job write access to the ledger that grades it) — but it means **nothing produces your weekly report or logs the repo arm unless you type the command.** | not blocking, but build the habit |
| 9 | **`docs/comparison_protocol.md:125` states the workflow runs `make report` at 06:00 UTC. It does not.** See §3. The doc's argument that `unverified` ordering is "the normal case" rests on that false premise. | not blocking (doc only) |

### Additional findings from this audit

| # | finding | flag |
| ---: | --- | --- |
| 10 | **`outputs/diagnostics/comparison_setup.md:191-204` ("Current state") is stale.** It still shows the pre-PAR table (`mean_fwd3` only, `beat_naive_share 0.000`) and the narrative "naive is ahead." The live ledger now reports `mean_par` with `beat_naive_share 1.000` on the same week. §Scoring and §Pre-registered rule *were* updated for PAR; only this closing section was missed. | not blocking (doc only) |
| 11 | **The shipped `models/*.joblib` carry a stale data revision.** Their card records a revision over **13** manifest entries; `main`'s manifest now has **53** (13 → 40 at the 2014–2025 backfill, → 53 at depth charts). The revision is recorded, never asserted at load — only library versions are. Refitting against the current manifest reproduces every R², half-width and the Week 8 table exactly, so this is bookkeeping staleness, not numeric drift. `dc_*` are correctly absent from `BASE_FEATURES`/`OPTIONAL_FEATURES` — the depth-chart walk-forward was a null and the feature was not kept. | not blocking |
| 12 | **QB's headline R² is inflated by the "did he even play" signal.** `outputs/diagnostics/followup_2025wk08.md` §1: rebuilt on `fwd3_played`, QB drops 0.435 → 0.266 while every other position moves ≤0.031. QB is the *weakest*-fitting position once that signal is removed, not the strongest. The model card's 0.428 is not wrong, but it does not mean what it looks like. Documented; no code change proposed. | not blocking |
| 13 | **TE conformal interval over-covers** (0.830 against a 0.80 target, flagged with ⚠️ in the card by design). Reproduced on this run. | not blocking |
| 14 | **No `prompt` arm has ever been logged.** The comparison has one week of one arm. Until a `prompt` claim is logged in a clean session, `make ledger` prints "nothing to pair" and no verdict exists. The infrastructure is ready; the discipline has not started. | not blocking — but it is the thing Week 1 is for |
| 15 | **Drop suggestions restart per tier** (`src/report.py:424-427`, documented as intentional): burn #1 and fallback #1 propose the same drop, because tiers are alternatives rather than a shopping list. Claiming two names in one week needs your own judgement on the second drop. | not blocking |

---

## What this audit did not touch

No code was changed, no bug was fixed, no PR was opened. The pipeline was run
and its regenerated artifacts restored. The only file added to the repository is
this one.

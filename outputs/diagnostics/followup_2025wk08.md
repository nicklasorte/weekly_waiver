# Follow-up: QB R² inversion, `proj_pts_hi` saturation, roster wiring

Answers to the three questions, the `neutral_opp` build, and a regenerated
2025 Week 8 report with a sample roster wired up. Data revision for every
number below (except the pre-existing `results_input_importance.txt`
appendix, computed earlier): `096fe02f45c5e7a4845c4f243c85797afed5170e5078bd00756ab48f066c16a2`
(`data/raw/MANIFEST.json` after a fresh `make data` run today; only
`games.csv` differs from the run that shipped the current model card, since
that file legitimately changes every week).

---

## 1. The QB R² inversion

**Confirmed: the hypothesis is right.** QB's headline R² is inflated by the
"did he even play" signal that `fwd3`'s zero-for-inactive convention creates,
and QB is not a strong-fitting position once that signal is taken away — it
becomes the weakest.

### fwd3 vs. fwd3_played, same rank objective, same split

Re-running `models.holdout_r2` (2022-24 train / 2025 test, the exact
methodology behind the model card) with the rank built on `fwd3_played`
instead of `fwd3`, everything else identical:

| position | R² on fwd3 (shipped) | R² on fwd3_played | Δ |
| --- | ---: | ---: | ---: |
| RB | 0.259 | 0.269 | +0.010 |
| WR | 0.325 | 0.344 | +0.019 |
| TE | 0.382 | 0.351 | −0.031 |
| QB | 0.435 | 0.266 | **−0.169** |

QB falls off a cliff; every other position moves by three hundredths of a
point or less. Under `fwd3_played`, QB (0.266) is no longer the best-fitting
position — it is tied for worst with RB (0.269), and both skill positions
(WR, TE) sit clearly above it.

### The same shipped model, evaluated only on rows where the player actually played

A second, sharper cut: take the model actually shipped (trained and ranked
on `fwd3`, exactly as in `models/QB.joblib` etc.), and instead of changing
the target, just restrict the *test-set evaluation* to player-weeks where
`fwd3_played` is not null — i.e., score the shipped QB model only on
quarterbacks who actually took snaps over the following three weeks:

| position | R² on full 2025 test | R² on played-only rows | frac. of test rows scored 0 |
| --- | ---: | ---: | ---: |
| RB | 0.259 | 0.220 | 28.7% |
| WR | 0.325 | 0.322 | 32.5% |
| TE | 0.382 | 0.334 | 23.9% |
| QB | **0.435** | **0.192** | 25.6% |

This isolates the mechanism directly. WR barely moves (0.325 → 0.322) even
though a third of its test rows are zeros — the model's skill among WRs who
play is basically the whole story already. QB moves from best (0.435) to
worst (0.192) of the four positions the instant the zeros are taken out of
the evaluation. The shipped QB model's headline number is mostly a
"will this guy take a snap" detector riding on `snap` share, which is a
near-deterministic signal (a wire QB is either the starter or he is not);
it is not evidence the model resolves QB scoring differences better than it
resolves WR or TE ones. If anything, among quarterbacks who do play, this
is the hardest position to rank.

This also reframes the "independent backtest had QB weakest" premise: the
backtest in `outputs/backtests/00_input_importance.py` regresses on raw
`fwd3` points directly (not the percentile rank the shipped models use), and
under that setup RB is the weakest position (R²=0.238), not QB. But its own
sensitivity appendix shows the same underlying effect from a different
angle: switching that backtest's raw-points target from `fwd3` to
`fwd3_played` also craters QB specifically (0.286 → 0.138, roughly halved)
while the other three positions move by 0.01–0.04. Whichever backtest or
methodology you use, QB is the one position whose apparent fit is almost
entirely a byproduct of the zero-for-inactive convention.

### Recommendation

**Keep training and serving on `fwd3`.** The zero-for-inactive convention is
not a modeling artifact to be fixed — it is what a waiver claim actually
returns, and a QB model that could not tell "will play" from "will not play"
would be strictly worse for the decision this report supports. Switching the
*served* target to `fwd3_played` would silently stop penalizing a claim on a
QB who is about to lose his job or get benched, which is exactly the outcome
a claim needs to be protected against.

What should change is how the number is **presented**. Right now the model
card's single R² column invites reading "QB: 0.435, our best-fitting
position" as "the QB projections are our most trustworthy ones." They are
not — among plausible plays, QB is the position this pipeline explains
least well (R²≈0.19–0.27 by either cut above), and the skill positions (WR,
TE) are where the model is doing real differentiation. Recommend adding a
played-only (or `fwd3_played`) R² column alongside the shipped one in the
model card specifically so QB's number is not read as a sign of unusual
model quality there. No code change made for this yet — flagging it for a
decision before touching `write_model_card`.

---

## 2. `proj_pts_hi` — derivation and saturation

**Confirmed: the ceiling is degenerate for most of the group the report
recommends, and it is happening for exactly the reason suspected.**

### Derivation

In `src/weekly.py`:

1. `score_hi = (score + conformal_half_width).clip(0, 1)` — the top of the
   model's 80% interval in rank space.
2. `points_scale(panel, position)` builds the position's empirical
   distribution of `fwd3` over the training universe (weeks 2-14, on the
   wire), sorted.
3. `proj_pts_hi = np.quantile(scale, score_hi.clip(*PROJECTION_CLIP))`, where
   `PROJECTION_CLIP = (0.01, 0.99)`.

The `0.99` clip in step 3 exists on purpose (the docstring on
`points_scale` explains it): without it, any `score_hi` that saturates at
`1.0` maps to the single highest `fwd3` ever recorded at the position, and
the range would read as "the record," not a forecast. The clip caps that at
the 99th-percentile outcome instead — a real, attainable number, but a
*fixed* one that many different `score_hi` values collapse onto.

### The saturation, quantified on 2025 Week 8

Any player whose `score + half_width ≥ 0.99` maps to the identical
99th-percentile `proj_pts_hi` for the position, regardless of how much
higher their score is than a teammate's. Counting how many of that week's
wire candidates hit this exactly:

| position | half-width | score threshold to saturate | saturated / total candidates | shared ceiling |
| --- | ---: | ---: | ---: | ---: |
| QB | 0.314 | 0.676 | 7 / 15 | 25.798 |
| TE | 0.286 | 0.704 | 9 / 39 | 16.463 |
| RB | 0.303 | 0.687 | 2 / 28 | 16.804 |
| WR | 0.289 | 0.701 | 3 / 52 | 14.806 |

(Half-widths shown are after the `neutral_opp` refit in section 4 below;
the effect and its size are the same under the previously-shipped models
that produced the numbers in the original question.)

This is precisely what was flagged: **7 of the week's 15 QB candidates —
including the entire top tier the report recommends — print the exact same
25.798 upper bound.** WR ranks 2 and 3 sharing 14.806 is the same mechanism
catching two scores (0.702 and 0.702, both ≥ the 0.701 threshold) at once.
The number is not wrong, exactly — it is a genuine 99th-percentile outcome
for the position — but it is not *that player's* upper bound. It is the same
number eight other players in this table would also get, and it happens
disproportionately to the group the report is built to recommend (high
score → more likely to clip), which is the worst place for false precision
to live.

### Recommendation

Do what was suggested rather than inventing a more clever number: when
`score_hi` is at or above the clip boundary, stop printing a point figure
for the top of the range and say plainly that the method does not bound it,
e.g. `1.6–14.8+ pts/wk (top of the wire; upside not bounded by this
method)`. That is honest about exactly what's known (the floor, from
`score_lo`, is unaffected by any of this) and what isn't (the ceiling, once
the model is confident enough to blow past the 99th percentile of history).
Not implemented yet — this is a presentation decision for `report.py` and/or
`weekly.py`, flagged rather than made.

---

## 3. `neutral_opp` status

Confirmed as described: `neutral_opp` has been sitting in
`src/models.py`'s `OPTIONAL_FEATURES` since before this session, with a
comment stating it was "not built yet," while `src/features.py` never
produced the column. Every model shipped to date was fit on the 13
`BASE_FEATURES` only. The play-by-play files were fetched into
`data/raw/play_by_play_{2022..2025}.csv.gz` (~19MB each, ~76MB total) and,
before this session, were not read by anything in `src/` — `features.py`
only ever joined `stats_player_week_*` and `snap_counts_*`. Both gaps are
closed in section 4.

---

## 4. Task A — `neutral_opp` built, refit, and measured

Added to `src/features.py`: `load_neutral_opp()` reads
`play_by_play_{year}.csv.gz` for each season, keeps regular-season plays
with `wp` (the possession team's pre-snap win probability — known before the
snap, so no lookahead) between 0.20 and 0.80, and counts targets
(`pass_attempt==1` with a `receiver_player_id`) plus carries
(`rush_attempt==1` with a `rusher_player_id`) per player-week. A season
whose pbp file is missing is skipped and reported, and left `NaN` for that
season rather than fabricated as zero (`HistGradientBoostingRegressor`
handles missing values natively, so this costs nothing at fit time). All
four seasons (2022-2025) had their pbp file on disk this run, so
`neutral_opp` is fully populated with real zeros, not NaNs, in the current
panel.

No other previously-dead input was added — implied team total, opponent
defense vs. position, team EPA, pace, PROE, vacated opportunity, red-zone
counts and aDOT were all left out as instructed.

### Result: null. `neutral_opp` does not move R² materially at any position.

| position | R² without neutral_opp | R² with neutral_opp | Δ | permutation importance | rank of 14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RB | 0.259 | 0.262 | +0.003 | 0.0073 | 8 |
| WR | 0.325 | 0.325 | +0.000 | −0.0007 | 12 |
| TE | 0.382 | 0.382 | +0.000 | 0.0025 | 9 |
| QB | 0.435 | 0.428 | −0.007 | 0.0058 | 6 |

Permutation importance (25 repeats, 2025 test set, model trained through
2024) puts `neutral_opp` in the bottom half of all 14 features at every
position, and its importance is within noise of zero everywhere except a
small, still-marginal positive contribution at RB. This flatly contradicts
the "~2x top-decile lift at every position" result from earlier testing
referenced in the brief — whatever produced that number, it is not
reproducing here against this panel, this universe, and this model.

Recorded as a null result per instructions, not silently dropped. Two
candidate explanations, neither chased further here since it wasn't asked
for: (1) `carries`, `carry_share`, `wopr_opp` and `eb_car_share` already
capture most of what "real offensive opportunity" would add on top of raw
volume, so a neutral-script cut of the same volume may be largely redundant
once those are already in the model; (2) the earlier "~2x lift" result may
have come from a univariate cut (neutral_opp vs. baseline fwd3, no other
features controlling for role) rather than the multivariate, held-out-season
setup used here — the same gap the `00_input_importance.py` backtest exists
to catch for other inputs. Left in the panel and in `OPTIONAL_FEATURES`
regardless of the null result, since it costs nothing to carry and the
question of whether it helps is exactly the kind of thing worth being able
to re-check against a future season without re-deriving it from pbp again.

The model card, weekly table and joblib files in this PR reflect models
refit with `neutral_opp` included (14 features). Conformal half-widths and
coverage shifted slightly as a normal consequence of refitting (TE's
coverage moved from 0.827 to 0.830, just past the ±0.03 flag threshold —
noted here as a coincidence of this particular refit, not a `neutral_opp`
effect worth a change on its own).

---

## 5. Task B — roster wired up

`src/report.py` already had `--roster` and full YAML/JSON support; the gap
was that nothing pointed at it by default. Changes:

- `DEFAULT_ROSTER_PATH = data/roster.yaml`. `python -m src.report` /
  `make report` now load it automatically when it exists, with no flag
  needed; `--roster <path>` still overrides it, and behavior with no roster
  file anywhere is unchanged (blank roster check, `DROP ???`, "record not
  configured").
- `pyyaml>=6.0` added to `requirements.txt` — the YAML path previously
  required it but it was never installed by `make install`.
- `data/roster.yaml` added to `.gitignore`: it is personal (your roster, your
  record), not a build artifact, matching the repo's existing stance of
  never putting credentials or personal league data in the tree.
- `data/roster.example.yaml` committed, documented inline: field-by-field
  comments on `name` (must match `player_display_name` exactly), `position`
  (QB/RB/WR/TE plus K and DST/D-ST for the mandatory slots), `status`
  (optional, drives the roster check), and `note` (free text, not read by
  the code).
- The mandatory-slot rule (`droppable()` in `report.py`) was already
  enforced in code before this change — never proposes dropping a K or
  DST/D-ST that is the roster's only one at that position. Verified below
  rather than re-implemented.

No ESPN-cookie integration was added or considered, per instruction.

### Verified against a sample roster (real 2025 players, fictional team)

15-man roster: 9 clean starters plus a bench built to exercise all three
roster-check paths at once — one player on a bye-week-9 team, one with a
real week-8 snap share drop, one with a hand-entered `status`. It ran locally
at `data/roster.yaml`, which is gitignored by design and so is not part of
this PR; its exact contents (`record: "5-2"`, same 15 players) are reproduced
verbatim in the committed `data/roster.example.yaml`, so the demonstration
below is fully reproducible by copying that file to `data/roster.yaml`.

- Record `5-2` appears in the horizon line, replacing "record not
  configured."
- Roster check produced all three flag types from real week-8 data plus one
  hand-entered status, nothing fabricated beyond the roster file itself:
  - **Dallas Goedert** — bye next week (Philadelphia is on bye in week 9)
  - **Bijan Robinson** — snaps down 21%, role slipping (real week-8
    `snap_jump`)
  - **Tez Johnson** — questionable (hand-entered `status` in the roster
    file)
- Every `DROP ???` resolved to a real bench name (Samaje Perine, Tez
  Johnson, Gunnar Helm across the three burn/fallback claims) — the
  kicker (Harrison Butker) and defense (Denver Broncos) never appear as a
  drop candidate, confirming the mandatory-slot rule holds with the roster
  wired in, not just in the code path that skips it when unset.

---

## Regenerated report: 2025 Week 8, with the sample roster

```markdown
# 2025 Week 8 — switching

Week 8, 5-2, **switching** mode: balance the playoff push against players who still hold value.

## Roster check

- **Dallas Goedert** — bye next week
- **Bijan Robinson** — snaps down 21%, role slipping
- **Tez Johnson** — questionable

## Top of the wire

- **Ty Johnson** (RB, BUF) — 32% snaps; 0.8–16.8 pts/wk
- **Sean Tucker** (RB, TB) — 41% carry share and 12 carries; 0.6–16.8 pts/wk
- **Rashee Rice** (WR, KC) — 9 targets and 28% target share; 1.6–14.8 pts/wk
- **Christian Watson** (WR, GB) — 38% air yards and 56% snaps; 0.8–14.8 pts/wk
- **Cade Otton** (TE, TB) — 21% target share and 5 targets; 2.8–16.5 pts/wk
- **Pat Freiermuth** (TE, PIT) — 4 targets and 14% target share; 2.5–16.5 pts/wk
- **Cam Ward** (QB, TEN) — 100% snaps; 11.1–25.8 pts/wk
- **Dillon Gabriel** (QB, CLE) — 100% snaps; 11.0–25.8 pts/wk

## Claims

**Burn the claim**

- `[KEEPER]` ADD Rashee Rice / DROP Samaje Perine (RB) — 9 targets and 28% target share; 1.6–14.8 pts/wk.
- `[STARTER]` ADD Ty Johnson / DROP Tez Johnson (WR) — 32% snaps; 0.8–16.8 pts/wk.

**Claim if tier 1 fails**

- `[KEEPER]` ADD Cade Otton / DROP Samaje Perine (RB) — 21% target share and 5 targets; 2.8–16.5 pts/wk.
- `[STARTER]` ADD Sean Tucker / DROP Tez Johnson (WR) — 41% carry share and 12 carries; 0.6–16.8 pts/wk.
- `[STARTER]` ADD Cam Ward / DROP Gunnar Helm (TE) — 100% snaps; 11.1–25.8 pts/wk.

**Watch list**

- `[STARTER]` WATCH Dillon Gabriel — 100% snaps; 11.0–25.8 pts/wk.
- `[STARTER]` WATCH Christian Watson — 38% air yards and 56% snaps; 0.8–14.8 pts/wk.
- `[STARTER]` WATCH Darnell Mooney — 90% snaps and 30% air yards; 0.8–14.8 pts/wk.
- `[STARTER]` WATCH Pat Freiermuth — 4 targets and 14% target share; 2.5–16.5 pts/wk.

## Standing rules

- Recommendations only. Nothing here places a transaction.
- Claim now. Hoarding priority to Week 10 cost ~69% of a claim's value.
- Ranges are 80% conformal intervals, not projections. They are wide because the models are.
- Claims are ordered by points above replacement at the same position, not by raw model score, which is not comparable across positions.
- No drop proposed here leaves you without a K or D/ST.
```

Note the QB and WR ranges in this table are the same saturated ceilings
discussed in section 2 (25.8 for both top QBs, 14.8 repeated across multiple
WRs) — left as-is here since the presentation fix was not implemented,
consistent with "don't fix what you find until we've looked at it."

`outputs/weekly/2025/wk08.csv` and `outputs/reports/2025/wk08.md` in this PR
are the actual files this run produced, not hand-copied text — running
`make weekly SEASON=2025 WEEK=8 && make report SEASON=2025 WEEK=8` against
this branch reproduces both exactly (roster picked up automatically from
`data/roster.yaml` if present, per section 5).

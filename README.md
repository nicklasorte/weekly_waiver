# weekly_waiver

Reproducible fantasy football waiver-wire analysis. Python 3.12.

The point of the repo is that every number it produces can be traced back to the
exact bytes it came from. nflverse revises history in place as stat corrections
land, so a result is only meaningful alongside the data revision it was computed
against — hence `data/raw/MANIFEST.json`, which is committed while the data
itself is not.

## Layout

```
src/fetch.py       download nflverse data, pin every file with a sha256
src/features.py    build the backward-looking player-week panel
src/models.py      fit and persist the per-position models
src/weekly.py      score the wire pool for one season/week
src/report.py      write the weekly markdown report
src/ledger.py      grade past recommendations after the fact
src/log_claim.py   log a comparison arm's picks in one command

data/raw/          downloaded source files (gitignored, except MANIFEST.json)
data/processed/    derived panel (gitignored)
outputs/           backtests, weekly tables, reports, claim ledger
```

## Setup

```bash
make install       # create .venv and install requirements.txt into it
make data          # download 2022-2026 into data/raw/
make panel         # build data/processed/panel.csv
make models        # fit models/ and regenerate the model card
make weekly SEASON=2025 WEEK=8    # score one week's wire pool
make test          # unit tests
```

`make install` builds `.venv` and every other target runs out of it, so there
is no `activate` step. Point `PY=` at another interpreter to use an environment
you manage yourself — `make install` then installs into that environment
instead of creating a venv. `BOOTSTRAP_PY=` chooses the interpreter the venv is
built from (default `python3.12`) and `VENV=` relocates it.

Every target takes `PY=` and `SEASONS=` overrides, e.g.
`make panel SEASONS="2022 2023 2024 2025"`.

`make data` skips files already on disk; `FORCE=1 make data` re-downloads them.
Seasons that have not started yet report `not published yet` and are skipped
rather than failing the run, so the same command works before Week 1.

### Pinned dependencies

`requirements.txt` pins every direct dependency to an exact version, and each
`models/*.joblib` records the versions it was fitted under. `src/models.py`
compares the two on load and **exits** if `scikit-learn`, `numpy`, `scipy` or
`pandas` differ.

This is deliberately a hard failure. The bundles are fitted estimators committed
to the repo, and the scheduled job unpickles them without ever refitting — so a
newer scikit-learn does not crash, it scores. The result is a weekly table of
slightly different numbers with nothing to signal that anything moved, which is
worse than an error, because there is no reason to distrust it.

To move a version: change the pin, `make install`, then `make models` to refit
under it. Bumping a pin without refitting is exactly what the check stops. The
model card records the resulting versions alongside the data revision.

`python` and `joblib` are recorded in each bundle but not asserted — the CI
runner picks its own CPython patch level (see [Known gaps](#known-gaps)), and
joblib only serialises.

## Data sources

All from [nflverse](https://github.com/nflverse):

| file | source |
| --- | --- |
| `snap_counts_{year}.csv` | nflverse-data release `snap_counts` |
| `stats_player_week_{year}.csv` | nflverse-data release `stats_player` |
| `play_by_play_{year}.csv.gz` | nflverse-data release `pbp` |
| `games.csv` | `nflverse/nfldata` |

## MANIFEST.json

Written by `src/fetch.py` on every run. Per file: source url, sha256 and byte
count. If a re-fetch changes a hash for a season that is already complete,
upstream revised history and previously computed results are no longer
comparable — `fetch.py` says so on the spot.

Fetch timestamps are **not** in it. Identical bytes must produce an identical
manifest, or the scheduled job's "nothing changed, don't commit" exit never
fires and the weekly commit history becomes a heartbeat rather than a record of
change. They go to `data/raw/FETCH_LOG.json` instead, which is untracked.
`outputs/weekly/LAST_MANIFEST.json` holds digests only for the same reason: when
the data changed, git records when; when it did not, there is nothing to record.

## The panel

`src/features.py` joins weekly stats to snap counts and writes one row per
player-week to `data/processed/panel.csv`. Scoring is half PPR with full PPR for
tight ends.

Everything on a row for week W is computable on the Monday after week W's games
— the morning claims are entered. Usage features use weeks 1..W inclusive, the
`on_wire` availability proxy uses weeks 1..W-1 (it asks whether the player was
rosterable going *into* the week), and the only forward-looking column is `fwd3`,
the training target. The module docstring states this, along with the two
judgement calls worth arguing about: where the empirical Bayes priors are fit,
and how byes are handled at both ends of the window.

## Weekly table

`make weekly` writes `outputs/weekly/{season}/wk{NN}.csv` — every wire-eligible
player ranked by model score, with the raw usage numbers that produced the score
in the same row. A bare score is not actionable: if a name surfaces, the reason
has to be visible next to it.

Omit `SEASON`/`WEEK` and the week is resolved from the schedule in
`data/raw/games.csv`. Never from calendar arithmetic — week boundaries move for
byes, international kickoffs and the five-day gap before Week 18, and a day
counter is wrong several times a season and wrong silently. `tests/test_weekly.py`
pins that behaviour against real schedule data.

This does **not** filter by availability in any particular league. It narrows a
few thousand player-weeks to a few dozen names; confirming who is actually free
is a separate manual step.

## What the scheduled job does, and does not do

`.github/workflows/weekly.yml` runs Tuesdays at 06:00 UTC, and on demand via
**workflow_dispatch**. It produces **the weekly table only**:

| | |
| --- | --- |
| runs | `make install`, `make test`, `make data`, `make panel`, `make weekly` |
| may commit | `outputs/weekly/**` and `data/raw/MANIFEST.json`, nothing else |
| never runs | `make models`, `make report`, `make ledger` |

**`make report` and `make ledger` are hand-run.** Nothing arrives on its own on a
Monday. The table is waiting for you in `outputs/weekly/{season}/wk{NN}.csv`, and
the report is a command you type:

```bash
git pull
make report SEASON=2025 WEEK=8      # then make ledger, once three weeks have passed
```

This is deliberate, not a gap. The commit step is an allowlist that fails the job
if a run touches anything outside those two paths, and `outputs/reports/`,
`outputs/ledger/`, `outputs/diagnostics/` and `models/` are all outside it —
they are permanent records and deliberate retrains, and a scheduled job that
quietly rewrote one would be discovered weeks later, if ever. Automating the
report would mean handing the job write access to the ledger that grades it.

**No commit is the expected outcome on a quiet week.** The manifest and the
weekly table both carry digests and no timestamps, so a run whose inputs and
output are unchanged stages nothing and exits without pushing. A Tuesday with no
commit means the wire was quiet, not that the job is broken — check the Actions
tab for a green run, which is the signal that it worked.

If nflverse revised a prior season since the last run, the job prints a loud
warning and keeps going — the new table is fine, but anything cached from before
the revision is no longer comparable.

## Report and ledger

`make report SEASON=2025 WEEK=8` writes `outputs/reports/{season}/wk{NN}.md` —
under 500 words, tiered into burn-the-claim / fallback / watch, every line
carrying the usage numbers behind it and a range rather than a point estimate.
Add `ROSTER=path` to resolve drops and run the roster check; without one the
report still tiers the wire and leaves `DROP ???`.

Claims are ordered by **points above replacement at the same position**, not by
raw model score. The score is a within-week percentile rank pooled across
positions, so it is not comparable between them — tiering on it puts a streaming
quarterback ahead of a genuinely valuable receiver. Replacement is the next
player you would actually take instead (the third-best available QB, the
seventh-best WR), not the pool median, which at QB is a backup who will not play.

Nothing in this repo places a transaction, and no proposed drop leaves the
roster without a K or D/ST.

Every recommendation is appended to `outputs/ledger/claims.csv`. `make ledger`
grades them once three weeks of forward data exist, against two benchmarks: the
best player who was on the wire that week (the ceiling) and whoever simply
scored the most points the week before (the hot hand — what most managers
actually do, and the bar worth clearing). This is the only thing here that tests
the workflow rather than the metrics. If a season of reports does not beat the
naive benchmark, the apparatus is decoration, and this is how that gets found
out.

## The three-arm comparison

`make ledger` also runs a controlled comparison of three ways of picking a
claim, scored identically by what the recommended player went on to do:

- **naive** — the highest-scoring available player from last week. Derived from
  the panel, never logged by hand, so it cannot drift.
- **prompt** — an LLM with web search and no access to this repo.
- **repo** — the candidate table plus judgement. `make report` logs this arm
  itself.

Only one arm is ever played on a real roster; the other two are paper. The top
three picks per arm per week are logged, which turns thirteen observations into
thirty-nine — still not many, and the output says so.

```bash
make log-claim ARM=prompt PLAYERS="First Name, Second Name, Third Name"
```

Season and week come from the schedule, positions from the panel. The `prompt`
arm must be produced **before** the candidate table is opened, in a separate
session; `CONTAMINATED=1` marks a week where that order broke and drops it from
the comparison. `docs/comparison_protocol.md` has the weekly ritual, what the
`logged_at` timestamps can and cannot prove, and why an excluded week is worth
more than a clean-looking one that is not.

The decision rule is pre-registered in the `src/ledger.py` module docstring, so
it cannot be revised once the numbers land. The most likely honest answer at
this sample size is "inconclusive", and the module prints a bootstrap interval
on the paired prompt-vs-repo difference next to the verdict so that a tie gets
reported as a tie.

## Does any of this beat the naive benchmark?

On twelve seasons of walk-forward replay, **no — but not for the reason the
three-season replay suggested.**

`outputs/backtests/02_walkforward_2014_2025.py` fetches 2013-2025, refits the
models per season on the seasons strictly before it (2014 on 2013, 2025 on
2013-2024), replays weeks 2-17, and scores the table's top three claims against
the three highest scorers from the previous week. Over 156 paired weeks — about
570 picks per arm — the repo arm loses by **2.30 ppg**, 95% CI [-3.12, -1.47].

**About 93% of that gap is positional composition, not ranking quality.** The
naive rule sorts on raw fantasy points, quarterbacks score the most raw fantasy
points, so the naive arm is 59% quarterbacks against the repo arm's 21%, and
`fwd3` rewards it for that. Points above replacement exists to undo exactly that
incomparability, and the scoring convention puts it straight back.

**Take position out and the difference is a null**: repo − naive = -0.12 ppg,
95% CI [-0.84, +0.63], which covers zero. Within position, this data does not
distinguish the model's ranking from sorting the wire by last week's box score.
That is a change from the three-season result, which put 61% of the gap on
position mix and still had the repo arm behind by 1.18 ppg [-2.23, -0.12] after
adjustment — the larger sample did not sharpen that into significance, it
dissolved it.

**Training window: do not recency-weight.** Every replay season was run twice,
trained on all prior seasons and on the most recent three. Across the nine
seasons where those differ, recent − expanding is **-0.32 ppg, 95% CI [-1.01,
+0.37]** — indistinguishable from nothing. Football did not change over this
span in a way these features can see, so the production models should keep using
all available history.

`outputs/diagnostics/walkforward_2014_2025.md` has the per-season tables, the
expanding-vs-recent comparison, the 2020 and 2021 structural breaks handled
separately rather than pooled, the roster-depth sensitivity, the contamination
audit, and what the replay does not test. Read it before trusting a weekly
table. `outputs/diagnostics/season_replay_2022_2025.md` is the earlier
three-season version, kept for the comparison.

### Two silent data defects this found

Both were found by building the panel back to 2013, and either would have
produced a clean-looking result computed on wrong data.

- **The three nflverse feeds disagree on team codes before 2020.**
  `stats_player_week` uses the modern franchise code in every season — a 2013
  Rams row says `LA` — while `snap_counts` and `games.csv` use the code in force
  at the time (`STL`). The stats-to-snaps merge joins on `team`, so every player
  on a relocated franchise was dropped: 2013 lost **12.1% of stat rows against a
  ~2.5% baseline**, three entire franchises. `forward_three` then failed to find
  their games and left `fwd3` — the training target — NaN for all of them. Fixed
  by `features.RELOCATIONS`. The last relocation was 2020, so this is a **no-op
  from 2020 on** and cannot change what the shipped pipeline produces.
- **A season asset can return HTTP 200 and contain only a header row.**
  `snap_counts_2012.csv` is exactly that, 154 bytes. A fetch-succeeded check
  passes it and the season silently contributes nothing.
  `features.require_rows` now asserts on parsed contents rather than on the
  fetch.

## Known gaps

Found, deliberately not fixed, and listed here so they are a decision rather
than a surprise. `outputs/diagnostics/` has the full findings behind each.

- **The CPython patch level is not pinned.** `requirements.txt` pins the
  libraries; nothing pins the interpreter. The workflow asks `setup-python` for
  `3.12` and gets whatever patch the runner's tool cache holds (3.12.14 at last
  check, against 3.12.3 locally). Each bundle records the patch it was fitted
  under, but the load check does not assert it — a runner cannot honour a pin it
  has no way to satisfy. Same-minor CPython is not a plausible source of numeric
  drift here; it is named because it is the one version in the chain nothing
  controls.
- **`git push` in the workflow has no retry or rebase.** `concurrency` stops the
  job racing itself but not a human pushing to `main` during its ~45-second
  window. It fails loudly, which is the right direction, but it fails *after*
  doing all the work — re-running is the whole pipeline again.
- **`actions/checkout@v4` and `actions/setup-python@v5` run on Node 20**, which
  the runner now force-upgrades to Node 24 with a deprecation warning. Harmless
  today, a version bump eventually.
- **`pts` is not a model feature, but `pts_lag1` is.** The panel row for week W
  carries the player's week-W fantasy points, `src/features.py` states they are
  known on the Monday claims are entered, and `src/weekly.py` prints them next to
  every candidate — but `BASE_FEATURES` in `src/models.py` includes only the
  *lagged* version. Week W's box score out-correlates its own lag with `fwd3` at
  every position, and it is the entire signal the naive benchmark uses. Adding it
  in the replay recovers about 0.5 ppg and does not close the gap, so it is a real
  defect and not the explanation for the result above. Left unfixed here because
  changing `BASE_FEATURES` means refitting and re-carding the shipped bundles,
  which is a deliberate retrain rather than a backtest.
- **The scheduled job produces the weekly table only.** `make report` and
  `make ledger` stay hand-run — see
  [What the scheduled job does, and does not do](#what-the-scheduled-job-does-and-does-not-do).
  Listed here as well because waiting for a report that is never coming is the
  most expensive way to learn it.

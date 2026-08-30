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

## Data sources

All from [nflverse](https://github.com/nflverse):

| file | source |
| --- | --- |
| `snap_counts_{year}.csv` | nflverse-data release `snap_counts` |
| `stats_player_week_{year}.csv` | nflverse-data release `stats_player` |
| `play_by_play_{year}.csv.gz` | nflverse-data release `pbp` |
| `games.csv` | `nflverse/nfldata` |

## MANIFEST.json

Written by `src/fetch.py` on every run. Per file: source url, sha256, byte
count and UTC fetch timestamp. If a re-fetch changes a hash for a season that is
already complete, upstream revised history and previously computed results are
no longer comparable — `fetch.py` says so on the spot.

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

`.github/workflows/weekly.yml` runs the whole chain Tuesdays at 06:00 UTC (and
on demand), then commits `outputs/` and `data/raw/MANIFEST.json`. If nflverse
revised a prior season since the last run, the job prints a loud warning and
keeps going — the new table is fine, but anything cached from before the
revision is no longer comparable.

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

# `proj_pts_hi` presentation, and the first real Actions run

Two answers: whether the saturated ceiling was actually fixed or only written
down, and what happened the first time the `weekly waiver table` workflow ran
on a GitHub runner.

---

## 1. `proj_pts_hi` — was it fixed, or only documented?

**Only documented.** Nothing in `src/report.py` changed in PR #8.

`outputs/diagnostics/followup_2025wk08.md` §2 measured the saturation, named
the cause, and proposed the exact wording — and then said so explicitly: *"Not
implemented yet — this is a presentation decision for `report.py` and/or
`weekly.py`, flagged rather than made."* PR #8 touched `report.py` only to add
`DEFAULT_ROSTER_PATH`; the two lines that render a range were left as they
were. The report that shipped on `main` prints the clipped ceiling as a
figure, exactly as before the diagnostics were written.

That is now fixed in **PR #9** (presentation only — no change to `weekly.py`,
the clip, the intervals, or any column in `outputs/weekly/*.csv`).

### What the report printed before the fix

Both of these lines are from the committed `outputs/reports/2025/wk08.md` at
`main` (10a4687):

```
Saturated     - **Cam Ward** (QB, TEN) — 100% snaps; 11.1–25.8 pts/wk
Not saturated - **Kirk Cousins** (QB, ATL) — ...; 2.2–25.8 pts/wk
```

(Cousins does not appear in the report — no candidate the report printed was
unsaturated — so his line is his real row from `outputs/weekly/2025/wk08.csv`
rendered through the same code path.)

This is the bug, in one place. Ward's `score_hi` is 1.0000: it hit the clip, so
his 25.8 is the position's 99th-percentile outcome, shared with five other QBs
in the same table. Cousins's `score_hi` is 0.9896: under the clip, so his 25.8
is his own bound. **The two rendered identically.** A reader on a Monday night
had no way to tell that one number described a player and the other described
the method's ceiling.

### What it prints after the fix

```
Saturated     - **Cam Ward** (QB, TEN) — 100% snaps; 11.1+ pts/wk (upside unbounded)
Not saturated - **Kirk Cousins** (QB, ATL) — ...; 2.2–25.8 pts/wk
```

plus one line in **Standing rules**, emitted only on weeks that actually
printed a saturated bound:

> - `0.8+ pts/wk (upside unbounded)` means the floor is that player's but the
>   ceiling is not: his interval ran past what this method resolves, where every
>   candidate would print the same 99th-percentile number for the position.
>   Upside not bounded by this method is what is known; a figure there would be
>   invented.

The floor keeps its number: a saturated ceiling says nothing about the bottom
of the interval. The per-line marker is short and the explanation is stated
once, because on this week's report the phrase would otherwise have repeated
twelve times and pushed the report from 462 to 517 words, over its own
500-word limit — repeating a sentence on every line hides the distinction just
as effectively as printing a fake number does.

### How bad it was on the shipped Week 8 table

Saturation test is `score_hi >= PROJECTION_CLIP[1]` (0.99), not `score_hi ==
1.0` — Darnell Mooney's 0.9932 clips without ever reaching 1.0, and a `== 1.0`
check would have called his line honest.

| position | half-width | score that saturates | saturated / candidates | shared ceiling |
| --- | ---: | ---: | ---: | ---: |
| QB | 0.314 | ≥ 0.676 | 6 / 15 | 25.8 |
| TE | 0.286 | ≥ 0.704 | 8 / 39 | 16.5 |
| WR | 0.289 | ≥ 0.701 | 3 / 52 | 14.8 |
| RB | 0.303 | ≥ 0.687 | 2 / 28 | 16.8 |

The concentration is the point, and it is worse than the per-position rates
suggest: **all nine players the Week 8 report printed were saturated** — every
name in *Top of the wire*, every ADD, every WATCH. Scoring well is what pushes
a bound past the clip, so the false precision landed on 100% of the players the
report recommended and on none of the ones it passed over. (§2 of the earlier
diagnostics reported 7/15 QBs rather than 6/15; that count came from a refit
during that session, this one is measured directly on the `wk08.csv` committed
to `main`.)

One honest limit on the fix: it distinguishes *clipped* from *unclipped*, not
"numbers that look different." Cousins's unclipped 25.8 rounds to the same
figure as Ward's clipped one, because his bound sits just under the clip. The
difference is that his is a statement about him.

---

## 2. First real Actions run: **FAILED**

- Run: <https://github.com/nicklasorte/weekly_waiver/actions/runs/33336409356>
- Trigger: `workflow_dispatch` on `main` @ `10a4687`, inputs `season=2025`,
  `week=8`
- Run number **1** — confirmed first execution of this workflow, ever
- Failed at step 5 of 9, *Unit tests*, 18 seconds in

Per your instruction this is **not fixed in this session**. The log is below
and the workflow file is untouched.

### Step results

| # | step | result |
| ---: | --- | --- |
| 2 | `actions/checkout@v4` | success |
| 3 | `actions/setup-python@v5` (3.12) | success |
| 4 | Install dependencies | success |
| 5 | **Unit tests** | **failure** |
| 6 | Fetch nflverse data | skipped |
| 7 | Build panel | skipped |
| 8 | Generate the weekly table | skipped |
| 9 | Commit results | skipped |

### The actual log

`pip install -r requirements.txt` succeeded, `pyyaml` included:

```
Collecting pyyaml>=6.0 (from -r requirements.txt (line 6))
  Downloading pyyaml-6.0.3-cp312-...-manylinux_2_28_x86_64.whl.metadata (2.4 kB)
...
Successfully installed joblib-1.5.3 narwhals-2.25.0 numpy-2.5.2 pandas-3.0.5
python-dateutil-2.9.0.post0 pyyaml-6.0.3 scikit-learn-1.9.0 scipy-1.18.1
six-1.17.0 threadpoolctl-3.6.0
```

Then, immediately:

```
##[group]Run make test
make test
...
error: no Python interpreter at '.venv/bin/python'
  run 'make install' to create it, or pass PY=<python> to use your own.
make: *** [Makefile:42: check-py] Error 1
##[error]Process completed with exit code 2.
```

### What this means

The job installs dependencies into the runner's system Python, but every `make`
target routes through `PY ?= $(VENV)/bin/python` and the `check-py` guard, which
finds no `.venv` and refuses. The workflow never runs `make install` and never
passes `PY=`, so `make test` — and `make data`, `make panel`, `make weekly`
behind it — cannot execute at all.

This is not a flake, a data problem, or a `pyyaml` problem. It is structural:
every `make` step in the job fails the same way, so a rerun fails identically.

Nothing in the failure touches project code. `make test` never imported a
module; the job died in the Makefile's own guard. The clean local run proved
the pipeline, not the runner — exactly the gap you flagged.

Where it came from: the workflow was written in 99d41b7 (PR #5), against a
Makefile that ran the ambient interpreter. 1d1ac90 (PR #7, *"Make `make install`
create the venv it depends on"*) introduced the `.venv` default and the
`check-py` guard and did not update `.github/workflows/weekly.yml`, which has
not been edited since PR #5. The workflow has been broken since PR #7 merged;
nothing surfaced it because it had never run. The fix is a one-line decision in
the workflow — add a `make install` step, or pass `PY=python` to the `make`
steps — deliberately left for you to make.

### What it committed: nothing

Step 9, *Commit results*, was skipped. **No commit was pushed by the runner.**
`main` is still at `10a4687` (the PR #8 merge), and no `outputs/` file or
`data/raw/MANIFEST.json` was touched by Actions.

So the answer to "make sure that does not conflict with or overwrite anything
from PR #8" is that the question did not arise on this run. It remains live for
the next one: the *Commit results* step does `git add outputs/ data/raw/MANIFEST.json`
and pushes to `main` unconditionally, so once the job gets that far it will
overwrite `data/raw/MANIFEST.json` and any `outputs/weekly/**` file it
regenerates. Worth deciding before the workflow is repaired, not after.

### Local comparison, since the runner produced no artifacts

Run here on `main` @ `10a4687` against nflverse as of 2026-08-30T21:26Z
(`make data && make panel && make weekly SEASON=2025 WEEK=8`):

- `outputs/weekly/2025/wk08.csv` came back **byte-identical** to the copy PR #8
  committed. The pipeline reproduces exactly against today's upstream data.
- `data/raw/MANIFEST.json` differs only in `generated_utc` / `fetched_utc`
  timestamps and in the `games.csv` SHA. `games.csv` was re-published upstream
  between the PR #8 fetch and this one — same byte count (2,177,172), different
  digest — which is the churn that file is expected to have. No play-by-play or
  stats file changed. Nothing downstream of it moved, as the identical table
  shows.
- Restored both files afterward; neither is part of PR #9.

So had the workflow reached its commit step on this run, it would have pushed a
`MANIFEST.json` timestamp/SHA update and no change to `wk08.csv`.

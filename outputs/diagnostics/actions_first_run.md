# `proj_pts_hi` presentation, and the first real Actions run

Two answers: whether the saturated ceiling was actually fixed or only written
down, and what happened the first time the `weekly waiver table` workflow ran
on a GitHub runner.

§3 was appended in a later session: the repair those two answers called for,
and the second run. The story of getting this workflow running is kept here
rather than split across files.

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

---

## 3. The repair, and the second run: **PASSED**

- Run: <https://github.com/nicklasorte/weekly_waiver/actions/runs/33339010648>
- Trigger: `workflow_dispatch` on `claude/weekly-workflow-repair-mqf0jt` @
  `be4f06a`, inputs `season=2025`, `week=8`
- Run number **2**; 9 of 9 steps green, 46 seconds wall clock
- Committed `c22adf4` to the branch — **two files, both allowed**

Run on the branch rather than `main` on purpose: the commit step had never
executed, and the first thing it does when it works is push. Exercising it
against the branch put the failure mode where a mistake was recoverable.

### 3.1 Decision 1 — `make install`, not `PY=python`

`make install` now runs before the first `make` target, and the job asserts the
interpreter it built:

```
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
Successfully installed joblib-1.5.3 narwhals-2.25.0 numpy-2.5.2 pandas-3.0.5
python-dateutil-2.9.0.post0 pyyaml-6.0.3 scikit-learn-1.9.0 scipy-1.18.1
six-1.17.0 threadpoolctl-3.6.0
Python 3.12.14
```

Every subsequent step then dispatched through the venv, which is the property
`PY=python` would have given up:

```
.venv/bin/python -m unittest discover -s tests     # 48 tests, OK
.venv/bin/python -m src.fetch 2022 2023 2024 2025 2026
.venv/bin/python -m src.features 2022 2023 2024 2025 2026
.venv/bin/python -m src.weekly --season 2025 --week 8
```

Cost of the choice, measured: the *Install dependencies* step took **21s** on a
cold pip cache. That is the whole premium, and it buys the runner and a
developer machine the same interpreter path.

One residual gap, stated because "match local exactly" is the reason the
decision went this way. The venv's *path* matches and its package set matches
exactly, but the CPython patch level does not: `setup-python` resolves
`python3.12` to its tool-cache **3.12.14**, while the container this session
ran in has **3.12.3**. Same minor, different patch, and nothing in the repo
pins it. It did not move any number this week (see §3.3), but it is the one
axis on which the two environments are still not identical.

### 3.2 Decision 2 — what the commit step may touch

The step now commits only `outputs/weekly/**` and `data/raw/MANIFEST.json`, and
fails the job if a *tracked* file changed anywhere else. What it did:

```
committing:
  data/raw/MANIFEST.json
  outputs/weekly/LAST_MANIFEST.json
[claude/weekly-workflow-repair-mqf0jt c22adf4] weekly table: 2026-08-30
 2 files changed, 17 insertions(+), 17 deletions(-)
   be4f06a..c22adf4  claude/weekly-workflow-repair-mqf0jt
```

Both are inside the allowed set. `outputs/reports/**`,
`outputs/diagnostics/**`, `outputs/ledger/**` and `models/**` were untouched,
as `git show --stat c22adf4` confirms.

The guard was tested against real changes on a scratch tree before it shipped,
not just reasoned about. Each case is the guard verbatim, run against a tree
dirtied to match:

| tree state | guard |
| --- | --- |
| `wk08.csv` + `MANIFEST.json` + `LAST_MANIFEST.json` + a new `wk09.csv` | commits all four |
| `outputs/diagnostics/dryrun_2025wk08.md` edited | **exit 1**, commits nothing |
| `outputs/ledger/claims.csv` + `models/QB.joblib` edited | **exit 1**, commits nothing |
| `outputs/reports/2025/wk08.md` edited | **exit 1**, commits nothing |

The failure path is loud and total: it names every offending file, prints
`git diff --stat`, emits a `::error::` annotation, and exits non-zero **before**
staging anything — so a stray write produces a red X and no commit, never a
partial one and never a silent skip.

The check runs twice on purpose. Once on the working tree, and once on the
*staged* set after `git add`. The second is redundant today; it is what keeps
the guarantee if someone later widens the `git add` line, which is exactly how
this step drifted the first time.

### 3.3 Did the runner's table match?

**Yes — byte-for-byte.** `outputs/weekly/2025/wk08.csv` does not appear in
`c22adf4`, and that absence *is* the result: the runner regenerated the file,
and `git` found no difference against the committed copy, so there was nothing
to stage.

Spelled out, since a null result is easy to mistake for "it didn't run":

| artifact | sha256 |
| --- | --- |
| committed at `7cfb39f` / `be4f06a` / `c22adf4` (blob `2aa524d1`) | `7710fbb6…56e322` |
| regenerated locally, this session | `7710fbb6…56e322` |
| regenerated on the runner | identical to the blob at `HEAD`, by `git diff` |

The runner did write the file — `2025 week 8: 134 wire candidates ->
outputs/weekly/2025/wk08.csv`, and the printed top-5 tables match the local run
row for row, to the last decimal. It then compared equal. Two machines, two
CPython patch levels, a fresh download of every input, and the same 134 rows.

So the mismatch that would have been a real signal did not occur.

### 3.4 What churned, and what that means for every future run

The two files it did commit are both timestamp/digest churn:

- `data/raw/MANIFEST.json` — `generated_utc`, thirteen `fetched_utc` stamps,
  and the `games.csv` digest. `games.csv` was republished upstream *again*
  between the local fetch at 20:53Z and the runner's at 22:23Z: same 2,177,172
  bytes, new sha256. The other twelve files are unchanged.
- `outputs/weekly/LAST_MANIFEST.json` — the recorded revision, following
  `games.csv`.

`make data` flagged it at fetch time (`<-- upstream revised, prior results are
not comparable`), and `make weekly`'s louder REVISED HISTORY banner correctly
stayed quiet: that check deliberately excludes `games.csv`, which carries no
season in its name and legitimately changes every week.

Worth knowing before the first unattended Tuesday: **the "nothing changed" exit
is effectively dead code.** `MANIFEST.json` embeds `generated_utc` and
`LAST_MANIFEST.json` embeds `recorded`, both of which move on every run. So the
job will push a commit every week whether or not the table changed — as it just
did, for a week whose table did not change at all. Left as-is rather than fixed
quietly; the honest options are to stop tracking those timestamps or to accept
the weekly noise commit, and that is a call to make deliberately.

### 3.5 Other drift found in the workflow — reported, not fixed

The workflow was last edited in `99d41b7` (PR #5) and the tree grew underneath
it. Two things were changed beyond the two decisions, and both are named here
rather than slipped in:

- **The *Install dependencies* step was `pip install -r requirements.txt`** —
  which in PR #5 *was* `make install`, verbatim, inlined. When PR #7 gave
  `install` a venv, the copy stopped tracking the original. Replaced by
  `make install`, which is Decision 1.
- **`${{ inputs.season }}` was interpolated straight into the shell command.**
  Now passed via `env:` and referenced as `"$SEASON"`. Blank behaves as before
  (`make weekly SEASON=` sets an empty make variable, which
  `$(if $(SEASON),…)` treats as unset), but a value with a space in it is now
  one argument instead of a shell fragment.

Found and **deliberately left alone**:

- **`git add outputs/` was not wrong when it was written.** In PR #5 `outputs/`
  held only `weekly/` and `backtests/`. `reports/` and `ledger/` arrived in
  PR #6, `diagnostics/` in PR #8. The line never changed; the tree grew under
  it until it covered three record directories. Same drift mechanism as the
  Makefile, which is why the replacement is an allowlist that fails closed
  rather than a wider `git add`.
- **`models/*.joblib` are loaded, never rebuilt, and `requirements.txt` is
  unpinned.** `make weekly` unpickles the committed bundles; the workflow never
  runs `make models` (correctly — Decision 2 forbids it). But `requirements.txt`
  says `scikit-learn>=1.3`, so nothing stops a runner from installing a version
  newer than the one the bundles were pickled under. Today it happens to match
  (1.9.0 both places, no `InconsistentVersionWarning` on load), which means this
  is latent, not benign: the day scikit-learn ships 1.10, the weekly table
  starts loading models through a version it was never fit under, and the
  failure mode is a warning at worst and silently different numbers at worst-
  worst. A pin, or a fit-version assertion at load, would close it.
- **`git push` has no retry or rebase.** `concurrency` stops the job racing
  itself, not a human pushing to `main` during the ~45s window. It would fail
  loudly, which is acceptable, but it would fail after doing all the work.
- **`make report` and `make ledger` are not in the workflow at all.** That is
  now deliberate rather than accidental — both write to directories the commit
  step is forbidden to touch — but it is worth stating that the scheduled job
  produces the table only. The report and the ledger stay hand-run.
- **`actions/checkout@v4` and `actions/setup-python@v5` are on Node 20**, which
  the runner now force-upgrades to Node 24 with a deprecation warning. Harmless
  today, a version bump eventually.

## 4. Version pinning and the noise commit — fixed

Both items from §3.4 and §3.5 are now closed. §4.1–4.3 cover the model/runtime
coupling, §4.4 the weekly noise commit, §4.5 what moved out of this file and
into the README.

### 4.1 The pin: every direct dependency, not just the numeric stack

`requirements.txt` went from six floors to six exact pins:

```
 pandas>=2.0            ->  pandas==3.0.5
 numpy>=1.24            ->  numpy==2.5.2
 scikit-learn>=1.3      ->  scikit-learn==1.9.0
 scipy>=1.10            ->  scipy==1.18.1
 joblib                 ->  joblib==1.5.3
 pyyaml>=6.0            ->  pyyaml==6.0.3
```

Everything, not only the numeric stack. This is an application with one
execution path, not a library other code resolves against, so a loose bound buys
nothing: no downstream consumer ever has to reconcile these ranges with its own.
And `pyyaml` is not decoration — it parses `data/roster.yaml`, which decides
which drops a report proposes, so its version reaches the output too. Drawing
the line at "numerics" would have left a parser free to move under the one file
a human hand-maintains.

Transitive dependencies (`narwhals`, `threadpoolctl`, `six`, `python-dateutil`)
are deliberately **not** pinned. Pinning those is a lockfile, with a lockfile's
maintenance, and the thing that actually protects the numbers is §4.2 rather
than the transitive closure of a requirements file. Worth noting that the exact
pins above resolve those four to the same versions the fitting session recorded,
so the practical gap today is zero.

### 4.2 The assertion: bundles carry their fit versions, and a mismatch exits

Each `models/*.joblib` now carries a `fit_versions` dict alongside the data
revision it already had:

```json
{"python": "3.12.3", "scikit-learn": "1.9.0", "numpy": "2.5.2",
 "scipy": "1.18.1", "pandas": "3.0.5", "joblib": "1.5.3"}
```

`src/models.py:check_fit_versions` runs inside `load_bundle`, which is the single
chokepoint every score goes through (`src/weekly.py` is the only caller). Four of
the six are asserted — `scikit-learn`, `numpy`, `scipy`, `pandas`. Those sit
between the stored trees and the printed number: scikit-learn owns the predict
path, numpy and scipy own the arithmetic under it, pandas builds the feature
frame that goes in.

`python` and `joblib` are recorded but **not** asserted, and the distinction is
the point rather than a hedge. The runner picks its own CPython patch level from
the tool cache; asserting a version nothing can pin would fail every Tuesday for
a non-difference (§4.5 carries this forward as a known gap rather than losing
it). `joblib` only serialises — if the bundle unpickled at all, it did its job.
Both are in the bundle so a forensic question later has an answer.

A mismatch **exits**; it does not warn. `make weekly` against a bundle claiming
scikit-learn 1.10.0 — the exact scenario §3.5 flagged as latent — now:

```
$ make weekly SEASON=2025 WEEK=8
.venv/bin/python -m src.weekly --season 2025 --week 8
WR.joblib was fitted under different library versions than the ones running now.

  library         fitted under    running now
  scikit-learn    1.10.0          1.9.0

Refusing to score: unpickling a fitted estimator through a version it
was not fit under produces numbers, not errors, and a weekly table you
have no reason to distrust is the worst way for this to fail.

  to fix: `make install` to get the pinned set in requirements.txt, or
          `make models` to refit the bundles under what is installed.
make: *** [Makefile:64: weekly] Error 1
```

That is a red X on the Actions run, which is the whole objective: the failure
mode being closed here is one where the job stays green and the numbers move.
A bundle with no `fit_versions` at all is refused on the same grounds — it
cannot be verified, so it is not assumed fine.

### 4.3 The committed bundles were stamped, not refitted

The four bundles predate the stamp, so they had to acquire one. Refitting would
have been the tidier-looking option and the wrong one: the panel has moved since
these were fit (`games.csv` alone was republished twice in the last two hours,
see §3.4), so `make models` would have produced *different* models and quietly
changed the shipped table under cover of a version-pinning change. Instead each
bundle was loaded and re-dumped with `fit_versions` added and nothing else
touched.

Provenance of each version, since a backfilled stamp is only as good as its
source:

| library | where the value comes from | strength |
| --- | --- | --- |
| `scikit-learn 1.9.0` | `_sklearn_version` inside the pickle itself | self-attested by the artifact |
| `numpy 2.5.2`, `pandas 3.0.5`, `scipy 1.18.1`, `joblib 1.5.3` | the pip install log of the session that fit them (`dryrun_2025wk08.md`, line 66) | recorded at the time, not reconstructed |
| `python 3.12.3` | same session's bootstrap interpreter, and this container's | recorded at the time |

Corroborated three ways: the models' `trained_on` is `2026-08-30`, the same day
as that session; the exact pins in §4.1 resolve to precisely that set today; and
those bundles load under it with no `InconsistentVersionWarning` (checked with
that warning promoted to an error). This is stronger than a guess and weaker
than a stamp written by the fitting code itself — the next `make models` produces
the latter, and every bundle after this one is self-recorded.

The re-dump was verified not to move anything:

```
QB: stamped, 232,352 -> 232,473 bytes, predictions bit-identical over 64 probes
RB: stamped, 249,648 -> 249,831 bytes, predictions bit-identical over 64 probes
TE: stamped, 241,328 -> 241,511 bytes, predictions bit-identical over 64 probes
WR: stamped, 249,456 -> 249,639 bytes, predictions bit-identical over 64 probes
```

The ~120-180 byte growth is the JSON-ish payload of the new key. `predict` was
compared before and after on 64 fixed probe rows and matched exactly, and the
conformal half-width, coverage, R², feature list, `n_train` and `data_revision`
all round-tripped unchanged.

**Confirmation the committed bundles still load clean under the pinned set** —
the whole chain was run end to end in a fresh `.venv` built from the new
`requirements.txt`:

| step | result |
| --- | --- |
| `make install` | pinned set installs, no resolver conflict |
| `make test` | 62 tests OK (was 48; §4.4 adds the other 14) |
| `make data` | 13 files, all re-downloaded |
| `make panel` | 22,615 rows |
| `make weekly SEASON=2025 WEEK=8` | 134 wire candidates, no version error |

The regenerated table hashes to `7710fbb6…56e322` — **the same digest as the
committed blob and as both prior runs in §3.3**. Pinning the environment and
stamping the bundles changed no number.

One incidental finding from that run: `games.csv` had been republished *again*
(`0d581221…` → `edc7d01f…`, same 2,177,172 bytes) between the runner's fetch at
22:23Z and this one. The table came out byte-identical anyway. That churn was
deliberately **not** committed here — the data revision recorded on `main` should
be set by a pipeline run on `main`, not by a verification run in a scratch
container, so `MANIFEST.json` in this change carries only the timestamp removal
below and keeps the digests the runner recorded.

### 4.4 The noise commit: timestamps are out of version control

Three fields moved every run and made the "nothing changed" exit dead code. All
three are gone from tracked files:

| file | removed | kept |
| --- | --- | --- |
| `data/raw/MANIFEST.json` | `generated_utc`, 13× `fetched_utc` | `url`, `sha256`, `bytes` |
| `outputs/weekly/LAST_MANIFEST.json` | `recorded` | `revision`, per-file `sha256` |

The digests stay — they are the reproducibility guarantee and the entire point of
the manifest. `data_revision()` never hashed the timestamps in the first place,
so the revision fingerprint is unchanged by this: `ebfa356a…` before and after.

The fetch times were useful enough to keep somewhere, so they go to
`data/raw/FETCH_LOG.json`, untracked (it falls under the existing `data/raw/*`
ignore; `.gitignore` now says why explicitly). `load_manifest` folds them back in
when the log is present, so `make data` does not forget when it last pulled
anything, and works fine when it is absent — a fresh clone simply has no fetch
history, which is true.

`LAST_MANIFEST.json`'s `recorded` date was not given a sidecar. That file's only
job is to say whether anything changed since last time; when something did, the
commit dates it, and when nothing did, there is nothing to date.

Placement mattered for one of these. An untracked sidecar under
`outputs/weekly/` would have been staged anyway — the commit step runs
`git add outputs/weekly`, which picks up untracked files in the directory. The
fetch log lives under `data/raw/`, where the allowlist names a single file.

Fourteen tests cover it (`tests/test_fetch.py`, `tests/test_models.py`). The one
that matters is byte-level rather than field-by-field: write the same manifest
twice with different clocks and the tracked file must be identical, while a
single changed digest must still move it. That is the actual property the commit
guard depends on, and asserting the absence of specific field names would pass
while some future field quietly reintroduced the problem.

What this buys, stated plainly: a commit from this workflow now means something
moved upstream or in the table. A Tuesday with no commit means the wire was
quiet. The green check on the Actions run is what says the job ran — not the
commit, which is now evidence of change rather than evidence of life.

### 4.5 The deferred items are in the README now, not only here

§3.5 listed four things found and left alone. A diagnostics file is a record of a
session, not a place anyone looks before a Monday, so they are now a **Known
gaps** section in `README.md` — CPython patch level unpinned, `git push` without
retry or rebase, checkout/setup-python on Node 20, and the scheduled job
producing the table only.

That last one got more than a bullet. `README.md` gained a **What the scheduled
job does, and does not do** section with the runs / may-commit / never-runs
table, the `make report` command spelled out as a thing a human types, and the
reason it is deliberate rather than missing: automating the report would mean
handing the scheduled job write access to the ledger that grades it. It also
states the new expected outcome — no commit on a quiet week is correct, and the
Actions tab is where "it ran" is confirmed.

### 4.6 The run on `main`: **PASSED**, and the commit means something

PR #12 merged as `e345c2f`; `workflow_dispatch` on `main` for 2025 week 8 is
[run #3](https://github.com/nicklasorte/weekly_waiver/actions/runs/33340045886),
green in 51 seconds. Every step passed on the first attempt — no repair needed
this time, which is what §3 was for.

| step | result |
| --- | --- |
| `make install` | pinned set installed on the runner, no resolver conflict |
| `make test` | 62 tests OK |
| `make data` | 13 files |
| `make panel` | built |
| `make weekly SEASON=2025 WEEK=8` | table written |
| Commit results | committed and pushed `93881a0` |

It **did** commit, and that is the result worth reading closely. The entire
commit is three lines:

```
 data/raw/MANIFEST.json            | 2 +-
 outputs/weekly/LAST_MANIFEST.json | 4 ++--

-  "sha256":   "0d581221cfddf39e30230ebc970dfa17f3404eed9cb1b24850ce238da22b9e21",
+  "sha256":   "edc7d01f94c9e23f1180c6678d6640374bc7ee5a72fa34c39991652ce1718ac6",
-  "revision": "ebfa356a7b9c5b6c6b298100da870da2865dd99f30b42b85f8aee242ac4ab8bd",
+  "revision": "383d351e4209704ac9073467a08f06d76ef0baff6a0f5befc15c5b0e61485eed",
-  "games.csv": "0d581221…",
+  "games.csv": "edc7d01f…",
```

`games.csv` was republished upstream again — to `edc7d01f…`, the same digest the
verification run in §4.3 independently saw an hour earlier — and the data
revision follows it. That is a real change to real data, which is exactly what a
commit from this job is now supposed to mean.

Two absences say as much as the diff does:

- **No timestamp lines.** Under the old code this same run would additionally
  have rewritten `generated_utc`, all thirteen `fetched_utc` stamps and the
  `recorded` date — 15 lines of churn wrapped around 3 lines of signal. It is
  now 3 lines of signal.
- **`outputs/weekly/2025/wk08.csv` was not committed.** The runner regenerated it
  and it compared equal to what is already in the repo, so it was never staged.
  Third machine, third run, same 134 rows — now under pinned libraries and a
  load-time version assertion rather than by luck.

So the honest reading of this run: the noise-commit fix did not make the commit
disappear, because there was something real to commit. It made the commit
legible. A run with nothing but a new clock behind it would have hit
`nothing changed` and pushed nothing, which is the case that used to be
unreachable.

One thing to keep in mind for the first unattended Tuesday: `games.csv` has now
been republished three times in under two hours. If that cadence holds through
the season, most Tuesdays will produce a one-line manifest commit rather than
none — still a record of change, but of upstream churn rather than of the table.
The weekly table itself is the file to watch in the diff; the manifest moving
alone means nflverse republished, not that this week's ranking did.

Also confirmed on the runner, incidentally: the deprecation warning §3.5 flagged
is still there and still harmless — `actions/checkout@v4` and
`actions/setup-python@v5` were force-upgraded to Node 24. It is in the README's
Known gaps now rather than only here.

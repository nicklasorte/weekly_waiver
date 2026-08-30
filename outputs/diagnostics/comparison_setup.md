> **Note:** this file was pushed to `main` ahead of the code it describes, which
> is in PR #10. Until that merges, `src/ledger.py` on `main` does not yet have
> the arm comparison in it.

# Three-arm comparison: setup

What was built, what it measures, and the rule that decides what it means —
written before the season's data exists.

## The arms

| arm | input | produced by |
| --- | --- | --- |
| `naive` | the wire pool's points from the week just played | derived, `src.ledger.naive_picks` |
| `prompt` | judgement, news, published stats | LLM with web search, no repo access |
| `repo` | the candidate table plus judgement | `make report`, or `make log-claim ARM=repo` |

Top three picks per arm per week. Only one arm is ever played on a real roster;
all three are scored the same way, so which one gets executed does not affect
the measurement.

The `naive` arm is never written to the ledger. It is recomputed from the panel
on every run, which means it cannot drift, cannot be tuned after the fact, and
cannot quietly become a different benchmark mid-season.

## Schema

`outputs/ledger/claims.csv` gains four columns:

| column | values |
| --- | --- |
| `arm` | `naive` / `prompt` / `repo` — `naive` never appears, being derived |
| `rank_within_arm` | 1, 2, 3 — blank for watch-list rows, which are not recommendations |
| `logged_at` | ISO 8601 UTC, stamped at the moment of logging |
| `contaminated` | `true` when the prompt arm saw the candidate table first |

`contaminated` is the fourth column and is not in the original request. The
contamination rule requires marking a row, and without somewhere to mark it the
rule is only prose.

**Backfill:** the nine pre-existing 2025 week 8 rows are tagged `arm=repo`, with
`rank_within_arm` 1–5 across the burn and fallback tiers in the edge order the
report already ranked them in, and blank for the four watch-list rows. Those
rows *are* the repo arm — the candidate table plus the tiering rules — so
leaving them null would have discarded real data about it. `logged_at` is
`2026-08-30T19:30:44+00:00`, the commit timestamp of `0f10bef`, which added
them. That is when the ledger received them, not necessarily when the pick was
made; the distinction does not matter here because that week has no prompt arm
to order against, and inventing a plausible Tuesday-morning timestamp would have
been worse than a slightly wrong true one.

De-duplication in `src/report.py` now keys on `(season, week, arm, player)`
rather than `(season, week, player)`. The prompt arm and the repo arm naming the
same player in the same week is the clearest possible evidence that the
candidate table added nothing that week, and the old key would have deleted
exactly that row.

## Scoring

Outcome for every pick, every arm, is `fwd3`: the player's mean points over
weeks W+1..W+3 under NCFOM scoring, **counting a week his team played and he did
not as 0.0**. Same convention the models train on. A recommendation who stops
playing is scored as the failure it is, rather than dropped from the average the
way `fwd3_played` would drop it.

Reported per arm: `n`, mean points captured, share of the weekly ceiling
captured (the best fwd3 anyone on the wire posted that week), and share of weeks
beating the naive arm head to head. "Beat" is strict — a tie is not a win, and
an arm that recommends the same player naive would have is scored as having
added nothing.

Then a paired `prompt` vs `repo` comparison on the weeks both arms covered, with
a bootstrap CI and a paired t-test on the differences.

Four decisions worth arguing with:

- **The week is the unit.** An arm's three picks in one week come from one wire
  pool against one slate of upcoming opponents, so they are not three
  independent observations of the arm. Every statistic pairs and resamples at
  the week level. Treating the 39 rows as 39 independent draws would understate
  the variance by roughly the within-week correlation and make a null result
  look significant.
- **Outcomes are looked up in the whole panel, not the wire pool.** `on_wire` is
  a proxy built from prior-week usage that the prompt arm has never seen and can
  reasonably disagree with. Scoring such a pick as unresolvable would penalise
  an arm for disagreeing with a heuristic rather than for being wrong about the
  player. Whether the pick was inside the pool is recorded per row instead.
- **A pick with no panel row that week still gets an outcome**, rebuilt from his
  surrounding weeks under the same fwd3 convention. A player who was inactive in
  week W is simply absent from the panel — and being inactive is a common reason
  a name is available. Dropping those picks would bias the comparison against
  whichever arm is most willing to recommend a player coming back from injury,
  which is the prompt arm, since news is the one input it has that the panel
  does not. In the 2025 data this is not hypothetical: a week 8 prompt pick of
  Tory Horton, who has no week 8 row, would otherwise have gone unscored.
- **The naive arm selects on `pts` alone**, without filtering to players whose
  fwd3 has resolved. Filtering would keep its average tidy by handing it
  foresight the other two arms do not have. Unresolved naive picks are dropped
  at grading time and counted.

## Contamination

The `prompt` arm is only a fair comparison if it was produced before the repo's
candidate table was opened, in a separate session. Weeks marked `contaminated`
are dropped from the head-to-head **for every arm**, not just for `prompt` —
keeping two arms of a week and dropping the third would put the arms on
different week sets, which is the one thing the paired design exists to avoid.
Excluded weeks are counted in the output rather than quietly removed.

`logged_at` makes the ordering auditable but not enforceable. The ledger reports
`clean` / `unverified` / `contaminated` per week, where `unverified` means the
repo rows are stamped first — which is the *normal* case, because CI runs
`make report` at 06:00 UTC Tuesday, hours before anyone is awake to write down a
prompt pick. So `unverified` weeks are kept by default and counted;
`make ledger STRICT_ORDER=1` drops them for a stricter read. Nothing in a git
repo can prove a file was not read. The `CONTAMINATED=1` flag is the only thing
that makes the rule real, and it works only if it is used against its user's
interest.

## Pre-registered decision rule

In the `src/ledger.py` module docstring, so that revising it shows up in a diff
as a revision to the rule. `verdict()` implements exactly this, in this order:

| condition | verdict |
| --- | --- |
| neither arm beats naive by >= 1.5 ppg | the analysis is decoration |
| prompt and repo within 1.0 ppg | repo adds nothing, keep the prompt |
| repo beats prompt by >= 1.5 ppg | the data layer earns its keep |
| anything else | inconclusive, reported as such |

The rule reads point estimates, and a point estimate is not evidence. With
around 13 weeks and the per-player variance in wire outcomes, a 1.5 ppg gap sits
comfortably inside the noise. So the verdict is printed *next to* the bootstrap
interval and never instead of it: when the interval covers zero, the output says
the arms are not distinguishable whatever branch the rule landed on, and that
this should be reported as a tie. Below eight paired weeks the output warns that
the t-test and the interval are being asked to do more than the data supports.

The point of pre-registering it is that "the analysis is decoration" is a result
the apparatus can return about itself. A tie reported as a tie is the correct
output for most seasons, and the module is written to make that the easy answer
rather than the disappointing one.

## Where things are

```
src/ledger.py                grading, the arm comparison, the decision rule
src/log_claim.py             make log-claim
docs/comparison_protocol.md  the weekly ritual and the contamination rule
outputs/ledger/claims.csv    the ledger, with the arm columns
outputs/ledger/arm_grades.csv    one row per graded (arm, week, rank)
outputs/ledger/arm_summary.csv   the per-arm table
tests/test_ledger.py         31 cases, mostly on the ways this goes wrong
```

## Current state

One week is logged (2025 wk 8), repo arm only, so there is nothing to pair yet:

```
  arm  weeks  n  mean_fwd3  ceiling_share  beat_naive_share
naive      1  3     10.660          0.526               NaN
 repo      1  3     10.522          0.519             0.000
```

The repo arm's three picks averaged 10.52 ppg against the naive arm's 10.66 —
naive is ahead, by an amount that means nothing whatsoever on one week. It is
recorded here because the first data point being unflattering is exactly the
kind of thing that gets forgotten by week 12.

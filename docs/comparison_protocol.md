# Three-arm comparison protocol

The question this season is answering is not "do the models predict fwd3". It is
two harder ones:

1. Does any of this beat a naive strategy?
2. Does the repo add anything over a well-prompted LLM with web search?

Both are answerable only by writing picks down before their outcomes exist, and
only if the arms are kept apart. The second requirement is the one that fails in
practice, so most of this document is about it.

## The arms

| arm | input | who produces it |
| --- | --- | --- |
| `naive` | the wire pool's `pts` for the week just played | nobody — derived by `src.ledger.naive_picks` |
| `prompt` | judgement, news, published stats | Claude with web search, no repo access |
| `repo` | the candidate table plus judgement | you, reading `outputs/weekly/{season}/wk{NN}.csv` |

Only one arm is ever executed on the real roster. The other two are paper. All
three are scored by what the recommended player actually did afterward, so which
one you play does not affect the measurement — play whichever you want, and log
all three the same way.

The top **three** picks per arm per week are logged, not just the top one.
Thirteen weeks of single picks resolves nothing; thirty-nine rows resolves
slightly more than nothing. That is an honest description of the improvement,
not a claim that it is enough.

The `naive` arm is never logged by hand. It is recomputed from the panel every
time the ledger runs, so it cannot drift, cannot be tuned after the fact, and
cannot quietly become a different benchmark halfway through the season.

## The weekly ritual

Tuesday morning, after Monday night resolves. The order is the protocol.

**Step 1 — prompt arm, first, in a separate session.**

Open a fresh Claude session with web search. Do not open this repo, the weekly
CSV, the report, or the panel. Ask for the three best waiver adds for your
league, with reasons. Then:

```bash
make log-claim ARM=prompt PLAYERS="First Name, Second Name, Third Name" \
  WHY="one line on what the reasoning was"
```

Names are in rank order. Season and week come from the schedule; positions come
from the panel. Override either when you need to: `PLAYERS="Tory Horton:WR"`,
`SEASON=2025 WEEK=8`.

**Step 2 — repo arm, second.**

Now open the candidate table and the report:

```bash
make weekly && make report SEASON=... WEEK=...
```

`make report` writes the repo arm's rows to the ledger itself, tagged
`arm=repo`, ranked across the burn and fallback tiers in edge order. There is
nothing to type. If you override the report's ranking with your own judgement,
log the override with `make log-claim ARM=repo PLAYERS="..."` and delete the
generated rows for that week — the arm is "candidate table plus judgement", so
your judgement is part of it, but the ledger has to say what you actually
recommended.

**Step 3 — play one of them.** Whichever you like. It does not affect scoring.

**Step 4 — grade, whenever you want.**

```bash
make ledger
```

Claims resolve three weeks after they are made, so week 8's picks are scoreable
once week 11 is in the books. Nothing needs to be run on a schedule.

## The contamination rule

**The `prompt` arm must be produced and logged BEFORE the repo's candidate table
is opened, in a separate session.**

This is the entire experiment. A prompt arm that has seen the model scores is
not measuring "LLM with web search"; it is measuring "the repo, laundered
through a second opinion", and it will look better than it is for a reason that
has nothing to do with either arm.

Concretely, before logging a prompt pick you must not have looked at, in that
week:

- `outputs/weekly/{season}/wk{NN}.csv` — the candidate table
- `outputs/reports/{season}/wk{NN}.md` — the report
- `data/processed/panel.csv`, or model scores, or the tiering
- a previous week's report, if you are about to recommend a name it surfaced

Reading the news, checking a box score, or looking at your own roster is fine.
That is what the prompt arm is.

**If the order breaks, say so.** Log the pick with `CONTAMINATED=1`:

```bash
make log-claim ARM=prompt PLAYERS="..." CONTAMINATED=1
```

That marks the row, and `src/ledger.py` drops the whole week from the head-to-head
— every arm, not just `prompt`, so no arm ends up averaged over a week set the
others do not share. The dropped weeks are counted out loud in the output.

Losing a week is cheap. A contaminated week reported as clean is not: it biases
the exact comparison the season is being run to make, and it is undetectable
afterward. A season of nine clean weeks beats a season of thirteen weeks where
four are quietly wrong.

### What the timestamps can and cannot prove

Every row carries `logged_at`, and the ledger checks whether the prompt rows are
stamped before the repo rows. It reports one of three states per week:

- **clean** — prompt rows are stamped strictly before the repo rows, or only one
  arm ran that week.
- **unverified** — the repo rows are stamped first. This is *not* evidence of
  contamination: `.github/workflows/weekly.yml` runs `make report` at 06:00 UTC
  on Tuesday, hours before anyone is awake to write down a prompt pick, so a
  perfectly clean week routinely looks like this. It only means the file cannot
  vouch for the order.
- **contaminated** — you marked it. Always excluded.

Because `unverified` is the normal case for a CI-generated repo arm, it is kept
by default and the count is printed. `make ledger STRICT_ORDER=1` excludes those
weeks too, for a stricter read of the same season.

This is the honest position: the timestamps make the ordering *auditable*, not
*enforceable*. Nothing in a git repo can prove you did not read a file. The
`CONTAMINATED=1` flag is the only thing that makes the rule real, and it works
only if you use it against your own interest.

## What is being compared

Points above replacement at position, not raw fantasy points. Every arm's picks
still resolve to `fwd3` and that column is kept on every graded row, but the
arm means, the paired difference and the pre-registered rule all read
`fwd3` minus the position's realised replacement level.

The reason is that raw points are not comparable across positions. A
quarterback averages roughly three times a running back, so an arm that sorts on
raw points claims quarterbacks and is paid for the position rather than the
pick — and you start one quarterback, so the second one is worth close to
nothing. The twelve-season walk-forward replay measured that effect at 93% of
the naive arm's margin.

**Amended 2026-08-31**, before the rule had ever been evaluated on live data.
The thresholds did not move; the quantity they are applied to did. The amendment
is recorded in the `src/ledger.py` module docstring, in
`outputs/diagnostics/comparison_setup.md`, in the README and here, because this
document's whole claim is that the rule cannot be quietly revised — so a
revision that is not quiet is the only kind it can survive.
`outputs/diagnostics/par_rescore_and_ablation.md` has the arithmetic, including
what the change does **not** fix: PAR re-prices positional composition rather
than removing it, and it re-prices it hard enough to move the walk-forward
verdict on its own.

## What the output will probably say

Most likely: inconclusive. With around thirteen weeks and the per-player
variance in wire outcomes, differences of one or two points per game are
comfortably inside the noise. `src/ledger.py` prints the pre-registered decision
rule's verdict next to a bootstrap confidence interval on the paired difference,
and says plainly when the interval covers zero.

That is the design working, not failing. The pre-registered rule (in the
`src/ledger.py` module docstring, where it cannot be quietly revised) exists so
that "the analysis is decoration" is a result the apparatus can return about
itself. A tie reported as a tie is the correct output for most seasons.

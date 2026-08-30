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
src/features.py    build the backward-looking player-week panel   (stub)
src/models.py      fit and persist the per-position models        (stub)
src/weekly.py      score the wire pool for one season/week        (stub)
src/report.py      write the weekly markdown report               (stub)
src/ledger.py      grade past recommendations after the fact      (stub)

data/raw/          downloaded source files (gitignored, except MANIFEST.json)
data/processed/    derived panel (gitignored)
outputs/           backtests, weekly tables, reports, claim ledger
```

## Setup

```bash
make install       # pip install -r requirements.txt
make data          # download 2022-2026 into data/raw/
```

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

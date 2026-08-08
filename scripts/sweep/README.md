# Benchmark sweeps and reports

`sweep.py` runs the benchmarks and writes one JSON file per sweep to
`bench/results/`, named `<machine>-<commit>-<date>.json`. `report.py` reads those
files and writes a site with two pages. CI publishes it to GitHub Pages on every
push to `main` (`.github/workflows/pages.yml`).

- `index.html` shows how each machine's numbers change from one sweep to the next.
- `explore.html` shows a single sweep in detail, down to per-stage timing.

Clicking a point on a trend chart, or a commit on a machine card, opens that
sweep in `explore.html`.

## Generating the site

```sh
uv run scripts/sweep/report.py --results-dir bench/results/ -o _site --serve
```

That writes the site and serves it at http://127.0.0.1:8000/index.html. Pass a
port to `--serve` to use another one, or drop the flag to only write the files.
Run the command again after you add or change a results file.

The pages are code only. Their data is written beside them and fetched at load:

| file | holds |
|---|---|
| `decode.js` | unpacks the columns, imported by both pages |
| `data/overview.json` | every sweep, trimmed, for `index.html` |
| `data/sweeps.json` | the sweep list `explore.html` offers |
| `data/sweeps/<result>.json` | one sweep in full, fetched when it is opened |

So the explorer downloads one sweep instead of all of them, and adding a sweep
leaves the other sweep files untouched for anything holding a cached copy. The
cost is that the pages have to be served over http — opening `_site/index.html`
from disk gets you an empty page, because the browser refuses the fetches. That
is what `--serve` is for.

Inside those files the runs are stored as columns rather than one object per
run, with the text in a shared table. `columnar.py` writes that and `decode.js`
reads it. Floats are cut to four significant figures, which is finer than any
page prints and about what the benchmarks resolve. `columnar.py` unpacks every
sweep it packs and refuses to hand back one that does not match, so a broken
encoding stops the build rather than reaching a page.

`report.py` looks for `bench/machines.toml` next to the results directory, then
one level up. Use `--machines` to point somewhere else.

## Machine names

Pass `--machine` when a machine's hostname changes between runs, or set
`CHUCKY_MACHINE`:

```sh
uv run scripts/sweep/sweep.py --all --machine reef-l40
```

`bench/machines.toml` says which names belong to the same machine and describes
each one. The comment at the top of that file explains the fields.

The overview groups machines that way. A **Group** checkbox turns the grouping
off, so you can check that the names under one machine really do agree. When a
comparison uses two of those names, the last sweep from one and the sweep before
it from another, the page says so.

A machine keeps its color as other machines are added. Eight colors are
available; machines past that appear in the tables but not in the chart, and the
page names them.

## What the numbers mean

- A point is the best passing run in that sweep for the scenario, codec,
  backend, and sink you picked. Data type, chunk size, and fill are searched
  instead of averaged. Hover a point to see which run won and how many it beat.
- Runs that did not pass are left out, and counted on the machine card instead.
- A gap means the sweep ran nothing matching the filter. It does not mean zero.
- A change compares a machine's latest sweep against its previous one, matching
  runs by id. Changes under 2% are shown as no real change.

## Schema changes

`models.py` holds the results schema, its version, and the migrations.
`retired_metrics` lists the metrics in a file whose meaning changed later, and
the overview leaves those out of comparisons instead of converting them. Bump
`CURRENT_VERSION` when a metric is renamed, removed, or changes meaning. Adding
one does not need a bump.

A stored sweep records only its version number, so add a line here when you
bump it.

### Version history

- **4** — Discard sink reports a fixed 4096-byte shard alignment, so a sweep
  with no output path measures the page-aligned pipeline. No timing comparable
  across this bump.
- **3** — `tail_gate` now measures the compression-to-aggregation delay rather
  than a device gate's wait. `tail_gate_ms` and `tail_gate_count` retired.
- **2** — `kick_sync_ms` and `kick_sync_count` retired. Stage `lod_dim0_fold`
  renamed to `lod_append_fold`.
- **1** — Predates this rule; not a single shape.

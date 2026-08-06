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
uv run scripts/sweep/report.py --results-dir bench/results/ -o _site
python3 -m http.server -d _site      # or open _site/index.html
```

The data is embedded in the pages when they are written, so they need no server
and load nothing. Run the command again after you add or change a results file.

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

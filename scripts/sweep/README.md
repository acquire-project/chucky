# Benchmark sweeps and the site they feed

`sweep.py` runs the bench binaries and writes one JSON file per sweep to
`bench/results/`. `report.py` turns every one of those files into a two-page
static site, which CI publishes to GitHub Pages on each push to `main`
(`.github/workflows/pages.yml`).

## Generating the site

```sh
uv run scripts/sweep/report.py --results-dir bench/results/ -o _site
python3 -m http.server -d _site      # or just open _site/index.html
```

Both pages embed their data at generation time, so they work straight off the
filesystem and there is nothing to fetch at load. Re-run the command after
adding or editing a results file.

| Page | Question it answers |
|---|---|
| `index.html` | Is this getting faster or slower, on which machine, and what moved last? |
| `explore.html` | Inside one sweep: every codec, chunk size, and per-stage timing. |

The overview links into the explorer (`explore.html?file=<results file name>`);
clicking a point on the trend chart or a commit on a machine card lands on that
sweep.

## One sweep is one machine at one commit

The overview groups sweeps by **machine name taken from the file name**, which
`sweep.py` writes as `<machine>-<commit>-<yyyymmdd>.json`. The name is the one a
person chose, so it stays stable when a cluster hands out a different hostname
for every allocation — `reef-l40` covers whichever `cw-us-e4a2-l40-*` node ran
it. The hostname and GPU from the file's `machine` block are shown on the
machine card. If a file name does not follow the pattern, the hostname is used
instead.

Colours are assigned per machine in order of first appearance, so adding a
machine never repaints the ones already on screen. The palette holds eight;
machines past that stay in the tables and drop out of the chart, which the page
says out loud.

## What the numbers mean

- A plotted point is the **best** passing run of that sweep for the selected
  scenario, codec, backend, and sink. Data type, chunk size, and fill are
  searched rather than averaged — hover a point for the winning configuration
  and how many runs it beat.
- Runs that did not pass are excluded and counted on the machine card instead.
- A missing point means that sweep ran nothing matching the filter, not zero.
- Changes are per machine, latest sweep against its previous sweep with
  matching runs, matched on the exact run id. Anything under ±2% is labelled
  "no real change".

## Schema changes

`models.py` owns the results schema, its version, and the migrations.
`retired_metrics` reports the metrics a file carries whose meaning changed
later; the overview drops those from comparisons rather than converting them.
Bump `CURRENT_VERSION` when a metric is renamed, removed, or changes meaning —
adding one does not need a bump.

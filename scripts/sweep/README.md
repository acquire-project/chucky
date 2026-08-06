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

`machines.toml` is picked up from beside the results (or one level up); point at
another with `--machines`.

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

## Which sweeps are the same machine

Every sweep names itself after its file, `<name>-<commit>-<yyyymmdd>.json`, which
`sweep.py` writes from `--machine` (or `CHUCKY_MACHINE`), falling back to the
hostname. Pass `--machine` where the hostname is not stable:

```sh
uv run scripts/sweep/sweep.py --all --machine reef-l40
```

That name alone is not enough to say what is the same machine — one box can boot
two systems, and a cluster hands out a different node every allocation. So
**`bench/machines.toml`** holds the grouping, matching on the sweep name or the
hostname inside the file, both with wildcards:

```toml
[[machine]]
name = "livescreen"
description = "Workstation running two systems"
names = ["LiveScreen-1", "livescreen-kubuntu"]
[machine.specs]
storage = "local nvme"
```

A machine whose name never changes needs nothing but the name. `specs` is
free-form and only worth filling in for what the sweep file does not already
record — the GPU is in there, the disk and the network are not, and they decide
as much about whether two runs compare. Both the description and the specs show
on the machine card. A sweep matching no entry keeps its own name and the page
says it is undescribed, so a new machine gets an entry deliberately.

Grouping is a view, not a fact baked into the data: untick **Group** on the
overview to split a machine back into the sweep names underneath and check that
its members really do agree. Where a comparison crosses two members — last
sweep on one system, previous on the other — the page says so rather than
quietly presenting it as one machine changing.

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

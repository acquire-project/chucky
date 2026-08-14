# Benchmark sweeps and reports

`sweep.py` runs the benchmarks and writes one JSON file per sweep to
`bench/results/`, named `<machine>-<commit>-<date>.json`. `report.py` reads those
files and writes a site with two pages. CI publishes it to GitHub Pages on every
push to `main` (`.github/workflows/pages.yml`).

- `index.html` shows how each machine's numbers change from one sweep to the next.
- `explore.html` shows a single sweep in detail, down to per-stage timing.

Clicking a point on a trend chart, or a commit on a machine card, opens that
sweep in `explore.html`.

The explorer picks a machine first and then one of its sweeps, newest at the
top, so it opens on the most recent sweep run anywhere. A control disappears
when the open sweep leaves it nothing to choose.

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
| `site.css` | the palette and the title bar, linked by both pages |
| `theme.js` | light or dark, applied before either page paints |
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

Both pages group machines that way: the overview's cards and the explorer's
machine list. A **Group** checkbox on the overview turns the grouping off, so
you can check that the names under one machine really do agree. When a
comparison uses two of those names, the last sweep from one and the sweep before
it from another, the page says so. The explorer marks the sweeps that ran under
another name, so `livescreen` shows which of its sweeps came from
`LiveScreen-1`.

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

## What a results file records

The `machine` block describes the sweep once: `name`, `hostname`, `gpu`,
`driver_version`, `cpu_count`, `commit`, `date`, and a `build` block.
`cpu_count` is the cores the process was allowed, which on a cluster is fewer
than the machine has. The build block describes the build directory: build
type, CUDA architectures, whether the GPU backend was on, C++ compiler, and
nvcomp path from `CMakeCache.txt`, plus the CUDA compiler version from the
`CMakeCUDACompiler.cmake` written beside it. A key the build directory cannot
answer is left out, and `gpu` and `driver_version` say `unknown` when there is
no `nvidia-smi`. The explorer shows either as unknown.

Each run records its `frames` and the `worker_threads` its pool ran on. The two
backends count different pools. The GPU number is the staging-copy pool, which
stops at three helpers. The CPU number is the pipeline pool, which takes one
thread per allowed core, so it matches `cpu_count`.

Each run records memory as an estimate and a measurement:

| field | holds |
|---|---|
| `memory_estimate_total_bytes` | the engine's estimate — device bytes on GPU, heap bytes on CPU |
| `memory_estimate_pinned_bytes` | pinned host bytes on GPU, 0 on CPU |
| `memory_host_baseline_bytes` | resident memory before the stream was created |
| `memory_host_peak_bytes` | most resident memory the process held during the run |
| `memory_device_used_bytes` | device memory the stream took, 0 on CPU |
| `memory_measured_bytes` | the figure to hold the estimate against: device memory on GPU, the host difference on CPU |

Compare `memory_measured_bytes` against the estimate. On the CPU it also
carries the benchmark's own source block, which is allocated after the baseline
is taken and reaches 64 MiB — more than a small stream's whole estimate — so
subtract that before reading the two as a ratio. The device figure does not
carry the block.

## Schema changes

`models.py` holds the results schema, its version, and the migrations.
`retired_metrics` lists the metrics in a file whose meaning changed later, and
the overview leaves those out of comparisons instead of converting them. Bump
`CURRENT_VERSION` when a metric is renamed, removed, or changes meaning. Adding
one does not need a bump.

A stored sweep records only its version number, so add a line here when you
bump it.

### Version history

- **6** — The CPU pipeline pool sizes itself from the cores the process is
  allowed rather than the cores the machine has. A sweep under a batch
  scheduler used to start a thread per core on a fraction of them, so no
  CPU-backend timing is comparable across this bump. GPU runs are unaffected.
- **5** — The writer groups appends into one transfer per staging buffer, so
  the pending-bytes high-water mark and the backpressure wait are sampled far
  less often. `peak_pending_mib`, `backpressure_ms` and `backpressure_count`
  retired.
- **4** — Discard sink reports a fixed 4096-byte shard alignment, so a sweep
  with no output path measures the page-aligned pipeline. No timing comparable
  across this bump.
- **3** — `tail_gate` now measures the compression-to-aggregation delay rather
  than a device gate's wait. `tail_gate_ms` and `tail_gate_count` retired.
- **2** — `kick_sync_ms` and `kick_sync_count` retired. Stage `lod_dim0_fold`
  renamed to `lod_append_fold`.
- **1** — Predates this rule; not a single shape.

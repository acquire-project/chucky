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

## Running the write-depth tier

The `iodepth` tier sweeps the write-queue depth of the filesystem sink, from
1 to 32 writes at once, at one and at four per shard file. Run it when the
write path changes.

    uv run scripts/sweep/sweep.py --tier iodepth

Read it in the explorer, not on the overview page: the overview picks the best
run per scenario, input, codec, backend and sink, and every run in this tier
matches on all five, so only the fastest depth would show. The run id carries
the depth — `__wif16__perfile4` is sixteen writes at once, four of them to any
one shard file.

## Running S3 benchmarks

The S3 tier measures against
[`s3-blackhole`](https://github.com/nclack/s3-blackhole), which consumes uploads
and discards them so the result is limited by the producer rather than object
storage. Start the benchmark server and wait for its health check:

```sh
docker compose up --build --wait s3-blackhole
```

The image builds the server's default branch rather than a fixed commit, so
Docker's layer cache decides which commit you get. Add `--no-cache` to that
build when you want the newest one. Nothing in the running server says which
commit it is, so take it from the `cargo install` line in the build output and
record it with any result you keep.

Then run the tier with arbitrary credentials and a bucket name. The bucket does
not need to exist because `s3-blackhole` stores nothing:

```sh
AWS_ACCESS_KEY_ID=blackhole AWS_SECRET_ACCESS_KEY=blackhole \
  uv run scripts/sweep/sweep.py \
    --tier s3 --backend cpu --s3-bucket chucky-bench
```

Use `--backend gpu` for a GPU-only sweep, or omit `--backend` to run both
backends.

Inspect the server-side counters at
http://127.0.0.1:9000/_s3_blackhole/stats, then stop the server with:

```sh
docker compose stop s3-blackhole
```

MinIO remains the backend for S3 integration tests, where stored data is read
back and validated. The `benchmark` Compose profile keeps `s3-blackhole` out of
the default test stack unless that service is selected explicitly.

### Accept threads

The image starts the server with `--shards 4`, overriding its default of one
accept thread per core. The server runs on the same machine as the sweep, and
every accept thread holds a core the sweep could be sending from. On a 64-core
node the server's own default measured 18% below four threads, and 23% below
two, on `256cube_single` with no compression. Across two nodes, where the server
has its own cores, four threads and sixty-four measure the same, so nothing is
given up by fixing it at four. Change it in `Dockerfile.s3-blackhole`.

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

## Inputs

The sweep fills frames three ways, and the choice decides how much a codec can
squeeze out: `zeros` compresses by thousands of times, the `xor` pattern by about
ten, `rand` barely at all. One line over all three jumps whenever a sweep adds or
drops an easier input, so the trend chart draws a panel per input. A **Same
scale** tickbox appears once there is more than one panel, putting every panel on
one value axis. Untick it to read a panel whose numbers are much smaller. The
**Input** filter picks the one the machine cards and the latest sweep use, and
clicking a panel title sets it. The trend table and the biggest changes cover
every input, with a column naming it.

## What the numbers mean

- A point is the best passing run in that sweep for the scenario, input, codec,
  backend, and sink you picked. Data type and chunk size are searched instead of
  averaged. Hover a point to see which run won and how many it beat.
- Runs that did not pass are left out, and counted on the machine card instead.
- A gap means the sweep ran nothing matching the filter. It does not mean zero.
- A change on a machine card compares the two most recent sweeps that ran the
  scenario and input, using the best run of each. A row in the biggest changes
  compares a machine's most recent sweep against the last sweep that ran the
  same configuration, matching runs by id. Changes under 2% are shown as no real
  change.

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

For a run with filesystem output, the write scheduling it used is recorded,
so a sweep is comparable only against one taken with the same settings:

| field | holds |
|---|---|
| `io_backend` | which write backend ran the requests; `threads` is the only one so far |
| `io_workers` | threads the sink's queue ran requests on |
| `io_writes_in_flight` | most requests handed to the backend at once, over every shard file |
| `io_writes_in_flight_per_file` | most on any one shard file; above one the file is pre-sized so its writes do not extend it |

What the write path then did is also recorded. Read `io_writes_in_flight_*`,
the depth reached, against `io_files_waiting_*`, the depth the scheduler had
to work with. A configured ceiling is not evidence the run got near it:

| field | holds |
|---|---|
| `io_files_waiting_mean` | shard files with a write waiting, averaged over time |
| `io_files_waiting_peak` | the most at once |
| `io_writes_in_flight_mean` | requests the backend was running at once, averaged over time: the depth reached, against the two rows above, which are the depth available |
| `io_writes_in_flight_peak` | the most at once |
| `io_files_opened` | shard files opened over the run |
| `io_files_open_peak` | the most open at once |
| `io_writes` | writes run by the queue |
| `io_bytes_copied` / `io_bytes_borrowed` | payload bytes owned by the write, against bytes borrowed from a pinned buffer |
| `io_queued_bytes_peak` / `io_queued_jobs_peak` | the largest backlog, in payload bytes and in jobs, counting a shard's finalizing truncate and close |
| `io_wait_ms_mean` / `io_wait_ms_max` | the wait before a write starts |
| `io_run_ms_mean` / `io_run_ms_max` | the time taken by a write once started |
| `io_write_sizes` | request-size histogram, as `{at_least, n}` in powers of two |

Under `stalls`, the CPU path reports each wait the producer makes on io on its
own, as a total and a count. They are a breakdown of the `sink` stage total
except for `append_extent`, which happens outside every stage. The GPU path
leaves all four at count zero, meaning not measured rather than no wait:

| field | holds |
|---|---|
| `io_fence_ms` / `io_fence_count` | the wait on an aggregate slot's previous writes, before the slot is filled again |
| `footer_buffer_ms` / `footer_buffer_count` | the wait on a shard's previous footer write, before its footer buffer is filled again |
| `append_extent_ms` / `append_extent_count` | the wait on shards closed since the append extent was last published |
| `flush_fence_ms` / `flush_fence_count` | the wait on every queued write, at flush |

The keys kept by the overview page are listed in `summary.py`, so a new counter
is dropped from that page unless it is named there. Everything not explicitly
dropped is taken by the explorer, so no change is needed there.

Measurement is from the first queued job of any kind until the run is flushed.
Covered: streaming and every shard's closing footer, truncate and close, but
not the startup before them. There is no payload on truncate and close, so
neither is counted toward the files-waiting figures; the footer write is. A
peak may therefore be the flush writing a footer to every open shard at once,
rather than streaming. None of these are recorded for a run with no filesystem
output.

Each run records memory as an estimate and a measurement:

| field | holds |
|---|---|
| `memory_estimate_total_bytes` | the engine's estimate — device bytes on GPU, heap bytes on CPU |
| `memory_estimate_pinned_bytes` | pinned host bytes on GPU, 0 on CPU |
| `memory_host_baseline_bytes` | resident memory before the stream was created |
| `memory_host_peak_bytes` | most resident memory the process held during the run |
| `memory_host_reading_failed` | true when either host reading was unavailable, since 0 is also a valid reading |
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

- **7** — The filesystem sink runs several writes at once instead of one, and
  pre-sizes a shard file when more than one of its writes may run together. No
  timing from a run with filesystem output is comparable across this bump; a
  run to the discard or S3 sink is unaffected.
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

# Retained Blosc experiments

The benchmark site's [**Blosc Pareto** analysis][pareto-analysis] compares the
RTX 5070 Laptop, RTX 5080, and L40 archives. The initial view includes all
systems, with input/chunk groups in rows and matching scales across system
columns. These are whole-system
measurements, including host work and transfers; they do not isolate GPU speed.
All three archives were measured using the
[`orca2_single` scenario](../../bench/bench_stream_orca2_single.c).

Build and serve the complete static site from the repository root:

```sh
uv run scripts/sweep/report.py --results-dir bench/results -o _site --serve
```

Open `http://127.0.0.1:8000/pareto.html`. No GPU, benchmark rerun, JavaScript
package installation, frontend framework, or external chart CDN is involved.
The existing Over time and Benchmark explorer pages keep their sweep behavior.

## Dataset contract

[datasets.json](datasets.json) is the version 1 source manifest. Each experiment
specifies a stable ID, archive directory, format adapter, matrix, workload,
hardware, methodology, retained files, and source metadata locations. Acquisition
dates and build details come from original provenance, including the 5080's
September 6 UTC timestamp despite its September 5 directory name.

[pareto_data.py](../../scripts/sweep/pareto_data.py) validates and normalizes
three supported formats:

| Format | Input | Validation |
|---|---|---|
| `summary-v1` | Historical summary CSV + provenance | Matrix, unique identities, ranges, repetitions, retained summary hash; raw repetition validation unavailable |
| `node-jsonl-v1` | Gzipped Node runner records + original summary | Raw hash, command/geometry/settings, status, warmups, every repetition, medians, ranges, memory and additional source metrics |
| `python-jsonl-v1` | Gzipped Python runner records, runs CSV, summary, collection manifest/build/validation | Collection hashes, raw/compact agreement, geometry/settings, warmups, every repetition and summary metrics |

The build writes `data/pareto/index.json` and one
`data/pareto/<experiment-id>.json` per experiment. Both have `version: 1`.
The index contains measurement definitions and experiment metadata. An experiment
file contains `experiment`, `workloads`, and `measurements`. Each measurement has:

| Field | Meaning |
|---|---|
| `id`, `experiment_id`, `configuration_id` | Globally unique point, experiment, and configuration identities (including codec level) |
| `workload_id` | Stable digest of full input, dtype, scenario, backend, sink, frame count, chunk shape and batch geometry |
| `fill`, `chunk_kib`, `codec`, `shuffle`, `block_kib`, `level`, `control` | Explicit configuration, with block 0 identifying a raw codec control |
| `throughput_gibs` | Original-input median, min, max; warmups excluded |
| `repetitions`, `compression_fold` | Measured repetition count and **reported padded-input** compression fold |
| `measured_device_gib` | Device free-memory delta median/min/max, not a sampled peak |
| `estimated_device_gib`, `estimated_pinned_gib` | Separate allocation estimates for device and pinned host memory |
| `source_metrics` | Every original summary field, preserving strings and field names |
| `samples` | Measured repetition numbers, throughput, memory bytes and one-based raw JSONL line references, or null for summary-only data |
| `provenance` | Original summary and metadata links, summary line, and raw archive link when available |

Numbers are never rounded during normalization or selection. Missing metrics are
null, never zero or inferred values. The summary-only 5070 retains its reported
throughput min–max, but has no raw samples, measured-memory ranges, or pinned
host estimates. Raw output includes additional per-run diagnostics and remains
available through the original archive links. `source_metrics` also retains old
frontier flags as historical fields; they do not drive current membership.

[pareto.mjs](../../scripts/sweep/pareto.mjs) is the sole frontier implementation,
imported by the browser and Node tests. It filters eligibility first, then groups
by experiment and full workload identity, plus codec for per-codec selection.
Cross-codec selection combines Blosc codecs. Both frontier modes maximize median
throughput and reported compression fold; memory remains a chart and filtering
quantity rather than a frontier objective. Raw controls are always visible and
never join a frontier.
Missing allocation estimates exclude a point from a budget that requires them.
Exact ties are retained. An allocation
budget always filters estimated **device** allocations, excluding pinned host
memory and additional runtime headroom. Overlay mode preserves all group boundaries.

## Adding an experiment

1. Retain source files under a new directory and give the experiment a new ID.
2. Add an entry to `datasets.json` using a supported format. Supply hardware,
   methodology, retained file paths and SHA256s, and point to original provenance.
3. Reuse a matching matrix and workload, or add their actual definitions. Never
   relabel different geometry as an existing workload to make it compare.
4. Run the validator and report build. The index, controls, columns, downloads,
   details and provenance links discover the experiment automatically.

You can pass `--pareto-manifest <path>` to report.py for another manifest. Its
archive paths are relative to its directory and must remain inside that directory.
The supported adapters validate their respective acquisition formats; a new
source format needs an adapter, not presentation changes. Archived hashes for
unretained binaries and original temporary paths are historical identifiers,
not dependencies that the report attempts to fetch.

## Retention and checksums

Raw files, original summary CSVs, numeric frontier/comparison/memory tables,
provenance, collection manifests, patches, scripts and logs are retained.
The previous figure generators, standalone HTML reports, and SVG/PNG/PDF figures
were superseded by the shared site. Numeric tables are historical snapshots and
test references; the build does not overwrite them.

The L40's original checksum inventory is preserved byte-for-byte as
`checksums.historical.sha256`. It still records the original documentation and
retired presentation artifacts. `checksums.sha256` is the reconciled inventory of
currently retained files. Original provenance, collection hashes and validation
records remain unchanged. The normalizer verifies retained source hashes and
the current inventory. Text hashes use repository LF bytes; CRLF-expanded Windows
checkouts may also verify against canonical LF without rewriting the files.
Gzip and decompressed raw-record hashes always require exact byte agreement.

## Validation and interface review

```sh
python scripts/sweep/pareto_data.py
uv run --with click --with rich --with pydantic python -m unittest discover -s scripts/sweep -p 'test_*.py'
uv run scripts/sweep/report.py --results-dir bench/results -o _site
node --test scripts/sweep/test_reports.mjs scripts/sweep/test_pareto.mjs
uv run scripts/sweep/test_pareto_browser.py
```

The browser test uses installed Edge on Windows, or Playwright Chromium elsewhere
(install with `uv run --with playwright==1.62.0 python -m playwright install chromium`).
It starts its own local server and saves desktop/mobile screenshots in both
themes to `.cache/pareto-browser-review`. It checks scale alignment, clipping,
keyboard and touch selection, sorting, CSV downloads, URL restoration, empty/error
states, navigation and an additional experiment. Node reference tests consume the
built `_site` data; set `PARETO_SITE` to test a different output directory.

The interface extends the existing Segoe/system font stack, with tabular numerals
and quiet grids. The comparison matrix carries the visual emphasis; controls and
provenance use the existing page palette (`#fcfcfb`, `#f9f9f7`, `#0b0b0b`,
`#52514e`). Blue LZ4 (`#126da8`) and orange Zstd (`#b34426`) gain lighter variants
in the dark theme. Squares, circles and triangles consistently mean none, byte
and bit shuffle. A visible keyboard focus and the sortable table provide access
to every point, including overlapping marks. Horizontal scales are logarithmic:
compression views begin at the meaningful 1× baseline and random-data plots use
their available range effectively. The Throughput × memory view always uses
estimated device allocation. Fitting remains an explicit, shareable choice
with a full-extent reset. Repetition bars show observed min–max, not confidence intervals.
The selected-setting panel limits measured values to three significant figures;
normalized JSON and CSV downloads retain the archived precision.

The implementation follows the reviewed
[frontend-design guidance](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md)
and [D3 guidance](https://github.com/benchflow-ai/skillsbench/blob/main/tasks/data-to-d3/environment/skills/d3-visualization/SKILL.md),
adapting them to this engineering comparison and the existing site. D3 7.9.0
and its ISC license are retained in `scripts/sweep/vendor`, shared by all tabs.

[pareto-analysis]: https://acquire-project.github.io/chucky/pareto.html

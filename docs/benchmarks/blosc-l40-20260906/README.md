# L40 Blosc measurements

PR264 at `2c131223a6d1a915df45b0d393bc20865bd74f92` was measured on an NVIDIA L40 on
2026-09-06, from 00:13:38 to 00:59:02 UTC (45.4 minutes).
All 1,200 executions passed: 200 configurations, each with one warmup and five
measured repetitions. The [performance guide](../../blosc-performance.md)
interprets the results and their memory costs.

The compression implementation is the unmodified PR revision. The retained
[benchmark-only patch](benchmark-controls.patch) adds shuffle/level CLI controls
and result metadata, using the controls from PR263 at
`95c2d7137aefd8843f80d75b73129b03cf0d3c09`. It preserves PR264's explicit
block-size validation. These private collection controls do not change the
regular sweep runner's supported matrix.

| Setting | Value |
|---|---|
| GPU | NVIDIA L40, 46,068 MiB reported VRAM, driver 580.126.20 |
| Node | `cw-us-e4a2-l40-234-085` |
| GPU UUID | `GPU-47053b1c-87c8-a0b2-c8a1-d1ff239f9775` |
| Build | Release, native sm_89, CUDA 13.1.115, GCC/G++ 14.3.0, nvCOMP 5.3.0.16 static |
| Scenario | `orca2_single`, u16, 100 frames, discard sink |
| Input shape | `(100, 2, 2048, 2304)`; 1.7578125 GiB ingested |
| Inputs | Deterministic uniform 12-bit random and coordinate XOR |
| Chunks | 256 KiB `(8,1,128,128)` or 1 MiB `(16,1,128,256)` |
| Actual batches | One epoch: 576 chunks / 144 MiB, or 288 chunks / 288 MiB |
| Target batch / device allocation budget | 64 MiB / 6 GiB |
| Append buffer | 33,554,432 u16 elements / 64 MiB, prefilled once and reused |
| Threads | Three worker threads, eight allocated CPU threads |
| Blosc | LZ4/Zstd, level 3, none/byte/bit shuffle |
| Blocks | 4/8/16/32/64/128/256/512/1024 KiB, omitting sizes above the chunk |
| Controls | Raw LZ4 level 1 and raw Zstd level 0; eight configurations |
| Order | One full warmup pass, then five measured passes; shuffled with RNG seed 169 |
| Processes | One new benchmark process per execution; configurations run sequentially |

The random pattern uses SplitMix64 with seed `0xdeadbeefcafebabe`, masked to
12 bits. The generators initialize a 16-frame pattern; the timed benchmark
prefills and repeatedly appends the same 64 MiB buffer. This is a controlled
synthetic workload, not 100 independent frames of microscopy data.

Throughput is original input GiB divided by the benchmark's append-and-flush
wall time, including append-buffer allocation and prefill. Stream creation is
timed separately. Compression fold divides padded chunk bytes by encoded bytes;
keeping 100 frames preserves the historical tail-padding fraction. Geometry,
settings, thread counts, and memory readings were checked on every execution.
The reported min–max is observed spread, not a confidence interval.

Memory estimates describe explicit device allocations for the whole stream;
`memory_estimate_total_bytes` excludes the separately reported pinned host bytes.
Observed memory compares free bytes before stream creation with after the run,
while the stream is alive. The initial CUDA context predates the baseline. This
is not a sampled peak; the signed residual is observed minus estimated bytes.
Memory tables combine fills only after checking that their estimates agree.

## Retained artifacts

| Artifact | Contents |
|---|---|
| [summary.csv](summary.csv) | 200 configuration summaries with median, min–max, compression fold, and device memory |
| [runs.csv](runs.csv) | All 1,200 executions, including the separately marked warmups |
| [results.jsonl.gz](results.jsonl.gz) | Full benchmark JSON, actual geometry, configuration, repeat, and UTC for each execution |
| [manifest.json](manifest.json) | All 200 commands and fixed measurement settings |
| [provenance.json](provenance.json), [build.json](build.json) | Source, build inputs, binary hashes, Slurm jobs, GPU readings before/after, and completion record |
| [validation.json](validation.json), [build.log](build.log), [gpu.log](gpu.log) | Passed correctness/CLI checks and the collection log |
| [pareto.svg](pareto.svg), [all candidates](pareto-all-candidates.svg) | Per-codec throughput/ratio frontiers across all shuffles |
| [pareto-frontier.csv](pareto-frontier.csv) | Per-codec frontier points with membership across both Blosc codecs |
| [pareto-memory-frontier.csv](pareto-memory-frontier.csv) | Frontier across both codecs maximizing speed/ratio and minimizing observed device memory |
| [memory.svg](memory.svg), [memory tables](memory-estimates.md), [memory CSV](memory-estimates.csv) | Observed deltas and 96 distinct allocation-estimate configurations |
| [comparison.svg](comparison.svg), [comparison table](comparison.md), [paired CSV](comparison.csv) | The matched L40 and historical RTX 5070 Laptop configurations |
| [analysis.json](analysis.json) | Timing spread, frontier changes, ratio agreement, and observed memory residuals |
| [pareto.html](pareto.html) | Self-contained figures and a filterable measurements table |
| [plot.py](plot.py) | Data validation, table generation, and Matplotlib figures; SVG, PNG, and PDF output |
| [sweep.py](sweep.py), [build.sh](build.sh), [gpu.sh](gpu.sh), [CLI checks](cli_check.py) | Exact collection harness and job scripts |
| [checksums.sha256](checksums.sha256) | SHA-256 for the retained files |

Raw per-execution stderr and binaries remain in the original shared run directory
recorded in `validation.json`; they are not required for plot generation. Full
JSON and compact individual measurements are bundled here. Binary and collection
input hashes are checked before measurement. The build and collection worktree
is separate from the active checkout.

## Comparison with the RTX 5070 Laptop

The [historical dataset](../blosc-rtx5070-20260905/README.md) has the same 200
configuration identities, input geometry, append size, and target batch/budget,
with one warmup and three measurements per configuration. Its old CSV `pareto`
field groups candidates by shuffle; this generator recomputes frontiers across
all three shuffles for both datasets. Raw controls are excluded from Blosc
frontiers. The three-objective frontier uses exact observed memory values, so
small memory-reading differences can create additional frontier points.

The earlier run used source `c519a05260cbda589bc323486170802a1c40c380`, nine
builds with different internal block-size constants, CUDA 13.2.51, Clang 21.1.8,
and driver 595.99.02. Both setups used nvCOMP 5.3.0.16. Source, compiler, toolkit,
driver, host, and GPU differ. This compares the two measured setups; it does
not isolate the effect of changing only the GPU. The historical archive is
preserved unchanged.

The figures follow the historical Pareto plot conventions: blue LZ4, orange
Zstd, squares for no shuffle, circles for byte shuffle, and triangles for
bitshuffle. XOR panels come first. Graphical legends identify the encodings;
point labels give block sizes with units, without repeating shuffle names.
The comparison uses solid lines and filled markers for L40, and dashed lines
and open markers for the RTX 5070 Laptop.

## Regenerate tables and figures

From the repository root, using Python 3.11 or later and the dependency pinned
in the script:

```sh
uv run --python 3.12 docs/benchmarks/blosc-l40-20260906/plot.py
```

To validate retained measurements without regenerating the plots:

```sh
uv run --python 3.12 docs/benchmarks/blosc-l40-20260906/plot.py --check
```

This needs no GPU, build, Slurm allocation, or new throughput measurements. The
generator checks raw/compact record agreement, all repetitions, medians/ranges,
ratios, allocation estimates, memory residuals, and fixed chunk/batch geometry.
It reads the preserved 5070 Laptop summary next to this directory.

## Collect new measurements

Use a new shared run directory. The archived job scripts contain the exact
cluster paths used here; adjust `run_root` and dependency paths for a different
location. `sweep.py` expects a detached worktree at `repo/` with the recorded
source revision and `benchmark-controls.patch` applied. Update the toolchain
paths in `record_build()` if the installation differs, then run `sweep.py prepare`
before building to record the new harness, patch, commands, and manifest hashes.

Build on a CPU compute node and run tests/measurements on an allocated L40.
The successful CPU job was `3559392`; the GPU job was
`3559405`. `build.sh` builds the two benchmarks and five
relevant test executables, runs both CPU checks, and records binary hashes.
`gpu.sh` checks those hashes, runs four CLI checks and three GPU tests, then
collects the matrix. The collector refuses to overwrite an existing raw run.

One matrix command, with the benchmark-only controls applied, is:

```sh
build/bench/bench_stream_orca2_single \
  --backend gpu --dtype u16 --frames 100 --fill rand \
  --chunk-bytes 1048576 --codec blosc-zstd --shuffle bit --level 3 \
  --blosc-block-bytes 1048576 --memory-budget 6G --batch-bytes 64M \
  --max-threads 3 --append-elements 33554432 --json
```

Retain each execution and its actual geometry. Increasing the frame count or
changing the repeated append buffer changes this workload and its padding;
save those experiments under a separate identity.

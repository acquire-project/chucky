# Chucky

[![CI][ci-badge]][ci]
[![codecov][codecov-badge]][codecov]

A high-performance streaming library for tiled transformation and compression of
large multidimensional arrays (tensors) using CUDA.

[Benchmarks][benchmarks]

## Overview

Chucky implements a GPU-accelerated streaming pipeline for writing compressed,
sharded [Zarr v3][zarr-v3] stores from high-throughput data sources (5+ GB/s).
Zarr v3 shards pack
multiple compressed chunks into a single file, reducing file count by orders of
magnitude compared to one-file-per-chunk layouts.

The pipeline stages are:

1. **Tiling** — partition the input tensor into fixed-size chunks
2. **Transpose** — scatter data so each chunk is contiguous in memory
3. **Compression** — batch-compress chunks on the GPU (Zstd, LZ4, or Blosc via nvCOMP)
4. **Aggregation** — pack compressed chunks into shards with an index
5. **Delivery** — D2H transfer and write shards to a Zarr v3 store

Output is [OME-NGFF v0.5][ome-ngff] with multiscale
LOD pyramids built on the fly: after the base level (L0) is chunked, the pipeline
scatters, reduces, and chunks each coarser level before compressing and delivering
it alongside L0.

**Supported element types:** u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64
(see `enum dtype` in `src/dtype.h`).

**Limits:** up to 32 dimensions for Zarr output (rank ≤ `MAX_ZARR_RANK`),
up to 32 LOD levels. Internal layout supports up to 64 dimensions.

## Getting Started

### Prerequisites

**Full (GPU) build:**

- **CUDA Toolkit** (12.8+) — CUDA runtime and nvcc compiler
- [**nvcomp**][nvcomp] (5.x) — NVIDIA compression library for GPU-accelerated
  codecs
- NVIDIA GPU with 8+ GB VRAM (16+ GB recommended for multiscale workloads)

**CPU-only build** — no CUDA or GPU required.

**Both:**

- **aws-c-s3** — Amazon S3 client library for S3 storage backend
- **lz4**, **zstd** — compression libraries
- **c-blosc** (optional) — enables CPU Blosc-LZ4/Blosc-Zstd and the GPU
  Blosc interoperability tests. GPU Blosc encoding uses nvCOMP and does not
  require c-blosc.
- **CMake** (3.18+) + **Ninja** — build system

The default build targets SM 100 (Blackwell). For other GPUs, set
`CMAKE_CUDA_ARCHITECTURES` at configure time.

### Build

The default preset enables the GPU backend. To force a CPU-only build, use the
`cpu-only` preset or pass `-DCHUCKY_ENABLE_GPU=OFF`. When
`CHUCKY_ENABLE_GPU` is not set explicitly, the CMake option still auto-detects
CUDA.

```
# Full build (GPU + CPU)
cmake --preset default
cmake --build build

# CPU-only build (no CUDA required)
cmake --preset cpu-only
cmake --build build
```

The default build uses four host output buffers and 32 filesystem I/O workers
per shard pool. Set `CHUCKY_OUTPUT_BUFFERS` (at least 2) and `CHUCKY_IO_WORKERS`
(at least 1) at configure time to change these defaults, for example:

```sh
cmake --preset default -DCHUCKY_OUTPUT_BUFFERS=8 -DCHUCKY_IO_WORKERS=16
```

CPU and GPU memory estimates include the configured output-buffer count. A
multiarray writer shares one output pool across its arrays. Explicit filesystem
scheduler limits take precedence over the worker default.

The easiest way to get the non-CUDA dependencies (lz4, zstd, c-blosc, aws-c-s3) is via
[vcpkg][vcpkg]. A `vcpkg.json` manifest is included in the repo:

```
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh   # or bootstrap-vcpkg.bat on Windows

cmake --preset default \
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

If you already have the dependencies installed (e.g. via your system package
manager or Nix), just use the default preset directly:

```
cmake --preset default
cmake --build build
```

If nvcomp or other dependencies are installed outside the default search paths,
create a `CMakeUserPresets.json` (git-ignored) to set `CMAKE_PREFIX_PATH`:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "default",
      "cacheVariables": {
        "CMAKE_PREFIX_PATH": "/path/to/nvcomp;/path/to/zstd"
      }
    }
  ]
}
```

Then build with your preset:

```
cmake --preset local
cmake --build build
```

### Run tests

```
ctest --test-dir build
```

### Docker

Run the full test suite (including S3 integration tests against MinIO):

```
docker compose up --build
docker compose down
```

This builds the project inside a CUDA container, starts a MinIO instance, and
runs `ctest`. GPU access uses [CDI][nvidia-cdi] (`nvidia.com/gpu=all`). MinIO stays running
after tests finish — `docker compose down` stops and removes everything.

To run a single test:

```
docker compose run test ctest --test-dir build -R test_zarr_s3_sink --output-on-failure
docker compose down
```

Build the image alone (no tests):

```
docker build -t chucky .
```

## Benchmarks

Several streaming benchmarks exercise the full pipeline on two representative
workloads (256³ cube and 4096×2304 Orca Quest 2 sensor) in single-scale,
multiscale, and multiscale-with-dim0-downsampling modes.

```
./build/bench/bench_stream_256cube_single [options]
./build/bench/bench_stream_orca2_multiscale [options]
```

**Available benchmarks:**

- `bench_stream_256cube_single` / `_multiscale` / `_multiscale_dim0`
- `bench_stream_orca2_single` / `_multiscale` / `_multiscale_dim0`
- `bench_stream_medfmt_single` / `_multiscale` / `_multiscale_dim0`
- `bench_stream_smallepoch_single` — a tiny epoch, one shard file at a time
- `bench_stream_smallepoch_4shards` — a tiny epoch, 4 concurrent shards
- `bench_stream_aqz_single`
- `bench_stream_256cube_two_streams` — two streams sharing one GPU

**Options:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--fill` | `xor`, `zeros`, `rand` | `xor` | Synthetic fill pattern for input data |
| `--codec` | `none`, `lz4`, `zstd`, `blosc-lz4`, `blosc-zstd` | `zstd` | Compression codec |
| `--blosc-block-bytes` | e.g. `16K`, `64K`, `4097` | Required for Blosc | Internal Blosc block size in bytes |
| `--blosc-shuffle` | `none`, `byte`, `bit` | `none` | Blosc filter; recorded with the level in benchmark JSON |
| `--level` | Integer; 0–9 for Blosc | Blosc: 3; LZ4: 1; Zstd: 0 | Compression level; Blosc level 0 stores input without compression |
| `--reduce` | `mean`, `min`, `max`, `median`, `max_sup`, `min_sup` | `mean` | LOD reduction method |
| `--duration` | seconds, > 0 | 1 | Minimum measured append duration; coverage may extend it and final drain is included |
| `--frames` | frame count | 0 (unbounded) | Minimum measured input; may be combined with duration |
| `--geometry-frames` | positive frame count | scenario reference | Reference extent used only to fit chunk, shard, epoch, and batch geometry |
| `--concurrent-shards` | positive integer | scenario target | Target number of active shard files; actual count depends on chunk and array geometry |
| `--warmup` | seconds, >= 0 | 0.25 | Minimum warmup; always at least 0.25 s and two batches, then drain and reset metrics |
| `--max-attempts` | positive integer | 5 | Maximum measurement attempts before insufficient coverage is an error |
| `--no-boundary-timing` | flag | off | Disable full-API samples to check their observer overhead |
| `--append-elements` | element count | bulk | Generated input: bytes must divide 64 MiB; images allow arbitrary sizes |
| `--full-memcpy-timing` | flag | off | GPU profiling: time every host copy instead of sampling small copies |
| `-o path` | output directory | omit to discard | Write Zarr output to disk |

Benchmarks report per-stage throughput and latency, compression ratio, memory
breakdown, and overall pipeline GiB/s.

Single-stream benchmarks use the same timing policy for CPU/GPU, single-scale
and multiscale, generated/image inputs, and discard/filesystem/S3/throttled sinks. For example:

```sh
./build/bench/bench_stream_smallepoch_single --backend gpu --codec none \
  --fill xor --dtype u16 --geometry-frames 65536 --chunk-bytes 1M \
  --batch-bytes 64M --memory-budget 5G --append-elements 256 --json
```

`--geometry-frames` fixes the layout reference. `--frames N` requests a minimum
measured input, and `--duration S` requests a minimum measured append time;
neither changes that layout. The underlying frame dimension is unbounded. Source
preparation and stream creation precede warmup. Generated inputs use a fixed 64 MiB source ring. Images use a preloaded,
chunk-padded ring; loading and padding are excluded from timing.

Warmup continues to a full batch boundary with no partial append-downsample
accumulator, drains earlier work and sink metadata, and resets stage timings,
append histograms, maxima, and work counters. It keeps the same pipeline and
allocations. No warmup input is in flight when measurement begins. Measurement
includes all its accepted input through final flush, close, and metadata
publication. `append_s` and `drain_s` partition this window; primary throughput
uses their sum. Output bytes count physical sink writes, including padding and
footers, and are read after drain. Output files include **both warmup and measured
input**; their extent is not just `--frames N`.

TTY and top-level JSON rates/stage metrics describe this same window. The
`measurement` object records the policy, requested and actual durations, warmup
and measured work, effective per-dimension geometry, epoch/batch/staging sizes,
and boundary samples. `throughput_in_gibs` counts submitted bytes;
`throughput_logical_gibs` excludes image padding. `compression_fold` counts
full decoded chunks, while `logical_compression_fold` counts logical input.
Version 12 adds images to the common version-11 window. Neither is directly
comparable to older whole-run or no-drain sustained rates. The specialized two-stream driver
retains its explicit fixed-frame, whole-run policy and rejects timing options.

The driver enforces coverage for every successful single-stream run, including
all sweep scenarios. It starts with 0.25 s warmup / 1 s measurement defaults.
Coverage requires at least 0.25 s and two batches of warmup, then 0.25 s, four
complete batches, and two generation transitions during measurement. Final drain
must take no more than 10% of the measured window. This is a conservative
screening budget, not a measured precision crossover. The batch
reuse count is a conservative lower bound after allowing two batch buffers.
These counts use input positions, independently of the geometry reference and
`min_append_shards`; that fitter constraint is no longer a coverage guarantee.
Warmup and measurement extend until the time and work minima are met, including
when `--warmup 0` or a tiny frame count is requested. After final close, a drain
over 10% rejects the attempt. The driver recreates the stream and sink using
the same fitted geometry and source, then repeats warmup and measurement with
an append target of at least twice the previous append time or 18 times the
observed drain, whichever is larger. It checks the actual drain again.

Only the final qualified attempt contributes rates and stage metrics. JSON and
TTY report its attempt number, effective duration target, and time spent on
discarded attempts. Filesystem retries replace the previous attempt's dataset;
S3 retries write to the same prefix. After `--max-attempts` (default five), an
unqualified run exits nonzero with `status: error`, `error: insufficient_coverage`,
and the final attempt's measurement diagnostics. It cannot report `PASS` with
insufficient coverage. Meeting these minima does not establish steady-state
accuracy or guarantee a particular statistical precision.

Validate representative short runs against longer references (`--warmup 2
--duration 5`, or longer for slow cases) with the same geometry, source, append
size, codec, sink, and worker allocation. Repeat paired runs in reverse order,
include `orca2_single --geometry-frames 200` and CPU controls, and compare rates,
variation, coverage, and drain fraction. Request longer runs where more precision
is needed. A blocking append or
warmup alignment can overshoot requested durations; use actual times.

Full-API latency samples cover calls crossing input batch/generation boundaries,
a possible staging grid (GCD of batch bytes and staging capacity), and the next
16 calls. Groups overlap and do not establish maxima for unsampled calls. The
backend append histogram uses the same warmup exclusion; its sampling scope
still depends on the backend. `--no-boundary-timing` checks observer overhead
without disabling the library timers.

Input buffering ([#272](https://github.com/acquire-project/chucky/issues/272))
remains a separate producer-latency feature. A queue can absorb caller stalls
while downstream initialization continues; it cannot warm the downstream
pipeline by itself. Compare direct and buffered runs with the same downstream
work and drained endpoints, reporting occupancy, backpressure, and downstream
delay separately from caller latency. Extra copying can reduce throughput,
especially with codec `none`; buffering is not enabled as a benchmark default.

## Architecture

The pipeline uses a four-stream CUDA model with double-buffered staging to
overlap H2D transfer, GPU compute (scatter, compress, aggregate), and D2H
delivery. Input arrives as contiguous byte spans via a `struct writer` interface;
the library handles all tiling, padding, and shard assembly internally. See
[docs/design.md][docs-design-md] for a detailed walkthrough, or
[docs/guide.md][docs-guide-md] for a quick orientation to the module structure.

Paced producers can opt into the [bounded input-buffering writer](docs/buffered-writer.md).
Filesystem streams [prepare one shard generation ahead](docs/shard-preparation.md)
by default, including cleanup of unused files at shutdown. Set
`config.disable_shard_preparation = 1` to open files on demand instead.

For writing directly to S3 (or S3-compatible stores), see the
[S3 storage guide][s3-storage-guide].

For Blosc, see the [binary format specification][blosc-format], the
[performance and GPU memory guide][blosc-performance], and the
[interactive Pareto analysis][blosc-pareto].

## API

The main entry points live in [`src/stream.gpu.h`][src-stream-gpu-h] (GPU) and
[`src/stream.cpu.h`][src-stream-cpu-h] (CPU). Both backends expose the same
`struct writer` interface for feeding data. For frames arriving on a regular
camera clock, use the [buffered input adapter](#camera-frames-at-a-regular-frequency)
described below.

```c
// 1. Estimate GPU memory requirements
struct tile_stream_memory_info info;
tile_stream_gpu_memory_estimate(&config, 0, &info);

// 2. Create the stream
struct tile_stream_gpu* s = tile_stream_gpu_create(&config, sink);

// 3. Get a writer and feed data
struct writer* w = tile_stream_gpu_writer(s);
struct slice frame = { .beg = data, .end = (const char*)data + nbytes };
writer_append(w, frame);   // call repeatedly as data arrives
writer_flush(w);           // finalize: takes no input after this
writer_close(w);           // wait for the writes, publish the shape

// 4. Query metrics and tear down
struct stream_metrics m = tile_stream_gpu_get_metrics(s);
tile_stream_gpu_destroy(s);
```

The CPU backend (`tile_stream_cpu_create` / `tile_stream_cpu_writer`) follows
the same pattern — swap `gpu` for `cpu` in the function names. The GPU backend
uses CUDA streams + nvcomp; the CPU backend uses OpenMP + zstd/lz4 and optional
c-blosc.

Configure the pipeline via `struct tile_stream_configuration` (codec, chunk
dimensions, shard layout, LOD reduction method, etc.). See
[`src/stream.gpu.h`][src-stream-gpu-h] or
[`src/stream.cpu.h`][src-stream-cpu-h] for the full API.

### Camera frames at a regular frequency

Use the [buffered input writer](src/writer.buffered.h) when frames arrive on a
camera's clock. Direct stream appends can spend time processing a batch or waiting
for output capacity. The buffered adapter copies accepted input into a bounded
queue and forwards it to the stream on one worker thread. The camera can reuse
accepted bytes as soon as append returns, while the stream continues processing.

These buffers have different roles:

| Buffer | Control | Purpose |
| --- | --- | --- |
| Camera pool | Acquisition application | Holds arriving frames until the writer accepts them |
| Buffered input queue | `buffered_writer_config.capacity_bytes` | Owns copies of accepted input while the forwarding thread feeds the stream |
| Stream processing buffers | `tile_stream_configuration.buffer_capacity_bytes` and `target_batch_bytes` | Size the stream's internal input capacity and processing batches |

Setting a stream buffer or batch to 64 MiB does **not** enable the buffered input
adapter. Create it explicitly after creating the stream and before starting the
camera. The example below allows 250 ms at 4 useful GiB/s for BBBC022's padded
frames; size the queue for your own submitted byte rate and pause allowance:

```c
#include "writer.buffered.h"

struct buffered_writer_config input_config = {
  .capacity_bytes = 1152u * 1024 * 1024,
  .max_drain_bytes = 64u * 1024 * 1024,
};
struct buffered_writer* input = buffered_writer_create(
  tile_stream_cpu_writer(stream), &input_config);
if (!input)
  return 1;
struct writer* frames = buffered_writer_as_writer(input);
```

Submit each frame with `writer_append_wait(frames, frame_slice)` and check its
error and unaccepted remainder before recycling the entire camera buffer.
`max_drain_bytes` caps each call to the stream; the worker can forward a smaller
available batch. Both byte limits must respect the stream's input granularity,
and the drain limit cannot exceed capacity. Zero drain limit uses capacity.

Size input capacity from the bytes actually submitted, including padding:

`capacity >= submitted_bytes_per_second * pause_seconds + active_drain_bytes`

Round up for whole frames during the pause and for the downstream input
granularity. The extra drain allowance covers input still owned by the forwarding
call. This assumes the queue has cleared earlier backlog before the pause.
Keep `max_drain_bytes` separate from capacity: increasing the latency allowance
need not increase the compression batch or drain size.

At 4 useful GiB/s, BBBC022's 520×696 uint16 frames padded to 544×704 for 16 KiB
chunks produce 4.23 GiB/s of submitted input. A 250 ms pause requires about
1.06 GiB, plus one 64 MiB drain; rounding up to a drain-sized block gives
1,152 MiB. The camera pool is additional. A 64 MiB queue alone holds only about
14.8 ms of input at that rate.

Creation touches every page of the adapter queue and waits for its forwarding
thread. CPU stream creation also touches its main payload buffers and waits for
its processing workers. Initialization takes time and commits memory before the
first append; include it in startup and memory budgets. The input queue uses
ordinary host memory. GPU staging and output buffers are page-locked separately.
On NUMA hosts, set CPU affinity and memory binding before creation; for one GPU,
use its local node. Touching pages does not prevent later NUMA-balancing faults.

Append blocks when the input queue is full. Keep enough camera buffers to absorb
that wait when acquisition cannot pause, and keep capture independent of the
append thread. The queue adds a copy, its configured
memory capacity, and one forwarding thread; measure total memory and buffer-return
latency for the intended frame rate. `buffered_writer_get_stats()` reports pending
bytes, peak occupancy, and time spent waiting for space.

Use the adapter exclusively while it owns the stream's writer interface. At the
end, call `buffered_writer_destroy(input)` and check its return value: it drains,
flushes, and closes the downstream writer before freeing the adapter. Destroy the
stream and its sink afterwards. Returning a camera buffer only means its input
was copied; output completion requires the final drain and the sink's completion
guarantees. Flush finalizes the stream, so do not flush after individual frames.

Test startup and accumulated backlog at the intended camera rate, and count
both the camera pool and adapter queue in the memory budget.

### GPU host placement

On Linux, GPU streams use the NUMA node of the current CUDA context's device
when its PCI topology is available. Stream-owned copy and delivery workers run
on that node's CPUs within the caller's allowed CPU set. Host staging and output
buffers are touched during creation and page-locked for CUDA transfers. Allocation
prefers GPU-local memory, with fallback to other allowed nodes if local RAM is
full; an explicit caller memory policy takes precedence.

Creation restores the calling thread's CPU affinity and memory policy. Each
append temporarily uses local CPUs within that thread's current affinity, then
restores it, including when a buffered writer forwards the append from another
thread. Separately created filesystem workers, input buffers, and other caller
threads retain their own placement. Missing topology, restricted system calls,
and platforms without this NUMA support keep the existing allocation behavior.

### Blosc configuration

The [format specification][blosc-format] describes the encoded bytes.
The [performance guide][blosc-performance] covers block-size selection,
shuffle, GPU memory estimates, and benchmarking.

Blosc requires an explicit block size on both CPU and GPU:

```c
config.codec = (struct codec_config){
    .id = CODEC_BLOSC_ZSTD,       // or CODEC_BLOSC_LZ4
    .level = 3,                  // 0 = store-only
    .shuffle = CODEC_SHUFFLE_BIT, // or CODEC_SHUFFLE_BYTE / CODEC_SHUFFLE_NONE
    .blosc_block_bytes = 16 * 1024,
};
```

`blosc_block_bytes` is the requested uncompressed size of an internal Blosc
block, independent of the multidimensional Zarr chunk shape. It is required
even at level 0: zero is invalid, not an automatic/default selection. Values
below 128 or above `(INT32_MAX - 255 * 4) / 3` are rejected instead of silently
clamped; the GPU backend additionally checks nvCOMP's per-input size limit.
Other codecs ignore this field.

On GPU the block size is capped by the chunk size, with a shorter final block
when needed. The encoder handles device alignment internally; callers simply append
their bytes, and block sizes need not be aligned or powers of two. Aligned block
sizes with no filter use the existing chunk layout directly. Shuffle and
misaligned block boundaries share an internal preparation buffer; memory
estimates include that scratch.

On CPU the value is passed explicitly to `blosc_compress_ctx`. C-Blosc's
[block splitting and element-alignment rules][block-splitting-and-element-alignment-rules]
can adjust the resulting frame block size; the encoder does not request automatic
selection or change global Blosc settings. GPU levels 1 through 9 all use
nvCOMP's single compression mode, while CPU Blosc honors the requested level.

Use the same codec configuration for the stream and its Zarr/NGFF sink. Zarr
metadata records the requested `blocksize`; each Blosc frame records its actual
block size. A GPU multiarray stream requires the same codec id and
`blosc_block_bytes` for all arrays sharing its codec instance.

Benchmarks require `--blosc-block-bytes` when a Blosc codec is selected:

```sh
./build/bench/bench_stream_orca2_single --backend gpu --codec blosc-zstd \
  --blosc-block-bytes 16K --blosc-shuffle bit
```

The benchmark executable exposes `--blosc-shuffle none|byte|bit` and `--level`.
Blosc JSON records `blosc_shuffle` and `blosc_level`; raw codecs record `level`.
Blosc defaults to level 3, and `--level 0` selects store-only mode regardless
of its position before or after `--codec`. The retained L40 tuning sweep used
the older [`--shuffle`/`--level` controls][blosc-benchmark-controls] preserved
with that archive. The command above matches the bitshuffle API example at
level 3.

The existing sweep suite explicitly selects 16 KiB for its Blosc cases; this is
a benchmark choice, not a codec API default.

## Status

Pre-release. Functional streaming pipeline with multiscale LOD support and Zarr
v3 sharded output. The API may change without notice. Under active development.
I plan to use this as a future backbone for [acquire-zarr][acquire-zarr].

[zarr-v3]: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
[ome-ngff]: https://ngff.openmicroscopy.org/0.5/
[nvcomp]: https://developer.nvidia.com/nvcomp
[vcpkg]: https://vcpkg.io/
[nvidia-cdi]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
[acquire-zarr]: https://github.com/acquire-project/acquire-zarr

[ci-badge]: https://github.com/acquire-project/chucky/actions/workflows/ci.yml/badge.svg
[ci]: https://github.com/acquire-project/chucky/actions/workflows/ci.yml
[codecov-badge]: https://codecov.io/gh/acquire-project/chucky/graph/badge.svg
[codecov]: https://codecov.io/gh/acquire-project/chucky
[benchmarks]: https://acquire-project.github.io/chucky/
[docs-design-md]: docs/design.md
[docs-guide-md]: docs/guide.md
[s3-storage-guide]: docs/s3-guide.md
[blosc-format]: docs/blosc-format.md
[blosc-performance]: docs/blosc-performance.md
[blosc-pareto]: https://acquire-project.github.io/chucky/pareto.html
[src-stream-gpu-h]: src/stream.gpu.h
[src-stream-cpu-h]: src/stream.cpu.h
[block-splitting-and-element-alignment-rules]: https://github.com/Blosc/c-blosc/blob/616f4b7343a8479f7e71dd3d7025bd92c9a6bbd0/blosc/blosc.c#L934-L1064
[blosc-benchmark-controls]: docs/benchmarks/blosc-l40-20260906/benchmark-controls.patch

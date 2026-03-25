# Developer Orientation Guide

Quick-reference guide to the chucky codebase: module layout, public API,
internal conventions, and key concepts.

## Module map

### Platform & utilities

| Target | Source | Purpose |
|--------|--------|---------|
| `dimension` | `src/dimension.c` | `struct dimension` builder/validation helpers |
| `chucky_log` | `src/log/` | Logging |
| `platform` | `src/platform/platform.*.c` | OS abstraction (page size, clock) |
| `platform_io` | `src/platform/platform_io.*.c` | Unbuffered file I/O |
| `platform_cmd` | `src/platform/platform_cmd.*.c` | Subprocess execution |
| `io_queue` | `src/zarr/io_queue.*.c` | Async I/O queue (threaded) |
| `index_ops` | `src/util/index.ops.c` | Mixed-radix index arithmetic |
| `crc32c` | `src/zarr/crc32c.c` | CRC-32C checksum |
| `writer` | `src/writer.c` | `struct writer` dispatch + backpressure helpers |

### Data model

| Header | Purpose |
|--------|---------|
| `dtype.h` | `enum dtype` (11 types) + `dtype_bpe()` |
| `dimension.h` | `struct dimension` + builder helpers (`dims_create`, `dims_set_chunk_sizes`, etc.) |
| `types.stream.h` | `struct tile_stream_configuration`, `struct tile_stream_status`, `struct stream_metrics` |
| `types.codec.h` | `enum compression_codec` (none / lz4 / zstd) |
| `types.lod.h` | `enum lod_reduce_method` (mean / min / max / median / max_suppressed / min_suppressed) |
| `defs.limits.h` | Compile-time limits: `MAX_RANK=64`, `MAX_ZARR_RANK=32`, `LOD_MAX_LEVELS=32` |

### Layout & planning

| Target | Source | Purpose |
|--------|--------|---------|
| `stream_config` | `src/stream/config.c` | Compute stream layouts, aggregate types, batch sizing |
| `lod_plan` | `src/lod/lod_plan.c` | Pure-C LOD plan: per-level shapes, counts, ends arrays |

### GPU backend

| Target | Source | Purpose |
|--------|--------|---------|
| `transpose` | `src/gpu/transpose.cu` | CUDA scatter kernel (input -> chunk pool) |
| `compress` | `src/gpu/compress.cu` | Batch compress via nvcomp (lz4 / zstd) |
| `aggregate` | `src/gpu/aggregate.cu` | Pack compressed chunks into shard buffers |
| `lod` | `src/gpu/lod.cu` | GPU LOD scatter + reduce (all 11 dtypes) |
| `stream` | `src/gpu/stream.c` + helpers | GPU pipeline orchestrator |

### CPU backend

| Target | Source | Purpose |
|--------|--------|---------|
| `transpose_cpu` | `src/cpu/transpose.cpp` | OpenMP scatter |
| `compress_cpu` | `src/cpu/compress.c` | zstd / lz4 compression (CPU) |
| `aggregate_cpu` | `src/cpu/aggregate.c` | Shard packing (CPU) |
| `lod_cpu` | `src/cpu/lod.cpp` | CPU LOD scatter + reduce |
| `stream_cpu` | `src/cpu/stream.c` | CPU pipeline orchestrator |

### Zarr storage

| Target | Source | Purpose |
|--------|--------|---------|
| `json_writer` | `src/zarr/json_writer.c` | JSON serialization for zarr metadata |
| `zarr_metadata` | `src/zarr/zarr_metadata.c` | Write zarr.json / .zarray / OME-NGFF metadata |
| `shard_delivery` | `src/zarr/shard_delivery.c` | Write shard index + CRC, deliver to shard_writer |
| `zarr_fs_sink` | `src/zarr/zarr_fs_sink.c` | `shard_sink` for local filesystem Zarr stores |
| `s3_client` | `src/zarr/s3_client.c` | AWS S3 multipart upload client |
| `zarr_s3_sink` | `src/zarr/zarr_s3_sink.c` | `shard_sink` for S3-backed Zarr stores |

## Public API headers

A library consumer needs:

- **`stream.gpu.h`** — GPU pipeline: create, destroy, writer, memory estimate
- **`stream.cpu.h`** — CPU pipeline: same interface, no CUDA dependency
- **`writer.h`** — `struct writer`, `struct slice`, `writer_append()`, `writer_flush()`
- **`dimension.h`** — `struct dimension` + builder/validation helpers
- **`dtype.h`** — `enum dtype`, `dtype_bpe()`, `dtype_zarr_string()`
- **`types.stream.h`** — `struct tile_stream_configuration`, metrics, status
- **`types.codec.h`** — `enum compression_codec`
- **`types.lod.h`** — `enum lod_reduce_method`
- **`defs.limits.h`** — compile-time limits
- **`zarr_fs_sink.h`** — filesystem Zarr sink
- **`zarr_s3_sink.h`** — S3 Zarr sink

## Internal conventions

**Error handling.** Functions return `int` (0 = success, non-zero = error).
Callers test with `if (func())`. Macros: `CHECK(label, expr)` for assertions
with goto-cleanup, `CU(label, expr)` for CUDA calls.

**Naming.** Dots for namespacing headers (`stream.gpu.h`, `types.stream.h`).
Underscores within identifiers (`platform_io`, `lod_plan`). Prefix
`tile_stream_` for the public streaming API.

**Memory.** `buffer_new` / `buffer_free` for pinned host or device memory with
`CUevent` synchronization. Host buffers use `CU_MEMHOSTALLOC_WRITECOMBINED` —
do not read from the host side; copy data out first.

**Writer vtable.** `struct writer` has two methods:
`append(self, slice) -> writer_result` and `flush(self) -> writer_result`.
Free functions `writer_append()` / `writer_flush()` dispatch through the vtable.

**Two-phase init.** `compute_stream_layouts()` does all pure-CPU layout math
(lifted shape, strides, chunk geometry), then the GPU path uploads to device
memory. `tile_stream_gpu_memory_estimate` reuses the same layout function.

## GPU vs CPU backends

Both backends implement the same pipeline stages behind the same
`struct writer` interface:

| Stage | GPU | CPU |
|-------|-----|-----|
| Scatter | CUDA kernel | OpenMP parallel loop |
| Compress | nvcomp (lz4 / zstd) | libzstd / liblz4 |
| Aggregate | CUDA kernel | Sequential packing |
| LOD | CUDA kernels (templated on dtype) | C++ templates + OpenMP |
| Orchestration | 4 CUDA streams, double-buffered | Single-threaded pipeline |

Create with `tile_stream_gpu_create()` or `tile_stream_cpu_create()`.
Both return a `struct writer*` via `_writer()` — downstream code is identical.

## Key concepts

- **epoch** — one full pass of the inner chunk dimensions; the append dimension
  advances by `chunk_size[0]` per epoch
- **batch** — `K` epochs grouped for compression; `K` is auto-tuned or set via
  `epochs_per_batch`
- **chunk** — Zarr's independently compressed unit; shaped by
  `dimension.chunk_size` per axis
- **shard** — a file containing multiple compressed chunks plus a binary index
- **lifted shape** — the input tensor reshaped as
  `(t[D-1], n[D-1], ..., t[0], n[0])` to expose chunk structure
- **append dimension** — dimension 0; may have `size=0` (unbounded, streams
  indefinitely)
- **inner dimensions** — dimensions 1..rank-1; fully known at stream creation
- **LOD level** — one layer of the multiscale pyramid; L0 = full resolution
- **compacted morton order** — Z-order curve over chunk indices within a shard,
  used to map LOD chunks to shard positions

## Further reading

- [docs/design.md](design.md) — full design walkthrough (problem, pipeline, memory model, API)
- [docs/streaming.md](streaming.md) — chunk lifetime math and ring-buffer proof
- [docs/sharding.md](sharding.md) — shard layout and index format
- [docs/s3-guide.md](s3-guide.md) — S3 storage backend setup

# Chucky

A high-performance streaming library for tiled transformation and compression of
large multidimensional arrays (tensors) using CUDA.

## Overview

Chucky implements a GPU-accelerated streaming pipeline for writing compressed,
sharded Zarr v3 stores from high-throughput data sources (2–5 GB/s).
The pipeline stages are:

1. **Tiling** — partition the input tensor into fixed-size tiles
2. **Transpose** — scatter data so each tile is contiguous in memory
3. **Compression** — batch-compress tiles on the GPU (zstd or lz4 via nvcomp)
4. **Aggregation** — pack compressed tiles into shards with an index
5. **Delivery** — D2H transfer and write shards to a Zarr v3 store

Output is OME-NGFF v0.5 with multiscale LOD pyramids built on the fly: after
the base level (L0) is tiled, the pipeline scatters, reduces, and tiles each
coarser level before compressing and delivering it alongside L0.

## Getting Started

### Prerequisites

An NVIDIA GPU with at least 8 GB VRAM is required at runtime.

- **CUDA Toolkit** (12.8+) — CUDA runtime and nvcc compiler
- **nvcomp** (5.x) — NVIDIA compression library for GPU
- **zstd** — Zstandard compression (CPU-side, used by tests)
- **CMake** (3.18+) + **Ninja** — build system
- NVIDIA GPU with 8+ GB VRAM (16 GB recommended for multiscale workloads)

The default build targets SM 100. For other GPUs, set `CMAKE_CUDA_ARCHITECTURES`
at configure time.

### Build

When the dependencies are in standard locations, use the default preset:

```
cmake --preset default
cmake --build build
```

If nvcomp or zstd are installed outside the default search paths, create a
`CMakeUserPresets.json` (git-ignored) to set `CMAKE_PREFIX_PATH`:

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

**Options:**

| Flag | Values | Default |
|------|--------|---------|
| `--fill` | `xor`, `zeros`, `thirds` | `xor` |
| `--codec` | `none`, `lz4`, `zstd` | `zstd` |
| `--reduce` | `mean`, `min`, `max`, `median`, `max_sup`, `min_sup` | `mean` |
| `-o path` | output directory | omit to discard |

Benchmarks report per-stage throughput and latency, compression ratio, memory
breakdown, and overall pipeline GB/s.

## Project Structure

```
src/       library sources (C/CUDA)
bench/     streaming benchmarks
tests/     unit and integration tests
docs/      design docs (design.md, streaming.md, sharding.md)
```

## Architecture

The pipeline uses a four-stream CUDA model with double-buffered staging to
overlap H2D transfer, GPU compute (scatter, compress, aggregate), and D2H
delivery. Input arrives as contiguous byte spans via a `struct writer` interface;
the library handles all tiling, padding, and shard assembly internally. See
[docs/design.md](docs/design.md) for a detailed walkthrough.

Pipeline stages:

- **H2D** — async copy from pinned host buffers to device staging
- **Scatter** — transpose kernel places elements into the tile pool
- **Compress + Aggregate** — batch nvcomp compression, then shard packing
- **D2H + Sink** — transfer shards to host and write via the shard sink

## API

The main entry points live in [`src/stream.h`](src/stream.h):

- `tile_stream_gpu_memory_estimate()` — estimate GPU memory without allocating
- `tile_stream_gpu_create()` / `tile_stream_gpu_destroy()` — lifecycle
- `tile_stream_gpu_writer()` — obtain a `struct writer` to append data
- `tile_stream_gpu_get_metrics()` — retrieve per-stage timing metrics

Configure the pipeline via `struct tile_stream_configuration` (codec, tile
dimensions, shard layout, LOD reduction method, etc.).


## Status

Functional streaming pipeline with multiscale LOD support and Zarr v3 sharded
output. Under active development.

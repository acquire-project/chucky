# Chucky

A high-performance streaming library for tiled transformation and compression of
large multidimensional arrays (tensors) using CUDA.

## Overview

Chucky implements a GPU-accelerated streaming pipeline for writing compressed,
sharded [Zarr v3][zarr-v3] stores from high-throughput data sources (2–5 GB/s).
Zarr v3 shards pack
multiple compressed chunks into a single file, reducing file count by orders of
magnitude compared to one-file-per-chunk layouts.

The pipeline stages are:

1. **Tiling** — partition the input tensor into fixed-size chunks
2. **Transpose** — scatter data so each chunk is contiguous in memory
3. **Compression** — batch-compress chunks on the GPU (zstd or lz4 via nvcomp)
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

- **CUDA Toolkit** (12.8+) — CUDA runtime and nvcc compiler
- [**nvcomp**][nvcomp] (5.x) — NVIDIA compression library for GPU-accelerated
  codecs
- **aws-c-s3** — Amazon S3 client library for S3 storage backend
- **zstd** — Zstandard compression (CPU-side, used by tests)
- **CMake** (3.18+) + **Ninja** — build system
- NVIDIA GPU with 8+ GB VRAM (16+ GB recommended for multiscale workloads)

The default build targets SM 100 (Blackwell). For other GPUs, set
`CMAKE_CUDA_ARCHITECTURES` at configure time.

### Build

The easiest way to get the non-CUDA dependencies (lz4, zstd, aws-c-s3) is via
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

**Options:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--fill` | `xor`, `zeros`, `thirds` | `xor` | Synthetic fill pattern for input data |
| `--codec` | `none`, `lz4`, `zstd` | `zstd` | Compression codec |
| `--reduce` | `mean`, `min`, `max`, `median`, `max_sup`, `min_sup` | `mean` | LOD reduction method |
| `-o path` | output directory | omit to discard | Write Zarr output to disk |

Benchmarks report per-stage throughput and latency, compression ratio, memory
breakdown, and overall pipeline GB/s.

## Architecture

The pipeline uses a four-stream CUDA model with double-buffered staging to
overlap H2D transfer, GPU compute (scatter, compress, aggregate), and D2H
delivery. Input arrives as contiguous byte spans via a `struct writer` interface;
the library handles all tiling, padding, and shard assembly internally. See
[docs/design.md](docs/design.md) for a detailed walkthrough, or
[docs/guide.md](docs/guide.md) for a quick orientation to the module structure.

For writing directly to S3 (or S3-compatible stores), see the
[S3 storage guide](docs/s3-guide.md).


## API

The main entry points live in [`src/stream.gpu.h`](src/stream.gpu.h) (GPU) and
[`src/stream.cpu.h`](src/stream.cpu.h) (CPU). Both backends expose the same
`struct writer` interface for feeding data.

```c
// 1. Estimate GPU memory requirements
struct tile_stream_memory_info info;
tile_stream_gpu_memory_estimate(&config, &info);

// 2. Create the stream
struct tile_stream_gpu* s = tile_stream_gpu_create(&config, sink);

// 3. Get a writer and feed data
struct writer* w = tile_stream_gpu_writer(s);
struct slice frame = { .beg = data, .end = (const char*)data + nbytes };
writer_append(w, frame);   // call repeatedly as data arrives
writer_flush(w);            // finalize

// 4. Query metrics and tear down
struct stream_metrics m = tile_stream_gpu_get_metrics(s);
tile_stream_gpu_destroy(s);
```

The CPU backend (`tile_stream_cpu_create` / `tile_stream_cpu_writer`) follows
the same pattern — swap `gpu` for `cpu` in the function names. The GPU backend
uses CUDA streams + nvcomp; the CPU backend uses OpenMP + zstd/lz4.

Configure the pipeline via `struct tile_stream_configuration` (codec, chunk
dimensions, shard layout, LOD reduction method, etc.). See
[`src/stream.gpu.h`](src/stream.gpu.h) or
[`src/stream.cpu.h`](src/stream.cpu.h) for the full API.

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

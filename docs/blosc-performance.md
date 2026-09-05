# Blosc performance and GPU memory

Use this guide to select `codec_config.blosc_block_bytes`, shuffle mode, and
codec. The [binary specification][blosc-format] defines the stored format;
the [README][blosc-configuration] describes configuration.

## Starting points

Set the block size explicitly. `16 * 1024` is a useful initial GPU comparison
point. A block is internal to a Zarr chunk. Changing it does not require
changing the chunk or shard shape.

Our RTX 5070 Laptop measurements suggest these candidates:

| Workload and priority | Codec/filter | First block sizes to compare |
|---|---|---|
| Low-compressibility input; throughput | Blosc-LZ4 | 4–8 KiB, with a memory check |
| Low-compressibility input | Blosc-Zstd | 8–32 KiB; 16 KiB was fastest |
| Highly repetitive input | Blosc-Zstd + bitshuffle | 32–64 KiB for speed; larger for ratio |
| Highly repetitive input | Blosc-LZ4 + bitshuffle | 256 KiB–1 MiB, capped by chunk length |

These are based on measurements using two synthetic fills, not actual microscopy
images. Hold the input, element type, chunk shape, batch geometry, and sink
fixed when comparing settings. Test representative images before adopting a
production policy.

For example, an explicit Zstd/bitshuffle comparison point is:

```c
config.codec = (struct codec_config){
    .id = CODEC_BLOSC_ZSTD,
    .level = 3,
    .shuffle = CODEC_SHUFFLE_BIT,
    .blosc_block_bytes = 64 * 1024,
};
```

On GPU, levels 1–9 have no effect so sweeping those levels adds no
compression-mode coverage. Level 0 is store-only. CPU C-Blosc does honor levels
and may adjust the actual block size or split blocks. A CPU tuning search over
64/128/256 KiB is a reasonable separate starting experiment, not a conclusion
from the GPU measurements below. Blosc's [CPU tuning report][cpu-tuning-report]
discusses the different cache/ratio tradeoff.

## Throughput–compression Pareto frontiers

A setting dominates another when its median throughput and compression fold
are both at least as high and one is strictly higher. A frontier contains
settings not dominated by another candidate in the comparison group. Here each
group fixes GPU, input, chunk geometry, and codec, and searches all block sizes
and all three shuffle choices. This is not the same as computing a separate
frontier for each shuffle mode.

![Blosc throughput and compression frontiers][pareto-plot]

Blue is LZ4 and orange is Zstd. Higher and farther right is better. Labels are
block sizes; all per-codec frontier points in this dataset use bitshuffle.
Faint points are dominated candidates. Hollow diamonds are raw-codec controls,
excluded from the Blosc frontier. The random-input panels zoom the ratio axis;
the XOR panels use a logarithmic ratio axis. Lines only connect tested points;
they do not predict intermediate settings.

[All candidates, without the random-input zoom][pareto-all-candidates]
and the [frontier table][pareto-frontier]
are available. The table also identifies nondominated points when both codecs
compete, rather than retaining a separate frontier for each codec.

Representative 1 MiB-chunk results, with bitshuffle throughout:

| Input | Codec | Block | Input GiB/s | Compression fold | Device GiB |
|---|---|---:|---:|---:|---:|
| Random | LZ4 | 4 KiB | 4.769 | 1.47469 | 4.766 |
| Random | LZ4 | 16 KiB | 3.633 | 1.48181 | 3.076 |
| Random | Zstd | 16 KiB | 1.107 | 1.48904 | 3.245 |
| XOR | LZ4 | 1 MiB | 5.284 | 131.02 | 2.523 |
| XOR | Zstd | 64 KiB | 5.167 | 125.30 | 3.349 |
| XOR | Zstd | 1 MiB | 3.692 | 360.65 | 3.349 |

On random input, small-block LZ4 buys substantial throughput for little ratio
loss but costs memory. On repetitive input, larger blocks can improve both
speed and ratio; block-size effects are not monotonic across codecs or filters.
The XOR example also shows why neither codec universally wins: LZ4 at 1 MiB
beats Zstd at 64 KiB in both objectives, but larger-block Zstd reaches ratios
LZ4 does not.

For a storage-limited pipeline, also measure encoded bytes and sustained sink
throughput. A faster discard-sink configuration can be slower end to end if
its output is larger. This sweep did not measure decompression throughput.

### Measurement scope

The retained run is `orca2_single` on an NVIDIA GeForce RTX 5070 Laptop GPU,
driver 595.99.02, CUDA 13.2.51, nvCOMP 5.3.0.16, Release build. It used u16 XOR
and deterministic uniform 12-bit random input, 100 frames, a discard sink,
`max_threads = 3`, a 6 GiB device budget, and a 64 MiB target batch size.

The nine block sizes were 4/8/16/32/64/128/256/512/1024 KiB, omitting requests
larger than a chunk. Each configuration had one warmup and three measured
runs; each pass randomized the matrix with seed 169. All 800 runs completed,
including 600 measured runs. Each block-size build passed GPU Blosc correctness
tests before measurement.

The actual minimum batch was one epoch, larger than the 64 MiB target:

| Chunk size | Shape `(t,c,y,x)` | Chunks per epoch | Padded input per batch |
|---|---|---:|---:|
| 256 KiB | `(8,1,128,128)` | 576 | 144 MiB |
| 1 MiB | `(16,1,128,256)` | 288 | 288 MiB |

Each run ingested 1.758 GiB. Compression fold is padded chunk input bytes
divided by encoded bytes, not original ingested bytes divided by file size.
Compare folds within a chunk geometry. Throughput is original input GiB per
second through the pipeline, not an isolated nvCOMP kernel rate.

Throughput bars show the minimum and maximum of three runs, not confidence
intervals. The median relative span `(max-min)/median` across configurations
was 2.1%, with a maximum of 19.1%. Exact-median frontier membership is therefore
descriptive; very close rankings should not be treated as decisive. Compression
fold was identical across repetitions.

## How block size affects GPU memory

Smaller blocks do **not** necessarily use less memory. At a fixed chunk shape
and batch size, reducing block size increases the number of independent
compression inputs. Metadata and compressor workspace can grow even while
each individual block shrinks.

![Blosc block size and stream GPU memory][memory-plot]

This plot fixes the XOR input and bitshuffle. Solid lines are measured device
memory deltas; dashed lines are the stream's explicit-allocation estimate.
It shows the whole stream, not just compression scratch.

### Memory terms

For `Q` chunk slots, `C` uncompressed bytes per chunk, and effective block size
`B = min(blosc_block_bytes, C)`, the codec reserves `P = Q * ceil(C/B)` block
slots. Its allocation accounting includes:

| Term | Block-size dependence |
|---|---|
| Final encoded-chunk pools | Two pools of `Q * codec_output_stride(codec, C)`; independent of `B` |
| Raw compressed-block scratch | `P` aligned worst-case codec-output slots; includes capacity for the shorter final block |
| Block sizes, offsets, and input/output pointers | Linear in `P`; on this 64-bit build, 40 bytes per block plus 8 bytes per chunk |
| nvCOMP workspace | Queried for `P` inputs, maximum input `B`, and total uncompressed bytes `Q*C`; codec/version dependent |
| Prepared input blocks | `P * align_up(B, input_alignment)` when the initial configuration enables shuffle, future shuffle storage is reserved, or internal block starts need alignment; one buffer serves all three purposes |

The estimator also includes the stream's staging, original chunk pools,
aggregation, and LOD allocations. These are separate from the codec-owned
scratch. Pinned host memory is reported separately from device memory.

The encoder owns alignment handling. Callers do not align append buffers or change
their data layout. Odd block sizes are valid, but can require extra scratch:
for 64 chunks of 1 MiB each, 4097-byte blocks require 64.0625 MiB of aligned
input scratch with this nvCOMP build. With no shuffle configured or reserved,
4096-byte blocks require none. Initial byte shuffle or bitshuffle always
allocates preparation storage, including for one-byte elements, even when the
reservation argument is disabled. The reservation argument adds capacity for
future shuffled bindings. A binding that needs unavailable storage is rejected
without changing the previous binding. Odd blocks use this same buffer for
alignment and filtering.

Copy, byte shuffle, and bitshuffle use one CUDA thread block per Blosc block,
sharing block addressing and incomplete-element tail handling. Filtering
writes directly to aligned block slots, and both nvCOMP and raw-block
fallback read the same prepared pointers. Aligned, unfiltered input bypasses
preparation, as does aligned byte shuffle of one-byte elements. Whole-frame
fallback always reads the original chunk.

Allocation sizes are computable from the configuration. Only the vendor's
workspace requirements and output bounds need nvCOMP sizing queries; these do
not require compression or allocating the codec buffers. There is no reason to
benchmark compression to produce a memory-versus-block-size table.

The [generated memory tables][memory-estimates]
use the estimator results already recorded by the throughput sweep, before
shuffle and alignment storage were unified. They remain historical results. The
generator checks that estimates agree across the two fills. For 1 MiB chunks,
288 MiB of padded input per batch, and bitshuffle:

| Block size | LZ4 device GiB | Zstd device GiB |
|---|---:|---:|
| 4 KiB | 4.754 | 3.082 |
| 16 KiB | 3.064 | 3.100 |
| 64 KiB | 2.642 | 3.202 |
| 256 KiB | 2.536 | 3.202 |
| 1 MiB | 2.510 | 3.202 |

These are whole-stream allocation estimates, not measured usage or codec-only
workspace. LZ4's workspace requirement falls with the block count; Zstd's grows
up to 64 KiB blocks and then plateaus in this build. The estimator asks the
installed library rather than hardcoding this version's workspace formula.

Higher compression ratios do not shrink these reserved buffers. Level 0
currently retains codec capacity even though it emits verbatim chunks. Likewise,
a shared multiarray codec retains capacity for its maximum geometry and any
required shuffle storage; the single-array estimate is not a whole-multiarray
memory total.

### Estimate accuracy and headroom

`tile_stream_gpu_memory_estimate()` reports explicitly requested allocations.
`codec_device_bytes()` and codec initialization use the same checked layout
calculation for every codec-owned allocation above; the stream adds its other
pools. Raw LZ4 and Zstd use the same batch setup with one block per chunk. Their
existing aligned output bounds are preserved; Blosc retains a logical frame
bound of `C + 16`, separate from the aligned output slot stride. Allocation
validation required no additional block-dependent accounting term.

`test_compress_blosc_gpu` now compares the codec estimate against the sum of
CUDA's actual allocation-range sizes, rather than only another copy of the
sizing formula. It covers both codecs, seven block sizes including an odd size
and a size larger than the chunk, all three filters independently of shuffle
reservation on/off, and levels 0/3: 168 configurations. The byte totals matched
exactly on this GPU, including initial filters with reservation disabled.

That does not make the estimate a prediction of all device memory consumed
by a process. CUDA allocation granularity, context/runtime/library residency,
and unrelated GPU users affect free-memory readings. In the original 1 MiB
chunk sweep, measured-minus-estimated stream usage was about 11–14 MiB for LZ4
and 148–150 MiB for Zstd. These differences do not identify a particular CUDA
module or prove that all overhead is retained throughout a run.

The regular benchmark now records `memory_device_overhead_bytes`: signed
device-memory delta minus the estimated device allocations, alongside the
existing estimate and measurement. It is null for CPU runs or unavailable
readings/estimates. Negative values are preserved. The measurement compares
`cuMemGetInfo` free bytes before stream creation with after the run, while the
stream is still alive. It is not a sampled peak, and the initial CUDA context
cost predates the baseline. The historical sweep used the same measurement
boundary; its overhead can be compared from the two recorded quantities, but
its results are not relabeled as containing the new field.

For budgeting, use the estimate for capacity planning and leave measured
headroom for runtime overhead and other users. Do not feed all currently free
VRAM to the layout advisor or assume a universal percentage covers cold-start
costs. Compare cold and warmed runs on each target GPU/nvCOMP combination;
record allocation deltas separately from a sampled high-water mark.

Run the allocation regression test with:

```sh
cmake --build build --target test_compress_blosc_gpu
ctest --test-dir build --output-on-failure -R '^test-test_compress_blosc_gpu$'
```

The test requires a usable GPU. Overhead observations belong to the normal
performance sweeps; run without competing GPU workloads for interpretable
free-memory readings. No separate memory benchmark is needed.

### Memory-aware frontiers

The throughput/ratio chart is only a two-objective frontier. Under a fixed
VRAM budget, first discard settings that do not fit, then recompute the
frontier. Alternatively, maximize throughput and compression while minimizing
memory simultaneously. The retained
[three-objective frontier][pareto-memory-frontier]
does this across both codecs using measured device deltas, separately for each
input and chunk geometry. It remains subject to the measurement limitations
above; tiny memory differences can create additional nondominated points.

## Incorporating Blosc into the regular sweeps

The recommended integration is a small regression matrix plus a separate
tuning tier. This is a proposal, not a claim that the runner already implements
the following tiers or repetition controls.

### Routine regression coverage

- Run GPU Blosc in the existing `compress`/`backend`, `fill`, and selected LOD
  cases. Remove the obsolete CPU-only classification when implementing this.
- Use a fixed, explicit 16 KiB block size, level 3, and bitshuffle for both
  Blosc codecs as a stable comparison configuration, not as a universal optimum.
  Keep raw LZ4/Zstd and no-compression controls.
- Exercise one small-block LZ4 case (4 KiB), one 64 KiB Zstd case, and one
  chunk-sized bitshuffled LZ4 case on a representative single-scale scenario.
  Keep no/byte/bit shuffle correctness coverage in tests; do not multiply every
  filesystem/S3/LOD case by the full tuning matrix.
- Compare CPU and GPU at identical explicit settings for parity, and label
  separately tuned backend configurations as tuned comparisons. Old CPU results
  with unknown block/shuffle settings are not equivalent baselines.

### Dedicated, opt-in Blosc tuning tier

Use `orca2_single`, u16, XOR and random, 256 KiB and 1 MiB chunks, all three
shuffles, and blocks 4/8/16/32/64/128/256/512/1024 KiB, excluding sizes larger
than the chunk. This reproduces 192 Blosc configurations plus eight raw-codec
controls. One warmup and three measured repetitions yield 800 executions.
Use a fixed seed to randomize each pass and preserve every repetition.

Keep this full matrix explicitly selected rather than silently adding it to
the routine `--all` set. A later real-image tier should use identified,
redistributable datasets with checksums and controlled input staging. Zeros
are useful for extremes and correctness, not for choosing an imaging default.

Freeze the actual chunk shape and epochs per batch during tuning. If a budget
cannot support them, record a capacity-limited result instead of silently
shrinking the layout. Run settings sequentially on the GPU. After selecting
speed-, ratio-, and memory-oriented candidates, validate those few against
the real filesystem or S3 sink.

### Runner and report status

The current matrices still schedule Blosc only on CPU and request 16 KiB
blocks. The following support is already available:

- `RunSpec` requires explicit `blosc_block_bytes` for Blosc runs; zero is
  invalid. The benchmark command and result metadata include the request,
  including error and timeout results.
- Run identities and resume checks distinguish block sizes. Archived
  unrecorded sizes remain unknown, and existing raw-codec identities and
  archived results are preserved.
- Both report pages offer block-request selectors and distinguish unknown
  settings from explicit sizes in comparisons.
- GPU-independent regression tests cover block-size validation, run identities,
  deduplication, resume, CLI propagation, matrix counts, report serialization,
  and block-aware filtering.

Remaining work for the expanded matrices:

1. Add explicit shuffle and level fields to Blosc run specs, commands, and
   results, including failures. Extend identities, resume checks, and comparison
   labels to include them. This branch's benchmark parser does not yet expose
   shuffle/level CLI controls; port/reuse the existing Blosc benchmark work.
   Validate returned settings against requests.
2. Add the routine GPU Blosc cases and opt-in tuning tier described above, with
   warmups, seeded run order, and measured repetitions. Keep each repetition
   distinct from its configuration identity.
3. Complete tuning metadata with actual chunk geometry, batch size, device
   budget, output bytes, driver/nvCOMP/build provenance, and capacity failures.
   Include codec workspace alongside the existing allocation estimates and
   measured device deltas; label runtime peak measurements separately when added.
4. Extend the regression tests to cover the new settings, failure metadata,
   repetitions, and expanded matrix counts. Keep validation and dry-run tests
   runnable without a GPU.
5. Add shuffle/level report filters and summaries showing median plus spread.
   Compute frontiers only within matching input/geometry/backend/sink groups;
   never pool CPU/GPU, synthetic fills, or differently padded chunk layouts.

The [sweep documentation][sweep-documentation] describes the current
runner and report behavior.

## Retained data and plot reproduction

The [artifact directory][benchmark-artifacts] includes
all 200 configuration summaries, provenance, estimator-derived memory tables,
and the plot generator. No new throughput sweep was needed to write this guide.

```sh
python3 docs/benchmarks/blosc-rtx5070-20260905/plot.py
```

This regenerates the memory tables, SVGs, two- and three-objective frontier CSVs, and a
self-contained HTML view using only Python's standard library. The standalone
run predates the explicit block-size API and used validated binaries with
different internal block-size constants; it must not be relabeled as a
measurement of every subsequent implementation change.

[blosc-format]: blosc-format.md
[blosc-configuration]: ../README.md#blosc-configuration
[cpu-tuning-report]: https://blosc.org/posts/beast-release/
[pareto-plot]: benchmarks/blosc-rtx5070-20260905/pareto.svg
[pareto-all-candidates]: benchmarks/blosc-rtx5070-20260905/pareto-all-candidates.svg
[pareto-frontier]: benchmarks/blosc-rtx5070-20260905/pareto-frontier.csv
[memory-plot]: benchmarks/blosc-rtx5070-20260905/memory.svg
[memory-estimates]: benchmarks/blosc-rtx5070-20260905/memory-estimates.md
[pareto-memory-frontier]: benchmarks/blosc-rtx5070-20260905/pareto-memory-frontier.csv
[sweep-documentation]: ../scripts/sweep/README.md
[benchmark-artifacts]: benchmarks/blosc-rtx5070-20260905/README.md

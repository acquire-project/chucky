# Blosc performance and GPU memory

Use this guide to select `codec_config.blosc_block_bytes`, shuffle mode, and
codec. The [binary specification][blosc-format] defines the stored format;
the [README][blosc-configuration] describes configuration.

## Starting points

Retune for the target GPU. The L40 measurements below have different frontiers
from the archived RTX 5070 Laptop run. Set the block size explicitly;
`16 * 1024` remains a useful fixed regression comparison. A block is internal
to a Zarr chunk, so changing it does not require
changing the chunk or shard shape.

The full L40 sweep suggests these candidates:

| Workload and priority | Codec/filter | First block sizes to compare on L40 |
|---|---|---|
| Low-compressibility input; throughput | Blosc-LZ4 + bitshuffle | 4–8 KiB, with a memory check; also compare 128–256 KiB |
| Low-compressibility input; Zstd speed/ratio | Blosc-Zstd + bitshuffle | Chunk-sized: 256 KiB or 1 MiB here; retain 16 KiB as a lower-memory comparison |
| Highly repetitive input; throughput | Blosc-LZ4 + byte shuffle | 4/8/16/32 KiB, with a memory check |
| Highly repetitive input; LZ4 ratio | Blosc-LZ4 + bitshuffle | Chunk-sized: 256 KiB or 1 MiB here |
| Highly repetitive input; Zstd speed/ratio | Blosc-Zstd + bitshuffle | 16–64 KiB for speed; larger blocks for ratio |

These are two synthetic fills, not actual microscopy images. Hold input,
element type, chunk shape, batch geometry, and sink fixed when comparing
settings. Test representative images before adopting a production policy.

For example, an explicit Zstd/bitshuffle comparison point is:

```c
config.codec = (struct codec_config){
    .id = CODEC_BLOSC_ZSTD,
    .level = 3,
    .shuffle = CODEC_SHUFFLE_BIT,
    .blosc_block_bytes = 64 * 1024,
};
```

For GPU Blosc, levels 1–9 have no effect so sweeping those levels adds no
compression-mode coverage. Level 0 is store-only. CPU C-Blosc does honor levels
and may adjust the actual block size or split blocks. A CPU tuning search over
64/128/256 KiB is a reasonable separate starting experiment, not a conclusion
from the GPU measurements below. Blosc's [CPU tuning report][cpu-tuning-report]
discusses the different cache/ratio tradeoff.

## Throughput–compression Pareto frontiers

A setting dominates another when its median throughput and compression fold
are both at least as high and one is strictly higher. A frontier contains
settings not dominated by another candidate in the comparison group. Each
group fixes GPU, input, chunk geometry, and codec, and searches all block sizes
and all three shuffle choices. Separate frontiers for each shuffle would keep
settings that another filter dominates.

![L40 Blosc throughput and compression frontiers][pareto-plot]

Blue is LZ4 and orange is Zstd. Higher and farther right is better. Labels are
block sizes with units (for example, 16 KiB or 1 MiB). The graphical legend
identifies shuffle modes: squares for none, circles for byte, and triangles
for bitshuffle. The L40 XOR frontiers include
byte-shuffled LZ4 points; restricting the search to bitshuffle would miss them.
Faint points are dominated candidates. Hollow diamonds are raw-codec controls,
excluded from the Blosc frontier. Random-input panels zoom the ratio axis;
XOR panels use a logarithmic ratio axis. Lines connect only tested points;
they do not predict intermediate settings.

[All candidates, without the random-input zoom][pareto-all-candidates], the
[frontier table][pareto-frontier], and a [filterable local view][pareto-html]
are available. The table also marks nondominated points when both Blosc codecs
compete, rather than retaining a separate frontier for each codec.

Representative L40 results with 1 MiB chunks:

| Input | Codec | Shuffle | Block | Input GiB/s | Compression fold | Observed device GiB |
|---|---|---|---:|---:|---:|---:|
| Random | LZ4 | bit | 4 KiB | 11.176 | 1.47469 | 4.764 |
| Random | LZ4 | bit | 256 KiB | 7.175 | 1.48431 | 2.547 |
| Random | Zstd | bit | 16 KiB | 2.181 | 1.48904 | 3.767 |
| Random | Zstd | bit | 1 MiB | 2.979 | 1.49292 | 4.038 |
| XOR | LZ4 | byte | 4 KiB | 10.785 | 2.44739 | 4.764 |
| XOR | LZ4 | byte | 32 KiB | 10.458 | 3.98582 | 2.793 |
| XOR | LZ4 | bit | 1 MiB | 7.876 | 131.02 | 2.521 |
| XOR | Zstd | bit | 64 KiB | 6.334 | 125.30 | 4.038 |
| XOR | Zstd | bit | 1 MiB | 4.718 | 360.65 | 4.038 |

On random input, small-block LZ4 buys throughput for little ratio loss but
costs memory. Chunk-sized Zstd improves both speed and ratio over 16 KiB on
this L40, while using more device memory. On XOR input, byte-shuffled LZ4 is
worth comparing for speed, chunk-sized bitshuffled LZ4 offers much higher
compression, and large-block bitshuffled Zstd reaches ratios LZ4 does not.
Block-size effects are not monotonic across codecs or filters.

For a storage-limited pipeline, also measure encoded bytes and sustained sink
throughput. A faster discard-sink configuration can be slower end to end if
its output is larger. This sweep did not measure decompression throughput.

### Comparison with the RTX 5070 Laptop

The exact-median frontiers changed in **all eight** input/chunk/codec
groups. [Paired measurements and frontier identities][frontier-comparison]
and the [comparison figure][comparison-plot] retain all 200 matched settings.
Compression folds matched the archived values at the recorded precision for
every configuration. XOR/Zstd timing ranges overlap substantially, so its
precise block ranking remains uncertain despite different median frontiers.

Two useful changes appear with 1 MiB chunks:

| Input and setting | RTX 5070 Laptop GiB/s | L40 GiB/s | Compression fold |
|---|---:|---:|---:|
| Random, Zstd + bitshuffle, 16 KiB | 1.107 | 2.181 | 1.48904 |
| Random, Zstd + bitshuffle, 1 MiB | 0.981 | 2.979 | 1.49292 |
| XOR, LZ4 + byte shuffle, 4 KiB | 4.392 | 10.785 | 2.44739 |
| XOR, LZ4 + bitshuffle, 1 MiB | 5.284 | 7.876 | 131.02 |

The random-input Zstd ordering reverses: on L40, the 1 MiB block is
37% faster than 16 KiB and compresses slightly better. For XOR, the laptop's
chunk-sized bitshuffled LZ4 point dominates the smaller byte-shuffled setting;
on L40 the smaller setting trades ratio for greater speed. Memory remains a
separate objective: a two-objective winner can require more VRAM.

The observed min–max throughput ranges for both reversals are disjoint
within each setup. This is an observation about these repetitions, not a
confidence interval.

These runs compare different measured setups. The older run used source
`c519a05`, internal block-size constants, CUDA 13.2.51, Clang 21.1.8, and driver
595.99.02. The L40 run uses PR264's explicit API, CUDA 13.1.115, GCC 14.3.0, and
driver 580.126.20; both use nvCOMP 5.3.0.16. Source, toolchain, driver, host, and
GPU differ, so this does not isolate the effect of changing only the GPU.

### Measurement scope

The retained L40 run is `orca2_single` at PR264 commit `2c13122`, measured on
2026-09-06 in Release mode with native sm_89 code. An archived benchmark-only
patch adds shuffle/level controls and metadata; the compression implementation
is unchanged from that revision. It used u16 XOR and deterministic uniform
12-bit random input, 100 frames, a discard sink, three worker threads, a 6 GiB
device allocation budget, and a 64 MiB target batch size. A prefilled 64 MiB
append buffer is reused, matching the earlier benchmark's default. This is a
controlled repeated input, not 100 independent random frames.

The nine block sizes were 4/8/16/32/64/128/256/512/1024 KiB, omitting requests
larger than a chunk. The 192 Blosc configurations cover both codecs, all three
shuffles, and level 3; eight raw LZ4/Zstd controls bring the total to 200.
One complete warmup pass precedes five measured passes. Each pass is shuffled
using RNG seed 169, and each execution uses a fresh process. All 1,200 runs
completed, including 1,000 measurements. The two CPU checks, three GPU
correctness tests, and four benchmark CLI checks passed before measurement.

The actual minimum batch was one epoch, larger than the 64 MiB target:

| Chunk size | Shape `(t,c,y,x)` | Chunks per epoch | Padded input per batch |
|---|---|---:|---:|
| 256 KiB | `(8,1,128,128)` | 576 | 144 MiB |
| 1 MiB | `(16,1,128,256)` | 288 | 288 MiB |

Every execution was checked against these shapes and batch sizes. Each run
ingested 1.758 GiB. Compression fold is padded chunk input bytes divided by
encoded bytes, not original ingested bytes divided by file size. Compare folds
within a chunk geometry. Throughput covers append-buffer allocation/prefill,
appends, and flush; stream creation is timed separately. It is a pipeline rate,
not an isolated nvCOMP kernel rate.

Throughput bars show the minimum and maximum of five runs, not confidence
intervals. The median relative span `(max-min)/median` across configurations
was 5.8%, with a maximum of 50.7%. Exact-median frontier membership is
descriptive; close rankings should not be treated as decisive. Compression
fold was identical across measured repetitions. The widest span was XOR with
256 KiB chunks, Zstd, bitshuffle, and 16 KiB blocks; 16–64 KiB XOR/Zstd choices
need further measurements to settle close rankings. The separate 5070 Laptop
archive retains its original three-repeat method and results.

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

The [generated memory tables][memory-estimates] use the allocation estimates
recorded by this L40 sweep. The generator checks that estimates agree across
both fills before combining them; no additional compression run is needed.
For 1 MiB chunks, 288 MiB of padded input per batch, and bitshuffle:

| Block size | LZ4 device GiB | Zstd device GiB |
|---|---:|---:|
| 4 KiB | 4.754 | 3.270 |
| 16 KiB | 3.064 | 3.331 |
| 64 KiB | 2.642 | 3.603 |
| 256 KiB | 2.536 | 3.603 |
| 1 MiB | 2.510 | 3.602 |

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

`test_compress_blosc_gpu` compares the codec estimate against the sum of
CUDA's actual allocation-range sizes, rather than only another copy of the
sizing formula. It covers both codecs, seven block sizes including an odd size
and a size larger than the chunk, all three filters independently of shuffle
reservation on/off, and levels 0/3: 168 configurations. The byte totals matched
exactly on the L40 before the sweep, including initial filters with reservation
disabled.

That does not make the estimate a prediction of all device memory consumed
by a process. CUDA allocation granularity, context/runtime/library residency,
and unrelated GPU users affect free-memory readings. Across the 1 MiB-chunk,
bitshuffled configurations, median observed-minus-estimated usage on L40 was
9–12 MiB for LZ4 and 444–446 MiB for Zstd. The archived laptop medians were
11–14 MiB and 148–150 MiB respectively. These differences do not identify a
particular CUDA module or prove that all overhead is retained throughout a run.

The regular benchmark now records `memory_device_overhead_bytes`: signed
device-memory delta minus the estimated device allocations, alongside the
existing estimate and measurement. It is null for CPU runs or unavailable
readings/estimates. Negative values are preserved. The measurement compares
`cuMemGetInfo` free bytes before stream creation with after the run, while the
stream is still alive. It is not a sampled peak, and the initial CUDA context
cost predates the baseline. The L40 records include the residual field. The
5070 Laptop archive used the same measurement boundary; its residual can be
computed from the two recorded quantities, but its records are not relabeled
as containing the newer field.

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
controls. One warmup and five measured repetitions yield 1,200 executions.
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

The current matrices still schedule Blosc only on CPU, with level 3, no shuffle,
and an explicit 16 KiB block request. This is a scheduling limitation; the GPU
backend supports both Blosc codecs. The following support is already available:

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
   shuffle/level CLI controls; port/reuse the existing Blosc benchmark work
   retained in the [L40 collection patch][benchmark-controls].
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

The [L40 artifact directory][benchmark-artifacts] retains all 200 summaries,
all 1,200 individual executions including warmups, compressed full JSON,
source/build/GPU provenance, the benchmark-only CLI patch, validation logs,
and the collection and plot scripts. The [5070 Laptop archive][historical-artifacts]
is preserved separately.

```sh
uv run --python 3.12 docs/benchmarks/blosc-l40-20260906/plot.py
```

The script pins Matplotlib and requires Python 3.11 or later. It validates
records and collection-input hashes, then regenerates memory tables, two- and
three-objective frontiers, matched historical comparisons, SVG/PNG/PDF figures,
and a self-contained HTML view. Add `--check` to validate without rewriting
outputs. Neither operation needs a GPU or new throughput measurements. The
artifact README gives the exact collection method and reproduction limits.

[blosc-format]: blosc-format.md
[blosc-configuration]: ../README.md#blosc-configuration
[cpu-tuning-report]: https://blosc.org/posts/beast-release/
[pareto-plot]: benchmarks/blosc-l40-20260906/pareto.svg
[pareto-all-candidates]: benchmarks/blosc-l40-20260906/pareto-all-candidates.svg
[pareto-frontier]: benchmarks/blosc-l40-20260906/pareto-frontier.csv
[memory-plot]: benchmarks/blosc-l40-20260906/memory.svg
[memory-estimates]: benchmarks/blosc-l40-20260906/memory-estimates.md
[pareto-memory-frontier]: benchmarks/blosc-l40-20260906/pareto-memory-frontier.csv
[sweep-documentation]: ../scripts/sweep/README.md
[benchmark-artifacts]: benchmarks/blosc-l40-20260906/README.md
[pareto-html]: benchmarks/blosc-l40-20260906/pareto.html
[frontier-comparison]: benchmarks/blosc-l40-20260906/comparison.md
[comparison-plot]: benchmarks/blosc-l40-20260906/comparison.svg
[historical-artifacts]: benchmarks/blosc-rtx5070-20260905/README.md
[benchmark-controls]: benchmarks/blosc-l40-20260906/benchmark-controls.patch

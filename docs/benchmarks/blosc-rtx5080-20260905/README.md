# RTX 5080 Blosc Pareto analysis

This experiment compares this Windows desktop with the RTX 5070 Laptop
measurements retained in [blosc-performance.md](../../blosc-performance.md).
The artifact date is September 5 in America/Los_Angeles; the run timestamps
are September 6 UTC. Results characterize this machine, software, and desktop
session. They are empirical frontiers over the tested matrix, not guaranteed
hardware throughput limits.

[Unified Blosc Pareto site and build instructions](../README.md) ·
[All measurements](summary.csv) · [Matching-configuration comparison](comparison.csv)

## Results and changes from the guide

All 800 executions passed: 200 warmups and 600 measured runs. All 200
configuration folds match the historical CSV at its reported precision and
remain constant across repetitions. The median relative throughput span
`(max-min)/median` is 7.15%, versus 2.1% historically; the maximum is 26.0%,
versus 19.1%. The three measured passes were about 79–82% as fast as their
matching warmups in median. Warmups are excluded from every result below.
This systematic difference reinforces the need to compare repeated measured
runs rather than isolated early timings.

Representative **1 MiB-chunk** comparisons, all with bitshuffle:

| Input | Codec | Block | RTX 5080 GiB/s | 5070 Laptop GiB/s | Change | Fold on both |
|---|---|---:|---:|---:|---:|---:|
| Random | LZ4 | 4 KiB | 6.490 | 4.769 | +36.1% | 1.47469 |
| Random | LZ4 | 16 KiB | 3.503 | 3.633 | -3.6% | 1.48181 |
| Random | Zstd | 16 KiB | 1.556 | 1.107 | +40.6% | 1.48904 |
| Random | Zstd | 1 MiB | 1.792 | 0.981 | +82.6% | 1.49292 |
| XOR | LZ4 | 1 MiB | 5.173 | 5.284 | -2.1% | 131.022 |
| XOR | Zstd | 64 KiB | 6.181 | 5.167 | +19.6% | 125.296 |
| XOR | Zstd | 256 KiB | 5.408 | 3.974 | +36.1% | 190.301 |
| XOR | Zstd | 1 MiB | 4.331 | 3.692 | +17.3% | 360.654 |

The small negative differences have overlapping repetition ranges and do not
establish a regression. The 5080 does not provide a uniform multiplier across
settings.

The **random / 1 MiB-chunk cross-codec frontier** is especially simple:

| Codec/filter | Block | Input GiB/s, median [min–max] | Fold | Explicit device GiB |
|---|---:|---:|---:|---:|
| LZ4 / bit | 4 KiB | 6.490 [6.184–6.490] | 1.47469 | 4.754 |
| LZ4 / bit | 8 KiB | 5.978 [5.808–6.258] | 1.47931 | 3.628 |
| LZ4 / bit | 512 KiB | 3.753 [3.627–3.825] | 1.48431 | 2.519 |
| Zstd / bit | 1 MiB | 1.792 [1.792–1.818] | 1.49292 | 3.529 |

Small-block LZ4 remains the throughput choice. Moving from 4 to 8 KiB saves
1.127 GiB of allocations, improves fold slightly, and reduces median speed
by 7.9%; their timing ranges overlap. The guide's random-input Zstd speed
recommendation changes: 1 MiB blocks dominate its smaller blocks here,
including 16 KiB: 1 MiB is 15.2% faster and compresses better. For 256 KiB chunks, the
random-input Zstd frontier similarly reduces to the chunk-sized 256 KiB
setting, though its speed differences are less separated from noise.

For **XOR / 1 MiB chunks**, Zstd/bitshuffle at 32 KiB delivers 6.490 GiB/s
and 106.653×; 64 KiB gives 6.181 GiB/s and 125.296×. Their ranges overlap.
At 256 KiB, Zstd reaches 190.301× and 5.408 GiB/s, dominating chunk-sized
LZ4's median speed and fold. Those two speeds also have overlapping ranges,
but the ratio difference is substantial. Chunk-sized LZ4 remains useful
under a memory limit: it allocates 2.510 GiB versus Zstd's 3.529 GiB.
Zstd at 1 MiB extends the frontier to 360.654× at 4.331 GiB/s.

The historical **all-bitshuffle frontier does not generalize**. On XOR with
256 KiB chunks, LZ4/byte shuffle at 16 KiB provides 6.477 GiB/s, 3.63556×,
and 1.657 GiB of allocations. It is a useful speed/memory alternative to
LZ4/bitshuffle at 4 KiB: 6.580 GiB/s, 2.50416×, and 2.502 GiB. On XOR with
1 MiB chunks, LZ4/byte shuffle at 4 KiB narrowly enters the speed/ratio
frontier at 6.517 GiB/s and 2.44739×. Its speed advantage over Zstd/bitshuffle
32 KiB is only 0.4%, well inside their ranges, with far worse compression
and more allocated memory; that frontier endpoint is not a compelling
practical choice.

For **memory**, the LZ4 estimates match the historical values for the tested
configurations. Zstd's installed-library estimates are higher even though
the nvCOMP version string is the same. At 1 MiB chunks with bitshuffle:

| Block | LZ4 allocation GiB, both | Zstd allocation GiB, 5080 | Zstd allocation GiB, historical |
|---|---:|---:|---:|
| 4 KiB | 4.754 | 3.235 | 3.082 |
| 16 KiB | 3.064 | 3.288 | 3.100 |
| 64 KiB | 2.642 | 3.529 | 3.202 |
| 256 KiB | 2.536 | 3.529 | 3.202 |
| 1 MiB | 2.510 | 3.529 | 3.202 |

Observed Zstd usage at 64 KiB–1 MiB is 3.791 GiB, versus about 3.349 GiB
historically. Observed-minus-estimated Zstd overhead is about 267–269 MiB
for these 1 MiB-chunk configurations, versus 148–150 MiB in the guide.
These are stream-lifetime allocation deltas, not sampled process peaks.
The experiment does not isolate whether GPU-dependent library sizing,
platform, toolchain, or implementation changes explain the estimate change.
Use the current estimator and leave headroom; the historical Zstd totals
are insufficient for budgeting this machine.

For a **storage-limited random / 1 MiB** pipeline, the optimistic model below
chooses Zstd/bitshuffle 1 MiB at a 1 GiB/s sink (about 1.333 GiB/s input),
LZ4/bitshuffle 512 KiB at 2 GiB/s (about 2.651 GiB/s input), and LZ4/bitshuffle
8 KiB at 4 GiB/s (about 5.283 GiB/s input). These are modeled choices;
the discard sweep did not validate a filesystem or S3 sink.

## Measurement and comparison scope

| Property | This run | Historical run |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16 GiB class | RTX 5070 Laptop |
| CPU | AMD Ryzen 5 7600X, 6 cores / 12 logical processors | Not retained |
| Platform | Windows 11 Pro, WDDM, active desktop | Linux |
| Host memory | 63.10 GiB visible to Windows | Not retained |
| NVIDIA driver | 591.86 | 595.99.02 |
| CUDA compiler | 13.3.73 | 13.2.51 |
| nvCOMP | 5.3.0.16 | 5.3.0.16 |
| Build | Release, MSVC 14.51, CUDA architecture 120 | Release, Clang 21.1.8, architectures 89/100 |
| Source | `19cd6d3` plus recorded benchmark CLI/report patch | `c519a05`, nine block-constant variants |

Both use `orca2_single`, u16, deterministic 12-bit random and XOR fills,
100 frames (1,887,436,800 ingested bytes), discard sink, three worker threads,
6 GiB device-allocation budget, and a 64 MiB target batch. The actual batch is
one epoch, which exceeds that target:

| Chunk bytes | Shape `(t,c,y,x)` | Chunks/batch | Padded batch | Epochs/run | Padded / ingested input |
|---|---|---:|---:|---:|---:|
| 256 KiB | `(8,1,128,128)` | 576 | 144 MiB | 13 | 1.04 |
| 1 MiB | `(16,1,128,256)` | 288 | 288 MiB | 7 | 1.12 |

The matrix has 192 Blosc configurations: two fills, two chunk geometries,
two codecs, all three shuffles, and 4/8/16/32/64/128/256/512/1024 KiB blocks
where a block fits within the chunk. Eight raw LZ4/Zstd controls complete the
200 configurations. Blosc level is 3; raw LZ4 uses 1 and raw Zstd uses 0.
Every configuration receives one warmup and three measured runs, in separate
randomized passes. The seed is 169; the Node runner's PRNG/order is not claimed
to reproduce the historical runner's exact order.

The runner validates returned shuffle, level, block size, chunk shape,
chunks/epoch, epochs/run, one-epoch batch, original and padded byte counts,
and thread count. It rejects layout changes instead of quietly combining
different geometries. `test-test_compress_blosc_gpu` passed before the sweep.
Reported compression fold divides padded input by bytes written to the discard
sink, including shard footers and their write alignment. Compressor payload
byte counts exclude those footer writes. The retained JSON does not expose
the exact sink byte total, so `sink_bytes_approx` and normalized sink ratios
retain the precision of the reported fold; payload byte counts are exact.
Each process has its own CUDA context; a prior warmup does not remove all
per-process cold-start effects. The reported throughput uses the benchmark's
pipeline wall interval and excludes its separately reported stream init time.

The desktop, browser, and a game process remained open. GPU telemetry is
retained in [provenance.json](provenance.json). No clocks were locked and no
other applications were stopped. Timings and device free-memory deltas can
therefore vary with other users of the machine. Source, compiler, OS, and
driver also differ from the historical run, so speedup is a comparison of
whole systems and software stacks, not an isolated GPU hardware speedup.

## What the Pareto bound means

For a fixed input and chunk geometry, a setting dominates another if its
median input throughput and padded compression fold are both at least as
large, with one strictly larger. The chart searches all filters together
and shows a frontier for each codec. `overall_frontier` in the CSV compares
both Blosc codecs together. Raw-codec controls are excluded from those
frontiers. Lines connect tested settings only.

For a required compression fold `R`, the measured speed envelope is
`max(T_i : fold_i >= R)` over the eligible tested configurations. Under an
allocation budget `M`, restrict eligibility to `estimate_i <= M` first.
The [budget tables](pareto-by-allocation-budget.csv) do this separately for
each input and geometry. Runtime headroom must be reserved in addition to
these explicit allocations.

The [three-objective table](pareto-memory-frontier.csv) records both versions
of the speed/ratio/memory frontier: one minimizing observed device deltas
and one minimizing explicit allocation estimates. Small observed differences
can create extra frontier members. Min–max bars describe the three runs;
they are not confidence intervals or proof that nearby settings differ.
Membership uses reported fold precision; tiny ratio differences can round
into ties.

## Random-input entropy reference

Independent uniformly random 12-bit samples stored in 16 bits have an ideal
expected compression fold of `16/12 = 1.33333` before padding and format
overhead. Multiplying by the padding factors above gives references of
1.38667 for 256 KiB chunks and 1.49333 for 1 MiB chunks. This explains why the
documented random-input ratios exceed 1.33333.

At 1 MiB chunks, the 1.49292 padded fold for 1 MiB Zstd/bitshuffle blocks is
about 1.33296 on ingested bytes, close to that reference. The 4 KiB
LZ4/bitshuffle setting gives 1.47469 padded, or about 1.31669 on ingested
bytes. Switching between those two saves only about 1.22% of encoded bytes.
The distinction matters when trading a large speed difference for a small
ratio gain. [Normalized ratios](random-entropy-reference.csv) use the recorded
fold and exact input/padding factors.

This is an entropy reference for independent samples, not a universal bound
on the deterministic test stream. The benchmark pre-fills and reuses a
64 MiB append buffer; these codecs operate on blocks of at most 1 MiB and
do not exploit repetition across that whole buffer. XOR is highly
structured and is not governed by the independent 12-bit reference.

## Storage bandwidth envelope

The measured sink discards output. For a sustained encoded-output bandwidth
`S`, an optimistic throughput model is
`min(T_i, S * ingested_bytes_i / encoded_bytes_i)`. Its two terms assume the
measured pipeline throughput remains achievable and storage overlaps fully.
This is a planning model, not a measured filesystem/S3 result.

Using the plotted fold directly would overestimate the storage benefit:
`ingested_bytes / encoded_bytes = fold / padding_factor` here. The
[sink-envelope table](sink-envelope.csv) selects the maximum modeled
throughput over both Blosc codecs at several sink rates. Actual storage
latency, contention, and decompression are outside this experiment.

## Reproduction and artifacts

The original summary and numeric analysis tables remain unchanged. The shared
[Blosc Pareto pipeline](../README.md) validates the raw records against these
summaries and supplies the interactive site. It replaces the standalone plots,
report page, and duplicate derivation script.

To collect new measurements, copy the runner into a new machine/date
directory at the same depth under `docs/benchmarks`, build the Release
`bench_stream_orca2_single` target, run the Blosc GPU correctness test, then
run `node <new-directory>/run.mjs`. The runner accepts an optional benchmark
executable path and refuses to overwrite existing raw results. It requires
`--blosc-shuffle` and returned `blosc_shuffle`/`blosc_level` metadata.

| Artifact | Contents |
|---|---|
| [raw-results.jsonl.gz](raw-results.jsonl.gz) | All 800 process commands, timestamps, stdout, stderr, and parsed results, including warmups |
| [provenance.json](provenance.json) | Hardware/build settings, executable/source hashes, exact benchmark patch, counts, GPU samples |
| [summary.csv](summary.csv) | All 200 configurations, medians, ranges, byte counts, stage metrics, memory, frontier flags |
| [comparison.csv](comparison.csv) | Matching current/historical settings, throughput changes, folds, and allocation changes |
| [memory-estimates.csv](memory-estimates.csv) | 96 geometry/codec/filter/block allocation estimates, checked to agree across fills |
| [pareto-frontier.csv](pareto-frontier.csv) | Per-codec frontier across filters, plus cross-codec membership |
| [pareto-memory-frontier.csv](pareto-memory-frontier.csv) | Three-objective membership using estimated or observed memory |
| [pareto-by-allocation-budget.csv](pareto-by-allocation-budget.csv) | Recomputed cross-codec frontiers at 1.5/2/2.5/3/4/6 GiB explicit-allocation limits |
| [analysis.json](analysis.json) | Aggregate checks, timing variability, cross-codec frontiers |
| [run.mjs](run.mjs) | Original dependency-free acquisition script |

Historical measurements remain in their original directory.

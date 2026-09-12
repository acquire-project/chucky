# Microscopy on L40 at 256 KiB

This 2026-09-12 run compares the six registered inputs from all five microscopy
datasets at one 256 KiB chunk target. All 60 configurations and 180 measured
executions passed. The [complete sweep](../../bench/results/reef-l40-432380c-20260912-microscopy-256k.json)
contains CPU and GPU controls, all raw repetitions, timing windows, source hashes,
chunk layouts, and memory and stage measurements. It loads in the standard
benchmark explorer; the public site receives it when these changes reach `main`.

GPU Blosc-LZ4 was the fastest compressed preset for OpenCell and BBBC010.
Raw GPU Zstd was fastest for JUMP-Scope, DynaCell, and COSEM. These rankings
apply to the cyclic replay described below.

The build is `432380c` in RelWithDebInfo mode, with CUDA 13.1.115,
SM 89 code, and nvCOMP 5.3.0.16. Slurm job 3660027 used one NVIDIA L40
(driver 580.126.20), eight allocated CPUs, and 32 GiB on
`cw-us-e4a2-l40-202-193`. The host has two Intel Xeon Platinum
8462Y+ processors; each benchmark used four workers. These rates measure the
whole pipeline, including host work and transfers.

Each execution used a preloaded, cyclic image source, a discard sink, a 64 MiB
batch target, at least 32 GiB of logical input, 0.25 seconds minimum warmup,
and one second minimum measured append time. Measurement includes final drain
and close and excludes initialization and warmup. Each configuration has three
fresh-process repetitions. Raw LZ4 uses level 1 and raw Zstd level 3. Both Blosc
presets use bitshuffle, level 3, and an explicit 16 KiB internal block request.

Each table cell gives **logical GiB/s / logical compression fold**. Logical bytes
count the original pixels and exclude padding; compression divides those bytes
by physical sink traffic. Throughput is the median of three executions;
compression uses their pooled byte counts. These are the tested presets at one
chunk size, without a search for optimal settings.

| Input | GPU LZ4 | GPU Zstd | GPU Blosc-LZ4 | GPU Blosc-Zstd |
|---|---:|---:|---:|---:|
| OpenCell DNA (uint16) | 2.38 / 1.02× | 2.28 / 1.33× | 6.20 / 1.28× | 2.83 / 1.49× |
| OpenCell protein (uint16) | 2.70 / 1× | 2.81 / 1.26× | 5.93 / 1.21× | 2.70 / 1.39× |
| BBBC010 brightfield (uint16) | 1.05 / 1.5× | 3.32 / 11.3× | 10.59 / 1.86× | 6.37 / 3.35× |
| JUMP-Scope fluorescence (uint16) | 2.73 / 1.01× | 7.73 / 5.12× | 5.72 / 1.27× | 3.01 / 1.5× |
| DynaCell phase (float32) | 3.14 / 0.996× | 9.63 / 4.29× | 4.34 / 1.11× | 2.30 / 1.15× |
| COSEM EM (uint8) | 3.14 / 0.996× | 9.38 / 4.9× | 3.81 / 1× | 2.01 / 1.02× |

The CPU controls also matter: on DynaCell, raw CPU LZ4 reached 5.03 GiB/s
versus 3.14 GiB/s on GPU, while both produced slight output expansion.

The explorer's primary throughput includes submitted padding; its microscopy
tooltip also gives the logical values used here. At this chunk target, spatial
padding adds 36.5% for OpenCell, 35.8% for BBBC010, and 4.9% for JUMP-Scope.
DynaCell and COSEM have no spatial padding. Use logical bytes when comparing
chunk sizes so extra padding does not appear to be extra useful work.

OpenCell contributes six source planes per pack. The other four samples contain
one plane each. All measured chunk shapes span four planes, with 64 KiB per
plane slice. The single-plane inputs therefore repeat the same slice four times
inside each chunk. Compression values describe that replay. A chunk-size study
must decide how temporal depth and repeated planes should vary, alongside
spatial padding and batch geometry.

The L40 allocation lasted 58.4 minutes including GPU checks
and report generation, or 0.974 GPU-hours. CPU-only build
and validation allocations used another 8.4 minutes with eight CPUs each.
Benchmark process time was distributed as follows.
Spread is `(maximum - minimum) / median` throughput across the three executions;
it is an observed range, not a confidence interval.

| Backend | Process minutes | Median spread | Maximum spread | Cases above 5% |
|---|---:|---:|---:|---:|
| GPU | 13.8 | 0.11% | 3.81% | 0/30 |
| CPU | 39.8 | 1.73% | 13.97% | 7/30 |

Accepted measurement windows ranged from 1.57 to 75.81 seconds; the shortest
append window was 1.57 seconds. Every execution passed coverage on its first
attempt.
All executions met the 32 GiB logical-input minimum. The one-second duration
setting was therefore not the main cost control in this run.

Duration and warmup have not been calibrated by this experiment. Before reducing
repetitions, compare smaller work budgets and warmup settings against longer
references at fixed geometry, using paired runs in both orders. The CPU spread
makes a single execution an especially weak basis for close comparisons. Use
this baseline to select a small calibration panel and contrasting images for
the later chunk-size study; keep its performance question and error tolerance
explicit before expanding the matrix.

Focused CPU and GPU correctness checks passed before data collection, including
independent image readback. CI passed for the tested source on CPU, macOS,
Windows, CUDA 12, and CUDA 13. The complete report built successfully with
35 sweep files and 8,431 runs.

Equivalent sweep command, inside an L40 allocation with a prepared build:

```sh
uv run scripts/sweep/sweep.py \
  --tier backend --scenario microscopy --chunk-bytes 256K \
  --repeats 3 --min-gib 32 --warmup 0.25 --duration 1 \
  --build-dir build-gpu --machine reef-l40 \
  --output bench/results/reef-l40-432380c-20260912-microscopy-256k.json
```

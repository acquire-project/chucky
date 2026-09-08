# Small GPU memcpy timing: implementation acceptance

Measured on Auk on 2026-09-08: AMD Ryzen AI 9 365, GeForce RTX 5070 Laptop GPU,
driver 595.99.02, CUDA 13.2, Clang 21.1.8, Release, CUDA architecture 120.
Parent is PR267 at `0752abd610d3232a6cc68a9e757cb1c392b877ab`, including PR266.
Candidate is the sampling implementation in this PR. No CPU algorithm,
copy implementation, context handling, scheduling, or payload buffering changed.

## Outcome

Default GPU timing samples one of every 64 copies below 8 KiB; larger copies
remain fully timed. See [the metric contract](../../memcpy-timing.md).

Completed physical output throughput, GiB/s; medians where two pairs were run:

| Append | Small epoch: parent → candidate | Change | Orca: parent → candidate | Change |
|---|---:|---:|---:|---:|
| 512 B | 4.414 → 7.203 | +63.2% | 4.177 → 7.060 | +69.0% |
| 2 KiB | 6.003 → 7.128 | +18.8% | 5.906 → 7.088 | +20.0% |
| 8 KiB | 9.579 → 9.555 | −0.26% | 8.691 → 8.663 | −0.32% |
| 64 MiB | 9.828 → 9.828 | ≈0% | 9.805 → 9.814 | +0.09% |

Both 512-byte pairs improved on each layout. Identical-parent bulk controls
varied by −0.028% and +0.018%; none of the candidate's four bulk pairs declined
materially. The 8 KiB checks are single pairs, not a precise crossover estimate.
Full memcpy profiling at 512 B reproduced parent throughput within 0.7%.

Across all 34 timed GPU runs (including controls), 4,117,098,441 append-body
observations had a maximum of 16.606 ms. External API timing around boundaries
peaked at 16.607 ms. No observed GPU append reached 100 ms. Final flush was
separate and peaked at 78.423 ms. This is evidence about these runs, not a
worst-case latency guarantee or a measurement of every API wrapper invocation.

The faster producer does expose more backpressure at saturation: Orca's
post-generation/post-batch waits at 512 B rose from roughly 0.01 ms to 1.2 ms;
at 2 KiB they reached 1.6 ms. Overall append maxima at those sizes stayed near
10 ms. At a common 3.5 GiB/s offered rate, the 512 B Orca pair sustained equal
throughput, with append maxima 10.139 → 9.974 ms and post-boundary maxima below
0.04 ms. This is consistent with higher producer pressure, not proof of an
isolated cause for every stall. Pacing used 4 MiB bursts outside append calls;
it is not a camera arrival-time simulation.

## CPU guard and its limits

The original unpinned CPU screens showed candidate declines of 8.6%/11.6% at
512 B (Orca/small epoch), and 4.3%/2.6% at bulk. These observations are retained,
not discarded as presumed noise. The small-call results triggered one bounded
confirmation block.

Inspection found identical `cpu/stream.body.c.o` disassembly, including
relocations. `stream.c.o` differs only in allocation/metric-return sizes and
the metadata clock offset following the enlarged metrics record. This does
not rule out placement, layout, linking, or allocation effects.

Confirmation pinned only the producer to logical CPU0 after creating the
worker pool. Workers retained their original affinity and count. The harness
was rebuilt with an optional producer-affinity control; comparisons are within
that block, not between different harness versions. Both versions are archived.

- Small-epoch 512 B: candidate +18.3% and +19.2% in the controlled pairs.
- Orca 512 B: completed-output rate −1.37% in both pairs, but input including
  drain +0.4%/+0.8%. One 576 MiB batch in a 20-second window is about 1.4% at
  these rates; completed-batch quantization is material here.
- CPU bulk: Orca −0.14%, small epoch −0.51% in the controlled repeat.
- Identical-parent 512 B controls: Orca +0.005%, small epoch +2.35%.

The large CPU declines did not reproduce under controlled producer placement.
That clears this bounded regression screen, not a claim of universal unchanged
CPU performance, nor a CPU optimization claim. No further tuning experiments
were performed. The CPU writer has no built-in append-body histogram today;
its zero internal count/max is unmeasured, not zero latency.

## Method and validation

42 initial timed runs plus 16 CPU confirmation runs, serialized. Each had a
5-second warmup and 20-second measurement; 64 MiB appends used 30 seconds to
reduce batch quantization. Recovery was 3 seconds. Configuration order was
seeded, with reversed order for repeated small-call and GPU bulk pairs. No
assistant builds, tests, or analysis jobs overlapped timed runs. Desktop load,
GPU clocks, and governors were not controlled.

GPU/none/discard/xor/u16; prepared 64 MiB source ring; 128 MiB staging; fixed
5 GiB layout-fit budget and reference extent, then unbounded streaming:

| Layout | Epoch | Batch | Generation |
|---|---:|---:|---:|
| Small epoch | 1 MiB | 512 MiB | 8 MiB |
| Orca | 288 MiB | 576 MiB | 864 MiB |

All GPU append-body timers remained enabled. External API timing covered input
positions crossing batch/generation boundaries, 64 MiB possible-staging
boundaries, and the following 16 calls. These labels are not internal flush
event traces. Initialization/source preparation were outside the timed loop.
The primary rate counts completed physical output after D2H; input throughput
including final drain is also retained.

Validation: 66/66 GPU-enabled non-S3 tests and 42/42 CPU-only non-S3 tests;
10 sweep/migration tests and 4 report-selection tests; CPU/GPU CLI smoke in
both timing modes; 12 successful fixed-input parent/candidate digest runs.
The new tests cover sampling phase/rollover, exact work and observed coverage,
8 KiB threshold and split-copy behavior, per-array phase and full-timing mode,
clock-call counts, readback/padding/index CRC, and JSON/text semantics.

One initial standalone-harness preflight failed because its runtime search
path found the CUDA stub library. Adding the repository's normal Nix driver
RPATH resolved it; all timed runs succeeded. Hashing runs are correctness-only:
their slow sink can create hundreds-of-ms stalls and is excluded from latency
and throughput summaries above. S3 integration tests were not run.

## Retained evidence

[pairs.csv](pairs.csv) contains every timed pair, including identical-build
controls, full profiling, paced input, original CPU screens, and confirmation.
No rows were selected out for the summary. Original source/binary hashes,
commands, raw JSON/stdout/stderr, environment snapshots, harness versions,
pre-registered method and confirmation plan are retained locally at:

`/tmp/chucky-sampled-memcpy-20260908-hUnhU6`

`summary.json` includes every successful timed run; `results/` also preserves
the failed preflight and all correctness runs. `artifacts/` contains both
harness source versions and binaries. The original CPU binaries were
reconstructed after confirmation and verified bit-for-bit against the hashes
recorded before the initial runs. This local archive is not a published data
download or part of the source installation.

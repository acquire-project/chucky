# RTX 5080 host microscopy Pareto measurements

Measured on the Windows host `oreb` on 2026-09-16. The CPU benchmark requested
32 compression threads in every execution and verified that the executable
reported 32. The retained public studies use local Windows filesystem storage
for the filesystem sink:

| Study | Inputs | Configurations | Samples | References | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Core CPU](../../../bench/studies/microscopy/rtx5080-core-cpu32-20260916/study.json) | 2 | 36 | 108 | 8 | 0 |
| [Transfer CPU](../../../bench/studies/microscopy/rtx5080-transfer-cpu32-20260916/study.json) | 5 | 50 | 100 | 20 | 0 |
| [Core GPU](../../../bench/studies/microscopy/rtx5080-core-gpu-20260916/study.json) | 2 | 48 | 144 | 8 | 0 |
| [Transfer GPU](../../../bench/studies/microscopy/rtx5080-transfer-gpu-20260916/study.json) | 5 | 50 | 100 | 20 | 1 |
| [DynaCell GPU refinement](../../../bench/studies/microscopy/rtx5080-refinement-gpu-20260916/study.json) | 1 | 16 | 48 | 4 | 1 |

All five studies are included in the microscopy report index. DynaCell's GPU
view selects the refinement study; its CPU view selects the transfer study.
The core comparison and DynaCell refinement use three planned samples per
configuration; the transfer screen uses two. Each
`study.json` retains the ordered plan, per-process results, benchmark and corpus
checksums, machine metadata, and the executable build record. The separate
[Blosc GPU sweep](../blosc-rtx5080-20260916/README.md) contains 200
configurations and its own Pareto summary.

The GPU executable was built with CUDA 13.3 and nvCOMP 5.3 for the RTX 5080;
37 GPU image readback checks passed before collection. GPU executions requested
four worker threads. The build used CMake `Release`, while the L40 comparison
build used `RelWithDebInfo`; compiler and library details are retained in each
study's build record.

Across the three GPU studies, 322 of 324 planned executions passed. The two
errors were `insufficient_coverage` on DynaCell filesystem output with Blosc
Zstd: one 256 KiB sample in the transfer screen and one 128 KiB sample in the
refinement. In each case, final drain exceeded the benchmark's 10% coverage
limit. The failed samples are retained in their respective archives and
excluded from throughput summaries; the affected configurations need further
confirmation. No out-of-memory error occurred. The runner continued after
both errors as requested.

The DynaCell GPU filesystem `none` 256 KiB median was 2.798 GiB/s in the
two-sample transfer screen and 2.347 GiB/s in the three-sample refinement.
These are separate measurement phases with appreciable variation; the report
uses refinement for DynaCell GPU and retains transfer for review. The observed
rankings should be treated as candidates for confirmation rather than a
precise performance ordering.

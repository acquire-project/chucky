# RTX 5080 Blosc Pareto rerun

Measured on this Windows desktop on 2026-09-16, from 15:45:37 to 16:02:25 UTC.
The retained run contains 200 configurations, each with one warmup and three
measured executions. All 800 benchmark processes passed. The successful Blosc
points produce 23 cross-codec throughput/compression frontier points for the
four fill/chunk workload groups. `failures.csv` is empty except for its header.
This archive is included in the site's Blosc Pareto report as a separate
September 16 experiment, alongside the September 5 RTX 5080 run.
Across the 200 matched configurations, the median ratio of new to September 5
median throughput is 1.350; all 200 reported compression folds are identical.
The new within-configuration median min–max span is 2.66%.

The matrix, input geometry, sink, randomization, validation, and benchmark
executable match the [September 5 RTX 5080 archive](../blosc-rtx5080-20260905/README.md).
The executable SHA256 is identical to that archive's recorded binary hash, so
its source revision is `19cd6d30068cbd7aec88048e54817ccae1228c7c`.
`provenance.json` also records the collection checkout as
`bc360b79d24724390f261a985480b51f237c541e` in its `source_commit` field;
that field describes the checkout used to launch the retained binary, not the
binary's source. The collector run used [run.executed.mjs](run.executed.mjs).
[run.mjs](run.mjs) corrects that source distinction for a future rerun and
requires the same validated executable hash.

- [raw-results.jsonl.gz](raw-results.jsonl.gz) retains every process command,
  stdout, stderr, exit code, validation result, and timing. The uncompressed
  SHA256 matches `provenance.json`.
- [summary.csv](summary.csv) contains the measured medians and observed ranges.
- [pareto-frontier.csv](pareto-frontier.csv) contains points undominated in
  median input throughput and reported compression fold within each fill/chunk
  group, across Blosc LZ4 and Zstd.
- [failures.csv](failures.csv) records failed configurations when present. The
  collector continues after a failed benchmark process and retains its error.

The graphics desktop was active during collection. These are whole-system
measurements under this run's CPU, GPU, driver, and background load; compare
their observed ranges before interpreting small differences from earlier runs.

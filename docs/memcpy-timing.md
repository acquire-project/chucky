# GPU host-copy timing

The GPU producer copies caller input into pinned staging memory. Reading a
clock twice and updating timing statistics on every small copy can consume a
substantial part of the append path. The default now times one out of every 64
copies smaller than 8 KiB, and every copy of 8 KiB or larger. The cutoff is a
provisional instrumentation policy, not a change to the copy implementation.
It applies to each actual copy fragment: an append may split at staging or
batch boundaries.

Sampling is per array. The first small copy is timed; subsequent 64-copy blocks
rotate the sampled position through all 64 offsets. This avoids always choosing
the same power-of-two staging offset, but is not random sampling and does not
guarantee an unbiased estimate of the population. In particular, a sampled
maximum is **not** the worst copy latency.

For profiling, set `tile_stream_configuration.full_memcpy_timing` to a nonzero
value, or pass `--full-memcpy-timing` to a benchmark. This times every host copy
and can significantly reduce throughput for very small appends. The CPU backend
ignores this option.

The additional public metrics fields change the C struct layout. Rebuild
callers with matching headers and library; do not mix old and new binaries.

The single-array GPU writer's per-append body-latency timer and histogram remain,
as are pipeline waits, batch/generation scheduling, and final-flush timing.
There is no additional buffering, deferred copy, or change to input ownership.
The existing append-body timer does not include the API wrapper's CUDA context
selection; use external API timing when that distinction matters.
The multiarray writer has no outer update-latency timer today; that is unchanged.

## Interpreting the metrics

`stream_metrics.memcpy_calls` and `memcpy_bytes` are exact GPU work totals,
including untimed copies. They count copy fragments, not append calls. They
remain zero on the CPU backend.

Every field in `stream_metrics.memcpy` describes only the measured copies:
`count`, byte counts, cumulative time, and extrema. Sampled times are never
scaled up to pretend they are exact whole-stage totals. An observed throughput
uses only the observed bytes and their observed time.

Benchmark text labels partial observations `Memcpy[smp]` and prints both exact
work and timed coverage. JSON adds `memcpy_work` with exact `calls`, `bytes`, and
`timing_scope` (`sampled` or `full`, based on actual coverage). When any copies
were untimed, the timing row is `stages.memcpy_sample`, **not** `stages.memcpy`.
This prevents existing report consumers from mistaking a partial time sum for
the whole memcpy stage. The normal `stages.memcpy` key is retained when every
copy was timed, including default-policy runs containing only large copies.
Empty streams and CPU streams do not fabricate GPU work totals.

Consumers that need exact work should read `memcpy_work`. Consumers that need
whole-stage copy time must run with full timing. Do not sum `memcpy_sample` time
into producer utilization or compare its maximum as if it covered all appends.

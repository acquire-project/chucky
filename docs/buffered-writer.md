# Buffered input writer

The [buffered writer](../src/writer.buffered.h) copies input into a bounded
queue for CPU/GPU tile streams. Use it to absorb temporary downstream stalls
when a paced producer needs to reuse input promptly. The extra copy can reduce
sustained throughput.

## Configure and use

`capacity_bytes` bounds queued and active input. `max_drain_bytes` limits each
downstream batch; zero uses the full capacity. Capacity must be nonzero and the
cap must not exceed it. Both values and input sizes must respect downstream
granularity, such as whole elements. Tune these example sizes to your workload:

```c
#include "writer.buffered.h"

int write_buffered(struct writer* downstream, struct slice input)
{
  const struct buffered_writer_config config = {
    .capacity_bytes = 128u * 1024 * 1024,
    .max_drain_bytes = 16u * 1024 * 1024,
  };
  struct buffered_writer* buffered = buffered_writer_create(downstream, &config);
  if (!buffered)
    return 1;
  struct writer* writer = buffered_writer_as_writer(buffered);
  int failed = writer_append_wait(writer, input).error != 0;
  failed |= buffered_writer_destroy(buffered); // flush, close, and join
  return failed;
}
```

## Input and ownership

Append copies available input and returns the unaccepted suffix in `rest`.
The accepted prefix can immediately be reused; `writer_append_wait` retries the
suffix. A full queue blocks without dropping input.

The worker drains the contiguous queued prefix up to the cap or ring boundary.
Drains can split or combine producer appends. The worker never waits to fill a
batch: it releases each completed batch and immediately continues while input
remains.

Keep downstream, its configuration, sink, and GPU context alive until adapter
destruction. Only the worker may use downstream. Serialize append, flush, close,
and destroy; stop concurrent stats readers before destroy. Custom writers must
allow calls from another thread and release consumed input before returning.

## Finish and handle errors

Flush stops acceptance and drains queued input unless downstream has terminated.
It still flushes downstream after failure. Close publishes metadata after flush;
before flush it does nothing. Both are idempotent and preserve errors.

A downstream failure or early completion stops forwarding. Remaining accepted
input is counted in `abandoned_bytes` and makes finalization fail. Later appends
accept nothing. Always check errors, including from destroy: it flushes, closes,
joins, and frees the adapter without destroying downstream.

## Measure

`buffered_writer_get_stats` reports byte occupancy, backpressure, and downstream
batch timing. Forwarded bytes measure consumption, not storage persistence.

Compare [bench_buffered_writer](../bench/bench_buffered_writer.c) with
`--buffer-bytes 0` and your chosen capacity; vary `--drain-bytes` at fixed capacity.
Test `--fps 0` and your expected frame rate. A cap limits bytes, not call
duration, and buffering cannot absorb sustained overload.

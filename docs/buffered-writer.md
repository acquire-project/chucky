# Buffered input writer

`writer.buffered.h` provides an optional copying adapter around `struct writer`.
It can absorb temporary downstream stalls for a paced producer, including CPU
and GPU tile streams. One worker forwards accepted input in order. Creating the
adapter does not change the underlying pipeline or its output format.

## Capacity and acceptance

Configure **`capacity_bytes`** and **`max_drain_bytes`**. The adapter treats input
as a byte stream: capacity, backpressure and drains do not depend on how callers
partition that stream into appends. Capacity must be nonzero. A zero drain cap
uses the full capacity; a cap greater than capacity is invalid. Capacity, cap
and input sizes must respect downstream granularity, such as whole elements.

The payload allocation is exactly `capacity_bytes` in a shared ring, plus fixed
bookkeeping and one worker thread and stack. The producer copies directly into
free space while the worker drains an older prefix. Active drains pin only their
own bytes. There are no per-append descriptors, allocations or packing copies.

Each drain submits up to `max_drain_bytes` of the contiguous queued prefix in
one downstream append, stopping at the physical end of the ring. The byte cap
can split a producer append or combine several appends, regardless of their
sizes. Partial downstream acceptance or stalls can require retries of the
remaining suffix.

The worker starts as soon as input is available, without waiting to fill a
batch. On return, that batch's bytes are reusable and the worker immediately
starts another if a backlog remains. For example, a 128 MiB capacity with a
16 MiB cap leaves at least 112 MiB outside the active drain for queued or new
input. A size cap does not bound call duration: downstream startup or flush work
can occur even for a small append.

An append waits for free capacity, copies as much input as fits up to the ring
boundary, and returns the unaccepted suffix in `writer_result.rest`. The caller
can immediately overwrite or free the accepted prefix. `writer_append_wait`
retries any suffix. An empty input consumes no capacity. A full queue blocks
without dropping or overwriting input. A bounded queue cannot compensate for
sustained overload or bound a call's duration if downstream stops returning.

Acceptance means the copy is queued, not that downstream consumed or persisted
it. Always check flush and close, even if all appends succeeded.

## Completion and errors

| Event | Queued input | Result |
| --- | --- | --- |
| Normal flush | Forward every accepted byte, then flush downstream | Success only if draining and downstream flush succeed |
| Downstream append fails or stops making progress | Stop forwarding; abandon the unconsumed accepted suffix | Sticky failure, including on later flush/close |
| Downstream returns `finished` with accepted bytes left | Abandon its unconsumed suffix and later queued bytes | Sticky failure; `abandoned_bytes` counts the loss |
| Downstream returns `finished` after consuming all accepted bytes | Nothing remains | Future appends return `finished`; flush still runs |
| Close after flush | Publish through downstream close | Wait for completion and report failures |
| Close before flush | Continue accepting | No-op, matching tile-stream writers |

After a terminal result, a later append returns its entire input unaccepted.
`rest` always refers to that call's original input, never to internal copies.
Previously accepted data cannot be recovered from the adapter. Inspect
`abandoned_bytes` when deciding how to handle an acquisition failure.
`forwarded_bytes` counts downstream consumption; a later I/O error may still
prevent those bytes from reaching storage.

Flush stops acceptance permanently. It calls downstream flush even after an
append error, so already submitted work can drain. Flush and close are
idempotent and preserve failures. Destroy runs both, joins the worker, frees the
adapter, and returns an error if either failed. It never destroys downstream.
Keep the sink alive through destruction of the underlying tile stream, too.

## Ownership and concurrency

The adapter borrows downstream. Its writer, configuration, sink, and GPU context
must outlive the adapter. While the adapter exists, only its worker may access
downstream, including downstream metrics/accessors. Destroy the adapter before
reading those metrics or using downstream separately.

The underlying writer must allow calls from a different thread and release its
reference to consumed input before append returns. CPU and GPU tile-stream
writers satisfy this contract; the GPU writer installs its own CUDA context
around each operation. Other thread-affine writers require their own adaptation.

Externally serialize calls to append, flush, close and destroy. A concurrent
observer may call `buffered_writer_get_stats`; stop observers before destroy.
Downstream callbacks must not call back into the adapter.

```c
#include "writer.buffered.h"

int write_buffered(struct writer* downstream, struct slice input,
                   size_t capacity_bytes, size_t max_drain_bytes)
{
  struct buffered_writer_config config = {
    .capacity_bytes = capacity_bytes,
    .max_drain_bytes = max_drain_bytes,
  };
  struct buffered_writer* buffered = buffered_writer_create(downstream, &config);
  if (!buffered)
    return 1;
  struct writer* writer = buffered_writer_as_writer(buffered);
  int failed = writer_append_wait(writer, input).error != 0;
  // The accepted prefix of input can be reused now. An error may leave a suffix.
  failed |= writer_flush(writer).error != 0;
  failed |= writer_close(writer).error != 0;
  failed |= buffered_writer_destroy(buffered);
  return failed;
}
```

## Measurement

Stats distinguish accepted, forwarded, abandoned and pending bytes. Pending
bytes include the active downstream append, but exclude downstream asynchronous
work after that append returns. The peak captures brief occupancy spikes even
if the observer samples infrequently.

`backpressure_ns` measures caller time waiting for capacity. `downstream_ns` and
`max_downstream_ns` measure drained batches, including partial-append retries.
`completed_batches` and `max_batch_bytes` describe batching without preserving
producer append boundaries or retaining timing records per append.

`bench_buffered_writer` compares direct and buffered appends using identical
1024 x 1024 uint16 frames, 256 x 256 chunks, 32 epochs per batch, four processing
threads, a 64-frame spatial/temporal XOR pattern, and a discard sink by default.
The pattern is prepared before timing. Creation is excluded; first-append
startup, append stalls, and final flush/close are included. The discard sink
isolates pipeline capacity; filesystem output is checked separately by CPU and
GPU readback tests, including metadata and immediate input reuse.

```sh
# Repeat in alternating order for both cpu/gpu and none/zstd.
build/bench/bench_buffered_writer --backend cpu --codec none --frames 16384 --buffer-bytes 0
build/bench/bench_buffered_writer --backend cpu --codec none --frames 16384 --buffer-bytes 134217728 --drain-bytes 16777216

# Paced arrival: fps=0 disables pacing; buffer-bytes=0 selects direct appends.
build/bench/bench_buffered_writer --backend gpu --codec zstd --frames 2048 --fps 500 --buffer-bytes 0
build/bench/bench_buffered_writer --backend gpu --codec zstd --frames 2048 --fps 500 --buffer-bytes 134217728 --drain-bytes 16777216
```

JSON reports capacity, effective drain cap, maximum batch size and peak pending
input in bytes. `--drain-bytes 0` uses the full configured capacity. Compare caps
at the same capacity to distinguish batching from stall absorption. The 16 MiB
example is a tunable starting point, not a universal optimum.

Only the benchmark knows about frames. It reports caller latency per frame and
downstream latency per batch. `submission_to_forward_ms` measures each frame's
submission until its first byte reaches the downstream call; this includes
copying and any preceding backpressure, rather than starting after acceptance.
Completion delay ends when the call containing that frame's last byte returns.
A drain may split a frame or combine parts of several frames. These observations
measure downstream consumption, not persistence. The observer grows its timing
array if a small byte cap produces more batches than frames.

The benchmark also reports lateness against scheduled arrival and backpressure.
The unpaced second-half byte rate includes the final drain, so accepted backlog
is never counted as completed work. Compare several capacities and include an
overloaded paced run. Smaller buffers may reduce copying/cache overhead but
absorb shorter stalls.

The extra copy consumes CPU and memory bandwidth. Buffering can reduce caller
stalls while increasing downstream delay and reducing maximum throughput,
especially with codec `none`. Select capacity and decide whether to enable the
adapter using measurements of the intended workload; there is no default
capacity or general throughput-parity guarantee.

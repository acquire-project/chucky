# Buffered input writer

`writer.buffered.h` provides an optional copying adapter around `struct writer`.
It can absorb temporary downstream stalls for a paced producer, including CPU
and GPU tile streams. One worker forwards accepted input in order. Creating the
adapter does not change the underlying pipeline or its output format.

## Capacity and acceptance

Choose `slot_bytes` and `slot_count` explicitly. The payload allocation is their
product; bookkeeping is fixed per slot, with one worker thread and its stack.
The slot being forwarded counts toward capacity. There are no allocations in
the adapter's append path. Slots are not combined: a short append occupies one
slot until downstream consumes it. Choose a slot size matching normal producer
appends, and a count covering the temporary backlog your workload can tolerate.
Both slot size and input sizes must respect downstream element granularity.

An append waits until one slot is free, copies up to `slot_bytes`, and returns
the unaccepted suffix in `writer_result.rest`. The accepted prefix is owned by
the adapter, so the caller can immediately overwrite or free that prefix.
`writer_append_wait` handles inputs larger than one slot. An empty input consumes
no slot. When full, the queue blocks; it does not overwrite old data or drop new
input. A bounded queue cannot compensate for sustained overload, and cannot
bound an append's duration if downstream stops returning.

Acceptance means the copy is queued, not that downstream consumed or persisted
it. Always check flush and close, even if all appends succeeded.

## Completion and errors

| Event | Queued input | Result |
| --- | --- | --- |
| Normal flush | Forward every accepted byte, then flush downstream | Success only if draining and downstream flush succeed |
| Downstream append fails or stops making progress | Stop forwarding; abandon the unconsumed accepted suffix | Sticky failure, including on later flush/close |
| Downstream returns `finished` with accepted bytes left | Abandon its unconsumed suffix and later queued slots | Sticky failure; `abandoned_bytes` counts the loss |
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
                   size_t slot_bytes, size_t slot_count)
{
  struct buffered_writer_config config = { slot_bytes, slot_count };
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
bytes and occupied slots include the active downstream append, but exclude
downstream asynchronous work after that append returns. Peaks capture brief
occupancy spikes even if the observer samples infrequently.

`backpressure_ns` measures caller time waiting for capacity. Queue time runs from
acceptance (after the copy) until the worker starts forwarding. Downstream time
includes partial-append retries. Completion time ends when downstream append
returns, not when the data reaches storage. Timing aggregates are per slot, not
weighted by bytes. These clocks exclude caller copy time from queue time.

`bench_buffered_writer` compares direct and buffered appends using identical
1024 x 1024 uint16 frames, 256 x 256 chunks, 32 epochs per batch, four processing
threads, a 64-frame spatial/temporal XOR pattern, and a discard sink by default.
The pattern is prepared before timing. Creation is excluded; first-append
startup, append stalls, and final flush/close are included. The discard sink
isolates pipeline capacity; filesystem output is checked separately by CPU and
GPU readback tests, including metadata and immediate input reuse.

```sh
# Repeat in alternating order for both cpu/gpu and none/zstd.
build/bench/bench_buffered_writer --backend cpu --codec none --frames 16384 --slots 0
build/bench/bench_buffered_writer --backend cpu --codec none --frames 16384 --slots 16

# Paced arrival: fps=0 disables pacing; slots=0 selects direct appends.
build/bench/bench_buffered_writer --backend gpu --codec zstd --frames 2048 --fps 500 --slots 0
build/bench/bench_buffered_writer --backend gpu --codec zstd --frames 2048 --fps 500 --slots 16
```

JSON reports caller and downstream append latency, submission-to-downstream
completion delay, lateness against scheduled frame arrival, peak queue
occupancy, queue delay and backpressure. The unpaced second-half byte rate
includes the final drain, so accepted backlog is never counted as completed
work. Compare several capacities and include an overloaded paced run. Smaller
queues may reduce copying/cache overhead but absorb shorter stalls.

The extra copy consumes CPU and memory bandwidth. Buffering can reduce caller
stalls while increasing downstream delay and reducing maximum throughput,
especially with codec `none`. Select capacity and decide whether to enable the
adapter using measurements of the intended workload; there is no default
capacity or general throughput-parity guarantee.

# GPU pipeline design and open work

**Status.** The five-step rebuild is finished and merged
(#143, #144, #148, #149, #151). "Completed rebuild" is history, kept because it
explains why the current shape looks the way it does.

This file is a working plan, not user-facing documentation. Reference docs for
using chucky are in `docs/`.

## Terms

- **Chunk** — the smallest block of data compressed and stored on its own.
- **Append dimension** — a dimension the stream grows along, outermost in
  storage order. These are the leading dimensions whose chunk size is one, and
  there is always at least one of them; `dims_n_append` holds the exact rule,
  including how a downsampled dimension shortens the run.
- **Epoch** — one chunk-deep layer of the array, spanning the full extent of
  every dimension that is not an append dimension. Its chunks all fill at the
  same time, which is what makes the layer the smallest thing that can go
  downstream to be compressed.
- **Batch** — the epochs compressed and gathered together, `epochs_per_batch`
  of them.
- **Append** — one call handing data to the writer. Appends accumulate in the
  staging buffer and do not reach the device on their own.
- **Dispatch** — one transfer of a filled staging buffer, and the scatter that
  places it. Covers as many epochs as the buffer spans.
- **Shard** — a group of chunks written as one file, ending with an index of
  each chunk's offset and size.
- **Sink** — whatever finished shards are handed to: files, object storage, or
  a test double.
- **Stage** — one step of work: bring data in, build reduced levels, compress,
  gather into shards, copy back, hand to the sink.
- **Slot** — the pipeline runs two batches at a time, so most buffers exist in
  pairs. The code calls the index `fc`.
- **Generation** — each reuse of a slot for a new batch. Most past bugs were
  one generation reading or overwriting a buffer the previous one still used.
- **Kick** — start a batch through compress and gather.
- **Drain** — wait for a kicked batch to finish, then hand its bytes to the
  sink.
- **Ordering rule** — a "this must happen before that" rule between two CUDA
  streams, or between the host and a stream. The code calls these *edges*.
- **Page-aligned sink** — a sink that requires every write to start at a
  multiple of its page size, such as a file opened for unbuffered writes. A
  batch then writes only whole pages and carries the leftover bytes, the
  **tail**, into the next batch.

## Pipeline structure

**Five CUDA streams** (`gpu_streams`, `src/gpu/schedule.h`): `h2d`, `compute`,
`compress`, `d2h`, and a separate `drain`. Drain-time copies get their own
stream because a compressed payload's size is known only after its metadata
has been copied back to the host. Keeping those exact-size copies on `drain`
stops them from queueing behind later metadata work on `d2h`.

**One table of ordering rules** (`src/gpu/ordering.{h,c}`). Every cross-stream
and host-to-stream rule is a named entry with its producer, its consumer, and
the buffer it protects. Debug builds check that the stream a rule is recorded
or waited on matches its declaration, and warn at shutdown about rules that
were recorded but never waited on. Timing-only events stay out of the table so
a measurement can never be mistaken for a rule.

**Pools carry the ordering** (`src/gpu/pool.{h,c}`). A pool binds a recycled
buffer to the two rules that order its generations: *ready* (producer to
consumer, contents are valid) and *consumed* (consumer to producer, safe to
reuse). Acquiring queues the wait before handing out the pointer; releasing
records completion. You cannot get the pointer without getting its ordering,
which is what makes the #140 and #141 bugs hard to write again.

**One scheduler** (`src/gpu/schedule.{h,c}`) owns stream creation, how deep the
pipeline runs, which stages run for the current settings, and where the
acquires and releases go. Stages are payload only: they compute and copy, and
do not decide ordering.

**Two workers.** Staging copies run on a small thread pool, and each kicked
batch is queued to a delivery worker and joined before its slot is refilled.
The producer thread decides what happens but no longer moves large amounts of
data itself.

**Dispatch groups appends** (#173). An append copies into the pinned staging
buffer and returns. Nothing reaches the device until that buffer fills or the
caller flushes, at which point one transfer moves the whole buffer and one
scatter places it. A dispatch is capped at the room left in the batch, since
that is what the chunk pool holds — or at one epoch for a multiscale stream,
whose linear buffer holds one. So `buffer_capacity_bytes` sets the transfer
size, and the append cursor runs ahead of the epochs the schedule has counted
by whatever is still staged.

### Cost of grouped dispatch

Frame-sized appends stopped paying a transfer and a kernel launch each. On one
L40: `256cube` at 128 KiB per append went 3.0 to 10.8 GiB/s, `smallepoch` at
512 B went 0.02 to 3.0 GiB/s, and dispatches fell from 153,600 and 2,000,000 to
150 and 8. Waits for a staging buffer fell from 35,146 to 19, and from 147,173
to none.

The producer pays in the copy. Appends used to land in the same first bytes of
the pinned buffer and stay in cache; they now sweep all of it, so that stage
went 584 to 1398 ms at the same call count. Write-combined staging memory
recovers about a third of that but costs the pooled large-append path more than
it gains, so it was measured and rejected. #183 tracks the copy threshold
instead.

### Flush and failure

A flush separates work that must happen from work that claims the output is
complete. Joining the delivery worker and draining the sink always run, because
queued writes point into buffers teardown frees. Finalizing a shard index and
writing the array shape run only when everything before them succeeded: a reader
cannot tell a complete array from one whose tail never arrived, so a failed
flush leaves the output short rather than making it look whole.

A dispatch that fails partway leaves epochs transferred but uncounted, and
nothing can un-enqueue them. The array records that and stops taking data. The
latch is per array, so one array's failure does not silence the others in a
multi-array stream.

### Host-coordinated aggregation

A page-aligned batch reads the tail its predecessor carried, and that tail
reaches the device only during the predecessor's delivery. The two are ordered
on the host rather than on the device.

Such a batch is first `PREPARED`: the scheduler acquires its chunk and
aggregate slots, uploads its batch tables, queues compression, builds the
handoff, and keeps the plan and pool views in the scheduler slot. The delivery
worker moves it to `SUBMITTED` only once the preceding generation's sink
delivery and tail upload have returned. Submitting queues aggregation on the
existing compress stream, records `AGG_DONE` and `POOL_CONSUMED`, and queues
the usual metadata copy back. The job becomes `DONE` after the exact-size
payload copy and sink delivery finish.

Generations are monotonic and start at 1. `tail_ready_generation` starts at 0,
so generation N may be submitted only once that counter reads N-1. Before
preparing generation N+1 the producer waits for N to be submitted, which keeps
the shared codec-size arrays, lookup tables, and shard tables alive through N's
aggregation without a second copy of each.

No CUDA work waits on state that a future host action must produce; the tail
dependency lives only in the worker's condition-variable state machine.
Batches that need no tail keep submitting aggregation directly.

Selection covers four cases, one per value of `schedule_mode`:

| Mode | When |
|---|---|
| `SCHEDULE_PIPELINED_DIRECT` | no alignment requirement; the producer submits aggregation and queues only the drain |
| `SCHEDULE_PIPELINED_HOST_COORDINATED` | page-aligned with a delivery worker |
| `SCHEDULE_DRAIN_BEFORE_KICK` | page-aligned with no worker, so the producer orders tail uploads by draining first |
| `SCHEDULE_DRAIN_AFTER_KICK` | multi-array, which swaps per-array state between calls |

Coordinator and sink errors are sticky, and they wake the producer and any
thread waiting to join. A job that has not yet submitted is cancelled without
aggregating. A job that has already submitted still drains, because its work is
queued on the device and its output has to reach the sink and release its pool
slot. During teardown the worker finishes the valid jobs and cancels the rest
before any stream is synchronized.

### Cost of host coordination

Aggregation is now enqueued only after the preceding batch's delivery returns
from its tail upload, where the removed device gate let the GPU run ahead on
its own. Measured against `ad1125b` on one L40, writing page-aligned output to
a local unbuffered store, four scenarios with two repetitions each, interleaved
in one allocation: `256cube` block -0.1%, `256cube` frame +7.0%, `smallepoch`
block +0.5%, `orca2` block -0.3% throughput. The producer's wait moved out of
the I/O fence and into the flush stall without changing wall time.

A fast sink is the case that most favors the run-ahead the gate provided. A
slower sink is unmeasured, as is Windows.

## Ordering rules

Twelve names cover seven distinct events plus two rules that are call-order
only. One event can back several names when it guards different buffers for
different readers; those extras are marked "shares" below.

| Name in code | producer → consumer | protects | how | per slot |
|---|---|---|---|---|
| `STAGING_SCATTER_DONE` | compute → h2d | staging input buffer, read before it is overwritten | event | yes |
| `STAGING_H2D_DONE` | h2d → compute | staging input buffer contents | event | yes |
| `STAGING_FREE` | h2d → host | host staging buffer safe to refill | shares `STAGING_H2D_DONE` | yes |
| `POOL_FILLED` | compute → compress | chunk pool batch contents | event | no |
| `LOD_DONE` | compute → compress | reduced-level chunks in the pool | event, owned by the reduced-level timer | yes |
| `AGG_DONE` | compress → d2h | gathered output slot | event | yes |
| `POOL_CONSUMED` | compress → compute | chunk pool reuse and re-zero (#140) | shares `AGG_DONE` | yes |
| `SLOT_DRAINED` | d2h or drain → compress | gathered slot reuse | event | yes |
| `D2H_DONE` | d2h or drain → host | host copy stable for the sink | shares `SLOT_DRAINED` | yes |
| `CHUNK_INDEX_READY` | d2h → host | chunk offsets and sizes; only with a codec | event | yes |
| `DRAIN_BEFORE_REKICK` | host | drain a slot before kicking it again | call-order rule | yes |
| `DELIVER_OLDEST_FIRST` | host | drains follow batch generation order | call-order rule | yes |

Three notes:

- **Borrow, don't own, when an event does double duty.** The reduced-level
  end-of-work timer also serves as `LOD_DONE`, recorded once and attached with
  `gpu_ordering_bind`. Test harnesses use the same hook to stand in for a
  producer.
- **The "recorded but never waited" check is weak for events**, because all of
  them are pre-signaled at startup so the first wait costs nothing. What caught
  a deleted wait in testing was the shutdown report of rules that saw no wait
  for the whole run.
- **Some rules are idle depending on settings** (`POOL_CONSUMED` on the
  multi-array path, `CHUNK_INDEX_READY` without a codec), so the shutdown
  warning fires by design there. Fixing that needs a per-configuration list of
  which rules should be live — see "Multi-array composition".

## Completed rebuild

Five steps, each shippable on its own and each replacing one piece in place.
There was never a second engine running alongside the first. All merged:

1. **One table of ordering rules** — #143. Moved every record and wait through
   `ordering.{h,c}`, tagged timing-only events, deleted three rules that turned
   out to be dead.
2. **One engine setup and teardown** — #144, shared by single-array and
   multi-array, replacing a field-by-field copy checklist.
3. **Pools that carry ordering** — #148. Staging, chunk pool, and gathered
   slots became pools with explicit ready and consumed generations.
4. **One scheduler** — #149. Stream creation, pipeline depth, and fallbacks
   moved into `schedule.{h,c}`; the old orchestration state was deleted.
5. **Workers** — #151. Staging copies and sink delivery moved off the producer
   thread.

Both shipped silent-corruption bugs (#140, #141) were lifetime bugs rather than
logic bugs: a buffer was reused or read while its previous user was still in
flight. The stage code forgot a rule, and the old structure allowed it.

## Open work

Three items, in the order I would do them.

### 1. Trustworthy measurements

This blocks every performance question and is the smallest piece of work here.
The numbers mislead in four ways:

- The scatter timer reads elapsed time without checking the result. When the
  event is not ready the sample is dropped silently — on some settings more
  than half the scattered bytes go uncounted.
- The reduced-level timers use one event pair per batch, re-recorded every
  epoch, so each reported row covers only the batch's last epoch.
- The producer's real waiting time is the `StagingFree` stall. It is measured
  and printed to the console table, but left out of the JSON the sweep analyzes
  (`bench/bench_report.c`). This has already caused one wrong diagnosis.
- The JSON keeps averages and rates but drops each stage's total time and
  sample count, so nothing can be re-derived from a sweep file.

Fixes: export `StagingFree` and per-stage total time and count in the JSON;
check the elapsed-time result and retry instead of dropping the sample;
accumulate reduced-level timings per epoch instead of per batch; keep the "too
small to be real" filter only for detecting pre-signaled events.

The public `tail_gate` metric and JSON key remain for compatibility. They now
measure the delay between compression and aggregation while the coordinator
waits for the preceding batch's tail, and no longer describe a device gate.

The discard sink reports a fixed 4096-byte shard alignment, so a bench run with
no output path measures the page-aligned pipeline. Before that change the
default bench measured the unaligned one and never reached the tail-carry code
at all. Sweep files below schema version 4 therefore measured a different
pipeline and cannot be compared with later ones.

*Done when:* a sweep JSON alone is enough to reconstruct each stage's total
time, and the producer's stall time appears in it. Some of this already exists
on the local `issue-101-append-latency` branch (append latency distribution,
issue #101).

### 2. Multi-array composition

Multi-array should use several pipelines sharing pooled buffers, rather than
one engine with state swapped in and out. That lets `SCHEDULE_DRAIN_AFTER_KICK`
go, along with its special pool-zeroing and teardown branches.

*Done when:* two pipeline shapes remain, the two pool-zero workarounds are
gone, and the shutdown warning about unused rules is clean without a
per-configuration exception list.

### 3. Pool-only buffer access

The pools work (#148) set out to reach "no pointer to a recycled buffer crosses
a stage boundary outside the pool API". It did not. `gpu_pool_at` hands out a
pointer with no ordering attached, and has seven non-test callers in
`src/gpu/stream.c`, `src/gpu/schedule.c`, and `src/multiarray/stream.gpu.c`.
Each is correct today because a comment says so. Two zero a pool; the rest read
at an offset inside a generation the caller already holds.

#173 added another way past the pool API: the scatter stepped a raw device
pointer from one epoch's region to the next. #181 removed it. The kernel finds
each region from the element index and a stride, so the host hands out no pool
pointers of its own. Both kinds of `gpu_pool_at` call above remain.

Work: give the pool an operation for each of those two uses, then remove
`gpu_pool_at`.

*Done when:* the rule is checkable with grep instead of by reading comments.

## Out of scope

- No changes to layout math (`src/stream/config.c`), kernels
  (`src/gpu/aggregate.cu`, `src/gpu/lod.cu`), codec backends, or the zarr and
  sink layer. Their signatures stay as they are.
- No CUDA Graphs. The pipeline is small enough that a declared table plus pools
  gives the auditability without a graph executor. Dispatch overhead did turn
  out to be a real cost (#173), and the answer was to dispatch less often
  rather than to record a graph.
- No revival of PR #139's batched output slots. Measurement showed batching the
  copy back from the device is worth at most 1%.

## Related

Issues #140 and #141 (the two corruption bugs), PR #142 (the tail-state
counter), issues #101, #162, #173, #181 and #183.

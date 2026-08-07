# GPU pipeline: how it works, and what's next

**Status.** The five-step rebuild is **finished and merged** (#143, #144, #148,
#149, #151). Nothing under "The rebuild that shipped" is pending — it is kept
as history and as the reason the current shape looks the way it does.

Open work is under "What's next".

This file lives in `dev/` because it is a working plan, not user-facing
documentation. Reference docs for how to *use* chucky are in `docs/`.

## Words used here

- **Stage** — one step of work: bring data in, build reduced levels, compress,
  gather into shards, copy back, hand to the sink.
- **Slot** — the pipeline runs two batches at a time, so most buffers exist in
  pairs. The code calls the index `fc`.
- **Generation** — each time a slot is reused for a new batch counts as a new
  generation. Most past bugs were one generation reading or overwriting a
  buffer while the previous generation was still using it.
- **Kick** — start a batch through compress and gather.
- **Drain** — wait for a kicked batch to finish, then hand its bytes to the sink.
- **Ordering rule** — a "this must happen before that" rule between two CUDA
  streams, or between the host and a stream. The code calls these *edges*.

## How the pipeline is put together

**Five CUDA streams** (`gpu_streams`, `src/gpu/schedule.h`): `h2d`, `compute`,
`compress`, `d2h`, and a separate `drain`. Drain-time copies get their own
stream because compressed payload extents are known only after the metadata
copy has completed on the host. Keeping the exact-size payload copies on
`drain` avoids inserting them behind later metadata work on `d2h`.

**One table of ordering rules** (`src/gpu/ordering.{h,c}`). Every cross-stream
and host-to-stream rule is a named entry with its producer, its consumer, and
the buffer it protects. Debug builds check that the stream a rule is recorded
or waited on matches its declaration, and warn at shutdown about rules that
were recorded but never waited on. Timing-only events are deliberately kept
out of the table so a measurement can never be mistaken for a rule.

**Pools carry the ordering** (`src/gpu/pool.{h,c}`). A pool binds a recycled
buffer to the two rules that order its generations: *ready* (producer to
consumer, contents are valid) and *consumed* (consumer to producer, safe to
reuse). Acquiring queues the wait before handing out the pointer; releasing
records completion. You cannot get the pointer without getting its ordering,
which is what makes the #140 and #141 bug class hard to write again.

**One scheduler** (`src/gpu/schedule.{h,c}`) owns stream creation, how deep the
pipeline runs, which stages run for the current settings, and where the
acquires and releases go. Stages are payload only: they compute and copy;
they do not decide ordering.

**Two workers.** Staging copies run on a small thread pool; each drain is
queued to a delivery worker at kick time and joined before its slot is
refilled. The producer thread decides what happens but no longer moves large
amounts of data itself.

**Host-coordinated page-aligned aggregation.** A page-aligned single-array
batch is first `PREPARED`: the scheduler acquires its chunk and aggregate
slots, uploads its batch tables, queues compression, builds the handoff, and
retains the plan and pool views in the scheduler slot. The delivery worker
changes it to `SUBMITTED` only when the preceding generation's sink delivery
and synchronous tail uploads have returned. Submission queues aggregation on
the existing compress stream, records `AGG_DONE`/`POOL_CONSUMED`, and queues
the normal metadata D2H kick. After the exact-size payload copy and sink
delivery finish, the job becomes `DONE`.

Jobs use monotonic generations starting at 1. `tail_ready_generation` starts
at 0, so generation N can be submitted only when it equals N-1. Before the
producer prepares N+1, it waits until N is submitted. That host barrier keeps
the shared codec-size arrays, LUTs, and shard tables live through N's
aggregation without adding a stream or double-buffering those tables.

No CUDA work waits on state that a future host action must produce. The tail
dependency exists only in the worker's condition-variable state machine. If
the worker is unavailable, single-array scheduling uses the existing
drain-before-kick depth-one path; multi-array scheduling still drains
immediately because it swaps per-array state between calls. Contiguous batches
do not read tail state and keep their direct aggregation submission path.

Coordinator and sink errors are sticky. They wake producer and join waiters,
and later `PREPARED` jobs become failed `DONE` jobs without aggregation. During
teardown the worker drains valid jobs or cancels invalid ones before any CUDA
stream is synchronized.

## The ordering rules

One event can back several named rules when it guards different buffers for
different readers — the extras are marked "shares" below. Twelve names, seven
distinct events, and two rules with no GPU primitive behind them.

| Name in code | producer → consumer | protects | how | per slot |
|---|---|---|---|---|
| `STAGING_SCATTER_DONE` | compute → h2d | staging input buffer, read before it is overwritten | event | yes |
| `STAGING_H2D_DONE` | h2d → compute | staging input buffer contents | event | yes |
| `STAGING_FREE` | h2d → host | host staging buffer safe to refill | shares `STAGING_H2D_DONE` | yes |
| `POOL_FILLED` | compute → compress | chunk pool batch contents | event | no |
| `LOD_DONE` | compute → compress | reduced-level chunks in the pool | event, owned by the LOD timer | yes |
| `AGG_DONE` | compress → d2h | gathered output slot | event | yes |
| `POOL_CONSUMED` | compress → compute | chunk pool reuse and re-zero (#140) | shares `AGG_DONE` | yes |
| `SLOT_DRAINED` | d2h or drain → compress | gathered slot reuse | event | yes |
| `D2H_DONE` | d2h or drain → host | host copy stable for the sink | shares `SLOT_DRAINED` | yes |
| `CHUNK_INDEX_READY` | d2h → host | chunk offsets and sizes; only with a codec | event | yes |
| `DRAIN_BEFORE_REKICK` | host | drain a slot before kicking it again | call-order rule | yes |
| `DELIVER_OLDEST_FIRST` | host | drains follow batch generation order | call-order rule | yes |

Notes worth keeping:

- **Borrow, don't own, when an event does double duty.** The LOD end-of-work
  timer also serves as `LOD_DONE`, recorded once and attached with
  `gpu_ordering_bind`. Test harnesses use the same hook to stand in for a
  producer.
- **The "recorded but never waited" check is weak for events**, because all of
  them are pre-signaled at startup so the first wait costs nothing. What
  actually caught a deleted wait in testing was the shutdown report of rules
  that saw no waits all run.
- **Some rules are idle depending on settings** (`POOL_CONSUMED` on the
  multi-array path, `CHUNK_INDEX_READY` without a codec), so the shutdown
  warning fires by design there. Fixing that needs a per-configuration list of
  which rules should be live — see "What's next", item 2.

## The rebuild that shipped

Five steps, each shippable on its own, each replacing one piece behind an
existing seam. No flag day, no second engine. All merged:

1. **One table of ordering rules** — #143. Moved every record and wait through
   `ordering.{h,c}`, tagged timing-only events, deleted three rules that
   turned out to be dead.
2. **One engine setup and teardown** — #144, shared by single-array and
   multi-array, replacing a field-by-field copy checklist.
3. **Pools that carry ordering** — #148. Staging, chunk pool, and gathered
   slots became pools with explicit ready/consumed generations.
4. **One scheduler** — #149. Stream creation, pipeline depth and fallbacks
   moved into `schedule.{h,c}`; the old orchestration state was deleted.
5. **Workers** — #151. Staging copies and sink delivery moved off the producer
   thread.

Why it was worth doing: two shipped silent-corruption bugs (#140, #141) were
both lifetime bugs, not logic bugs — a buffer was reused or read while the
previous user was still in flight. The stage code forgot a rule and the old
structure allowed it.

## What's next

Three items, in the order I'd do them.

### 1. Make the measurements trustworthy

This is the blocker for every performance question, and it is the smallest
piece of work here. Today the numbers mislead in four specific ways:

- The scatter timer reads elapsed time without checking the result. When the
  event isn't ready the sample is silently dropped — on some settings more
  than half the scattered bytes go uncounted.
- The reduced-level timers use one event pair per batch, re-recorded every
  epoch, so each reported row is only the batch's last epoch. The reported
  total is one epoch's worth of time where it should be the whole batch's.
- The producer's real waiting time is the `StagingFree` stall. It is measured,
  and printed to the console table, but left out of the JSON the sweep
  analyzes (`bench/bench_report.c` — the console block prints it, the JSON
  block does not). This has already caused one wrong diagnosis.
- The JSON keeps averages and rates but drops each stage's total time and
  sample count, so you cannot re-derive anything from a sweep file.

Fixes: export `StagingFree` and per-stage total time and count in the JSON;
check the elapsed-time result and retry instead of dropping the sample;
accumulate reduced-level timings per epoch instead of per batch; keep the "too
small to be real" filter only for detecting pre-signaled events.

The public `tail_gate` metric and JSON key remain for compatibility. They now
measure the compression-to-aggregation delay while the coordinator waits for
the preceding batch's tail readiness; they do not represent a device gate.

*Done when:* a sweep JSON alone is enough to reconstruct each stage's total
time, and the producer's stall time appears in it. Some of this already exists
on the local `issue-101-append-latency` branch (append latency distribution,
issue #101).

### 2. Make multi-array compose

`schedule_depth` still has three values:

- `SCHEDULE_PIPELINED` — the normal single-array path. Page-aligned batches use
  host-coordinated submission; contiguous batches submit directly.
- `SCHEDULE_DRAIN_BEFORE_KICK` — the single-array fallback when the delivery
  worker cannot be created. A deterministic test forces this selection and
  exercises page-aligned tail carry.
- `SCHEDULE_DRAIN_AFTER_KICK` — the multi-array path, where per-array state is
  swapped between calls and every kick therefore drains immediately.

The remaining work is to make multi-array use several pipelines sharing pooled
buffers rather than one engine with state swapped in and out, then delete
`SCHEDULE_DRAIN_AFTER_KICK`. That also removes its special pool-zeroing and
teardown branches.

*Done when:* two pipeline shapes remain, the two pool-zero workarounds are
gone, and the shutdown warning about unused rules is clean without needing a
per-configuration exception list.

### 3. Close the last way to get a raw pointer

The pools work (#148) set out to reach "no pointer to a recycled buffer
crosses a stage boundary outside the pool API". It didn't. `gpu_pool_at`
hands out a pointer with no ordering attached, and has five non-test callers
(`src/gpu/stream.c`, `src/gpu/schedule.c`, `src/multiarray/stream.gpu.c`).
Each is correct today because a comment says so. Three of them are pool
zeroing.

Work: give the pool an operation for "zero this slot for its next use" and one
for reading at an offset inside a generation the caller already holds, then
remove `gpu_pool_at`.

*Done when:* the rule is checkable with grep instead of by reading comments.

## Not doing

- No changes to layout math (`src/stream/config.c`), kernels
  (`src/gpu/aggregate.cu`, `src/gpu/lod.cu`), codec backends, or the
  zarr/sink layer. Their signatures stay as they are.
- No CUDA Graphs. The pipeline is small enough that a declared table plus
  pools gives the auditability without a graph executor. Revisit only if
  dispatch overhead shows up as a real cost.
- No revival of PR #139's batched output slots. Measurement showed batching
  the copy back from the device is worth at most 1%.

## Related

Issues #140 and #141 (the two corruption bugs), PR #142 (the tail-state
counter), issues #101 and #162.

# Filesystem write scheduling

**Status.** Step 1 is done. Steps 2-5 are planned. Issue #178.

Only one write at a time is run by the filesystem sink: each queued job is a
blocking `pwrite`, so however deep the backlog, only one request reaches the
drive. This file is the working plan for changing that.

## Terms

- **Write in flight** — handed to the operating system, completion not yet
  seen.
- **Queue depth** — writes in flight across every shard file at once. Always
  both words: bare "depth" already means pipeline depth in `src/gpu/`.
- **Queue depth per file** — the part of that depth on one shard file. No
  gain above one while a file is still growing: on xfs the file's lock is taken
  for itself by an extending write, so requests are accepted by the kernel and
  then run one at a time. Only worth raising once the file is pre-sized.
- **Pre-sizing** — setting a file's size up front, so later writes are inside
  the file rather than extensions of it. Trimmed back to its real size at
  finalize.
- **File generation** — one open of one shard file. A pool slot is reused
  across many shard files, each reuse a new generation. No write may ever be
  applied to a generation other than its own.

## Current write path

`io_queue` (`src/zarr/io_queue.h`) is a growable ring buffer of closures with
one worker thread. Three kinds of job from `shard_pool_fs`: a copying write, a
zero-copy write over borrowed pinned memory, and the truncate and close of a
shard file at finalize.

### Ordering guarantees from one worker

Correctness, given jobs run in the order posted:

- In `pool_fs_open`, `fs_slot_finalize` is called: a close is *queued*, the
  slot marked free at once, the next file opened, no wait on that close. With
  one in-order worker the close cannot be run ahead of the writes before it,
  and the new descriptor cannot be the old one, still open — neither true once
  completions are unordered.
- The shard footer is written at `data_cursor`, after the data writes;
  truncate and close are last, after every write to that file.
- In `pool_fs_flush` a sequence number is recorded and waited on, meaning
  "everything posted so far has finished".
- In `shard_delivery.c`, a shard's footer buffer is fenced against reuse while
  its zero-copy write is queued, and footer, truncate, and close are fenced
  before the array's shape is allowed to name that data.
- In `io_event_wait` the wait is abandoned as soon as shutdown is set, so a
  fence taken at the same time as a destroy is allowed to return before its
  writes are done — covered only by the worker join in `io_queue_destroy`.

### Pools shared across levels and fields

One pool — one queue, one worker — per `store->create_pool` call. One for all
levels of a multiscale array, each level given a disjoint range of slots, so
every shard and level is put through a single worker. Worse for a plate: one
pool sized for a *single* field of view, handed to every field, so slot indices
are shared — safe only because fields are written one at a time and their opens
and closes are kept apart by the single worker. Untested.

## Cost of one write at a time

About 1.4x from more queue depth, flat past 16. Measured on xfs over an md
RAID10 of 8 NVMe drives, 48 MiB writes, queue depth taken from shard files
written at once. One in flight is today's sink.

| queue depth | GB/s | vs today |
|---:|---:|---:|
| 1 | 12.07 | — |
| 8 | 15.56 | 1.29x |
| 16 | 17.15 | 1.42x |
| 32 | 17.54 | 1.45x |
| 64 | 17.51 | 1.45x |

Today's sink is already at 69% of this array's ceiling: a 48 MiB write is split
by the array into hundreds of device requests. The gap should be larger on a
single drive, where the problem was first seen.

Every data write is already 2-48 MiB — nothing to combine, nothing to split.
The only small writes are shard footers at 4-8 KiB, one per file, about 2.6 ms
across a whole run.

Full measurements in issue #178, including between-node variation: no single
number is safe to quote.

## Plan

Five changes, each a pull request off `main`, opened after the previous
merge. Not stacked: with squash merges, every child would be rebased onto a
commit its history does not contain.

### 1. Write-path measurements

Done. No behavior change; counters only.

Main number: **distinct shard files with a write waiting at once**, as a high-
water mark and a time-weighted average — the queue depth available to a
scheduler, capped by pool size.

Also counted: shard files opened and the most open at once, summed across
levels, not guessed from pool size; a request-size histogram as a regression
check; per-request latency; copied against borrowed pending bytes; the high-
water mark of queued work. Full field list in `scripts/sweep/README.md`. Fence
stalls were already recorded as `io_fence_stall`, CPU and GPU.

New scenario `smallepoch_4shards`: both inner dimensions split in two, for four
shard files at the same chunk size — a four-times-larger epoch. In
`smallepoch_single` there is one chunk per inner dimension — one shard file, no
point raising its concurrent-shard target. Compare the pair on the write path,
not on throughput. Kept: the only scenario with a single growing file, and the
only place pre-sizing can be measured.

### 2. Correctness at queue depth one

Done, merged as `54b03fa` (#216). Output is byte-identical and failure
behavior is unchanged.

Requests are described rather than wrapped in closures, descriptors and their
syscalls sit behind a backend interface, and every open file carries a token,
so no late write reaches a recycled descriptor. Room is claimed before a
payload is copied. A truncate or close is held behind the writes posted ahead
of it on its own file. Retirement walks a tail cursor over finished slots, so
the watermark follows from the structure rather than from the dispatch order,
and a backend may report an outcome after `execute` has returned. Still one
worker.

The byte ceiling ships unlimited. The mechanism is tested; step 3 picks the
number from sweep data.

The synchronous failing-truncate hook stayed. A fake backend does not replace
it, because `cpu_stream_flush_body` checks for an error before it finalizes the
shards and never looks again. That gap is #218. The fault hooks still compiled
into the shipping backend are #219.

### 3. More queue depth

Raise the worker count, schedule ready files round-robin, and add pre-sizing so
queue depth per file can be raised above one. Add the selecting flags and
record them in the results file: `--io-writes-in-flight`,
`--io-writes-in-flight-per-file`, `--io-workers`, `--io-backend`. Sweep writes
in flight from 1 to 32.

Two things step 2 left for this step, both harmless while one worker runs
everything in order and neither harmless afterwards.

Dispatch picks the next request by scanning the window from the tail, and the
readiness test for each candidate scans the window again. That is quadratic in
the size of the window, under the queue's lock. It costs nothing today: the
filesystem backend finishes every request inside `execute`, so nothing is ever
running while the worker scans, the oldest slot is always the next one to run,
and the scan stops on its first step. The moment a backend answers
`IO_SUBMITTED`, or a second worker starts, the front of the window fills with
running requests and every dispatch walks past all of them. At the default
ceiling of 1024 requests that is around half a million comparisons per dispatch
while holding the lock the producer needs to post. The fix is to stop scanning:
the file token already carries a dense index, so the queue's open-file table can
be indexed rather than searched, and each file can carry the sequence number of
its oldest request and of its first waiting barrier. Readiness is then one
comparison.

The same change should merge the queue's open-file table with the one in
`io_queue_stats.c`. They have the same shape and the same key, and they are
updated from adjacent lines under the same lock, so every post and every retire
walks two lists instead of one.

*Done when:* the reef-l40 XFS matrix is green.

### 4. io_uring on Linux

An io_uring backend behind the same interface, opt-in, with a fallback to
threads when a ring is not allowed by the kernel or the container. At chucky's
write sizes the ceiling is reached by one ring alone, and by blocking writes
too, so evidence is needed first.

The backend interface step 2 built is general enough to hold a ring, but three
things are worth settling before the first one is written. A backend has no way
to say "not now", which a full submission queue needs; the alternatives today
are to block a worker or to fail the request. The queue never calls into the
backend at teardown, so a backend with its own thread has nowhere to stop it.
And the request handed to `execute` is the worker's own copy, which dies when
the call returns, so a backend that finishes later has to copy what it needs.

A short write is also unowned. `platform_pwrite` loops internally, so it never
reports one; a ring does. Retrying the remainder belongs in the backend, not in
the queue.

*Done when:* the XFS matrix is green, then an NFS matrix before it is made the
default.

### 5. Overlapped writes on Windows

Blocked on hardware. Until then Windows is on the thread backend, with the
completion-ports (IOCP) rewrite. No pre-sizing there: on NTFS a file's valid
data length is not extended by setting its size, and the call that would is
privileged, with old disk contents exposed.

## Settled decisions

- Registered files: dropped. No measurable difference across three runs.
- Combining or splitting writes: dropped.
- Unbuffered writes: kept. Buffered was worse on throughput and tail latency.
- Pre-sizing with `ftruncate`, not `fallocate`: same speed at chucky's write
  sizes, no space reserved so peak disk usage is unchanged, `fallocate`
  unsupported on the NFS mount. The cost: on a full disk, failure is at the
  write, not the open.

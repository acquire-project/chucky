# Filesystem write scheduling

**Status.** Steps 1 and 2 are done. Step 3 is being written, with two fixes
spun out of step 2 alongside it. Steps 4 and 5 are planned. Issue #178.

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

`io_queue` (`src/zarr/io_queue.h`) is a window of described requests
(`src/zarr/io_request.h`) drained by one worker thread. Four operations: a
write, whose payload is either copied or borrowed from pinned memory, and the
truncate and close of a shard file at finalize, which are barriers, and a
no-op used as a fence marker. Every open file carries a token, and a request
naming a generation that has closed is refused.

### Ordering guarantees

After step 2 most of these no longer depend on the order the worker runs things
in:

- A write cannot reach a recycled descriptor. `pool_fs_open` still queues a
  close, frees the slot and opens the next file without waiting, but the new
  file gets a new generation, and late requests naming the old one are refused.
- Truncate and close cannot pass the writes ahead of them on their own file,
  because they are barriers. Writes carry their own offsets, so among
  themselves they need no order.
- In `pool_fs_flush` a sequence number is recorded and waited on, meaning
  "everything posted so far has finished". Retirement walks a cursor over
  finished slots, so this holds however the completions arrive.
- In `shard_delivery.c`, a shard's footer buffer is fenced against reuse while
  its zero-copy write is queued, and footer, truncate, and close are fenced
  before the array's shape is allowed to name that data.

One case still rests on the worker. In `io_event_wait` the wait is abandoned as
soon as shutdown is set, so a fence taken at the same time as a destroy may
return before its writes are done — covered only by the worker join in
`io_queue_destroy`.

### Pools shared across levels and fields

One pool — one queue, one worker — per `store->create_pool` call. One for all
levels of a multiscale array, each level given a disjoint range of slots, so
every shard and level is put through a single worker. Worse for a plate: one
pool sized for a *single* field of view, handed to every field, so slot indices
are shared and two fields open at once write into each other's files. That is
#211, and its fix changes how slots are allocated. Untested either way: no test
in the tree writes array data through a plate field sink.

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

Full measurements are in #178, including between-node variation: no single
number is safe to quote. The program that produced them is not in the
repository and is not being added, so everything measured from here on is
measured through the sweep, on the real write path.

## Plan

Five steps, each of them one or more pull requests off `main`, opened after the
previous merge. Not stacked: with squash merges, every child
would be rebased onto a commit its history does not contain.

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

Both rewrite `io_backend.fs.c` and `shard_pool_fs.c`, so they have to be taken
in order, #218 first. #218 does not delete the truncate hook; it moves the
failure onto the worker, which is the point of the fix. #219 therefore has one
more hook to carry out of shipping code, not one fewer.

### 3. More queue depth

Issue #213. Raise the worker count, schedule ready files round-robin, add the
flags that select it — `--io-writes-in-flight`, `--io-workers`, `--io-backend`
and `--io-writes-in-flight-per-file` — and record them in the results file. The
byte ceiling gets its number here, from step 1's `io_queued_bytes_peak`.

Four pieces around it, each its own pull request, in this order.

**Index the open-file table** (#225). Dispatch picks the next request by
scanning the window from the tail, and the readiness test for each candidate
scans it again — quadratic, under the queue's lock. It is free today because the
filesystem backend finishes every request inside `execute`, so the oldest slot
is always next and the scan stops on its first step. With a second worker, or a
backend that answers `IO_SUBMITTED`, the front of the window fills with running
requests and every dispatch walks past them: around half a million comparisons
at the default ceiling of 1024, while holding the lock the producer needs to
post. The file token already carries a dense index, so the table can be indexed
rather than searched, and each file can carry the sequence number of its oldest
request and of its first waiting barrier. The same change merges that table with
the one in `io_queue_stats.c`, which has the same shape and the same key and is
updated under the same lock.

**Record a queue depth in the sweep** (#226). A run's id is built from the
scenario, codec, fill, backend, dtype, chunk size and sink, so 32 runs that
differ only in writes in flight share one id and the last one written wins. The
S3 tier shows the shape: an optional field on the run spec, a suffix on the id,
a column in the results file. Add a tier over the scenarios holding more than
one shard file, a schema version and its migration, a line in
`scripts/sweep/README.md`, and a report that reads depth as an axis.

**Pre-size shard files** (#227). `ftruncate` at open, trimmed back at finalize,
so a write lands inside the file instead of extending it. Taken after the
workers, because depth per file above one buys nothing without it and it buys
nothing on its own. `smallepoch_single` is the only scenario that can measure
it, and Windows does not get it at all — see step 5.

**Publish the measurement** (#228). A sweep at those depths, its results file
its own pull request, as with earlier sweeps. Step 4 depends on it: a ring is
worth writing only if threads do not already reach the same depth.

*Done when:* the reef-l40 XFS matrix is green and the sweep is published.

### 4. io_uring on Linux

An io_uring backend behind the same interface, opt-in, with a fallback to
threads when a ring is not allowed by the kernel or the container. At chucky's
write sizes the ceiling is reached by one ring alone, and by blocking writes
too, so evidence is needed first.

**Close four gaps in the backend interface** (#229), as its own pull request
before any ring is written. The interface step 2 built is general enough to hold
one, but a backend has no way to say "not now", which a full submission queue
needs; the alternatives today are to block a worker or to fail the request. The
queue never calls into the backend at teardown, so a backend with a thread of
its own has nowhere to stop it. The request handed to `execute` is the worker's
own copy and dies when the call returns, so a backend that finishes later has to
copy what it needs. And a short write is unowned: `platform_pwrite` loops
internally so it never reports one, a ring does, and retrying the remainder
belongs in the backend rather than in the queue.

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
- macOS keeps the thread backend. Neither step 4 nor step 5 touches it, and
  nothing else is planned for it.
- The S3 pool is out of scope. It writes on the calling thread through the AWS
  client and never reaches `io_queue`, and the last L40 sweep put it at about
  what a local write costs.
- The queue-depth microbenchmark is not being added to the repository; its
  numbers stay in #178.

## Issues

- #178, the epic.
- #208, step 1, merged as `fee70ed`.
- #212 and #216, step 2, merged as `54b03fa`.
- #213, step 3, with #225, #226, #227 and #228 around it.
- #229, then #214, step 4. #215, step 5.
- #217, a fence waiter left holding a freed lock. Fixed inside #216.
- #218, a flush misses the errors of the work it queued.
- #219, the fault hooks compiled into the shipping backend.
- #211, plate fields share pool slots. A different epic, and its fix changes how
  slots are allocated.

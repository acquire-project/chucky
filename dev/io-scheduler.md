# Filesystem write scheduling

**Status.** Step 1 is done. Steps 2-5 are planned. Issue #178.

The filesystem sink runs one write at a time. Each queued job does a blocking
`pwrite`, so however deep the backlog gets, the drive sees one request. This
file is the working plan for changing that.

It is not user-facing documentation. Reference docs are in `docs/`.

## Terms

- **Write in flight** — a write the application has handed to the operating
  system and not yet seen finish.
- **Writes in flight, total** — the number outstanding across every shard file
  at once. This is the number storage cares about. The rest of this file calls
  it **depth**.
- **Writes in flight per file** — how many of those are to the same shard file.
  Above one it buys nothing while a file is still growing: on xfs a write that
  extends a file takes the file's lock for itself, so the kernel accepts many
  requests and then runs them one after another. Pre-sizing the file is what
  makes this number matter.
- **Pre-sizing** — setting a file's size up front with `ftruncate`, so later
  writes land inside the file rather than extending it. Finalize already
  trims the file back to its real size.
- **File generation** — one open of one shard file. A slot in the pool is
  reused for many shard files over a run, and each reuse is a new generation.
  A write must never reach a generation other than the one it was issued for.

## Current write path

`io_queue` (`src/zarr/io_queue.h`) is a growable ring buffer of closures with
one worker thread. `shard_pool_fs` posts three kinds of job to it: a copying
write, a zero-copy write that borrows pinned memory, and the truncate and
close that finalize a shard file.

### Ordering guarantees from one worker

Correctness rests on that single worker running jobs in the order they were
posted:

- `pool_fs_open` calls `fs_slot_finalize`, which *queues* a close and marks the
  slot free straight away, then opens the next shard file. Nothing waits for
  that close. With one in-order worker the close cannot overtake the writes
  ahead of it, and the new file's descriptor cannot collide with the old one
  because the old one is still open. Neither holds once completions are
  unordered.
- The shard footer is written at `data_cursor`, after the data writes.
- Truncate and close come after every write to that file.
- `pool_fs_flush` records a sequence number and waits for it, which means
  "everything posted so far has finished".

Two more places depend only on the order jobs run in:

- `shard_delivery.c` fences a shard's footer buffer against reuse while its
  zero-copy write is still queued, and fences footer, truncate, and close
  before the array's shape is allowed to name that data.
- `io_event_wait` gives up as soon as shutdown is set, so a fence taken at the
  same time as a destroy can return before its writes land. Only the worker
  join inside `io_queue_destroy` covers that gap today.

### Pools shared across levels and fields

Each `store->create_pool` call creates one pool, and therefore one queue and
one worker. A multiscale array creates one for all of its levels, each level
taking a disjoint range of pool slots, so a multiscale run really does put
every shard and every level through a single worker. A plate goes further: it
creates one pool sized for a *single* field of view and hands it to every
field, so fields share slot indices. That is safe only because fields are
written one at a time and the single worker keeps their opens and closes
apart. Nothing tests it.

## Cost of one write at a time

Depth is worth about 1.4x, and the gain is flat past 16. Measured on xfs over
an md RAID10 of 8 NVMe drives, writing 48 MiB at a time, with depth taken
from the number of shard files being written at once. Depth 1 is today's sink.

| writes in flight | GB/s | vs today |
|---:|---:|---:|
| 1 | 12.07 | — |
| 8 | 15.56 | 1.29x |
| 16 | 17.15 | 1.42x |
| 32 | 17.54 | 1.45x |
| 64 | 17.51 | 1.45x |

Today's sink already reaches 69% of this array's ceiling, because the array
splits a 48 MiB write into hundreds of device requests before they reach the
drives. It already gets at the drives some of what depth would give. On a
single drive the gap should be larger. That is where the problem first showed
up.

Every data write is already 2-48 MiB, so there is nothing to combine and
nothing to split. The only small writes are shard footers at 4-8 KiB, one per
file, about 2.6 ms across a whole run.

Issue #178 has the full measurements, including the variation between nodes
that makes any single number unsafe to quote.

## Plan

There are five changes. Each is a pull request off `main`, opened after the
one before merges. The repo squash-merges, so stacking them would cost a
rebase onto a squashed commit at every merge.

### 1. Write-path measurements

Done. No behavior change; counters only.

The main number is **how many distinct shard files have a write waiting at
once**, as a high-water mark and as a time-weighted average. That is the depth
a scheduler could take. Writes in flight is not recorded: it is one by
construction today, so measuring it would restate the problem. If only two or
three files ever have a write waiting, the pool size caps throughput and no
scheduler can reach depth 16.

The counters also cover shard files opened and the most open at once, summed
across levels rather than guessed from the pool size; a request-size histogram
as a regression check; per-request latency; copied against borrowed pending
bytes; and the high-water mark of queued work. `scripts/sweep/README.md` has
the full field list and says what window each covers. Fence stalls were
already recorded as `io_fence_stall` on both the CPU and GPU paths.

Added a `smallepoch_4shards_single` scenario. `smallepoch_single` has one
chunk along each of its inner dimensions, so it produces one shard file and
raising its concurrent-shard target does nothing. The new scenario splits both
inner dimensions in two to give four. Its chunk is the same size, so its epoch
is four times as large. Compare the pair on what the write path did, not on
throughput. The single-shard scenario stays, because it is the only one that
writes to a single growing file and so the only place pre-sizing can be
measured.

### 2. Correctness at one write in flight

Replace closures with described write requests. Give every open file an opaque
token so a late write cannot reach a recycled descriptor. Reserve request and
byte credits before allocating or copying a payload, so the memory limit
holds. Make truncate and close barriers that only run once that file's writes
are done. Retire completions through a sliding window with a watermark, so
`wait` keeps its current meaning. Define what partial, zero-byte, failed, and
cancelled completions do. Keep one worker.

The throttled benchmark sink (`bench/sink_throttled.c`) has an `io_queue` of
its own, where one write at a time is the point. It models a device's
bandwidth by sleeping inside jobs, and it has to keep working.

Rewrite the injection tests against a fake backend that can complete work out
of order, and add cases for the orderings that cannot happen today. Several
existing tests pin the current behavior directly and will have to be rewritten
rather than kept: some assert jobs finish in the order they were posted, some
assert an exact pending-byte total after every post, and several rely on one
blocking job holding up everything behind it, which only works with one
worker.

Two hazards need cases of their own, because nothing covers them now: a plate
reusing one pool slot for two different fields of view, and a fence taken at
the same time as a destroy.

The synchronous failing-truncate hook can stop being a special case. It is
synchronous only because a queued failure would land behind the footer write
and leave no outstanding work for the error path to drain. A fake backend can
hold the footer write and fail the truncate at the same time.

*Done when:* output is byte-identical and failure behavior is unchanged.

### 3. Several writes in flight

Raise the worker count, schedule ready files round-robin, and add pre-sizing
so the per-file limit can go above one. Add the flags that select this and
write them into the results file, so a sweep records what it ran:
`--io-writes-in-flight`, `--io-writes-in-flight-per-file`, `--io-workers`,
`--io-backend`. Sweep depth from 1 to 32.

*Done when:* the reef-l40 XFS matrix passes.

### 4. io_uring on Linux

An io_uring backend behind the same interface, opt-in, falling back to threads
when the kernel or the container will not allow a ring. At chucky's write
sizes one ring reaches the ceiling on its own, and blocking writes reach it
too, so this step needs evidence before it lands.

*Done when:* the XFS matrix passes, then an NFS matrix before it becomes the
default.

### 5. Overlapped writes on Windows

Blocked on hardware. Windows stays on the thread backend until then, and the
rewrite onto completion ports (IOCP) waits with it. Pre-sizing does not carry
over: setting a file's size on NTFS does not extend its valid data length, and
the call that would is privileged and exposes old disk contents.

## Settled decisions

- Registered files: dropped. No measurable difference across three runs.
- Combining or splitting writes: dropped.
- Unbuffered writes: kept. Buffered was worse on both throughput and tail
  latency.
- Pre-sizing uses `ftruncate`, not `fallocate`. The two perform the same at
  chucky's write sizes, `ftruncate` reserves no space so peak disk usage does
  not rise, and `fallocate` is not supported on the NFS mount. The cost is
  that a full disk fails the write instead of the open.

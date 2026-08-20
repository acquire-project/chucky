# Filesystem write scheduling

**Status.** Planning and instrumentation. Issue #178.

The filesystem sink runs one write at a time. Each queued job does a blocking
`pwrite`, so however deep the backlog gets, the drive sees one request. This
file is the working plan for changing that. It is not user-facing
documentation; reference docs are in `docs/`.

## Terms

- **Write in flight** — a write the application has handed to the operating
  system and not yet seen finish.
- **Writes in flight, total** — the number outstanding across every shard file
  at once. This is the number storage cares about, and the number the current
  sink pins at one.
- **Writes in flight per file** — how many of those are to the same shard file.
  Worth nothing above one on a file that is still growing: on xfs a write that
  grows a file takes the file's lock for itself, so the kernel accepts many
  requests and then runs them one after another. Pre-sizing the file is what
  makes this number matter.
- **Pre-sizing** — setting a file's size up front with `ftruncate`, so later
  writes land inside the file rather than extending it. Finalize already
  trims the file back to its real size.
- **File generation** — one open of one shard file. A slot in the pool is
  reused for many shard files over a run, and each reuse is a new generation.
  A write must never reach a generation other than the one it was issued for.

## What is there today

`io_queue` (`src/zarr/io_queue.h`) is a growable ring of closures with one
worker thread. `shard_pool_fs` posts three kinds of job to it: a copying
write, a zero-copy write that borrows pinned memory, and the truncate and
close that finalize a shard file.

Correctness rests on that single worker running jobs in the order they were
posted:

- `pool_fs_open` calls `fs_slot_finalize`, which *queues* a close and marks the
  slot free straight away, then opens the next shard file. Nothing waits for
  that close. With one FIFO worker the close cannot overtake the writes ahead
  of it, and the new file's descriptor cannot collide with the old one because
  the old one is still open. Neither holds once completions are unordered.
- The shard footer is written at `data_cursor`, after the data writes.
- Truncate and close come after every write to that file.
- `pool_fs_flush` records a sequence number and waits for it, which means
  "everything posted so far has finished".

Two more places lean on the same ordering:

- `shard_delivery.c` fences a shard's footer buffer against reuse while its
  zero-copy write is still queued, and fences footer, truncate, and close
  before the array's shape is allowed to name that data. Both are arguments
  about sequence order, nothing else.
- `io_event_wait` gives up as soon as shutdown is set, so a fence taken at the
  same time as a destroy can return before its writes land. Only the worker
  join inside `io_queue_destroy` covers that gap today.

One pool, and therefore one queue and one worker, is created per
`store->create_pool` call. A multiscale array creates one for all of its
levels, each level taking a disjoint range of pool slots, so a multiscale run
really does put every shard and every level through a single worker. A plate
goes further: it creates one pool sized for a *single* field of view and hands
it to every field, so slot indices alias between fields. That is safe only
because fields are written one at a time and the single worker keeps their
opens and closes apart. Nothing tests it.

The throttled benchmark sink (`bench/sink_throttled.c`) has an `io_queue` of
its own, where serializing is the whole point — it models a device's bandwidth
by sleeping inside jobs. It has to keep working.

## Why it costs something

Measured on xfs over an md RAID10 of 8 NVMe drives, 48 MiB writes, depth taken
from the number of shard files being written at once. Depth 1 is today's sink.

| writes in flight | GB/s | vs today |
|---:|---:|---:|
| 1 | 12.07 | — |
| 8 | 15.56 | 1.29x |
| 16 | 17.15 | 1.42x |
| 32 | 17.54 | 1.45x |
| 64 | 17.51 | 1.45x |

Flat past 16. Today's sink already reaches 69% of this array's ceiling,
because a 48 MiB write is split into hundreds of device requests before it
reaches the drives — the array supplies at the device level some of what
depth would supply. On a single drive the gap is expected to be larger, which
is where the problem was first seen.

Every data write is already 2-48 MB, so there is nothing to combine and
nothing to split. The only small writes are shard footers at 4-8 KiB, one per
file, about 2.6 ms across a whole run.

Full measurements, including the node-to-node variation that makes any single
number unsafe to quote on its own, are in issue #178.

## Plan

Five changes, each its own pull request, each branching from `main` after the
one before it merges. The chain is linear and the gate between steps is a
measurement rather than a review, so stacking them would buy nothing and cost
a rebase onto a squashed commit at every merge.

### 1. Measure the write path

No behavior change. Adds the counters needed to say what the sink is actually
doing.

The headline number is **how many distinct shard files have a write waiting at
once**, as a high-water mark and as a time-weighted average. Not how many have
a write in flight: that is one by construction today, so measuring it would
only restate the problem. What is unknown, and what decides whether any of
this is worth building, is how much depth is *available* to take. If only two
or three files ever have a write waiting, the pool size caps throughput and no
scheduler can reach depth 16.

Also: open file generations, summed across levels rather than guessed from the
pool size; request-size histogram, kept as a regression check rather than a
lever; per-request latency; copied against borrowed pending bytes; and the
high-water mark of queued work. Fence stalls are already recorded as
`io_fence_stall` on both the CPU and GPU paths.

Adding a counter does not need a results-schema bump — the version is stamped
on the Python side and only a rename or a change of meaning requires one. But
`scripts/sweep/summary.py` lists the keys the overview page keeps, so a new
counter is dropped from that page unless it is named there. The explorer takes
everything it is not told to drop, so it needs no change.

Adds a `smallepoch_4shards_single` scenario. Today `smallepoch_single` has one
chunk along each of its inner dimensions, so it produces exactly one shard file
and raising its concurrent-shard target does nothing. The new scenario splits
both inner dimensions in two to give four. Its chunk is the same size, so its
epoch is four times as large; the pair is comparable on what the write path did
rather than on throughput. The single-shard scenario stays, because it is the
only one that exercises a single growing file and so the only place pre-sizing
can be measured.

### 2. Correctness at one write in flight

Replace closures with described write requests. Give every open file an opaque
token so a late write cannot reach a recycled descriptor. Reserve request and
byte credits before allocating or copying a payload, so the memory bound is
real. Make truncate and close barriers that only run once that file's writes
are done. Retire completions through a sliding window with a watermark, so
`wait` keeps its current meaning. Define what partial, zero-byte, failed, and
cancelled completions do. Keep one worker.

Rewrite the injection tests against a fake backend that can complete work out
of order, and add cases for the orderings that cannot happen today. Several
existing tests pin the current behavior directly and will have to be rewritten
rather than kept: one asserts jobs finish in the order they were posted, two
assert an exact pending-byte total after every post, and five rely on a single
gate job blocking every job behind it — which only works while there is one
worker.

Two hazards need cases of their own, because nothing covers them now: a plate
reusing one pool slot for two different fields of view, and a fence taken at
the same time as a destroy.

The synchronous failing-truncate hook can stop being a special case. It is
synchronous only because a queued failure would land behind the footer write
and leave no outstanding work for the error path to drain. A fake backend can
hold the footer write and fail the truncate at the same time, which is the
state the test was always trying to describe.

Gate: byte-identical output and unchanged failure behavior.

### 3. Several writes at once

Raise the worker count, schedule ready files round-robin, and add pre-sizing
so the per-file limit can go above one. Adds the flags that select all of
this, written into the results file so a sweep records what it ran:
`--io-writes-in-flight`, `--io-writes-in-flight-per-file`, `--io-workers`,
`--io-backend`. Sweep total writes in flight from 1 to
32; the curve is flat past 16 on the array measured above.

Gate: the reef-l40 XFS matrix.

### 4. io_uring on Linux

Behind the same backend interface, opt-in, falling back to threads when the
kernel or the container will not allow a ring. At chucky's write sizes one
ring reaches the ceiling on its own, and blocking writes reach it too, so this
step has to earn its place on evidence.

Gate: the XFS matrix, then an NFS smoke matrix before it becomes the default.

### 5. IOCP on Windows

Blocked on hardware. Windows stays on the thread backend until then. Pre-sizing
does not carry over unchanged: setting a file's size on NTFS does not extend
its valid data length, and the call that would is privileged and exposes
whatever was on the disk before.

## Decisions already made

- Registered files: dropped. No measurable difference across three runs.
- Combining or splitting writes: dropped. The sizes are already right.
- Unbuffered writes: kept. Buffered was worse on both throughput and tail
  latency.
- Pre-sizing uses `ftruncate`, not `fallocate`. The two perform the same at
  chucky's write sizes, `ftruncate` reserves no space so peak disk usage does
  not rise, and `fallocate` is not supported on the NFS mount. The cost is
  that a full disk surfaces mid-write instead of at open.

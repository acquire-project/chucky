# Filesystem write scheduling

**Status.** Steps 1 to 3 are done. Steps 4 and 5 are planned. Issue #178.

Only one write at a time was run by the filesystem sink: each queued job was a
blocking `pwrite`, so however deep the backlog, only one request reached the
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

## The write path this started from

Steps 2 and 3 have replaced most of this. It is kept because the rest of the
plan reads against it.

`io_queue` (`src/zarr/io_queue.h`) was a growable ring buffer of closures with
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

One pool — one queue — per `store->create_pool` call. One for all
levels of a multiscale array, each level given a disjoint range of slots, so
every shard and level is put through a single worker. Worse for a plate: one
pool sized for a *single* field of view, handed to every field, so slot indices
are shared — safe only because fields are written one at a time and their opens
and closes are kept apart by the single worker. Untested.

## Cost of one write at a time

About 1.4x from more queue depth, flat past 16. Measured on xfs over an md
RAID10 of 8 NVMe drives, 48 MiB writes, queue depth taken from shard files
written at once. One in flight is the sink this started from.

| queue depth | GB/s | vs one |
|---:|---:|---:|
| 1 | 12.07 | — |
| 8 | 15.56 | 1.29x |
| 16 | 17.15 | 1.42x |
| 32 | 17.54 | 1.45x |
| 64 | 17.51 | 1.45x |

The one-write-at-a-time sink was already at 69% of this array's ceiling: a 48 MiB write is split
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

- Described write requests instead of closures.
- An opaque token on every open file, so no late write can be applied to a
  recycled descriptor.
- Request and byte credits reserved before a payload is allocated or copied.
  The byte ceiling starts unlimited: there is no memory limit today to keep, and
  step 3 picks a number from sweep data.
- Truncate and close as barriers, run only once that file's writes are done.
- Completions retired through a sliding window with a watermark, so the
  meaning of `wait` is unchanged.
- Defined behavior for partial, zero-byte, failed, and cancelled completions.
- Still one worker.

Keep `bench/sink_throttled.c` working: a second `io_queue` where one write at a
time is the point, bandwidth modeled by sleeping inside jobs.

Rewrite the injection tests against a fake backend able to complete out of
order, and add cases for the orderings impossible today. Several are pinned to
today's behavior and must be rewritten: jobs finishing in the order posted; an
exact pending-byte total after every post; one blocking job holding up
everything behind it — only possible with one worker.

One uncovered hazard above still needs a case: a fence at the same time as a
destroy. The plate's shared pool slot moved to #211, whose fix changes how slots
are allocated.

The synchronous failing-truncate hook stays. A fake backend does not replace it.
`cpu_stream_flush_body` checks for an error before it finalizes the shards and
never drains afterwards, so a flush reports a truncate failure only when
`finalize_shards` returns non-zero. Make the truncate asynchronous and the flush
reports success, then publishes a fence and an append extent for a shard whose
size is wrong. The fake backend covers the queue-level case instead: a failed
truncate still lets the close run.

*Done when:* output is byte-identical and failure behavior is unchanged.

### 3. More queue depth

Done.

- Several workers per queue, with a ceiling on requests in flight across every
  file and a second ceiling per file.
- Files take turns, so a backlog on one shard file cannot keep the others
  idle. The dispatch scan is gone: each file keeps its requests on a list of
  their own, and readiness is one comparison against the head of that list
  plus a flag for a barrier already running.
- The open-file table is indexed by the token's file index rather than
  searched, and the table `io_queue_stats.c` kept alongside it is merged into
  it. A backend hands an index back when the close naming it runs, which is
  before that close retires, so a new file naming the same index takes the
  entry over and the old close retires against a generation that no longer
  matches.
- Shard files are pre-sized with `ftruncate` when more than one write per file
  may run. The size is every chunk at its worst-case compressed size plus the
  footer, and `finalize` already trims it back. Not on Windows, where only
  writing moves a file's valid data length, so `platform_presize_helps`
  answers no there and the step is skipped.
- `--io-workers`, `--io-writes-in-flight`, `--io-writes-in-flight-per-file` and
  `--io-backend` select all of it, and the results file records what the run
  resolved them to. New sweep tier `iodepth` covers 1 to 32 writes in flight,
  one and four per file.
- Defaults: eight workers, eight writes in flight, four per file. The byte
  credit is 2 GiB, above the deepest backlog measured — 1392 MiB, from an
  uncompressed 256cube run with one write at a time.

*Done:* the reef-l40 XFS matrix is green, 72 of 72, in
`bench/results/reef-l40-130fda8-20260823.json`. The node writes its shards to
a local `/tmp` that is xfs over an md array.

Output throughput, GiB/s, against writes in flight:

| scenario | codec | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---:|---:|---:|---:|---:|---:|
| orca2_single | none | 2.16 | 2.42 | 2.54 | 2.56 | 2.47 | 2.42 |
| smallepoch_4shards | none | 0.52 | 0.56 | 0.59 | 0.60 | 0.57 | 0.58 |
| smallepoch_single | none | 0.59 | 0.58 | 0.56 | 0.59 | 0.54 | 0.57 |
| orca2_single | zstd | 0.36 | 0.37 | 0.37 | 0.37 | 0.37 | 0.36 |

Read these against the same node, not against the table further up. That
table is an md RAID10 of 8 drives, on a `cpu-turin-gp-l` node. An L40 node's
`/tmp` is an md **RAID1 mirror of two drives**, so a write goes to both
members and there is no striping to gain from. Neither is encrypted; the
difference is drive count and array shape. `dev/io-depth/io_depth` on the L40
node, 48 MiB writes over 8 files, one per file:

| writes in flight | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| GB/s | 2.24 | 1.49 | 2.27 | 2.59 | 2.43 | 2.11 |

So the node tops out near 2.6 GB/s and peaks at eight, not sixteen.

**The sink reaches that ceiling.** `orca2_single` uncompressed writes 2.79 GB/s
at eight in flight, against the 2.59 GB/s the microbenchmark gets on the same
node, and the queue behind it holds a 1.1 GiB backlog with writes waiting
209 ms before they start. Nothing upstream is holding the write path back
there; the drive is the limit, and the 1.2x is the whole of what this node has
to give. The microbenchmark's own gain over the same range is 1.16x.

A compressed run does not move. Compression is the ceiling there, so the write
path is not what is being waited on.

Queue depth per file does not show up in the sweep — 1.21x against 1.19x on
`orca2_single`, inside the spread. That is the expected answer rather than a
disappointing one: about twenty shard files have a write waiting at once, so
the depth is already there and taking it per file adds nothing. The scenario
where per-file depth is the only lever is `smallepoch_single`, and its runs
last 0.05 s and queue 16 MiB, which is too little to resolve anything.

The microbenchmark can resolve it. One file, four workers:

| | GB/s |
|---|---:|
| one write per file | 1.98 |
| four per file, no pre-sizing | 2.10 |
| four per file, pre-sized | 2.18 |

Pre-sizing is worth about 10% on a file being written by itself, which is why
it stays on by default even though no bench scenario is shaped to show it.

### On the eight-drive array

The tier carries a CPU-backend arm so it can run where the fast drive is. On
`cpu-turin-gp-l-243-241`, an md RAID10 of eight drives, `--backend cpu`,
`bench/results/turin-raid10-ab4b2f0-20260824.json`:

| scenario | codec | 1 | 2 | 4 | 8 | 16 | 32 | best/1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| orca2_single, GB/s written | none | 6.12 | 6.89 | 7.84 | 8.22 | 8.08 | 7.90 | 1.34x |

**1.34x, against 1.21x on the mirror.** A faster drive is worth measuring on:
the gain is larger and it peaks in the same place.

It also moves the limit. At one write in flight the queue holds a 1 GiB
backlog and writes wait 59 ms; by sixteen the wait is 6 ms and by thirty-two
it is 0.4 ms, so the queue is draining as fast as it fills. The drive is no
longer what is waited on — the rest of the pipeline is, at about 8 GB/s
against the array's 16.4. That is where the next write-path work has to look.

Both sweeps peak at **eight** writes in flight and fall back past it, which is
why the defaults are eight rather than the sixteen the microbenchmark table
suggests. The microbenchmark keeps improving to thirty-two because it has no
pipeline in front of it.

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

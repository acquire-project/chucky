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
- **Depth allowed** — the cap `--io-writes-in-flight` sets on queue depth, and
  `--io-writes-in-flight-per-file` on queue depth per file. The two are a
  minimum, not a product: four per file across sixteen files still runs only
  as many at once as the first cap allows.
- **Depth reached** — the depth a run actually got to, recorded as
  `io_writes_in_flight_mean`. Never assume it equals the depth allowed.
- **Ceiling** — a drive's maximum rate, never a setting.
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

One pool — one queue — per `store->create_pool` call. One for all
levels of a multiscale array, each level given a disjoint range of slots, so
every shard and level is put through a single worker. Worse for a plate: one
pool sized for a *single* field of view, handed to every field, so slot indices
are shared and two fields open at once write into each other's files. That is
#211, and its fix changes how slots are allocated. Untested either way: no test
in the tree writes array data through a plate field sink.

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

Done.

- Several workers per queue, with a cap on queue depth and a second on queue
  depth per file.
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
  writing moves a file's valid data length, so `platform_should_presize_shard`
  answers no there and the step is skipped.
- `--io-workers`, `--io-writes-in-flight`, `--io-writes-in-flight-per-file` and
  `--io-backend` select all of it, and the results file records what the run
  resolved them to. New sweep tier `iodepth` covers depth allowed from 1 to 32,
  at one and four per file.
- Defaults: eight workers, depth allowed eight, four per file. The byte
  credit is 2 GiB, above the deepest backlog measured — 1392 MiB, from an
  uncompressed 256cube run with one write at a time.

*Done:* the reef-l40 XFS matrix is green, 72 of 72, in
`bench/results/reef-l40-130fda8-20260823.json`. The node writes its shards to
a local `/tmp` that is xfs over an md array.

Output throughput, GiB/s, against depth allowed:

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

| depth allowed | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| GB/s | 2.24 | 1.49 | 2.27 | 2.59 | 2.43 | 2.11 |

Those last three rows all reached depth 7.8, not 16 and 32: eight files at one
write per file cannot go deeper. Read the corrected curve below instead.

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

**1.34x, against 1.21x on the mirror.** A faster drive is worth measuring on.

It also moves the limit. At one write in flight the queue holds a 1 GiB
backlog and writes wait 59 ms; by thirty-two the wait is 0.4 ms, so the queue
drains as fast as it fills and the drive is no longer what is waited on.

### Depth allowed is not depth reached

`io_writes_in_flight_mean` records the depth a run actually reached, because
the cap it was given is no evidence it got near it. Two things only show up
once that is read.

The array rewards depth past eight, and the earlier claim here that it did not
was an artifact: a depth sweep over 8 files at one write per file cannot
exceed depth 8 however many workers it is given. Over 64 files the achieved
depth tracks the setting, and the drive behaves as the table at the top of
this file says:

| depth asked | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| depth reached | 1.00 | 2.00 | 4.00 | 7.99 | 15.93 | 31.39 | 58.16 |
| GB/s | 11.77 | 13.44 | 14.23 | 16.38 | 17.52 | 17.56 | 17.50 |

**The sink cannot use that depth.** `orca2_single` writes about sixteen shard
files at a time, and with one write per file the depth *is* the count of those
holding queued work. Allowed thirty-two, it reaches 11.8 while 11.7 files hold
work — the same number. Four per file adds only 0.4 on top, because with a
dozen files busy each is handed about one write anyway. Allowing more than
about ten is allowance nothing claims.

So the defaults are eight, and the reason is not the drive. `orca2_single` on
the eight-drive array at 4000 frames, three repeats, spread under 1%:

| depth allowed | depth reached | GB/s |
|---:|---:|---:|
| 4 | 3.90 | 11.71 |
| 8 | 6.84 | **12.47** |
| 16 | 10.55 | 11.65 |
| 32 | 10.10 | 11.54 |

**Use a run of several seconds for this.** At the scenario's own 200 frames
the run lasts half a second, nearly all of it startup and the closing flush,
and two passes of the tier then disagree about whether eight or sixteen is
faster. That is why the tier overrides the frame count.

More depth is slower, and why is not settled. It is not the worker threads
taking cores from the pipeline: holding the pipeline to 32 threads on a
96-core node, leaving 64 idle, eight still beats sixteen by 9%. What does move
with depth is how long a single write takes — `io_run_ms_mean` goes from 10.0
ms at eight to 18.4 at sixteen while the sink's rate does not improve — and
the producer waits on io, so it feels a write's latency rather than the
aggregate rate. Which wait is not settled. The one fence the CPU path times
reads 0.01 ms over seven calls, so it is not that one, and the producer's
other waits on io are timed inside the `sink` stage rather than reported on
their own.

The gap left is the pipeline: 11.4 GB/s against the 16.4 the array gives at
the same depth. Since depth follows the count of shard files holding work,
writing more shard files at once is the lever that would move it, not a
larger cap.

### The shared file server

`/bio`, which is `/mnt/main0`, is an NFS version 3 mount, one mebibyte per
request, sixteen TCP connections (`nconnect=16`), over an NVMe store. Every
login and compute node has it. A file written on a compute node was
checksummed from a second machine, so the bytes do reach the server.

**A node caps near 8.8 GB/s, and the server does not.** One point run by one,
two and four nodes at once totals 7.4, 12.3 and 25.0 GB/s. The limit belongs
to a node, not to the file server: 8.8 GB/s is about 70 Gb/s, which is roughly
what one of a node's 100 Gb/s ports carries in practice.

From one node, 8 MiB writes over 256 files, unbuffered:

| writes in flight | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| GB/s | 8.26 | 7.88 | 8.76 | 7.90 |
| median write, ms | 17 | 24 | 79 | 147 |

Thirty-two is the knee, and past it only the latency grows. Buffered writes
are no faster and cost ten times the system time — 24 s against 1.4 s for
32 GiB — so the shipping path's O_DIRECT is right here too. Request size
matters up to about 8 MiB: at sixteen writes in flight, 1 MiB gives 3.14 GB/s
and 8 MiB gives 7.20, flat above that.

One write costs about ten times what it costs on a local drive, 51 ms for
48 MiB against 5 ms on the eight-drive array, so throughput comes from depth
alone. The default of eight writes in flight is chosen for a local array and
is too low for this mount.

The server is shared with the rest of the cluster. Two repeats of one point
differ by up to 40%, so a single reading is an estimate.

**The sink reaches about half of what a node can.** `orca2_single`
uncompressed writes 3.9 to 4.8 GB/s to the mount, and eight of those streams
at once on one node total 8.78 — the node's own ceiling. The shortfall is in
one stream rather than in the mount or the server.

| streams at once | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| total GB/s | 3.86 | 5.41 | 6.78 | **8.78** |
| depth reached, each | 12.6 | 10.3 | 9.5 | 8.3 |

The queue is starved rather than backed up: it waits 0.4 ms and reaches depth
12.6, while the mount is fastest at about thirty-two writes at a time. A
higher cap helps up to sixteen — 2.7 GB/s at four against 4.8 at sixteen — and
then flattens, because the depth reached stays near twelve however high the
cap goes. What is missing is writes the producer never posts, and that is the
same pipeline limit the eight-drive array shows, widened by the mount's higher
latency.

**The faster target depends on the node.** An L40 node's mirror gives 1.6 to
2.8 GB/s, so the file server is three times faster there. A `cpu-turin-gp-l`
node's array gives 17.4 GB/s, more than twice the file server.
`orca2_single` uncompressed, 2000 frames, two repeats, each pair measured in
one allocation:

| node | target | best GB/s | at |
|---|---|---:|---|
| L40 | local mirror | 2.19 | 4 in flight |
| L40 | file server | 5.12 | 32 in flight, 4 per file |
| turin | local array | 7.10 | 16 in flight |
| turin | file server | 4.77 | 16 in flight |

Compare the pairs with each other and not with the 4000-frame table above:
2000 frames is short enough to read low, as that section says.

**Writes to one file are serialized.** On a single file, depth adds latency in
proportion and no throughput — 1.05 GB/s at one write in flight against 1.35
at thirty-two, while the median write goes from 47 ms to 1204. Pre-sizing
cannot help there, because there is no extending-write lock to avoid. Per-file
depth still helps `orca2_single`, but only by raising the total across its
sixteen shard files.

### 4. io_uring on Linux

An io_uring backend behind the same interface, opt-in, with a fallback to
threads when a ring is not allowed by the kernel or the container. At chucky's
write sizes the ceiling is reached by one ring alone, and by blocking writes
too, so evidence is needed first.

**Close four gaps in the backend interface** (#229), as its own pull request
before any ring is written. The interface step 2 built is general enough to hold
one, and these four were what it was short of. A backend with no room now
answers "not now" instead of blocking a worker or failing the request, and the
queue offers that request again with its place in line kept. A backend can carry
a stop, called once at teardown, so one running a thread of its own has
somewhere to release it. The request handed to `execute` is the queue's own copy
and outlives the call, so a backend that finishes later reads it where it lies
rather than copying what it needs. And a short write is the backend's to finish:
`platform_pwrite` loops internally so it never reports one, a ring does, and the
part still to be done is handed back for retrying.

*Done when:* the XFS matrix is green, then an NFS matrix before it is made the
default. The file server's own numbers are above, so that matrix has a
baseline to be read against.

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

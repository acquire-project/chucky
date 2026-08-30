# Shard tail carry-over

## What this is for

Chucky streams compressed image data into [zarr v3 sharded
files](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html).
For throughput on local SSDs we open shard files with `O_DIRECT` (Linux),
`F_NOCACHE` (macOS), or `FILE_FLAG_NO_BUFFERING` (Windows). Unbuffered IO has a
hard constraint: every write's *offset*, *length*, and *source pointer* must be
multiples of the device's page alignment.

We also want the on-disk shard file to have no inter-batch padding zeros: the
shard's index records each chunk's actual byte offset and size, and any slack
inside the file is wasted bytes.

Compressed chunks have variable size, so they don't naturally land on page
boundaries. The "tail carry-over" mechanism reconciles these two demands by
deferring the sub-page tail of each batch and prepending it to the next batch.

## Concepts

| term                | meaning                                                                                  |
|---------------------|------------------------------------------------------------------------------------------|
| **chunk**           | independently compressed unit; the smallest thing the codec produces.                    |
| **shard**           | one file holding many chunks plus a `[offset, size]` index at the end.                   |
| **append dim**      | leftmost dimension(s) that grow over time (e.g. T, Z). Data streams in along these.      |
| **shard generation**| span of `chunks_per_shard_append` epochs along the append dim that fills one shard file. |
| **batch**           | the streaming unit; the pipeline processes N epochs end-to-end per batch.                |
| **page alignment**  | the device's required alignment for unbuffered IO (typically 4 KiB).                     |
| **leading tail**    | sub-page bytes from the prior batch we couldn't write yet — they prefix the next batch.  |

## The data path

A batch flows through four logical stages on the GPU:

1. **Compute** (`src/{cpu,gpu}/`): scatters input voxels into per-LOD chunk
   pools, compresses each chunk, and emits one variable-size compressed blob
   per chunk.
2. **Aggregate** (`src/gpu/aggregate.cu`): copies real chunks into one compact,
   tail-free device buffer in shard-major delivery order. Chunk offsets are
   absolute within that compact buffer; there is no page padding or reserved
   per-shard capacity on the device.
3. **Materialize** (`src/gpu/d2h.materializer.c` and
   `src/zarr/shard_delivery.c`): after the preceding batch has committed its
   host tail, partitions the batch into physical-shard generation runs. Each
   run gets a page-aligned region in the pinned host slot, its committed tail
   is copied to the front, and the run's compact payload is copied from the
   device immediately after it.
4. **Deliver** (`src/zarr/shard_delivery.c`): walks those host runs
   and calls the active shard writer's `write_direct` (zero-copy reference
   to caller memory) or `write` (copying bounce). The writer queues async
   pwrite jobs onto a per-pool io_queue.

Aggregation can run ahead of delivery. Indexed codecs also copy their
offset/size metadata as soon as aggregation completes, but payload
materialization and sink delivery remain generation-ordered because the
preceding delivery establishes the only authoritative tail state.

## Device and host aggregate buffers

`page = sink->required_shard_alignment()` (zero means buffered IO; the rest of
this doc assumes `page > 0`). GPU slots have separate capacities. The device
allocation contains at most `real_chunk_count * max_comp_chunk_bytes` and
holds only compact payload. The pinned-host allocation is larger because each
possible physical run needs independent alignment and room for a carried
prefix smaller than one page.

Every non-empty physical run produces one payload D2H copy into a layout of
the following form:

```
pinned_slot
  physical run 0 (page-aligned): [tail_in || payload || slack]
  physical run 1 (page-aligned): [tail_in || payload || slack]
  ...
```

Fixed-size output builds its host offsets and sizes from the existing
permutation geometry. Variable-size output first lands the device-generated
offset/size arrays and then resolves exact run source ranges. All payload
copies are enqueued before one batch-ready event and one host wait.

### CPU legacy aggregate buffer

The CPU aggregator and `deliver_to_shards_batch` retain the older
shard-capacity layout below. The host-run materializer is currently GPU-only;
this split keeps CPU output unchanged.

```
ws->data
├── LOD 0 segment (page-aligned)
│   ├── shard 0 region (shard_capacity bytes, page-aligned)
│   │   ├── [0, tail_in)                    leading tail (from prior batch)
│   │   ├── [tail_in, tail_in + run_real)   chunks for this batch
│   │   └── slack to shard_capacity
│   ├── shard 1 region
│   └── ...
├── LOD 1 segment
└── ...
```

`shard_capacity = align_up(active_count_max * cps_inner * max_comp_chunk_bytes
+ page, page)`. The `+ page` reserves room for a possible leading tail
(`< page` bytes); the rest is the worst-case real chunk bytes for one batch.

Inside one CPU batch, chunks pack tightly per shard. Multiple shard generations
that happen to fall inside the same batch are also contiguous — the
aggregator does **no per-generation padding**. Generation crossings are
handled at delivery time.

## Per-shard live state

Per-LOD `struct shard_state` owns two contiguous pools, each
`shard_inner_count` slots wide:

| pool                      | purpose                                                  |
|---------------------------|----------------------------------------------------------|
| `tail_buf_pool`           | sub-page carry-over bytes between batches (one per shard)|
| `footer_buf_pool`         | shard-footer scratch (one per shard, page-aligned)       |

Each `struct active_shard` is one inner shard's slice into those pools, plus
its own per-shard state:

| field            | purpose                                                                       |
|------------------|-------------------------------------------------------------------------------|
| `writer`         | open shard-file handle from `sink->open(...)`; `NULL` between generations     |
| `data_cursor`    | next file offset to write at                                                  |
| `index`          | `[chunks_per_shard_total][2]` `(offset, size)` pairs, written at finalize     |
| `tail_buf`       | slice into `tail_buf_pool`; valid bytes in `tail_bytes` (always `< page`)     |
| `footer_buf`     | slice into `footer_buf_pool`; capacity `footer_capacity`, built lazily        |
| `footer_io_done` | io fence for `footer_buf`; wait before refill, record after every `write_direct` |

`footer_capacity = align_up(page + chunks_per_shard_total*16 + 4, page)`:
one page for the trailing sub-page data, the index (16 bytes per chunk), the
4-byte CRC, padded.

## Delivery: runs and their outcomes

Both delivery paths walk a batch in **runs**. A run is a contiguous span of
epochs that fits inside the current shard generation:

```
run_len       = min(remaining_in_shard, remaining_in_batch)
run_finalizes = (run_len == remaining_in_shard)
```

For the GPU, `host_batch_build_compact` creates one `host_batch_run` and one
page-aligned host region for each `(inner shard, generation run)` pair. The
run records its compact device source origin, committed tail prefix, and
rebased chunk views. The CPU's unchanged `deliver_to_shards_batch` instead
finds the run in its shard-capacity aggregate region.

There are three run outcomes.

### Non-finalizing run (batch ends mid-generation)

The run produces some bytes for shard si; more of the same generation will
arrive in a future batch.

1. `total_run = tail_in + run_real`; `tail_in` is the leading tail from the
   prior batch and is non-zero only on the first run for this shard in this
   batch.
2. `write_bytes = (total_run / page) * page` — page-aligned floor.
3. Write `[src, src + write_bytes)`. GPU host runs are independently aligned,
   so the normal filesystem path can use `write_direct`; sinks without that
   operation use `write`.
4. Save the `< page` remainder into `sh->tail_buf` for the next batch.
5. Advance `sh->data_cursor` by `write_bytes`.

### Finalizing run (this run completes the generation)

The shard's last data goes here; we close out by writing the **footer** —
the page-aligned `[trailing<page || index || CRC || zero-pad]` blob. This is
**two-step** when `total_run >= page`:

1. **Page-floor write**. Same as non-finalizing: write `(total_run / page) *
   page` bytes via `write_direct` from the agg buffer. Advance `data_cursor`.
2. **Footer write**. `wait_fence(sh->footer_io_done)` so any prior reuser of
   `sh->footer_buf` has retired, build the footer into it, `write_direct` it
   at the new `data_cursor`, then `record_fence` back into `footer_io_done`.
3. **Truncate** the file to `data_cursor + logical_bytes` to drop the
   trailing zero-pad on disk.
4. **Finalize** (close).
5. Reset per-shard state: `writer = NULL`, `data_cursor = 0`, `tail_bytes = 0`,
   `index` cleared.

When `total_run < page`, step 1 is skipped and the footer includes all of
`total_run` as the leading "trailing" portion.

### Intra-batch fresh-gen run

When `epochs_per_batch > chunks_per_shard_append`, one batch can finalize the
same shard slot multiple times — finalize gen N, open gen N+1, possibly
finalize gen N+1 too. The footer write for each finalize uses the same
`sh->footer_buf`; the per-shard fence (`footer_io_done`) makes the reuse
safe regardless of how many finalizes happen in a batch.

On the GPU, the non-finalizing run that follows an intra-batch finalize gets a
new page-aligned host region and therefore an additional D2H span. The legacy
CPU aggregate can still place that run at a non-page-aligned position and use
the copying `write` path.

## Carrying tails across batches

**Same generation, next batch.** The non-finalizing run left
`sh->tail_bytes > 0` and the bytes saved in `sh->tail_buf`. When that next
batch reaches ordered GPU materialization, the planner copies the committed
host tail into the front of the run's aligned pinned region and enqueues the
fresh compact device payload at `region_base + tail_bytes`. Delivery sees one
contiguous `[tail || fresh chunks]` region. No tail length or content is sent
to the GPU. The CPU path performs the equivalent prefix assembly in its
legacy aggregate layout. Every `write_bytes` write lands at a `data_cursor`
that is a multiple of `page`, so no inter-batch padding lands on disk.

**Across generation boundary, fresh batch.** After the prior batch finalized
a generation, the next batch starts with `bytes_consumed[si] = 0`,
`writer = NULL`, `tail_bytes = 0`. The next run opens a new shard file via
`sink->open(...)` and writes from a page-aligned source — including the
footer later. This is the "ideal" steady-state path.

## End-of-stream finalize

`finalize_shards` runs when the writer is flushed and any shard is still
open (a partial generation at the end of the stream). It writes each
remaining shard's footer through the same `write_footer` helper used by
delivery — `wait_fence(footer_io_done)`, build, `write_direct`,
`record_fence` — then truncates, closes, and resets state.

## Async, lifetime, and the io_queue

`write_direct` (`fs_slot_write_direct` → `pwrite_ref_job`) stores **only a
pointer** to caller memory and a target offset. The io_queue worker reads
that memory at job-execute time, which can be much later than queue time.
That's safe only as long as the source memory stays alive and unchanged.
Two lifetime mechanisms cover this:

- **Pinned host run buffer** — the materialization slot is double-buffered.
  Each batch records
  a fence (`sink->record_fence`) after queueing all of its writes; the next
  batch waits on that fence (`sink->wait_fence`) before reusing the slot.
  The new remainder is copied into persistent tail storage before that slot
  can be released, and pinned-slot pointers stay valid across async IO.
- **Footer buffer** — every `active_shard` carries its own `footer_io_done`
  event. `write_footer` waits on it before refilling `footer_buf` and
  records a fresh event after each `write_direct`. Reuse is therefore
  bounded to "after the prior IO retired", regardless of how many batches or
  finalizes happen in between.

`write` (`pwrite_job`) avoids the lifetime question entirely by malloc'ing a
new buffer and copying the source bytes into it. The job carries its own
memory; the caller can release the source immediately. Sinks that don't
provide `write_direct` (e.g. S3) take this path for the footer write.

## GPU materialization and scheduling notes

GPU aggregation performs one size permutation, one exclusive scan, and one
compact gather. It has no shard-bias kernel, device tail buffers, or tail H2D
upload. Fixed output synthesizes the same compact index on the host; indexed
output copies the scan metadata before planning payload spans.

Compression and aggregation for the next slot may run while an older batch is
being delivered. Payload D2H and sink delivery are ordered by batch
generation, so the materializer reads only committed `shard_state`. A D2H or
sink failure makes delivery sticky-failed: later materializations are
cancelled and their slot leases retired rather than committing more shard
state.

## Where to look in the code

| concept                          | file                                  |
|----------------------------------|---------------------------------------|
| run planning and delivery        | `src/zarr/shard_delivery.c`           |
| `shard_state` / `active_shard`   | `src/zarr/shard_delivery.h`           |
| CPU aggregator                   | `src/cpu/aggregate.c`                 |
| compact GPU aggregator           | `src/gpu/aggregate.cu`                |
| D2H materializer lifecycle       | `src/gpu/d2h.materializer.{h,c}`      |
| scheduling / failure ordering    | `src/gpu/schedule.c`                  |
| compact and legacy layouts       | `src/stream/types.aggregate.{h,c}`    |
| FS shard pool, `pwrite_ref_job`  | `src/zarr/shard_pool_fs.c`            |
| O_DIRECT alignment watchdog      | `fs_slot_write` in `shard_pool_fs.c`  |
| writer interface (`shard_writer`)| `src/writer.h`                        |

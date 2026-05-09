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

A batch flows through three stages:

1. **Compute** (`src/{cpu,gpu}/`): scatters input voxels into per-LOD chunk
   pools, compresses each chunk, and emits one variable-size compressed blob
   per chunk.
2. **Aggregate** (`src/{cpu,gpu}/aggregate.*`): copies the scattered
   compressed chunks into one contiguous workspace buffer, packed per shard
   in delivery order. The buffer is page-aligned at the base; each shard
   gets its own region.
3. **Deliver** (`src/zarr/shard_delivery.c`): walks the aggregated buffer
   and calls the active shard writer's `write_direct` (zero-copy reference
   to caller memory) or `write` (copying bounce). The writer queues async
   pwrite jobs onto a per-pool io_queue.

Each stage hands off via simple POD structs (`struct aggregate_result`,
`struct active_shard`); no hidden state.

## The aggregate buffer

`page = sink->required_shard_alignment()` (zero means buffered IO; the rest of
this doc assumes `page > 0`). The aggregator lays out its workspace
hierarchically: per-LOD segments, each containing per-shard regions.

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

Inside one batch, chunks pack tightly per shard. Multiple shard generations
that happen to fall inside the same batch are also contiguous — the
aggregator does **no per-generation padding**. Generation crossings are
handled at delivery time.

## Per-shard live state

Per-LOD `struct shard_state` owns two contiguous pools, each
`shard_inner_count` slots wide:

| pool                      | purpose                                                  |
|---------------------------|----------------------------------------------------------|
| `tail_buf_pool`           | sub-page carry-over bytes between batches (one per shard)|
| `bundle_buf_pool`         | finalize bundle scratch (one per shard, page-aligned)    |

Each `struct active_shard` is one inner shard's slice into those pools, plus
its own per-shard state:

| field         | purpose                                                                       |
|---------------|-------------------------------------------------------------------------------|
| `writer`      | open shard-file handle from `sink->open(...)`; `NULL` between generations     |
| `data_cursor` | next file offset to write at                                                  |
| `index`       | `[chunks_per_shard_total][2]` `(offset, size)` pairs, written at finalize     |
| `tail_buf`    | slice into `tail_buf_pool`; valid bytes in `tail_bytes` (always `< page`)     |
| `bundle_buf`  | slice into `bundle_buf_pool`; capacity `bundle_capacity`, content built lazily|

`bundle_capacity = align_up(page + chunks_per_shard_total*16 + 4, page)`:
one page for the trailing sub-page data, the index (16 bytes per chunk), the
4-byte CRC, padded.

## Delivery: runs and their outcomes

`deliver_to_shards_batch` walks the batch in **runs**. A run is a contiguous
span of epochs that fits inside the current shard generation:

```
run_len       = min(remaining_in_shard, remaining_in_batch)
run_finalizes = (run_len == remaining_in_shard)
```

For each `(si, run)` pair the aggregator has already laid the run's chunk
bytes at `result->data + si * shard_capacity + bytes_consumed[si]`, where
`bytes_consumed[si]` accumulates across this batch's prior runs for shard si.

There are three run paths.

### Non-finalizing run (batch ends mid-generation)

The run produces some bytes for shard si; more of the same generation will
arrive in a future batch.

1. `total_run = tail_in + run_real`; `tail_in` is the leading tail from the
   prior batch and is non-zero only on the first run for this shard in this
   batch.
2. `write_bytes = (total_run / page) * page` — page-aligned floor.
3. Write `[src, src + write_bytes)`. `write_direct` if `src` is page-aligned
   (the common case), else `write` (bounce-copy).
4. Save the `< page` remainder into `sh->tail_buf` for the next batch.
5. Advance `sh->data_cursor` by `write_bytes`.

### Finalizing run (this run completes the generation)

The shard's last data goes here; we close out by writing the index. This is
**two-step** when `total_run >= page`:

1. **Page-floor write**. Same as non-finalizing: write `(total_run / page) *
   page` bytes via `write_direct` from the agg buffer. Advance `data_cursor`.
2. **Bundle write**. Build `[trailing<page || index || CRC || zero-pad-to-page]`
   into `sh->bundle_buf`, write it via `write_direct` from the bundle slot at
   the new `data_cursor`. The bundle slot lives with the stream, so
   `write_direct`'s pointer is safe across the async write.
3. **Truncate** the file to `data_cursor + logical_bytes` to drop the
   trailing zero-pad on disk.
4. **Finalize** (close).
5. Reset per-shard state: `writer = NULL`, `data_cursor = 0`, `tail_bytes = 0`,
   `index` cleared.

When `total_run < page`, step 1 is skipped and the bundle includes all of
`total_run` as the leading "trailing" portion.

### Intra-batch finalize (same shard finalizes twice in one batch)

Triggered when `epochs_per_batch > chunks_per_shard_append`. The first
finalize for shard si in this batch follows the path above and uses
`sh->bundle_buf` for `write_direct` (zero-copy). The **second** finalize must
not reuse `bundle_buf` directly: the prior finalize's `pwrite_ref_job` may
still be sitting in the io_queue holding a pointer to the same bytes.
Overwriting them races the worker.

So the second-and-later intra-batch finalize for the same shard
**bounce-copies** the bundle through a malloc'd buffer (`write`, not
`write_direct`). The caller signals this via `is_first_run_for_shard`
(equivalent to `bytes_consumed[si] == 0`).

The *fresh-gen non-finalizing run* that follows an intra-batch finalize also
bounces, because its source pointer in the agg buffer is mid-shard, not
page-aligned.

How often does this fire? It depends on
`epochs_per_batch / chunks_per_shard_append`. Workloads that size batches
at or below one generation never hit this path.

## Carrying tails across batches

**Same generation, next batch.** The non-finalizing run left
`sh->tail_bytes > 0` and the bytes saved in `sh->tail_buf`. On the next
batch, the aggregator anchors shard si's first chunk at
`shard_base + tail_in` and copies `sh->tail_buf` into
`[shard_base, shard_base + tail_in)` (CPU memcpy or
`copy_leading_tail_k` on GPU). Delivery then sees one contiguous
`[tail || fresh chunks]` region whose source is `shard_base` — page-aligned.
Every `write_bytes` write lands at a `data_cursor` that is a multiple of
`page`, so no inter-batch padding ever lands on disk.

**Across generation boundary, fresh batch.** After the prior batch finalized
a generation, the next batch starts with `bytes_consumed[si] = 0`,
`writer = NULL`, `tail_bytes = 0`. The next run opens a new shard file via
`sink->open(...)` and writes from a page-aligned source — including the
finalize bundle later. This is the "ideal" steady-state path and the one
intra-batch-finalize avoids by bouncing.

## End-of-stream finalize

`finalize_shards` runs when the writer is flushed and any shard is still
open (a partial generation at the end of the stream). It builds each
remaining shard's bundle into `sh->bundle_buf` (or a malloc'd buffer for
sinks without `write_direct`, like S3), `write_direct`s it, truncates,
finalizes, and resets state. By construction there is no pending IO on
these shards — they were never finalized in a prior batch — so reusing
`sh->bundle_buf` is always safe here.

## Async, lifetime, and the io_queue

`write_direct` (`fs_slot_write_direct` → `pwrite_ref_job`) stores **only a
pointer** to caller memory and a target offset. The io_queue worker reads
that memory at job-execute time, which can be much later than queue time.
That's safe only as long as the source memory stays alive and unchanged.
Two lifetime mechanisms cover this:

- **Aggregate buffer** — the agg slot is double-buffered. Each batch records
  a fence (`sink->record_fence`) after queueing all of its writes; the next
  batch waits on that fence (`sink->wait_fence`) before reusing the slot.
  So agg-buffer pointers stay valid across the async drain.
- **Bundle pool** — `shard_state.bundle_buf_pool` lives for the lifetime of
  the stream. Stream destroy waits on the sink's fence before freeing the
  pool. The only intra-stream race (intra-batch second finalize on the same
  shard) is handled by bouncing instead of reusing the slot.

`write` (`pwrite_job`) avoids the lifetime question entirely by malloc'ing a
new buffer and copying the source bytes into it. The job carries its own
memory; the caller can release the source immediately.

## GPU aggregator notes

The CPU and GPU aggregators produce the same per-shard layout. The GPU side
uses two small kernels:

- `add_shard_bias_k` (one block per shard): reads `bias_s = s * shard_capacity
  + d_tail_bytes_prev[s] - d_offsets[s * tps_group]` once into shared memory,
  then every thread in the block adds `bias_s` to its chunk's offset. The
  shared-memory hop is required: thread 0's write to `d_offsets[base]` would
  otherwise clobber the value other threads still need to read.
- `copy_leading_tail_k` (one block per shard): copies the prior batch's
  ragged tail from `d_tail_carry[s * page]` into the head of shard s's
  region in `d_aggregated`.

There is no per-generation bias kernel; intra-batch generation crossings
are handled at delivery time, same as on CPU.

## Where to look in the code

| concept                          | file                                  |
|----------------------------------|---------------------------------------|
| run walking, three outcomes      | `src/zarr/shard_delivery.c`           |
| `shard_state` / `active_shard`   | `src/zarr/shard_delivery.h`           |
| CPU aggregator                   | `src/cpu/aggregate.c`                 |
| GPU aggregator + kernels         | `src/gpu/aggregate.cu`                |
| layout / `shard_capacity`        | `src/stream/types.aggregate.{h,c}`    |
| FS shard pool, `pwrite_ref_job`  | `src/zarr/shard_pool_fs.c`            |
| O_DIRECT alignment watchdog      | `fs_slot_write` in `shard_pool_fs.c`  |
| writer interface (`shard_writer`)| `src/writer.h`                        |

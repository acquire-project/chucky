# Shard tail carry-over

How aggregated chunks reach disk in the unbuffered (O_DIRECT / F_NOCACHE /
FILE_FLAG_NO_BUFFERING) sink, and how per-shard ragged tails roll forward
across batches so the on-disk file contains no inter-batch padding.

## Aggregate buffer layout

`page = sink->required_shard_alignment()`. When `page > 0` the workspace data
buffer is laid out hierarchically:

```
ws->data
├── LOD 0 segment (page-aligned, data_segment_offset = 0)
│   ├── shard 0 region (shard_capacity bytes, page-aligned)
│   │   ├── [0, tail_in)                     leading tail (from prior batch)
│   │   ├── [tail_in, tail_in + run_real)    chunks for this batch
│   │   └── slack to shard_capacity
│   ├── shard 1 region
│   └── ...
├── LOD 1 segment
└── ...
```

`shard_capacity = align_up(active_count_max × cps_inner × max_comp_chunk_bytes
+ page, page)`. The one page of slack reserves room for an incoming leading
tail (< page).

Chunks within one batch pack tightly per shard. Generations within one batch
are contiguous in the aggregate buffer — there is **no per-generation
padding**. Intra-batch generation crossings are handled at delivery time, not
in the aggregator.

## Per-shard state (`struct active_shard`)

| field         | purpose                                                          |
|---------------|------------------------------------------------------------------|
| `writer`      | open shard-file handle; `NULL` between generations               |
| `data_cursor` | next file offset to write at                                     |
| `tail_buf`    | page-sized scratch holding the carry-forward bytes               |
| `tail_bytes`  | how many bytes of `tail_buf` are valid                           |
| `bundle_buf`  | per-shard bundle slot (slice of `shard_state.bundle_buf_pool`)   |
| `index`       | `[chunks_per_shard_total][2]` (offset, size) pairs for the index |

`tail_buf` and `bundle_buf` are slices of contiguous pools owned by
`shard_state` (`tail_buf_pool`, `bundle_buf_pool`). Both lifetime-bounded by
the stream — they outlive any single async write.

`bundle_capacity = align_up(page + chunks_per_shard_total × 16 + 4, page)`:
one page for the trailing sub-page data, the index, the CRC, page-padded.

## The run concept

`deliver_to_shards_batch` walks the batch in **runs**. A run is a contiguous
span of epochs that fits in the current shard generation:

```
run_len       = min(remaining_in_shard, remaining_in_batch)
run_finalizes = (run_len == remaining_in_shard)
```

For each `(si, run)` pair the aggregator has already laid out the run's chunk
bytes at `result->data + si × shard_capacity + bytes_consumed[si]`, where
`bytes_consumed[si]` is the cumulative agg-buffer offset advanced by prior
runs in this batch.

## Three run outcomes

### Non-finalizing run (batch ends mid-generation)

1. `total_run = tail_in + run_real` (tail_in is the leading tail, only on the
   first run for this shard in this batch).
2. `write_bytes = (total_run / page) × page` — page-aligned floor.
3. Write `[src, src + write_bytes)` via `write_direct` (zero-copy async) when
   `src` is page-aligned, else via `write` (copy).
4. Save `total_run - write_bytes` bytes (always `< page`) into `sh->tail_buf`,
   record in `h_tail_bytes[si]`.
5. `sh->data_cursor += write_bytes`.

### Finalizing run (this run completes the shard generation)

The finalize is **two-step** when `total_run > page`:

1. **Page-floor write**. Identical to non-finalizing: `write_bytes = (total_run
   / page) × page` bytes via `write_direct` from the agg slot. Advances
   `data_cursor` by `write_bytes`.
2. **Bundle write**. Build `[trailing<page || index || CRC || zero-pad-to-page]`
   in `sh->bundle_buf`, write via `write_direct` from the bundle slot at
   `data_cursor`. Lifetime is bounded by `shard_state` (no malloc'd buffer to
   free).
3. **Truncate** to `data_cursor + logical_bytes` to drop trailing pad zeros
   from the on-disk file.
4. **Finalize** (close).
5. Reset per-shard state: `writer = NULL`, `data_cursor = 0`, `tail_bytes = 0`,
   index cleared.

When `total_run < page`, step 1 is skipped and the bundle includes all of
`total_run` as the "trailing" portion.

### Intra-batch finalize (same shard finalizes twice in one batch)

Triggered when `epochs_per_batch > chunks_per_shard_append`. The first
finalize of a shard in a batch follows the path above and uses
`sh->bundle_buf` for `write_direct` (zero-copy). The **second** finalize of
the same shard within the same batch must bounce-copy the bundle through a
malloc'd buffer because the prior finalize's `pwrite_ref_job` may still be
pending in the io_queue. Reusing `sh->bundle_buf` while the worker still
holds a pointer to it would cause a data race.

Detection: the caller (`deliver_to_shards_batch`) passes
`is_first_run_for_shard` to `deliver_run_finalizing`, which selects between
`bundle_buf` (zero-copy) and a one-shot malloc bounce.

The fresh-gen run that follows an intra-batch finalize (writing data into
the new generation's file) also bounces because its source pointer in the
agg slot is mid-shard, not page-aligned.

Frequency of intra-batch finalize is governed by the ratio of
`epochs_per_batch` to `chunks_per_shard_append`. Workloads with batches
sized at or below one shard generation never hit this path.

## Batch boundary (same generation)

The non-finalizing path leaves a sub-page tail in `sh->tail_buf` and
`h_tail_bytes[si] > 0`. On the next batch:

1. Aggregator anchors shard `si`'s first chunk at
   `shard_base + tail_in`.
2. Leading-tail-copy memcpys `sh->tail_buf` into `[shard_base,
   shard_base + tail_in)`.
3. Deliver sees `src = result->data + shard_base + 0` (`bytes_consumed` is
   reset per batch); the tail is at the front, fresh chunks follow — one
   contiguous page-aligned region.

Every batch's `write_bytes` write lands at a `data_cursor` that is a
multiple of `page`. No inter-batch padding ever lands on disk.

## Generation boundary (across batches)

After a finalizing run, the next batch starts with `bytes_consumed[si] = 0`
(it's a fresh allocation in `deliver_to_shards_batch`) and the file is fresh
(`writer = NULL`, opened on first run via `sink->open`). `tail_in = 0` (the
finalize zeroed it). So the new generation's first run has page-aligned
`src` and goes through `write_direct`, including the bundle on finalize.

## GPU aggregator

`add_shard_bias_k` (one block per shard) computes a single per-shard bias
`bias_s = s × shard_capacity + d_tail_bytes_prev[s] − d_offsets[s × tps_group]`
into shared memory and applies it to every chunk's offset in shard `s`.
After the exclusive prefix sum and bias addition, shard `s`'s first chunk
lands at `s × shard_capacity + tail_in`, with subsequent chunks packed
tightly. `copy_leading_tail_k` (one block per shard) stages the prior batch's
ragged tail at the head of each shard's region before chunks pack just past
it. There is no per-generation bias kernel — generations within one batch
are contiguous in the agg buffer.

## Async coupling

All `write_direct` / `write` / `truncate` / close jobs go on the same
per-pool FIFO `io_queue`. The pool exposes `record_fence` / `wait_fence`;
each batch ends with one fence so the next slot reuse waits for every IO
from this batch to drain. That's why `write_direct` can return immediately
while the source buffer (the aggregated workspace, or the bundle slot) stays
alive — slot rotation and stream lifetime gate the fence, not the write
call.

`shard_pool_fs.c` adds a debug-build assertion in `fs_slot_write` that
`offset` and `nbytes` are multiples of `w->alignment` when `w->alignment > 0`,
catching upstream bugs that would otherwise silently corrupt O_DIRECT writes.

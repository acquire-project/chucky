# Shard tail carry-over

How aggregated chunks reach disk in the unbuffered (O_DIRECT / F_NOCACHE /
FILE_FLAG_NO_BUFFERING) sink, and how per-shard ragged tails roll forward
across batches so the on-disk file contains no inter-batch padding.

## Buffer layout (carry-over mode)

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
+ max_gens × page, page)` where
`max_gens = ceil(active_count_max / chunks_per_shard_append) + 1`. The extra
pages reserve room for an incoming leading tail (< page) plus up to one page
of pad after every generation that completes inside the batch.

The CPU aggregator's per-shard prefix-sum anchors the first chunk at
`shard_base + tail_in` and walks chunks epoch-by-epoch. After each generation
completes (`epoch_in_shard` reaches `chunks_per_shard_append`), the cursor is
advanced up to the next page boundary in the agg buffer. Chunks within a
single generation pack tightly behind the leading tail; consecutive
generations within one batch are page-aligned with respect to each other.

The pad lives only in the agg buffer. The on-disk file still gets each
generation truncated to its unpadded logical size by the prior generation's
finalize, so the on-disk layout is unchanged.

## Per-shard state (`struct active_shard`)

| field         | purpose                                                          |
|---------------|------------------------------------------------------------------|
| `writer`      | open shard-file handle; `NULL` between generations               |
| `data_cursor` | next file offset to write at                                     |
| `tail_buf`    | page-sized scratch holding the carry-forward bytes               |
| `tail_bytes`  | how many bytes of `tail_buf` are valid                           |
| `index`       | `[chunks_per_shard_total][2]` (offset, size) pairs for the index |

## The run concept

`deliver_to_shards_batch` walks the batch in **runs**. A run is a contiguous
span of epochs that fits in the current shard generation:

```
run_len       = min(remaining_in_shard, remaining_in_batch)
run_finalizes = (run_len == remaining_in_shard)
```

For each `(si, run)` pair the aggregator has already laid out the run's chunk
bytes at `result->data + si*shard_capacity + bytes_consumed[si]`, where
`bytes_consumed[si]` is the cumulative agg-buffer offset advanced by prior
runs in this batch.

## Two run outcomes

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

1. Build a temporary `fbuf = [remaining_data || index || CRC]`, padded up to a
   page.
2. Copy-write the bundle to the file. Must be `write` (not `write_direct`)
   because `fbuf` is freed immediately and `write_direct` is async.
3. Truncate the file to `data_cursor + logical_bytes` to drop the trailing
   page-padding zeros.
4. Close (async close job).
5. Reset: `writer = NULL`, `data_cursor = 0`, clear `index`, `tail_bytes = 0`,
   `h_tail_bytes[si] = 0`.

## Batch boundary (same generation)

The non-finalizing path leaves a sub-page tail in `sh->tail_buf` and
`h_tail_bytes[si] > 0`. On the next batch:

1. Aggregator anchors shard `si`'s first chunk at `shard_base + tail_in`.
2. Leading-tail-copy memcpys `sh->tail_buf` into `[shard_base, shard_base + tail_in)`.
3. Deliver sees `src = result->data + shard_base + 0` (`bytes_consumed` is
   reset per batch); the tail is at the front, fresh chunks follow — one
   contiguous page-aligned region.

Every batch's `write_bytes` write lands at a `data_cursor` that is a multiple
of `page`. No inter-batch padding ever lands on disk.

## Generation boundary (across batches)

After a finalizing run, the next batch starts with `bytes_consumed[si] = 0`
(it's a fresh allocation in `deliver_to_shards_batch`) and the file is fresh
(`writer = NULL`, opened on first run via `sink->open`). `tail_in = 0` (the
finalize zeroed it). So the new generation's first run has page-aligned `src`
and goes through `write_direct`.

## Generation boundary (intra-batch)

When `run_finalizes` mid-batch and more runs follow for the same `si`:

1. Finalize bundle written, file truncated and closed, state reset.
2. Outer loop opens a new file via `sink->open(level, shard_epoch * shard_inner_count + si)`.
3. `bytes_consumed[si] += total_run` advances the agg-buffer cursor, then
   when `requires_gen_pads` is set on the layout (CPU pipeline) it is rounded
   up to the next page so the next gen's `src` lands page-aligned.
4. `h_tail_bytes[si]` was zeroed by the finalize, so gen 1 has no carry-in.

Because the aggregator already padded gen 1's first chunk up to the next
page when laying out the agg buffer, gen 1's `src = data + shard_base +
align_up(total_run_gen0, page)` is page-aligned and the run takes
`write_direct` like every other non-finalizing run. No copying-write
fallback for steady-state delivery.

The GPU aggregator (`aggregate.cu`) does not yet pad gen boundaries — it
leaves `requires_gen_pads = 0` on the layout, so deliver does NOT advance
`bytes_consumed[si]` over a non-existent pad. Post-finalize runs on the GPU
path remain non-page-aligned and fall back to the copying `write` path; the
GPU port of the pad logic is a separate follow-up tracked in the plan
section below.

## Async coupling

All `write` / `write_direct` / `truncate` / close jobs go on the same per-pool
FIFO `io_queue`. The pool exposes `record_fence` / `wait_fence`; each batch
ends with one fence so the next slot reuse waits for every IO from this batch
to drain. That's why `write_direct` can return immediately while the source
buffer (the aggregated workspace) stays alive — slot rotation is gated by the
fence, not by the write call.

---

# GPU follow-up

The CPU aggregator now pads intra-batch generation boundaries up to the
next page so deliver always takes `write_direct`. The GPU side
(`aggregate.cu`: `compute_bias_k` / `apply_bias_k`) still produces
gen-tight, unpadded layouts — every shard places gen `g+1`'s first chunk
immediately after gen `g`'s last chunk inside the same shard region. When
`epochs_per_batch > chunks_per_shard_append`, the GPU's post-finalize runs
fall back to the copying `write` path.

To bring the GPU to parity, the bias kernels need to become gen-aware along
the same lines as the CPU prefix-sum: walk per-shard, compute bias =
`shard_base + tail_in + sum(per-gen pad)`, and have deliver flip the
`requires_gen_pads` flag on the layout. The CPU edit can ship independently;
the GPU port carries the same risk profile (small edit, no on-disk format
change, no async coupling change).

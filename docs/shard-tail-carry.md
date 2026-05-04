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
+ page, page)`. The `+page` reservation guarantees room for an incoming tail
plus a worst-case batch.

The aggregator's per-shard prefix-sum anchors the first chunk at
`shard_base + tail_in`, so the leading-tail-copy step stages the prior batch's
tail bytes in `[shard_base, shard_base + tail_in)` and the new chunks pack
tightly behind them.

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
3. `bytes_consumed[si]` is **not** reset, so `src_offset_in_shard` skips over
   gen 0's region in the agg buffer to find gen 1's chunks.
4. `h_tail_bytes[si]` was zeroed by the finalize, so gen 1 has no carry-in.

Gen 1's `src` lives at `shard_base + total_run_gen0` in the agg buffer, which
is **not** page-aligned in general (`total_run_gen0 = tail_in + run_real_gen0`,
neither term page-aligned). The `is_first_run_for_shard` gate in deliver is
false (`bytes_consumed[si] > 0`), so the alignment check fails and the run
falls back to the copying `write` path.

This is the only steady-state case where data takes the copying path. It only
arises when `epochs_per_batch > chunks_per_shard_append` so a single batch
spans more than one generation. The current test suite does not exercise this
configuration.

## Async coupling

All `write` / `write_direct` / `truncate` / close jobs go on the same per-pool
FIFO `io_queue`. The pool exposes `record_fence` / `wait_fence`; each batch
ends with one fence so the next slot reuse waits for every IO from this batch
to drain. That's why `write_direct` can return immediately while the source
buffer (the aggregated workspace) stays alive — slot rotation is gated by the
fence, not by the write call.

---

# Plan: pad gen boundaries in the agg buffer

Goal: eliminate the copying-write fallback for intra-batch generation
transitions so every non-finalizing write hits `write_direct`.

## Idea

Pad each generation boundary in the per-shard agg-buffer region up to the
next page boundary, mirroring the pad in `bytes_consumed[si]` on the deliver
side. Each fresh generation then starts at a page-aligned `src` and the
alignment gate in deliver always succeeds.

The pad lives only in the agg buffer; the file still gets `truncate`'d to
the unpadded logical size by the prior generation's finalize, so on-disk
layout is unchanged.

## Aggregator change

The CPU aggregator's per-shard prefix-sum becomes gen-aware. New input:
`epoch_in_shard_at_batch_start[lv]` — per-LOD scalar, all inner shards share
the same gen progression.

```
cur = shard_base + tail_in
e   = epoch_in_shard_at_batch_start[lv]
for k in [0, n_active):
  for j in [0, cps_inner):
    offsets[..k,j] = cur
    cur += sizes[..k,j]
  e += 1
  if e >= chunks_per_shard_append:
    cur = align_up(cur, page_size)
    e = 0
```

The leading-tail copy is unchanged — still applies only to gen 0.

## Deliver change

After a finalizing run, advance `bytes_consumed[si]` past the agg-buffer pad:

```c
bytes_consumed[si] = align_up(bytes_consumed[si] + total_run, page_size);
```

The `is_first_run_for_shard` gate in the alignment check can be dropped —
every run for a freshly-opened generation now has a page-aligned `src`.

## Sizing

`shard_capacity` grows by up to `(max_gens_per_batch - 1) × page`:

```
max_gens_per_batch = ceil(active_count_max / chunks_per_shard_append)
shard_capacity = align_up(worst + max_gens_per_batch × page, page)
```

In practice one extra page per generation boundary, negligible.

## Index offsets stay correct

`chunk_off = sh->data_cursor + (result->offsets[j] - shard_base - src_offset_in_shard)`
still computes the right file offset. The agg-buffer pad is invisible to the
file because each generation's file starts fresh with `data_cursor = 0`.

## GPU path

`aggregate.cu` (`compute_bias_k` / `apply_bias_k`) has the same shape and the
same gap. The CPU change can ship independently; the GPU edit is structurally
identical and a separate follow-up.

## Test coverage

A regression test is required: a config with
`epochs_per_batch > chunks_per_shard_append` exercises intra-batch generation
boundaries. The fix should turn the existing copying-write fallback into a
direct write — countable via the `counting_sink` shim already used by
`test_unbuffered_zero_copy`.

## Risk

Low. The change is two small edits (aggregator prefix-sum, deliver cursor)
plus a size bump and a test. No on-disk format change, no async-coupling
change.

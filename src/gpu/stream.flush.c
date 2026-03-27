#include "gpu/stream.flush.h"

#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.lod.h"

#include "gpu/lod.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#include <string.h>

// --- Helpers ---

// Build compress_agg_input from current state.
static struct compress_agg_input
make_compress_input(struct tile_stream_gpu* s, int fc, uint32_t n_epochs)
{
  struct flush_slot_gpu* fs = &s->flush.slot[fc];
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = fs->active_levels_mask,
    .epochs_per_batch = s->batch.epochs_per_batch,
    .pool_buf = s->pools.buf[fc],
    .lod_done = (s->levels.enable_multiscale && s->lod.t_end)
                  ? s->lod.t_end
                  : NULL,
  };
  memcpy(
    in.batch_active_masks, fs->batch_active_masks, n_epochs * sizeof(uint32_t));
  memcpy(in.epoch_events, s->batch.pool_events, n_epochs * sizeof(CUevent));
  return in;
}

// --- Epoch accumulation ---

// Return pointer to the current epoch's chunk region within the super-pool.
static inline void*
pool_epoch_ptr(struct tile_stream_gpu* s, uint32_t epoch_in_batch)
{
  const size_t bytes_per_element = dtype_bpe(s->config.dtype);
  return (char*)s->pools.buf[s->pools.current] +
         (uint64_t)epoch_in_batch * s->levels.total_chunks *
           s->layout.chunk_stride * bytes_per_element;
}

// Run LOD pipeline for the current epoch, or handle non-multiscale case.
// Updates flush slot batch_active_masks and active_levels_mask.
int
flush_run_epoch_lod(struct tile_stream_gpu* s)
{
  struct flush_slot_gpu* fs = &s->flush.slot[s->pools.current];
  uint32_t active_mask;

  if (!s->levels.enable_multiscale || !s->lod.d_linear) {
    // Non-multiscale: all levels (just L0) are active
    active_mask = 1;
  } else {
    CHECK(Error,
          lod_run_epoch(&s->lod,
                        &s->levels,
                        pool_epoch_ptr(s, s->batch.accumulated),
                        s->config.dtype,
                        s->config.reduce_method,
                        s->config.append_reduce_method,
                        &s->dims,
                        s->streams.compute,
                        &active_mask) == 0);
  }

  fs->batch_active_masks[s->batch.accumulated] = active_mask;
  fs->active_levels_mask |= active_mask;
  return 0;

Error:
  return 1;
}

// Drain previous flush, kick async pipeline on completed pool,
// swap to fresh pool, reset batch state.
static struct writer_result
drain_kick_and_swap(struct tile_stream_gpu* s)
{
  const uint32_t K = s->batch.epochs_per_batch;
  const int completed_pool = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush.slot[completed_pool];

  // Wait for any previous flush to finish delivery
  struct writer_result r = flush_drain_pending(s);
  if (r.error)
    return r;

  // Launch async compress->aggregate->D2H on completed pool
  fs->batch_epoch_count = (int)s->batch.accumulated;
  if (flush_kick_batch(s, completed_pool, s->batch.accumulated))
    return writer_error();

  // Swap to fresh pool and zero it for next batch
  s->pools.current ^= 1;
  size_t pool_bytes = (uint64_t)K * s->levels.total_chunks *
                      s->layout.chunk_stride *
                      dtype_bpe(s->config.dtype);
  CU(Error,
     cuMemsetD8Async(s->pools.buf[s->pools.current],
                     0,
                     pool_bytes,
                     s->streams.compute));

  // Reset batch accumulation
  s->batch.accumulated = 0;
  s->flush.slot[s->pools.current].active_levels_mask = 0;
  memset(s->flush.slot[s->pools.current].batch_active_masks,
         0,
         sizeof(s->flush.slot[s->pools.current].batch_active_masks));

  // Mark completed pool as pending delivery
  s->flush.pending = 1;
  s->flush.current = completed_pool;
  return writer_ok();

Error:
  return writer_error();
}

// Accumulate one epoch into the current batch, or flush when batch is full.
// Called at each epoch boundary.
struct writer_result
flush_accumulate_epoch(struct tile_stream_gpu* s)
{
  if (flush_run_epoch_lod(s))
    return writer_error();

  CU(Error,
     cuEventRecord(s->batch.pool_events[s->batch.accumulated],
                   s->streams.compute));
  s->batch.accumulated++;

  if (s->batch.accumulated < s->batch.epochs_per_batch)
    return writer_ok();

  return drain_kick_and_swap(s);

Error:
  return writer_error();
}

// --- Batch flush pipeline ---

// Kick compress + aggregate + D2H for a batch of n_epochs epochs.
int
flush_kick_batch(struct tile_stream_gpu* s, int fc, uint32_t n_epochs)
{
  struct compress_agg_input in = make_compress_input(s, fc, n_epochs);
  struct flush_handoff handoff = { 0 };

  CHECK(Error,
        compress_agg_kick(&s->compress_agg,
                          &in,
                          &s->levels,
                          &s->batch,
                          &s->dims,
                          s->streams.compress,
                          &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(&s->d2h_deliver,
                         &handoff,
                         &s->levels,
                         &s->batch,
                         &s->dims,
                         &s->config,
                         s->shard_sink,
                         s->streams.d2h) == 0);

  // Save handoff for drain
  s->flush.pending_handoff = handoff;

  return 0;

Error:
  return 1;
}

// Kick compress->aggregate->D2H->deliver for a single epoch from the
// super-pool.
static struct writer_result
kick_and_deliver_one_epoch(struct tile_stream_gpu* s,
                           int fc,
                           uint32_t epoch_in_batch,
                           uint32_t active_mask)
{
  // Build single-epoch input
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = 1,
    .active_levels_mask = active_mask,
    .epochs_per_batch = s->batch.epochs_per_batch,
    .lod_done = (s->levels.enable_multiscale && s->lod.t_end)
                  ? s->lod.t_end
                  : NULL,
  };
  in.batch_active_masks[0] = active_mask;
  in.epoch_events[0] = s->batch.pool_events[epoch_in_batch];

  // Point pool_buf at the specific epoch
  const size_t chunk_bytes =
    s->layout.chunk_stride * dtype_bpe(s->config.dtype);
  in.pool_buf = s->pools.buf[fc] + (uint64_t)epoch_in_batch *
                                      s->levels.total_chunks * chunk_bytes;

  struct flush_handoff handoff = { 0 };
  CHECK(Error,
        compress_agg_kick(&s->compress_agg,
                          &in,
                          &s->levels,
                          &s->batch,
                          &s->dims,
                          s->streams.compress,
                          &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(&s->d2h_deliver,
                         &handoff,
                         &s->levels,
                         &s->batch,
                         &s->dims,
                         &s->config,
                         s->shard_sink,
                         s->streams.d2h) == 0);

  return d2h_deliver_drain(&s->d2h_deliver,
                           &handoff,
                           &s->levels,
                           &s->batch,
                           &s->dims,
                           &s->layout,
                           &s->config,
                           s->shard_sink,
                           &s->lod,
                           &s->metrics,
                           &s->metadata_update_clock);

Error:
  return writer_error();
}

// --- Public interface ---

struct writer_result
flush_drain_pending(struct tile_stream_gpu* s)
{
  if (!s->flush.pending)
    return writer_ok();

  s->flush.pending = 0;
  return d2h_deliver_drain(&s->d2h_deliver,
                           &s->flush.pending_handoff,
                           &s->levels,
                           &s->batch,
                           &s->dims,
                           &s->layout,
                           &s->config,
                           s->shard_sink,
                           &s->lod,
                           &s->metrics,
                           &s->metadata_update_clock);
}

struct writer_result
flush_accumulated_sync(struct tile_stream_gpu* s)
{
  if (s->batch.accumulated == 0)
    return writer_ok();

  const int fc = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush.slot[fc];

  if (s->batch.accumulated == s->batch.epochs_per_batch) {
    // Full batch: use the batch pipeline (LUT-accelerated aggregate)
    fs->batch_epoch_count = (int)s->batch.accumulated;
    if (flush_kick_batch(s, fc, s->batch.accumulated))
      return writer_error();

    struct writer_result r = d2h_deliver_drain(&s->d2h_deliver,
                                               &s->flush.pending_handoff,
                                               &s->levels,
                                               &s->batch,
                                               &s->dims,
                                               &s->layout,
                                               &s->config,
                                               s->shard_sink,
                                               &s->lod,
                                               &s->metrics,
                                               &s->metadata_update_clock);
    s->batch.accumulated = 0;
    s->flush.slot[s->pools.current].active_levels_mask = 0;
    memset(s->flush.slot[s->pools.current].batch_active_masks,
           0,
           sizeof(s->flush.slot[s->pools.current].batch_active_masks));
    return r;
  }

  // Partial batch: process each epoch individually
  for (uint32_t e = 0; e < s->batch.accumulated; ++e) {
    uint32_t mask = fs->batch_active_masks[e];
    if (!mask)
      continue;

    struct writer_result r = kick_and_deliver_one_epoch(s, fc, e, mask);
    if (r.error)
      return r;
  }

  s->batch.accumulated = 0;
  fs->active_levels_mask = 0;
  memset(fs->batch_active_masks, 0, sizeof(fs->batch_active_masks));
  return writer_ok();
}

struct writer_result
flush_partial_append(struct tile_stream_gpu* s)
{
  if (!s->dims.append_downsample || !s->levels.enable_multiscale)
    return writer_ok();

  const struct lod_plan* p = &s->lod.plan;
  const size_t bytes_per_element = dtype_bpe(s->config.dtype);
  const enum dtype dtype = s->config.dtype;

  // Check if any level has pending data
  uint32_t active_levels_mask = 0;
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (s->lod.append_accum.counts[lv] > 0)
      active_levels_mask |= (1u << lv);
  }

  if (!active_levels_mask)
    return writer_ok();

  const int fc = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush.slot[fc];
  fs->active_levels_mask = active_levels_mask;
  fs->batch_epoch_count = 1; // partial append flush is always 1 epoch

  // Zero LOD level regions of pool, emit partial accums, scatter to chunks
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    uint64_t n_elements = p->batch_count * p->lod_nelem[lv];

    // Compute offset of this level within the packed accumulator
    uint64_t accum_offset = 0;
    for (int k = 1; k < lv; ++k)
      accum_offset += p->batch_count * p->lod_nelem[k];

    size_t accum_bpe = dtype_accum_bpe(dtype, s->config.append_reduce_method);

    struct lod_span lev = lod_spans_at(&p->levels, lv);
    CUdeviceptr morton_lv = s->lod.d_morton + lev.beg * bytes_per_element;
    CUdeviceptr accum_lv = s->lod.append_accum.d_accum + accum_offset * accum_bpe;

    // Emit with actual count (not period) -- mean divides by actual count
    CHECK(Error,
          lod_accum_emit(morton_lv,
                         accum_lv,
                         dtype,
                         s->config.append_reduce_method,
                         n_elements,
                         s->lod.append_accum.counts[lv],
                         s->streams.compute) == 0);

    s->lod.append_accum.counts[lv] = 0;

    // Zero and scatter to chunk pool (epoch 0 in current pool)
    CUdeviceptr dst =
      s->pools.buf[s->pools.current] +
      s->levels.chunk_offset[lv] * s->layout.chunk_stride * bytes_per_element;
    size_t lv_pool_bytes =
      s->levels.chunk_count[lv] * s->layout.chunk_stride * bytes_per_element;
    CU(Error, cuMemsetD8Async(dst, 0, lv_pool_bytes, s->streams.compute));

    CHECK(Error,
          lod_morton_to_chunks_lut(dst,
                                   morton_lv,
                                   s->lod.d_morton_chunk_lut[lv],
                                   s->lod.d_morton_batch_chunk_offsets[lv],
                                   dtype,
                                   p->lod_nelem[lv],
                                   p->batch_count,
                                   s->streams.compute) == 0);
  }

  CU(Error, cuEventRecord(s->pools.ready[fc], s->streams.compute));
  if (s->lod.t_end)
    CU(Error, cuEventRecord(s->lod.t_end, s->streams.compute));

  CU(Error, cuEventRecord(s->batch.pool_events[0], s->streams.compute));
  if (flush_kick_batch(s, fc, 1))
    return writer_error();
  return d2h_deliver_drain(&s->d2h_deliver,
                           &s->flush.pending_handoff,
                           &s->levels,
                           &s->batch,
                           &s->dims,
                           &s->layout,
                           &s->config,
                           s->shard_sink,
                           &s->lod,
                           &s->metrics,
                           &s->metadata_update_clock);

Error:
  return writer_error();
}

#include "stream.flush.h"

#include "flush.compress_agg.h"
#include "flush.d2h_deliver.h"
#include "stream.lod.h"

#include "lod.h"
#include "prelude.cuda.h"
#include "prelude.h"

#include <string.h>

// --- Helpers ---

// Build compress_agg_input from current state.
static struct compress_agg_input
make_compress_input(struct flush_context* ctx, int fc, uint32_t n_epochs)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = fs->active_levels_mask,
    .epochs_per_batch = ctx->batch->epochs_per_batch,
    .pool_buf = ctx->pools->buf[fc],
    .lod_done = (ctx->levels->enable_multiscale && ctx->lod->t_end)
                  ? ctx->lod->t_end
                  : NULL,
  };
  memcpy(
    in.batch_active_masks, fs->batch_active_masks, n_epochs * sizeof(uint32_t));
  memcpy(in.epoch_events, ctx->batch->pool_events, n_epochs * sizeof(CUevent));
  return in;
}

// --- Epoch accumulation ---

// Return pointer to the current epoch's chunk region within the super-pool.
static inline void*
pool_epoch_ptr(struct flush_context* ctx, uint32_t epoch_in_batch)
{
  const size_t bpe = lod_dtype_bpe(ctx->config->dtype);
  return (char*)ctx->pools->buf[ctx->pools->current] +
         (uint64_t)epoch_in_batch * ctx->levels->total_chunks *
           ctx->layout->chunk_stride * bpe;
}

// Run LOD pipeline for the current epoch, or handle non-multiscale case.
// Updates flush slot batch_active_masks and active_levels_mask.
int
flush_run_epoch_lod(struct flush_context* ctx)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[ctx->pools->current];
  uint32_t active_mask;

  if (!ctx->levels->enable_multiscale || !ctx->lod->d_linear) {
    // Non-multiscale: all levels (just L0) are active
    active_mask = 1;
  } else {
    CHECK(Error,
          lod_run_epoch(ctx->lod,
                        ctx->levels,
                        ctx->layout,
                        pool_epoch_ptr(ctx, ctx->batch->accumulated),
                        ctx->config->dtype,
                        ctx->config->reduce_method,
                        ctx->config->dim0_reduce_method,
                        ctx->streams.compute,
                        &active_mask) == 0);
  }

  fs->batch_active_masks[ctx->batch->accumulated] = active_mask;
  fs->active_levels_mask |= active_mask;
  return 0;

Error:
  return 1;
}

// Drain previous flush, kick async pipeline on completed pool,
// swap to fresh pool, reset batch state.
static struct writer_result
drain_kick_and_swap(struct flush_context* ctx)
{
  const uint32_t K = ctx->batch->epochs_per_batch;
  const int completed_pool = ctx->pools->current;
  struct flush_slot_gpu* fs = &ctx->flush->slot[completed_pool];

  // Wait for any previous flush to finish delivery
  struct writer_result r = flush_drain_pending(ctx);
  if (r.error)
    return r;

  // Launch async compress->aggregate->D2H on completed pool
  fs->batch_epoch_count = (int)ctx->batch->accumulated;
  if (flush_kick_batch(ctx, completed_pool, ctx->batch->accumulated))
    return writer_error();

  // Swap to fresh pool and zero it for next batch
  ctx->pools->current ^= 1;
  size_t pool_bytes = (uint64_t)K * ctx->levels->total_chunks *
                      ctx->layout->chunk_stride *
                      lod_dtype_bpe(ctx->config->dtype);
  CU(Error,
     cuMemsetD8Async(ctx->pools->buf[ctx->pools->current],
                     0,
                     pool_bytes,
                     ctx->streams.compute));

  // Reset batch accumulation
  ctx->batch->accumulated = 0;
  ctx->flush->slot[ctx->pools->current].active_levels_mask = 0;
  memset(ctx->flush->slot[ctx->pools->current].batch_active_masks,
         0,
         sizeof(ctx->flush->slot[ctx->pools->current].batch_active_masks));

  // Mark completed pool as pending delivery
  ctx->flush->pending = 1;
  ctx->flush->current = completed_pool;
  return writer_ok();

Error:
  return writer_error();
}

// Accumulate one epoch into the current batch, or flush when batch is full.
// Called at each epoch boundary.
struct writer_result
flush_accumulate_epoch(struct flush_context* ctx)
{
  if (flush_run_epoch_lod(ctx))
    return writer_error();

  CU(Error,
     cuEventRecord(ctx->batch->pool_events[ctx->batch->accumulated],
                   ctx->streams.compute));
  ctx->batch->accumulated++;

  if (ctx->batch->accumulated < ctx->batch->epochs_per_batch)
    return writer_ok();

  return drain_kick_and_swap(ctx);

Error:
  return writer_error();
}

// --- Batch flush pipeline ---

// Kick compress + aggregate + D2H for a batch of n_epochs epochs.
int
flush_kick_batch(struct flush_context* ctx, int fc, uint32_t n_epochs)
{
  struct compress_agg_input in = make_compress_input(ctx, fc, n_epochs);
  struct flush_handoff handoff = { 0 };

  CHECK(Error,
        compress_agg_kick(ctx->compress_agg,
                          &in,
                          ctx->levels,
                          ctx->batch,
                          ctx->streams.compress,
                          &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(ctx->d2h_deliver,
                         &handoff,
                         ctx->levels,
                         ctx->batch,
                         ctx->config,
                         ctx->streams.d2h) == 0);

  // Save handoff for drain
  ctx->flush->pending_handoff = handoff;

  return 0;

Error:
  return 1;
}

// Kick compress->aggregate->D2H->deliver for a single epoch from the
// super-pool.
static struct writer_result
kick_and_deliver_one_epoch(struct flush_context* ctx,
                           int fc,
                           uint32_t epoch_in_batch,
                           uint32_t active_mask)
{
  // Build single-epoch input
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = 1,
    .active_levels_mask = active_mask,
    .epochs_per_batch = ctx->batch->epochs_per_batch,
    .lod_done = (ctx->levels->enable_multiscale && ctx->lod->t_end)
                  ? ctx->lod->t_end
                  : NULL,
  };
  in.batch_active_masks[0] = active_mask;
  in.epoch_events[0] = ctx->batch->pool_events[epoch_in_batch];

  // Point pool_buf at the specific epoch
  const size_t chunk_bytes =
    ctx->layout->chunk_stride * lod_dtype_bpe(ctx->config->dtype);
  in.pool_buf = ctx->pools->buf[fc] + (uint64_t)epoch_in_batch *
                                        ctx->levels->total_chunks * chunk_bytes;

  struct flush_handoff handoff = { 0 };
  CHECK(Error,
        compress_agg_kick(ctx->compress_agg,
                          &in,
                          ctx->levels,
                          ctx->batch,
                          ctx->streams.compress,
                          &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(ctx->d2h_deliver,
                         &handoff,
                         ctx->levels,
                         ctx->batch,
                         ctx->config,
                         ctx->streams.d2h) == 0);

  return d2h_deliver_drain(ctx->d2h_deliver,
                           &handoff,
                           ctx->levels,
                           ctx->batch,
                           ctx->layout,
                           ctx->config,
                           ctx->lod,
                           ctx->metrics,
                           ctx->metadata_update_clock);

Error:
  return writer_error();
}

// --- Public interface ---

struct writer_result
flush_drain_pending(struct flush_context* ctx)
{
  if (!ctx->flush->pending)
    return writer_ok();

  ctx->flush->pending = 0;
  return d2h_deliver_drain(ctx->d2h_deliver,
                           &ctx->flush->pending_handoff,
                           ctx->levels,
                           ctx->batch,
                           ctx->layout,
                           ctx->config,
                           ctx->lod,
                           ctx->metrics,
                           ctx->metadata_update_clock);
}

struct writer_result
flush_accumulated_sync(struct flush_context* ctx)
{
  if (ctx->batch->accumulated == 0)
    return writer_ok();

  const int fc = ctx->pools->current;
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];

  if (ctx->batch->accumulated == ctx->batch->epochs_per_batch) {
    // Full batch: use the batch pipeline (LUT-accelerated aggregate)
    fs->batch_epoch_count = (int)ctx->batch->accumulated;
    if (flush_kick_batch(ctx, fc, ctx->batch->accumulated))
      return writer_error();

    struct writer_result r = d2h_deliver_drain(ctx->d2h_deliver,
                                               &ctx->flush->pending_handoff,
                                               ctx->levels,
                                               ctx->batch,
                                               ctx->layout,
                                               ctx->config,
                                               ctx->lod,
                                               ctx->metrics,
                                               ctx->metadata_update_clock);
    ctx->batch->accumulated = 0;
    ctx->flush->slot[ctx->pools->current].active_levels_mask = 0;
    memset(ctx->flush->slot[ctx->pools->current].batch_active_masks,
           0,
           sizeof(ctx->flush->slot[ctx->pools->current].batch_active_masks));
    return r;
  }

  // Partial batch: process each epoch individually
  for (uint32_t e = 0; e < ctx->batch->accumulated; ++e) {
    uint32_t mask = fs->batch_active_masks[e];
    if (!mask)
      continue;

    struct writer_result r = kick_and_deliver_one_epoch(ctx, fc, e, mask);
    if (r.error)
      return r;
  }

  ctx->batch->accumulated = 0;
  fs->active_levels_mask = 0;
  memset(fs->batch_active_masks, 0, sizeof(fs->batch_active_masks));
  return writer_ok();
}

struct writer_result
flush_partial_dim0(struct flush_context* ctx)
{
  if (!ctx->levels->dim0_downsample || !ctx->levels->enable_multiscale)
    return writer_ok();

  const struct lod_plan* p = &ctx->lod->plan;
  const size_t bpe = lod_dtype_bpe(ctx->config->dtype);
  const enum lod_dtype dtype = ctx->config->dtype;

  // Check if any level has pending data
  uint32_t active_levels_mask = 0;
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (ctx->lod->dim0.counts[lv] > 0)
      active_levels_mask |= (1u << lv);
  }

  if (!active_levels_mask)
    return writer_ok();

  const int fc = ctx->pools->current;
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];
  fs->active_levels_mask = active_levels_mask;
  fs->batch_epoch_count = 1; // partial dim0 flush is always 1 epoch

  // Zero LOD level regions of pool, emit partial accums, scatter to chunks
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    uint64_t n_elements = p->batch_count * p->lod_counts[lv];

    // Compute offset of this level within the packed accumulator
    uint64_t accum_offset = 0;
    for (int k = 1; k < lv; ++k)
      accum_offset += p->batch_count * p->lod_counts[k];

    size_t accum_bpe = lod_accum_bpe(dtype, ctx->config->dim0_reduce_method);

    struct lod_span lev = lod_spans_at(&p->levels, lv);
    CUdeviceptr morton_lv = ctx->lod->d_morton + lev.beg * bpe;
    CUdeviceptr accum_lv = ctx->lod->dim0.d_accum + accum_offset * accum_bpe;

    // Emit with actual count (not period) -- mean divides by actual count
    CHECK(Error,
          lod_accum_emit(morton_lv,
                         accum_lv,
                         dtype,
                         ctx->config->dim0_reduce_method,
                         n_elements,
                         ctx->lod->dim0.counts[lv],
                         ctx->streams.compute) == 0);

    ctx->lod->dim0.counts[lv] = 0;

    // Zero and scatter to chunk pool (epoch 0 in current pool)
    CUdeviceptr dst =
      ctx->pools->buf[ctx->pools->current] +
      ctx->levels->chunk_offset[lv] * ctx->layout->chunk_stride * bpe;
    size_t lv_pool_bytes =
      ctx->levels->chunk_count[lv] * ctx->layout->chunk_stride * bpe;
    CU(Error, cuMemsetD8Async(dst, 0, lv_pool_bytes, ctx->streams.compute));

    CHECK(Error,
          lod_morton_to_chunks_lut(dst,
                                   morton_lv,
                                   ctx->lod->d_morton_chunk_lut[lv],
                                   ctx->lod->d_morton_batch_chunk_offsets[lv],
                                   dtype,
                                   p->lod_counts[lv],
                                   p->batch_count,
                                   ctx->streams.compute) == 0);
  }

  CU(Error, cuEventRecord(ctx->pools->ready[fc], ctx->streams.compute));
  if (ctx->lod->t_end)
    CU(Error, cuEventRecord(ctx->lod->t_end, ctx->streams.compute));

  CU(Error, cuEventRecord(ctx->batch->pool_events[0], ctx->streams.compute));
  if (flush_kick_batch(ctx, fc, 1))
    return writer_error();
  return d2h_deliver_drain(ctx->d2h_deliver,
                           &ctx->flush->pending_handoff,
                           ctx->levels,
                           ctx->batch,
                           ctx->layout,
                           ctx->config,
                           ctx->lod,
                           ctx->metrics,
                           ctx->metadata_update_clock);

Error:
  return writer_error();
}

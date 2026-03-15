#include "stream_internal.h"

#include "aggregate.h"
#include "compress.h"
#include "lod.h"
#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "shard_delivery.h"

#include <string.h>

// --- Helpers ---

// How many epochs fire for level `lv` within a batch of `n_epochs` epochs.
// L0 fires every epoch; dim0-downsampled level lv fires every 2^lv epochs.
// Returns 0 if the level doesn't fire at all in this batch.
static uint32_t
level_active_epochs(const struct flush_context* ctx, int lv, uint32_t n_epochs)
{
  uint32_t full = ctx->flush->levels[lv].batch_active_count;
  if (n_epochs >= ctx->batch->epochs_per_batch)
    return full;
  uint32_t period = (ctx->levels->dim0_downsample && lv > 0) ? (1u << lv) : 1;
  return (n_epochs >= period) ? n_epochs / period : 0;
}

// Count actual active epochs for a level from per-epoch masks.
// For infrequent dim0 levels (period > K, batch_active_count == 0),
// level_active_epochs returns 0 even when the level fired.  This function
// falls back to scanning the per-epoch masks in that case.
static uint32_t
level_actual_active_count(const struct flush_context* ctx,
                          const struct flush_slot_gpu* fs,
                          int lv,
                          uint32_t n_epochs)
{
  uint32_t n = level_active_epochs(ctx, lv, n_epochs);
  if (n > 0)
    return n;
  // Infrequent level: count from actual per-epoch masks
  for (uint32_t e = 0; e < n_epochs; ++e)
    if (fs->batch_active_masks[e] & (1u << lv))
      n++;
  return n;
}

static void
record_flush_metrics(struct flush_context* ctx, int fc)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];

  if (ctx->levels->enable_multiscale && ctx->lod->t_start) {
    const size_t bpe = ctx->config->bytes_per_element;
    const size_t scatter_bytes = ctx->layout->epoch_elements * bpe;
    const size_t morton_bytes =
      ctx->lod->plan.levels.ends[ctx->lod->plan.nlod - 1] * bpe;
    const size_t unified_pool_bytes =
      ctx->levels->total_tiles * ctx->layout->tile_stride * bpe;

    accumulate_metric_cu(&ctx->metrics->lod_gather,
                         ctx->lod->t_start,
                         ctx->lod->t_scatter_end,
                         scatter_bytes);
    accumulate_metric_cu(&ctx->metrics->lod_reduce,
                         ctx->lod->t_scatter_end,
                         ctx->lod->t_reduce_end,
                         morton_bytes);
    if (ctx->levels->dim0_downsample) {
      size_t accum_bpe = lod_accum_bpe(bpe, ctx->config->dim0_reduce_method);
      size_t dim0_bytes = ctx->lod->dim0.total_elements * accum_bpe;
      accumulate_metric_cu(&ctx->metrics->lod_dim0_fold,
                           ctx->lod->t_reduce_end,
                           ctx->lod->t_dim0_end,
                           dim0_bytes);
    }
    accumulate_metric_cu(&ctx->metrics->lod_morton_tile,
                         ctx->lod->t_dim0_end,
                         ctx->lod->t_end,
                         unified_pool_bytes);
  }

  {
    const size_t pool_bytes =
      (uint64_t)fs->batch_epoch_count * ctx->levels->total_tiles *
      ctx->layout->tile_stride * ctx->config->bytes_per_element;

    accumulate_metric_cu(&ctx->metrics->compress,
                         fs->t_compress_start,
                         fs->t_compress_end,
                         pool_bytes);

    // Use actual aggregated bytes from h_offsets (available after D2H sync).
    size_t agg_bytes = 0;
    const uint32_t n_epochs = (uint32_t)fs->batch_epoch_count;
    for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
      if (!(fs->active_levels_mask & (1u << lv)))
        continue;
      uint32_t active_count = level_actual_active_count(ctx, fs, lv, n_epochs);
      if (active_count == 0)
        continue;
      struct level_flush_state* lvl = &ctx->flush->levels[lv];
      uint64_t batch_covering =
        (uint64_t)active_count * lvl->agg_layout.covering_count;
      agg_bytes += lvl->agg[fc].h_offsets[batch_covering];
    }

    accumulate_metric_cu(&ctx->metrics->aggregate,
                         fs->t_compress_end,
                         fs->t_aggregate_end,
                         agg_bytes);
    accumulate_metric_cu(
      &ctx->metrics->d2h, fs->t_d2h_start, fs->ready, agg_bytes);
  }
}

// Wait for pending IO fences on aggregate slots before reuse.
static void
wait_io_fences(struct flush_context* ctx, int fc, uint32_t level_mask)
{
  if (!ctx->config->shard_sink->wait_fence)
    return;
  for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;
    struct aggregate_slot* agg = &ctx->flush->levels[lv].agg[fc];
    if (agg->io_done.seq > 0)
      ctx->config->shard_sink->wait_fence(
        ctx->config->shard_sink, (uint8_t)lv, agg->io_done);
  }
}

// Record compress-start, compress, record compress-end.
static int
kick_compress(struct flush_context* ctx,
              struct flush_slot_gpu* fs,
              const void* d_input,
              uint64_t n_tiles)
{
  const size_t tile_bytes =
    ctx->layout->tile_stride * ctx->config->bytes_per_element;

  CU(Error, cuEventRecord(fs->t_compress_start, ctx->streams.compress));
  CHECK(Error,
        codec_compress(ctx->codec,
                       d_input,
                       tile_bytes,
                       (void*)fs->d_compressed,
                       n_tiles,
                       ctx->streams.compress) == 0);
  CU(Error, cuEventRecord(fs->t_compress_end, ctx->streams.compress));
  return 0;

Error:
  return 1;
}

// Aggregate one epoch's tiles for a single level.
// epoch_in_compressed: index of this epoch within the compressed buffer.
static int
aggregate_epoch_level(struct flush_context* ctx,
                      int fc,
                      int lv,
                      uint32_t epoch_in_compressed)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];
  struct level_flush_state* lvl = &ctx->flush->levels[lv];
  struct aggregate_slot* agg = &lvl->agg[fc];

  void* d_comp = (char*)fs->d_compressed +
                 ((uint64_t)epoch_in_compressed * ctx->levels->total_tiles +
                  ctx->levels->tile_offset[lv]) *
                   ctx->codec->max_output_size;
  size_t* d_sizes = ctx->codec->d_comp_sizes +
                    (uint64_t)epoch_in_compressed * ctx->levels->total_tiles +
                    ctx->levels->tile_offset[lv];

  CHECK(Error,
        aggregate_by_shard_async(
          &lvl->agg_layout, d_comp, d_sizes, agg, ctx->streams.compress) == 0);
  return 0;

Error:
  return 1;
}

// Synchronize D2H, record metrics, deliver to sinks.
// Precondition: fs->batch_epoch_count and fs->active_levels_mask must be set.
static struct writer_result
sync_and_deliver(struct flush_context* ctx, int fc)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];

  CU(Error, cuEventSynchronize(fs->ready));
  record_flush_metrics(ctx, fc);

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;

    for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
      if (!(fs->active_levels_mask & (1u << lv)))
        continue;

      uint32_t active_count =
        level_actual_active_count(ctx, fs, lv, (uint32_t)fs->batch_epoch_count);
      if (active_count == 0)
        continue;

      struct level_flush_state* lvl = &ctx->flush->levels[lv];

      size_t level_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  &lvl->shard,
                                  &lvl->agg[fc],
                                  active_count,
                                  ctx->config->shard_sink,
                                  ctx->config->shard_alignment,
                                  &level_bytes))
        goto Error;
      sink_bytes += level_bytes;

      if (ctx->config->shard_sink->record_fence)
        lvl->agg[fc].io_done = ctx->config->shard_sink->record_fence(
          ctx->config->shard_sink, (uint8_t)lv);
    }

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&ctx->metrics->sink, sink_ms, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

// Periodic metadata update (dim0 extent per level).
static void
maybe_update_metadata(struct flush_context* ctx)
{
  if (!ctx->config->shard_sink->update_dim0)
    return;

  struct platform_clock peek = *ctx->metadata_update_clock;
  float elapsed = platform_toc(&peek);
  if (elapsed < ctx->config->metadata_update_interval_s)
    return;

  *ctx->metadata_update_clock = peek;
  const struct dimension* dims = ctx->config->dimensions;
  for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
    struct shard_state* ss = &ctx->flush->levels[lv].shard;
    uint64_t dim0_tiles =
      ss->shard_epoch * ss->tiles_per_shard_0 + ss->epoch_in_shard;
    uint64_t dim0_extent = dim0_tiles * dims[0].tile_size;
    ctx->config->shard_sink->update_dim0(
      ctx->config->shard_sink, (uint8_t)lv, dim0_extent);
  }
}

// Wait for D2H on the given flush slot, record timing, deliver to sinks.
static struct writer_result
wait_and_deliver(struct flush_context* ctx, int fc)
{
  wait_io_fences(ctx, fc, ~0u);
  struct writer_result r = sync_and_deliver(ctx, fc);
  if (!r.error)
    maybe_update_metadata(ctx);
  return r;
}

// Two-phase D2H: transfer offsets first (small), synchronize, then only
// actual compressed bytes.
// n_epochs: number of epochs in this batch (1 for single-epoch path).
// level_mask: which levels are active.
static int
two_phase_d2h(struct flush_context* ctx,
              struct flush_slot_gpu* fs,
              int fc,
              uint32_t level_mask,
              uint32_t n_epochs)
{
  // Phase 1: D2H offsets only
  for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;

    uint32_t active_count = 1;
    if (n_epochs > 1) {
      active_count = level_actual_active_count(ctx, fs, lv, n_epochs);
      if (active_count == 0)
        continue;
    }

    struct level_flush_state* lvl = &ctx->flush->levels[lv];
    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t covering = (uint64_t)active_count * lvl->agg_layout.covering_count;

    CU(Error,
       cuMemcpyDtoHAsync(agg->h_offsets,
                         (CUdeviceptr)agg->d_offsets,
                         (covering + 1) * sizeof(size_t),
                         ctx->streams.d2h));
  }
  CU(Error, cuEventRecord(fs->ready, ctx->streams.d2h));
  CU(Error, cuEventSynchronize(fs->ready));

  // Phase 2: D2H only actual compressed bytes per level
  for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;

    uint32_t active_count = 1;
    if (n_epochs > 1) {
      active_count = level_actual_active_count(ctx, fs, lv, n_epochs);
      if (active_count == 0)
        continue;
    }

    struct level_flush_state* lvl = &ctx->flush->levels[lv];
    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t covering = (uint64_t)active_count * lvl->agg_layout.covering_count;

    size_t actual = agg->h_offsets[covering];
    if (ctx->config->shard_alignment > 0)
      actual += ctx->config->shard_alignment;
    size_t cap =
      agg_pool_bytes((uint64_t)active_count * ctx->levels->tile_count[lv],
                     ctx->codec->max_output_size,
                     lvl->agg_layout.covering_count,
                     lvl->agg_layout.tps_inner,
                     lvl->agg_layout.page_size);
    if (actual > cap)
      actual = cap;

    CU(Error,
       cuMemcpyDtoHAsync(agg->h_aggregated,
                         (CUdeviceptr)agg->d_aggregated,
                         actual,
                         ctx->streams.d2h));
    CU(Error, cuEventRecord(agg->ready, ctx->streams.d2h));
  }

  CU(Error, cuEventRecord(fs->ready, ctx->streams.d2h));

  return 0;

Error:
  return 1;
}

// --- Batch flush pipeline ---

// Kick compress + aggregate + D2H for a batch of n_epochs epochs.
// fc: flush slot index (0 or 1, matches pools.current before swap).
// n_epochs: number of epochs in this batch (may be < K for final flush).
int
flush_kick_batch(struct flush_context* ctx, int fc, uint32_t n_epochs)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];

  // Wait for all per-epoch pool-ready events
  for (uint32_t e = 0; e < n_epochs; ++e)
    CU(Error,
       cuStreamWaitEvent(ctx->streams.compress, ctx->batch->pool_events[e], 0));
  if (ctx->levels->enable_multiscale && ctx->lod->t_end)
    CU(Error, cuStreamWaitEvent(ctx->streams.compress, ctx->lod->t_end, 0));

  // Compress all epochs as one batch
  uint64_t batch_tiles = (uint64_t)n_epochs * ctx->levels->total_tiles;
  CHECK(Error,
        kick_compress(ctx, fs, (void*)ctx->pools->buf[fc], batch_tiles) == 0);

  // Per-level batch aggregate on compress stream
  for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
    if (!(fs->active_levels_mask & (1u << lv)))
      continue;

    uint32_t active_count = level_active_epochs(ctx, lv, n_epochs);

    struct level_flush_state* lvl = &ctx->flush->levels[lv];
    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t tiles_lv = ctx->levels->tile_count[lv];

    if (active_count == 0) {
      // Infrequent dim0 level (period > K): scan actual per-epoch masks.
      for (uint32_t e = 0; e < n_epochs; ++e) {
        if (!(fs->batch_active_masks[e] & (1u << lv)))
          continue;
        active_count++;
        CHECK(Error, aggregate_epoch_level(ctx, fc, lv, e) == 0);
      }
      if (active_count == 0)
        continue;
    } else {
      uint64_t batch_tile_count = (uint64_t)active_count * tiles_lv;
      uint64_t batch_covering =
        (uint64_t)active_count * lvl->agg_layout.covering_count;

      int use_luts = (ctx->batch->epochs_per_batch > 1 && lvl->d_batch_gather &&
                      active_count == lvl->batch_active_count);
      if (use_luts) {
        CHECK(Error,
              aggregate_batch_by_shard_async(
                (void*)fs->d_compressed,
                ctx->codec->d_comp_sizes,
                (const uint32_t*)(uintptr_t)lvl->d_batch_gather,
                (const uint32_t*)(uintptr_t)lvl->d_batch_perm,
                batch_tile_count,
                batch_covering,
                ctx->codec->max_output_size,
                &lvl->agg_layout,
                agg,
                ctx->streams.compress) == 0);
      } else {
        // K=1 or partial batch: per-epoch aggregate
        for (uint32_t a = 0; a < active_count; ++a) {
          uint32_t period = 1;
          if (ctx->levels->dim0_downsample && lv > 0)
            period = 1u << lv;
          uint32_t pool_epoch = (n_epochs == 1) ? 0 : (a + 1) * period - 1;
          CHECK(Error, aggregate_epoch_level(ctx, fc, lv, pool_epoch) == 0);
        }
      }
    }
  }

  CU(Error, cuEventRecord(fs->t_aggregate_end, ctx->streams.compress));

  CU(Error, cuStreamWaitEvent(ctx->streams.d2h, fs->t_aggregate_end, 0));
  CU(Error, cuEventRecord(fs->t_d2h_start, ctx->streams.d2h));
  CHECK(Error,
        two_phase_d2h(ctx, fs, fc, fs->active_levels_mask, n_epochs) == 0);

  return 0;

Error:
  return 1;
}

// Kick compress->aggregate->D2H->deliver for a single epoch from the
// super-pool.
// epoch_in_batch: which epoch within the super-pool to process.
static struct writer_result
kick_and_deliver_one_epoch(struct flush_context* ctx,
                           int fc,
                           uint32_t epoch_in_batch,
                           uint32_t active_mask)
{
  struct flush_slot_gpu* fs = &ctx->flush->slot[fc];
  const size_t tile_bytes =
    ctx->layout->tile_stride * ctx->config->bytes_per_element;

  CU(Error,
     cuStreamWaitEvent(
       ctx->streams.compress, ctx->batch->pool_events[epoch_in_batch], 0));
  if (ctx->levels->enable_multiscale && ctx->lod->t_end)
    CU(Error, cuStreamWaitEvent(ctx->streams.compress, ctx->lod->t_end, 0));

  // Compress single epoch
  void* epoch_pool = (char*)ctx->pools->buf[fc] + (uint64_t)epoch_in_batch *
                                                    ctx->levels->total_tiles *
                                                    tile_bytes;
  CHECK(Error,
        kick_compress(ctx, fs, epoch_pool, ctx->levels->total_tiles) == 0);

  // Aggregate per level
  for (int lv = 0; lv < ctx->levels->nlod; ++lv) {
    if (!(active_mask & (1u << lv)))
      continue;
    CHECK(Error, aggregate_epoch_level(ctx, fc, lv, 0) == 0);
  }

  CU(Error, cuEventRecord(fs->t_aggregate_end, ctx->streams.compress));

  // Wait for previous IO fences before D2H
  wait_io_fences(ctx, fc, active_mask);

  CU(Error, cuStreamWaitEvent(ctx->streams.d2h, fs->t_aggregate_end, 0));
  CU(Error, cuEventRecord(fs->t_d2h_start, ctx->streams.d2h));
  CHECK(Error, two_phase_d2h(ctx, fs, fc, active_mask, 1) == 0);

  // Set up for sync_and_deliver
  fs->batch_epoch_count = 1;
  fs->active_levels_mask = active_mask;
  fs->batch_active_masks[0] = active_mask;

  return sync_and_deliver(ctx, fc);

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
  return wait_and_deliver(ctx, ctx->flush->current);
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

    struct writer_result r = wait_and_deliver(ctx, fc);
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
  const size_t bpe = ctx->config->bytes_per_element;
  enum lod_dtype dtype = (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

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

  // Zero LOD level regions of pool, emit partial accums, scatter to tiles
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    uint64_t n_elements = p->batch_count * p->lod_counts[lv];

    // Compute offset of this level within the packed accumulator
    uint64_t accum_offset = 0;
    for (int k = 1; k < lv; ++k)
      accum_offset += p->batch_count * p->lod_counts[k];

    size_t accum_bpe = lod_accum_bpe(bpe, ctx->config->dim0_reduce_method);

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

    // Zero and scatter to tile pool (epoch 0 in current pool)
    CUdeviceptr dst =
      ctx->pools->buf[ctx->pools->current] +
      ctx->levels->tile_offset[lv] * ctx->layout->tile_stride * bpe;
    size_t lv_pool_bytes =
      ctx->levels->tile_count[lv] * ctx->layout->tile_stride * bpe;
    CU(Error, cuMemsetD8Async(dst, 0, lv_pool_bytes, ctx->streams.compute));

    CHECK(Error,
          lod_morton_to_tiles_lut(dst,
                                  morton_lv,
                                  ctx->lod->d_morton_tile_lut[lv],
                                  ctx->lod->d_morton_batch_tile_offsets[lv],
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
  return wait_and_deliver(ctx, fc);

Error:
  return writer_error();
}

#include "gpu/stream.flush.h"

#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.lod.h"

#include "gpu/lod.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <string.h>

// --- Helpers ---

// Build compress_agg_input from current state.
static struct compress_agg_input
make_compress_input(struct stream_engine* e,
                    struct stream_context* ctx,
                    int fc,
                    uint32_t n_epochs)
{
  struct flush_slot_gpu* fs = &e->flush.slot[fc];
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = fs->active_levels_mask,
    .epochs_per_batch = e->batch.epochs_per_batch,
    .pool_buf = e->pools.buf[fc],
    .lod_done = (ctx->levels.enable_multiscale && e->lod.timing[fc].t_end)
                  ? e->lod.timing[fc].t_end
                  : NULL,
  };
  memcpy(
    in.batch_active_masks, fs->batch_active_masks, n_epochs * sizeof(uint32_t));
  memcpy(in.epoch_events, e->batch.pool_events, n_epochs * sizeof(CUevent));
  return in;
}

// --- Epoch accumulation ---

// Return pointer to the current epoch's chunk region within the super-pool.
static inline void*
pool_epoch_ptr(struct stream_engine* e,
               struct stream_context* ctx,
               uint32_t epoch_in_batch)
{
  const size_t bytes_per_element = dtype_bpe(ctx->config.dtype);
  return (char*)e->pools.buf[e->pools.current] +
         (uint64_t)epoch_in_batch * ctx->levels.total_chunks *
           ctx->layout.chunk_stride * bytes_per_element;
}

// Run LOD pipeline for the current epoch, or handle non-multiscale case.
// Updates flush slot batch_active_masks and active_levels_mask.
int
flush_run_epoch_lod(struct stream_engine* e, struct stream_context* ctx)
{
  struct flush_slot_gpu* fs = &e->flush.slot[e->pools.current];
  uint32_t active_mask;

  if (!ctx->levels.enable_multiscale || !e->lod.d_linear) {
    // Non-multiscale: all levels (just L0) are active
    active_mask = 1;
  } else {
    CHECK(Error,
          lod_run_epoch(&e->lod,
                        e->pools.current,
                        &ctx->levels,
                        pool_epoch_ptr(e, ctx, e->batch.accumulated),
                        ctx->config.dtype,
                        ctx->config.reduce_method,
                        ctx->config.append_reduce_method,
                        &ctx->dims,
                        e->streams.compute,
                        &active_mask) == 0);
  }

  fs->batch_active_masks[e->batch.accumulated] = active_mask;
  fs->active_levels_mask |= active_mask;
  return 0;

Error:
  return 1;
}

// Pipeline the batch boundary: launch compress+aggregate for the new batch
// on the compress stream, drain the previous batch's bulk D2H + delivery on
// the d2h stream, then queue the new batch's offset D2H.  This ordering
// keeps the d2h stream free for the previous batch's bulk transfer before
// the new batch's aggregate-end wait occupies it.
static struct writer_result
drain_kick_and_swap(struct stream_engine* e, struct stream_context* ctx)
{
  const uint32_t K = e->batch.epochs_per_batch;
  const int completed_pool = e->pools.current;
  struct flush_slot_gpu* fs = &e->flush.slot[completed_pool];

  // Save the previous pending state so we can drain it after the
  // compress+aggregate kick but before the d2h kick.
  struct flush_handoff prev_handoff = e->flush.pending_handoff;
  int had_pending = e->flush.pending;
  e->flush.pending = 0;

  // Phase A: kick compress+aggregate for the new batch (compress stream).
  // This starts GPU work immediately — no host blocking.
  fs->batch_epoch_count = (int)e->batch.accumulated;
  struct compress_agg_input in =
    make_compress_input(e, ctx, completed_pool, e->batch.accumulated);
  struct flush_handoff new_handoff = { 0 };
  CHECK(Error,
        compress_agg_kick(&e->compress_agg,
                          &in,
                          &ctx->levels,
                          &e->batch,
                          &ctx->dims,
                          e->streams.compress,
                          &new_handoff) == 0);

  // Phase B: drain the PREVIOUS batch.  This issues bulk D2H on d2h_stream
  // (unblocked — the new batch's aggregate-end wait hasn't been queued yet).
  if (had_pending) {
    struct platform_clock stall_clk = { 0 };
    platform_toc(&stall_clk);
    struct writer_result r = d2h_deliver_drain(&e->d2h_deliver,
                                               &prev_handoff,
                                               &ctx->levels,
                                               &e->batch,
                                               &ctx->dims,
                                               &ctx->layout,
                                               &ctx->config,
                                               ctx->sink,
                                               &e->lod,
                                               &e->metrics,
                                               &e->metadata_update_clock);
    float ms = (float)(platform_toc(&stall_clk) * 1000.0);
    accumulate_metric_ms(&e->metrics.flush_stall, ms, 0, 0);
    if (r.error)
      return r;
  }

  // Phase C: queue the new batch's offset D2H on d2h_stream (non-blocking).
  // This goes AFTER the previous drain's bulk D2H on d2h_stream, avoiding
  // the aggregate-end wait from blocking the prior batch's transfer.
  CHECK(Error,
        d2h_deliver_kick(&e->d2h_deliver,
                         &new_handoff,
                         &ctx->levels,
                         &e->batch,
                         &ctx->dims,
                         e->streams.d2h) == 0);

  // Save handoff for the next drain.
  e->flush.pending_handoff = new_handoff;
  e->flush.pending = 1;
  e->flush.current = completed_pool;

  // Swap to fresh pool and zero it for next batch.
  e->pools.current ^= 1;
  size_t pool_bytes = (uint64_t)K * ctx->levels.total_chunks *
                      ctx->layout.chunk_stride * dtype_bpe(ctx->config.dtype);
  CU(Error,
     cuMemsetD8Async(
       e->pools.buf[e->pools.current], 0, pool_bytes, e->streams.compute));

  // Reset batch accumulation.
  e->batch.accumulated = 0;
  e->flush.slot[e->pools.current].active_levels_mask = 0;
  memset(e->flush.slot[e->pools.current].batch_active_masks,
         0,
         sizeof(e->flush.slot[e->pools.current].batch_active_masks));

  return writer_ok();

Error:
  return writer_error();
}

// Accumulate one epoch into the current batch, or flush when batch is full.
// Called at each epoch boundary.
struct writer_result
flush_accumulate_epoch(struct stream_engine* e, struct stream_context* ctx)
{
  if (flush_run_epoch_lod(e, ctx))
    return writer_error();

  CU(Error,
     cuEventRecord(e->batch.pool_events[e->batch.accumulated],
                   e->streams.compute));
  e->batch.accumulated++;

  if (e->batch.accumulated < e->batch.epochs_per_batch)
    return writer_ok();

  if (e->sync_flush) {
    // Synchronous path: flush the full batch immediately (no pool swap).
    // Used by multiarray where double-buffered pipeline state doesn't
    // compose across array switches.
    struct writer_result r = flush_accumulated_sync(e, ctx);
    // Zero pool for next batch.
    if (!r.error) {
      size_t bpe = dtype_bpe(ctx->config.dtype);
      size_t pool_bytes = (uint64_t)e->batch.epochs_per_batch *
                          ctx->levels.total_chunks * ctx->layout.chunk_stride *
                          bpe;
      CU(SyncError,
         cuMemsetD8Async(
           e->pools.buf[e->pools.current], 0, pool_bytes, e->streams.compute));
    }
    return r;
  SyncError:
    return writer_error();
  }

  return drain_kick_and_swap(e, ctx);

Error:
  return writer_error();
}

// --- Batch flush pipeline ---

// Kick compress + aggregate + D2H for a batch of n_epochs epochs.
int
flush_kick_batch(struct stream_engine* e,
                 struct stream_context* ctx,
                 int fc,
                 uint32_t n_epochs)
{
  struct compress_agg_input in = make_compress_input(e, ctx, fc, n_epochs);
  struct flush_handoff handoff = { 0 };

  CHECK(Error,
        compress_agg_kick(&e->compress_agg,
                          &in,
                          &ctx->levels,
                          &e->batch,
                          &ctx->dims,
                          e->streams.compress,
                          &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(&e->d2h_deliver,
                         &handoff,
                         &ctx->levels,
                         &e->batch,
                         &ctx->dims,
                         e->streams.d2h) == 0);

  // Save handoff for drain
  e->flush.pending_handoff = handoff;

  return 0;

Error:
  return 1;
}

// --- Public interface ---

struct writer_result
flush_drain_pending(struct stream_engine* e, struct stream_context* ctx)
{
  if (!e->flush.pending)
    return writer_ok();

  e->flush.pending = 0;
  return d2h_deliver_drain(&e->d2h_deliver,
                           &e->flush.pending_handoff,
                           &ctx->levels,
                           &e->batch,
                           &ctx->dims,
                           &ctx->layout,
                           &ctx->config,
                           ctx->sink,
                           &e->lod,
                           &e->metrics,
                           &e->metadata_update_clock);
}

struct writer_result
flush_accumulated_sync(struct stream_engine* e, struct stream_context* ctx)
{
  if (e->batch.accumulated == 0)
    return writer_ok();

  const int fc = e->pools.current;
  struct flush_slot_gpu* fs = &e->flush.slot[fc];

  fs->batch_epoch_count = (int)e->batch.accumulated;
  if (flush_kick_batch(e, ctx, fc, e->batch.accumulated))
    return writer_error();

  struct writer_result r = d2h_deliver_drain(&e->d2h_deliver,
                                             &e->flush.pending_handoff,
                                             &ctx->levels,
                                             &e->batch,
                                             &ctx->dims,
                                             &ctx->layout,
                                             &ctx->config,
                                             ctx->sink,
                                             &e->lod,
                                             &e->metrics,
                                             &e->metadata_update_clock);
  if (r.error)
    return r;

  e->batch.accumulated = 0;
  e->flush.slot[e->pools.current].active_levels_mask = 0;
  memset(e->flush.slot[e->pools.current].batch_active_masks,
         0,
         sizeof(e->flush.slot[e->pools.current].batch_active_masks));
  return r;
}

struct writer_result
flush_partial_append(struct stream_engine* e, struct stream_context* ctx)
{
  if (!ctx->dims.append_downsample || !ctx->levels.enable_multiscale)
    return writer_ok();

  const struct lod_plan* p = &e->lod.plan;
  const size_t bytes_per_element = dtype_bpe(ctx->config.dtype);
  const enum dtype dtype = ctx->config.dtype;

  // Check if any level has pending data
  uint32_t active_levels_mask = 0;
  for (int lv = 1; lv < p->levels.nlod; ++lv) {
    if (e->lod.append_accum.counts[lv] > 0)
      active_levels_mask |= (1u << lv);
  }

  if (!active_levels_mask)
    return writer_ok();

  const int fc = e->pools.current;
  struct flush_slot_gpu* fs = &e->flush.slot[fc];
  fs->active_levels_mask = active_levels_mask;
  fs->batch_active_masks[0] = active_levels_mask;
  fs->batch_epoch_count = 1;

  for (int lv = 1; lv < p->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    uint64_t n_elements =
      p->levels.level[lv].fixed_dims_count * p->levels.level[lv].lod_nelem;

    uint64_t accum_offset = 0;
    for (int k = 1; k < lv; ++k)
      accum_offset +=
        p->levels.level[k].fixed_dims_count * p->levels.level[k].lod_nelem;

    size_t accum_bpe = dtype_bpe(dtype);

    struct lod_span lev = lod_spans_at(&p->level_spans, lv);
    CUdeviceptr morton_lv = e->lod.d_morton + lev.beg * bytes_per_element;
    CUdeviceptr accum_lv =
      e->lod.append_accum.d_accum + accum_offset * accum_bpe;

    CHECK(Error,
          lod_accum_emit(morton_lv,
                         accum_lv,
                         dtype,
                         ctx->config.append_reduce_method,
                         n_elements,
                         e->lod.append_accum.counts[lv],
                         e->streams.compute) == 0);

    e->lod.append_accum.counts[lv] = 0;

    CUdeviceptr dst = e->pools.buf[e->pools.current] +
                      ctx->levels.level[lv].chunk_offset *
                        ctx->layout.chunk_stride * bytes_per_element;
    size_t lv_pool_bytes = ctx->levels.level[lv].chunk_count *
                           ctx->layout.chunk_stride * bytes_per_element;
    CU(Error, cuMemsetD8Async(dst, 0, lv_pool_bytes, e->streams.compute));

    CHECK(Error,
          lod_morton_to_chunks_lut(dst,
                                   morton_lv,
                                   e->lod.d_morton_chunk_lut[lv],
                                   e->lod.d_morton_fixed_dims_chunk_offsets[lv],
                                   dtype,
                                   p->levels.level[lv].lod_nelem,
                                   p->levels.level[lv].fixed_dims_count,
                                   e->streams.compute) == 0);
  }

  CU(Error, cuEventRecord(e->pools.ready[fc], e->streams.compute));
  if (e->lod.timing[fc].t_end)
    CU(Error, cuEventRecord(e->lod.timing[fc].t_end, e->streams.compute));

  CU(Error, cuEventRecord(e->batch.pool_events[0], e->streams.compute));
  if (flush_kick_batch(e, ctx, fc, 1))
    return writer_error();
  return d2h_deliver_drain(&e->d2h_deliver,
                           &e->flush.pending_handoff,
                           &ctx->levels,
                           &e->batch,
                           &ctx->dims,
                           &ctx->layout,
                           &ctx->config,
                           ctx->sink,
                           &e->lod,
                           &e->metrics,
                           &e->metadata_update_clock);

Error:
  return writer_error();
}

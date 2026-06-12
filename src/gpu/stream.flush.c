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
  return (struct compress_agg_input){
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = fs->active_levels_mask,
    .batch_active_masks = fs->batch_active_masks,
    .pool = &e->pools.p,
    .lod_active = ctx->levels.enable_multiscale &&
                  gpu_ordering_event(&e->ord, GPU_EDGE_LOD_DONE, fc) != NULL,
    .epochs_per_batch = e->batch.epochs_per_batch,
  };
}

// --- Epoch accumulation ---

// Run LOD pipeline for the current epoch, or handle non-multiscale case.
// Updates flush slot batch_active_masks and active_levels_mask.
int
flush_run_epoch_lod(struct stream_engine* e, struct stream_context* ctx)
{
  struct flush_slot_gpu* fs = &e->flush.slot[e->pools.current];
  uint32_t active_mask;

  if (!ctx->levels.enable_multiscale || !e->lod_shared.d_linear) {
    // Non-multiscale: all levels (just L0) are active
    active_mask = 1;
  } else {
    CHECK(Error,
          lod_run_epoch(&e->lod,
                        &e->lod_shared,
                        &e->ord,
                        e->pools.current,
                        &ctx->levels,
                        stream_engine_pool_epoch(e, ctx, e->batch.accumulated),
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

// Drain one pending fc slot: host-sync on ready[fc] (near-zero in steady
// state), run delivery, clear pending. Accumulates flush_stall metric.
static struct writer_result
drain_fc(struct stream_engine* e, struct stream_context* ctx, int fc)
{
  if (!e->flush.pending[fc])
    return writer_ok();

  // The tail gate's GEQ threshold assumes drains follow kick order.
  gpu_edge_host_rule(&e->ord,
                     GPU_EDGE_DELIVER_OLDEST_FIRST,
                     !e->flush.pending[fc ^ 1] ||
                       e->flush.pending_seq[fc] < e->flush.pending_seq[fc ^ 1]);

  struct platform_clock stall_clk = { 0 };
  platform_toc(&stall_clk);
  struct writer_result r = d2h_deliver_drain(&e->d2h_deliver,
                                             &e->flush.pending_handoff[fc],
                                             &ctx->levels,
                                             &ctx->dims,
                                             &ctx->layout,
                                             &ctx->config,
                                             ctx->sink,
                                             &e->lod,
                                             &e->lod_shared,
                                             &e->metrics,
                                             &e->metadata_update_clock,
                                             e->streams.d2h);
  float ms = (float)(platform_toc(&stall_clk) * 1000.0);
  accumulate_metric_ms(&e->metrics.flush_stall, ms, 0, 0);

  e->flush.pending[fc] = 0;
  return r;
}

// Pipeline the batch boundary (lazy drain model):
//   1. Deliver the PRIOR batch at this fc (if any). In steady state that
//      batch's D2H has long since completed, so the host-sync is near-zero.
//   2. Kick compress+aggregate for the new batch on this fc.
//   3. Kick the full D2H (offset + bulk) on the d2h stream — non-blocking.
//   4. Record the new handoff as pending for this fc; the drain of this
//      batch will happen when fc is reused (typically 2 rounds later).
//   5. Swap pools, zero the fresh one, reset batch accumulation.
// Shard writes stay in batch order because we always deliver the fc we're
// about to reuse, which holds the oldest undelivered batch.
static struct writer_result
drain_kick_and_swap(struct stream_engine* e, struct stream_context* ctx)
{
  const uint32_t K = e->batch.epochs_per_batch;
  const int completed_pool = e->pools.current;
  struct flush_slot_gpu* fs = &e->flush.slot[completed_pool];

  // Phase 1: deliver the PRIOR batch at fc=completed_pool (if any).
  {
    struct writer_result r = drain_fc(e, ctx, completed_pool);
    if (r.error)
      return r;
  }

  // Tail-gate fallback: without the gate (stream memops unsupported, or
  // the counter failed to map at init) the tail upload cannot be ordered
  // device-side (gate state in gpu_ordering, src/gpu/ordering.h), so also
  // drain the other, newer pending batch — order stays oldest-first,
  // pipeline degrades to depth 1 — and the kick below sees published tail
  // state.
  if (e->compress_agg.ar.page_size > 0 &&
      e->compress_agg.ar.total_shards > 0 &&
      !gpu_ordering_gate_supported(&e->ord)) {
    struct writer_result r = drain_fc(e, ctx, completed_pool ^ 1);
    if (r.error)
      return r;
  }

  gpu_edge_host_rule(&e->ord,
                     GPU_EDGE_DRAIN_BEFORE_REKICK,
                     !e->flush.pending[completed_pool]);
  fs->batch_epoch_count = (int)e->batch.accumulated;
  struct compress_agg_input in =
    make_compress_input(e, ctx, completed_pool, e->batch.accumulated);
  struct flush_handoff new_handoff = { 0 };
  CHECK(Error,
        compress_agg_kick(&e->compress_agg,
                          &in,
                          &ctx->levels,
                          e->streams.compress,
                          &new_handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(&e->d2h_deliver,
                         &new_handoff,
                         ctx->sink,
                         e->streams.d2h) == 0);

  // Phase 4: mark this fc pending delivery.
  e->flush.pending_handoff[completed_pool] = new_handoff;
  e->flush.pending_seq[completed_pool] = e->flush.next_seq++;
  e->flush.pending[completed_pool] = 1;

  // Phase 5: swap to fresh pool and zero it for next batch.
  e->pools.current ^= 1;
  // The aggregate's last pool read gates reuse; re-zeroing any earlier
  // corrupts the in-flight batch (#140).
  struct gpu_pool_view fresh;
  CHECK(Error,
        gpu_pool_acquire_produce(
          &e->pools.p, e->pools.current, e->streams.compute, &fresh) == 0);
  size_t pool_bytes = (uint64_t)K * ctx->levels.total_chunks *
                      ctx->layout.chunk_stride * dtype_bpe(ctx->config.dtype);
  CU(Error,
     cuMemsetD8Async(gpu_pool_view_d(fresh), 0, pool_bytes, e->streams.compute));

  // Reset batch accumulation.
  e->batch.accumulated = 0;
  e->flush.slot[e->pools.current].active_levels_mask = 0;
  memset(e->flush.slot[e->pools.current].batch_active_masks,
         0,
         (size_t)K * sizeof(uint32_t));

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

  e->batch.accumulated++;

  if (e->batch.accumulated < e->batch.epochs_per_batch)
    return writer_ok();

  // Release pool-filled once after the last epoch's scatter; compute-stream
  // ordering means this subsumes per-epoch ready signals.
  CHECK(Error,
        gpu_pool_release_produce(
          &e->pools.p, e->pools.current, e->streams.compute) == 0);

  if (e->sync_flush) {
    // Synchronous path: flush the full batch immediately (no pool swap).
    // Used by multiarray where double-buffered pipeline state doesn't
    // compose across array switches.
    struct writer_result r = flush_accumulated_sync(e, ctx);
    // Host-ordered re-acquire: the sync drain above host-completed this
    // fc's D2H, so no device wait is queued.
    if (!r.error) {
      size_t bpe = dtype_bpe(ctx->config.dtype);
      size_t pool_bytes = (uint64_t)e->batch.epochs_per_batch *
                          ctx->levels.total_chunks * ctx->layout.chunk_stride *
                          bpe;
      CU(SyncError,
         cuMemsetD8Async(
           gpu_pool_view_d(gpu_pool_at(&e->pools.p, e->pools.current, 0)),
           0,
           pool_bytes,
           e->streams.compute));
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

// Kick compress + aggregate + full D2H for a batch of n_epochs epochs on
// the given fc. Records the handoff as pending for this fc. Caller must
// drain pending[fc] before calling again on the same fc.
int
flush_kick_batch(struct stream_engine* e,
                 struct stream_context* ctx,
                 int fc,
                 uint32_t n_epochs)
{
  gpu_edge_host_rule(&e->ord, GPU_EDGE_DRAIN_BEFORE_REKICK,
                     !e->flush.pending[fc]);
  struct compress_agg_input in = make_compress_input(e, ctx, fc, n_epochs);
  struct flush_handoff handoff = { 0 };

  CHECK(Error,
        compress_agg_kick(&e->compress_agg,
                          &in,
                          &ctx->levels,
                          e->streams.compress,
                          &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(&e->d2h_deliver,
                         &handoff,
                         ctx->sink,
                         e->streams.d2h) == 0);

  // Save handoff for drain; mark this fc pending.
  e->flush.pending_handoff[fc] = handoff;
  e->flush.pending_seq[fc] = e->flush.next_seq++;
  e->flush.pending[fc] = 1;

  return 0;

Error:
  return 1;
}

// --- Public interface ---

// Drain all pending batches in kick order. Called at stream flush.
struct writer_result
flush_drain_pending(struct stream_engine* e, struct stream_context* ctx)
{
  // Up to 2 pending fcs (one per slot); drain oldest first.
  for (int i = 0; i < 2; ++i) {
    int pick = -1;
    uint64_t pick_seq = UINT64_MAX;
    for (int fc = 0; fc < 2; ++fc) {
      if (e->flush.pending[fc] && e->flush.pending_seq[fc] < pick_seq) {
        pick = fc;
        pick_seq = e->flush.pending_seq[fc];
      }
    }
    if (pick < 0)
      break;
    struct writer_result r = drain_fc(e, ctx, pick);
    if (r.error)
      return r;
  }
  return writer_ok();
}

struct writer_result
flush_accumulated_sync(struct stream_engine* e, struct stream_context* ctx)
{
  if (e->batch.accumulated == 0)
    return writer_ok();

  const int fc = e->pools.current;
  struct flush_slot_gpu* fs = &e->flush.slot[fc];

  fs->batch_epoch_count = (int)e->batch.accumulated;

  // Release pool-filled after all scatter ops for this (possibly partial)
  // batch.
  if (gpu_pool_release_produce(&e->pools.p, e->pools.current, e->streams.compute))
    return writer_error();

  if (flush_kick_batch(e, ctx, fc, e->batch.accumulated))
    return writer_error();

  struct writer_result r = drain_fc(e, ctx, fc);
  if (r.error)
    return r;

  e->batch.accumulated = 0;
  e->flush.slot[e->pools.current].active_levels_mask = 0;
  memset(e->flush.slot[e->pools.current].batch_active_masks,
         0,
         (size_t)e->batch.epochs_per_batch * sizeof(uint32_t));
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
    CUdeviceptr morton_lv =
      e->lod_shared.d_morton + lev.beg * bytes_per_element;
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

    // Produce-phase write within the generation acquired at the last swap
    // (the current pool is still being filled).
    CUdeviceptr dst =
      gpu_pool_view_d(gpu_pool_at(&e->pools.p,
                                  e->pools.current,
                                  ctx->levels.level[lv].chunk_offset *
                                    ctx->layout.chunk_stride *
                                    bytes_per_element));
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

  if (gpu_ordering_event(&e->ord, GPU_EDGE_LOD_DONE, fc))
    CHECK(Error,
          gpu_edge_record(
            &e->ord, GPU_EDGE_LOD_DONE, fc, e->streams.compute) == 0);

  CHECK(Error,
        gpu_pool_release_produce(
          &e->pools.p, e->pools.current, e->streams.compute) == 0);
  if (flush_kick_batch(e, ctx, fc, 1))
    return writer_error();
  return drain_fc(e, ctx, fc);

Error:
  return writer_error();
}

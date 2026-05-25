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
                    int output_idx,
                    uint32_t n_epochs)
{
  struct flush_slot_gpu* fs = &e->flush.slot[fc];
  return (struct compress_agg_input){
    .fc = fc,
    .output_idx = output_idx,
    .n_epochs = n_epochs,
    .active_levels_mask = fs->active_levels_mask,
    .batch_active_masks = fs->batch_active_masks,
    .pool_buf = e->pools.buf[fc],
    .pool_ready = e->batch.pool_ready,
    .lod_done =
      (ctx->levels.enable_multiscale && e->lod_shared.timing[fc].t_end)
        ? e->lod_shared.timing[fc].t_end
        : NULL,
    .prev_d2h_done = e->d2h_deliver.ready[output_idx],
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

// Resets slot state on success so the next kick starts fresh. B>1 stacking
// skips drain between in-slot batches, preserving the cursor.
static struct writer_result
drain_output(struct stream_engine* e, struct stream_context* ctx, int oi)
{
  if (!e->flush.pending[oi])
    return writer_ok();

  struct platform_clock stall_clk = { 0 };
  platform_toc(&stall_clk);
  struct writer_result r = d2h_deliver_drain(&e->d2h_deliver,
                                             &e->flush.pending_handoff[oi],
                                             &ctx->levels,
                                             &e->batch,
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

  e->flush.pending[oi] = 0;
  output_slot_close_reset(&e->compress_agg.output[oi]);
  e->compress_agg.slot_host_acc_desc_entries[oi] = 0;
  return r;
}

static int
kick_compress_agg(struct stream_engine* e,
                  struct stream_context* ctx,
                  int fc,
                  int output_idx,
                  uint32_t n_epochs,
                  struct flush_handoff* handoff_out)
{
  struct compress_agg_input in =
    make_compress_input(e, ctx, fc, output_idx, n_epochs);
  return compress_agg_kick(&e->compress_agg,
                           &in,
                           &ctx->levels,
                           &e->batch,
                           &ctx->dims,
                           e->streams.compress,
                           handoff_out);
}

static int
kick_d2h_and_mark_pending(struct stream_engine* e,
                          struct stream_context* ctx,
                          int output_idx,
                          const struct flush_handoff* handoff);

static int
post_kick_review(struct stream_engine* e,
                 struct stream_context* ctx,
                 int active_oi,
                 struct flush_handoff* scratch,
                 int* out_target)
{
  CU(Error,
     cuEventSynchronize(e->compress_agg.output[active_oi].host_func_done));
  const volatile struct d_routing* r = e->compress_agg.h_routing;
  const int target = r->target_slot_idx;
  const int close = r->close_prior_slot_idx;
  const uint32_t cap = e->compress_agg.output[active_oi].batches_per_slot_cap;
  if (cap == 1) {
    CHECK(Error, target == active_oi);
    CHECK(Error, close == -1);
    CHECK(Error, e->compress_agg.h_close_signal[0] == 0);
    CHECK(Error, e->compress_agg.h_close_signal[1] == 0);
  }
  if (close >= 0) {
    CHECK(Error, close != target);
    CHECK(Error,
          kick_d2h_and_mark_pending(
            e, ctx, close, &e->flush.pending_handoff[close]) == 0);
    e->compress_agg.h_close_signal[close] = 0;
  }
  scratch->output_idx = target;
  scratch->output = &e->compress_agg.output[target];
  e->flush.pending_handoff[target] = *scratch;
  *out_target = target;
  return 0;
Error:
  return 1;
}

static int
kick_d2h_and_mark_pending(struct stream_engine* e,
                          struct stream_context* ctx,
                          int output_idx,
                          const struct flush_handoff* handoff)
{
  CHECK(Error,
        d2h_deliver_kick(&e->d2h_deliver,
                         handoff,
                         &ctx->levels,
                         &e->batch,
                         &ctx->dims,
                         ctx->sink,
                         e->streams.d2h) == 0);
  e->flush.pending_seq[output_idx] = e->flush.next_seq++;
  e->flush.pending[output_idx] = 1;
  return 0;
Error:
  return 1;
}

static int
pool_swap_and_reset_accum(struct stream_engine* e, struct stream_context* ctx)
{
  const uint32_t K = e->batch.epochs_per_batch;
  e->pools.current ^= 1;
  size_t pool_bytes = (uint64_t)K * ctx->levels.total_chunks *
                      ctx->layout.chunk_stride * dtype_bpe(ctx->config.dtype);
  CU(Error,
     cuMemsetD8Async(
       e->pools.buf[e->pools.current], 0, pool_bytes, e->streams.compute));
  e->batch.accumulated = 0;
  e->flush.slot[e->pools.current].active_levels_mask = 0;
  memset(e->flush.slot[e->pools.current].batch_active_masks,
         0,
         (size_t)K * sizeof(uint32_t));
  return 0;
Error:
  return 1;
}

// Batch order is preserved by draining the slot we're about to reuse.
static struct writer_result
drain_kick_and_swap(struct stream_engine* e, struct stream_context* ctx)
{
  const int fc = e->pools.current;
  const int oi = e->flush.output_current;
  struct flush_slot_gpu* fs = &e->flush.slot[fc];

  {
    struct writer_result r = drain_output(e, ctx, oi);
    if (r.error)
      return r;
  }

  fs->batch_epoch_count = (int)e->batch.accumulated;
  struct flush_handoff scratch = { 0 };
  CHECK(Error,
        kick_compress_agg(e, ctx, fc, oi, e->batch.accumulated, &scratch) == 0);

  int target = -1;
  CHECK(Error, post_kick_review(e, ctx, oi, &scratch, &target) == 0);

  const uint32_t cap = e->compress_agg.output[oi].batches_per_slot_cap;
  if (cap > 1) {
    e->flush.output_current = target;
  } else {
    CHECK(Error,
          kick_d2h_and_mark_pending(
            e, ctx, target, &e->flush.pending_handoff[target]) == 0);
    e->flush.output_current ^= 1;
  }
  CHECK(Error, pool_swap_and_reset_accum(e, ctx) == 0);

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

  // Record pool_ready once after the last epoch's scatter; compute-stream
  // ordering means this subsumes per-epoch ready signals.
  CU(Error, cuEventRecord(e->batch.pool_ready, e->streams.compute));

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

int
flush_kick_batch(struct stream_engine* e,
                 struct stream_context* ctx,
                 int fc,
                 int output_idx,
                 uint32_t n_epochs)
{
  struct flush_handoff scratch = { 0 };
  CHECK(Error,
        kick_compress_agg(e, ctx, fc, output_idx, n_epochs, &scratch) == 0);
  int target = -1;
  CHECK(Error, post_kick_review(e, ctx, output_idx, &scratch, &target) == 0);
  CHECK(Error,
        kick_d2h_and_mark_pending(
          e, ctx, target, &e->flush.pending_handoff[target]) == 0);
  return 0;

Error:
  return 1;
}

// --- Public interface ---

struct writer_result
flush_drain_pending(struct stream_engine* e, struct stream_context* ctx)
{
  for (int oi = 0; oi < 2; ++oi) {
    if (!e->flush.pending[oi] &&
        e->compress_agg.slot_host_acc_desc_entries[oi] > 0) {
      if (kick_d2h_and_mark_pending(
            e, ctx, oi, &e->flush.pending_handoff[oi]) != 0)
        return writer_error();
    }
  }
  for (int i = 0; i < 2; ++i) {
    int pick = -1;
    uint64_t pick_seq = UINT64_MAX;
    for (int oi = 0; oi < 2; ++oi) {
      if (e->flush.pending[oi] && e->flush.pending_seq[oi] < pick_seq) {
        pick = oi;
        pick_seq = e->flush.pending_seq[oi];
      }
    }
    if (pick < 0)
      break;
    struct writer_result r = drain_output(e, ctx, pick);
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
  const int oi = e->flush.output_current;
  struct flush_slot_gpu* fs = &e->flush.slot[fc];

  fs->batch_epoch_count = (int)e->batch.accumulated;

  if (cuEventRecord(e->batch.pool_ready, e->streams.compute) != CUDA_SUCCESS)
    return writer_error();

  if (flush_kick_batch(e, ctx, fc, oi, e->batch.accumulated))
    return writer_error();

  struct writer_result r = drain_output(e, ctx, oi);
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
  const int oi = e->flush.output_current;
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
  if (e->lod_shared.timing[fc].t_end)
    CU(Error,
       cuEventRecord(e->lod_shared.timing[fc].t_end, e->streams.compute));

  CU(Error, cuEventRecord(e->batch.pool_ready, e->streams.compute));
  if (flush_kick_batch(e, ctx, fc, oi, 1))
    return writer_error();
  return drain_output(e, ctx, oi);

Error:
  return writer_error();
}

#include "gpu/schedule.h"

#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.lod.h"

#include "gpu/lod.h"
#include "gpu/ordering.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <string.h>

// --- Streams ---

int
gpu_streams_init(struct gpu_streams* s)
{
  CU(Fail, cuStreamCreate(&s->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->d2h, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->drain, CU_STREAM_NON_BLOCKING));
  return 0;
Fail:
  return 1;
}

void
gpu_streams_destroy(struct gpu_streams* s)
{
  cu_stream_destroy(s->h2d);
  cu_stream_destroy(s->compute);
  cu_stream_destroy(s->compress);
  cu_stream_destroy(s->d2h);
  cu_stream_destroy(s->drain);
}

void
gpu_streams_register(const struct gpu_streams* s, struct gpu_ordering* ord)
{
  gpu_ordering_register_stream(ord, GPU_STREAM_H2D, s->h2d);
  gpu_ordering_register_stream(ord, GPU_STREAM_COMPUTE, s->compute);
  gpu_ordering_register_stream(ord, GPU_STREAM_COMPRESS, s->compress);
  gpu_ordering_register_stream(ord, GPU_STREAM_D2H, s->d2h);
  gpu_ordering_register_stream(ord, GPU_STREAM_DRAIN, s->drain);
}

// --- Selection ---

void
schedule_select(struct gpu_scheduler* sched,
                const struct compress_agg_array* ar,
                const struct gpu_ordering* gate_ord)
{
  const int page_aligned = ar->page_size > 0 && ar->total_shards > 0;
  if (!gate_ord)
    sched->depth = SCHEDULE_DRAIN_AFTER_KICK;
  else if (page_aligned && !gpu_ordering_gate_supported(gate_ord))
    sched->depth = SCHEDULE_DRAIN_BEFORE_KICK;
  else
    sched->depth = SCHEDULE_PIPELINED;
}

int
schedule_lod_active(const struct gpu_ordering* ord, int enable_multiscale)
{
  return enable_multiscale &&
         gpu_ordering_event(ord, GPU_EDGE_LOD_DONE, 0) != NULL;
}

// --- Compress+aggregate kick ---

int
schedule_compress_agg_kick(struct compress_agg_stage* stage,
                           const struct compress_agg_input* in,
                           const struct level_geometry* levels,
                           struct gpu_pool* chunk_pool,
                           int lod_active,
                           CUstream compress_stream,
                           struct flush_handoff* out)
{
  struct compress_agg_plan plan;
  CHECK(Error,
        compress_agg_prepare(stage, in, levels, compress_stream, &plan) == 0);

  struct gpu_pool_view pool_buf;
  CHECK(Error,
        gpu_pool_acquire_consume(
          chunk_pool, in->fc, compress_stream, &pool_buf) == 0);
  // LOD chunks land in the pool through a second producer edge.
  if (lod_active)
    CHECK(Error,
          gpu_edge_wait(
            chunk_pool->ord, GPU_EDGE_LOD_DONE, in->fc, compress_stream) == 0);
  // Aggregate must not overwrite agg[fc] before the prior D2H on the same
  // fc has read it.
  struct gpu_pool_view slot;
  CHECK(Error,
        gpu_pool_acquire_produce(
          &stage->agg_pool, in->fc, compress_stream, &slot) == 0);

  CHECK(Error,
        compress_agg_compress(stage, in, levels, pool_buf, compress_stream) ==
          0);

  // Page-aligned path only. The aggregate dispatch reads tail state that
  // the previous kick's delivery uploads AFTER this enqueue (gate state in
  // gpu_ordering, src/gpu/ordering.h). The wait goes after compress, which
  // reads no tail state.
  {
    const size_t page_size = stage->ar.per_lod_agg_layouts[0].page_size;
    const int enable = plan.layout.total_batch_chunks > 0 &&
                       stage->ar.total_shards > 0 && page_size > 0;
    CHECK(Error,
          gpu_pool_acquire_consume_gen(&stage->tail, compress_stream, enable) ==
            0);
  }

  CHECK(Error,
        compress_agg_aggregate(
          stage, &plan, in->fc, slot.p, pool_buf, compress_stream) == 0);
  // Also releases the chunk pool for re-zero: POOL_CONSUMED aliases this
  // record (#140).
  CHECK(Error,
        gpu_pool_release_produce(&stage->agg_pool, in->fc, compress_stream) ==
          0);

  compress_agg_fill_handoff(stage, in, &plan, out);
  return 0;

Error:
  return 1;
}

// --- Helpers ---

static struct compress_agg_input
make_compress_input(struct stream_engine* e, int fc, uint32_t n_epochs)
{
  struct schedule_slot* s = &e->sched.slot[fc];
  return (struct compress_agg_input){
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = s->active_levels_mask,
    .batch_active_masks = s->batch_active_masks,
    .epochs_per_batch = e->sched.epochs_per_batch,
  };
}

static void
reset_fill_slot(struct stream_engine* e)
{
  struct schedule_slot* s = &e->sched.slot[e->sched.fill];
  e->sched.accumulated = 0;
  s->active_levels_mask = 0;
  memset(s->batch_active_masks,
         0,
         (size_t)e->sched.epochs_per_batch * sizeof(uint32_t));
}

// Run LOD for the current epoch (multiscale), or mark L0 active; fold the
// epoch's mask into the fill slot.
static int
run_epoch_lod(struct stream_engine* e, struct stream_context* ctx)
{
  struct schedule_slot* s = &e->sched.slot[e->sched.fill];
  uint32_t active_mask;

  if (!e->sched.lod_active) {
    active_mask = 1;
  } else {
    CHECK(Error,
          lod_run_epoch(&e->lod,
                        &e->lod_shared,
                        &e->ord,
                        e->sched.fill,
                        &ctx->levels,
                        stream_engine_pool_epoch(e, ctx, e->sched.accumulated),
                        ctx->config.dtype,
                        ctx->config.reduce_method,
                        ctx->config.append_reduce_method,
                        &ctx->dims,
                        e->streams.compute,
                        &active_mask) == 0);
  }

  s->batch_active_masks[e->sched.accumulated] = active_mask;
  s->active_levels_mask |= active_mask;
  return 0;

Error:
  return 1;
}

// Drain one kicked slot: host-sync on its handoff (near-zero in steady
// state), run delivery, clear. Accumulates flush_stall.
static struct writer_result
drain_slot(struct stream_engine* e, struct stream_context* ctx, int fc)
{
  struct schedule_slot* s = &e->sched.slot[fc];
  if (!s->kicked)
    return writer_ok();

  // The tail gate's GEQ threshold assumes drains follow kick order.
  gpu_edge_host_rule(&e->ord,
                     GPU_EDGE_DELIVER_OLDEST_FIRST,
                     !e->sched.slot[fc ^ 1].kicked ||
                       s->kick_seq < e->sched.slot[fc ^ 1].kick_seq);

  struct platform_clock stall_clk = { 0 };
  platform_toc(&stall_clk);
  struct writer_result r = d2h_deliver_drain(&e->d2h_deliver,
                                             &s->handoff,
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

  s->kicked = 0;
  return r;
}

// Kick compress -> aggregate -> D2H for n_epochs on slot fc and record the
// handoff for the later drain.
static int
kick_batch(struct stream_engine* e,
           struct stream_context* ctx,
           int fc,
           uint32_t n_epochs)
{
  gpu_edge_host_rule(
    &e->ord, GPU_EDGE_DRAIN_BEFORE_REKICK, !e->sched.slot[fc].kicked);
  struct compress_agg_input in = make_compress_input(e, fc, n_epochs);
  struct flush_handoff handoff = { 0 };

  CHECK(Error,
        schedule_compress_agg_kick(&e->compress_agg,
                                   &in,
                                   &ctx->levels,
                                   &e->pools.p,
                                   e->sched.lod_active,
                                   e->streams.compress,
                                   &handoff) == 0);

  CHECK(Error,
        d2h_deliver_kick(&e->d2h_deliver,
                         &handoff,
                         ctx->sink,
                         e->streams.d2h) == 0);

  e->sched.slot[fc].handoff = handoff;
  e->sched.slot[fc].kick_seq = e->sched.next_seq++;
  e->sched.slot[fc].kicked = 1;

  return 0;

Error:
  return 1;
}

// Pipelined batch boundary: drain the slot about to be refilled (it holds
// the oldest undelivered batch, so shard writes stay in batch order), kick
// the new batch on it, swap, zero the fresh slot.
static struct writer_result
drain_kick_and_swap(struct stream_engine* e, struct stream_context* ctx)
{
  const uint32_t K = e->sched.epochs_per_batch;
  const int fc = e->sched.fill;

  {
    struct writer_result r = drain_slot(e, ctx, fc);
    if (r.error)
      return r;
  }

  // Without the gate the tail upload cannot be ordered device-side, so also
  // drain the other, newer kicked batch — order stays oldest-first, depth
  // degrades to 1 — and the kick below sees published tail state.
  if (e->sched.depth == SCHEDULE_DRAIN_BEFORE_KICK) {
    struct writer_result r = drain_slot(e, ctx, fc ^ 1);
    if (r.error)
      return r;
  }

  CHECK(Error, kick_batch(e, ctx, fc, e->sched.accumulated) == 0);

  e->sched.fill ^= 1;
  // The aggregate's last pool read gates reuse; re-zeroing any earlier
  // corrupts the in-flight batch (#140).
  struct gpu_pool_view fresh;
  CHECK(Error,
        gpu_pool_acquire_produce(
          &e->pools.p, e->sched.fill, e->streams.compute, &fresh) == 0);
  size_t pool_bytes = (uint64_t)K * ctx->levels.total_chunks *
                      ctx->layout.chunk_stride * dtype_bpe(ctx->config.dtype);
  CU(Error,
     cuMemsetD8Async(gpu_pool_view_d(fresh), 0, pool_bytes, e->streams.compute));

  reset_fill_slot(e);

  return writer_ok();

Error:
  return writer_error();
}

// --- Producer-thread entry points ---

struct writer_result
schedule_accumulate_epoch(struct stream_engine* e, struct stream_context* ctx)
{
  if (run_epoch_lod(e, ctx))
    return writer_error();

  e->sched.accumulated++;

  if (e->sched.accumulated < e->sched.epochs_per_batch)
    return writer_ok();

  // Release pool-filled once after the last epoch's scatter; compute-stream
  // ordering means this subsumes per-epoch ready signals.
  CHECK(Error,
        gpu_pool_release_produce(
          &e->pools.p, e->sched.fill, e->streams.compute) == 0);

  if (e->sched.depth == SCHEDULE_DRAIN_AFTER_KICK) {
    struct writer_result r = schedule_flush_accumulated(e, ctx);
    // Host-ordered re-acquire: the drain above host-completed this slot's
    // D2H, so no device wait is queued.
    if (!r.error) {
      size_t bpe = dtype_bpe(ctx->config.dtype);
      size_t pool_bytes = (uint64_t)e->sched.epochs_per_batch *
                          ctx->levels.total_chunks * ctx->layout.chunk_stride *
                          bpe;
      CU(SyncError,
         cuMemsetD8Async(
           gpu_pool_view_d(gpu_pool_at(&e->pools.p, e->sched.fill, 0)),
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

int
schedule_add_partial_epoch(struct stream_engine* e, struct stream_context* ctx)
{
  if (run_epoch_lod(e, ctx))
    return 1;
  e->sched.accumulated++;
  return 0;
}

struct writer_result
schedule_drain_kicked(struct stream_engine* e, struct stream_context* ctx)
{
  for (int i = 0; i < 2; ++i) {
    int pick = -1;
    uint64_t pick_seq = UINT64_MAX;
    for (int fc = 0; fc < 2; ++fc) {
      struct schedule_slot* s = &e->sched.slot[fc];
      if (s->kicked && s->kick_seq < pick_seq) {
        pick = fc;
        pick_seq = s->kick_seq;
      }
    }
    if (pick < 0)
      break;
    struct writer_result r = drain_slot(e, ctx, pick);
    if (r.error)
      return r;
  }
  return writer_ok();
}

struct writer_result
schedule_flush_accumulated(struct stream_engine* e, struct stream_context* ctx)
{
  if (e->sched.accumulated == 0)
    return writer_ok();

  const int fc = e->sched.fill;

  // Release pool-filled after all scatter ops for this (possibly partial)
  // batch.
  if (gpu_pool_release_produce(&e->pools.p, fc, e->streams.compute))
    return writer_error();

  if (kick_batch(e, ctx, fc, e->sched.accumulated))
    return writer_error();

  struct writer_result r = drain_slot(e, ctx, fc);
  if (r.error)
    return r;

  reset_fill_slot(e);
  return r;
}

struct writer_result
schedule_flush_partial_append(struct stream_engine* e,
                              struct stream_context* ctx)
{
  if (!ctx->dims.append_downsample || !ctx->levels.enable_multiscale)
    return writer_ok();

  const struct lod_plan* p = &e->lod.plan;
  const size_t bytes_per_element = dtype_bpe(ctx->config.dtype);
  const enum dtype dtype = ctx->config.dtype;

  uint32_t active_levels_mask = 0;
  for (int lv = 1; lv < p->levels.nlod; ++lv) {
    if (e->lod.append_accum.counts[lv] > 0)
      active_levels_mask |= (1u << lv);
  }

  if (!active_levels_mask)
    return writer_ok();

  const int fc = e->sched.fill;
  struct schedule_slot* fs = &e->sched.slot[fc];
  fs->active_levels_mask = active_levels_mask;
  fs->batch_active_masks[0] = active_levels_mask;

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
    // (the fill slot is still being filled).
    CUdeviceptr dst =
      gpu_pool_view_d(gpu_pool_at(&e->pools.p,
                                  fc,
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
        gpu_pool_release_produce(&e->pools.p, fc, e->streams.compute) == 0);
  if (kick_batch(e, ctx, fc, 1))
    return writer_error();
  return drain_slot(e, ctx, fc);

Error:
  return writer_error();
}

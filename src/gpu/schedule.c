#include "gpu/schedule.h"

#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.lod.h"

#include "gpu/ordering.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

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

// --- D2H kick / drain ---

// Wait for any pending IO fence on the unified slot (across all LODs that
// share it). Accumulates wall time into io_fence_stall.
static void
wait_io_fences(struct aggregate_slot* slot,
               struct shard_sink* sink,
               struct stream_metrics* metrics)
{
  if (!sink->wait_fence)
    return;
  struct platform_clock clk = { 0 };
  platform_toc(&clk);
  if (slot->io_done.seq > 0)
    sink->wait_fence(sink, slot->io_done);
  if (metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&metrics->io_fence_stall, ms, 0, 0);
  }
}

int
schedule_d2h_kick(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  struct shard_sink* sink,
                  CUstream d2h_stream)
{
  const int fc = handoff->fc;

  // io_done is host-owned slot bookkeeping; its fence must retire before
  // any device acquire, so peek rather than acquire here.
  wait_io_fences(gpu_pool_at(handoff->agg_host, fc, 0).p, sink, stage->metrics);

  struct gpu_pool_view v;
  CHECK(Error,
        gpu_pool_acquire_consume(handoff->agg_pool, fc, d2h_stream, &v) == 0);

  int dispatch_err = d2h_deliver_kick(stage, handoff, v.p, d2h_stream);

  // Passthrough never polls the chunk index (its drain waits on the
  // slot-drained edge recorded after the kick's bulk copy).
  if (!dispatch_err && !handoff->passthrough &&
      gpu_pool_release_produce(handoff->agg_index, fc, d2h_stream))
    dispatch_err = 1;

  // Always release the passthrough slot (SLOT_DRAINED) even on dispatch
  // error: the drain's host poll blocks on it and would hang otherwise.
  if (handoff->passthrough)
    CHECK(Error,
          gpu_pool_release_consume(handoff->agg_pool, fc, d2h_stream) == 0);

  return dispatch_err;

Error:
  return 1;
}

struct writer_result
schedule_d2h_drain(struct d2h_deliver_stage* stage,
                   const struct flush_handoff* handoff,
                   const struct level_geometry* levels,
                   const struct dim_info* dims,
                   const struct tile_stream_layout* layout,
                   const struct tile_stream_configuration* config,
                   struct shard_sink* sink,
                   const struct lod_state* lod,
                   const struct lod_shared_state* lod_shared,
                   struct stream_metrics* metrics,
                   struct platform_clock* metadata_update_clock)
{
  const int fc = handoff->fc;
  struct aggregate_slot* slot = NULL;
  int err = 1;

  if (sink->has_error && sink->has_error(sink))
    goto Done;

  {
    struct platform_clock kick_clk = { 0 };
    platform_toc(&kick_clk);

    if (handoff->passthrough) {
      struct gpu_pool_view hv;
      if (gpu_pool_host_acquire_consume(handoff->agg_host, fc, &hv))
        goto Done;
      slot = hv.p;
    } else {
      struct gpu_pool_view iv;
      if (gpu_pool_host_acquire_consume(handoff->agg_index, fc, &iv))
        goto Done;
      slot = iv.p;

      int dispatch_err = d2h_deliver_drain_copy(stage, handoff, slot);

      // Always release the slot (SLOT_DRAINED), even if the D2H dispatch
      // above failed: the host poll below and the next kick's acquire block
      // on it and would hang otherwise. Release-on-error is harmless
      // because the stream is already in an error state and the next op
      // will short-circuit.
      if (gpu_pool_release_consume(handoff->agg_pool, fc, stage->drain_stream))
        goto Done;

      if (dispatch_err)
        goto Done;
      if (gpu_pool_host_acquire_consume(handoff->agg_host, fc, NULL))
        goto Done;
    }

    float kick_ms = platform_toc(&kick_clk) * 1000.0f;
    accumulate_metric_ms(&metrics->kick_sync_stall, kick_ms, 0, 0);
  }

  {
    // The consumed direction is the deliver-oldest-first host rule, so no
    // device wait is queued; the acquire hands out the array whose tail
    // buffers this delivery uploads.
    struct gpu_pool_view tv = { 0 };
    if (gpu_pool_host_acquire_produce(handoff->tail, 0, &tv))
      goto Done;
    err = d2h_deliver_drain_sink(stage,
                                 handoff,
                                 slot,
                                 tv.p,
                                 levels,
                                 dims,
                                 layout,
                                 config,
                                 sink,
                                 lod,
                                 lod_shared,
                                 metrics)
            .error;
  }

Done:
  // The drained kick's tail generation releases exactly once, on failure
  // exits too — the gate threshold counts kicks, so a skipped release
  // leaves the gate unsatisfiable and destroy's auto-flush hangs polling.
  // Tail-state content is moot once the drain has failed.
  gpu_pool_release_produce_gen(handoff->tail);
  if (err)
    return writer_error();
  if (d2h_deliver_update_metadata(
        handoff, dims, config, sink, metadata_update_clock))
    return writer_error();
  return writer_ok();
}

void
schedule_quiesce_output(struct stream_engine* e, struct shard_sink* sink)
{
  cuStreamSynchronize(e->streams.d2h);
  // Host-ordered slot access: the sync above quiesced the slots.
  for (int fc = 0; fc < 2; ++fc) {
    struct aggregate_slot* slot =
      gpu_pool_at(&e->compress_agg.agg_host, fc, 0).p;
    if (slot->io_done.seq > 0 && sink->wait_fence)
      sink->wait_fence(sink, slot->io_done);
    slot->io_done.seq = 0;
  }
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
  struct writer_result r = schedule_d2h_drain(&e->d2h_deliver,
                                              &s->handoff,
                                              &ctx->levels,
                                              &ctx->dims,
                                              &ctx->layout,
                                              &ctx->config,
                                              ctx->sink,
                                              &e->lod,
                                              &e->lod_shared,
                                              &e->metrics,
                                              &e->metadata_update_clock);
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
        schedule_d2h_kick(&e->d2h_deliver,
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

  uint32_t active_levels_mask = lod_partial_append_mask(&e->lod);
  if (!active_levels_mask)
    return writer_ok();

  const int fc = e->sched.fill;
  struct schedule_slot* fs = &e->sched.slot[fc];
  fs->active_levels_mask = active_levels_mask;
  fs->batch_active_masks[0] = active_levels_mask;

  // Produce-phase writes within the generation acquired at the last swap
  // (the fill slot is still being filled).
  CHECK(Error,
        lod_emit_partial_append(&e->lod,
                                &e->lod_shared,
                                &ctx->levels,
                                &ctx->layout,
                                ctx->config.dtype,
                                ctx->config.append_reduce_method,
                                active_levels_mask,
                                gpu_pool_at(&e->pools.p, fc, 0),
                                e->streams.compute) == 0);

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

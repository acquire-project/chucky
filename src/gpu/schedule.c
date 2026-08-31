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

#include <assert.h>
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
gpu_streams_sync(const struct gpu_streams* s)
{
  cu_stream_sync(s->h2d);
  cu_stream_sync(s->compute);
  cu_stream_sync(s->compress);
  cu_stream_sync(s->d2h);
  cu_stream_sync(s->drain);
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
                const struct gpu_delivery* delivery)
{
  const int page_aligned = ar->page_size > 0 && ar->total_shards > 0;
  const int worker = delivery && delivery->thread;
  if (!delivery)
    sched->mode = SCHEDULE_DRAIN_AFTER_KICK;
  else if (!page_aligned)
    sched->mode = SCHEDULE_PIPELINED_DIRECT;
  else
    sched->mode =
      worker ? SCHEDULE_PIPELINED_HOST_COORDINATED : SCHEDULE_DRAIN_BEFORE_KICK;
  if (sched->next_generation == 0)
    sched->next_generation = 1;
}

int
schedule_lod_active(const struct gpu_ordering* ord, int enable_multiscale)
{
  return enable_multiscale &&
         gpu_ordering_event(ord, GPU_EDGE_LOD_DONE, 0) != NULL;
}

// --- Compress+aggregate kick ---

int
schedule_compress_agg_prepare(struct compress_agg_stage* stage,
                              const struct compress_agg_input* in,
                              const struct level_geometry* levels,
                              struct gpu_pool* chunk_pool,
                              int lod_active,
                              CUstream compress_stream,
                              struct schedule_slot* slot)
{
  // Ahead of the pool acquires, so these uploads overlap the previous batch's
  // drain instead of queueing behind the wait for it.
  CHECK(Error,
        compress_agg_prepare(stage, in, levels, compress_stream, &slot->plan) ==
          0);

  CHECK(Error,
        gpu_pool_acquire_consume(
          chunk_pool, in->fc, compress_stream, &slot->pool_buf) == 0);
  // LOD chunks land in the pool through a second producer edge.
  if (lod_active)
    CHECK(Error,
          gpu_edge_wait(
            chunk_pool->ord, GPU_EDGE_LOD_DONE, in->fc, compress_stream) == 0);
  // Aggregate must not overwrite agg[fc] before the prior D2H on the same
  // fc has read it.
  CHECK(Error,
        gpu_pool_acquire_produce(
          &stage->agg_pool, in->fc, compress_stream, &slot->aggregate_slot) ==
          0);

  CHECK(Error,
        compress_agg_compress(
          stage, in, levels, slot->pool_buf, compress_stream) == 0);

  compress_agg_fill_handoff(stage, in, &slot->plan, &slot->handoff);
  return 0;

Error:
  return 1;
}

int
schedule_compress_agg_submit(struct compress_agg_stage* stage,
                             struct schedule_slot* slot,
                             CUstream compress_stream)
{
  const int fc = slot->handoff.fc;

  CHECK(Error,
        compress_agg_aggregate(stage,
                               &slot->plan,
                               fc,
                               slot->aggregate_slot.p,
                               slot->pool_buf,
                               compress_stream) == 0);
  // Also releases the chunk pool for re-zero: POOL_CONSUMED aliases this
  // record (#140).
  CHECK(Error,
        gpu_pool_release_produce(&stage->agg_pool, fc, compress_stream) == 0);

  return 0;

Error:
  return 1;
}

int
schedule_compress_agg_kick(struct compress_agg_stage* stage,
                           const struct compress_agg_input* in,
                           const struct level_geometry* levels,
                           struct gpu_pool* chunk_pool,
                           int lod_active,
                           CUstream compress_stream,
                           struct flush_handoff* out)
{
  struct schedule_slot slot = { 0 };
  CHECK(Error,
        schedule_compress_agg_prepare(
          stage, in, levels, chunk_pool, lod_active, compress_stream, &slot) ==
          0);
  CHECK(Error,
        schedule_compress_agg_submit(stage, &slot, compress_stream) == 0);

  *out = slot.handoff;
  return 0;

Error:
  return 1;
}

// --- D2H kick / drain ---

// Wait time already attributed to the ordering edges.
static float
edge_stall_total_ms(const struct stream_metrics* m)
{
  float t = 0;
  const size_t n = sizeof(m->edge_stall) / sizeof(m->edge_stall[0]);
  for (size_t i = 0; i < n; ++i)
    t += m->edge_stall[i].ms;
  return t;
}

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
    const float polls_before = edge_stall_total_ms(metrics);

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

    float block_ms = platform_toc(&kick_clk) * 1000.0f;
    float own_ms = block_ms - (edge_stall_total_ms(metrics) - polls_before);
    accumulate_metric_ms(
      &metrics->drain_dispatch, own_ms > 0 ? own_ms : 0.0f, 0, 0);
  }

  err = d2h_deliver_drain_sink(stage,
                               handoff,
                               slot,
                               handoff->shards,
                               levels,
                               layout,
                               config,
                               sink,
                               metrics)
          .error;

Done:
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

// --- Delivery worker ---

static struct writer_result
drain_payload(struct stream_engine* e, struct stream_context* ctx, int fc)
{
  return schedule_d2h_drain(&e->d2h_deliver,
                            &e->sched.slot[fc].handoff,
                            &ctx->levels,
                            &ctx->dims,
                            &ctx->layout,
                            &ctx->config,
                            ctx->sink,
                            &e->metrics,
                            &e->metadata_update_clock);
}

static int
submit_payload(struct stream_engine* e, struct stream_context* ctx, int fc)
{
  struct schedule_slot* s = &e->sched.slot[fc];
  CHECK(Error,
        schedule_compress_agg_submit(
          &e->compress_agg, s, e->streams.compress) == 0);
  CHECK(Error,
        schedule_d2h_kick(
          &e->d2h_deliver, &s->handoff, ctx->sink, e->streams.d2h) == 0);
  return 0;

Error:
  return 1;
}

// d->mu must be held. Once one generation fails, a later prepared job must
// never aggregate from tail state that was not produced by its predecessor,
// so it is cancelled before anything is queued for it. A job that already
// submitted still has its aggregation and device-to-host copy queued on the
// device, and its output has to reach the sink and release its pool slot, so
// it is left for the loop to drain.
static void
delivery_fail_pending_locked(struct gpu_delivery* d)
{
  d->sticky_error = 1;
  for (int i = 0; i < 2; ++i) {
    struct delivery_job* j = &d->job[i];
    if (j->state == DELIVERY_JOB_PREPARED) {
      j->result = writer_error();
      j->state = DELIVERY_JOB_DONE;
    }
  }
  platform_cond_broadcast(d->cv);
}

// d->mu must be held.
static void
delivery_park(struct gpu_delivery* d, enum delivery_hold_point at)
{
  while (d->hold_at == at && !d->stop)
    platform_cond_wait(d->cv, d->mu);
}

static void
delivery_main(void* arg)
{
  struct gpu_delivery* d = (struct gpu_delivery*)arg;
  CUresult context_result = cuCtxSetCurrent(d->cuda);
  if (context_result != CUDA_SUCCESS)
    handle_curesult(
      LOG_ERROR, context_result, __FILE__, __LINE__, "cuCtxSetCurrent");

  platform_mutex_lock(d->mu);
  if (context_result != CUDA_SUCCESS)
    delivery_fail_pending_locked(d);

  for (;;) {
    int fc = -1;
    for (int i = 0; i < 2; ++i) {
      struct delivery_job* j = &d->job[i];
      if ((j->state == DELIVERY_JOB_PREPARED ||
           j->state == DELIVERY_JOB_SUBMITTED) &&
          (fc < 0 || j->generation < d->job[fc].generation))
        fc = i;
    }
    if (fc < 0) {
      if (d->stop)
        break;
      platform_cond_wait(d->cv, d->mu);
      continue;
    }

    struct delivery_job* j = &d->job[fc];
    const uint64_t generation = j->generation;

    if (j->state == DELIVERY_JOB_PREPARED) {
      // No GPU operation is queued until the preceding delivery has returned
      // from its synchronous tail uploads.
      if (d->tail_ready_generation != generation - 1) {
        if (d->stop) {
          delivery_fail_pending_locked(d);
          continue;
        }
        platform_cond_wait(d->cv, d->mu);
        continue;
      }

      struct stream_engine* e = j->e;
      struct stream_context* ctx = j->ctx;
      platform_mutex_unlock(d->mu);
      int submit_error = submit_payload(e, ctx, fc);
      platform_mutex_lock(d->mu);

      if (submit_error) {
        j->result = writer_error();
        j->state = DELIVERY_JOB_DONE;
        delivery_fail_pending_locked(d);
        continue;
      }

      j->state = DELIVERY_JOB_SUBMITTED;
      d->submitted_generation = generation;
      platform_cond_broadcast(d->cv);
    }

    delivery_park(d, DELIVERY_HOLD_BEFORE_DRAIN);
    if (j->state != DELIVERY_JOB_SUBMITTED)
      continue;

    const struct delivery_job* other = &d->job[fc ^ 1];
    int oldest = other->state == DELIVERY_JOB_EMPTY ||
                 other->state == DELIVERY_JOB_DONE ||
                 generation < other->generation;
    struct stream_engine* e = j->e;
    struct stream_context* ctx = j->ctx;
    platform_mutex_unlock(d->mu);

    gpu_edge_host_rule(&e->ord, GPU_EDGE_DELIVER_OLDEST_FIRST, oldest);
    struct writer_result r = drain_payload(e, ctx, fc);

    platform_mutex_lock(d->mu);
    j->result = r;
    j->state = DELIVERY_JOB_DONE;
    if (r.error) {
      delivery_fail_pending_locked(d);
    } else {
      // d2h_deliver_drain_sink returned only after synchronous tail uploads.
      d->tail_ready_generation = generation;
      platform_cond_broadcast(d->cv);
    }

    delivery_park(d, DELIVERY_HOLD_AFTER_DRAIN);
  }
  platform_mutex_unlock(d->mu);
}

int
gpu_delivery_init(struct gpu_delivery* d, CUcontext cuda)
{
  memset(d, 0, sizeof(*d));
  if (!cuda)
    return 1;
  d->cuda = cuda;
  d->mu = platform_mutex_new();
  d->cv = platform_cond_new();
  CHECK(Fail, d->mu && d->cv);
  d->thread = platform_thread_start(delivery_main, d);
  CHECK(Fail, d->thread);
  return 0;

Fail:
  platform_cond_free(d->cv);
  platform_mutex_free(d->mu);
  memset(d, 0, sizeof(*d));
  return 1;
}

static int
gpu_delivery_enqueue(struct gpu_delivery* d,
                     struct stream_engine* e,
                     struct stream_context* ctx,
                     int fc,
                     uint64_t generation,
                     enum delivery_job_state state)
{
  if (!d->thread)
    return 1;
  platform_mutex_lock(d->mu);
  // The drain-before-rekick rule keeps a slot's previous job joined before
  // its next kick can enqueue here.
  assert(d->job[fc].state == DELIVERY_JOB_EMPTY);
  d->job[fc] = (struct delivery_job){
    .state = state,
    .generation = generation,
    .e = e,
    .ctx = ctx,
  };
  int error = d->sticky_error;
  if (error) {
    d->job[fc].result = writer_error();
    d->job[fc].state = DELIVERY_JOB_DONE;
  } else if (state == DELIVERY_JOB_SUBMITTED) {
    d->submitted_generation = generation;
  }
  platform_cond_broadcast(d->cv);
  platform_mutex_unlock(d->mu);
  return error;
}

int
gpu_delivery_enqueue_prepared(struct gpu_delivery* d,
                              struct stream_engine* e,
                              struct stream_context* ctx,
                              int fc,
                              uint64_t generation)
{
  return gpu_delivery_enqueue(d, e, ctx, fc, generation, DELIVERY_JOB_PREPARED);
}

int
gpu_delivery_enqueue_submitted(struct gpu_delivery* d,
                               struct stream_engine* e,
                               struct stream_context* ctx,
                               int fc,
                               uint64_t generation)
{
  // Aggregation is already queued, so without a worker the producer can drain
  // this slot itself when it comes to refill it.
  if (!d->thread)
    return 0;
  return gpu_delivery_enqueue(
    d, e, ctx, fc, generation, DELIVERY_JOB_SUBMITTED);
}

int
gpu_delivery_wait_submitted(struct gpu_delivery* d, uint64_t generation)
{
  if (!d->thread)
    return 1;
  platform_mutex_lock(d->mu);
  while (!d->sticky_error && d->submitted_generation < generation)
    platform_cond_wait(d->cv, d->mu);
  int error = d->sticky_error;
  platform_mutex_unlock(d->mu);
  return error;
}

int
gpu_delivery_pending(struct gpu_delivery* d, int fc)
{
  if (!d->thread)
    return 0;
  platform_mutex_lock(d->mu);
  int p = d->job[fc].state != DELIVERY_JOB_EMPTY;
  platform_mutex_unlock(d->mu);
  return p;
}

struct writer_result
gpu_delivery_join(struct gpu_delivery* d, int fc)
{
  platform_mutex_lock(d->mu);
  while (d->job[fc].state != DELIVERY_JOB_DONE)
    platform_cond_wait(d->cv, d->mu);
  struct writer_result r = d->job[fc].result;
  d->job[fc] = (struct delivery_job){ 0 };
  platform_mutex_unlock(d->mu);
  return r;
}

void
gpu_delivery_stop_join(struct gpu_delivery* d)
{
  if (!d->thread)
    return;
  platform_mutex_lock(d->mu);
  d->stop = 1;
  platform_cond_broadcast(d->cv);
  platform_mutex_unlock(d->mu);
  platform_thread_join(d->thread);
  d->thread = NULL;
  platform_cond_free(d->cv);
  d->cv = NULL;
  platform_mutex_free(d->mu);
  d->mu = NULL;
}

void
gpu_delivery_set_hold(struct gpu_delivery* d, enum delivery_hold_point at)
{
  if (!d->thread)
    return;
  platform_mutex_lock(d->mu);
  d->hold_at = at;
  platform_cond_broadcast(d->cv);
  platform_mutex_unlock(d->mu);
}

// The mutex is gone once the worker has been joined, but the counters it
// protected stay readable.
enum delivery_job_state
gpu_delivery_job_state(struct gpu_delivery* d, int fc, uint64_t* generation)
{
  if (d->mu)
    platform_mutex_lock(d->mu);
  enum delivery_job_state state = d->job[fc].state;
  if (generation)
    *generation = d->job[fc].generation;
  if (d->mu)
    platform_mutex_unlock(d->mu);
  return state;
}

void
gpu_delivery_generations(struct gpu_delivery* d,
                         uint64_t* submitted,
                         uint64_t* tail_ready)
{
  if (d->mu)
    platform_mutex_lock(d->mu);
  if (submitted)
    *submitted = d->submitted_generation;
  if (tail_ready)
    *tail_ready = d->tail_ready_generation;
  if (d->mu)
    platform_mutex_unlock(d->mu);
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
                        &ctx->layout,
                        stream_engine_pool_epoch(e, ctx, e->sched.accumulated),
                        ctx->config.dtype,
                        ctx->config.reduce_method,
                        ctx->config.append_reduce_method,
                        &ctx->dims,
                        e->streams.compute,
                        &active_mask) == 0);
    lod_collect_timing(&e->lod_shared, &e->metrics);
  }

  s->batch_active_masks[e->sched.accumulated] = active_mask;
  s->active_levels_mask |= active_mask;
  return 0;

Error:
  return 1;
}

static struct writer_result
drain_slot(struct stream_engine* e, struct stream_context* ctx, int fc)
{
  struct schedule_slot* s = &e->sched.slot[fc];
  if (!s->kicked)
    return writer_ok();

  // Sink delivery and tail publication always follow generation order.
  gpu_edge_host_rule(&e->ord,
                     GPU_EDGE_DELIVER_OLDEST_FIRST,
                     !e->sched.slot[fc ^ 1].kicked ||
                       s->generation < e->sched.slot[fc ^ 1].generation);

  struct platform_clock stall_clk = { 0 };
  platform_toc(&stall_clk);
  struct writer_result r = gpu_delivery_pending(&e->delivery, fc)
                             ? gpu_delivery_join(&e->delivery, fc)
                             : drain_payload(e, ctx, fc);
  float ms = (float)(platform_toc(&stall_clk) * 1000.0);
  accumulate_metric_ms(&e->metrics.flush_stall, ms, 0, 0);

  s->kicked = 0;
  return r;
}

static int
kick_batch(struct stream_engine* e,
           struct stream_context* ctx,
           int fc,
           uint32_t n_epochs)
{
  gpu_edge_host_rule(
    &e->ord, GPU_EDGE_DRAIN_BEFORE_REKICK, !e->sched.slot[fc].kicked);
  struct compress_agg_input in = make_compress_input(e, fc, n_epochs);
  struct schedule_slot* s = &e->sched.slot[fc];

  const uint64_t generation = e->sched.next_generation;

  // Shared codec-size arrays, LUTs, and shard tables can be overwritten only
  // after the previous aggregation has been enqueued on this same stream.
  if (e->sched.mode == SCHEDULE_PIPELINED_HOST_COORDINATED && generation > 1)
    CHECK(Error,
          gpu_delivery_wait_submitted(&e->delivery, generation - 1) == 0);

  CHECK(Error,
        schedule_compress_agg_prepare(&e->compress_agg,
                                      &in,
                                      &ctx->levels,
                                      &e->pools.p,
                                      e->sched.lod_active,
                                      e->streams.compress,
                                      s) == 0);

  s->generation = generation;
  e->sched.next_generation = generation + 1;

  if (e->sched.mode == SCHEDULE_PIPELINED_HOST_COORDINATED) {
    // The worker owns submission so aggregation cannot read tail state until
    // generation-1's synchronous upload has completed.
    int enqueue_error =
      gpu_delivery_enqueue_prepared(&e->delivery, e, ctx, fc, generation);
    s->kicked = 1;
    CHECK(Error, enqueue_error == 0);
    return 0;
  }

  CHECK(Error,
        schedule_compress_agg_submit(
          &e->compress_agg, s, e->streams.compress) == 0);
  CHECK(Error,
        schedule_d2h_kick(
          &e->d2h_deliver, &s->handoff, ctx->sink, e->streams.d2h) == 0);

  s->kicked = 1;

  // Depth-1 schedules drain inline. Contiguous pipelined batches have already
  // submitted aggregation and queue only their drain.
  if (e->sched.mode == SCHEDULE_PIPELINED_DIRECT)
    CHECK(Error,
          gpu_delivery_enqueue_submitted(
            &e->delivery, e, ctx, fc, generation) == 0);

  return 0;

Error:
  return 1;
}

// The slot about to be refilled holds the oldest undelivered batch, so
// draining it first keeps shard writes in batch order.
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

  // Without a worker, drain the other kicked batch too. The next direct
  // aggregation is enqueued only after its predecessor's tail upload returns.
  if (e->sched.mode == SCHEDULE_DRAIN_BEFORE_KICK) {
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
  CU(
    Error,
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
  // batch_active_masks holds one entry per epoch of a batch, and the slot's
  // pool region holds one epoch each. A failed kick can leave the batch full
  // without resetting, and counting another epoch would run past both.
  if (e->sched.accumulated >= e->sched.epochs_per_batch)
    return writer_error();

  if (run_epoch_lod(e, ctx))
    return writer_error();

  e->sched.accumulated++;

  if (e->sched.accumulated < e->sched.epochs_per_batch)
    return writer_ok();

  // Release pool-filled once after the last epoch's scatter; compute-stream
  // ordering means this subsumes per-epoch ready signals. The drain-after-kick
  // path releases inside schedule_flush_accumulated, so skip it here to avoid
  // re-recording the same edge on the same stream.
  if (e->sched.mode != SCHEDULE_DRAIN_AFTER_KICK)
    CHECK(Error,
          gpu_pool_release_produce(
            &e->pools.p, e->sched.fill, e->streams.compute) == 0);

  if (e->sched.mode == SCHEDULE_DRAIN_AFTER_KICK) {
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
  if (e->sched.accumulated >= e->sched.epochs_per_batch)
    return 1;
  if (run_epoch_lod(e, ctx))
    return 1;
  e->sched.accumulated++;
  return 0;
}

// A worker left running owns shard state the caller reads and buffers destroy
// frees, so every kicked slot is drained even after one fails.
struct writer_result
schedule_drain_kicked(struct stream_engine* e, struct stream_context* ctx)
{
  struct writer_result first = writer_ok();
  for (int i = 0; i < 2; ++i) {
    int pick = -1;
    uint64_t pick_generation = UINT64_MAX;
    for (int fc = 0; fc < 2; ++fc) {
      struct schedule_slot* s = &e->sched.slot[fc];
      if (s->kicked && s->generation < pick_generation) {
        pick = fc;
        pick_generation = s->generation;
      }
    }
    if (pick < 0)
      break;
    struct writer_result r = drain_slot(e, ctx, pick);
    if (r.error && !first.error)
      first = r;
  }
  return first;
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
          gpu_edge_record(&e->ord, GPU_EDGE_LOD_DONE, fc, e->streams.compute) ==
            0);

  CHECK(Error,
        gpu_pool_release_produce(&e->pools.p, fc, e->streams.compute) == 0);
  if (kick_batch(e, ctx, fc, 1))
    return writer_error();
  return drain_slot(e, ctx, fc);

Error:
  return writer_error();
}

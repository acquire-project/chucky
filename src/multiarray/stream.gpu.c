#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/schedule.h"
#include "gpu/stream.engine.h"
#include "gpu/stream.ingest.h"
#include "gpu/stream.lod.h"

#include "defs.limits.h"
#include "gpu/prelude.cuda.h"
#include "multiarray.gpu.h"
#include "stream/config.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

// ---- Per-array descriptor ----
// Extends stream_context with the per-array engine state that is swapped
// into/out of the engine on array switch.

struct array_descriptor_gpu
{
  struct stream_context ctx;
  struct computed_stream_layouts cl; // owned, freed on destroy

  // Whole-struct swapped on bind/unbind. st.lod owns plan, layouts[], CSRs,
  // append accumulator device memory, and LOD LUTs — but
  // NOT d_linear/d_morton/timing, which are shared and owned by the engine.
  // st.agg owns per-LOD layouts, shard_state, and the per-array tail buffers.
  struct engine_array_state st;

  int flushed;      // 1 once finalized; no further input is taken
  int closed;       // 1 once close drained and, on success, published
  int close_failed; // outcome of that close, re-reported by later calls
};

// ---- Main struct ----

struct multiarray_tile_stream_gpu
{
  struct multiarray_writer writer;
  struct stream_engine engine;
  int n_arrays;
  int active; // -1 = none
  struct array_descriptor_gpu* arrays;
};

// ---- Forward declarations ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data);
static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self);
static struct multiarray_writer_result
close_impl(struct multiarray_writer* self);

// ---- Bind / Unbind ----
// Copy per-array mutable state between descriptor and engine sub-structs.

static int
bind_context(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  // Whole-struct handoff; shared engine resources (sized to maxima) are
  // untouched. Tails remain in the per-array host shard state.
  return stream_engine_bind_array(e, &desc->st, &desc->ctx);
}

static void
unbind_context(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  // Without this, the next array would inherit stale fences from a
  // different sink or reuse aggregate buffers the prior D2H copy is still
  // reading.
  schedule_quiesce_output(e, desc->ctx.sink);

  // Wholesale: shard_state, accumulator counts, and schedule progress all
  // mutate over a batch.
  desc->st.sched = e->sched;
  desc->st.lod = e->lod;
  desc->st.agg = e->compress_agg.ar;

  // Clear the engine-side per-array slice so a stale pointer can't survive
  // an unbind/destroy that isn't followed by a bind.
  memset(&e->compress_agg.ar, 0, sizeof(e->compress_agg.ar));
}

// ---- Per-array init ----

static int
init_array_descriptor(struct array_descriptor_gpu* desc,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct engine_limits* lim)
{
  if (!codec_is_gpu_supported(config->codec.id))
    return 1;

  desc->ctx.config = *config;
  desc->ctx.sink = sink;
  desc->ctx.shard_alignment = shard_sink_required_shard_alignment(sink);

  if (compute_stream_layouts(config,
                             codec_alignment(config->codec.id),
                             codec_max_output_size,
                             desc->ctx.shard_alignment,
                             &desc->cl))
    return 1;

  // Fold this array into the shared-resource maxima before
  // engine_array_state_init moves cl->plan out.
  if (engine_limits_accumulate(lim, &desc->cl, config))
    return 1;

  // No delivery coordinator on the multiarray sync-flush path: every kick
  // drains immediately before per-array state is swapped.
  return engine_array_state_init(&desc->st, &desc->ctx, &desc->cl, NULL);
}

static void
destroy_array_descriptor(struct array_descriptor_gpu* desc)
{
  if (!desc)
    return;
  engine_array_state_destroy(&desc->st);
  computed_stream_layouts_free(&desc->cl);
}

// ---- Array switching ----

static int
switch_to_array(struct multiarray_tile_stream_gpu* ms, int array_index)
{
  struct stream_engine* e = &ms->engine;

  if (ms->active >= 0) {
    struct array_descriptor_gpu* departing = &ms->arrays[ms->active];

    // A failed array holds no staging and never reads the pool again, so
    // leaving it needs nothing. Holding the switch on it would strand every
    // other array in the stream.
    if (!departing->ctx.append_failed) {
      // Reject switch mid-epoch
      if (departing->ctx.cursor_elements %
            departing->ctx.layout.epoch_elements !=
          0)
        return multiarray_writer_not_flushable;

      // A failure in either step belongs to the array being left, which reports
      // it when flushed. The array being switched to stays usable.
      if (!stream_dispatch_staged(e).error && e->sched.accumulated > 0 &&
          schedule_flush_accumulated(e, &departing->ctx).error)
        departing->ctx.append_failed = 1;
    }

    unbind_context(e, departing);
    ms->active = -1;
  }

  CHECK(Fail, bind_context(e, &ms->arrays[array_index]) == 0);

  // Zero both pools for the incoming array. This is the correctness-critical
  // zero: it ensures no stale data from the departing array leaks into the
  // incoming array's scatter. (schedule_accumulate_epoch's drain-after-kick
  // path also zeros the per-array portion of the current pool as an
  // optimization for the common batch-boundary case, but that only covers
  // one pool and only the per-array size — this full zero covers both pools
  // at the max size.)
  // Host-ordered access: the departing array's immediate drains completed
  // every batch, so no produce wait is queued.
  for (int i = 0; i < 2; ++i)
    CU(Fail,
       cuMemsetD8Async(gpu_pool_view_d(gpu_pool_at(&e->pools.p, i, 0)),
                       0,
                       e->pool_bytes,
                       e->streams.compute));

  // Commit the switch only after the incoming geometry is bound and both
  // shared pools are safe for its scatter path. A failed setup is retryable.
  ms->active = array_index;
  return 0;

Fail:
  ms->active = -1;
  return multiarray_writer_fail;
}

// ---- Writer: update ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);
  if (array_index < 0 || array_index >= ms->n_arrays)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_fail,
      .rest = data,
    };

  // Ahead of the context push: refusing input touches no GPU state.
  if (ms->arrays[array_index].flushed)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_finished,
      .rest = data,
    };

  const int pushed = cu_ctx_push(ms->engine.cuda);
  if (pushed < 0)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_fail,
      .rest = data,
    };

  // One exit from here, so the context is popped once.
  struct multiarray_writer_result out = { .error = multiarray_writer_ok,
                                          .rest = data };

  if (array_index != ms->active)
    out.error = switch_to_array(ms, array_index);

  if (out.error == multiarray_writer_ok) {
    struct array_descriptor_gpu* desc = &ms->arrays[array_index];
    struct writer_result r = stream_append_body(&ms->engine, &desc->ctx, data);
    // `writer_finished` here means the array is at capacity
    // (total_element_limit); an already-finalized array is refused above.
    out.error = r.error;
    out.rest = r.rest;
  }

  cu_ctx_pop(pushed);
  return out;
}

// ---- Writer: flush ----

static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);
  const int pushed = cu_ctx_push(ms->engine.cuda);
  if (pushed < 0)
    return (struct multiarray_writer_result){ .error = multiarray_writer_fail };

  // One array failing must not leave the others unfinalized, so the whole loop
  // runs and the first failure is what gets reported.
  int failed = 0;

  // Empty staging before the loop binds another array.
  if (ms->active >= 0) {
    struct array_descriptor_gpu* desc = &ms->arrays[ms->active];
    if (stream_dispatch_staged(&ms->engine).error)
      failed = 1;
    unbind_context(&ms->engine, desc);
    ms->active = -1;
  }

  // Flush each array that has data
  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    // Finalizing twice would re-finalize an already-closed sink.
    if (desc->flushed)
      continue;
    if (desc->ctx.cursor_elements == 0 && desc->st.sched.accumulated == 0) {
      desc->flushed = 1;
      continue;
    }

    // A bind can fail while selecting this array's codec geometry. Do not
    // flush against whatever per-array state the engine held previously.
    if (bind_context(&ms->engine, desc)) {
      failed = 1;
      continue;
    }
    ms->active = a;

    struct writer_result r = stream_flush_body(&ms->engine, &desc->ctx);

    unbind_context(&ms->engine, desc);
    ms->active = -1;
    // Latched even on failure: a flush that died partway may already have
    // closed shards, so taking more input would append past them. Those writes
    // are new, so an earlier close no longer covers them.
    desc->flushed = 1;
    desc->closed = 0;
    if (r.error)
      failed = 1;
  }

  cu_ctx_pop(pushed);
  return (struct multiarray_writer_result){
    .error = failed ? multiarray_writer_fail : multiarray_writer_ok,
  };
}

// Takes each array's own aggregate state rather than binding it to the engine:
// binding would quiesce the output and kick delivery work that closing does not
// need.
static struct multiarray_writer_result
close_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);
  const int pushed = cu_ctx_push(ms->engine.cuda);
  if (pushed < 0)
    return (struct multiarray_writer_result){ .error = multiarray_writer_fail };

  int failed = 0;
  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    if (desc->closed || !desc->flushed || !desc->ctx.sink) {
      failed |= desc->close_failed;
      continue;
    }
    // desc->st.agg is only refreshed by unbind_context, so a still-bound array
    // has its live shard state in the engine.
    struct compress_agg_array* agg =
      (ms->active == a) ? &ms->engine.compress_agg.ar : &desc->st.agg;
    desc->close_failed = (stream_close_body(agg, &desc->ctx).error != 0);
    failed |= desc->close_failed;
    desc->closed = 1;
  }

  cu_ctx_pop(pushed);
  return (struct multiarray_writer_result){
    .error = failed ? multiarray_writer_fail : multiarray_writer_ok,
  };
}

// ---- Create / Destroy ----

void
multiarray_tile_stream_gpu_destroy(struct multiarray_tile_stream_gpu* ms)
{
  if (!ms)
    return;

  const int pushed = cu_ctx_push(ms->engine.cuda);

  // Auto-finalize any unflushed arrays so destroy is a safe commit point
  // for callers that didn't explicitly flush. Errors are logged but not
  // propagated — destroy returns void.
  {
    struct multiarray_writer_result r = flush_impl(&ms->writer);
    if (r.error)
      log_error("GPU multiarray auto-flush failed during destroy");
  }

  // Resolve every worker job before synchronizing streams or freeing buffers.
  gpu_delivery_stop_join(&ms->engine.delivery);
  gpu_streams_sync(&ms->engine.streams);

  // After the join, so the writes it lets the worker issue are waited out too.
  if (ms->arrays && close_impl(&ms->writer).error)
    log_error("GPU multiarray close failed during destroy");

  // Copy state can still name a per-array output pool after a failed
  // cancellation, so tear down the shared stages before the descriptors.
  stream_engine_destroy(&ms->engine);

  if (ms->arrays) {
    for (int a = 0; a < ms->n_arrays; ++a)
      destroy_array_descriptor(&ms->arrays[a]);
    free(ms->arrays);
  }

  cu_ctx_pop(pushed);
  free(ms);
}

struct multiarray_tile_stream_gpu*
multiarray_tile_stream_gpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[],
  int enable_metrics)
{
  // enable_metrics is ignored: CUDA events are recorded for stream sync
  // regardless, so metrics collection has no meaningful opt-out on the GPU
  // path. See multiarray.gpu.h.
  (void)enable_metrics;

  if (n_arrays <= 0)
    return NULL;

  struct multiarray_tile_stream_gpu* ms =
    (struct multiarray_tile_stream_gpu*)calloc(1, sizeof(*ms));
  if (!ms)
    return NULL;

  ms->n_arrays = n_arrays;
  ms->active = -1;
  ms->writer.update = update_impl;
  ms->writer.flush = flush_impl;
  ms->writer.close = close_impl;

  ms->arrays = (struct array_descriptor_gpu*)calloc(
    n_arrays, sizeof(struct array_descriptor_gpu));
  CHECK(Fail, ms->arrays);

  struct engine_limits lim;
  memset(&lim, 0, sizeof(lim));

  for (int a = 0; a < n_arrays; ++a)
    CHECK(Fail,
          init_array_descriptor(&ms->arrays[a], &configs[a], sinks[a], &lim) ==
            0);

  // Validate: all arrays must use the same codec (shared codec instance).
  for (int a = 1; a < n_arrays; ++a) {
    if (ms->arrays[a].ctx.config.codec.id !=
        ms->arrays[0].ctx.config.codec.id) {
      log_error("GPU multiarray: all arrays must use the same codec");
      goto Fail;
    }
  }

  // Label scatter as "Copy" only when every array uses multiscale (matches
  // single-array GPU).  When any array is non-multiscale, the scatter kernel
  // runs directly into the chunk pool, so keep the generic label.
  int all_multiscale = 1;
  for (int a = 0; a < n_arrays; ++a) {
    if (!ms->arrays[a].ctx.levels.enable_multiscale) {
      all_multiscale = 0;
      break;
    }
  }

  CHECK(Fail,
        stream_engine_init(&ms->engine,
                           &lim,
                           ms->arrays[0].ctx.config.codec.id,
                           all_multiscale) == 0);

  return ms;

Fail:
  multiarray_tile_stream_gpu_destroy(ms);
  return NULL;
}

// ---- Accessors ----

struct multiarray_writer*
multiarray_tile_stream_gpu_writer(struct multiarray_tile_stream_gpu* ms)
{
  return &ms->writer;
}

struct stream_metrics
multiarray_tile_stream_gpu_get_metrics(
  const struct multiarray_tile_stream_gpu* ms)
{
  struct stream_metrics metrics = ms->engine.metrics;
  for (int i = 0; i < ms->n_arrays; ++i)
    host_output_pool_accumulate_metrics(ms->arrays[i].st.agg.output_pool,
                                        &metrics);
  return metrics;
}

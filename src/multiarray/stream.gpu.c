#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.engine.h"
#include "gpu/stream.flush.h"
#include "gpu/stream.ingest.h"
#include "gpu/stream.lod.h"

#include "gpu/prelude.cuda.h"
#include "multiarray.gpu.h"
#include "stream/config.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

// ---- Per-array descriptor ----
// Extends stream_context with mutable per-array state that is swapped
// into/out of the engine on array switch.

struct array_descriptor_gpu
{
  struct stream_context ctx;
  struct computed_stream_layouts cl; // owned, freed on destroy

  // Per-array LOD state (owns plan, layouts[], layout_gpu[], CSRs, append
  // accumulator device memory, and LOD LUTs — but NOT d_linear/d_morton/timing,
  // which are shared and owned by the engine).
  struct lod_state array_lod;

  // Mutable per-array state (saved/restored via bind/unbind)
  uint32_t batch_accumulated;
  int pools_current;
  struct flush_slot_gpu flush_slots[2];
  int flush_pending;
  int flush_current;
  struct flush_handoff flush_pending_handoff;
  struct shard_state shard[LOD_MAX_LEVELS];
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];
  uint32_t batch_active_count[LOD_MAX_LEVELS];
  int flushed; // 1 once flush body has run for this array
};

// ---- Pool maxima (computed across all arrays) ----

struct pool_maxima
{
  size_t pool_bytes;
  size_t buffer_capacity;
  size_t compressed_bytes;
  uint64_t codec_batch;
  size_t chunk_bytes;
  uint32_t epochs_per_batch;
  int max_nlod;
  size_t max_output_size;
  size_t lod_linear_bytes; // max across arrays
  size_t lod_morton_bytes; // max across arrays
  int any_multiscale;      // 1 if any array uses multiscale

  struct
  {
    uint64_t batch_chunks;
    uint64_t batch_covering;
    size_t batch_agg_bytes;
    uint64_t lut_len;
  } level[LOD_MAX_LEVELS];
};

// ---- Main struct ----

struct multiarray_tile_stream_gpu
{
  struct multiarray_writer writer;
  struct stream_engine engine;
  int n_arrays;
  int active; // -1 = none
  int max_nlod;
  struct array_descriptor_gpu* arrays;
};

// ---- Forward declarations ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data);
static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self);

// ---- Helpers ----

static inline size_t
max_sz(size_t a, size_t b)
{
  return a > b ? a : b;
}

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
  return a > b ? a : b;
}

static inline uint32_t
max_u32(uint32_t a, uint32_t b)
{
  return a > b ? a : b;
}

// ---- Bind / Unbind ----
// Copy per-array mutable state between descriptor and engine sub-structs.

static void
bind_context(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  // Set batch K from the array's own config, not the engine max.
  // Shared buffers are sized to max K, but each array should fill/flush at
  // its own K so that batch_active_count and active_count agree.
  e->batch.epochs_per_batch = desc->cl.epochs_per_batch;
  e->batch.accumulated = desc->batch_accumulated;
  e->pools.current = desc->pools_current;
  for (int i = 0; i < 2; ++i)
    e->flush.slot[i] = desc->flush_slots[i];
  e->flush.pending = desc->flush_pending;
  e->flush.current = desc->flush_current;
  e->flush.pending_handoff = desc->flush_pending_handoff;
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    e->compress_agg.levels[lv].shard = desc->shard[lv];
    e->compress_agg.levels[lv].agg_layout = desc->agg_layout[lv];
    e->compress_agg.levels[lv].batch_active_count =
      desc->batch_active_count[lv];
  }
  e->d2h_deliver.nlod = desc->ctx.levels.nlod;
  e->d2h_deliver.shard_alignment = desc->ctx.shard_alignment;

  // Per-array LOD state is now a single struct assignment.  Shared LOD
  // resources (d_linear, d_morton, timing) live in e->lod_shared and are
  // untouched by bind/unbind — the type system enforces this.
  e->lod = desc->array_lod;
}

static void
unbind_context(struct stream_engine* e, struct array_descriptor_gpu* desc)
{
  desc->batch_accumulated = e->batch.accumulated;
  desc->pools_current = e->pools.current;
  for (int i = 0; i < 2; ++i)
    desc->flush_slots[i] = e->flush.slot[i];
  desc->flush_pending = e->flush.pending;
  desc->flush_current = e->flush.current;
  desc->flush_pending_handoff = e->flush.pending_handoff;
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    desc->shard[lv] = e->compress_agg.levels[lv].shard;
    desc->agg_layout[lv] = e->compress_agg.levels[lv].agg_layout;
    desc->batch_active_count[lv] =
      e->compress_agg.levels[lv].batch_active_count;
  }

  // Save per-array LOD mutable state. counts[] and total_elements track
  // running append-accumulator state across epochs.
  if (desc->ctx.levels.enable_multiscale) {
    memcpy(desc->array_lod.append_accum.counts,
           e->lod.append_accum.counts,
           sizeof(desc->array_lod.append_accum.counts));
    desc->array_lod.append_accum.total_elements =
      e->lod.append_accum.total_elements;
  }
}

// ---- Per-array init ----

static int
init_array_descriptor(struct array_descriptor_gpu* desc,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct pool_maxima* mx)
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

  desc->ctx.layout = desc->cl.layouts[0];
  desc->ctx.levels = desc->cl.levels;
  desc->ctx.dims = desc->cl.dims;

  // Initialize per-array LOD state: transfer plan from cl, copy layouts,
  // upload level layouts + LUTs + CSRs. Does NOT allocate d_linear/d_morton
  // or timing events — those are engine-owned shared resources.
  desc->array_lod.plan = desc->cl.plan;
  desc->cl.plan = (struct lod_plan){ 0 }; // ownership transferred
  for (int lv = 0; lv < desc->cl.levels.nlod; ++lv)
    desc->array_lod.layouts[lv] = desc->cl.layouts[lv];

  if (lod_state_init(&desc->array_lod, &desc->ctx.levels, &desc->ctx.config))
    return 1;

  // Alias L0 layout GPU pointers from array_lod into ctx (for scatter).
  desc->ctx.layout_gpu = desc->array_lod.layout_gpu[0];

  if (desc->ctx.levels.enable_multiscale && desc->ctx.dims.append_downsample) {
    if (lod_state_init_accumulators(&desc->array_lod, &desc->ctx.config))
      return 1;
  }

  const uint32_t K = desc->cl.epochs_per_batch;
  const size_t bpe = dtype_bpe(config->dtype);
  const uint64_t total_chunks = desc->ctx.levels.total_chunks;
  const uint64_t chunk_stride = desc->ctx.layout.chunk_stride;

  // max_cursor
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&desc->ctx.dims);
    if (dims[0].size > 0) {
      desc->ctx.max_cursor_elements = desc->ctx.layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        desc->ctx.max_cursor_elements *=
          ceildiv(dims[d].size, dims[d].chunk_size);
    }
  }

  // Buffer capacity (page-aligned)
  desc->ctx.config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // Update pool maxima
  mx->pool_bytes =
    max_sz(mx->pool_bytes, (size_t)K * total_chunks * chunk_stride * bpe);
  mx->buffer_capacity =
    max_sz(mx->buffer_capacity, desc->ctx.config.buffer_capacity_bytes);
  mx->compressed_bytes = max_sz(
    mx->compressed_bytes, (size_t)K * total_chunks * desc->cl.max_output_size);
  mx->codec_batch = max_u64(mx->codec_batch, (uint64_t)K * total_chunks);
  mx->chunk_bytes = max_sz(mx->chunk_bytes, chunk_stride * bpe);
  mx->epochs_per_batch = max_u32(mx->epochs_per_batch, K);
  mx->max_output_size = max_sz(mx->max_output_size, desc->cl.max_output_size);

  if (desc->ctx.levels.nlod > mx->max_nlod)
    mx->max_nlod = desc->ctx.levels.nlod;

  // LOD buffer sizes (for engine's shared d_linear / d_morton).
  if (desc->ctx.levels.enable_multiscale) {
    mx->any_multiscale = 1;
    size_t linear_bytes = desc->ctx.layout.epoch_elements * bpe;
    mx->lod_linear_bytes = max_sz(mx->lod_linear_bytes, linear_bytes);
    uint64_t total_vals = desc->array_lod.plan.level_spans
                            .ends[desc->array_lod.plan.levels.nlod - 1];
    size_t morton_bytes = total_vals * bpe;
    mx->lod_morton_bytes = max_sz(mx->lod_morton_bytes, morton_bytes);
  }

  // Per-level aggregate + shard state
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    const struct level_layout_info* li = &desc->cl.per_level[lv];
    desc->agg_layout[lv] = li->agg_layout;
    desc->batch_active_count[lv] = li->batch_active_count;

    uint32_t slot_count =
      li->batch_active_count > 0 ? li->batch_active_count : 1;
    uint64_t chunks_lv = desc->ctx.levels.level[lv].chunk_count;
    uint64_t batch_chunks = (uint64_t)slot_count * chunks_lv;
    uint64_t batch_covering =
      (uint64_t)slot_count * li->agg_layout.covering_count;
    size_t batch_agg_bytes = agg_pool_bytes(batch_chunks,
                                            desc->cl.max_output_size,
                                            li->agg_layout.covering_count,
                                            li->agg_layout.cps_inner,
                                            li->agg_layout.page_size);

    mx->level[lv].batch_chunks =
      max_u64(mx->level[lv].batch_chunks, batch_chunks);
    mx->level[lv].batch_covering =
      max_u64(mx->level[lv].batch_covering, batch_covering);
    mx->level[lv].batch_agg_bytes =
      max_sz(mx->level[lv].batch_agg_bytes, batch_agg_bytes);
    mx->level[lv].lut_len = max_u64(mx->level[lv].lut_len, batch_chunks);

    if (init_shard_state(&desc->shard[lv], li))
      return 1;
  }

  return 0;
}

static void
destroy_array_descriptor(struct array_descriptor_gpu* desc)
{
  if (!desc)
    return;
  for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv) {
    struct shard_state* ss = &desc->shard[lv];
    if (ss->shards) {
      for (uint64_t i = 0; i < ss->shard_inner_count; ++i)
        free(ss->shards[i].index);
      free(ss->shards);
    }
    aggregate_layout_destroy(&desc->agg_layout[lv]);
  }
  // array_lod owns everything except d_linear/d_morton/timing (which stay 0
  // in the per-array struct). ctx.layout_gpu aliases array_lod.layout_gpu[0]
  // and is freed via array_lod destroy.
  lod_state_destroy(&desc->array_lod);
  computed_stream_layouts_free(&desc->cl);
}

// ---- Shared resource allocation ----

static int
init_shared_resources(struct multiarray_tile_stream_gpu* ms,
                      const struct pool_maxima* mx)
{
  struct stream_engine* e = &ms->engine;

  // CUDA streams
  CU(Fail, cuStreamCreate(&e->streams.h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&e->streams.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&e->streams.compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&e->streams.d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i)
    CU(Fail, cuEventCreate(&e->pools.ready[i], CU_EVENT_DEFAULT));

  CHECK(Fail,
        ingest_init(&e->stage, mx->buffer_capacity, e->streams.compute) == 0);

  e->pool_bytes = mx->pool_bytes;
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&e->pools.buf[i], mx->pool_bytes));
    CU(Fail,
       cuMemsetD8Async(e->pools.buf[i], 0, mx->pool_bytes, e->streams.compute));
  }

  e->batch.epochs_per_batch = mx->epochs_per_batch;
  for (uint32_t i = 0; i < mx->epochs_per_batch; ++i) {
    CU(Fail, cuEventCreate(&e->batch.pool_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(e->batch.pool_events[i], e->streams.compute));
  }

  CHECK(Fail,
        codec_init(&e->compress_agg.codec,
                   ms->arrays[0].ctx.config.codec.id,
                   mx->chunk_bytes,
                   mx->codec_batch) == 0);

  for (int fc = 0; fc < 2; ++fc) {
    size_t comp_sz = mx->codec_batch * e->compress_agg.codec.max_output_size;
    if (comp_sz > 0)
      CU(Fail, cuMemAlloc(&e->compress_agg.d_compressed[fc], comp_sz));
    CU(Fail,
       cuEventCreate(&e->compress_agg.t_compress_start[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&e->compress_agg.t_compress_end[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&e->compress_agg.t_aggregate_end[fc], CU_EVENT_DEFAULT));
  }

  for (int lv = 0; lv < ms->max_nlod; ++lv) {
    struct level_flush_state* lvl = &e->compress_agg.levels[lv];
    for (int fc = 0; fc < 2; ++fc) {
      if (mx->level[lv].batch_covering > 0) {
        CHECK(Fail,
              aggregate_batch_slot_init(&lvl->agg[fc],
                                        mx->level[lv].batch_chunks,
                                        mx->level[lv].batch_covering,
                                        mx->level[lv].batch_agg_bytes) == 0);
        CU(Fail, cuEventRecord(lvl->agg[fc].ready, e->streams.compute));
      }
    }
    if (mx->level[lv].lut_len > 0) {
      size_t lut_bytes = mx->level[lv].lut_len * sizeof(uint32_t);
      CU(Fail, cuMemAlloc(&lvl->d_batch_gather, lut_bytes));
      CU(Fail, cuMemAlloc(&lvl->d_batch_perm, lut_bytes));
    }
  }

  CHECK(Fail,
        d2h_deliver_init(&e->d2h_deliver,
                         e->compress_agg.levels,
                         ms->max_nlod,
                         0,
                         e->streams.compute) == 0);
  e->d2h_deliver.metrics = &e->metrics;

  CU(Fail, cuEventRecord(e->pools.ready[0], e->streams.compute));
  CU(Fail, cuEventRecord(e->pools.ready[1], e->streams.compute));

  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail,
       cuEventRecord(e->compress_agg.t_compress_start[fc], e->streams.compute));
    CU(Fail,
       cuEventRecord(e->compress_agg.t_compress_end[fc], e->streams.compute));
    CU(Fail,
       cuEventRecord(e->compress_agg.t_aggregate_end[fc], e->streams.compute));
  }

  // Shared LOD buffers (sized to max across arrays). Only allocated if any
  // array uses multiscale.  The struct lod_shared_state / lod_state split
  // keeps engine-owned resources separate from per-array state, so bind/unbind
  // never touches these fields.
  if (mx->any_multiscale) {
    CHECK(Fail,
          lod_shared_state_init(&e->lod_shared,
                                mx->lod_linear_bytes,
                                mx->lod_morton_bytes,
                                e->streams.compute) == 0);
  }

  CU(Fail, cuStreamSynchronize(e->streams.compute));

  return 0;

Fail:
  return 1;
}

// ---- Array switching ----

static int
switch_to_array(struct multiarray_tile_stream_gpu* ms, int array_index)
{
  struct stream_engine* e = &ms->engine;

  if (ms->active >= 0) {
    struct array_descriptor_gpu* departing = &ms->arrays[ms->active];

    // Reject switch mid-epoch
    if (departing->ctx.cursor_elements % departing->ctx.layout.epoch_elements !=
        0)
      return multiarray_writer_not_flushable;

    // Flush departing array's accumulated batch. With sync_flush=1,
    // stream_flush_body uses the synchronous path (no pool swap or
    // pending state), so it's safe to call during switch.
    if (e->batch.accumulated > 0) {
      struct writer_result r = flush_accumulated_sync(e, &departing->ctx);
      if (r.error)
        return multiarray_writer_fail;
    }

    unbind_context(e, departing);
  }

  ms->active = array_index;
  bind_context(e, &ms->arrays[array_index]);

  // Zero both pools for the incoming array. This is the correctness-critical
  // zero: it ensures no stale data from the departing array leaks into the
  // incoming array's scatter. (flush_accumulate_epoch's sync path also zeros
  // the per-array portion of the current pool as an optimization for the
  // common batch-boundary case, but that only covers one pool and only the
  // per-array size — this full zero covers both pools at the max size.)
  for (int i = 0; i < 2; ++i)
    CU(Fail,
       cuMemsetD8Async(e->pools.buf[i], 0, e->pool_bytes, e->streams.compute));

  return 0;

Fail:
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

  struct array_descriptor_gpu* desc = &ms->arrays[array_index];

  // If this array has already been flushed (capacity reached with inline
  // flush, or explicit flush), further appends are a no-op that report
  // `finished` with the full input unconsumed.
  if (desc->flushed)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_finished,
      .rest = data,
    };

  // Switch arrays if needed
  if (array_index != ms->active) {
    int err = switch_to_array(ms, array_index);
    if (err)
      return (struct multiarray_writer_result){ .error = err, .rest = data };
  }

  struct writer_result r = stream_append_body(&ms->engine, &desc->ctx, data);

  // `writer_finished` here means "stream is at capacity"; finalization
  // happens on explicit `flush()` or on destroy, not here.
  return (struct multiarray_writer_result){
    .error = r.error,
    .rest = r.rest,
  };
}

// ---- Writer: flush ----

static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);

  // Save current array's state
  if (ms->active >= 0)
    unbind_context(&ms->engine, &ms->arrays[ms->active]);

  // Flush each array that has data
  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    // Already-flushed arrays (either by inline flush on capacity or by a
    // prior explicit flush) re-entering the body would re-finalize an
    // already-finalized sink.
    if (desc->flushed)
      continue;
    if (desc->ctx.cursor_elements == 0 && desc->batch_accumulated == 0) {
      desc->flushed = 1;
      continue;
    }

    ms->active = a;
    bind_context(&ms->engine, desc);

    struct writer_result r = stream_flush_body(&ms->engine, &desc->ctx);
    if (r.error)
      goto Error;

    unbind_context(&ms->engine, desc);
    desc->flushed = 1;
  }

  ms->active = -1;
  return (struct multiarray_writer_result){ .error = multiarray_writer_ok };

Error:
  if (ms->active >= 0)
    unbind_context(&ms->engine, &ms->arrays[ms->active]);
  ms->active = -1;
  return (struct multiarray_writer_result){ .error = multiarray_writer_fail };
}

// ---- Create / Destroy ----

static void
sync_all(struct gpu_streams* streams)
{
  if (streams->h2d)
    cuStreamSynchronize(streams->h2d);
  if (streams->compute)
    cuStreamSynchronize(streams->compute);
  if (streams->compress)
    cuStreamSynchronize(streams->compress);
  if (streams->d2h)
    cuStreamSynchronize(streams->d2h);
}

void
multiarray_tile_stream_gpu_destroy(struct multiarray_tile_stream_gpu* ms)
{
  if (!ms)
    return;

  // Auto-finalize any unflushed arrays so destroy is a safe commit point
  // for callers that didn't explicitly flush. Errors here are swallowed —
  // the stream is tearing down, there's no one to report to.
  (void)flush_impl(&ms->writer);

  sync_all(&ms->engine.streams);

  if (ms->arrays) {
    for (int a = 0; a < ms->n_arrays; ++a)
      destroy_array_descriptor(&ms->arrays[a]);
    free(ms->arrays);
  }

  struct stream_engine* e = &ms->engine;

  for (uint32_t i = 0; i < e->batch.epochs_per_batch; ++i)
    cu_event_destroy(e->batch.pool_events[i]);

  d2h_deliver_destroy(&e->d2h_deliver);

  codec_free(&e->compress_agg.codec);
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(e->compress_agg.d_compressed[fc]);
    cu_event_destroy(e->compress_agg.t_compress_start[fc]);
    cu_event_destroy(e->compress_agg.t_compress_end[fc]);
    cu_event_destroy(e->compress_agg.t_aggregate_end[fc]);
  }
  for (int lv = 0; lv < ms->max_nlod; ++lv) {
    struct level_flush_state* lvl = &e->compress_agg.levels[lv];
    for (int fc = 0; fc < 2; ++fc)
      aggregate_slot_destroy(&lvl->agg[fc]);
    cu_mem_free(lvl->d_batch_gather);
    cu_mem_free(lvl->d_batch_perm);
  }

  // The per-array fields of e->lod are views of the last-bound descriptor
  // (freed above by destroy_array_descriptor).  Engine-owned shared LOD
  // resources live in e->lod_shared.
  lod_shared_state_destroy(&e->lod_shared);

  for (int i = 0; i < 2; ++i) {
    cu_mem_free(e->pools.buf[i]);
    cu_event_destroy(e->pools.ready[i]);
  }

  ingest_destroy(&e->stage);

  cu_stream_destroy(e->streams.h2d);
  cu_stream_destroy(e->streams.compute);
  cu_stream_destroy(e->streams.compress);
  cu_stream_destroy(e->streams.d2h);

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

  ms->arrays = (struct array_descriptor_gpu*)calloc(
    n_arrays, sizeof(struct array_descriptor_gpu));
  CHECK(Fail, ms->arrays);

  // Phase 1: compute per-array layouts and pool maxima
  struct pool_maxima mx;
  memset(&mx, 0, sizeof(mx));

  for (int a = 0; a < n_arrays; ++a)
    CHECK(Fail,
          init_array_descriptor(&ms->arrays[a], &configs[a], sinks[a], &mx) ==
            0);

  ms->max_nlod = mx.max_nlod;

  // Validate: all arrays must use the same codec (shared codec instance).
  for (int a = 1; a < n_arrays; ++a) {
    if (ms->arrays[a].ctx.config.codec.id !=
        ms->arrays[0].ctx.config.codec.id) {
      log_error("GPU multiarray: all arrays must use the same codec");
      goto Fail;
    }
  }

  // Phase 2: upload per-array aggregate layouts to GPU
  for (int a = 0; a < n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    for (int lv = 0; lv < desc->ctx.levels.nlod; ++lv)
      CHECK(Fail, aggregate_layout_upload(&desc->agg_layout[lv]) == 0);
  }

  // Phase 3: allocate shared GPU resources
  // (Per-array L0 layout_gpu is aliased from array_lod.layout_gpu[0], which
  // was uploaded during lod_state_init in init_array_descriptor.)
  CHECK(Fail, init_shared_resources(ms, &mx) == 0);

  // Use synchronous flush path — the double-buffered pipeline doesn't
  // compose across array switches.
  ms->engine.sync_flush = 1;

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
  ms->engine.metrics = stream_engine_init_metrics(all_multiscale);
  ms->engine.metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&ms->engine.metadata_update_clock);

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
  return ms->engine.metrics;
}

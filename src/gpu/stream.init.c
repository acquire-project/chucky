#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.flush.h"
#include "gpu/stream.ingest.h"
#include "gpu/stream.lod.h"

#include "defs.limits.h"
#include "gpu/prelude.cuda.h"
#include "lod/lod_plan.h"
#include "platform/platform.h"
#include "stream/config.h"
#include "util/prelude.h"
#include "writer.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

// --- Engine limits ---

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

int
engine_limits_accumulate(struct engine_limits* lim,
                         const struct computed_stream_layouts* cl,
                         const struct tile_stream_configuration* config)
{
  const uint32_t K = cl->epochs_per_batch;
  const size_t bpe = dtype_bpe(config->dtype);
  const uint64_t total_chunks = cl->levels.total_chunks;
  const uint64_t chunk_stride = cl->layouts[0].chunk_stride;

  CHECK(Fail, K >= 1);
  CHECK_MUL_OVERFLOW(Fail, K, total_chunks, UINT64_MAX);

  lim->buffer_capacity =
    max_sz(lim->buffer_capacity,
           (config->buffer_capacity_bytes + 4095) & ~(size_t)4095);
  lim->pool_bytes =
    max_sz(lim->pool_bytes, (size_t)K * total_chunks * chunk_stride * bpe);
  lim->chunk_bytes = max_sz(lim->chunk_bytes, chunk_stride * bpe);
  lim->codec_batch = max_u64(lim->codec_batch, (uint64_t)K * total_chunks);
  if (K > lim->epochs_per_batch)
    lim->epochs_per_batch = K;
  if (cl->levels.nlod > lim->max_nlod)
    lim->max_nlod = cl->levels.nlod;

  if (cl->levels.enable_multiscale) {
    lim->any_multiscale = 1;
    lim->lod_linear_bytes =
      max_sz(lim->lod_linear_bytes, cl->layouts[0].epoch_elements * bpe);
    const uint64_t morton_vals =
      cl->plan.level_spans.ends[cl->plan.levels.nlod - 1];
    lim->lod_morton_bytes = max_sz(lim->lod_morton_bytes, morton_vals * bpe);
  }

  // Max batch layout assuming each LOD fires its worst-case active count.
  {
    struct batch_aggregate_layout ml;
    struct aggregate_layout per_lod_layouts[LOD_MAX_LEVELS];
    uint32_t per_lod_max[LOD_MAX_LEVELS] = { 0 };
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      per_lod_layouts[lv] = cl->per_level[lv].agg_layout;
      per_lod_max[lv] = cl->per_level[lv].batch_active_count;
    }
    CHECK(Fail,
          batch_aggregate_layout_init(&ml,
                                      per_lod_layouts,
                                      per_lod_max,
                                      (uint8_t)cl->levels.nlod,
                                      cl->per_level[0].agg_layout.page_size) ==
            0);
    lim->max_total_batch_chunks =
      max_u64(lim->max_total_batch_chunks, ml.total_batch_chunks);
    lim->max_total_batch_covering =
      max_u64(lim->max_total_batch_covering, ml.total_batch_covering);
    lim->max_total_data_bytes =
      max_sz(lim->max_total_data_bytes, ml.total_data_bytes);
  }

  {
    uint64_t total_shards = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv)
      total_shards += cl->per_level[lv].agg_layout.num_shards;
    lim->max_total_shards = max_u64(lim->max_total_shards, total_shards);
  }

  return 0;

Fail:
  return 1;
}

// --- Shared engine init / teardown ---

static int
init_cuda_streams(struct gpu_streams* streams)
{
  CU(Fail, cuStreamCreate(&streams->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->d2h, CU_STREAM_NON_BLOCKING));

  return 0;
Fail:
  return 1;
}

static void
destroy_cuda_streams(struct gpu_streams* streams)
{
  cu_stream_destroy(streams->h2d);
  cu_stream_destroy(streams->compute);
  cu_stream_destroy(streams->compress);
  cu_stream_destroy(streams->d2h);
}

static void
destroy_chunk_pools(struct pool_state* pools)
{
  for (int i = 0; i < 2; ++i)
    cu_mem_free(pools->buf[i]);
}

int
stream_engine_init(struct stream_engine* e,
                   const struct engine_limits* lim,
                   enum compression_codec codec_id,
                   int scatter_is_copy)
{
  CHECK(Fail, init_cuda_streams(&e->streams) == 0);
  CHECK(Fail, gpu_ordering_init(&e->ord, e->streams.compute) == 0);
  gpu_ordering_register_stream(&e->ord, GPU_STREAM_H2D, e->streams.h2d);
  gpu_ordering_register_stream(&e->ord, GPU_STREAM_COMPUTE, e->streams.compute);
  gpu_ordering_register_stream(
    &e->ord, GPU_STREAM_COMPRESS, e->streams.compress);
  gpu_ordering_register_stream(&e->ord, GPU_STREAM_D2H, e->streams.d2h);

  CHECK(Fail,
        ingest_init(
          &e->stage, lim->buffer_capacity, &e->ord, e->streams.compute) == 0);

  e->pool_bytes = lim->pool_bytes;
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&e->pools.buf[i], lim->pool_bytes));
    CU(Fail,
       cuMemsetD8Async(e->pools.buf[i], 0, lim->pool_bytes, e->streams.compute));
  }

  e->batch.epochs_per_batch = lim->epochs_per_batch;

  CHECK(Fail,
        compress_agg_init_shared(
          &e->compress_agg, lim, codec_id, &e->ord, e->streams.compute) == 0);

  // shard_alignment is per-array; set by stream_engine_bind_array.
  CHECK(Fail,
        d2h_deliver_init(&e->d2h_deliver, 0, &e->ord, e->streams.compute) == 0);

  // Shared LOD buffers (sized to the max across arrays).
  if (lim->any_multiscale) {
    CHECK(Fail,
          lod_shared_state_init(&e->lod_shared,
                                lim->lod_linear_bytes,
                                lim->lod_morton_bytes,
                                e->streams.compute) == 0);
    // t_end doubles as GPU_EDGE_LOD_DONE; already seeded by the init above.
    for (int fc = 0; fc < 2; ++fc)
      gpu_ordering_bind(
        &e->ord, GPU_EDGE_LOD_DONE, fc, e->lod_shared.timing[fc].t_end);
  }

  CU(Fail, cuStreamSynchronize(e->streams.compute));

  e->metrics = stream_engine_init_metrics(scatter_is_copy);
  e->d2h_deliver.metrics = &e->metrics;
  stream_engine_attach_edge_stalls(e);
  e->metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&e->metadata_update_clock);

  return 0;

Fail:
  return 1; // caller tears down via stream_engine_destroy
}

void
stream_engine_destroy(struct stream_engine* e)
{
  d2h_deliver_destroy(&e->d2h_deliver);
  compress_agg_destroy_shared(&e->compress_agg);
  lod_shared_state_destroy(&e->lod_shared);
  destroy_chunk_pools(&e->pools);
  ingest_destroy(&e->stage);
  gpu_ordering_destroy(&e->ord);
  destroy_cuda_streams(&e->streams);
}

// --- Per-array state ---

int
engine_array_state_init(struct engine_array_state* st,
                        struct stream_context* ctx,
                        struct computed_stream_layouts* cl,
                        struct gpu_ordering* gate_ord,
                        CUstream gate_stream)
{
  memset(st, 0, sizeof(*st));

  ctx->layout = cl->layouts[0]; // host fields; d_* still NULL
  ctx->levels = cl->levels;
  ctx->dims = cl->dims;
  ctx->config.buffer_capacity_bytes =
    (ctx->config.buffer_capacity_bytes + 4095) & ~(size_t)4095;

  st->batch.epochs_per_batch = cl->epochs_per_batch;

  // Move LOD plan and level layouts (always, including L0). st->lod owns
  // plan, layouts[], layout_gpu[], CSRs, accumulators, and LOD LUTs — but
  // NOT d_linear/d_morton/timing, which are engine-owned shared resources.
  st->lod.plan = cl->plan;
  cl->plan = (struct lod_plan){ 0 }; // ownership transferred
  for (int lv = 0; lv < cl->levels.nlod; ++lv)
    st->lod.layouts[lv] = cl->layouts[lv];

  CHECK(Fail, lod_state_init(&st->lod, &ctx->levels, &ctx->config) == 0);
  // Alias L0 layout GPU pointers from LOD state into context (for scatter).
  ctx->layout_gpu = st->lod.layout_gpu[0];

  if (ctx->levels.enable_multiscale && ctx->dims.append_downsample)
    CHECK(Fail, lod_state_init_accumulators(&st->lod, &ctx->config) == 0);

  // Per-slot batch_active_masks sized to this array's K.
  for (int fc = 0; fc < 2; ++fc) {
    st->flush.slot[fc].batch_active_masks =
      (uint32_t*)calloc(cl->epochs_per_batch, sizeof(uint32_t));
    CHECK(Fail, st->flush.slot[fc].batch_active_masks);
  }

  CHECK(Fail, compress_agg_array_init(&st->agg, cl, gate_ord, gate_stream) == 0);

  // total_element_limit: configured stream length (0 = unbounded). Lets the
  // append body detect the at-capacity case without recomputing each call.
  {
    const struct dimension* dims = ctx->config.dimensions;
    const uint8_t na = dim_info_n_append(&ctx->dims);
    if (dims[0].size > 0) {
      ctx->total_element_limit = ctx->layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        ctx->total_element_limit *= ceildiv(dims[d].size, dims[d].chunk_size);
    }
  }

  return 0;

Fail:
  engine_array_state_destroy(st);
  return 1;
}

void
engine_array_state_destroy(struct engine_array_state* st)
{
  if (!st)
    return;
  for (int fc = 0; fc < 2; ++fc) {
    free(st->flush.slot[fc].batch_active_masks);
    st->flush.slot[fc].batch_active_masks = NULL;
  }
  compress_agg_array_destroy(&st->agg);
  lod_state_destroy(&st->lod);
}

int
stream_engine_bind_array(struct stream_engine* e,
                         const struct engine_array_state* st,
                         const struct stream_context* ctx)
{
  e->batch = st->batch;
  e->pools.current = st->pool_current;
  e->flush = st->flush;
  e->lod = st->lod;
  e->compress_agg.ar = st->agg;
  e->d2h_deliver.shard_alignment = ctx->shard_alignment;

  // Per-array shard_capacity table is constant; upload on bind so the
  // device-side d_shard_capacity reflects the active array's shard sizes.
  // (h_base_offsets / h_tps_group / h_offsets_base are per-batch scratch,
  // refreshed by the kick.)
  if (st->agg.total_shards > 0 && st->agg.h_shard_capacity)
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)e->compress_agg.shards.d_shard_capacity,
                    st->agg.h_shard_capacity,
                    st->agg.total_shards * sizeof(size_t)));

  // Invalidate the LUT cache: per-array layouts differ.
  e->compress_agg.lut_cache_valid = 0;
  return 0;

Fail:
  return 1;
}

// --- Create / Destroy ---

static void
sync(CUstream stream)
{
  if (stream)
    cuStreamSynchronize(stream);
}

void
tile_stream_gpu_destroy(struct tile_stream_gpu* s)
{
  if (!s)
    return;

  // Auto-finalize any unwritten data so destroy is a safe commit point for
  // callers that didn't explicitly flush. Errors are logged but not
  // propagated — destroy returns void.
  if (!s->flushed) {
    struct writer_result r = stream_flush_body(&s->engine, &s->ctx);
    if (r.error)
      log_error("GPU stream auto-flush failed during destroy");
    s->flushed = 1;
  }

  // A failed flush can leave a kick parked on the tail gate; release it or
  // the syncs below never return.
  gpu_edge_release_all(&s->engine.ord);

  // Ensure all GPU work completes before tearing down events/memory.
  sync(s->engine.streams.h2d);
  sync(s->engine.streams.compute);
  sync(s->engine.streams.compress);
  sync(s->engine.streams.d2h);

  // s->ar owns the per-array allocations; the engine holds a bound copy of
  // the same pointers, so destroy strictly after engine teardown would
  // double-free — free once, here.
  engine_array_state_destroy(&s->ar);
  stream_engine_destroy(&s->engine);
  free(s);
}

struct tile_stream_gpu*
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct shard_sink* sink)
{
  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));

  CHECK(FailPhase1, config && sink);

  if (!codec_is_gpu_supported(config->codec.id)) {
    log_error("codec %d is not supported on GPU", config->codec.id);
    goto FailPhase1;
  }

  // Phase 1: CPU-only layout computation.
  CHECK(FailPhase1,
        compute_stream_layouts(config,
                               codec_alignment(config->codec.id),
                               codec_max_output_size,
                               shard_sink_required_shard_alignment(sink),
                               &cl) == 0);

  // Phase 2: allocate engine (shared) + array state, then bind.
  struct tile_stream_gpu* out =
    (struct tile_stream_gpu*)calloc(1, sizeof(*out));
  CHECK(FailPhase1b, out);

  out->ctx.config = *config;
  out->ctx.sink = sink;
  out->ctx.shard_alignment = shard_sink_required_shard_alignment(sink);
  tile_stream_gpu_init_writer(out);

  struct engine_limits lim;
  memset(&lim, 0, sizeof(lim));
  CHECK(FailPhase2, engine_limits_accumulate(&lim, &cl, config) == 0);
  CHECK(FailPhase2,
        stream_engine_init(&out->engine,
                           &lim,
                           config->codec.id,
                           cl.levels.enable_multiscale) == 0);
  // The pipelined (non-sync-flush) path arms the tail gate.
  CHECK(FailPhase2,
        engine_array_state_init(&out->ar,
                                &out->ctx,
                                &cl,
                                &out->engine.ord,
                                out->engine.streams.compute) == 0);
  CHECK(FailPhase2,
        stream_engine_bind_array(&out->engine, &out->ar, &out->ctx) == 0);

  computed_stream_layouts_free(&cl);
  return out;

FailPhase2:
  tile_stream_gpu_destroy(out);
FailPhase1b:
  computed_stream_layouts_free(&cl);
FailPhase1:
  return NULL;
}

// --- Accessors ---

const struct tile_stream_layout*
tile_stream_gpu_layout(const struct tile_stream_gpu* s)
{
  return &s->ctx.layout;
}

struct writer*
tile_stream_gpu_writer(struct tile_stream_gpu* s)
{
  return &s->writer;
}

uint64_t
tile_stream_gpu_cursor(const struct tile_stream_gpu* s)
{
  return s->ctx.cursor_elements;
}

struct tile_stream_status
tile_stream_gpu_status(const struct tile_stream_gpu* s)
{
  return (struct tile_stream_status){
    .nlod = s->ctx.levels.nlod,
    .append_downsample = s->ctx.dims.append_downsample,
    .epochs_per_batch = s->engine.batch.epochs_per_batch,
    .max_compressed_size = s->engine.compress_agg.codec.max_output_size,
    .dtype = s->ctx.config.dtype,
    .codec = s->ctx.config.codec,
    .codec_batch_size = s->engine.compress_agg.codec.batch_size,
    .batch_accumulated = s->engine.batch.accumulated,
    .pool_current = s->engine.pools.current,
    .flush_pending = s->engine.flush.pending[0] || s->engine.flush.pending[1],
  };
}

// --- Memory estimate ---

int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                size_t shard_alignment,
                                struct tile_stream_memory_info* info)
{
  if (!info)
    return 1;

  memset(info, 0, sizeof(*info));

  struct computed_stream_layouts cl;
  if (compute_stream_layouts(config,
                             codec_alignment(config->codec.id),
                             codec_max_output_size,
                             shard_alignment,
                             &cl))
    return 1;

  const uint8_t rank = config->rank;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  const size_t buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;
  const uint64_t chunk_stride = cl.layouts[0].chunk_stride;
  const uint64_t chunks_per_epoch = cl.layouts[0].chunks_per_epoch;
  const uint64_t total_chunks = cl.levels.total_chunks;
  const uint32_t K = cl.epochs_per_batch;
  const int nlod = cl.levels.nlod;
  const size_t max_output_size = cl.max_output_size;

  const size_t chunk_bytes = chunk_stride * bytes_per_element;
  const uint64_t codec_batch = (uint64_t)K * total_chunks;
  const size_t nvcomp_temp =
    codec_temp_bytes(config->codec.id, chunk_bytes, codec_batch);

  const size_t staging_bytes = 2 * buffer_capacity_bytes;
  const size_t staging_host = 2 * buffer_capacity_bytes;

  const size_t chunk_pool_bytes =
    2 * (uint64_t)K * total_chunks * chunk_stride * bytes_per_element;

  // CODEC_NONE skips the d_compressed[fc] allocation (aggregate reads pool
  // directly); see compress_agg_init / compress_agg_kick.
  const size_t compressed_pool_bytes =
    (config->codec.id == CODEC_NONE)
      ? 0
      : 2 * (uint64_t)K * total_chunks * max_output_size;

  size_t codec_bytes = 0;
  codec_bytes += codec_batch * sizeof(size_t); // d_comp_sizes
  codec_bytes += codec_batch * sizeof(size_t); // d_uncomp_sizes
  if (config->codec.id != CODEC_NONE)
    codec_bytes += 2 * codec_batch * sizeof(void*); // d_ptrs
  codec_bytes += nvcomp_temp;

  size_t aggregate_device = 0;
  size_t aggregate_host = 0;

  for (int lv = 0; lv < nlod; ++lv) {
    const struct level_layout_info* li = &cl.per_level[lv];
    uint64_t covering_count = li->agg_layout.covering_count;
    uint64_t cps_inner_lv = li->agg_layout.cps_inner;

    uint32_t batch_count = li->batch_active_count;
    if (batch_count == 0)
      batch_count = 1;

    uint64_t batch_chunks =
      (uint64_t)batch_count * cl.levels.level[lv].chunk_count;
    uint64_t batch_covering = (uint64_t)batch_count * covering_count;
    size_t batch_agg_bytes = agg_pool_bytes(batch_chunks,
                                            max_output_size,
                                            covering_count,
                                            cps_inner_lv,
                                            shard_alignment);

    size_t agg_layout_dev =
      2 * (rank - 1) * sizeof(uint64_t) + 2 * (rank - 1) * sizeof(int64_t);

    size_t cub_temp = 0;
    aggregate_cub_temp_bytes(batch_covering, &cub_temp);

    size_t slot_dev = (batch_covering + 1) * sizeof(size_t) +
                      (batch_covering + 1) * sizeof(size_t) +
                      batch_chunks * sizeof(uint32_t) + batch_agg_bytes +
                      cub_temp;

    size_t slot_host = batch_agg_bytes + (batch_covering + 1) * sizeof(size_t) +
                       batch_covering * sizeof(size_t);

    size_t lut_dev = 0;
    if (K > 1)
      lut_dev = batch_chunks * sizeof(uint32_t) * 2;

    aggregate_device += agg_layout_dev + 2 * slot_dev + lut_dev;
    aggregate_host += 2 * slot_host;
  }

  // Shard state: active_shard arrays + index buffers (host heap).
  size_t shard_heap = 0;
  for (int lv = 0; lv < nlod; ++lv) {
    const struct level_layout_info* li = &cl.per_level[lv];
    shard_heap += li->shard_inner_count *
                  (sizeof(struct active_shard) +
                   2 * li->chunks_per_shard_total * sizeof(uint64_t));
  }

  size_t lod_device = 0;

  lod_device += 2 * rank * sizeof(uint64_t);
  lod_device += 2 * rank * sizeof(int64_t);

  if (cl.levels.enable_multiscale) {
    const struct lod_plan* plan = &cl.plan;

    lod_device += cl.layouts[0].epoch_elements * bytes_per_element;
    uint64_t total_lod_vals = plan->level_spans.ends[plan->levels.nlod - 1];
    lod_device += total_lod_vals * bytes_per_element;

    lod_device += rank * sizeof(uint64_t);
    if (plan->lod_ndim > 0)
      lod_device += plan->lod_ndim * sizeof(uint64_t);

    if (plan->lod_ndim > 0) {
      lod_device += plan->levels.level[0].lod_nelem * sizeof(uint32_t);
      lod_device += plan->fixed_dims_count * sizeof(uint32_t);
    }

    for (int l = 0; l < plan->levels.nlod - 1; ++l) {
      // CSR reduce LUTs (computed from level_dims; actual alloc happens later
      // via reduce_csr_gpu_alloc).
      const struct level_dims* src_ld = &plan->levels.level[l];
      const struct level_dims* dst_ld = &plan->levels.level[l + 1];
      uint64_t src_total = src_ld->fixed_dims_count * src_ld->lod_nelem;
      uint64_t dst_total = dst_ld->fixed_dims_count * dst_ld->lod_nelem;
      if (src_total == 0 || dst_total == 0)
        continue;
      lod_device += (dst_total + 1) * sizeof(uint64_t);
      lod_device += src_total * sizeof(uint64_t);
    }

    for (int lv = 1; lv < plan->levels.nlod; ++lv) {
      // Per-level morton scatter LUTs (level 0 already counted above)
      lod_device += plan->levels.level[lv].lod_nelem * sizeof(uint32_t);
      lod_device += plan->levels.level[lv].fixed_dims_count * sizeof(uint32_t);
    }

    for (int lv = 1; lv < plan->levels.nlod; ++lv) {
      lod_device += 2 * rank * sizeof(uint64_t);
      lod_device += 2 * rank * sizeof(int64_t);
    }

    if (cl.dims.append_downsample) {
      size_t accum_bpe = dtype_bpe(config->dtype);
      uint64_t total_elems = 0;
      for (int lv = 1; lv < plan->levels.nlod; ++lv)
        total_elems += plan->levels.level[lv].fixed_dims_count *
                       plan->levels.level[lv].lod_nelem;
      lod_device += total_elems * accum_bpe;
      lod_device += total_elems;
      lod_device += (uint64_t)plan->levels.nlod * sizeof(uint32_t);
    }
  }

  // FIXME: use designated initializers here
  info->staging_bytes = staging_bytes;
  info->chunk_pool_bytes = chunk_pool_bytes;
  info->compressed_pool_bytes = compressed_pool_bytes;
  info->aggregate_bytes = aggregate_device;
  info->lod_bytes = lod_device;
  info->codec_bytes = codec_bytes;
  info->shard_bytes = shard_heap;

  info->device_bytes = staging_bytes + chunk_pool_bytes +
                       compressed_pool_bytes + aggregate_device + lod_device +
                       codec_bytes;
  info->host_pinned_bytes = staging_host + aggregate_host;

  info->chunks_per_epoch = chunks_per_epoch;
  info->total_chunks = total_chunks;
  info->max_output_size = max_output_size;
  info->nlod = nlod;
  info->epochs_per_batch = K;

  computed_stream_layouts_free(&cl);
  return 0;
}

int
tile_stream_gpu_advise_layout(struct tile_stream_configuration* config,
                              size_t target_chunk_bytes,
                              size_t min_chunk_bytes,
                              const int* ratios,
                              size_t budget_bytes,
                              size_t min_shard_bytes,
                              uint32_t target_concurrent_shards,
                              uint32_t min_append_shards,
                              size_t shard_alignment,
                              struct advise_layout_diagnostic* diag)
{
  if (diag) {
    memset(diag, 0, sizeof(*diag));
    diag->budget_bytes = budget_bytes;
    diag->parts_limit = MAX_PARTS_PER_SHARD;
  }

  const size_t bytes_per_element = dtype_bpe(config->dtype);
  if (bytes_per_element == 0 || budget_bytes == 0) {
    if (diag)
      diag->reason = ADVISE_INVALID_CONFIG;
    return 1;
  }

  const uint32_t user_k = config->epochs_per_batch;
  const size_t floor =
    min_chunk_bytes > bytes_per_element ? min_chunk_bytes : bytes_per_element;
  if (diag)
    diag->floor_chunk_bytes = floor;

  // Track last-iteration context so the diagnostic can describe the reason the
  // solver bailed after exhausting all chunk sizes.
  enum advise_layout_reason last_reason = ADVISE_BUDGET_EXCEEDED;
  size_t last_chunk_bytes = 0;
  uint32_t last_k = 0;
  size_t last_device_bytes = 0;
  uint64_t last_cps_total = 0;

  for (size_t target = target_chunk_bytes; target >= floor; target >>= 1) {
    // Phase 1: fit chunks + K to memory budget. Start with auto-derived K
    // (or user-supplied K if non-zero); if device_bytes exceeds budget, halve
    // K and retry. User-supplied K is authoritative and isn't reduced.
    if (dims_budget_chunk_bytes(config->dimensions,
                                config->rank,
                                target,
                                bytes_per_element,
                                ratios)) {
      // Non-recoverable input (e.g. pinned dims exceed target at this step).
      // Halving target can only make it worse — bail.
      last_reason = ADVISE_CHUNK_BUDGET_INFEASIBLE;
      last_chunk_bytes = target;
      break;
    }

    uint64_t chunk_vol = 1;
    for (uint8_t d = 0; d < config->rank; ++d)
      chunk_vol *= config->dimensions[d].chunk_size;
    last_chunk_bytes = (size_t)(chunk_vol * bytes_per_element);

    config->epochs_per_batch = user_k;
    int fit = 0;
    for (;;) {
      struct tile_stream_memory_info mem;
      if (tile_stream_gpu_memory_estimate(config, shard_alignment, &mem)) {
        if (diag) {
          diag->reason = ADVISE_INVALID_CONFIG;
          diag->chunk_bytes = last_chunk_bytes;
          diag->epochs_per_batch = config->epochs_per_batch;
        }
        return 1;
      }
      last_k = mem.epochs_per_batch;
      last_device_bytes = mem.device_bytes;
      if (mem.device_bytes <= budget_bytes) {
        config->epochs_per_batch = mem.epochs_per_batch;
        fit = 1;
        break;
      }
      last_reason = ADVISE_BUDGET_EXCEEDED;
      last_cps_total = 0;
      if (user_k || mem.epochs_per_batch <= 1)
        break;
      config->epochs_per_batch = mem.epochs_per_batch / 2;
    }
    if (!fit)
      continue;

    // Phase 2: shard geometry (parts budget + concurrency target + byte floor).
    // Return 1 = MIN_SHARD_TOO_SMALL (retryable by shrinking chunks);
    // return 2 = PARTS_LIMIT infeasible even with inner fully split — halving
    // chunks only grows inner_cps_prod, so bail immediately.
    int sg = dims_set_shard_geometry(config->dimensions,
                                     config->rank,
                                     min_shard_bytes,
                                     target_concurrent_shards,
                                     min_append_shards,
                                     bytes_per_element);
    if (sg == 1) {
      last_reason = ADVISE_MIN_SHARD_TOO_SMALL;
      last_cps_total = 0;
      continue;
    }
    if (sg == 2) {
      last_reason = ADVISE_PARTS_LIMIT_EXCEEDED;
      uint64_t cps_total = 1;
      for (uint8_t d = 0; d < config->rank; ++d) {
        uint64_t cps = config->dimensions[d].chunks_per_shard;
        if (cps == 0)
          cps = 1;
        cps_total *= cps;
      }
      last_cps_total = cps_total;
      break;
    }
    if (sg != 0) {
      last_reason = ADVISE_INVALID_CONFIG;
      break;
    }

    if (diag) {
      uint64_t cps_total = 1;
      for (uint8_t d = 0; d < config->rank; ++d) {
        uint64_t cps = config->dimensions[d].chunks_per_shard;
        if (cps == 0)
          cps = 1;
        cps_total *= cps;
      }
      uint8_t na = dims_n_append(config->dimensions, config->rank);
      uint64_t inner_shards_prod = 1;
      for (uint8_t d = na; d < config->rank; ++d) {
        uint64_t size = config->dimensions[d].size;
        uint64_t cs = config->dimensions[d].chunk_size;
        uint64_t cps = config->dimensions[d].chunks_per_shard;
        if (cs == 0 || cps == 0)
          continue;
        uint64_t n_chunks = (size + cs - 1) / cs;
        inner_shards_prod *= (n_chunks + cps - 1) / cps;
      }
      diag->reason = ADVISE_OK;
      diag->chunk_bytes = last_chunk_bytes;
      diag->epochs_per_batch = config->epochs_per_batch;
      diag->device_bytes = last_device_bytes;
      diag->chunks_per_shard_total = cps_total;
      diag->actual_concurrent_shards = inner_shards_prod;
      diag->actual_shard_bytes = last_chunk_bytes * cps_total;
      diag->min_append_shards_overrode_min_shard_bytes =
        (min_append_shards > 1 && min_shard_bytes > 0) ? 1 : 0;
    }
    return 0;
  }

  config->epochs_per_batch = user_k;
  if (diag) {
    diag->reason = last_reason;
    diag->chunk_bytes = last_chunk_bytes;
    diag->epochs_per_batch = last_k;
    diag->device_bytes = last_device_bytes;
    diag->chunks_per_shard_total = last_cps_total;
  }
  return 1;
}

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

static void
destroy_cuda_streams_and_events(struct gpu_streams* streams,
                                struct pool_state* pools)
{
  cu_stream_destroy(streams->h2d);
  cu_stream_destroy(streams->compute);
  cu_stream_destroy(streams->compress);
  cu_stream_destroy(streams->d2h);

  for (int i = 0; i < 2; ++i)
    cu_event_destroy(pools->ready[i]);
}

static void
destroy_chunk_pools(struct pool_state* pools)
{
  for (int i = 0; i < 2; ++i)
    cu_mem_free(pools->buf[i]);
}

static void
destroy_batch_events(struct batch_state* batch)
{
  cu_event_destroy(batch->pool_ready);
}

static void
destroy_flush_pipeline(struct flush_pipeline* fp);

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

  // Ensure all GPU work completes before tearing down events/memory.
  sync(s->engine.streams.h2d);
  sync(s->engine.streams.compute);
  sync(s->engine.streams.compress);
  sync(s->engine.streams.d2h);

  destroy_batch_events(&s->engine.batch);
  destroy_flush_pipeline(&s->engine.flush);
  d2h_deliver_destroy(&s->engine.d2h_deliver);
  compress_agg_destroy(&s->engine.compress_agg, s->ctx.levels.nlod);
  destroy_chunk_pools(&s->engine.pools);
  lod_state_destroy(&s->engine.lod);
  lod_shared_state_destroy(&s->engine.lod_shared);
  ingest_destroy(&s->engine.stage);
  destroy_cuda_streams_and_events(&s->engine.streams, &s->engine.pools);
  free(s);
}

// --- Create ---

static int
init_cuda_streams_and_events(struct gpu_streams* streams,
                             struct pool_state* pools)
{
  CU(Fail, cuStreamCreate(&streams->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&pools->ready[i], CU_EVENT_DEFAULT));
  }

  return 0;
Fail:
  return 1;
}

static int
init_chunk_pools(struct pool_state* pools,
                 const struct level_geometry* levels,
                 uint64_t chunk_stride,
                 size_t bytes_per_element,
                 uint32_t epochs_per_batch,
                 CUstream compute)
{
  const size_t pool_bytes = (uint64_t)epochs_per_batch * levels->total_chunks *
                            chunk_stride * bytes_per_element;

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&pools->buf[i], pool_bytes));
    CU(Fail, cuMemsetD8Async(pools->buf[i], 0, pool_bytes, compute));
  }

  return 0;
Fail:
  return 1;
}

// Create and seed the batch pool-ready event.
static int
init_batch_events(struct batch_state* batch, CUstream compute)
{
  CU(Fail, cuEventCreate(&batch->pool_ready, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(batch->pool_ready, compute));
  return 0;
Fail:
  return 1;
}

// Allocate per-slot batch_active_masks for the flush pipeline. K entries each.
static int
init_flush_pipeline(struct flush_pipeline* fp, uint32_t K)
{
  for (int fc = 0; fc < 2; ++fc) {
    fp->slot[fc].batch_active_masks = (uint32_t*)calloc(K, sizeof(uint32_t));
    if (!fp->slot[fc].batch_active_masks)
      return 1;
  }
  return 0;
}

static int
init_output_ledger(struct flush_pipeline* fp, const struct aggregate_slot* slot)
{
  struct output_slot_capacity capacity = {
    .data_bytes = slot->slot_capacity_bytes,
    .desc_entries = slot->slot_desc_capacity,
    .batch_records = slot->batches_per_slot_cap,
  };
  return output_slot_ledger_init(&fp->output, capacity) == OUTPUT_LEDGER_OK ? 0
                                                                            : 1;
}

static void
destroy_flush_pipeline(struct flush_pipeline* fp)
{
  for (int fc = 0; fc < 2; ++fc) {
    free(fp->slot[fc].batch_active_masks);
    fp->slot[fc].batch_active_masks = NULL;
  }
}

static int
seed_events(const struct pool_state* pools, CUstream compute)
{
  CU(Fail, cuEventRecord(pools->ready[0], compute));
  CU(Fail, cuEventRecord(pools->ready[1], compute));
  return 0;
Fail:
  return 1;
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

  // Phase 2: Allocate and initialize tile_stream_gpu.
  struct tile_stream_gpu* out =
    (struct tile_stream_gpu*)calloc(1, sizeof(*out));
  CHECK(FailPhase1b, out);

  out->ctx.config = *config;
  out->ctx.sink = sink;
  out->ctx.shard_alignment = shard_sink_required_shard_alignment(sink);
  out->ctx.levels = cl.levels;
  out->ctx.dims = cl.dims;
  tile_stream_gpu_init_writer(out);

  out->ctx.config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // Copy L0 layout (host fields; d_* still NULL).
  out->ctx.layout = cl.layouts[0];

  // Move LOD plan and level layouts (always, including L0).
  out->engine.lod.plan = cl.plan;
  cl.plan = (struct lod_plan){ 0 }; // ownership transferred
  for (int lv = 0; lv < cl.levels.nlod; ++lv)
    out->engine.lod.layouts[lv] = cl.layouts[lv];

  // Copy batch info.
  CHECK(FailPhase2, cl.epochs_per_batch >= 1);
  out->engine.batch.epochs_per_batch = cl.epochs_per_batch;
  out->engine.batch.accumulated = 0;

  // Allocate per-slot batch_active_masks sized to K.
  CHECK(FailPhase2,
        init_flush_pipeline(&out->engine.flush, cl.epochs_per_batch) == 0);

  // GPU allocation and init.
  CHECK(FailPhase2,
        init_cuda_streams_and_events(&out->engine.streams,
                                     &out->engine.pools) == 0);
  CHECK(FailPhase2,
        ingest_init(&out->engine.stage,
                    out->ctx.config.buffer_capacity_bytes,
                    out->engine.streams.compute) == 0);
  CHECK(FailPhase2,
        lod_state_init(&out->engine.lod, &out->ctx.levels, &out->ctx.config) ==
          0);
  // Alias L0 layout GPU pointers from LOD state into context.
  out->ctx.layout_gpu = out->engine.lod.layout_gpu[0];
  CHECK(FailPhase2,
        init_chunk_pools(&out->engine.pools,
                         &out->ctx.levels,
                         out->ctx.layout.chunk_stride,
                         dtype_bpe(config->dtype),
                         out->engine.batch.epochs_per_batch,
                         out->engine.streams.compute) == 0);

  // Initialize the two pipeline stages.
  CHECK(FailPhase2,
        compress_agg_init(&out->engine.compress_agg,
                          &cl,
                          config,
                          out->engine.streams.compute) == 0);
  CHECK(FailPhase2,
        init_output_ledger(&out->engine.flush,
                           &out->engine.compress_agg.output[0]) == 0);
  CHECK(FailPhase2,
        d2h_deliver_init(&out->engine.d2h_deliver,
                         out->ctx.shard_alignment,
                         out->engine.streams.compute) == 0);

  CHECK(FailPhase2,
        init_batch_events(&out->engine.batch, out->engine.streams.compute) ==
          0);
  if (out->ctx.levels.enable_multiscale) {
    const size_t bpe = dtype_bpe(out->ctx.config.dtype);
    const size_t linear_bytes = out->engine.lod.layouts[0].epoch_elements * bpe;
    const uint64_t morton_total_vals =
      out->engine.lod.plan.level_spans
        .ends[out->engine.lod.plan.levels.nlod - 1];
    const size_t morton_bytes = morton_total_vals * bpe;
    CHECK(FailPhase2,
          lod_shared_state_init(&out->engine.lod_shared,
                                linear_bytes,
                                morton_bytes,
                                out->engine.streams.compute) == 0);
    if (out->ctx.dims.append_downsample)
      CHECK(FailPhase2,
            lod_state_init_accumulators(&out->engine.lod, &out->ctx.config) ==
              0);
  }
  CHECK(FailPhase2,
        seed_events(&out->engine.pools, out->engine.streams.compute) == 0);

  CU(FailPhase2, cuStreamSynchronize(out->engine.streams.compute));

  // Precompute total_element_limit (configured stream length) so the body can
  // detect the at-capacity case without recomputing each call.
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&out->ctx.dims);
    if (dims[0].size > 0) {
      out->ctx.total_element_limit = out->ctx.layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        out->ctx.total_element_limit *=
          ceildiv(dims[d].size, dims[d].chunk_size);
    }
  }

  out->engine.metrics =
    stream_engine_init_metrics(out->ctx.levels.enable_multiscale);
  out->engine.d2h_deliver.metrics = &out->engine.metrics;

  // Initialize metadata update timer
  out->engine.metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&out->engine.metadata_update_clock);

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
    // True if any output slot owns data that has not been fully delivered.
    .flush_pending = output_slot_ledger_has_work(&s->engine.flush.output),
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

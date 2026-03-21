#include "flush.compress_agg.h"
#include "flush.d2h_deliver.h"
#include "stream.ingest.h"
#include "stream.lod.h"

#include "lod.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.config.h"

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
destroy_l0_layout_gpu(struct tile_stream_layout_gpu* gpu)
{
  cu_mem_free((CUdeviceptr)gpu->d_lifted_shape);
  cu_mem_free((CUdeviceptr)gpu->d_lifted_strides);
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
  for (uint32_t i = 0; i < batch->epochs_per_batch; ++i)
    cu_event_destroy(batch->pool_events[i]);
}

void
tile_stream_gpu_destroy(struct tile_stream_gpu* s)
{
  if (!s)
    return;

  // Ensure all GPU work completes before tearing down events/memory.
  if (s->streams.h2d)
    cuStreamSynchronize(s->streams.h2d);
  if (s->streams.compute)
    cuStreamSynchronize(s->streams.compute);
  if (s->streams.compress)
    cuStreamSynchronize(s->streams.compress);
  if (s->streams.d2h)
    cuStreamSynchronize(s->streams.d2h);

  destroy_batch_events(&s->batch);
  d2h_deliver_destroy(&s->d2h_deliver);
  compress_agg_destroy(&s->compress_agg, s->levels.nlod);
  destroy_chunk_pools(&s->pools);
  lod_state_destroy(&s->lod);
  ingest_destroy(&s->stage);
  destroy_l0_layout_gpu(&s->layout_gpu);
  destroy_cuda_streams_and_events(&s->streams, &s->pools);
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

// Upload layout arrays to GPU. Returns 0 on success.
static int
upload_stream_layout(struct tile_stream_layout_gpu* gpu,
                     const struct tile_stream_layout* layout)
{
  const size_t shape_bytes = layout->lifted_rank * sizeof(uint64_t);
  const size_t strides_bytes = layout->lifted_rank * sizeof(int64_t);
  CU(Fail, cuMemAlloc((CUdeviceptr*)&gpu->d_lifted_shape, shape_bytes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&gpu->d_lifted_strides, strides_bytes));
  CU(Fail,
     cuMemcpyHtoD(
       (CUdeviceptr)gpu->d_lifted_shape, layout->lifted_shape, shape_bytes));
  CU(Fail,
     cuMemcpyHtoD((CUdeviceptr)gpu->d_lifted_strides,
                  layout->lifted_strides,
                  strides_bytes));
  return 0;
Fail:
  return 1;
}

static int
init_chunk_pools(struct pool_state* pools,
                 const struct level_geometry* levels,
                 uint64_t chunk_stride,
                 size_t bpe,
                 uint32_t epochs_per_batch,
                 CUstream compute)
{
  const size_t pool_bytes =
    (uint64_t)epochs_per_batch * levels->total_chunks * chunk_stride * bpe;

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&pools->buf[i], pool_bytes));
    CU(Fail, cuMemsetD8Async(pools->buf[i], 0, pool_bytes, compute));
  }

  return 0;
Fail:
  return 1;
}

// Allocate and seed per-epoch batch pool events.
static int
init_batch_events(struct batch_state* batch, CUstream compute)
{
  const uint32_t K = batch->epochs_per_batch;
  for (uint32_t i = 0; i < K; ++i) {
    CU(Fail, cuEventCreate(&batch->pool_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(batch->pool_events[i], compute));
  }
  return 0;
Fail:
  return 1;
}

static int
seed_events(const struct pool_state* pools,
            const struct lod_state* lod,
            CUstream compute)
{
  CU(Fail, cuEventRecord(pools->ready[0], compute));
  CU(Fail, cuEventRecord(pools->ready[1], compute));

  if (lod->t_start) {
    CU(Fail, cuEventRecord(lod->t_start, compute));
    CU(Fail, cuEventRecord(lod->t_scatter_end, compute));
    CU(Fail, cuEventRecord(lod->t_reduce_end, compute));
    CU(Fail, cuEventRecord(lod->t_dim0_end, compute));
    CU(Fail, cuEventRecord(lod->t_end, compute));
  }

  return 0;
Fail:
  return 1;
}

static struct stream_metrics
init_metrics(int enable_multiscale)
{
  return (struct stream_metrics){
    .memcpy = mk_stream_metric("Memcpy"),
    .h2d = mk_stream_metric("H2D"),
    .scatter = mk_stream_metric(enable_multiscale ? "Copy" : "Scatter"),
    .lod_gather = mk_stream_metric("LOD Gather"),
    .lod_reduce = mk_stream_metric("LOD Reduce"),
    .lod_dim0_fold = mk_stream_metric("Dim0 Fold"),
    .lod_morton_chunk = mk_stream_metric("LOD to chunks"),
    .compress = mk_stream_metric("Compress"),
    .aggregate = mk_stream_metric("Aggregate"),
    .d2h = mk_stream_metric("D2H"),
    .sink = mk_stream_metric("Sink"),
  };
}

struct tile_stream_gpu*
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct shard_sink* sink)
{
  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));

  CHECK(FailPhase1, config && sink);

  // Phase 1: CPU-only layout computation.
  CHECK(FailPhase1,
        compute_stream_layouts(config,
                               codec_alignment(config->codec),
                               codec_max_output_size,
                               &cl) == 0);

  // Phase 2: Allocate and initialize tile_stream_gpu.
  struct tile_stream_gpu* out =
    (struct tile_stream_gpu*)calloc(1, sizeof(*out));
  CHECK(FailPhase1b, out);

  out->config = *config;
  out->shard_sink = sink;
  out->levels = cl.levels;
  tile_stream_gpu_init_writer(out);

  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // Copy L0 layout (host fields; d_* still NULL).
  out->layout = cl.l0;

  // Move LOD plan and level layouts.
  if (cl.levels.enable_multiscale) {
    out->lod.plan = cl.plan;
    cl.plan = (struct lod_plan){ 0 }; // ownership transferred
    for (int lv = 1; lv < cl.levels.nlod; ++lv)
      out->lod.layouts[lv] = cl.lod_layouts[lv];
  }

  // Copy batch info.
  CHECK(FailPhase2, (cl.epochs_per_batch & (cl.epochs_per_batch - 1)) == 0);
  out->batch.epochs_per_batch = cl.epochs_per_batch;
  out->batch.accumulated = 0;

  // GPU allocation and init.
  CHECK(FailPhase2,
        init_cuda_streams_and_events(&out->streams, &out->pools) == 0);
  CHECK(FailPhase2, upload_stream_layout(&out->layout_gpu, &out->layout) == 0);
  CHECK(FailPhase2,
        ingest_init(&out->stage,
                    out->config.buffer_capacity_bytes,
                    out->streams.compute) == 0);
  CHECK(FailPhase2,
        lod_state_init(&out->lod, &out->levels, &out->layout, &out->config) ==
          0);
  CHECK(FailPhase2,
        init_chunk_pools(&out->pools,
                         &out->levels,
                         out->layout.chunk_stride,
                         dtype_bpe(config->dtype),
                         out->batch.epochs_per_batch,
                         out->streams.compute) == 0);

  // Initialize the two pipeline stages.
  CHECK(FailPhase2,
        compress_agg_init(
          &out->compress_agg, &cl, config, out->streams.compute) == 0);
  CHECK(FailPhase2,
        d2h_deliver_init(&out->d2h_deliver,
                         out->compress_agg.levels,
                         out->levels.nlod,
                         out->streams.compute) == 0);

  CHECK(FailPhase2, init_batch_events(&out->batch, out->streams.compute) == 0);
  if (out->levels.enable_multiscale) {
    CHECK(FailPhase2,
          lod_state_init_buffers(&out->lod, &out->layout, out->config.dtype) ==
            0);
    if (out->levels.dim0_downsample)
      CHECK(FailPhase2,
            lod_state_init_accumulators(&out->lod, &out->config) == 0);
  }
  CHECK(FailPhase2,
        seed_events(&out->pools, &out->lod, out->streams.compute) == 0);

  CU(FailPhase2, cuStreamSynchronize(out->streams.compute));

  out->metrics = init_metrics(out->levels.enable_multiscale);

  // Initialize metadata update timer
  out->metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&out->metadata_update_clock);

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
  return &s->layout;
}

struct writer*
tile_stream_gpu_writer(struct tile_stream_gpu* s)
{
  return &s->writer;
}

uint64_t
tile_stream_gpu_cursor(const struct tile_stream_gpu* s)
{
  return s->cursor;
}

struct tile_stream_status
tile_stream_gpu_status(const struct tile_stream_gpu* s)
{
  return (struct tile_stream_status){
    .nlod = s->levels.nlod,
    .dim0_downsample = s->levels.dim0_downsample,
    .epochs_per_batch = s->batch.epochs_per_batch,
    .max_compressed_size = s->compress_agg.codec.max_output_size,
    .dtype = s->config.dtype,
    .codec = s->config.codec,
    .codec_batch_size = s->compress_agg.codec.batch_size,
    .batch_accumulated = s->batch.accumulated,
    .pool_current = s->pools.current,
    .flush_pending = s->flush.pending,
  };
}

// --- Memory estimate ---

int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_memory_info* info)
{
  if (!info)
    return 1;

  memset(info, 0, sizeof(*info));

  struct computed_stream_layouts cl;
  if (compute_stream_layouts(
        config, codec_alignment(config->codec), codec_max_output_size, &cl))
    return 1;

  const uint8_t rank = config->rank;
  const size_t bpe = dtype_bpe(config->dtype);
  const size_t buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;
  const uint64_t chunk_stride = cl.l0.chunk_stride;
  const uint64_t chunks_per_epoch = cl.l0.chunks_per_epoch;
  const uint64_t total_chunks = cl.levels.total_chunks;
  const uint32_t K = cl.epochs_per_batch;
  const int nlod = cl.levels.nlod;
  const size_t max_output_size = cl.max_output_size;

  const size_t chunk_bytes = chunk_stride * bpe;
  const uint64_t codec_batch = (uint64_t)K * total_chunks;
  const size_t nvcomp_temp =
    codec_temp_bytes(config->codec, chunk_bytes, codec_batch);

  const size_t staging_bytes = 2 * buffer_capacity_bytes;
  const size_t staging_host = 2 * buffer_capacity_bytes;

  const size_t chunk_pool_bytes =
    2 * (uint64_t)K * total_chunks * chunk_stride * bpe;

  const size_t compressed_pool_bytes =
    2 * (uint64_t)K * total_chunks * max_output_size;

  size_t codec_bytes = 0;
  codec_bytes += codec_batch * sizeof(size_t); // d_comp_sizes
  codec_bytes += codec_batch * sizeof(size_t); // d_uncomp_sizes
  if (config->codec != CODEC_NONE)
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

    uint64_t batch_chunks = (uint64_t)batch_count * cl.levels.chunk_count[lv];
    uint64_t batch_covering = (uint64_t)batch_count * covering_count;
    size_t batch_agg_bytes = agg_pool_bytes(batch_chunks,
                                            max_output_size,
                                            covering_count,
                                            cps_inner_lv,
                                            config->shard_alignment);

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

    lod_device += cl.l0.epoch_elements * bpe;
    uint64_t total_lod_vals = plan->levels.ends[plan->nlod - 1];
    lod_device += total_lod_vals * bpe;

    lod_device += rank * sizeof(uint64_t);
    if (plan->lod_ndim > 0)
      lod_device += plan->lod_ndim * sizeof(uint64_t);

    if (plan->lod_ndim > 0) {
      lod_device += plan->lod_nelem[0] * sizeof(uint32_t);
      lod_device += plan->batch_count * sizeof(uint32_t);
    }

    for (int l = 0; l < plan->nlod - 1; ++l) {
      lod_device += plan->lod_ndim * sizeof(uint64_t);
      lod_device += plan->lod_ndim * sizeof(uint64_t);

      struct lod_span seg = lod_segment(plan, l);
      uint64_t n_parents = lod_span_len(seg);
      lod_device += n_parents * sizeof(uint64_t);
    }

    for (int lv = 1; lv < plan->nlod; ++lv) {
      lod_device += 2 * rank * sizeof(uint64_t);
      lod_device += 2 * rank * sizeof(int64_t);
    }

    if (cl.levels.dim0_downsample) {
      size_t accum_bpe =
        dtype_accum_bpe(config->dtype, config->dim0_reduce_method);
      uint64_t total_elems = 0;
      for (int lv = 1; lv < plan->nlod; ++lv)
        total_elems += plan->batch_count * plan->lod_nelem[lv];
      lod_device += total_elems * accum_bpe;
      lod_device += total_elems;
      lod_device += (uint64_t)plan->nlod * sizeof(uint32_t);
    }
  }

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
tile_stream_gpu_advise_chunk_sizes(
    struct tile_stream_configuration* config,
    size_t target_chunk_bytes,
    const uint8_t* ratios,
    size_t budget_bytes)
{
  const size_t bpe = dtype_bpe(config->dtype);
  if (bpe == 0 || budget_bytes == 0)
    return 1;

  for (size_t target = target_chunk_bytes; target >= bpe; target >>= 1) {
    dims_budget_chunk_bytes(
      config->dimensions, config->rank, target, bpe, ratios);
    struct tile_stream_memory_info mem;
    if (tile_stream_gpu_memory_estimate(config, &mem))
      return 1;
    if (mem.device_bytes <= budget_bytes)
      return 0;
  }
  return 1;
}

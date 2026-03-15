#include "flush_compress_agg.h"
#include "flush_d2h_deliver.h"
#include "stream_ingest.h"
#include "stream_lod.h"

#include "index.ops.h"
#include "lod.h"
#include "metric.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>

static inline uint32_t
next_pow2_u32(uint32_t v)
{
  if (v == 0)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

// Compute epochs_per_batch (K) from config and total tiles per epoch.
// Returns K as a power of 2, clamped to MAX_BATCH_EPOCHS.
static uint32_t
compute_epochs_per_batch(const struct tile_stream_configuration* config,
                         uint64_t total_tiles_per_epoch)
{
  uint32_t K = config->epochs_per_batch;
  if (K == 0) {
    uint32_t target = config->target_batch_tiles;
    if (target == 0)
      target = 1024;
    K = (uint32_t)ceildiv(target, total_tiles_per_epoch);
    K = next_pow2_u32(K);
  }
  if (K > MAX_BATCH_EPOCHS)
    K = MAX_BATCH_EPOCHS;
  return K;
}

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
destroy_l0_layout(struct stream_layout* layout)
{
  cu_mem_free((CUdeviceptr)layout->d_lifted_shape);
  cu_mem_free((CUdeviceptr)layout->d_lifted_strides);
}

static void
destroy_tile_pools(struct pool_state* pools)
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
  destroy_batch_events(&s->batch);
  d2h_deliver_destroy(&s->d2h_deliver);
  compress_agg_destroy(&s->compress_agg, s->levels.nlod);
  destroy_tile_pools(&s->pools);
  lod_state_destroy(&s->lod);
  ingest_destroy(&s->stage);
  destroy_l0_layout(&s->layout);
  destroy_cuda_streams_and_events(&s->streams, &s->pools);
  free(s);
}

// --- Create ---

// Extract forward permutation from dims[d].storage_position.
// forward[j] = acquisition dim d such that dims[d].storage_position == j.
// Returns 0 on success.
static int
resolve_storage_order(uint8_t rank,
                      const struct dimension* dims,
                      uint8_t* forward)
{
  // dims[0].storage_position must be 0
  if (dims[0].storage_position != 0)
    return 1;

  // Invert: forward[storage_position] = acq_dim
  uint32_t seen = 0;
  uint8_t tmp[HALF_MAX_RANK];
  for (int d = 0; d < rank; ++d) {
    uint8_t j = dims[d].storage_position;
    if (j >= rank)
      return 1;
    if (seen & (1u << j))
      return 1;
    seen |= (1u << j);
    tmp[j] = (uint8_t)d;
  }

  // forward[0] must be 0 (dim 0 stays outermost)
  if (tmp[0] != 0)
    return 1;

  if (forward)
    memcpy(forward, tmp, rank);
  return 0;
}

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

// Compute host-side stream_layout fields for a single level (no GPU).
static void
compute_level_layout(struct stream_layout* layout,
                     uint8_t rank,
                     size_t bpe,
                     const struct dimension* dims,
                     const uint64_t* level_shape,
                     enum compression_codec codec,
                     const uint8_t* storage_order)
{
  layout->lifted_rank = 2 * rank;
  layout->tile_elements = 1;

  uint64_t tile_count[HALF_MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] =
      (level_shape[i] == 0) ? 1 : ceildiv(level_shape[i], dims[i].tile_size);
    layout->lifted_shape[2 * i] = tile_count[i];
    layout->lifted_shape[2 * i + 1] = dims[i].tile_size;
    layout->tile_elements *= dims[i].tile_size;
  }

  {
    size_t alignment = codec_alignment(codec);
    size_t tile_bytes = layout->tile_elements * bpe;
    size_t padded_bytes = align_up(tile_bytes, alignment);
    layout->tile_stride = padded_bytes / bpe;
  }

  {
    uint64_t ts[HALF_MAX_RANK];
    for (int i = 0; i < rank; ++i)
      ts[i] = dims[i].tile_size;
    compute_lifted_strides(rank,
                           ts,
                           tile_count,
                           storage_order,
                           (int64_t)layout->tile_stride,
                           layout->lifted_strides);
  }

  layout->tiles_per_epoch = layout->lifted_strides[0] / layout->tile_stride;
  layout->epoch_elements = layout->tiles_per_epoch * layout->tile_elements;
  layout->lifted_strides[0] = 0; // collapse epoch dim
  layout->tile_pool_bytes = layout->tiles_per_epoch * layout->tile_stride * bpe;

  layout->d_lifted_shape = NULL;
  layout->d_lifted_strides = NULL;
}

// Upload pre-computed stream_layout arrays to GPU. Returns 0 on success.
static int
upload_stream_layout(struct stream_layout* layout)
{
  const size_t shape_bytes = layout->lifted_rank * sizeof(uint64_t);
  const size_t strides_bytes = layout->lifted_rank * sizeof(int64_t);
  CU(Fail, cuMemAlloc((CUdeviceptr*)&layout->d_lifted_shape, shape_bytes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&layout->d_lifted_strides, strides_bytes));
  CU(Fail,
     cuMemcpyHtoD(
       (CUdeviceptr)layout->d_lifted_shape, layout->lifted_shape, shape_bytes));
  CU(Fail,
     cuMemcpyHtoD((CUdeviceptr)layout->d_lifted_strides,
                  layout->lifted_strides,
                  strides_bytes));
  return 0;
Fail:
  return 1;
}

static int
init_tile_pools(struct pool_state* pools,
                const struct level_geometry* levels,
                uint64_t tile_stride,
                size_t bpe,
                uint32_t epochs_per_batch,
                CUstream compute)
{
  const size_t pool_bytes =
    (uint64_t)epochs_per_batch * levels->total_tiles * tile_stride * bpe;

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

// Validate a tile_stream_configuration.
// Returns 0 on success, non-zero on invalid config.
static int
validate_config(const struct tile_stream_configuration* config)
{
  CHECK(Fail, config);
  CHECK(Fail, config->bytes_per_element > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= HALF_MAX_RANK);
  CHECK(Fail, config->dimensions);

  for (int d = 0; d < config->rank; ++d) {
    if (config->dimensions[d].tile_size == 0) {
      log_error("dims[%d].tile_size must be > 0", d);
      goto Fail;
    }
  }
  {
    uint64_t tile_elements = 1;
    for (int d = 0; d < config->rank; ++d)
      tile_elements *= config->dimensions[d].tile_size;
    if (tile_elements <= 1)
      log_warn("total tile elements is %llu (tile_size=1 in all dims?)",
               (unsigned long long)tile_elements);
  }

  // Validate unbounded dim0: tiles_per_shard must be specified
  if (config->dimensions[0].size == 0 &&
      config->dimensions[0].tiles_per_shard == 0) {
    log_error("dims[0].size=0 (unbounded) requires tiles_per_shard > 0");
    goto Fail;
  }

  if (resolve_storage_order(config->rank, config->dimensions, NULL)) {
    log_error("invalid storage_order permutation");
    goto Fail;
  }

  {
    int dim0_ds = 0;
    uint32_t lod_mask = 0;
    for (int d = 0; d < config->rank; ++d) {
      if (config->dimensions[d].downsample) {
        lod_mask |= (1u << d);
        if (d == 0) {
          dim0_ds = 1;
          enum lod_reduce_method m = config->dim0_reduce_method;
          if (m != lod_reduce_mean && m != lod_reduce_min &&
              m != lod_reduce_max) {
            log_error("dim0 reduce method must be mean, min, or max");
            goto Fail;
          }
        }
      }
    }
    if (dim0_ds && (lod_mask & ~1u) == 0) {
      log_error(
        "dim0 downsample requires at least one spatial dim downsampled");
      goto Fail;
    }
  }

  return 0;
Fail:
  return 1;
}

void
computed_stream_layouts_free(struct computed_stream_layouts* cl)
{
  if (!cl)
    return;
  if (cl->levels.enable_multiscale)
    lod_plan_free(&cl->plan);
}

int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       struct computed_stream_layouts* out)
{
  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const struct dimension* dims = config->dimensions;

  uint8_t storage_order[HALF_MAX_RANK];
  CHECK(Fail, resolve_storage_order(rank, dims, storage_order) == 0);

  uint32_t lod_mask = 0;
  int dim0_downsample = 0;
  for (int d = 0; d < rank; ++d) {
    if (dims[d].downsample) {
      lod_mask |= (1u << d);
      if (d == 0)
        dim0_downsample = 1;
    }
  }
  int enable_multiscale = (lod_mask & ~1u) != 0;

  memset(out, 0, sizeof(*out));
  out->levels.enable_multiscale = enable_multiscale;
  out->levels.dim0_downsample = dim0_downsample;
  out->levels.nlod = 1;

  // --- L0 layout ---
  {
    uint64_t l0_shape[HALF_MAX_RANK];
    for (int d = 0; d < rank; ++d)
      l0_shape[d] = dims[d].size;
    compute_level_layout(
      &out->l0, rank, bpe, dims, l0_shape, config->codec, storage_order);
  }

  // --- LOD plan ---
  if (enable_multiscale) {
    uint32_t lod_mask = 0;
    for (int d = 0; d < rank; ++d)
      if (dims[d].downsample)
        lod_mask |= (1u << d);

    uint64_t shape[HALF_MAX_RANK];
    uint64_t tile_shape[HALF_MAX_RANK];
    shape[0] = dims[0].tile_size;
    for (int d = 1; d < rank; ++d)
      shape[d] = dims[d].size;
    for (int d = 0; d < rank; ++d)
      tile_shape[d] = dims[d].tile_size;

    CHECK(Fail,
          lod_plan_init(&out->plan,
                        rank,
                        shape,
                        tile_shape,
                        lod_mask,
                        LOD_MAX_LEVELS,
                        dim0_downsample) == 0);

    out->levels.nlod = out->plan.nlod;

    for (int lv = 1; lv < out->plan.nlod; ++lv)
      compute_level_layout(&out->lod_layouts[lv],
                           rank,
                           bpe,
                           dims,
                           out->plan.shapes[lv],
                           config->codec,
                           storage_order);
  }

  // --- Level geometry ---
  out->levels.tile_count[0] = out->l0.tiles_per_epoch;
  out->levels.tile_offset[0] = 0;
  out->levels.total_tiles = out->l0.tiles_per_epoch;

  for (int lv = 1; lv < out->levels.nlod; ++lv) {
    out->levels.tile_count[lv] = out->lod_layouts[lv].tiles_per_epoch;
    out->levels.tile_offset[lv] = out->levels.total_tiles;
    out->levels.total_tiles += out->levels.tile_count[lv];
  }

  // --- Epochs per batch (K) ---
  out->epochs_per_batch =
    compute_epochs_per_batch(config, out->levels.total_tiles);

  // --- Codec-derived max_output_size ---
  {
    const size_t chunk_bytes = out->l0.tile_stride * bpe;
    out->max_output_size = codec_max_output_size(config->codec, chunk_bytes);
    if (config->codec != CODEC_NONE && out->max_output_size == 0)
      goto Fail;
  }

  // --- Per-level aggregate layout and shard geometry ---
  for (int lv = 0; lv < out->levels.nlod; ++lv) {
    uint64_t tile_count[HALF_MAX_RANK];
    uint64_t tiles_per_shard[HALF_MAX_RANK];

    if (lv == 0) {
      for (int d = 0; d < rank; ++d) {
        tile_count[d] =
          (dims[d].size == 0) ? 1 : ceildiv(dims[d].size, dims[d].tile_size);
        uint64_t tps = dims[d].tiles_per_shard;
        tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
      }
    } else {
      const uint64_t* lv_shape = out->plan.shapes[lv];
      for (int d = 0; d < rank; ++d) {
        tile_count[d] = ceildiv(lv_shape[d], dims[d].tile_size);
        uint64_t tps = dims[d].tiles_per_shard;
        tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
      }
    }

    uint64_t tiles_lv = out->levels.tile_count[lv];

    uint32_t batch_count = out->epochs_per_batch;
    if (dim0_downsample && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count =
        (out->epochs_per_batch >= period) ? out->epochs_per_batch / period : 0;
    }
    out->per_level[lv].batch_active_count = batch_count;

    uint64_t so_tile_count[HALF_MAX_RANK], so_tiles_per_shard[HALF_MAX_RANK];
    for (int j = 0; j < rank; ++j) {
      so_tile_count[j] = tile_count[storage_order[j]];
      so_tiles_per_shard[j] = tiles_per_shard[storage_order[j]];
    }

    CHECK(Fail,
          aggregate_layout_compute(&out->per_level[lv].agg_layout,
                                   rank,
                                   so_tile_count,
                                   so_tiles_per_shard,
                                   tiles_lv,
                                   out->max_output_size,
                                   config->shard_alignment) == 0);

    {
      uint64_t tps0 = tiles_per_shard[0];
      if (dim0_downsample && lv > 0) {
        uint64_t divisor = 1ull << lv;
        tps0 = (tps0 > divisor) ? tps0 / divisor : 1;
      }
      out->per_level[lv].tiles_per_shard_0 = tps0;
    }
    out->per_level[lv].tiles_per_shard_inner = 1;
    for (int d = 1; d < rank; ++d)
      out->per_level[lv].tiles_per_shard_inner *= tiles_per_shard[d];
    out->per_level[lv].tiles_per_shard_total =
      out->per_level[lv].tiles_per_shard_0 *
      out->per_level[lv].tiles_per_shard_inner;

    out->per_level[lv].shard_inner_count = 1;
    for (int d = 1; d < rank; ++d)
      out->per_level[lv].shard_inner_count *=
        ceildiv(tile_count[d], tiles_per_shard[d]);
  }

  return 0;
Fail:
  computed_stream_layouts_free(out);
  return 1;
}

struct stream_metric
mk_stream_metric(const char* name)
{
  return (struct stream_metric){ .name = name, .best_ms = 1e30 };
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
    .lod_morton_tile = mk_stream_metric("LOD to tiles"),
    .compress = mk_stream_metric("Compress"),
    .aggregate = mk_stream_metric("Aggregate"),
    .d2h = mk_stream_metric("D2H"),
    .sink = mk_stream_metric("Sink"),
  };
}

struct tile_stream_gpu*
tile_stream_gpu_create(const struct tile_stream_configuration* config)
{
  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));

  CHECK(FailPhase1, config && config->shard_sink);
  CHECK(FailPhase1, validate_config(config) == 0);

  // Phase 1: CPU-only layout computation.
  CHECK(FailPhase1, compute_stream_layouts(config, &cl) == 0);

  // Phase 2: Allocate and initialize tile_stream_gpu.
  struct tile_stream_gpu* out =
    (struct tile_stream_gpu*)calloc(1, sizeof(*out));
  CHECK(FailPhase1b, out);

  out->config = *config;
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
  CHECK(FailPhase2, upload_stream_layout(&out->layout) == 0);
  CHECK(FailPhase2,
        ingest_init(&out->stage,
                    out->config.buffer_capacity_bytes,
                    out->streams.compute) == 0);
  CHECK(FailPhase2,
        lod_state_init(&out->lod, &out->levels, &out->layout, &out->config) ==
          0);
  CHECK(FailPhase2,
        init_tile_pools(&out->pools,
                        &out->levels,
                        out->layout.tile_stride,
                        config->bytes_per_element,
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
          lod_state_init_buffers(
            &out->lod, &out->layout, out->config.bytes_per_element) == 0);
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

const struct stream_layout*
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
    .bytes_per_element = s->config.bytes_per_element,
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
  if (validate_config(config))
    return 1;

  memset(info, 0, sizeof(*info));

  struct computed_stream_layouts cl;
  if (compute_stream_layouts(config, &cl))
    return 1;

  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const size_t buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;
  const uint64_t tile_stride = cl.l0.tile_stride;
  const uint64_t tiles_per_epoch = cl.l0.tiles_per_epoch;
  const uint64_t total_tiles = cl.levels.total_tiles;
  const uint32_t K = cl.epochs_per_batch;
  const int nlod = cl.levels.nlod;
  const size_t max_output_size = cl.max_output_size;

  const size_t chunk_bytes = tile_stride * bpe;
  const uint64_t codec_batch = (uint64_t)K * total_tiles;
  const size_t nvcomp_temp =
    codec_temp_bytes(config->codec, chunk_bytes, codec_batch);

  const size_t staging_bytes = 2 * buffer_capacity_bytes;
  const size_t staging_host = 2 * buffer_capacity_bytes;

  const size_t tile_pool_bytes =
    2 * (uint64_t)K * total_tiles * tile_stride * bpe;

  const size_t compressed_pool_bytes =
    2 * (uint64_t)K * total_tiles * max_output_size;

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
    uint64_t tps_inner_lv = li->agg_layout.tps_inner;

    uint32_t batch_count = li->batch_active_count;
    if (batch_count == 0)
      batch_count = 1;

    uint64_t batch_tiles = (uint64_t)batch_count * cl.levels.tile_count[lv];
    uint64_t batch_covering = (uint64_t)batch_count * covering_count;
    size_t batch_agg_bytes = agg_pool_bytes(batch_tiles,
                                            max_output_size,
                                            covering_count,
                                            tps_inner_lv,
                                            config->shard_alignment);

    size_t agg_layout_dev =
      2 * (rank - 1) * sizeof(uint64_t) + 2 * (rank - 1) * sizeof(int64_t);

    size_t slot_dev = (batch_covering + 1) * sizeof(size_t)
                      + (batch_covering + 1) * sizeof(size_t)
                      + batch_tiles * sizeof(uint32_t)
                      + batch_agg_bytes;

    size_t slot_host = batch_agg_bytes
                       + (batch_covering + 1) * sizeof(size_t)
                       + batch_covering * sizeof(size_t);

    size_t lut_dev = 0;
    if (K > 1)
      lut_dev = batch_tiles * sizeof(uint32_t) * 2;

    aggregate_device += agg_layout_dev + 2 * slot_dev + lut_dev;
    aggregate_host += 2 * slot_host;
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
      lod_device += plan->lod_counts[0] * sizeof(uint32_t);
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
      size_t accum_bpe = lod_accum_bpe(bpe, config->dim0_reduce_method);
      uint64_t total_elems = 0;
      for (int lv = 1; lv < plan->nlod; ++lv)
        total_elems += plan->batch_count * plan->lod_counts[lv];
      lod_device += total_elems * accum_bpe;
      lod_device += total_elems;
      lod_device += (uint64_t)plan->nlod * sizeof(uint32_t);
    }
  }

  info->staging_bytes = staging_bytes;
  info->tile_pool_bytes = tile_pool_bytes;
  info->compressed_pool_bytes = compressed_pool_bytes;
  info->aggregate_bytes = aggregate_device;
  info->lod_bytes = lod_device;
  info->codec_bytes = codec_bytes;

  info->device_bytes = staging_bytes + tile_pool_bytes + compressed_pool_bytes +
                       aggregate_device + lod_device + codec_bytes;
  info->host_pinned_bytes = staging_host + aggregate_host;

  info->tiles_per_epoch = tiles_per_epoch;
  info->total_tiles = total_tiles;
  info->max_output_size = max_output_size;
  info->nlod = nlod;
  info->epochs_per_batch = K;

  computed_stream_layouts_free(&cl);
  return 0;
}

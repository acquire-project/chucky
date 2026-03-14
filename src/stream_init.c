#include "metric.h"
#include "stream_internal.h"

#include "aggregate.h"
#include "compress.h"
#include "crc32c.h"
#include "index.ops.h"
#include "lod.h"
#include "lod_plan.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "shard_delivery.h"

#include <stdlib.h>
#include <string.h>

static inline CUresult
cu_event_destroy(CUevent e)
{
  return e ? cuEventDestroy(e) : CUDA_SUCCESS;
}

static inline CUresult
cu_stream_destroy(CUstream s)
{
  return s ? cuStreamDestroy(s) : CUDA_SUCCESS;
}

static inline CUresult
cu_mem_free(CUdeviceptr p)
{
  return p ? cuMemFree(p) : CUDA_SUCCESS;
}

static inline CUresult
cu_mem_free_host(void* p)
{
  return p ? cuMemFreeHost(p) : CUDA_SUCCESS;
}

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
destroy_level_state(struct level_flush_state* lls)
{
  aggregate_layout_destroy(&lls->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&lls->agg[i]);
  CUWARN(cu_mem_free(lls->d_batch_gather));
  CUWARN(cu_mem_free(lls->d_batch_perm));
  if (lls->shard.shards) {
    for (uint64_t i = 0; i < lls->shard.shard_inner_count; ++i)
      free(lls->shard.shards[i].index);
    free(lls->shard.shards);
  }
}

static void
destroy_cuda_streams_and_events(struct gpu_streams* streams,
                                struct staging_state* stage,
                                struct pool_state* pools,
                                struct flush_pipeline* flush)
{
  CUWARN(cu_stream_destroy(streams->h2d));
  CUWARN(cu_stream_destroy(streams->compute));
  CUWARN(cu_stream_destroy(streams->compress));
  CUWARN(cu_stream_destroy(streams->d2h));

  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stage->slot[i];
    CUWARN(cu_event_destroy(ss->t_h2d_end));
    CUWARN(cu_event_destroy(ss->t_h2d_start));
    CUWARN(cu_event_destroy(ss->t_scatter_start));
    CUWARN(cu_event_destroy(ss->t_scatter_end));
  }

  for (int i = 0; i < 2; ++i)
    CUWARN(cu_event_destroy(pools->ready[i]));

  for (int i = 0; i < 2; ++i) {
    struct flush_slot_gpu* fs = &flush->slot[i];
    CUWARN(cu_event_destroy(fs->t_compress_end));
    CUWARN(cu_event_destroy(fs->t_compress_start));
    CUWARN(cu_event_destroy(fs->t_aggregate_end));
    CUWARN(cu_event_destroy(fs->t_d2h_start));
    CUWARN(cu_event_destroy(fs->ready));
  }
}

static void
destroy_l0_layout(struct stream_layout* layout)
{
  CUWARN(cu_mem_free((CUdeviceptr)layout->d_lifted_shape));
  CUWARN(cu_mem_free((CUdeviceptr)layout->d_lifted_strides));
}

static void
destroy_staging_buffers(struct staging_state* stage)
{
  for (int i = 0; i < 2; ++i) {
    CUWARN(cu_mem_free_host(stage->slot[i].h_in));
    CUWARN(cu_mem_free(stage->slot[i].d_in));
  }
}

static void
destroy_tile_pools(struct pool_state* pools)
{
  for (int i = 0; i < 2; ++i)
    CUWARN(cu_mem_free(pools->buf[i]));
}

static void
destroy_compression(struct codec* c, struct flush_pipeline* flush)
{
  codec_free(c);
  for (int i = 0; i < 2; ++i)
    CUWARN(cu_mem_free(flush->slot[i].d_compressed));
}

static void
destroy_aggregate_and_shards(struct flush_pipeline* flush, int nlod)
{
  for (int lv = 0; lv < nlod; ++lv)
    destroy_level_state(&flush->levels[lv]);
}

static void
destroy_batch_events(struct batch_state* batch)
{
  for (uint32_t i = 0; i < batch->epochs_per_batch; ++i)
    CUWARN(cu_event_destroy(batch->pool_events[i]));
}

void
tile_stream_gpu_destroy(struct tile_stream_gpu* s)
{
  if (!s)
    return;
  destroy_batch_events(&s->batch);
  destroy_aggregate_and_shards(&s->flush, s->levels.nlod);
  destroy_compression(&s->codec, &s->flush);
  destroy_tile_pools(&s->pools);
  lod_state_destroy(&s->lod);
  destroy_staging_buffers(&s->stage);
  destroy_l0_layout(&s->layout);
  destroy_cuda_streams_and_events(&s->streams, &s->stage, &s->pools, &s->flush);
  *s = (struct tile_stream_gpu){ 0 };
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
                             struct staging_state* stage,
                             struct pool_state* pools,
                             struct flush_pipeline* flush)
{
  CU(Fail, cuStreamCreate(&streams->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&streams->d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&stage->slot[i].t_h2d_end, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->slot[i].t_h2d_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->slot[i].t_scatter_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->slot[i].t_scatter_end, CU_EVENT_DEFAULT));
  }

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&pools->ready[i], CU_EVENT_DEFAULT));
  }

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&flush->slot[i].t_compress_end, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&flush->slot[i].t_compress_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&flush->slot[i].t_aggregate_end, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&flush->slot[i].t_d2h_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&flush->slot[i].ready, CU_EVENT_DEFAULT));
  }

  return 0;
Fail:
  return 1;
}

static int
init_l0_layout(struct stream_layout* layout,
               uint8_t rank,
               size_t bpe,
               const struct dimension* dims,
               enum compression_codec codec,
               const uint8_t* storage_order)
{
  // Build the "lifted" layout for the scatter transpose kernel.
  //
  // Each dimension d with shape S and tile size T is split into two:
  //   lifted_shape = (t_{D-1}, n_{D-1}, ..., t_0, n_0)
  // where t_i = ceil(S_i / T_i) (tile count) and n_i = T_i (within-tile).
  //
  // Strides are computed so the scatter kernel writes contiguous tiles:
  //   within-tile strides (n_i) are C-order within a tile,
  //   grid strides (t_i) jump between tile slots in the pool.
  //
  // The outermost grid stride (strides[0]) spans a full epoch. After
  // computing tiles_per_epoch from it, strides[0] is set to 0 so that
  // all epochs map to the same pool layout — the epoch offset is added
  // separately via current_pool_epoch().
  layout->lifted_rank = 2 * rank;
  layout->tile_elements = 1;

  uint64_t tile_count[HALF_MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] =
      (dims[i].size == 0) ? 1 : ceildiv(dims[i].size, dims[i].tile_size);
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

  {
    const size_t shape_bytes = layout->lifted_rank * sizeof(uint64_t);
    const size_t strides_bytes = layout->lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&layout->d_lifted_shape, shape_bytes));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&layout->d_lifted_strides, strides_bytes));
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)layout->d_lifted_shape,
                    layout->lifted_shape,
                    shape_bytes));
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)layout->d_lifted_strides,
                    layout->lifted_strides,
                    strides_bytes));
  }

  return 0;
Fail:
  return 1;
}

static int
init_staging_buffers(struct staging_state* stage, size_t buffer_capacity_bytes)
{
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemHostAlloc(&stage->slot[i].h_in, buffer_capacity_bytes, 0));
    CU(Fail, cuMemAlloc(&stage->slot[i].d_in, buffer_capacity_bytes));
  }

  return 0;
Fail:
  return 1;
}

static int
init_tile_pools(struct pool_state* pools,
                struct level_geometry* levels,
                const struct stream_layout* layout,
                const struct lod_state* lod,
                size_t bpe,
                uint32_t epochs_per_batch,
                CUstream compute)
{
  // Compute total_tiles and level tile offsets from L0 + LOD layouts
  // These are per-epoch counts; the pool holds K epochs, where K is
  // epochs_per_batch.
  levels->tile_count[0] = layout->tiles_per_epoch;
  levels->tile_offset[0] = 0;
  levels->total_tiles = layout->tiles_per_epoch;

  for (int lv = 1; lv < levels->nlod; ++lv) {
    levels->tile_count[lv] = lod->layouts[lv].tiles_per_epoch;
    levels->tile_offset[lv] = levels->total_tiles;
    levels->total_tiles += levels->tile_count[lv];
  }

  // Pool holds K epochs worth of tiles
  const size_t pool_bytes = (uint64_t)epochs_per_batch * levels->total_tiles *
                            layout->tile_stride * bpe;

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&pools->buf[i], pool_bytes));
    CU(Fail, cuMemsetD8Async(pools->buf[i], 0, pool_bytes, compute));
  }

  return 0;
Fail:
  return 1;
}

static int
init_compression(struct codec* c,
                 struct flush_pipeline* flush,
                 enum compression_codec codec_type,
                 size_t bpe,
                 uint32_t K,
                 uint64_t total_tiles,
                 uint64_t tile_stride)
{
  const uint64_t M = (uint64_t)K * total_tiles;
  const size_t tile_bytes = tile_stride * bpe;

  CHECK(Fail, codec_init(c, codec_type, tile_bytes, M) == 0);

  for (int fc = 0; fc < 2; ++fc) {
    struct flush_slot_gpu* fs = &flush->slot[fc];
    CU(Fail, cuMemAlloc(&fs->d_compressed, M * c->max_output_size));
  }

  return 0;
Fail:
  return 1;
}

static int
init_aggregate_and_shards(struct flush_pipeline* flush,
                          const struct tile_stream_configuration* config,
                          const struct level_geometry* levels,
                          uint32_t K,
                          size_t max_output_size,
                          const struct lod_plan* plan,
                          CUstream compute,
                          const uint8_t* storage_order)
{
  const uint8_t rank = config->rank;
  const struct dimension* dims = config->dimensions;

  crc32c_init();

  for (int lv = 0; lv < levels->nlod; ++lv) {
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
      const uint64_t* lv_shape = plan->shapes[lv];
      for (int d = 0; d < rank; ++d) {
        tile_count[d] = ceildiv(lv_shape[d], dims[d].tile_size);
        uint64_t tps = dims[d].tiles_per_shard;
        tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
      }
    }

    uint64_t tiles_lv = levels->tile_count[lv];

    // Epochs per batch for this level.
    // L0 fires every epoch; higher levels fire every 2^lv epochs.
    // When period > K the level fires less than once per batch, so
    // batch_active_count is 0 (no LUTs) but we still size slots for 1.
    uint32_t batch_count = K;
    if (levels->dim0_downsample && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count = (K >= period) ? K / period : 0;
    }
    flush->levels[lv].batch_active_count = batch_count;

    // Permute tile_count and tiles_per_shard into storage order for aggregate.
    // Aggregate decomposes tile indices assuming tile-pool dimension ordering,
    // which is now storage order.
    uint64_t so_tile_count[HALF_MAX_RANK], so_tiles_per_shard[HALF_MAX_RANK];
    for (int j = 0; j < rank; ++j) {
      so_tile_count[j] = tile_count[storage_order[j]];
      so_tiles_per_shard[j] = tiles_per_shard[storage_order[j]];
    }

    // Aggregate layout: per-epoch tile geometry (used by the single-epoch
    // aggregate path for K=1 or partial batch fallback).
    CHECK(Fail,
          aggregate_layout_init(&flush->levels[lv].agg_layout,
                                rank,
                                so_tile_count,
                                so_tiles_per_shard,
                                tiles_lv,
                                max_output_size,
                                config->shard_alignment) == 0);

    // Aggregate slots sized for batch (at least 1 for infrequent dim0 levels).
    uint32_t slot_count = batch_count > 0 ? batch_count : 1;
    uint64_t batch_tiles = (uint64_t)slot_count * tiles_lv;
    uint64_t batch_covering =
      (uint64_t)slot_count * flush->levels[lv].agg_layout.covering_count;
    size_t batch_agg_bytes =
      agg_pool_bytes(batch_tiles,
                     max_output_size,
                     flush->levels[lv].agg_layout.covering_count,
                     flush->levels[lv].agg_layout.tps_inner,
                     flush->levels[lv].agg_layout.page_size);

    for (int i = 0; i < 2; ++i) {
      CHECK(Fail,
            aggregate_batch_slot_init(&flush->levels[lv].agg[i],
                                      batch_tiles,
                                      batch_covering,
                                      batch_agg_bytes) == 0);
      CU(Fail, cuEventRecord(flush->levels[lv].agg[i].ready, compute));
    }

    // Shard state
    struct shard_state* ss = &flush->levels[lv].shard;
    {
      uint64_t tps0 = tiles_per_shard[0];
      // Dim0-downsampled levels emit every 2^lv epochs, so each emission
      // covers 2^lv times more temporal range -> fewer tiles per shard.
      if (levels->dim0_downsample && lv > 0) {
        uint64_t divisor = 1ull << lv;
        tps0 = (tps0 > divisor) ? tps0 / divisor : 1;
      }
      ss->tiles_per_shard_0 = tps0;
    }
    ss->tiles_per_shard_inner = 1;
    for (int d = 1; d < rank; ++d)
      ss->tiles_per_shard_inner *= tiles_per_shard[d];
    ss->tiles_per_shard_total =
      ss->tiles_per_shard_0 * ss->tiles_per_shard_inner;

    ss->shard_inner_count = 1;
    for (int d = 1; d < rank; ++d)
      ss->shard_inner_count *= ceildiv(tile_count[d], tiles_per_shard[d]);

    ss->shards = (struct active_shard*)calloc(ss->shard_inner_count,
                                              sizeof(struct active_shard));
    CHECK(Fail, ss->shards);

    size_t index_bytes = 2 * ss->tiles_per_shard_total * sizeof(uint64_t);
    for (uint64_t i = 0; i < ss->shard_inner_count; ++i) {
      ss->shards[i].index = (uint64_t*)malloc(index_bytes);
      CHECK(Fail, ss->shards[i].index);
      memset(ss->shards[i].index, 0xFF, index_bytes);
    }

    ss->epoch_in_shard = 0;
    ss->shard_epoch = 0;
  }

  return 0;
Fail:
  return 1;
}

// Precompute per-level gather and permutation LUTs for batch aggregate.
//
// gather[a * tiles_lv + j] maps batch-tile (a, j) to its position in the
// compressed buffer: pool_epoch(a) * total_tiles + level_offset + j.
//
// perm[a * tiles_lv + j] maps batch-tile (a, j) to the shard-ordered
// output position: shard_perm(j) * batch_count + a.  This interleaves
// epochs within each shard so tiles appear in epoch order.
//
// Must be called AFTER init_aggregate_and_shards (needs agg_layout).
static int
init_batch_luts(struct flush_pipeline* flush,
                const struct level_geometry* levels,
                uint32_t K)
{
  if (K <= 1)
    return 0;

  for (int lv = 0; lv < levels->nlod; ++lv) {
    struct level_flush_state* lvl = &flush->levels[lv];
    uint32_t batch_count = lvl->batch_active_count;
    uint64_t tiles_lv = levels->tile_count[lv];
    uint64_t lut_len = (uint64_t)batch_count * tiles_lv;

    if (lut_len == 0)
      continue;

    uint32_t* h_gather = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
    uint32_t* h_perm = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
    CHECK(Fail2, h_gather && h_perm);

    const struct aggregate_layout* al = &lvl->agg_layout;

    for (uint32_t a = 0; a < batch_count; ++a) {
      // Level lv fires at pool epochs period-1, 2*period-1, etc.
      uint32_t period = 1;
      if (levels->dim0_downsample && lv > 0)
        period = 1u << lv;
      uint32_t pool_epoch = (a + 1) * period - 1;

      for (uint64_t j = 0; j < tiles_lv; ++j) {
        uint64_t idx = (uint64_t)a * tiles_lv + j;

        // gather: map batch-tile to compressed buffer position
        h_gather[idx] = (uint32_t)(pool_epoch * levels->total_tiles +
                                   levels->tile_offset[lv] + j);

        // perm: shard-order position via lifted strides, interleaved by epoch
        uint64_t perm_pos = 0;
        uint64_t rest = j;
        for (int d = al->lifted_rank - 1; d >= 0; --d) {
          uint64_t coord = rest % al->lifted_shape[d];
          rest /= al->lifted_shape[d];
          perm_pos += coord * (uint64_t)al->lifted_strides[d];
        }
        h_perm[idx] = (uint32_t)(perm_pos * batch_count + a);
      }
    }

    CU(Fail2, cuMemAlloc(&lvl->d_batch_gather, lut_len * sizeof(uint32_t)));
    CU(Fail2,
       cuMemcpyHtoD(lvl->d_batch_gather, h_gather, lut_len * sizeof(uint32_t)));

    CU(Fail2, cuMemAlloc(&lvl->d_batch_perm, lut_len * sizeof(uint32_t)));
    CU(Fail2,
       cuMemcpyHtoD(lvl->d_batch_perm, h_perm, lut_len * sizeof(uint32_t)));

    free(h_gather);
    free(h_perm);
    continue;

  Fail2:
    free(h_gather);
    free(h_perm);
    goto Fail;
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
seed_events(const struct staging_state* stage,
            const struct pool_state* pools,
            const struct flush_pipeline* flush,
            const struct lod_state* lod,
            CUstream compute)
{
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(stage->slot[i].t_h2d_end, compute));
    CU(Fail, cuEventRecord(stage->slot[i].t_h2d_start, compute));
    CU(Fail, cuEventRecord(stage->slot[i].t_scatter_start, compute));
    CU(Fail, cuEventRecord(stage->slot[i].t_scatter_end, compute));
  }
  CU(Fail, cuEventRecord(pools->ready[0], compute));
  CU(Fail, cuEventRecord(pools->ready[1], compute));
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(flush->slot[i].t_compress_end, compute));
    CU(Fail, cuEventRecord(flush->slot[i].t_compress_start, compute));
    CU(Fail, cuEventRecord(flush->slot[i].t_aggregate_end, compute));
    CU(Fail, cuEventRecord(flush->slot[i].t_d2h_start, compute));
    CU(Fail, cuEventRecord(flush->slot[i].ready, compute));
  }

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
// On success, writes resolved storage_order, *enable_multiscale,
// *dim0_downsample. Returns 0 on success, non-zero on invalid config.
static int
validate_config(const struct tile_stream_configuration* config,
                uint8_t* storage_order,
                int* enable_multiscale,
                int* dim0_downsample)
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

  if (resolve_storage_order(config->rank, config->dimensions, storage_order)) {
    log_error("invalid storage_order permutation");
    goto Fail;
  }

  // Compute lod_mask from dimensions (uniform: includes dim 0 if marked).
  uint32_t lod_mask = 0;
  int dim0_ds = 0;
  for (int d = 0; d < config->rank; ++d) {
    if (config->dimensions[d].downsample) {
      lod_mask |= (1u << d);
      if (d == 0) {
        dim0_ds = 1;
        // Validate dim0 reduce method: only mean/min/max supported
        enum lod_reduce_method m = config->dim0_reduce_method;
        if (m != lod_reduce_mean && m != lod_reduce_min &&
            m != lod_reduce_max) {
          log_error("dim0 reduce method must be mean, min, or max");
          goto Fail;
        }
      }
    }
  }
  // dim0 downsampling requires at least one spatial dim also downsampled
  if (dim0_ds && (lod_mask & ~1u) == 0) {
    log_error("dim0 downsample requires at least one spatial dim downsampled");
    goto Fail;
  }

  *enable_multiscale = (lod_mask & ~1u) != 0;
  *dim0_downsample = dim0_ds;

  return 0;
Fail:
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

int
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct tile_stream_gpu* out)
{
  uint8_t resolved_storage_order[HALF_MAX_RANK];
  int enable_multiscale, dim0_downsample;

  CHECK(EarlyFail, out);
  CHECK(EarlyFail, config && config->shard_sink);
  CHECK(EarlyFail,
        validate_config(config,
                        resolved_storage_order,
                        &enable_multiscale,
                        &dim0_downsample) == 0);

  *out = (struct tile_stream_gpu){
    .config = *config,
    .levels = { .nlod = 1,
                .enable_multiscale = enable_multiscale,
                .dim0_downsample = dim0_downsample },
  };
  tile_stream_gpu_init_vtable(out);

  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  CHECK(Fail,
        init_cuda_streams_and_events(
          &out->streams, &out->stage, &out->pools, &out->flush) == 0);
  CHECK(Fail,
        init_l0_layout(&out->layout,
                       config->rank,
                       config->bytes_per_element,
                       config->dimensions,
                       config->codec,
                       resolved_storage_order) == 0);
  CHECK(Fail,
        init_staging_buffers(&out->stage, out->config.buffer_capacity_bytes) ==
          0);
  CHECK(Fail,
        lod_state_init(&out->lod,
                       &out->levels,
                       &out->layout,
                       &out->config,
                       resolved_storage_order) == 0);

  // Compute epochs_per_batch (K) after lod_layouts sets total_tiles/nlod.
  // nlod is set by lod_state_init; total_tiles is set by init_tile_pools.
  // We need total_tiles to compute K, but init_tile_pools needs K for sizing.
  // Compute per-epoch total_tiles here (same logic as init_tile_pools).
  {
    uint64_t total_tiles_per_epoch = out->layout.tiles_per_epoch;
    for (int lv = 1; lv < out->levels.nlod; ++lv)
      total_tiles_per_epoch += out->lod.layouts[lv].tiles_per_epoch;

    uint32_t K = compute_epochs_per_batch(config, total_tiles_per_epoch);
    // Ensure K is a power of 2
    CHECK(Fail, (K & (K - 1)) == 0);
    out->batch.epochs_per_batch = K;
    out->batch.accumulated = 0;
  }

  CHECK(Fail,
        init_tile_pools(&out->pools,
                        &out->levels,
                        &out->layout,
                        &out->lod,
                        config->bytes_per_element,
                        out->batch.epochs_per_batch,
                        out->streams.compute) == 0);
  CHECK(Fail,
        init_compression(&out->codec,
                         &out->flush,
                         config->codec,
                         config->bytes_per_element,
                         out->batch.epochs_per_batch,
                         out->levels.total_tiles,
                         out->layout.tile_stride) == 0);
  CHECK(Fail,
        init_aggregate_and_shards(&out->flush,
                                  &out->config,
                                  &out->levels,
                                  out->batch.epochs_per_batch,
                                  out->codec.max_output_size,
                                  &out->lod.plan,
                                  out->streams.compute,
                                  resolved_storage_order) == 0);
  CHECK(Fail,
        init_batch_luts(
          &out->flush, &out->levels, out->batch.epochs_per_batch) == 0);
  CHECK(Fail, init_batch_events(&out->batch, out->streams.compute) == 0);
  if (out->levels.enable_multiscale) {
    CHECK(Fail,
          lod_state_init_buffers(
            &out->lod, &out->layout, out->config.bytes_per_element) == 0);
    if (out->levels.dim0_downsample)
      CHECK(Fail, lod_state_init_accumulators(&out->lod, &out->config) == 0);
  }
  CHECK(
    Fail,
    seed_events(
      &out->stage, &out->pools, &out->flush, &out->lod, out->streams.compute) ==
      0);

  CU(Fail, cuStreamSynchronize(out->streams.compute));

  out->metrics = init_metrics(out->levels.enable_multiscale);

  // Initialize metadata update timer
  out->metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&out->metadata_update_clock);

  return 0;

Fail:
  tile_stream_gpu_destroy(out);
EarlyFail:
  return 1;
}

// --- Memory estimate ---

int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_memory_info* info)
{
  if (!info)
    return 1;

  uint8_t storage_order[HALF_MAX_RANK];
  int enable_multiscale, dim0_ds;
  if (validate_config(config, storage_order, &enable_multiscale, &dim0_ds))
    return 1;

  memset(info, 0, sizeof(*info));

  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const struct dimension* dims = config->dimensions;
  const size_t buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // --- L0 layout math (mirrors init_l0_layout) ---

  uint64_t tile_elements = 1;
  uint64_t tile_count[HALF_MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] =
      (dims[i].size == 0) ? 1 : ceildiv(dims[i].size, dims[i].tile_size);
    tile_elements *= dims[i].tile_size;
  }

  const size_t alignment = codec_alignment(config->codec);
  const size_t tile_bytes = tile_elements * bpe;
  const size_t padded_bytes = align_up(tile_bytes, alignment);
  const uint64_t tile_stride = padded_bytes / bpe;

  uint64_t tiles_per_epoch = 1;
  for (int d = 1; d < rank; ++d)
    tiles_per_epoch *= tile_count[d];

  const uint64_t epoch_elements = tiles_per_epoch * tile_elements;

  // --- LOD plan (CPU only) ---

  uint32_t lod_mask = 0;
  for (int d = 0; d < rank; ++d) {
    if (dims[d].downsample)
      lod_mask |= (1u << d);
  }

  struct lod_plan plan = { 0 };
  int nlod = 1;

  if (enable_multiscale) {
    uint64_t shape[HALF_MAX_RANK];
    uint64_t tile_shape[HALF_MAX_RANK];
    shape[0] = dims[0].tile_size;
    for (int d = 1; d < rank; ++d)
      shape[d] = dims[d].size;
    for (int d = 0; d < rank; ++d)
      tile_shape[d] = dims[d].tile_size;

    if (lod_plan_init(
          &plan, rank, shape, tile_shape, lod_mask, LOD_MAX_LEVELS, dim0_ds))
      return 1;

    nlod = plan.nlod;
  }

  // --- Per-level tile counts (mirrors init_tile_pools) ---

  uint64_t level_tile_count[LOD_MAX_LEVELS];
  memset(level_tile_count, 0, sizeof(level_tile_count));
  level_tile_count[0] = tiles_per_epoch;
  uint64_t total_tiles = tiles_per_epoch;

  for (int lv = 1; lv < nlod; ++lv) {
    const uint64_t* lv_shape = plan.shapes[lv];
    uint64_t lv_tiles = 1;
    for (int d = 1; d < rank; ++d)
      lv_tiles *= ceildiv(lv_shape[d], dims[d].tile_size);
    level_tile_count[lv] = lv_tiles;
    total_tiles += lv_tiles;
  }

  // --- Compute K ---
  uint32_t K = compute_epochs_per_batch(config, total_tiles);

  // --- Codec queries (no GPU allocation) ---

  const size_t chunk_bytes = tile_stride * bpe;
  const size_t max_output_size =
    codec_max_output_size(config->codec, chunk_bytes);
  if (config->codec != CODEC_NONE && max_output_size == 0)
    goto Fail;

  const uint64_t codec_batch = (uint64_t)K * total_tiles;
  const size_t nvcomp_temp =
    codec_temp_bytes(config->codec, chunk_bytes, codec_batch);

  // --- Tally: staging (device + host pinned) ---

  const size_t staging_bytes = 2 * buffer_capacity_bytes;
  const size_t staging_host = 2 * buffer_capacity_bytes;

  // --- Tally: tile pools (device) ---

  const size_t tile_pool_bytes =
    2 * (uint64_t)K * total_tiles * tile_stride * bpe;

  // --- Tally: compressed pool in flush slots (device) ---

  const size_t compressed_pool_bytes =
    2 * (uint64_t)K * total_tiles * max_output_size;

  // --- Tally: codec device arrays ---

  size_t codec_bytes = 0;
  codec_bytes += codec_batch * sizeof(size_t); // d_comp_sizes
  codec_bytes += codec_batch * sizeof(size_t); // d_uncomp_sizes
  if (config->codec != CODEC_NONE)
    codec_bytes += 2 * codec_batch * sizeof(void*); // d_ptrs
  codec_bytes += nvcomp_temp;                       // d_temp

  // --- Tally: aggregate (device + host pinned) ---

  size_t aggregate_device = 0;
  size_t aggregate_host = 0;

  for (int lv = 0; lv < nlod; ++lv) {
    uint64_t tc[HALF_MAX_RANK];
    uint64_t tps[HALF_MAX_RANK];

    if (lv == 0) {
      for (int d = 0; d < rank; ++d) {
        tc[d] = tile_count[d];
        tps[d] =
          (dims[d].tiles_per_shard == 0) ? tc[d] : dims[d].tiles_per_shard;
      }
    } else {
      const uint64_t* lv_shape = plan.shapes[lv];
      for (int d = 0; d < rank; ++d) {
        tc[d] = ceildiv(lv_shape[d], dims[d].tile_size);
        tps[d] =
          (dims[d].tiles_per_shard == 0) ? tc[d] : dims[d].tiles_per_shard;
      }
    }

    // covering_count = prod(ceildiv(tc[d], tps[d]) * tps[d]) for d=1..rank-1
    uint64_t covering_count = 1;
    for (int d = 1; d < rank; ++d)
      covering_count *= ceildiv(tc[d], tps[d]) * tps[d];

    uint32_t batch_count = K;
    if (dim0_ds && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count = (K >= period) ? K / period : 1;
    }

    uint64_t tps_inner_lv = 1;
    for (int d = 1; d < rank; ++d)
      tps_inner_lv *= tps[d];

    uint64_t batch_tiles = (uint64_t)batch_count * level_tile_count[lv];
    uint64_t batch_covering = (uint64_t)batch_count * covering_count;
    size_t batch_agg_bytes = agg_pool_bytes(batch_tiles,
                                            max_output_size,
                                            covering_count,
                                            tps_inner_lv,
                                            config->shard_alignment);

    // aggregate_layout: device lifted shape + strides
    size_t agg_layout_dev =
      2 * (rank - 1) * sizeof(uint64_t) + 2 * (rank - 1) * sizeof(int64_t);

    // aggregate_slot x 2: device arrays (batch-sized)
    size_t slot_dev = (batch_covering + 1) * sizeof(size_t) // d_permuted_sizes
                      + (batch_covering + 1) * sizeof(size_t) // d_offsets
                      + batch_tiles * sizeof(uint32_t)        // d_perm
                      + batch_agg_bytes;                      // d_aggregated

    // aggregate_slot x 2: host pinned (batch-sized)
    size_t slot_host = batch_agg_bytes                         // h_aggregated
                       + (batch_covering + 1) * sizeof(size_t) // h_offsets
                       + batch_covering * sizeof(size_t); // h_permuted_sizes

    // Batch LUTs (per level, if K > 1)
    size_t lut_dev = 0;
    if (K > 1)
      lut_dev = batch_tiles * sizeof(uint32_t) * 2; // gather + perm

    aggregate_device += agg_layout_dev + 2 * slot_dev + lut_dev;
    aggregate_host += 2 * slot_host;
  }

  // --- Tally: LOD buffers + shape arrays (device) ---

  size_t lod_device = 0;

  // L0 layout arrays (always present)
  lod_device += 2 * rank * sizeof(uint64_t); // d_lifted_shape
  lod_device += 2 * rank * sizeof(int64_t);  // d_lifted_strides

  if (enable_multiscale) {
    // d_linear + d_morton
    lod_device += epoch_elements * bpe;
    uint64_t total_lod_vals = plan.levels.ends[plan.nlod - 1];
    lod_device += total_lod_vals * bpe;

    // Global shape arrays
    lod_device += rank * sizeof(uint64_t); // d_full_shape
    if (plan.lod_ndim > 0)
      lod_device += plan.lod_ndim * sizeof(uint64_t); // d_lod_shape

    // Gather LUT + batch offsets
    if (plan.lod_ndim > 0) {
      lod_device += plan.lod_counts[0] * sizeof(uint32_t); // d_gather_lut
      lod_device += plan.batch_count * sizeof(uint32_t);   // d_batch_offsets
    }

    // Per reduce-level arrays (0..nlod-2)
    for (int l = 0; l < plan.nlod - 1; ++l) {
      lod_device += plan.lod_ndim * sizeof(uint64_t); // d_child_shapes
      lod_device += plan.lod_ndim * sizeof(uint64_t); // d_parent_shapes

      struct lod_span seg = lod_segment(&plan, l);
      uint64_t n_parents = lod_span_len(seg);
      lod_device += n_parents * sizeof(uint64_t); // d_level_ends
    }

    // Per LOD level (1..nlod-1): layout + shape arrays
    for (int lv = 1; lv < plan.nlod; ++lv) {
      lod_device += 2 * rank * sizeof(uint64_t); // d_lifted_shape
      lod_device += 2 * rank * sizeof(int64_t);  // d_lifted_strides
    }

    // Dim0 accumulator: single buffer + level-ID LUT + counts
    if (dim0_ds) {
      size_t accum_bpe =
        (config->dim0_reduce_method == lod_reduce_mean && bpe == 2) ? 4 : bpe;
      uint64_t total_elems = 0;
      for (int lv = 1; lv < plan.nlod; ++lv)
        total_elems += plan.batch_count * plan.lod_counts[lv];
      lod_device += total_elems * accum_bpe;                // d_accum
      lod_device += total_elems;                            // d_level_ids (u8)
      lod_device += (uint64_t)plan.nlod * sizeof(uint32_t); // d_counts
    }
  }

  // --- Sum totals ---

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

  if (enable_multiscale)
    lod_plan_free(&plan);

  return 0;

Fail:
  if (enable_multiscale)
    lod_plan_free(&plan);
  return 1;
}

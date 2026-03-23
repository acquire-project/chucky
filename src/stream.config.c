#include "stream.config.h"

#include "index.ops.h"
#include "prelude.h"
#include "types.aggregate.h"

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

// Compute epochs_per_batch (K) from config and total chunks per epoch.
// Returns K as a power of 2, clamped to MAX_BATCH_EPOCHS.
static uint32_t
compute_epochs_per_batch(const struct tile_stream_configuration* config,
                         uint64_t total_chunks_per_epoch)
{
  uint32_t K = config->epochs_per_batch;
  if (K == 0) {
    uint32_t target = config->target_batch_chunks;
    if (target == 0)
      target = 1024;
    K = (uint32_t)ceildiv(target, total_chunks_per_epoch);
    K = next_pow2_u32(K);
  }
  if (K > MAX_BATCH_EPOCHS)
    K = MAX_BATCH_EPOCHS;
  return K;
}

// Extract forward permutation from dims[d].storage_position.
// forward[j] = acquisition dim d such that dims[d].storage_position == j.
// Returns 0 on success.
static int
resolve_storage_order(uint8_t rank,
                      const struct dimension* dims,
                      uint8_t* forward)
{
  if (dims[0].storage_position != 0)
    return 1;

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

  if (tmp[0] != 0)
    return 1;

  if (forward)
    memcpy(forward, tmp, rank);
  return 0;
}

// Compute host-side tile_stream_layout fields for a single level (no GPU).
// Returns 0 on success, 1 on overflow.
static int
compute_level_layout(struct tile_stream_layout* layout,
                     uint8_t rank,
                     size_t bpe,
                     const struct dimension* dims,
                     const uint64_t* level_shape,
                     size_t alignment,
                     const uint8_t* storage_order)
{
  layout->lifted_rank = 2 * rank;
  layout->chunk_elements = 1;

  uint64_t chunk_count[HALF_MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    chunk_count[i] =
      (level_shape[i] == 0) ? 1 : ceildiv(level_shape[i], dims[i].chunk_size);
    layout->lifted_shape[2 * i] = chunk_count[i];
    layout->lifted_shape[2 * i + 1] = dims[i].chunk_size;
    layout->chunk_elements *= dims[i].chunk_size;
  }

  {
    size_t chunk_bytes = layout->chunk_elements * bpe;
    size_t padded_bytes = align_up(chunk_bytes, alignment);
    layout->chunk_stride = padded_bytes / bpe;
  }

  {
    uint64_t ts[HALF_MAX_RANK];
    for (int i = 0; i < rank; ++i)
      ts[i] = dims[i].chunk_size;
    compute_lifted_strides(rank,
                           ts,
                           chunk_count,
                           storage_order,
                           (int64_t)layout->chunk_stride,
                           layout->lifted_strides);
  }

  layout->chunks_per_epoch = layout->lifted_strides[0] / layout->chunk_stride;
  CHECK_MUL_OVERFLOW(Fail, layout->chunks_per_epoch, layout->chunk_elements,
                      UINT64_MAX);
  layout->epoch_elements = layout->chunks_per_epoch * layout->chunk_elements;
  layout->lifted_strides[0] = 0; // collapse epoch dim
  layout->chunk_pool_bytes =
    layout->chunks_per_epoch * layout->chunk_stride * bpe;
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
  CHECK(Fail, dtype_bpe(config->dtype) > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= HALF_MAX_RANK);
  CHECK(Fail, config->dimensions);
  CHECK(Fail, dims_validate(config->dimensions, config->rank) == 0);

  {
    uint64_t chunk_elements = 1;
    for (int d = 0; d < config->rank; ++d)
      chunk_elements *= config->dimensions[d].chunk_size;
    if (chunk_elements <= 1)
      log_warn("total chunk elements is %llu (chunk_size=1 in all dims?)",
               (unsigned long long)chunk_elements);
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
      log_error("dim0 downsample requires at least one inner dim downsampled");
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
compute_stream_layouts(
  const struct tile_stream_configuration* config,
  size_t codec_alignment,
  size_t (*max_output_size_fn)(enum compression_codec, size_t chunk_bytes),
  struct computed_stream_layouts* out)
{
  const uint8_t rank = config->rank;
  const size_t bpe = dtype_bpe(config->dtype);
  const struct dimension* dims = config->dimensions;

  memset(out, 0, sizeof(*out));

  CHECK(Fail, validate_config(config) == 0);

  uint8_t storage_order[HALF_MAX_RANK];
  CHECK(Fail, resolve_storage_order(rank, dims, storage_order) == 0);

  int dim0_downsample = dims[0].downsample;
  // LOD mask excludes dim0 — dim0 downsampling is handled separately.
  // Must match lod_plan_init_from_dims().
  uint32_t lod_mask = 0;
  for (int d = 1; d < rank; ++d)
    if (dims[d].downsample)
      lod_mask |= (1u << d);
  int enable_multiscale = lod_mask != 0;

  out->levels.enable_multiscale = enable_multiscale;
  out->levels.dim0_downsample = dim0_downsample;
  out->levels.nlod = 1;

  // --- L0 layout ---
  {
    uint64_t l0_shape[HALF_MAX_RANK];
    for (int d = 0; d < rank; ++d)
      l0_shape[d] = dims[d].size;
    CHECK(Fail,
          compute_level_layout(
            &out->l0, rank, bpe, dims, l0_shape, codec_alignment,
            storage_order) == 0);
  }

  // --- LOD plan ---
  if (enable_multiscale) {
    uint64_t shape[HALF_MAX_RANK];
    uint64_t chunk_shape[HALF_MAX_RANK];
    shape[0] = dims[0].chunk_size;
    for (int d = 1; d < rank; ++d)
      shape[d] = dims[d].size;
    for (int d = 0; d < rank; ++d)
      chunk_shape[d] = dims[d].chunk_size;

    CHECK(Fail,
          lod_plan_init(
            &out->plan, rank, shape, chunk_shape, lod_mask, LOD_MAX_LEVELS) ==
            0);

    out->levels.nlod = out->plan.nlod;

    for (int lv = 1; lv < out->plan.nlod; ++lv)
      CHECK(Fail,
            compute_level_layout(&out->lod_layouts[lv],
                                 rank,
                                 bpe,
                                 dims,
                                 out->plan.shapes[lv],
                                 codec_alignment,
                                 storage_order) == 0);
  }

  // --- Level geometry ---
  out->levels.chunk_count[0] = out->l0.chunks_per_epoch;
  out->levels.chunk_offset[0] = 0;
  out->levels.total_chunks = out->l0.chunks_per_epoch;

  for (int lv = 1; lv < out->levels.nlod; ++lv) {
    out->levels.chunk_count[lv] = out->lod_layouts[lv].chunks_per_epoch;
    out->levels.chunk_offset[lv] = out->levels.total_chunks;
    out->levels.total_chunks += out->levels.chunk_count[lv];
  }

  // --- Epochs per batch (K) ---
  out->epochs_per_batch =
    compute_epochs_per_batch(config, out->levels.total_chunks);

  // --- Codec-derived max_output_size ---
  {
    const size_t chunk_bytes = out->l0.chunk_stride * bpe;
    out->max_output_size = max_output_size_fn(config->codec, chunk_bytes);
    if (config->codec != CODEC_NONE && out->max_output_size == 0)
      goto Fail;
  }

  // --- Per-level aggregate layout and shard geometry ---
  for (int lv = 0; lv < out->levels.nlod; ++lv) {
    uint64_t chunk_count[HALF_MAX_RANK];
    uint64_t chunks_per_shard[HALF_MAX_RANK];

    if (lv == 0) {
      for (int d = 0; d < rank; ++d) {
        chunk_count[d] =
          (dims[d].size == 0) ? 1 : ceildiv(dims[d].size, dims[d].chunk_size);
        uint64_t cps = dims[d].chunks_per_shard;
        chunks_per_shard[d] = (cps == 0) ? chunk_count[d] : cps;
      }
    } else {
      const uint64_t* lv_shape = out->plan.shapes[lv];
      for (int d = 0; d < rank; ++d) {
        chunk_count[d] = ceildiv(lv_shape[d], dims[d].chunk_size);
        uint64_t cps = dims[d].chunks_per_shard;
        chunks_per_shard[d] = (cps == 0) ? chunk_count[d] : cps;
      }
    }

    uint64_t chunks_lv = out->levels.chunk_count[lv];

    uint32_t batch_count = out->epochs_per_batch;
    if (dim0_downsample && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count =
        (out->epochs_per_batch >= period) ? out->epochs_per_batch / period : 0;
    }
    out->per_level[lv].batch_active_count = batch_count;

    uint64_t so_chunk_count[HALF_MAX_RANK], so_chunks_per_shard[HALF_MAX_RANK];
    for (int j = 0; j < rank; ++j) {
      so_chunk_count[j] = chunk_count[storage_order[j]];
      so_chunks_per_shard[j] = chunks_per_shard[storage_order[j]];
    }

    CHECK(Fail,
          aggregate_layout_compute(&out->per_level[lv].agg_layout,
                                   rank,
                                   so_chunk_count,
                                   so_chunks_per_shard,
                                   chunks_lv,
                                   out->max_output_size,
                                   config->shard_alignment) == 0);

    {
      uint64_t cps0 = chunks_per_shard[0];
      if (dim0_downsample && lv > 0) {
        uint64_t divisor = 1ull << lv;
        cps0 = (cps0 > divisor) ? cps0 / divisor : 1;
      }
      out->per_level[lv].chunks_per_shard_0 = cps0;
    }
    out->per_level[lv].chunks_per_shard_inner = 1;
    for (int d = 1; d < rank; ++d)
      out->per_level[lv].chunks_per_shard_inner *= chunks_per_shard[d];
    out->per_level[lv].chunks_per_shard_total =
      out->per_level[lv].chunks_per_shard_0 *
      out->per_level[lv].chunks_per_shard_inner;

    out->per_level[lv].shard_inner_count = 1;
    for (int d = 1; d < rank; ++d)
      out->per_level[lv].shard_inner_count *=
        ceildiv(chunk_count[d], chunks_per_shard[d]);
  }

  return 0;
Fail:
  computed_stream_layouts_free(out);
  return 1;
}

struct stream_metric
mk_stream_metric(const char* name)
{
  return (struct stream_metric){ .name = name, .best_ms = 1e30f };
}

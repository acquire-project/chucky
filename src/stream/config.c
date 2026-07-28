#include "stream/config.h"

#include "dimension.h"
#include "dtype.h"
#include "stream/dim_info.h"
#include "stream/types.aggregate.h"
#include "types.stream.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// Compute epochs_per_batch (K) from config and uncompressed bytes per epoch.
// bytes_per_epoch sums chunks_per_epoch * chunk_stride * bpe across LOD levels.
// User-supplied K is authoritative; otherwise auto-K = ceildiv(target, bpe).
// Auto-K is clamped to AUTO_K_MAX as a safety backstop against configs with
// pathologically small bytes_per_epoch (e.g., unit tests with tiny chunks);
// callers who need more can set epochs_per_batch explicitly.
// Floors at 1.
#define AUTO_K_MAX (1u << 16)

static uint32_t
compute_epochs_per_batch(const struct tile_stream_configuration* config,
                         size_t bytes_per_epoch)
{
  if (config->epochs_per_batch > 0)
    return config->epochs_per_batch;

  size_t target = config->target_batch_bytes;
  if (target == 0)
    target = (size_t)512 << 20; // 512 MiB default
  if (bytes_per_epoch == 0)
    return 1;
  uint64_t K = ceildiv(target, bytes_per_epoch);
  if (K < 1)
    K = 1;
  if (K > AUTO_K_MAX)
    K = AUTO_K_MAX;
  return (uint32_t)K;
}

// Extract forward permutation from dims[d].storage_position.
// forward[j] = acquisition dim d such that dims[d].storage_position == j.
// Returns 0 on success.
static int
resolve_storage_order(uint8_t rank,
                      uint8_t n_append,
                      const struct dimension* dims,
                      uint8_t* forward)
{
  for (int d = 0; d < n_append; ++d)
    if (dims[d].storage_position != d)
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

  for (int d = 0; d < n_append; ++d)
    if (tmp[d] != d)
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
                     uint8_t n_append,
                     size_t bytes_per_element,
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
    size_t chunk_bytes = layout->chunk_elements * bytes_per_element;
    size_t padded_bytes = align_up(chunk_bytes, alignment);
    layout->chunk_stride = padded_bytes / bytes_per_element;
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

  // chunks_per_epoch = product of chunk_count for inner dims (d >= n_append)
  layout->chunks_per_epoch = 1;
  for (int d = n_append; d < rank; ++d)
    layout->chunks_per_epoch *= chunk_count[d];
  CHECK_MUL_OVERFLOW(
    Fail, layout->chunks_per_epoch, layout->chunk_elements, UINT64_MAX);
  layout->epoch_elements = layout->chunks_per_epoch * layout->chunk_elements;
  // Collapse all append dims
  for (int d = 0; d < n_append; ++d)
    layout->lifted_strides[2 * d] = 0;
  layout->chunk_pool_bytes =
    layout->chunks_per_epoch * layout->chunk_stride * bytes_per_element;
  return 0;
Fail:
  return 1;
}

// Validate a tile_stream_configuration.
// On success, stores the resolved dim partition in *di (slices point into
// dims). Returns 0 on success, non-zero on invalid config.
static int
validate_config(const struct tile_stream_configuration* config,
                struct dim_info* di,
                const struct dimension* dims)
{
  CHECK(Fail, config);
  CHECK(Fail, dtype_bpe(config->dtype) > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= HALF_MAX_RANK);
  CHECK(Fail, config->dimensions);

  if (config->max_threads < 0) {
    log_error("max_threads must be >= 0 (got %d)", config->max_threads);
    goto Fail;
  }

  // dim_info_init validates dims and computes the partition.
  CHECK(Fail, dim_info_init(di, dims, config->rank) == 0);

  {
    uint64_t chunk_elements = 1;
    for (int d = 0; d < config->rank; ++d)
      chunk_elements *= dims[d].chunk_size;
    if (chunk_elements <= 1)
      log_warn("total chunk elements is %llu (chunk_size=1 in all dims?)",
               (unsigned long long)chunk_elements);
  }

  if (config->codec.id == CODEC_LZ4_NON_STANDARD && config->codec.level == 0) {
    log_error("LZ4 requires level >= 1 (LZ4 HC levels 1..12)");
    goto Fail;
  }

  if (codec_is_blosc(config->codec.id) && config->codec.level > 9) {
    log_error("blosc level must be 0..9 (got %d)", config->codec.level);
    goto Fail;
  }

  if (config->max_nlod < 0) {
    log_error("max_nlod must be >= 0 (got %d)", config->max_nlod);
    goto Fail;
  }
  if (config->max_nlod > LOD_MAX_LEVELS) {
    log_error(
      "max_nlod %d exceeds limit (%d)", config->max_nlod, LOD_MAX_LEVELS);
    goto Fail;
  }

  {
    uint8_t na = dim_info_n_append(di);
    if (resolve_storage_order(config->rank, na, dims, NULL)) {
      log_error("invalid storage_order permutation");
      goto Fail;
    }
  }

  if (config->codec.id == CODEC_LZ4_NON_STANDARD)
    log_warn("LZ4 raw block format is not interoperable with existing zarr v3 "
             "readers");

  {
    if (di->append_downsample) {
      enum lod_reduce_method m = config->append_reduce_method;
      if (m != lod_reduce_mean && m != lod_reduce_min && m != lod_reduce_max) {
        log_error("append reduce method must be mean, min, or max");
        goto Fail;
      }
      if (di->lod_mask == 0) {
        log_error(
          "append downsample requires at least one inner dim downsampled");
        goto Fail;
      }
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
  dims_free_names(cl->dims_owned, cl->rank);
  lod_plan_free(&cl->plan);
}

int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       size_t codec_alignment,
                       size_t (*max_output_size_fn)(enum compression_codec,
                                                    size_t chunk_bytes),
                       size_t shard_alignment,
                       struct computed_stream_layouts* out)
{
  const uint8_t rank = config->rank;
  const size_t bytes_per_element = dtype_bpe(config->dtype);

  memset(out, 0, sizeof(*out));

  CHECK(Fail, dims_copy(out->dims_owned, config->dimensions, rank) == 0);
  out->rank = rank;
  const struct dimension* dims = out->dims_owned;

  CHECK(Fail, validate_config(config, &out->dims, dims) == 0);
  const uint8_t na = dim_info_n_append(&out->dims);

  uint8_t storage_order[HALF_MAX_RANK];
  CHECK(Fail, resolve_storage_order(rank, na, dims, storage_order) == 0);

  // --- LOD plan (always runs) ---
  int max_levels;
  if (config->max_nlod == 0)
    max_levels = LOD_MAX_LEVELS;
  else
    max_levels = config->max_nlod;
  CHECK(
    Fail,
    lod_plan_init_from_epoch_dims(
      &out->plan, dims, rank, na, max_levels, config->preserve_aspect_ratio) ==
      0);
  int enable_multiscale = out->plan.lod_mask != 0;
  out->levels.enable_multiscale = enable_multiscale;
  out->levels.nlod = enable_multiscale ? out->plan.levels.nlod : 1;

  // --- All level layouts ---
  for (int lv = 0; lv < out->levels.nlod; ++lv) {
    uint64_t lv_shape[HALF_MAX_RANK];
    level_dims_get_shape(&out->plan.levels.level[lv], rank, lv_shape);
    CHECK(Fail,
          compute_level_layout(&out->layouts[lv],
                               rank,
                               na,
                               bytes_per_element,
                               dims,
                               lv_shape,
                               codec_alignment,
                               storage_order) == 0);
  }

  // --- Level geometry (single loop) ---
  out->levels.total_chunks = 0;
  size_t bytes_per_epoch = 0;
  for (int lv = 0; lv < out->levels.nlod; ++lv) {
    out->levels.level[lv].chunk_count = out->layouts[lv].chunks_per_epoch;
    out->levels.level[lv].chunk_offset = out->levels.total_chunks;
    out->levels.total_chunks += out->levels.level[lv].chunk_count;
    bytes_per_epoch += (size_t)out->layouts[lv].chunks_per_epoch *
                       out->layouts[lv].chunk_stride * bytes_per_element;
  }

  // --- Epochs per batch (K) ---
  out->epochs_per_batch = compute_epochs_per_batch(config, bytes_per_epoch);

  // --- Codec-derived max_output_size ---
  {
    const size_t chunk_bytes = out->layouts[0].chunk_stride * bytes_per_element;
    out->max_output_size = max_output_size_fn(config->codec.id, chunk_bytes);
    if (config->codec.id != CODEC_NONE && out->max_output_size == 0) {
      log_error("codec %d: max_output_size is 0 (unsupported codec?)",
                config->codec.id);
      goto Fail;
    }
  }

  // --- Per-level aggregate layout and shard geometry ---
  uint64_t chunk_size[HALF_MAX_RANK], cps[HALF_MAX_RANK];
  for (int d = 0; d < rank; ++d) {
    chunk_size[d] = dims[d].chunk_size;
    cps[d] = dims[d].chunks_per_shard;
  }

  // L0 uses the full array shape; L1+ use the epoch-split plan shapes.
  uint64_t array_shape[HALF_MAX_RANK];
  for (int d = 0; d < rank; ++d)
    array_shape[d] = dims[d].size;

  for (int lv = 0; lv < out->levels.nlod; ++lv) {
    uint64_t lv_shape[HALF_MAX_RANK];
    if (lv == 0)
      memcpy(lv_shape, array_shape, rank * sizeof(uint64_t));
    else {
      level_dims_get_shape(&out->plan.levels.level[lv], rank, lv_shape);
      // Append dims use the original array shape, not the epoch-truncated
      // plan shape, so shard geometry matches the full array.
      for (int d = 0; d < na; ++d)
        lv_shape[d] = array_shape[d];
    }
    struct dim_extent de[HALF_MAX_RANK];
    for (int d = 0; d < rank; ++d) {
      de[d].size = lv_shape[d];
      de[d].chunk_size = (uint32_t)chunk_size[d];
    }
    uint64_t shard_inner_count = dim_extent_compute_shards(de, rank, na, cps);

    uint64_t chunks_lv = out->levels.level[lv].chunk_count;

    // Upper bound on epochs firing at this level within a K-epoch batch.
    // K may not divide period, so ceildiv covers the case where the emission
    // boundary falls inside a batch (e.g., K=3, period=2: batches alternate
    // between 1 and 2 level-1 emissions — buffers must fit the worst case).
    uint32_t batch_count = out->epochs_per_batch;
    if (out->dims.append_downsample && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count = (uint32_t)ceildiv(out->epochs_per_batch, period);
    }
    out->per_level[lv].batch_active_count = batch_count;

    uint64_t so_chunk_count[HALF_MAX_RANK], so_chunks_per_shard[HALF_MAX_RANK];
    for (int j = 0; j < rank; ++j) {
      so_chunk_count[j] = de[storage_order[j]].chunk_count;
      so_chunks_per_shard[j] = de[storage_order[j]].chunks_per_shard;
    }

    uint64_t cps_append = 1;
    for (int d = 0; d < na; ++d)
      cps_append *= de[d].chunks_per_shard;
    if (out->dims.append_downsample && lv > 0) {
      uint64_t divisor = 1ull << lv;
      cps_append = (cps_append > divisor) ? cps_append / divisor : 1;
    }
    out->per_level[lv].chunks_per_shard_append = cps_append;

    CHECK(Fail,
          aggregate_layout_compute(&out->per_level[lv].agg_layout,
                                   rank,
                                   na,
                                   so_chunk_count,
                                   so_chunks_per_shard,
                                   chunks_lv,
                                   out->max_output_size,
                                   batch_count,
                                   shard_alignment,
                                   cps_append) == 0);
    out->per_level[lv].chunks_per_shard_inner = 1;
    for (int d = na; d < rank; ++d)
      out->per_level[lv].chunks_per_shard_inner *= de[d].chunks_per_shard;
    out->per_level[lv].chunks_per_shard_total =
      out->per_level[lv].chunks_per_shard_append *
      out->per_level[lv].chunks_per_shard_inner;

    out->per_level[lv].shard_inner_count = shard_inner_count;
  }

  return 0;
Fail:
  computed_stream_layouts_free(out);
  return 1;
}

struct stream_metric
mk_stream_metric(const char* name, enum metric_owner owner)
{
  return (
    struct stream_metric){ .name = name, .owner = owner, .best_ms = 1e30f };
}

const char*
metric_owner_name(enum metric_owner o)
{
  switch (o) {
    case METRIC_OWNER_PRODUCER:
      return "producer";
    case METRIC_OWNER_DRAIN:
      return "drain";
    case METRIC_OWNER_H2D:
      return "h2d";
    case METRIC_OWNER_COMPUTE:
      return "compute";
    case METRIC_OWNER_COMPRESS:
      return "compress";
    case METRIC_OWNER_D2H:
      return "d2h";
    case METRIC_OWNER_NONE:
      break;
  }
  return "unset";
}

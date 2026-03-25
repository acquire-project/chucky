#include "multiarray.cpu.h"
#include "stream/config.h"
#include "zarr/shard_delivery.h"

#include "cpu/aggregate.h"
#include "cpu/compress.h"
#include "cpu/lod.h"
#include "cpu/transpose.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// ---- Per-array descriptor ----

struct array_descriptor
{
  struct tile_stream_configuration config;
  struct computed_stream_layouts cl;
  struct tile_stream_layout layout;
  struct level_geometry levels;
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];
  uint32_t batch_agg_active_count[LOD_MAX_LEVELS];
  struct shard_state shard[LOD_MAX_LEVELS];
  struct shard_sink* sink;
  uint64_t cursor;
  uint64_t max_cursor;
  uint32_t batch_accumulated;
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS];
  uint32_t dim0_counts[LOD_MAX_LEVELS];
  void* dim0_accum;
  int pool_fully_covered;
};

// ---- Aggregate output slot (one per level) ----

struct cpu_agg_slot
{
  void* data;
  size_t data_capacity;
  size_t* offsets;
  size_t* chunk_sizes;
};

// ---- Main struct ----

struct multiarray_tile_stream_cpu
{
  struct multiarray_writer writer;
  int n_arrays;
  int active; // -1 = none

  struct array_descriptor* arrays;

  // Shared pools — sized for max across all arrays.
  void* chunk_pool;
  size_t chunk_pool_bytes;
  void* compressed;
  size_t compressed_bytes;
  size_t* comp_sizes;
  size_t comp_sizes_count;

  // Shared aggregate workspace.
  struct cpu_agg_slot agg_slots[LOD_MAX_LEVELS];
  size_t* agg_permuted_sizes;

  // Shared LUT storage (recomputed on switch).
  uint32_t* agg_perm[LOD_MAX_LEVELS];
  size_t agg_perm_cap[LOD_MAX_LEVELS];
  uint32_t* batch_gather[LOD_MAX_LEVELS];
  size_t batch_gather_cap[LOD_MAX_LEVELS];
  uint32_t* batch_agg_perm[LOD_MAX_LEVELS];
  size_t batch_agg_perm_cap[LOD_MAX_LEVELS];
  uint32_t* scatter_lut;
  size_t scatter_lut_cap;
  uint64_t* scatter_batch_offsets;
  size_t scatter_batch_offsets_cap;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  size_t morton_lut_cap[LOD_MAX_LEVELS];
  uint64_t* lod_batch_offsets[LOD_MAX_LEVELS];
  size_t lod_batch_offsets_cap[LOD_MAX_LEVELS];

  // Shared LOD buffers (multiscale only).
  void* linear;
  size_t linear_bytes;
  void* lod_values;
  size_t lod_values_bytes;

  struct stream_metrics metrics;
};

// ---- Forward declarations ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data);
static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self);

static int
flush_batch(struct multiarray_tile_stream_cpu* ms,
            struct array_descriptor* desc,
            uint32_t n_epochs,
            const uint32_t* active_masks);

static int
scatter_epoch(struct multiarray_tile_stream_cpu* ms,
              struct array_descriptor* desc,
              uint32_t epoch_in_batch,
              uint32_t* out_mask);

static void
recompute_luts(struct multiarray_tile_stream_cpu* ms, int array_index);

// ---- Helpers ----

// Max of two size_t values.
static inline size_t
max_sz(size_t a, size_t b)
{
  return a > b ? a : b;
}

// ---- Create / Destroy ----

struct multiarray_tile_stream_cpu*
multiarray_tile_stream_cpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[])
{
  if (n_arrays <= 0 || !configs || !sinks)
    return NULL;

  struct multiarray_tile_stream_cpu* ms =
    (struct multiarray_tile_stream_cpu*)calloc(1, sizeof(*ms));
  if (!ms)
    return NULL;

  ms->n_arrays = n_arrays;
  ms->active = -1;

  ms->arrays =
    (struct array_descriptor*)calloc((size_t)n_arrays, sizeof(struct array_descriptor));
  CHECK(Fail, ms->arrays);

  // Per-array init + compute maxima.
  size_t max_chunk_pool = 0;
  size_t max_compressed = 0;
  size_t max_comp_sizes = 0;
  uint64_t max_batch_C = 0;
  size_t max_linear = 0;
  size_t max_lod_values = 0;

  // Per-level maxima for aggregate slots and LUTs.
  size_t max_agg_data[LOD_MAX_LEVELS] = { 0 };
  uint64_t max_agg_batch_C[LOD_MAX_LEVELS] = { 0 };
  size_t max_agg_perm[LOD_MAX_LEVELS] = { 0 };
  size_t max_batch_gather[LOD_MAX_LEVELS] = { 0 };
  size_t max_scatter_lut = 0;
  size_t max_scatter_batch_offsets = 0;
  size_t max_morton_lut[LOD_MAX_LEVELS] = { 0 };
  size_t max_lod_batch_offsets[LOD_MAX_LEVELS] = { 0 };

  for (int i = 0; i < n_arrays; ++i) {
    if (configs[i].dtype == dtype_f16)
      goto Fail;

    struct array_descriptor* d = &ms->arrays[i];
    d->config = configs[i];
    d->sink = sinks[i];

    if (compute_stream_layouts(
          &configs[i], 1, compress_cpu_max_output_size, &d->cl))
      goto Fail;

    d->layout = d->cl.layouts[0];
    d->levels = d->cl.levels;

    const uint32_t K = d->cl.epochs_per_batch;
    const size_t bpe = dtype_bpe(configs[i].dtype);
    const uint64_t total_chunks = d->levels.total_chunks;
    const size_t chunk_stride_bytes = d->layout.chunk_stride * bpe;
    const size_t max_out = d->cl.max_output_size;

    d->pool_fully_covered =
      (d->layout.chunk_stride == d->layout.chunk_elements);

    // Compute max_cursor.
    {
      const struct dimension* dims = configs[i].dimensions;
      d->max_cursor =
        (dims[0].size > 0)
          ? ceildiv(dims[0].size, dims[0].chunk_size) * d->layout.epoch_elements
          : 0;
    }

    // Pool sizes for this array.
    size_t pool = (uint64_t)K * total_chunks * chunk_stride_bytes;
    size_t comp = (uint64_t)K * total_chunks * max_out;
    size_t sizes = (uint64_t)K * total_chunks;

    max_chunk_pool = max_sz(max_chunk_pool, pool);
    max_compressed = max_sz(max_compressed, comp);
    max_comp_sizes = max_sz(max_comp_sizes, sizes);

    // Per-level shard + aggregate state.
    for (int lv = 0; lv < d->levels.nlod; ++lv) {
      const struct level_layout_info* li = &d->cl.per_level[lv];
      d->agg_layout[lv] = li->agg_layout;

      const struct aggregate_layout* al = &d->agg_layout[lv];
      uint32_t K_l = li->batch_active_count;
      d->batch_agg_active_count[lv] = K_l;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = al->covering_count;
      uint64_t M_lv = al->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      // Track per-level maxima for shared LUTs.
      max_agg_perm[lv] = max_sz(max_agg_perm[lv], M_lv);

      if (batch_C > max_agg_batch_C[lv])
        max_agg_batch_C[lv] = batch_C;

      size_t data_lv = agg_pool_bytes(
        (uint64_t)slot_count * M_lv,
        al->max_comp_chunk_bytes,
        C_lv,
        al->cps_inner,
        al->page_size);
      max_agg_data[lv] = max_sz(max_agg_data[lv], data_lv);

      if (K_l > 1) {
        uint64_t lut_len = (uint64_t)K_l * M_lv;
        max_batch_gather[lv] = max_sz(max_batch_gather[lv], lut_len);
      }

      // Init shard_state for this array+level.
      struct shard_state* ss = &d->shard[lv];
      ss->chunks_per_shard_0 = li->chunks_per_shard_0;
      ss->chunks_per_shard_inner = li->chunks_per_shard_inner;
      ss->chunks_per_shard_total = li->chunks_per_shard_total;
      ss->shard_inner_count = li->shard_inner_count;
      ss->shards = (struct active_shard*)calloc(
        li->shard_inner_count, sizeof(struct active_shard));
      CHECK(Fail, ss->shards);
      for (uint64_t si = 0; si < li->shard_inner_count; ++si) {
        ss->shards[si].index = (uint64_t*)malloc(
          li->chunks_per_shard_total * 2 * sizeof(uint64_t));
        CHECK(Fail, ss->shards[si].index);
        memset(ss->shards[si].index,
               0xFF,
               li->chunks_per_shard_total * 2 * sizeof(uint64_t));
      }
    }

    // LOD sizes.
    if (d->levels.enable_multiscale) {
      max_linear =
        max_sz(max_linear, d->layout.epoch_elements * bpe);

      uint64_t total_lod_elements =
        d->cl.plan.levels.ends[d->cl.plan.nlod - 1];
      max_lod_values = max_sz(max_lod_values, total_lod_elements * bpe);

      max_scatter_lut =
        max_sz(max_scatter_lut, d->cl.plan.lod_nelem[0]);
      max_scatter_batch_offsets =
        max_sz(max_scatter_batch_offsets, d->cl.plan.batch_count);

      for (int lv = 0; lv < d->levels.nlod; ++lv) {
        max_morton_lut[lv] =
          max_sz(max_morton_lut[lv], d->cl.plan.lod_nelem[lv]);
        max_lod_batch_offsets[lv] =
          max_sz(max_lod_batch_offsets[lv], d->cl.plan.batch_count);
      }

      // Dim0 accum: per-array, not shared.
      if (d->levels.dim0_downsample) {
        uint64_t dim0_total = 0;
        for (int lv = 1; lv < d->cl.plan.nlod; ++lv)
          dim0_total += d->cl.plan.batch_count * d->cl.plan.lod_nelem[lv];
        if (dim0_total > 0) {
          d->dim0_accum = calloc(dim0_total, bpe);
          CHECK(Fail, d->dim0_accum);
        }
        memset(d->dim0_counts, 0, sizeof(d->dim0_counts));
      }
    }
  }

  // Allocate shared pools.
  if (max_chunk_pool > 0) {
    ms->chunk_pool = calloc(1, max_chunk_pool);
    CHECK(Fail, ms->chunk_pool);
    ms->chunk_pool_bytes = max_chunk_pool;
  }

  if (max_compressed > 0) {
    ms->compressed = malloc(max_compressed);
    CHECK(Fail, ms->compressed);
    ms->compressed_bytes = max_compressed;
  }

  if (max_comp_sizes > 0) {
    ms->comp_sizes = (size_t*)calloc(max_comp_sizes, sizeof(size_t));
    CHECK(Fail, ms->comp_sizes);
    ms->comp_sizes_count = max_comp_sizes;
  }

  // Shared aggregate slots (sized for max per-level).
  {
    for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
      struct cpu_agg_slot* as = &ms->agg_slots[lv];
      uint64_t batch_C = max_agg_batch_C[lv];
      if (batch_C > 0) {
        as->offsets = (size_t*)malloc((batch_C + 1) * sizeof(size_t));
        as->chunk_sizes = (size_t*)calloc(batch_C, sizeof(size_t));
        CHECK(Fail, as->offsets && as->chunk_sizes);
      }
      if (max_agg_data[lv] > 0) {
        as->data = malloc(max_agg_data[lv]);
        as->data_capacity = max_agg_data[lv];
        CHECK(Fail, as->data);
      }
    }

    if (max_batch_C > 0) {
      ms->agg_permuted_sizes = (size_t*)calloc(max_batch_C, sizeof(size_t));
      CHECK(Fail, ms->agg_permuted_sizes);
    }
  }

  // Shared LUT storage.
  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    if (max_agg_perm[lv] > 0) {
      ms->agg_perm[lv] =
        (uint32_t*)malloc(max_agg_perm[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->agg_perm[lv]);
      ms->agg_perm_cap[lv] = max_agg_perm[lv];
    }
    if (max_batch_gather[lv] > 0) {
      ms->batch_gather[lv] =
        (uint32_t*)malloc(max_batch_gather[lv] * sizeof(uint32_t));
      ms->batch_agg_perm[lv] =
        (uint32_t*)malloc(max_batch_gather[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->batch_gather[lv] && ms->batch_agg_perm[lv]);
      ms->batch_gather_cap[lv] = max_batch_gather[lv];
      ms->batch_agg_perm_cap[lv] = max_batch_gather[lv];
    }
    if (max_morton_lut[lv] > 0) {
      ms->morton_lut[lv] =
        (uint32_t*)malloc(max_morton_lut[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->morton_lut[lv]);
      ms->morton_lut_cap[lv] = max_morton_lut[lv];
    }
    if (max_lod_batch_offsets[lv] > 0) {
      ms->lod_batch_offsets[lv] =
        (uint64_t*)calloc(max_lod_batch_offsets[lv], sizeof(uint64_t));
      CHECK(Fail, ms->lod_batch_offsets[lv]);
      ms->lod_batch_offsets_cap[lv] = max_lod_batch_offsets[lv];
    }
  }

  if (max_scatter_lut > 0) {
    ms->scatter_lut =
      (uint32_t*)malloc(max_scatter_lut * sizeof(uint32_t));
    CHECK(Fail, ms->scatter_lut);
    ms->scatter_lut_cap = max_scatter_lut;
  }
  if (max_scatter_batch_offsets > 0) {
    ms->scatter_batch_offsets =
      (uint64_t*)calloc(max_scatter_batch_offsets, sizeof(uint64_t));
    CHECK(Fail, ms->scatter_batch_offsets);
    ms->scatter_batch_offsets_cap = max_scatter_batch_offsets;
  }

  // Shared LOD buffers.
  if (max_linear > 0) {
    ms->linear = calloc(1, max_linear);
    CHECK(Fail, ms->linear);
    ms->linear_bytes = max_linear;
  }
  if (max_lod_values > 0) {
    ms->lod_values = calloc(1, max_lod_values);
    CHECK(Fail, ms->lod_values);
    ms->lod_values_bytes = max_lod_values;
  }

  ms->writer.update = update_impl;
  ms->writer.flush = flush_impl;

  return ms;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  return NULL;
}

void
multiarray_tile_stream_cpu_destroy(struct multiarray_tile_stream_cpu* ms)
{
  if (!ms)
    return;

  if (ms->arrays) {
    for (int i = 0; i < ms->n_arrays; ++i) {
      struct array_descriptor* d = &ms->arrays[i];
      for (int lv = 0; lv < d->levels.nlod; ++lv) {
        struct shard_state* ss = &d->shard[lv];
        if (ss->shards) {
          for (uint64_t si = 0; si < ss->shard_inner_count; ++si)
            free(ss->shards[si].index);
          free(ss->shards);
        }
      }
      free(d->dim0_accum);
      computed_stream_layouts_free(&d->cl);
    }
    free(ms->arrays);
  }

  free(ms->chunk_pool);
  free(ms->compressed);
  free(ms->comp_sizes);

  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    free(ms->agg_slots[lv].data);
    free(ms->agg_slots[lv].offsets);
    free(ms->agg_slots[lv].chunk_sizes);
    free(ms->agg_perm[lv]);
    free(ms->batch_gather[lv]);
    free(ms->batch_agg_perm[lv]);
    free(ms->morton_lut[lv]);
    free(ms->lod_batch_offsets[lv]);
  }
  free(ms->agg_permuted_sizes);
  free(ms->scatter_lut);
  free(ms->scatter_batch_offsets);
  free(ms->linear);
  free(ms->lod_values);
  free(ms);
}

struct multiarray_writer*
multiarray_tile_stream_cpu_writer(struct multiarray_tile_stream_cpu* ms)
{
  return ms ? &ms->writer : NULL;
}

// ---- LUT recomputation ----

static void
recompute_luts(struct multiarray_tile_stream_cpu* ms, int array_index)
{
  struct array_descriptor* d = &ms->arrays[array_index];

  // Per-level agg_perm + batch LUTs.
  for (int lv = 0; lv < d->levels.nlod; ++lv) {
    const struct aggregate_layout* al = &d->agg_layout[lv];
    uint64_t M_lv = al->chunks_per_epoch;

    // Single-epoch permutation.
    for (uint64_t i = 0; i < M_lv; ++i)
      ms->agg_perm[lv][i] = (uint32_t)ravel(
        al->lifted_rank, al->lifted_shape, al->lifted_strides, i);

    // Batch LUTs (K_l > 1 only).
    uint32_t K_l = d->batch_agg_active_count[lv];
    if (K_l > 1) {
      uint64_t total_chunks = d->levels.total_chunks;
      for (uint32_t a = 0; a < K_l; ++a) {
        uint32_t period =
          (d->levels.dim0_downsample && lv > 0) ? (1u << lv) : 1;
        uint32_t pool_epoch = (a + 1) * period - 1;

        for (uint64_t j = 0; j < M_lv; ++j) {
          uint64_t idx = (uint64_t)a * M_lv + j;
          ms->batch_gather[lv][idx] =
            (uint32_t)(pool_epoch * total_chunks +
                       d->levels.chunk_offset[lv] + j);
          uint32_t perm_pos = (uint32_t)ravel(
            al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
          ms->batch_agg_perm[lv][idx] = perm_pos * K_l + a;
        }
      }
    }
  }

  // LOD LUTs (multiscale only).
  if (d->levels.enable_multiscale) {
    const struct lod_plan* plan = &d->cl.plan;

    lod_cpu_build_scatter_lut(plan, ms->scatter_lut);
    lod_cpu_build_scatter_batch_offsets(plan, ms->scatter_batch_offsets);

    for (int lv = 0; lv < d->levels.nlod; ++lv) {
      const struct tile_stream_layout* layout_lv = &d->cl.layouts[lv];
      lod_cpu_build_chunk_lut(plan, lv, layout_lv, ms->morton_lut[lv]);

      for (uint64_t bi = 0; bi < plan->batch_count; ++bi) {
        uint64_t remainder = bi;
        int64_t offset = 0;
        for (int k = plan->batch_ndim - 1; k >= 0; --k) {
          uint64_t coord = remainder % plan->batch_shape[k];
          remainder /= plan->batch_shape[k];
          int dm = plan->batch_map[k];
          uint64_t cs = layout_lv->lifted_shape[2 * dm + 1];
          uint64_t ci = coord / cs;
          uint64_t wi = coord % cs;
          offset += (int64_t)ci * layout_lv->lifted_strides[2 * dm];
          offset += (int64_t)wi * layout_lv->lifted_strides[2 * dm + 1];
        }
        ms->lod_batch_offsets[lv][bi] =
          (uint64_t)offset +
          d->levels.chunk_offset[lv] * layout_lv->chunk_stride;
      }
    }
  }
}

// ---- Batch flush ----

static int
flush_batch(struct multiarray_tile_stream_cpu* ms,
            struct array_descriptor* desc,
            uint32_t n_epochs,
            const uint32_t* active_masks)
{
  const size_t bpe = dtype_bpe(desc->config.dtype);
  const size_t max_out = desc->cl.max_output_size;
  const uint64_t total_chunks = desc->levels.total_chunks;

  // Compress all epochs at once.
  if (compress_cpu(desc->config.codec,
                   ms->chunk_pool,
                   desc->layout.chunk_stride * bpe,
                   ms->compressed,
                   max_out,
                   ms->comp_sizes,
                   desc->layout.chunk_elements * bpe,
                   n_epochs * total_chunks))
    return 1;

  // Aggregate + deliver per-level.
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    uint32_t active_count = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (active_masks[e] & (1u << lv))
        active_count++;
    if (active_count == 0)
      continue;

    if (active_count == desc->batch_agg_active_count[lv] &&
        desc->batch_agg_active_count[lv] > 1) {
      // Batch aggregate.
      struct aggregate_cpu_workspace ws = {
        .perm = ms->batch_agg_perm[lv],
        .permuted_sizes = ms->agg_permuted_sizes,
        .data = ms->agg_slots[lv].data,
        .data_capacity = ms->agg_slots[lv].data_capacity,
        .offsets = ms->agg_slots[lv].offsets,
        .chunk_sizes = ms->agg_slots[lv].chunk_sizes,
      };
      struct aggregate_result ar;
      if (aggregate_cpu_batch_into(ms->compressed,
                                   ms->comp_sizes,
                                   ms->batch_gather[lv],
                                   &desc->agg_layout[lv],
                                   active_count,
                                   &ws,
                                   &ar))
        return 1;

      size_t sink_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  &desc->shard[lv],
                                  &ar,
                                  active_count,
                                  desc->sink,
                                  desc->config.shard_alignment,
                                  &sink_bytes))
        return 1;
    } else {
      // Per-epoch fallback.
      for (uint32_t e = 0; e < n_epochs; ++e) {
        if (!(active_masks[e] & (1u << lv)))
          continue;

        uint64_t comp_base =
          (uint64_t)e * total_chunks + desc->levels.chunk_offset[lv];

        const void* comp_lv =
          (const char*)ms->compressed + comp_base * max_out;
        const size_t* sizes_lv = ms->comp_sizes + comp_base;

        struct aggregate_cpu_workspace ws = {
          .perm = ms->agg_perm[lv],
          .permuted_sizes = ms->agg_permuted_sizes,
          .data = ms->agg_slots[lv].data,
          .data_capacity = ms->agg_slots[lv].data_capacity,
          .offsets = ms->agg_slots[lv].offsets,
          .chunk_sizes = ms->agg_slots[lv].chunk_sizes,
        };
        struct aggregate_result ar;
        if (aggregate_cpu_into(
              comp_lv, sizes_lv, &desc->agg_layout[lv], &ws, &ar))
          return 1;

        size_t sink_bytes = 0;
        if (deliver_to_shards_batch((uint8_t)lv,
                                    &desc->shard[lv],
                                    &ar,
                                    1,
                                    desc->sink,
                                    desc->config.shard_alignment,
                                    &sink_bytes))
          return 1;
      }
    }
  }

  return 0;
}

// ---- Scatter epoch (multiscale) ----

static int
scatter_epoch(struct multiarray_tile_stream_cpu* ms,
              struct array_descriptor* desc,
              uint32_t epoch_in_batch,
              uint32_t* out_mask)
{
  const size_t bpe = dtype_bpe(desc->config.dtype);
  void* epoch_pool =
    (char*)ms->chunk_pool + (uint64_t)epoch_in_batch *
                              desc->levels.total_chunks *
                              desc->layout.chunk_stride * bpe;

  if (!desc->levels.enable_multiscale) {
    *out_mask = 1;
    return 0;
  }

  CHECK(Error,
        lod_cpu_gather(&desc->cl.plan,
                       ms->linear,
                       ms->lod_values,
                       ms->scatter_lut,
                       ms->scatter_batch_offsets,
                       desc->config.dtype) == 0);

  CHECK(Error,
        lod_cpu_reduce(&desc->cl.plan,
                       ms->lod_values,
                       desc->config.dtype,
                       desc->config.reduce_method) == 0);

  uint32_t active_levels_mask = 1; // L0 always active

  if (desc->levels.dim0_downsample && desc->dim0_accum) {
    CHECK(Error,
          lod_cpu_dim0_fold(&desc->cl.plan,
                            ms->lod_values,
                            desc->dim0_accum,
                            desc->dim0_counts,
                            desc->config.dtype,
                            desc->config.dim0_reduce_method) == 0);

    for (int lv = 1; lv < desc->cl.plan.nlod; ++lv) {
      desc->dim0_counts[lv]++;
      uint32_t period = 1u << lv;
      if (desc->dim0_counts[lv] >= period) {
        CHECK(Error,
              lod_cpu_dim0_emit(&desc->cl.plan,
                                ms->lod_values,
                                desc->dim0_accum,
                                lv,
                                desc->dim0_counts[lv],
                                desc->config.dtype,
                                desc->config.dim0_reduce_method) == 0);
        desc->dim0_counts[lv] = 0;
        active_levels_mask |= (1u << lv);
      }
    }
  }

  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    const struct tile_stream_layout* layout = &desc->cl.layouts[lv];
    CHECK(Error,
          lod_cpu_morton_to_chunks(&desc->cl.plan,
                                   ms->lod_values,
                                   epoch_pool,
                                   lv,
                                   layout,
                                   ms->morton_lut[lv],
                                   ms->lod_batch_offsets[lv],
                                   desc->config.dtype) == 0);
  }

  *out_mask = active_levels_mask;
  return 0;

Error:
  return 1;
}

// ---- Writer: update ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data)
{
  struct multiarray_tile_stream_cpu* ms =
    container_of(self, struct multiarray_tile_stream_cpu, writer);

  // Bounds check.
  if (array_index < 0 || array_index >= ms->n_arrays)
    return (struct multiarray_writer_result){
      .error = multiarray_writer_fail,
      .rest = data,
    };

  struct array_descriptor* desc = &ms->arrays[array_index];

  // Switch arrays if needed.
  if (array_index != ms->active) {
    if (ms->active >= 0) {
      struct array_descriptor* departing = &ms->arrays[ms->active];

      // Reject switch if departing cursor not at epoch boundary.
      if (departing->cursor % departing->layout.epoch_elements != 0)
        return (struct multiarray_writer_result){
          .error = multiarray_writer_not_flushable,
          .rest = data,
        };

      // Flush departing batch if any epochs accumulated.
      if (departing->batch_accumulated > 0) {
        if (flush_batch(ms,
                        departing,
                        departing->batch_accumulated,
                        departing->batch_active_masks))
          return (struct multiarray_writer_result){
            .error = multiarray_writer_fail,
            .rest = data,
          };
        departing->batch_accumulated = 0;
      }
    }

    ms->active = array_index;
    recompute_luts(ms, array_index);

    // Zero chunk pool if not fully covered.
    if (!desc->pool_fully_covered)
      memset(ms->chunk_pool, 0, ms->chunk_pool_bytes);

    // Zero lod_values if multiscale.
    if (desc->levels.enable_multiscale && ms->lod_values)
      memset(ms->lod_values, 0, ms->lod_values_bytes);
  }

  const size_t bpe = dtype_bpe(desc->config.dtype);
  const uint8_t* src = (const uint8_t*)data.beg;
  const uint8_t* end = (const uint8_t*)data.end;
  const uint64_t max_cursor = desc->max_cursor;

  while (src < end) {
    if (max_cursor > 0 && desc->cursor >= max_cursor) {
      // Flush this array completely.
      if (desc->batch_accumulated > 0) {
        if (flush_batch(ms,
                        desc,
                        desc->batch_accumulated,
                        desc->batch_active_masks))
          return (struct multiarray_writer_result){
            .error = multiarray_writer_fail,
            .rest = { .beg = src, .end = end },
          };
        desc->batch_accumulated = 0;
      }
      return (struct multiarray_writer_result){
        .error = multiarray_writer_finished,
        .rest = { .beg = src, .end = end },
      };
    }

    const uint64_t epoch_remaining =
      desc->layout.epoch_elements -
      (desc->cursor % desc->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    uint64_t elements =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    if (max_cursor > 0) {
      uint64_t cap = max_cursor - desc->cursor;
      if (elements > cap)
        elements = cap;
    }

    const uint64_t bytes = elements * bpe;

    // Scatter into pool (or LOD linear buffer for multiscale).
    if (desc->levels.enable_multiscale) {
      uint64_t epoch_offset = desc->cursor % desc->layout.epoch_elements;
      memcpy((char*)ms->linear + epoch_offset * bpe, src, bytes);
    } else {
      void* epoch_pool =
        (char*)ms->chunk_pool + (uint64_t)desc->batch_accumulated *
                                  desc->levels.total_chunks *
                                  desc->layout.chunk_stride * bpe;
      if (transpose_cpu(epoch_pool,
                        src,
                        bytes,
                        (uint8_t)bpe,
                        desc->cursor,
                        desc->layout.lifted_rank,
                        desc->layout.lifted_shape,
                        desc->layout.lifted_strides))
        return (struct multiarray_writer_result){
          .error = multiarray_writer_fail,
          .rest = { .beg = src, .end = end },
        };
    }

    desc->cursor += elements;
    src += bytes;

    // Epoch boundary.
    if (desc->cursor % desc->layout.epoch_elements == 0 &&
        desc->cursor > 0) {
      uint32_t active_mask = 1;
      if (desc->levels.enable_multiscale) {
        if (scatter_epoch(ms, desc, desc->batch_accumulated, &active_mask))
          return (struct multiarray_writer_result){
            .error = multiarray_writer_fail,
            .rest = { .beg = src, .end = end },
          };
      }

      desc->batch_active_masks[desc->batch_accumulated] = active_mask;
      desc->batch_accumulated++;

      if (desc->batch_accumulated == desc->cl.epochs_per_batch) {
        if (flush_batch(
              ms, desc, desc->batch_accumulated, desc->batch_active_masks))
          return (struct multiarray_writer_result){
            .error = multiarray_writer_fail,
            .rest = { .beg = src, .end = end },
          };
        desc->batch_accumulated = 0;

        if (!desc->pool_fully_covered)
          memset(ms->chunk_pool,
                 0,
                 (uint64_t)desc->cl.epochs_per_batch *
                   desc->levels.total_chunks *
                   desc->layout.chunk_stride * bpe);
      }

      if (desc->levels.enable_multiscale && ms->lod_values) {
        size_t lod_bytes =
          desc->cl.plan.levels.ends[desc->cl.plan.nlod - 1] * bpe;
        memset(ms->lod_values, 0, lod_bytes);
      }
    }
  }

  return (struct multiarray_writer_result){
    .error = multiarray_writer_ok,
    .rest = { .beg = src, .end = end },
  };
}

// ---- Writer: flush ----

static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_cpu* ms =
    container_of(self, struct multiarray_tile_stream_cpu, writer);

  // For the currently active array: handle partial epoch.
  if (ms->active >= 0) {
    struct array_descriptor* desc = &ms->arrays[ms->active];
    if (desc->cursor % desc->layout.epoch_elements != 0) {
      uint32_t active_mask = 1;
      if (desc->levels.enable_multiscale) {
        if (scatter_epoch(ms, desc, desc->batch_accumulated, &active_mask))
          goto Error;
      }
      desc->batch_active_masks[desc->batch_accumulated] = active_mask;
      desc->batch_accumulated++;
    }
  }

  // Flush batch for each array that has accumulated epochs.
  for (int i = 0; i < ms->n_arrays; ++i) {
    struct array_descriptor* desc = &ms->arrays[i];
    if (desc->batch_accumulated == 0)
      continue;

    if (i != ms->active) {
      ms->active = i;
      recompute_luts(ms, i);

      if (!desc->pool_fully_covered)
        memset(ms->chunk_pool, 0, ms->chunk_pool_bytes);
      if (desc->levels.enable_multiscale && ms->lod_values)
        memset(ms->lod_values, 0, ms->lod_values_bytes);
    }

    if (flush_batch(ms, desc, desc->batch_accumulated,
                    desc->batch_active_masks))
      goto Error;
    desc->batch_accumulated = 0;
  }

  // Drain partial dim0 accumulators (multiscale dim0 downsample).
  for (int i = 0; i < ms->n_arrays; ++i) {
    struct array_descriptor* desc = &ms->arrays[i];
    if (!desc->levels.dim0_downsample || !desc->dim0_accum)
      continue;

    uint32_t drain_mask = 0;

    for (int lv = 1; lv < desc->cl.plan.nlod; ++lv) {
      if (desc->dim0_counts[lv] > 0) {
        if (i != ms->active) {
          ms->active = i;
          recompute_luts(ms, i);
        }

        if (lod_cpu_dim0_emit(&desc->cl.plan,
                              ms->lod_values,
                              desc->dim0_accum,
                              lv,
                              desc->dim0_counts[lv],
                              desc->config.dtype,
                              desc->config.dim0_reduce_method))
          goto Error;
        desc->dim0_counts[lv] = 0;

        const struct tile_stream_layout* layout_lv = &desc->cl.layouts[lv];
        if (lod_cpu_morton_to_chunks(&desc->cl.plan,
                                     ms->lod_values,
                                     ms->chunk_pool,
                                     lv,
                                     layout_lv,
                                     ms->morton_lut[lv],
                                     ms->lod_batch_offsets[lv],
                                     desc->config.dtype))
          goto Error;
        drain_mask |= (1u << lv);
      }
    }

    if (drain_mask) {
      desc->batch_active_masks[0] = drain_mask;
      if (flush_batch(ms, desc, 1, desc->batch_active_masks))
        goto Error;
    }
  }

  // Finalize partial shards + update dim0.
  for (int i = 0; i < ms->n_arrays; ++i) {
    struct array_descriptor* desc = &ms->arrays[i];

    for (int lv = 0; lv < desc->levels.nlod; ++lv) {
      if (desc->shard[lv].epoch_in_shard > 0) {
        if (finalize_shards(&desc->shard[lv], desc->config.shard_alignment))
          goto Error;
      }
    }

    if (desc->sink->update_dim0) {
      const struct dimension* dims = desc->config.dimensions;
      for (int lv = 0; lv < desc->levels.nlod; ++lv) {
        struct shard_state* ss = &desc->shard[lv];
        uint64_t d0c =
          ss->shard_epoch * ss->chunks_per_shard_0 + ss->epoch_in_shard;
        if (desc->sink->update_dim0(
              desc->sink, (uint8_t)lv, d0c * dims[0].chunk_size))
          goto Error;
      }
    }
  }

  return (struct multiarray_writer_result){
    .error = multiarray_writer_ok,
  };

Error:
  return (struct multiarray_writer_result){
    .error = multiarray_writer_fail,
  };
}

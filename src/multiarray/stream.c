#include "multiarray.cpu.h"
#include "cpu/pipeline.h"
#include "stream/config.h"
#include "zarr/shard_delivery.h"

#include "cpu/compress.h"
#include "cpu/transpose.h"
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
  uint32_t batch_active_count[LOD_MAX_LEVELS];
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
  size_t* shard_order_sizes;

  // Shared LUT storage (recomputed on switch).
  uint32_t* chunk_to_shard_map[LOD_MAX_LEVELS];
  uint32_t* batch_gather[LOD_MAX_LEVELS];
  uint32_t* batch_chunk_to_shard_map[LOD_MAX_LEVELS];
  uint32_t* scatter_lut;
  uint64_t* scatter_batch_offsets;
  uint32_t* morton_lut[LOD_MAX_LEVELS];
  uint64_t* lod_batch_offsets[LOD_MAX_LEVELS];

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
  size_t max_chunk_to_shard_map_len[LOD_MAX_LEVELS] = { 0 };
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
      d->batch_active_count[lv] = K_l;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = al->covering_count;
      uint64_t M_lv = al->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      // Track per-level maxima for shared LUTs.
      max_chunk_to_shard_map_len[lv] = max_sz(max_chunk_to_shard_map_len[lv], M_lv);

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
      ms->shard_order_sizes = (size_t*)calloc(max_batch_C, sizeof(size_t));
      CHECK(Fail, ms->shard_order_sizes);
    }
  }

  // Shared LUT storage.
  for (int lv = 0; lv < LOD_MAX_LEVELS; ++lv) {
    if (max_chunk_to_shard_map_len[lv] > 0) {
      ms->chunk_to_shard_map[lv] =
        (uint32_t*)malloc(max_chunk_to_shard_map_len[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->chunk_to_shard_map[lv]);
    }
    if (max_batch_gather[lv] > 0) {
      ms->batch_gather[lv] =
        (uint32_t*)malloc(max_batch_gather[lv] * sizeof(uint32_t));
      ms->batch_chunk_to_shard_map[lv] =
        (uint32_t*)malloc(max_batch_gather[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->batch_gather[lv] && ms->batch_chunk_to_shard_map[lv]);
    }
    if (max_morton_lut[lv] > 0) {
      ms->morton_lut[lv] =
        (uint32_t*)malloc(max_morton_lut[lv] * sizeof(uint32_t));
      CHECK(Fail, ms->morton_lut[lv]);
    }
    if (max_lod_batch_offsets[lv] > 0) {
      ms->lod_batch_offsets[lv] =
        (uint64_t*)calloc(max_lod_batch_offsets[lv], sizeof(uint64_t));
      CHECK(Fail, ms->lod_batch_offsets[lv]);
    }
  }

  if (max_scatter_lut > 0) {
    ms->scatter_lut =
      (uint32_t*)malloc(max_scatter_lut * sizeof(uint32_t));
    CHECK(Fail, ms->scatter_lut);
  }
  if (max_scatter_batch_offsets > 0) {
    ms->scatter_batch_offsets =
      (uint64_t*)calloc(max_scatter_batch_offsets, sizeof(uint64_t));
    CHECK(Fail, ms->scatter_batch_offsets);
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
    free(ms->chunk_to_shard_map[lv]);
    free(ms->batch_gather[lv]);
    free(ms->batch_chunk_to_shard_map[lv]);
    free(ms->morton_lut[lv]);
    free(ms->lod_batch_offsets[lv]);
  }
  free(ms->shard_order_sizes);
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
  struct lut_targets luts = {
    .scatter_lut = ms->scatter_lut,
    .scatter_batch_offsets = ms->scatter_batch_offsets,
  };
  for (int lv = 0; lv < d->levels.nlod; ++lv) {
    luts.chunk_to_shard_map[lv] = ms->chunk_to_shard_map[lv];
    luts.batch_gather[lv] = ms->batch_gather[lv];
    luts.batch_chunk_to_shard_map[lv] = ms->batch_chunk_to_shard_map[lv];
    luts.morton_lut[lv] = ms->morton_lut[lv];
    luts.lod_batch_offsets[lv] = ms->lod_batch_offsets[lv];
  }
  cpu_pipeline_compute_luts(
    &d->cl, &d->levels, d->batch_active_count, d->agg_layout, &luts);
}

// ---- Pipeline helpers ----

static struct flush_batch_params
make_flush_params(struct multiarray_tile_stream_cpu* ms,
                  struct array_descriptor* desc)
{
  const size_t bpe = dtype_bpe(desc->config.dtype);
  struct flush_batch_params p = {
    .codec = desc->config.codec,
    .chunk_pool = ms->chunk_pool,
    .chunk_stride_bytes = desc->layout.chunk_stride * bpe,
    .chunk_bytes = desc->layout.chunk_elements * bpe,
    .compressed = ms->compressed,
    .max_output_size = desc->cl.max_output_size,
    .comp_sizes = ms->comp_sizes,
    .total_chunks = desc->levels.total_chunks,
    .nlod = desc->levels.nlod,
    .shard_order_sizes = ms->shard_order_sizes,
    .sink = desc->sink,
    .shard_alignment = desc->config.shard_alignment,
    .metrics = NULL, // multiarray skips timing
  };
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    p.levels[lv] = (struct flush_level_view){
      .agg_layout = &desc->agg_layout[lv],
      .batch_active_count = desc->batch_active_count[lv],
      .chunk_offset = desc->levels.chunk_offset[lv],
      .chunk_to_shard_map = ms->chunk_to_shard_map[lv],
      .batch_chunk_to_shard_map = ms->batch_chunk_to_shard_map[lv],
      .batch_gather = ms->batch_gather[lv],
      .agg_slot = &ms->agg_slots[lv],
      .shard = &desc->shard[lv],
    };
  }
  return p;
}

static struct scatter_epoch_params
make_scatter_params(struct multiarray_tile_stream_cpu* ms,
                    struct array_descriptor* desc)
{
  struct scatter_epoch_params p = {
    .dtype = desc->config.dtype,
    .reduce_method = desc->config.reduce_method,
    .dim0_reduce_method = desc->config.dim0_reduce_method,
    .cl = &desc->cl,
    .chunk_pool = ms->chunk_pool,
    .linear = ms->linear,
    .lod_values = ms->lod_values,
    .scatter_lut = ms->scatter_lut,
    .scatter_batch_offsets = ms->scatter_batch_offsets,
    .dim0_accum = desc->dim0_accum,
    .dim0_counts = desc->dim0_counts,
    .metrics = NULL, // multiarray skips timing
  };
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    p.morton_lut[lv] = ms->morton_lut[lv];
    p.lod_batch_offsets[lv] = ms->lod_batch_offsets[lv];
  }
  return p;
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
        struct flush_batch_params fp = make_flush_params(ms, departing);
        if (cpu_pipeline_flush_batch(
              &fp, departing->batch_accumulated, departing->batch_active_masks))
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
        struct flush_batch_params fp = make_flush_params(ms, desc);
        if (cpu_pipeline_flush_batch(
              &fp, desc->batch_accumulated, desc->batch_active_masks))
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
        struct scatter_epoch_params sp = make_scatter_params(ms, desc);
        if (cpu_pipeline_scatter_epoch(
              &sp, desc->batch_accumulated, &active_mask))
          return (struct multiarray_writer_result){
            .error = multiarray_writer_fail,
            .rest = { .beg = src, .end = end },
          };
      }

      desc->batch_active_masks[desc->batch_accumulated] = active_mask;
      desc->batch_accumulated++;

      if (desc->batch_accumulated == desc->cl.epochs_per_batch) {
        struct flush_batch_params fp = make_flush_params(ms, desc);
        if (cpu_pipeline_flush_batch(
              &fp, desc->batch_accumulated, desc->batch_active_masks))
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
        struct scatter_epoch_params sp = make_scatter_params(ms, desc);
        if (cpu_pipeline_scatter_epoch(
              &sp, desc->batch_accumulated, &active_mask))
          goto Error;
      }
      desc->batch_active_masks[desc->batch_accumulated] = active_mask;
      desc->batch_accumulated++;
    }
  }

  // Flush the active array's batch (if any).
  if (ms->active >= 0) {
    struct array_descriptor* desc = &ms->arrays[ms->active];
    if (desc->batch_accumulated > 0) {
      struct flush_batch_params fp = make_flush_params(ms, desc);
      if (cpu_pipeline_flush_batch(
            &fp, desc->batch_accumulated, desc->batch_active_masks))
        goto Error;
      desc->batch_accumulated = 0;
    }
  }

  // Drain partial dim0 accumulators (multiscale dim0 downsample).
  for (int i = 0; i < ms->n_arrays; ++i) {
    struct array_descriptor* desc = &ms->arrays[i];
    if (!desc->levels.dim0_downsample || !desc->dim0_accum)
      continue;

    // Ensure LUTs are current for this array.
    if (i != ms->active) {
      ms->active = i;
      recompute_luts(ms, i);
    }

    struct dim0_drain_params dp = {
      .cl = &desc->cl,
      .dtype = desc->config.dtype,
      .dim0_reduce_method = desc->config.dim0_reduce_method,
      .lod_values = ms->lod_values,
      .dim0_accum = desc->dim0_accum,
      .dim0_counts = desc->dim0_counts,
      .chunk_pool = ms->chunk_pool,
      .metrics = NULL,
    };
    for (int lv = 0; lv < desc->levels.nlod; ++lv) {
      dp.morton_lut[lv] = ms->morton_lut[lv];
      dp.lod_batch_offsets[lv] = ms->lod_batch_offsets[lv];
    }

    uint32_t drain_mask = 0;
    if (cpu_pipeline_dim0_drain(&dp, &drain_mask))
      goto Error;

    if (drain_mask) {
      desc->batch_active_masks[0] = drain_mask;
      struct flush_batch_params fp = make_flush_params(ms, desc);
      if (cpu_pipeline_flush_batch(&fp, 1, desc->batch_active_masks))
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

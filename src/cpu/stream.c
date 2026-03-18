#include "stream.internal.h"

#include "aggregate.h"
#include "compress.h"
#include "index.ops.h"
#include "lod.h"
#include "platform.h"
#include "prelude.h"
#include "transpose.h"

#include <stdlib.h>
#include <string.h>

// ---- Forward declarations ----

static struct writer_result
cpu_append(struct writer* self, struct slice input);
static struct writer_result
cpu_flush(struct writer* self);

// ---- Create / Destroy ----

struct tile_stream_cpu*
tile_stream_cpu_create(const struct tile_stream_configuration* config)
{
  if (!config || !config->shard_sink)
    return NULL;
  if (config->dtype == lod_dtype_f16)
    return NULL;

  struct tile_stream_cpu* s =
    (struct tile_stream_cpu*)calloc(1, sizeof(*s));
  if (!s)
    return NULL;

  s->config = *config;

  // CPU codec alignment is 1 (no nvcomp alignment needed).
  if (compute_stream_layouts(
        config, 1, compress_cpu_max_output_size, &s->cl))
    goto Fail;

  s->layout = s->cl.l0;
  s->levels = s->cl.levels;

  // Force K=1 on CPU.
  s->cl.epochs_per_batch = 1;

  const size_t bpe = lod_dtype_bpe(config->dtype);
  const uint64_t total_chunks = s->levels.total_chunks;
  const size_t chunk_stride_bytes = s->layout.chunk_stride * bpe;
  const size_t max_out = s->cl.max_output_size;

  // Chunk pool: one epoch's worth across all levels.
  s->chunk_pool = calloc(total_chunks, chunk_stride_bytes);
  CHECK(Fail, s->chunk_pool);

  s->pool_fully_covered =
    (s->layout.chunk_stride == s->layout.chunk_elements);

  // Compressed output buffer.
  s->compressed = malloc(total_chunks * max_out);
  CHECK(Fail, s->compressed);

  s->comp_sizes = (size_t*)calloc(total_chunks, sizeof(size_t));
  CHECK(Fail, s->comp_sizes);

  // Per-level shard + aggregate state.
  {
    uint64_t max_C = 0;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct level_layout_info* li = &s->cl.per_level[lv];
      s->agg_layout[lv] = li->agg_layout;
      const struct aggregate_layout* al = &s->agg_layout[lv];

      if (al->covering_count > max_C)
        max_C = al->covering_count;

      // Per-level perm.
      s->agg_perm[lv] =
        (uint32_t*)malloc(al->chunks_per_epoch * sizeof(uint32_t));
      CHECK(Fail, s->agg_perm[lv]);
      for (uint64_t i = 0; i < al->chunks_per_epoch; ++i)
        s->agg_perm[lv][i] =
          (uint32_t)ravel(al->lifted_rank,
                          al->lifted_shape, al->lifted_strides, i);

      // Per-level aggregate output buffers.
      uint64_t C_lv = al->covering_count;
      size_t data_lv = agg_pool_bytes(al->chunks_per_epoch,
                                       al->max_comp_chunk_bytes,
                                       C_lv,
                                       al->cps_inner, al->page_size);
      {
        struct cpu_agg_slot* as = &s->agg_slots[lv];
        if (C_lv > 0) {
          as->offsets = (size_t*)malloc((C_lv + 1) * sizeof(size_t));
          as->chunk_sizes = (size_t*)calloc(C_lv, sizeof(size_t));
          CHECK(Fail, as->offsets && as->chunk_sizes);
        }
        if (data_lv > 0) {
          as->data = malloc(data_lv);
          as->data_capacity = data_lv;
          CHECK(Fail, as->data);
        }
      }

      struct shard_state* ss = &s->shard[lv];
      ss->chunks_per_shard_0 = li->chunks_per_shard_0;
      ss->chunks_per_shard_inner = li->chunks_per_shard_inner;
      ss->chunks_per_shard_total = li->chunks_per_shard_total;
      ss->shard_inner_count = li->shard_inner_count;
      ss->shards =
        (struct active_shard*)calloc(li->shard_inner_count, sizeof(struct active_shard));
      CHECK(Fail, ss->shards);
      for (uint64_t si = 0; si < li->shard_inner_count; ++si) {
        ss->shards[si].index =
          (uint64_t*)malloc(li->chunks_per_shard_total * 2 * sizeof(uint64_t));
        CHECK(Fail, ss->shards[si].index);
        memset(ss->shards[si].index,
               0xFF,
               li->chunks_per_shard_total * 2 * sizeof(uint64_t));
      }
    }

    // Shared permuted_sizes scratch (sized to max C).
    if (max_C > 0) {
      s->agg_permuted_sizes = (size_t*)calloc(max_C, sizeof(size_t));
      CHECK(Fail, s->agg_permuted_sizes);
    }
  }

  // LOD buffers (multiscale only).
  if (s->levels.enable_multiscale) {
    // Linear epoch buffer: input is accumulated here before LOD scatter.
    s->linear = calloc(s->layout.epoch_elements, bpe);
    CHECK(Fail, s->linear);

    uint64_t total_lod_elements =
      s->cl.plan.levels.ends[s->cl.plan.nlod - 1];
    s->lod_values = calloc(total_lod_elements, bpe);
    CHECK(Fail, s->lod_values);

    // Dim0 accumulator: total elements in levels 1+ (packed).
    if (s->levels.dim0_downsample) {
      uint64_t dim0_total = 0;
      for (int lv = 1; lv < s->cl.plan.nlod; ++lv)
        dim0_total += s->cl.plan.batch_count * s->cl.plan.lod_counts[lv];
      if (dim0_total > 0) {
        s->dim0_accum = calloc(dim0_total, bpe);
        CHECK(Fail, s->dim0_accum);
      }
      memset(s->dim0_counts, 0, sizeof(s->dim0_counts));
    }

    // Precompute L0 scatter LUT + batch offsets for gather.
    {
      const struct lod_plan* plan = &s->cl.plan;
      s->scatter_lut =
        (uint32_t*)malloc(plan->lod_counts[0] * sizeof(uint32_t));
      CHECK(Fail, s->scatter_lut);
      lod_cpu_build_scatter_lut(plan, s->scatter_lut);

      s->scatter_batch_offsets =
        (uint64_t*)calloc(plan->batch_count, sizeof(uint64_t));
      CHECK(Fail, s->scatter_batch_offsets);
      lod_cpu_build_scatter_batch_offsets(plan, s->scatter_batch_offsets);
    }

    // Precompute morton-to-chunk LUTs and batch offsets for each level.
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct lod_plan* plan = &s->cl.plan;
      const struct tile_stream_layout* layout_lv =
        (lv == 0) ? &s->layout : &s->cl.lod_layouts[lv];
      uint64_t lod_count = plan->lod_counts[lv];

      s->morton_lut[lv] = (uint32_t*)malloc(lod_count * sizeof(uint32_t));
      CHECK(Fail, s->morton_lut[lv]);
      lod_cpu_build_chunk_lut(plan, lv, layout_lv, s->morton_lut[lv]);

      s->lod_batch_offsets[lv] =
        (uint64_t*)calloc(plan->batch_count, sizeof(uint64_t));
      CHECK(Fail, s->lod_batch_offsets[lv]);
      for (uint64_t bi = 0; bi < plan->batch_count; ++bi) {
        uint64_t remainder = bi;
        int64_t offset = 0;
        for (int k = plan->batch_ndim - 1; k >= 0; --k) {
          uint64_t coord = remainder % plan->batch_shape[k];
          remainder /= plan->batch_shape[k];
          int d = plan->batch_map[k];
          uint64_t cs = layout_lv->lifted_shape[2 * d + 1];
          uint64_t ci = coord / cs;
          uint64_t wi = coord % cs;
          offset += (int64_t)ci * layout_lv->lifted_strides[2 * d];
          offset += (int64_t)wi * layout_lv->lifted_strides[2 * d + 1];
        }
        s->lod_batch_offsets[lv][bi] =
          (uint64_t)offset +
          s->levels.chunk_offset[lv] * layout_lv->chunk_stride;
      }
    }
  }

  // Metrics.
  s->metrics.memcpy = mk_stream_metric("memcpy");
  s->metrics.scatter = mk_stream_metric("scatter");
  s->metrics.compress = mk_stream_metric("compress");
  s->metrics.aggregate = mk_stream_metric("aggregate");
  s->metrics.sink = mk_stream_metric("sink");
  if (s->levels.enable_multiscale) {
    s->metrics.lod_gather = mk_stream_metric("lod_gather");
    s->metrics.lod_reduce = mk_stream_metric("lod_reduce");
    s->metrics.lod_dim0_fold = mk_stream_metric("lod_dim0_fold");
    s->metrics.lod_morton_chunk = mk_stream_metric("lod_morton");
  }

  s->writer.append = cpu_append;
  s->writer.flush = cpu_flush;

  platform_toc(&s->metadata_update_clock);

  return s;

Fail:
  tile_stream_cpu_destroy(s);
  return NULL;
}

void
tile_stream_cpu_destroy(struct tile_stream_cpu* s)
{
  if (!s)
    return;

  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->shard[lv];
    if (ss->shards) {
      for (uint64_t si = 0; si < ss->shard_inner_count; ++si)
        free(ss->shards[si].index);
      free(ss->shards);
    }
    free(s->agg_perm[lv]);
    free(s->morton_lut[lv]);
    free(s->lod_batch_offsets[lv]);
  }

  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct cpu_agg_slot* as = &s->agg_slots[lv];
    free(as->data);
    free(as->offsets);
    free(as->chunk_sizes);
  }
  free(s->agg_permuted_sizes);

  free(s->chunk_pool);
  free(s->compressed);
  free(s->comp_sizes);
  free(s->scatter_lut);
  free(s->scatter_batch_offsets);
  free(s->linear);
  free(s->lod_values);
  free(s->dim0_accum);
  computed_stream_layouts_free(&s->cl);
  free(s);
}

// ---- Accessors ----

struct stream_metrics
tile_stream_cpu_get_metrics(const struct tile_stream_cpu* s)
{
  return s->metrics;
}

const struct tile_stream_layout*
tile_stream_cpu_layout(const struct tile_stream_cpu* s)
{
  return &s->layout;
}

struct writer*
tile_stream_cpu_writer(struct tile_stream_cpu* s)
{
  return &s->writer;
}

uint64_t
tile_stream_cpu_cursor(const struct tile_stream_cpu* s)
{
  return s->cursor;
}

// ---- Memory estimate ----

int
tile_stream_cpu_memory_estimate(
  const struct tile_stream_configuration* config,
  struct tile_stream_cpu_memory_info* info)
{
  if (!info)
    return 1;

  memset(info, 0, sizeof(*info));

  struct computed_stream_layouts cl;
  if (compute_stream_layouts(
        config, 1, compress_cpu_max_output_size, &cl))
    return 1;

  // Force K=1 on CPU.
  cl.epochs_per_batch = 1;

  const size_t bpe = lod_dtype_bpe(config->dtype);
  const uint64_t total_chunks = cl.levels.total_chunks;
  const size_t chunk_stride_bytes = cl.l0.chunk_stride * bpe;
  const size_t max_out = cl.max_output_size;

  info->chunk_pool_bytes = total_chunks * chunk_stride_bytes;
  info->compressed_pool_bytes = total_chunks * max_out;
  info->comp_sizes_bytes = total_chunks * sizeof(size_t);

  // Aggregate: double-buffered slots + per-level perm + shared scratch.
  {
    size_t agg = 0;
    uint64_t max_C = 0;
    for (int lv = 0; lv < cl.levels.nlod; ++lv) {
      const struct aggregate_layout* al = &cl.per_level[lv].agg_layout;
      if (al->covering_count > max_C)
        max_C = al->covering_count;
      uint64_t C_lv = al->covering_count;
      size_t data_lv = agg_pool_bytes(al->chunks_per_epoch,
                                       al->max_comp_chunk_bytes,
                                       C_lv,
                                       al->cps_inner, al->page_size);

      // Per-level perm (1x).
      agg += al->chunks_per_epoch * sizeof(uint32_t);

      // Per-level: data + offsets + chunk_sizes (1x).
      agg += data_lv +
             (C_lv + 1) * sizeof(size_t) +
             C_lv * sizeof(size_t);
    }
    // Shared permuted_sizes scratch (1x max_C).
    if (max_C > 0)
      agg += max_C * sizeof(size_t);
    info->aggregate_bytes = agg;
  }

  // LOD buffers (multiscale only).
  {
    size_t lod = 0;
    if (cl.levels.enable_multiscale) {
      lod += cl.l0.epoch_elements * bpe; // linear
      uint64_t total_lod_elements =
        cl.plan.levels.ends[cl.plan.nlod - 1];
      lod += total_lod_elements * bpe; // lod_values

      // Dim0 accumulator.
      if (cl.levels.dim0_downsample) {
        uint64_t dim0_total = 0;
        for (int lv = 1; lv < cl.plan.nlod; ++lv)
          dim0_total += cl.plan.batch_count * cl.plan.lod_counts[lv];
        lod += dim0_total * bpe;
      }

      // Scatter LUT + batch offsets for L0 gather.
      lod += cl.plan.lod_counts[0] * sizeof(uint32_t);        // scatter_lut
      lod += cl.plan.batch_count * sizeof(uint64_t);           // scatter_batch_offsets

      // Morton-to-chunk LUTs + batch offsets per level.
      for (int lv = 0; lv < cl.levels.nlod; ++lv) {
        lod += cl.plan.lod_counts[lv] * sizeof(uint32_t);     // morton_lut
        lod += cl.plan.batch_count * sizeof(uint64_t);         // batch_offsets
      }
    }
    info->lod_bytes = lod;
  }

  // Shard state: active_shard arrays + index buffers.
  {
    size_t shards = 0;
    for (int lv = 0; lv < cl.levels.nlod; ++lv) {
      const struct level_layout_info* li = &cl.per_level[lv];
      shards += li->shard_inner_count * sizeof(struct active_shard);
      shards += li->shard_inner_count * li->chunks_per_shard_total *
                2 * sizeof(uint64_t);
    }
    info->shard_bytes = shards;
  }

  info->heap_bytes = sizeof(struct tile_stream_cpu) +
                     info->chunk_pool_bytes +
                     info->compressed_pool_bytes +
                     info->comp_sizes_bytes +
                     info->aggregate_bytes +
                     info->lod_bytes +
                     info->shard_bytes;

  info->chunks_per_epoch = cl.l0.chunks_per_epoch;
  info->total_chunks = total_chunks;
  info->max_output_size = max_out;
  info->nlod = cl.levels.nlod;

  computed_stream_layouts_free(&cl);
  return 0;
}

// ---- Epoch processing ----

// Compress all chunks, then aggregate + deliver per active level.
static int
flush_epoch(struct tile_stream_cpu* s, uint32_t active_levels_mask)
{
  const size_t bpe = lod_dtype_bpe(s->config.dtype);
  const size_t max_out = s->cl.max_output_size;

  // Compress all chunks in one batch (all levels contiguous in pool).
  {
    struct platform_clock clk = { 0 };
    platform_toc(&clk);

    if (compress_cpu(s->config.codec,
                     s->chunk_pool,
                     s->layout.chunk_stride * bpe,
                     s->compressed,
                     max_out,
                     s->comp_sizes,
                     s->layout.chunk_stride * bpe,
                     s->levels.total_chunks)) {
      return 1;
    }

    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(
      &s->metrics.compress, ms,
      s->levels.total_chunks * s->layout.chunk_stride * bpe);
  }

  // Aggregate per active level.
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    uint64_t chunk_offset = s->levels.chunk_offset[lv];

    struct platform_clock clk = { 0 };
    platform_toc(&clk);

    const void* comp_lv =
      (const char*)s->compressed + chunk_offset * max_out;
    const size_t* sizes_lv = s->comp_sizes + chunk_offset;

    struct aggregate_cpu_workspace ws = {
      .perm = s->agg_perm[lv],
      .permuted_sizes = s->agg_permuted_sizes,
      .data = s->agg_slots[lv].data,
      .data_capacity = s->agg_slots[lv].data_capacity,
      .offsets = s->agg_slots[lv].offsets,
      .chunk_sizes = s->agg_slots[lv].chunk_sizes,
    };
    struct aggregate_result ar;
    if (aggregate_cpu_into(
          comp_lv, sizes_lv, &s->agg_layout[lv], &ws, &ar)) {
      return 1;
    }

    float ms = (float)(platform_toc(&clk) * 1000.0);
    size_t agg_bytes = ar.offsets[s->agg_layout[lv].covering_count];
    accumulate_metric_ms(&s->metrics.aggregate, ms, agg_bytes);
  }

  // Deliver to shards.
  {
    struct platform_clock clk = { 0 };
    platform_toc(&clk);
    size_t total_sink_bytes = 0;

    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      if (!(active_levels_mask & (1u << lv)))
        continue;

      struct aggregate_result ar = {
        .data = s->agg_slots[lv].data,
        .offsets = s->agg_slots[lv].offsets,
        .chunk_sizes = s->agg_slots[lv].chunk_sizes,
      };

      size_t sink_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  &s->shard[lv],
                                  &ar,
                                  1,
                                  s->config.shard_sink,
                                  s->config.shard_alignment,
                                  &sink_bytes)) {
        return 1;
      }
      total_sink_bytes += sink_bytes;
    }

    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&s->metrics.sink, ms, total_sink_bytes);
  }

  return 0;
}

// Scatter one epoch into chunk pool. Returns active_levels_mask via out_mask.
static int
scatter_epoch(struct tile_stream_cpu* s, uint32_t* out_mask)
{
  const size_t bpe = lod_dtype_bpe(s->config.dtype);

  if (!s->levels.enable_multiscale) {
    // Simple path: transpose was already done during append.
    *out_mask = 1;
    return 0;
  }

  // Multiscale path: scatter linear buffer to morton, reduce, then
  // scatter each level to chunk pool.
  struct platform_clock clk = { 0 };
  platform_toc(&clk);

  CHECK(Error,
        lod_cpu_gather(&s->cl.plan, s->linear, s->lod_values,
                       s->scatter_lut, s->scatter_batch_offsets,
                       s->config.dtype) == 0);

  float scatter_ms = (float)(platform_toc(&clk) * 1000.0);
  accumulate_metric_ms(&s->metrics.lod_gather, scatter_ms,
                        s->layout.epoch_elements * bpe);

  platform_toc(&clk);
  CHECK(Error, lod_cpu_reduce(&s->cl.plan, s->lod_values, s->config.dtype,
                               s->config.reduce_method) == 0);

  float ms = (float)(platform_toc(&clk) * 1000.0);
  accumulate_metric_ms(&s->metrics.lod_reduce, ms,
                        s->cl.plan.levels.ends[s->cl.plan.nlod - 1] * bpe);

  // Dim0 fold/emit: accumulate levels 1+ across epochs.
  // Without dim0_downsample, only L0 is scattered to the chunk pool.
  uint32_t active_levels_mask = 1; // L0 always active
  if (s->levels.dim0_downsample && s->dim0_accum) {
    struct platform_clock dim0_clk = { 0 };
    platform_toc(&dim0_clk);

    CHECK(Error,
          lod_cpu_dim0_fold(&s->cl.plan,
                            s->lod_values,
                            s->dim0_accum,
                            s->dim0_counts,
                            s->config.dtype,
                            s->config.dim0_reduce_method) == 0);

    for (int lv = 1; lv < s->cl.plan.nlod; ++lv) {
      s->dim0_counts[lv]++;
      uint32_t period = 1u << lv;
      if (s->dim0_counts[lv] >= period) {
        CHECK(Error,
              lod_cpu_dim0_emit(&s->cl.plan,
                                s->lod_values,
                                s->dim0_accum,
                                lv,
                                s->dim0_counts[lv],
                                s->config.dtype,
                                s->config.dim0_reduce_method) == 0);
        s->dim0_counts[lv] = 0;
        active_levels_mask |= (1u << lv);
      }
    }

    float dim0_ms = (float)(platform_toc(&dim0_clk) * 1000.0);
    size_t dim0_bytes = 0;
    for (int lv = 1; lv < s->cl.plan.nlod; ++lv)
      dim0_bytes += s->cl.plan.batch_count * s->cl.plan.lod_counts[lv] * bpe;
    accumulate_metric_ms(&s->metrics.lod_dim0_fold, dim0_ms, dim0_bytes);
  }

  platform_toc(&clk);
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    const struct tile_stream_layout* layout =
      (lv == 0) ? &s->layout : &s->cl.lod_layouts[lv];

    CHECK(Error,
          lod_cpu_morton_to_chunks(&s->cl.plan,
                                   s->lod_values,
                                   s->chunk_pool,
                                   lv,
                                   layout,
                                   s->morton_lut[lv],
                                   s->lod_batch_offsets[lv],
                                   s->config.dtype) == 0);
  }

  ms = (float)(platform_toc(&clk) * 1000.0);
  accumulate_metric_ms(&s->metrics.lod_morton_chunk, ms,
                        s->levels.total_chunks * s->layout.chunk_stride * bpe);

  *out_mask = active_levels_mask;
  return 0;

Error:
  return 1;
}

// ---- Writer callbacks ----

static struct writer_result
cpu_append(struct writer* self, struct slice input)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);
  const size_t bpe = lod_dtype_bpe(s->config.dtype);
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  const uint64_t dim0_size = s->config.dimensions[0].size;
  const uint64_t max_cursor =
    (dim0_size > 0) ? ceildiv(dim0_size, s->config.dimensions[0].chunk_size) *
                        s->layout.epoch_elements
                    : 0;

  while (src < end) {
    if (dim0_size > 0 && s->cursor >= max_cursor) {
      struct writer_result fr = cpu_flush(&s->writer);
      if (fr.error)
        return writer_error_at(src, end);
      return writer_finished_at(src, end);
    }

    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    uint64_t elements =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    if (dim0_size > 0) {
      uint64_t cap = max_cursor - s->cursor;
      if (elements > cap)
        elements = cap;
    }

    const uint64_t bytes = elements * bpe;

    // Scatter into chunk pool (or LOD buffer for multiscale).
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      if (s->levels.enable_multiscale) {
        // Accumulate into linear epoch buffer; LOD scatter happens at
        // epoch boundary in scatter_epoch().
        uint64_t epoch_offset = s->cursor % s->layout.epoch_elements;
        memcpy((char*)s->linear + epoch_offset * bpe, src, bytes);
      } else {
        CHECK(Error,
              transpose_cpu(s->chunk_pool,
                            src,
                            bytes,
                            (uint8_t)bpe,
                            s->cursor,
                            s->layout.lifted_rank,
                            s->layout.lifted_shape,
                            s->layout.lifted_strides) == 0);
      }

      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(&s->metrics.scatter, ms, bytes);
    }

    s->cursor += elements;
    src += bytes;

    // Epoch boundary: process the completed epoch.
    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      uint32_t active_mask = 1; // L0 always active
      if (s->levels.enable_multiscale) {
        CHECK(Error, scatter_epoch(s, &active_mask) == 0);
      }
      CHECK(Error, flush_epoch(s, active_mask) == 0);

      // Clear buffers for next epoch (skip when scatter will overwrite all).
      if (!s->pool_fully_covered)
        memset(s->chunk_pool,
               0,
               s->levels.total_chunks * s->layout.chunk_stride * bpe);
      if (s->lod_values) {
        size_t lod_bytes =
          s->cl.plan.levels.ends[s->cl.plan.nlod - 1] * bpe;
        memset(s->lod_values, 0, lod_bytes);
      }

      // Periodic metadata update.
      if (s->config.shard_sink->update_dim0) {
        struct platform_clock peek = s->metadata_update_clock;
        float elapsed = platform_toc(&peek);
        if (elapsed >= s->config.metadata_update_interval_s) {
          s->metadata_update_clock = peek;
          const struct dimension* dims = s->config.dimensions;
          for (int lv = 0; lv < s->levels.nlod; ++lv) {
            struct shard_state* ss = &s->shard[lv];
            uint64_t d0c =
              ss->shard_epoch * ss->chunks_per_shard_0 + ss->epoch_in_shard;
            s->config.shard_sink->update_dim0(
              s->config.shard_sink, (uint8_t)lv, d0c * dims[0].chunk_size);
          }
        }
      }
    }
  }

  return (struct writer_result){ .error = 0, .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

static struct writer_result
cpu_flush(struct writer* self)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);

  // Flush partial epoch.
  if (s->cursor % s->layout.epoch_elements != 0) {
    uint32_t active_mask = 1;
    if (s->levels.enable_multiscale) {
      if (scatter_epoch(s, &active_mask))
        return writer_error();
    }
    if (flush_epoch(s, active_mask))
      return writer_error();
  }

  // Drain any partial dim0 accumulators (levels that haven't emitted yet).
  if (s->levels.dim0_downsample && s->dim0_accum) {
    const size_t bpe = lod_dtype_bpe(s->config.dtype);
    struct platform_clock dim0_clk = { 0 };
    platform_toc(&dim0_clk);

    uint32_t drain_mask = 0;
    for (int lv = 1; lv < s->cl.plan.nlod; ++lv) {
      if (s->dim0_counts[lv] > 0) {
        if (lod_cpu_dim0_emit(&s->cl.plan,
                              s->lod_values,
                              s->dim0_accum,
                              lv,
                              s->dim0_counts[lv],
                              s->config.dtype,
                              s->config.dim0_reduce_method))
          return writer_error();
        s->dim0_counts[lv] = 0;

        // Scatter emitted level from morton space to chunk pool.
        const struct tile_stream_layout* layout_lv = &s->cl.lod_layouts[lv];
        if (lod_cpu_morton_to_chunks(&s->cl.plan,
                                     s->lod_values,
                                     s->chunk_pool,
                                     lv,
                                     layout_lv,
                                     s->morton_lut[lv],
                                     s->lod_batch_offsets[lv],
                                     s->config.dtype))
          return writer_error();
        drain_mask |= (1u << lv);
      }
    }

    float dim0_ms = (float)(platform_toc(&dim0_clk) * 1000.0);
    size_t dim0_bytes = 0;
    for (int lv = 1; lv < s->cl.plan.nlod; ++lv)
      dim0_bytes += s->cl.plan.batch_count * s->cl.plan.lod_counts[lv] * bpe;
    accumulate_metric_ms(&s->metrics.lod_dim0_fold, dim0_ms, dim0_bytes);

    // Compress + aggregate + deliver drained levels.
    if (drain_mask) {
      if (flush_epoch(s, drain_mask))
        return writer_error();
    }
  }

  // Emit partial shards.
  uint64_t dim0_chunks[LOD_MAX_LEVELS];
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->shard[lv];
    dim0_chunks[lv] =
      ss->shard_epoch * ss->chunks_per_shard_0 + ss->epoch_in_shard;
  }

  {
    struct platform_clock emit_clk = { 0 };
    platform_toc(&emit_clk);

    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      if (s->shard[lv].epoch_in_shard > 0) {
        if (emit_shards(&s->shard[lv], s->config.shard_alignment))
          return writer_error();
      }
    }

    float emit_ms = (float)(platform_toc(&emit_clk) * 1000.0);
    accumulate_metric_ms(&s->metrics.sink, emit_ms, 0);
  }

  // Final metadata.
  if (s->config.shard_sink->update_dim0) {
    const struct dimension* dims = s->config.dimensions;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      uint64_t dim0_extent = dim0_chunks[lv] * dims[0].chunk_size;
      s->config.shard_sink->update_dim0(
        s->config.shard_sink, (uint8_t)lv, dim0_extent);
    }
  }

  return writer_ok();
}

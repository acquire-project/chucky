#include "stream.internal.h"

#include "aggregate.h"
#include "compress.h"
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

  // Compressed output buffer.
  s->compressed = malloc(total_chunks * max_out);
  CHECK(Fail, s->compressed);

  s->comp_sizes = (size_t*)calloc(total_chunks, sizeof(size_t));
  CHECK(Fail, s->comp_sizes);

  // Per-level shard + aggregate state.
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    const struct level_layout_info* li = &s->cl.per_level[lv];
    s->agg_layout[lv] = li->agg_layout;

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
  }

  free(s->chunk_pool);
  free(s->compressed);
  free(s->comp_sizes);
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

// ---- Epoch processing ----

// Process one complete epoch: compress + aggregate + deliver for active levels.
// active_levels_mask: bit lv set means level lv has data to process.
static int
flush_epoch(struct tile_stream_cpu* s, uint32_t active_levels_mask)
{
  const size_t bpe = lod_dtype_bpe(s->config.dtype);
  const size_t max_out = s->cl.max_output_size;

  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    uint64_t chunk_count = s->levels.chunk_count[lv];
    uint64_t chunk_offset = s->levels.chunk_offset[lv];
    const struct tile_stream_layout* layout =
      (lv == 0) ? &s->layout : &s->cl.lod_layouts[lv];

    // Compress this level's chunks.
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      const void* pool_lv =
        (const char*)s->chunk_pool + chunk_offset * layout->chunk_stride * bpe;
      void* comp_lv = (char*)s->compressed + chunk_offset * max_out;
      size_t* sizes_lv = s->comp_sizes + chunk_offset;

      CHECK(Error,
            compress_cpu(s->config.codec,
                         pool_lv,
                         layout->chunk_stride * bpe,
                         comp_lv,
                         max_out,
                         sizes_lv,
                         layout->chunk_stride * bpe,
                         chunk_count) == 0);

      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(
        &s->metrics.compress, ms, chunk_count * layout->chunk_stride * bpe);
    }

    // Aggregate into shard order.
    struct aggregate_result ar;
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      const void* comp_lv = (const char*)s->compressed + chunk_offset * max_out;
      const size_t* sizes_lv = s->comp_sizes + chunk_offset;

      CHECK(Error,
            aggregate_cpu(comp_lv, sizes_lv, &s->agg_layout[lv], &ar) == 0);

      float ms = (float)(platform_toc(&clk) * 1000.0);
      size_t agg_bytes =
        ar.offsets[s->agg_layout[lv].covering_count];
      accumulate_metric_ms(&s->metrics.aggregate, ms, agg_bytes);
    }

    // Deliver to shards.
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      size_t sink_bytes = 0;
      CHECK(Error,
            deliver_to_shards_batch((uint8_t)lv,
                                    &s->shard[lv],
                                    &ar,
                                    1, // n_active = 1 (K=1)
                                    s->config.shard_sink,
                                    s->config.shard_alignment,
                                    &sink_bytes) == 0);

      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(&s->metrics.sink, ms, sink_bytes);
    }

    aggregate_cpu_result_free(&ar);
  }

  return 0;

Error:
  return 1;
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
        lod_cpu_scatter(
          &s->cl.plan, s->linear, s->lod_values, s->config.dtype) == 0);

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
  }

  platform_toc(&clk);
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    const struct tile_stream_layout* layout =
      (lv == 0) ? &s->layout : &s->cl.lod_layouts[lv];

    // Build per-batch chunk pool offsets (batch dims map to layout strides).
    const struct lod_plan* plan = &s->cl.plan;
    uint64_t* batch_offsets =
      (uint64_t*)calloc(plan->batch_count, sizeof(uint64_t));
    CHECK(Error, batch_offsets);

    for (uint64_t bi = 0; bi < plan->batch_count; ++bi) {
      uint64_t remainder = bi;
      int64_t offset = 0;
      for (int k = plan->batch_ndim - 1; k >= 0; --k) {
        uint64_t coord = remainder % plan->batch_shape[k];
        remainder /= plan->batch_shape[k];
        int d = plan->batch_map[k];
        uint64_t cs = layout->lifted_shape[2 * d + 1];
        uint64_t ci = coord / cs;
        uint64_t wi = coord % cs;
        offset += (int64_t)ci * layout->lifted_strides[2 * d];
        offset += (int64_t)wi * layout->lifted_strides[2 * d + 1];
      }
      // Add the level's chunk pool offset.
      batch_offsets[bi] =
        (uint64_t)offset +
        s->levels.chunk_offset[lv] * layout->chunk_stride;
    }

    CHECK(Error,
          lod_cpu_morton_to_chunks(plan,
                                   s->lod_values,
                                   s->chunk_pool,
                                   lv,
                                   layout,
                                   batch_offsets,
                                   s->config.dtype) == 0);
    free(batch_offsets);
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

      // Clear buffers for next epoch.
      memset(s->chunk_pool,
             0,
             s->levels.total_chunks * s->layout.chunk_stride * bpe);
      if (s->linear)
        memset(s->linear, 0, s->layout.epoch_elements * bpe);
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
        // Scatter this level to chunk pool and flush it
        // (simplified: reuse scatter_epoch logic for just this level)
        s->dim0_counts[lv] = 0;
      }
    }
  }

  // Emit partial shards.
  uint64_t dim0_chunks[LOD_MAX_LEVELS];
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->shard[lv];
    dim0_chunks[lv] =
      ss->shard_epoch * ss->chunks_per_shard_0 + ss->epoch_in_shard;
  }

  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (s->shard[lv].epoch_in_shard > 0) {
      if (emit_shards(&s->shard[lv], s->config.shard_alignment))
        return writer_error();
    }
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

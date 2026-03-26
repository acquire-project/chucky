#include "stream/config.h"
#include "cpu/stream.internal.h"

#include "cpu/compress.h"
#include "cpu/transpose.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// ---- Forward declarations ----

static struct writer_result
cpu_append(struct writer* self, struct slice input);
static struct writer_result
cpu_flush(struct writer* self);

// ---- Create / Destroy ----

struct tile_stream_cpu*
tile_stream_cpu_create(const struct tile_stream_configuration* config,
                       struct shard_sink* sink)
{
  if (!config || !sink)
    return NULL;
  if (config->dtype == dtype_f16)
    return NULL;

  struct tile_stream_cpu* s = (struct tile_stream_cpu*)calloc(1, sizeof(*s));
  if (!s)
    return NULL;

  s->config = *config;
  s->shard_sink = sink;

  // CPU codec alignment is 1 (no nvcomp alignment needed).
  if (compute_stream_layouts(config, 1, compress_cpu_max_output_size, &s->cl))
    goto Fail;

  s->layout = s->cl.layouts[0];
  s->levels = s->cl.levels;

  const uint32_t K = s->cl.epochs_per_batch;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  const uint64_t total_chunks = s->levels.total_chunks;
  const size_t chunk_stride_bytes = s->layout.chunk_stride * bytes_per_element;
  const size_t max_out = s->cl.max_output_size;

  // Chunk pool: K epochs' worth across all levels.
  s->chunk_pool = calloc((uint64_t)K * total_chunks, chunk_stride_bytes);
  CHECK(Fail, s->chunk_pool);

  s->pool_fully_covered = (s->layout.chunk_stride == s->layout.chunk_elements);

  // Compressed output buffer (K epochs).
  s->compressed = malloc((uint64_t)K * total_chunks * max_out);
  CHECK(Fail, s->compressed);

  s->comp_sizes = (size_t*)calloc((uint64_t)K * total_chunks, sizeof(size_t));
  CHECK(Fail, s->comp_sizes);

  // Per-level shard + aggregate state.
  {
    uint64_t max_batch_C = 0;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct level_layout_info* li = &s->cl.per_level[lv];
      s->agg_layout[lv] = li->agg_layout;
      const struct aggregate_layout* agg = &s->agg_layout[lv];

      uint32_t K_l = li->batch_active_count;
      s->batch_active_count[lv] = K_l;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = agg->covering_count;
      uint64_t M_lv = agg->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      // Per-level perm (single-epoch, for fallback).
      s->chunk_to_shard_map[lv] = (uint32_t*)malloc(M_lv * sizeof(uint32_t));
      CHECK(Fail, s->chunk_to_shard_map[lv]);

      // Per-level aggregate output buffers (batch-scaled).
      size_t data_lv = agg_pool_bytes((uint64_t)slot_count * M_lv,
                                      agg->max_comp_chunk_bytes,
                                      C_lv,
                                      agg->cps_inner,
                                      agg->page_size);
      {
        struct cpu_agg_slot* as = &s->agg_slots[lv];
        if (batch_C > 0) {
          as->offsets = (size_t*)malloc((batch_C + 1) * sizeof(size_t));
          as->chunk_sizes = (size_t*)calloc(batch_C, sizeof(size_t));
          CHECK(Fail, as->offsets && as->chunk_sizes);
        }
        if (data_lv > 0) {
          as->data = malloc(data_lv);
          as->data_capacity_bytes = data_lv;
          CHECK(Fail, as->data);
        }
      }

      // Batch aggregate LUTs (K_l > 1 only).
      if (K_l > 1) {
        uint64_t lut_len = (uint64_t)K_l * M_lv;

        s->batch_gather[lv] = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
        s->batch_chunk_to_shard_map[lv] = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
        CHECK(Fail, s->batch_gather[lv] && s->batch_chunk_to_shard_map[lv]);
      }

      CHECK(Fail, init_shard_state(&s->shard[lv], li) == 0);
    }

    // Shared permuted_sizes scratch (sized to max batch C).
    if (max_batch_C > 0) {
      s->shard_order_sizes = (size_t*)calloc(max_batch_C, sizeof(size_t));
      CHECK(Fail, s->shard_order_sizes);
    }
  }

  // LOD buffers (multiscale only).
  if (s->levels.enable_multiscale) {
    // Linear epoch buffer: input is accumulated here before LOD scatter.
    s->linear = calloc(s->layout.epoch_elements, bytes_per_element);
    CHECK(Fail, s->linear);

    uint64_t total_lod_elements = s->cl.plan.levels.ends[s->cl.plan.nlod - 1];
    s->lod_values = calloc(total_lod_elements, bytes_per_element);
    CHECK(Fail, s->lod_values);

    // Append accumulator: total elements in levels 1+ (packed).
    // Uses the source dtype (not a wider accumulator) for integer mean —
    // overflow_safe_add_shift prevents overflow at the cost of rounding per
    // fold.
    if (s->levels.append_downsample) {
      uint64_t append_total = 0;
      for (int lv = 1; lv < s->cl.plan.nlod; ++lv)
        append_total += s->cl.plan.batch_count * s->cl.plan.lod_nelem[lv];
      if (append_total > 0) {
        s->append_accum = calloc(append_total, bytes_per_element);
        CHECK(Fail, s->append_accum);
      }
      memset(s->append_counts, 0, sizeof(s->append_counts));
    }

    // Allocate L0 scatter LUT + batch offsets.
    {
      const struct lod_plan* plan = &s->cl.plan;
      s->scatter_lut =
        (uint32_t*)malloc(plan->lod_nelem[0] * sizeof(uint32_t));
      CHECK(Fail, s->scatter_lut);

      s->scatter_batch_offsets =
        (uint64_t*)calloc(plan->batch_count, sizeof(uint64_t));
      CHECK(Fail, s->scatter_batch_offsets);
    }

    // Allocate morton-to-chunk LUTs and batch offsets for each level.
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      const struct lod_plan* plan = &s->cl.plan;
      uint64_t lod_count = plan->lod_nelem[lv];

      s->morton_lut[lv] = (uint32_t*)malloc(lod_count * sizeof(uint32_t));
      CHECK(Fail, s->morton_lut[lv]);

      s->lod_batch_offsets[lv] =
        (uint64_t*)calloc(plan->batch_count, sizeof(uint64_t));
      CHECK(Fail, s->lod_batch_offsets[lv]);
    }
  }

  // Fill all LUTs.
  {
    struct lut_targets luts = {
      .scatter_lut = s->scatter_lut,
      .scatter_batch_offsets = s->scatter_batch_offsets,
    };
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      luts.chunk_to_shard_map[lv] = s->chunk_to_shard_map[lv];
      luts.batch_gather[lv] = s->batch_gather[lv];
      luts.batch_chunk_to_shard_map[lv] = s->batch_chunk_to_shard_map[lv];
      luts.morton_lut[lv] = s->morton_lut[lv];
      luts.lod_batch_offsets[lv] = s->lod_batch_offsets[lv];
    }
    cpu_pipeline_compute_luts(
      &s->cl, &s->levels, s->batch_active_count, s->agg_layout, &luts);
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
    s->metrics.lod_append_fold = mk_stream_metric("lod_append_fold");
    s->metrics.lod_morton_chunk = mk_stream_metric("lod_morton");
  }

  // Precompute max_cursor so cpu_append doesn't recompute each call.
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = s->levels.n_append;
    if (dims[0].size > 0) {
      s->max_cursor_elements = s->layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        s->max_cursor_elements *= ceildiv(dims[d].size, dims[d].chunk_size);
    } else {
      s->max_cursor_elements = 0;
    }
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
    free(s->chunk_to_shard_map[lv]);
    free(s->batch_gather[lv]);
    free(s->batch_chunk_to_shard_map[lv]);
    free(s->morton_lut[lv]);
    free(s->lod_batch_offsets[lv]);

    struct cpu_agg_slot* as = &s->agg_slots[lv];
    free(as->data);
    free(as->offsets);
    free(as->chunk_sizes);
  }
  free(s->shard_order_sizes);

  free(s->chunk_pool);
  free(s->compressed);
  free(s->comp_sizes);
  free(s->scatter_lut);
  free(s->scatter_batch_offsets);
  free(s->linear);
  free(s->lod_values);
  free(s->append_accum);
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
  return s->cursor_elements;
}

// ---- Memory estimate ----

// Compute memory sizing from pre-computed layouts.
// Used by memory_estimate for reporting.
static void
compute_memory_info(const struct computed_stream_layouts* cl,
                    size_t bytes_per_element,
                    struct tile_stream_cpu_memory_info* info)
{
  const uint32_t K = cl->epochs_per_batch;
  const uint64_t total_chunks = cl->levels.total_chunks;
  const size_t chunk_stride_bytes = cl->layouts[0].chunk_stride * bytes_per_element;
  const size_t max_out = cl->max_output_size;

  info->chunk_pool_bytes = (uint64_t)K * total_chunks * chunk_stride_bytes;
  info->compressed_pool_bytes = (uint64_t)K * total_chunks * max_out;
  info->comp_sizes_bytes = (uint64_t)K * total_chunks * sizeof(size_t);

  // Aggregate: per-level perm + batch-scaled slots + LUTs + shared scratch.
  {
    size_t agg = 0;
    uint64_t max_batch_C = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      const struct level_layout_info* li = &cl->per_level[lv];
      const struct aggregate_layout* al = &li->agg_layout;
      uint32_t K_l = li->batch_active_count;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = al->covering_count;
      uint64_t M_lv = al->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      size_t data_lv = agg_pool_bytes((uint64_t)slot_count * M_lv,
                                      al->max_comp_chunk_bytes,
                                      C_lv,
                                      al->cps_inner,
                                      al->page_size);

      agg += M_lv * sizeof(uint32_t); // per-epoch perm
      agg += data_lv +
             (batch_C + 1) * sizeof(size_t) + // offsets (batch-scaled)
             batch_C * sizeof(size_t);        // chunk_sizes (batch-scaled)

      // Batch LUTs (gather + perm).
      if (K_l > 1) {
        uint64_t lut_len = (uint64_t)K_l * M_lv;
        agg += 2 * lut_len * sizeof(uint32_t);
      }
    }
    if (max_batch_C > 0)
      agg += max_batch_C * sizeof(size_t); // shared permuted_sizes scratch
    info->aggregate_bytes = agg;
  }

  // LOD buffers (multiscale only).
  {
    size_t lod = 0;
    if (cl->levels.enable_multiscale) {
      lod += cl->layouts[0].epoch_elements * bytes_per_element; // linear
      uint64_t total_lod_elements = cl->plan.levels.ends[cl->plan.nlod - 1];
      lod += total_lod_elements * bytes_per_element; // lod_values

      if (cl->levels.append_downsample) {
        uint64_t append_total = 0;
        for (int lv = 1; lv < cl->plan.nlod; ++lv)
          append_total += cl->plan.batch_count * cl->plan.lod_nelem[lv];
        lod += append_total * bytes_per_element; // append_accum
      }

      lod += cl->plan.lod_nelem[0] * sizeof(uint32_t); // scatter_lut
      lod += cl->plan.batch_count * sizeof(uint64_t);   // scatter_batch_offsets

      for (int lv = 0; lv < cl->levels.nlod; ++lv) {
        lod += cl->plan.lod_nelem[lv] * sizeof(uint32_t); // morton_lut
        lod += cl->plan.batch_count * sizeof(uint64_t);    // batch_offsets
      }
    }
    info->lod_bytes = lod;
  }

  // Shard state: active_shard arrays + index buffers.
  {
    size_t shards = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      const struct level_layout_info* li = &cl->per_level[lv];
      shards += li->shard_inner_count * sizeof(struct active_shard);
      shards += li->shard_inner_count * li->chunks_per_shard_total * 2 *
                sizeof(uint64_t);
    }
    info->shard_bytes = shards;
  }

  info->heap_bytes = sizeof(struct tile_stream_cpu) + info->chunk_pool_bytes +
                     info->compressed_pool_bytes + info->comp_sizes_bytes +
                     info->aggregate_bytes + info->lod_bytes +
                     info->shard_bytes;

  info->chunks_per_epoch = cl->layouts[0].chunks_per_epoch;
  info->total_chunks = total_chunks;
  info->max_output_size = max_out;
  info->nlod = cl->levels.nlod;
  info->epochs_per_batch = K;
}

int
tile_stream_cpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_cpu_memory_info* info)
{
  if (!info)
    return 1;

  memset(info, 0, sizeof(*info));

  struct computed_stream_layouts cl;
  if (compute_stream_layouts(config, 1, compress_cpu_max_output_size, &cl))
    return 1;

  compute_memory_info(&cl, dtype_bpe(config->dtype), info);
  computed_stream_layouts_free(&cl);
  return 0;
}

int
tile_stream_cpu_advise_chunk_sizes(struct tile_stream_configuration* config,
                                   size_t target_chunk_bytes,
                                   const uint8_t* ratios,
                                   size_t budget_bytes)
{
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  if (bytes_per_element == 0 || budget_bytes == 0)
    return 1;

  for (size_t target = target_chunk_bytes; target >= bytes_per_element; target >>= 1) {
    dims_budget_chunk_bytes(
      config->dimensions, config->rank, target, bytes_per_element, ratios);
    struct tile_stream_cpu_memory_info mem;
    if (tile_stream_cpu_memory_estimate(config, &mem))
      return 1;
    if (mem.heap_bytes <= budget_bytes)
      return 0;
  }
  return 1;
}

// ---- Pipeline helpers ----
// Keep in sync with multiarray/stream.c::make_flush_params.

static struct flush_batch_params
make_flush_params(struct tile_stream_cpu* s)
{
  const size_t bytes_per_element = dtype_bpe(s->config.dtype);
  struct flush_batch_params p = {
    .codec = s->config.codec,
    .chunk_pool = s->chunk_pool,
    .chunk_stride_bytes = s->layout.chunk_stride * bytes_per_element,
    .chunk_bytes = s->layout.chunk_elements * bytes_per_element,
    .compressed = s->compressed,
    .max_output_size_bytes = s->cl.max_output_size,
    .comp_sizes = s->comp_sizes,
    .total_chunks = s->levels.total_chunks,
    .nlod = s->levels.nlod,
    .shard_order_sizes_bytes = s->shard_order_sizes,
    .sink = s->shard_sink,
    .shard_alignment_bytes = s->config.shard_alignment,
    .metrics = &s->metrics,
  };
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    p.levels[lv] = (struct flush_level_view){
      .agg_layout = &s->agg_layout[lv],
      .batch_active_count = s->batch_active_count[lv],
      .chunk_offset = s->levels.chunk_offset[lv],
      .chunk_to_shard_map = s->chunk_to_shard_map[lv],
      .batch_chunk_to_shard_map = s->batch_chunk_to_shard_map[lv],
      .batch_gather = s->batch_gather[lv],
      .agg_slot = &s->agg_slots[lv],
      .shard = &s->shard[lv],
    };
  }
  return p;
}

// Keep in sync with multiarray/stream.c::make_scatter_params.
static struct scatter_epoch_params
make_scatter_params(struct tile_stream_cpu* s)
{
  struct scatter_epoch_params p = {
    .dtype = s->config.dtype,
    .reduce_method = s->config.reduce_method,
    .append_reduce_method = s->config.append_reduce_method,
    .cl = &s->cl,
    .chunk_pool = s->chunk_pool,
    .linear = s->linear,
    .lod_values = s->lod_values,
    .scatter_lut = s->scatter_lut,
    .scatter_batch_offsets = s->scatter_batch_offsets,
    .append_accum = s->append_accum,
    .append_counts = s->append_counts,
    .metrics = &s->metrics,
  };
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    p.morton_lut[lv] = s->morton_lut[lv];
    p.lod_batch_offsets[lv] = s->lod_batch_offsets[lv];
  }
  return p;
}

// ---- Writer callbacks ----

static struct writer_result
cpu_append(struct writer* self, struct slice input)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);
  const size_t bytes_per_element = dtype_bpe(s->config.dtype);
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  const uint64_t max_cursor = s->max_cursor_elements;

  while (src < end) {
    if (max_cursor > 0 && s->cursor_elements >= max_cursor) {
      struct writer_result fr = cpu_flush(&s->writer);
      if (fr.error)
        return writer_error_at(src, end);
      return writer_finished_at(src, end);
    }

    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor_elements % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bytes_per_element;
    uint64_t elements =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    if (max_cursor > 0) {
      uint64_t cap = max_cursor - s->cursor_elements;
      if (elements > cap)
        elements = cap;
    }

    const uint64_t bytes = elements * bytes_per_element;

    // Scatter into chunk pool (or LOD buffer for multiscale).
    {
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      if (s->levels.enable_multiscale) {
        // Accumulate into linear epoch buffer; LOD scatter happens at
        // epoch boundary in scatter_epoch().
        uint64_t epoch_offset = s->cursor_elements % s->layout.epoch_elements;
        memcpy((char*)s->linear + epoch_offset * bytes_per_element, src, bytes);
      } else {
        // Transpose into the current epoch's pool slice.
        void* epoch_pool =
          (char*)s->chunk_pool + (uint64_t)s->batch_accumulated *
                                   s->levels.total_chunks *
                                   s->layout.chunk_stride * bytes_per_element;
        CHECK(Error,
              transpose_cpu(epoch_pool,
                            src,
                            bytes,
                            (uint8_t)bytes_per_element,
                            s->cursor_elements,
                            s->layout.lifted_rank,
                            s->layout.lifted_shape,
                            s->layout.lifted_strides) == 0);
      }

      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(&s->metrics.scatter, ms, bytes, 0);
    }

    s->cursor_elements += elements;
    src += bytes;

    // Epoch boundary: accumulate into batch, flush when full.
    if (s->cursor_elements % s->layout.epoch_elements == 0 && s->cursor_elements > 0) {
      uint32_t active_mask = 1; // L0 always active
      if (s->levels.enable_multiscale) {
        struct scatter_epoch_params sp = make_scatter_params(s);
        CHECK(Error,
              cpu_pipeline_scatter_epoch(&sp, s->batch_accumulated, &active_mask) == 0);
      }

      CHECK(Error, s->batch_accumulated < MAX_BATCH_EPOCHS);
      s->batch_active_masks[s->batch_accumulated] = active_mask;
      s->batch_accumulated++;

      if (s->batch_accumulated == s->cl.epochs_per_batch) {
        struct flush_batch_params fp = make_flush_params(s);
        CHECK(Error,
              cpu_pipeline_flush_batch(&fp, s->batch_accumulated, s->batch_active_masks) == 0);
        s->batch_accumulated = 0;

        // Clear full batch pool for next batch (each epoch's slice
        // starts zeroed, so no per-epoch clearing is needed).
        if (!s->pool_fully_covered)
          memset(s->chunk_pool,
                 0,
                 (uint64_t)s->cl.epochs_per_batch * s->levels.total_chunks *
                   s->layout.chunk_stride * bytes_per_element);
      }

      if (s->lod_values) {
        size_t lod_bytes = s->cl.plan.levels.ends[s->cl.plan.nlod - 1] * bytes_per_element;
        memset(s->lod_values, 0, lod_bytes);
      }

      // Periodic metadata update.
      if (s->shard_sink->update_append) {
        struct platform_clock peek = s->metadata_update_clock;
        float elapsed = platform_toc(&peek);
        if (elapsed >= s->config.metadata_update_interval_s) {
          s->metadata_update_clock = peek;
          const struct dimension* dims = s->config.dimensions;
          const uint8_t na = s->levels.n_append;
          for (int lv = 0; lv < s->levels.nlod; ++lv) {
            struct shard_state* ss = &s->shard[lv];
            uint64_t total_ac = ss->shard_epoch * ss->chunks_per_shard_append + ss->epoch_in_shard;
            uint64_t append_sizes[HALF_MAX_RANK];
            for (int d = 1; d < na; ++d)
              append_sizes[d] = dims[d].size;
            uint64_t iac = 1;
            for (int d = 1; d < na; ++d)
              iac *= ceildiv(dims[d].size, dims[d].chunk_size);
            append_sizes[0] = (iac > 0) ? ceildiv(total_ac, iac) * dims[0].chunk_size : 0;
            if (s->shard_sink->update_append(s->shard_sink, (uint8_t)lv, na, append_sizes))
              goto Error;
          }
        }
      }
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

static struct writer_result
cpu_flush(struct writer* self)
{
  struct tile_stream_cpu* s =
    container_of(self, struct tile_stream_cpu, writer);

  // Flush partial epoch into the batch.
  if (s->cursor_elements % s->layout.epoch_elements != 0) {
    uint32_t active_mask = 1;
    if (s->levels.enable_multiscale) {
      struct scatter_epoch_params sp = make_scatter_params(s);
      if (cpu_pipeline_scatter_epoch(&sp, s->batch_accumulated, &active_mask))
        return writer_error();
    }
    if (s->batch_accumulated >= MAX_BATCH_EPOCHS)
      return writer_error();
    s->batch_active_masks[s->batch_accumulated] = active_mask;
    s->batch_accumulated++;
  }

  // Flush any accumulated batch (partial or full).
  if (s->batch_accumulated > 0) {
    struct flush_batch_params fp = make_flush_params(s);
    if (cpu_pipeline_flush_batch(&fp, s->batch_accumulated, s->batch_active_masks))
      return writer_error();
    s->batch_accumulated = 0;
  }

  // Drain any partial append accumulators (levels that haven't emitted yet).
  if (s->levels.append_downsample && s->append_accum) {
    struct append_drain_params dp = {
      .cl = &s->cl,
      .dtype = s->config.dtype,
      .append_reduce_method = s->config.append_reduce_method,
      .lod_values = s->lod_values,
      .append_accum = s->append_accum,
      .append_counts = s->append_counts,
      .chunk_pool = s->chunk_pool,
      .metrics = &s->metrics,
    };
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      dp.morton_lut[lv] = s->morton_lut[lv];
      dp.lod_batch_offsets[lv] = s->lod_batch_offsets[lv];
    }

    uint32_t drain_mask = 0;
    if (cpu_pipeline_append_drain(&dp, &drain_mask))
      return writer_error();

    // Compress + aggregate + deliver drained levels (single-epoch batch).
    if (drain_mask) {
      s->batch_active_masks[0] = drain_mask;
      struct flush_batch_params fp = make_flush_params(s);
      if (cpu_pipeline_flush_batch(&fp, 1, s->batch_active_masks))
        return writer_error();
    }
  }

  // Emit partial shards.
  uint64_t append_chunks[LOD_MAX_LEVELS];
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->shard[lv];
    append_chunks[lv] =
      ss->shard_epoch * ss->chunks_per_shard_append + ss->epoch_in_shard;
  }

  {
    struct platform_clock emit_clk = { 0 };
    platform_toc(&emit_clk);

    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      if (s->shard[lv].epoch_in_shard > 0) {
        if (finalize_shards(&s->shard[lv], s->config.shard_alignment))
          return writer_error();
      }
    }

    float emit_ms = (float)(platform_toc(&emit_clk) * 1000.0);
    accumulate_metric_ms(&s->metrics.sink, emit_ms, 0, 0);
  }

  // Final metadata.
  if (s->shard_sink->update_append) {
    const struct dimension* dims = s->config.dimensions;
    const uint8_t na = s->levels.n_append;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      uint64_t total_ac = append_chunks[lv];
      uint64_t append_sizes[HALF_MAX_RANK];
      for (int d = 1; d < na; ++d)
        append_sizes[d] = dims[d].size;
      uint64_t iac = 1;
      for (int d = 1; d < na; ++d)
        iac *= ceildiv(dims[d].size, dims[d].chunk_size);
      append_sizes[0] = (iac > 0) ? ceildiv(total_ac, iac) * dims[0].chunk_size : 0;
      if (s->shard_sink->update_append(s->shard_sink, (uint8_t)lv, na, append_sizes))
        return writer_error();
    }
  }

  return writer_ok();
}

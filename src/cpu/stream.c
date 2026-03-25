#include "stream/config.h"
#include "cpu/stream.internal.h"

#include "cpu/aggregate.h"
#include "cpu/compress.h"
#include "cpu/lod.h"
#include "cpu/transpose.h"
#include "platform/platform.h"
#include "util/index.ops.h"
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
  const size_t bpe = dtype_bpe(config->dtype);
  const uint64_t total_chunks = s->levels.total_chunks;
  const size_t chunk_stride_bytes = s->layout.chunk_stride * bpe;
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
      const struct aggregate_layout* al = &s->agg_layout[lv];

      uint32_t K_l = li->batch_active_count;
      s->batch_agg_active_count[lv] = K_l;
      uint32_t slot_count = K_l > 0 ? K_l : 1;
      uint64_t C_lv = al->covering_count;
      uint64_t M_lv = al->chunks_per_epoch;
      uint64_t batch_C = (uint64_t)slot_count * C_lv;

      if (batch_C > max_batch_C)
        max_batch_C = batch_C;

      // Per-level perm (single-epoch, for fallback).
      s->agg_perm[lv] = (uint32_t*)malloc(M_lv * sizeof(uint32_t));
      CHECK(Fail, s->agg_perm[lv]);
      for (uint64_t i = 0; i < M_lv; ++i)
        s->agg_perm[lv][i] = (uint32_t)ravel(
          al->lifted_rank, al->lifted_shape, al->lifted_strides, i);

      // Per-level aggregate output buffers (batch-scaled).
      size_t data_lv = agg_pool_bytes((uint64_t)slot_count * M_lv,
                                      al->max_comp_chunk_bytes,
                                      C_lv,
                                      al->cps_inner,
                                      al->page_size);
      {
        struct cpu_agg_slot* as = &s->agg_slots[lv];
        if (batch_C > 0) {
          as->offsets = (size_t*)malloc((batch_C + 1) * sizeof(size_t));
          as->chunk_sizes = (size_t*)calloc(batch_C, sizeof(size_t));
          CHECK(Fail, as->offsets && as->chunk_sizes);
        }
        if (data_lv > 0) {
          as->data = malloc(data_lv);
          as->data_capacity = data_lv;
          CHECK(Fail, as->data);
        }
      }

      // Batch aggregate LUTs (K_l > 1 only).
      if (K_l > 1) {
        uint64_t lut_len = (uint64_t)K_l * M_lv;

        s->batch_gather[lv] = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
        s->batch_agg_perm[lv] = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
        CHECK(Fail, s->batch_gather[lv] && s->batch_agg_perm[lv]);

        for (uint32_t a = 0; a < K_l; ++a) {
          uint32_t period =
            (s->levels.dim0_downsample && lv > 0) ? (1u << lv) : 1;
          uint32_t pool_epoch = (a + 1) * period - 1;

          for (uint64_t j = 0; j < M_lv; ++j) {
            uint64_t idx = (uint64_t)a * M_lv + j;
            s->batch_gather[lv][idx] =
              (uint32_t)(pool_epoch * total_chunks +
                         s->levels.chunk_offset[lv] + j);
            uint32_t perm_pos = (uint32_t)ravel(
              al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
            s->batch_agg_perm[lv][idx] = perm_pos * K_l + a;
          }
        }
      }

      struct shard_state* ss = &s->shard[lv];
      ss->chunks_per_shard_0 = li->chunks_per_shard_0;
      ss->chunks_per_shard_inner = li->chunks_per_shard_inner;
      ss->chunks_per_shard_total = li->chunks_per_shard_total;
      ss->shard_inner_count = li->shard_inner_count;
      ss->shards = (struct active_shard*)calloc(li->shard_inner_count,
                                                sizeof(struct active_shard));
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

    // Shared permuted_sizes scratch (sized to max batch C).
    if (max_batch_C > 0) {
      s->agg_permuted_sizes = (size_t*)calloc(max_batch_C, sizeof(size_t));
      CHECK(Fail, s->agg_permuted_sizes);
    }
  }

  // LOD buffers (multiscale only).
  if (s->levels.enable_multiscale) {
    // Linear epoch buffer: input is accumulated here before LOD scatter.
    s->linear = calloc(s->layout.epoch_elements, bpe);
    CHECK(Fail, s->linear);

    uint64_t total_lod_elements = s->cl.plan.levels.ends[s->cl.plan.nlod - 1];
    s->lod_values = calloc(total_lod_elements, bpe);
    CHECK(Fail, s->lod_values);

    // Dim0 accumulator: total elements in levels 1+ (packed).
    // Uses the source dtype (not a wider accumulator) for integer mean —
    // overflow_safe_add_shift prevents overflow at the cost of rounding per
    // fold.
    if (s->levels.dim0_downsample) {
      uint64_t dim0_total = 0;
      for (int lv = 1; lv < s->cl.plan.nlod; ++lv)
        dim0_total += s->cl.plan.batch_count * s->cl.plan.lod_nelem[lv];
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
        (uint32_t*)malloc(plan->lod_nelem[0] * sizeof(uint32_t));
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
        &s->cl.layouts[lv];
      uint64_t lod_count = plan->lod_nelem[lv];

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

  // Precompute max_cursor so cpu_append doesn't recompute each call.
  {
    const struct dimension* dims = config->dimensions;
    s->max_cursor =
      (dims[0].size > 0)
        ? ceildiv(dims[0].size, dims[0].chunk_size) * s->layout.epoch_elements
        : 0;
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
    free(s->batch_gather[lv]);
    free(s->batch_agg_perm[lv]);
    free(s->morton_lut[lv]);
    free(s->lod_batch_offsets[lv]);

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

// Compute memory sizing from pre-computed layouts.
// Used by memory_estimate for reporting.
static void
compute_memory_info(const struct computed_stream_layouts* cl,
                    size_t bpe,
                    struct tile_stream_cpu_memory_info* info)
{
  const uint32_t K = cl->epochs_per_batch;
  const uint64_t total_chunks = cl->levels.total_chunks;
  const size_t chunk_stride_bytes = cl->layouts[0].chunk_stride * bpe;
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
      lod += cl->layouts[0].epoch_elements * bpe; // linear
      uint64_t total_lod_elements = cl->plan.levels.ends[cl->plan.nlod - 1];
      lod += total_lod_elements * bpe; // lod_values

      if (cl->levels.dim0_downsample) {
        uint64_t dim0_total = 0;
        for (int lv = 1; lv < cl->plan.nlod; ++lv)
          dim0_total += cl->plan.batch_count * cl->plan.lod_nelem[lv];
        lod += dim0_total * bpe; // dim0_accum
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
  const size_t bpe = dtype_bpe(config->dtype);
  if (bpe == 0 || budget_bytes == 0)
    return 1;

  for (size_t target = target_chunk_bytes; target >= bpe; target >>= 1) {
    dims_budget_chunk_bytes(
      config->dimensions, config->rank, target, bpe, ratios);
    struct tile_stream_cpu_memory_info mem;
    if (tile_stream_cpu_memory_estimate(config, &mem))
      return 1;
    if (mem.heap_bytes <= budget_bytes)
      return 0;
  }
  return 1;
}

// ---- Epoch processing ----

// Compress all K epochs, then aggregate + deliver per-level.
// Uses batch aggregate when the full batch is available and K_l > 1;
// falls back to per-epoch aggregate otherwise.
static int
flush_batch(struct tile_stream_cpu* s,
            uint32_t n_epochs,
            const uint32_t* active_masks)
{
  const size_t bpe = dtype_bpe(s->config.dtype);
  const size_t max_out = s->cl.max_output_size;
  const uint64_t total_chunks = s->levels.total_chunks;

  // Compress all K epochs at once (pool is contiguous).
  {
    struct platform_clock clk = { 0 };
    platform_toc(&clk);

    if (compress_cpu(s->config.codec,
                     s->chunk_pool,
                     s->layout.chunk_stride * bpe,
                     s->compressed,
                     max_out,
                     s->comp_sizes,
                     s->layout.chunk_elements * bpe,
                     n_epochs * total_chunks)) {
      return 1;
    }

    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&s->metrics.compress,
                         ms,
                         n_epochs * total_chunks * s->layout.chunk_elements *
                           bpe,
                         0);
  }

  // Aggregate + deliver per-level.
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    // Count active epochs for this level.
    uint32_t active_count = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (active_masks[e] & (1u << lv))
        active_count++;
    if (active_count == 0)
      continue;

    if (active_count == s->batch_agg_active_count[lv] &&
        s->batch_agg_active_count[lv] > 1) {
      // Batch aggregate.
      struct platform_clock clk = { 0 };
      platform_toc(&clk);

      struct aggregate_cpu_workspace ws = {
        .perm = s->batch_agg_perm[lv],
        .permuted_sizes = s->agg_permuted_sizes,
        .data = s->agg_slots[lv].data,
        .data_capacity = s->agg_slots[lv].data_capacity,
        .offsets = s->agg_slots[lv].offsets,
        .chunk_sizes = s->agg_slots[lv].chunk_sizes,
      };
      struct aggregate_result ar;
      if (aggregate_cpu_batch_into(s->compressed,
                                   s->comp_sizes,
                                   s->batch_gather[lv],
                                   &s->agg_layout[lv],
                                   active_count,
                                   &ws,
                                   &ar))
        return 1;

      uint64_t batch_C =
        (uint64_t)active_count * s->agg_layout[lv].covering_count;
      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(&s->metrics.aggregate, ms, ar.offsets[batch_C], 0);

      // Batch deliver.
      platform_toc(&clk);
      size_t sink_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  &s->shard[lv],
                                  &ar,
                                  active_count,
                                  s->shard_sink,
                                  s->config.shard_alignment,
                                  &sink_bytes))
        return 1;
      ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(&s->metrics.sink, ms, sink_bytes, 0);
    } else {
      // Per-epoch fallback.
      for (uint32_t e = 0; e < n_epochs; ++e) {
        if (!(active_masks[e] & (1u << lv)))
          continue;

        uint64_t comp_base =
          (uint64_t)e * total_chunks + s->levels.chunk_offset[lv];

        struct platform_clock clk = { 0 };
        platform_toc(&clk);

        const void* comp_lv = (const char*)s->compressed + comp_base * max_out;
        const size_t* sizes_lv = s->comp_sizes + comp_base;

        struct aggregate_cpu_workspace ws = {
          .perm = s->agg_perm[lv],
          .permuted_sizes = s->agg_permuted_sizes,
          .data = s->agg_slots[lv].data,
          .data_capacity = s->agg_slots[lv].data_capacity,
          .offsets = s->agg_slots[lv].offsets,
          .chunk_sizes = s->agg_slots[lv].chunk_sizes,
        };
        struct aggregate_result ar;
        if (aggregate_cpu_into(comp_lv, sizes_lv, &s->agg_layout[lv], &ws, &ar))
          return 1;

        float ms = (float)(platform_toc(&clk) * 1000.0);
        size_t agg_bytes = ar.offsets[s->agg_layout[lv].covering_count];
        accumulate_metric_ms(&s->metrics.aggregate, ms, agg_bytes, 0);

        // Deliver.
        platform_toc(&clk);
        size_t sink_bytes = 0;
        if (deliver_to_shards_batch((uint8_t)lv,
                                    &s->shard[lv],
                                    &ar,
                                    1,
                                    s->shard_sink,
                                    s->config.shard_alignment,
                                    &sink_bytes))
          return 1;
        ms = (float)(platform_toc(&clk) * 1000.0);
        accumulate_metric_ms(&s->metrics.sink, ms, sink_bytes, 0);
      }
    }
  }

  return 0;
}

// Scatter one epoch into chunk pool. Returns active_levels_mask via out_mask.
// epoch_in_batch: index within the current batch (0..K-1), used to offset pool.
static int
scatter_epoch(struct tile_stream_cpu* s,
              uint32_t epoch_in_batch,
              uint32_t* out_mask)
{
  const size_t bpe = dtype_bpe(s->config.dtype);
  void* epoch_pool = (char*)s->chunk_pool + (uint64_t)epoch_in_batch *
                                              s->levels.total_chunks *
                                              s->layout.chunk_stride * bpe;

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
        lod_cpu_gather(&s->cl.plan,
                       s->linear,
                       s->lod_values,
                       s->scatter_lut,
                       s->scatter_batch_offsets,
                       s->config.dtype) == 0);

  float scatter_ms = (float)(platform_toc(&clk) * 1000.0);
  accumulate_metric_ms(
    &s->metrics.lod_gather, scatter_ms, s->layout.epoch_elements * bpe, 0);

  platform_toc(&clk);
  CHECK(Error,
        lod_cpu_reduce(&s->cl.plan,
                       s->lod_values,
                       s->config.dtype,
                       s->config.reduce_method) == 0);

  float ms = (float)(platform_toc(&clk) * 1000.0);
  accumulate_metric_ms(&s->metrics.lod_reduce,
                       ms,
                       s->cl.plan.levels.ends[s->cl.plan.nlod - 1] * bpe,
                       0);

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
      dim0_bytes += s->cl.plan.batch_count * s->cl.plan.lod_nelem[lv] * bpe;
    accumulate_metric_ms(&s->metrics.lod_dim0_fold, dim0_ms, dim0_bytes, 0);
  }

  platform_toc(&clk);
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    const struct tile_stream_layout* layout =
      &s->cl.layouts[lv];

    CHECK(Error,
          lod_cpu_morton_to_chunks(&s->cl.plan,
                                   s->lod_values,
                                   epoch_pool,
                                   lv,
                                   layout,
                                   s->morton_lut[lv],
                                   s->lod_batch_offsets[lv],
                                   s->config.dtype) == 0);
  }

  ms = (float)(platform_toc(&clk) * 1000.0);
  accumulate_metric_ms(&s->metrics.lod_morton_chunk,
                       ms,
                       s->levels.total_chunks * s->layout.chunk_stride * bpe,
                       0);

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
  const size_t bpe = dtype_bpe(s->config.dtype);
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  const uint64_t max_cursor = s->max_cursor;

  while (src < end) {
    if (max_cursor > 0 && s->cursor >= max_cursor) {
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

    if (max_cursor > 0) {
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
        // Transpose into the current epoch's pool slice.
        void* epoch_pool =
          (char*)s->chunk_pool + (uint64_t)s->batch_accumulated *
                                   s->levels.total_chunks *
                                   s->layout.chunk_stride * bpe;
        CHECK(Error,
              transpose_cpu(epoch_pool,
                            src,
                            bytes,
                            (uint8_t)bpe,
                            s->cursor,
                            s->layout.lifted_rank,
                            s->layout.lifted_shape,
                            s->layout.lifted_strides) == 0);
      }

      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(&s->metrics.scatter, ms, bytes, 0);
    }

    s->cursor += elements;
    src += bytes;

    // Epoch boundary: accumulate into batch, flush when full.
    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      uint32_t active_mask = 1; // L0 always active
      if (s->levels.enable_multiscale) {
        CHECK(Error, scatter_epoch(s, s->batch_accumulated, &active_mask) == 0);
      }

      s->batch_active_masks[s->batch_accumulated] = active_mask;
      s->batch_accumulated++;

      if (s->batch_accumulated == s->cl.epochs_per_batch) {
        CHECK(Error,
              flush_batch(s, s->batch_accumulated, s->batch_active_masks) == 0);
        s->batch_accumulated = 0;

        // Clear full batch pool for next batch (each epoch's slice
        // starts zeroed, so no per-epoch clearing is needed).
        if (!s->pool_fully_covered)
          memset(s->chunk_pool,
                 0,
                 (uint64_t)s->cl.epochs_per_batch * s->levels.total_chunks *
                   s->layout.chunk_stride * bpe);
      }

      if (s->lod_values) {
        size_t lod_bytes = s->cl.plan.levels.ends[s->cl.plan.nlod - 1] * bpe;
        memset(s->lod_values, 0, lod_bytes);
      }

      // Periodic metadata update.
      if (s->shard_sink->update_dim0) {
        struct platform_clock peek = s->metadata_update_clock;
        float elapsed = platform_toc(&peek);
        if (elapsed >= s->config.metadata_update_interval_s) {
          s->metadata_update_clock = peek;
          const struct dimension* dims = s->config.dimensions;
          for (int lv = 0; lv < s->levels.nlod; ++lv) {
            struct shard_state* ss = &s->shard[lv];
            uint64_t d0c =
              ss->shard_epoch * ss->chunks_per_shard_0 + ss->epoch_in_shard;
            if (s->shard_sink->update_dim0(
                  s->shard_sink, (uint8_t)lv, d0c * dims[0].chunk_size))
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
  if (s->cursor % s->layout.epoch_elements != 0) {
    uint32_t active_mask = 1;
    if (s->levels.enable_multiscale) {
      if (scatter_epoch(s, s->batch_accumulated, &active_mask))
        return writer_error();
    }
    s->batch_active_masks[s->batch_accumulated] = active_mask;
    s->batch_accumulated++;
  }

  // Flush any accumulated batch (partial or full).
  if (s->batch_accumulated > 0) {
    if (flush_batch(s, s->batch_accumulated, s->batch_active_masks))
      return writer_error();
    s->batch_accumulated = 0;
  }

  // Drain any partial dim0 accumulators (levels that haven't emitted yet).
  if (s->levels.dim0_downsample && s->dim0_accum) {
    const size_t bpe = dtype_bpe(s->config.dtype);
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
        // Drain always uses batch slot 0 (batch was just reset).
        const struct tile_stream_layout* layout_lv = &s->cl.layouts[lv];
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
      dim0_bytes += s->cl.plan.batch_count * s->cl.plan.lod_nelem[lv] * bpe;
    accumulate_metric_ms(&s->metrics.lod_dim0_fold, dim0_ms, dim0_bytes, 0);

    // Compress + aggregate + deliver drained levels (single-epoch batch).
    if (drain_mask) {
      s->batch_active_masks[0] = drain_mask;
      if (flush_batch(s, 1, s->batch_active_masks))
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
        if (finalize_shards(&s->shard[lv], s->config.shard_alignment))
          return writer_error();
      }
    }

    float emit_ms = (float)(platform_toc(&emit_clk) * 1000.0);
    accumulate_metric_ms(&s->metrics.sink, emit_ms, 0, 0);
  }

  // Final metadata.
  if (s->shard_sink->update_dim0) {
    const struct dimension* dims = s->config.dimensions;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      uint64_t dim0_extent = dim0_chunks[lv] * dims[0].chunk_size;
      if (s->shard_sink->update_dim0(
            s->shard_sink, (uint8_t)lv, dim0_extent))
        return writer_error();
    }
  }

  return writer_ok();
}

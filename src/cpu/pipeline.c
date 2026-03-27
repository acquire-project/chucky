#include "cpu/pipeline.h"

#include "cpu/aggregate.h"
#include "cpu/compress.h"
#include "cpu/lod.h"
#include "platform/platform.h"
#include "util/index.ops.h"
#include "util/metric.h"
#include "util/prelude.h"

// ---- flush_batch helpers ----

static struct aggregate_cpu_workspace
make_agg_workspace(const struct flush_level_view* lvl,
                   uint32_t* perm,
                   size_t* permuted_sizes)
{
  return (struct aggregate_cpu_workspace){
    .perm = perm,
    .permuted_sizes = permuted_sizes,
    .data = lvl->agg_slot->data,
    .data_capacity = lvl->agg_slot->data_capacity_bytes,
    .offsets = lvl->agg_slot->offsets,
    .chunk_sizes = lvl->agg_slot->chunk_sizes,
  };
}

// Deliver an aggregate result to shards, with optional metrics.
static int
deliver_aggregate(int lv,
                  const struct flush_batch_params* p,
                  const struct flush_level_view* lvl,
                  struct aggregate_result* ar,
                  uint32_t active_count)
{
  struct platform_clock sink_clk = { 0 };
  if (p->metrics)
    platform_toc(&sink_clk);

  size_t sink_bytes = 0;
  if (deliver_to_shards_batch((uint8_t)lv, lvl->shard, ar, active_count,
                              p->sink, p->shard_alignment_bytes, &sink_bytes))
    return 1;

  if (p->metrics) {
    float ms = (float)(platform_toc(&sink_clk) * 1000.0);
    accumulate_metric_ms(&p->metrics->sink, ms, sink_bytes, 0);
  }
  return 0;
}

// Aggregate + deliver using the batch path (K_l > 1 with full batch).
static int
aggregate_and_deliver_batch(int lv,
                            const struct flush_batch_params* p,
                            const struct flush_level_view* lvl,
                            uint32_t active_count)
{
  struct platform_clock clk = { 0 };
  if (p->metrics)
    platform_toc(&clk);

  struct aggregate_cpu_workspace ws =
    make_agg_workspace(lvl, lvl->batch_chunk_to_shard_map, p->shard_order_sizes_bytes);
  struct aggregate_result ar;
  if (aggregate_cpu_batch_into(p->compressed, p->comp_sizes,
                               lvl->batch_gather, lvl->agg_layout,
                               active_count, &ws, &ar))
    return 1;

  if (p->metrics) {
    uint64_t batch_C =
      (uint64_t)active_count * lvl->agg_layout->covering_count;
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&p->metrics->aggregate, ms, ar.offsets[batch_C], 0);
  }

  return deliver_aggregate(lv, p, lvl, &ar, active_count);
}

// Aggregate + deliver one epoch using the per-epoch fallback path.
static int
aggregate_and_deliver_epoch(int lv,
                            const struct flush_batch_params* p,
                            const struct flush_level_view* lvl,
                            uint64_t comp_base)
{
  struct platform_clock clk = { 0 };
  if (p->metrics)
    platform_toc(&clk);

  const void* comp_lv =
    (const char*)p->compressed + comp_base * p->max_output_size_bytes;
  const size_t* sizes_lv = p->comp_sizes + comp_base;

  struct aggregate_cpu_workspace ws =
    make_agg_workspace(lvl, lvl->chunk_to_shard_map, p->shard_order_sizes_bytes);
  struct aggregate_result ar;
  if (aggregate_cpu_into(comp_lv, sizes_lv, lvl->agg_layout, &ws, &ar))
    return 1;

  if (p->metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    size_t agg_bytes = ar.offsets[lvl->agg_layout->covering_count];
    accumulate_metric_ms(&p->metrics->aggregate, ms, agg_bytes, 0);
  }

  return deliver_aggregate(lv, p, lvl, &ar, 1);
}

// ---- flush_batch ----

int
cpu_pipeline_flush_batch(const struct flush_batch_params* p,
                         uint32_t n_epochs,
                         const uint32_t* active_masks)
{
  const uint64_t total_chunks = p->total_chunks;

  // Compress all K epochs at once (pool is contiguous).
  {
    struct platform_clock clk = { 0 };
    if (p->metrics)
      platform_toc(&clk);

    if (compress_cpu(p->codec, p->chunk_pool, p->chunk_stride_bytes,
                     p->compressed, p->max_output_size_bytes, p->comp_sizes,
                     p->chunk_bytes, n_epochs * total_chunks))
      return 1;

    if (p->metrics) {
      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(
        &p->metrics->compress, ms, n_epochs * total_chunks * p->chunk_bytes, 0);
    }
  }

  // Aggregate + deliver per-level.
  for (int lv = 0; lv < p->nlod; ++lv) {
    const struct flush_level_view* lvl = &p->levels[lv];

    uint32_t active_count = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (active_masks[e] & (1u << lv))
        active_count++;
    if (active_count == 0)
      continue;

    if (active_count == lvl->batch_active_count &&
        lvl->batch_active_count > 1) {
      if (aggregate_and_deliver_batch(lv, p, lvl, active_count))
        return 1;
    } else {
      for (uint32_t e = 0; e < n_epochs; ++e) {
        if (!(active_masks[e] & (1u << lv)))
          continue;
        uint64_t comp_base =
          (uint64_t)e * total_chunks + lvl->chunk_offset;
        if (aggregate_and_deliver_epoch(lv, p, lvl, comp_base))
          return 1;
      }
    }
  }

  return 0;
}

// ---- scatter_epoch ----

int
cpu_pipeline_scatter_epoch(const struct scatter_epoch_params* p,
                           uint32_t epoch_in_batch,
                           uint32_t* out_mask)
{
  const size_t bytes_per_element = dtype_bpe(p->dtype);
  const struct level_geometry* levels = &p->cl->levels;
  void* epoch_pool =
    (char*)p->chunk_pool + (uint64_t)epoch_in_batch *
                             levels->total_chunks *
                             p->cl->layouts[0].chunk_stride * bytes_per_element;

  if (!levels->enable_multiscale) {
    *out_mask = 1;
    return 0;
  }

  // Multiscale path: scatter linear → morton, reduce, append fold/emit,
  // then scatter each level to chunk pool.
  struct platform_clock clk = { 0 };
  if (p->metrics)
    platform_toc(&clk);

  CHECK(Error,
        lod_cpu_gather(&p->cl->plan,
                       p->linear,
                       p->lod_values,
                       p->scatter_lut,
                       p->scatter_batch_offsets,
                       p->dtype) == 0);

  if (p->metrics) {
    float scatter_ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&p->metrics->lod_gather,
                         scatter_ms,
                         p->cl->layouts[0].epoch_elements * bytes_per_element,
                         0);
  }

  if (p->metrics)
    platform_toc(&clk);

  CHECK(Error,
        lod_cpu_reduce(&p->cl->plan,
                       p->lod_values,
                       p->dtype,
                       p->reduce_method) == 0);

  if (p->metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&p->metrics->lod_reduce,
                         ms,
                         p->cl->plan.levels.ends[p->cl->plan.nlod - 1] * bytes_per_element,
                         0);
  }

  // Append fold/emit: accumulate levels 1+ across epochs.
  // Without append downsample, all inner LOD levels are ready every epoch.
  const int append_downsample = p->cl->dims.append_downsample;
  uint32_t active_levels_mask =
    (append_downsample && p->append_accum)
      ? 1
      : (uint32_t)((1u << levels->nlod) - 1);
  if (append_downsample && p->append_accum) {
    struct platform_clock append_clk = { 0 };
    if (p->metrics)
      platform_toc(&append_clk);

    CHECK(Error,
          lod_cpu_append_fold(&p->cl->plan,
                            p->lod_values,
                            p->append_accum,
                            p->append_counts,
                            p->dtype,
                            p->append_reduce_method) == 0);

    for (int lv = 1; lv < p->cl->plan.nlod; ++lv) {
      p->append_counts[lv]++;
      uint32_t period = 1u << lv;
      if (p->append_counts[lv] >= period) {
        CHECK(Error,
              lod_cpu_append_emit(&p->cl->plan,
                                p->lod_values,
                                p->append_accum,
                                lv,
                                p->append_counts[lv],
                                p->dtype,
                                p->append_reduce_method) == 0);
        p->append_counts[lv] = 0;
        active_levels_mask |= (1u << lv);
      }
    }

    if (p->metrics) {
      float append_ms = (float)(platform_toc(&append_clk) * 1000.0);
      size_t append_bytes = 0;
      for (int lv = 1; lv < p->cl->plan.nlod; ++lv)
        append_bytes += p->cl->plan.batch_count * p->cl->plan.lod_nelem[lv] * bytes_per_element;
      accumulate_metric_ms(
        &p->metrics->lod_append_fold, append_ms, append_bytes, 0);
    }
  }

  if (p->metrics)
    platform_toc(&clk);

  for (int lv = 0; lv < levels->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;
    const struct tile_stream_layout* layout = &p->cl->layouts[lv];

    CHECK(Error,
          lod_cpu_morton_to_chunks(&p->cl->plan,
                                   p->lod_values,
                                   epoch_pool,
                                   lv,
                                   layout,
                                   p->morton_lut[lv],
                                   p->lod_batch_offsets[lv],
                                   p->dtype) == 0);
  }

  if (p->metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(
      &p->metrics->lod_morton_chunk,
      ms,
      levels->total_chunks * p->cl->layouts[0].chunk_stride * bytes_per_element,
      0);
  }

  *out_mask = active_levels_mask;
  return 0;

Error:
  return 1;
}

// ---- LUT computation ----

void
cpu_pipeline_compute_luts(const struct computed_stream_layouts* cl,
                          const struct level_geometry* levels,
                          const uint32_t batch_active_count[LOD_MAX_LEVELS],
                          const struct aggregate_layout agg_layout[LOD_MAX_LEVELS],
                          struct lut_targets* out)
{
  // Per-level chunk-to-shard map + batch LUTs.
  for (int lv = 0; lv < levels->nlod; ++lv) {
    const struct aggregate_layout* agg = &agg_layout[lv];
    uint64_t M_lv = agg->chunks_per_epoch;

    // Single-epoch permutation.
    for (uint64_t i = 0; i < M_lv; ++i)
      out->chunk_to_shard_map[lv][i] = (uint32_t)ravel(
        agg->lifted_rank, agg->lifted_shape, agg->lifted_strides, i);

    // Batch LUTs (K_l > 1 only).
    uint32_t K_l = batch_active_count[lv];
    if (K_l > 1) {
      uint64_t total_chunks = levels->total_chunks;
      for (uint32_t a = 0; a < K_l; ++a) {
        uint32_t period =
          (cl->dims.append_downsample && lv > 0) ? (1u << lv) : 1;
        uint32_t pool_epoch = (a + 1) * period - 1;

        for (uint64_t j = 0; j < M_lv; ++j) {
          uint64_t idx = (uint64_t)a * M_lv + j;
          out->batch_gather[lv][idx] =
            (uint32_t)(pool_epoch * total_chunks +
                       levels->chunk_offset[lv] + j);
          uint32_t perm_pos = (uint32_t)ravel(
            agg->lifted_rank, agg->lifted_shape, agg->lifted_strides, j);
          out->batch_chunk_to_shard_map[lv][idx] = perm_pos * K_l + a;
        }
      }
    }
  }

  // LOD LUTs (multiscale only).
  if (levels->enable_multiscale) {
    const struct lod_plan* plan = &cl->plan;

    lod_cpu_build_scatter_lut(plan, out->scatter_lut);
    lod_cpu_build_scatter_batch_offsets(plan, out->scatter_batch_offsets);

    for (int lv = 0; lv < levels->nlod; ++lv) {
      const struct tile_stream_layout* layout_lv = &cl->layouts[lv];
      lod_cpu_build_chunk_lut(plan, lv, layout_lv, out->morton_lut[lv]);

      // Convert flat batch index → lifted-space chunk pool offset.
      // Decomposes bi into per-dimension coordinates, then maps each
      // coordinate to (chunk_index, within-chunk) in lifted space.
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
        out->lod_batch_offsets[lv][bi] =
          (uint64_t)offset +
          levels->chunk_offset[lv] * layout_lv->chunk_stride;
      }
    }
  }
}

// ---- append drain ----

int
cpu_pipeline_append_drain(const struct append_drain_params* p,
                          uint32_t* out_drain_mask)
{
  const size_t bytes_per_element = dtype_bpe(p->dtype);
  const struct lod_plan* plan = &p->cl->plan;

  struct platform_clock append_clk = { 0 };
  if (p->metrics)
    platform_toc(&append_clk);

  uint32_t drain_mask = 0;
  for (int lv = 1; lv < plan->nlod; ++lv) {
    if (p->append_counts[lv] > 0) {
      CHECK(Error,
            lod_cpu_append_emit(plan,
                              p->lod_values,
                              p->append_accum,
                              lv,
                              p->append_counts[lv],
                              p->dtype,
                              p->append_reduce_method) == 0);
      p->append_counts[lv] = 0;

      // Scatter emitted level from morton space to chunk pool.
      const struct tile_stream_layout* layout_lv = &p->cl->layouts[lv];
      CHECK(Error,
            lod_cpu_morton_to_chunks(plan,
                                     p->lod_values,
                                     p->chunk_pool,
                                     lv,
                                     layout_lv,
                                     p->morton_lut[lv],
                                     p->lod_batch_offsets[lv],
                                     p->dtype) == 0);
      drain_mask |= (1u << lv);
    }
  }

  if (p->metrics) {
    float append_ms = (float)(platform_toc(&append_clk) * 1000.0);
    size_t append_bytes = 0;
    for (int lv = 1; lv < plan->nlod; ++lv)
      append_bytes += plan->batch_count * plan->lod_nelem[lv] * bytes_per_element;
    accumulate_metric_ms(&p->metrics->lod_append_fold, append_ms, append_bytes, 0);
  }

  *out_drain_mask = drain_mask;
  return 0;

Error:
  return 1;
}

#include "cpu/pipeline.h"

#include "cpu/aggregate.h"
#include "cpu/compress.h"
#include "cpu/lod.h"
#include "platform/platform.h"
#include "util/index.ops.h"
#include "util/metric.h"
#include "util/prelude.h"

// ---- flush_batch ----

int
cpu_pipeline_flush_batch(const struct flush_batch_params* p,
                         uint32_t n_epochs,
                         const uint32_t* active_masks)
{
  const size_t max_out = p->max_output_size;
  const uint64_t total_chunks = p->total_chunks;

  // Compress all K epochs at once (pool is contiguous).
  {
    struct platform_clock clk = { 0 };
    if (p->metrics)
      platform_toc(&clk);

    if (compress_cpu(p->codec,
                     p->chunk_pool,
                     p->chunk_stride_bytes,
                     p->compressed,
                     max_out,
                     p->comp_sizes,
                     p->chunk_bytes,
                     n_epochs * total_chunks))
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

    // Count active epochs for this level.
    uint32_t active_count = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (active_masks[e] & (1u << lv))
        active_count++;
    if (active_count == 0)
      continue;

    if (active_count == lvl->batch_active_count &&
        lvl->batch_active_count > 1) {
      // Batch aggregate.
      struct platform_clock clk = { 0 };
      if (p->metrics)
        platform_toc(&clk);

      struct aggregate_cpu_workspace ws = {
        .perm = lvl->batch_chunk_to_shard_map,
        .permuted_sizes = p->shard_order_sizes,
        .data = lvl->agg_slot->data,
        .data_capacity = lvl->agg_slot->data_capacity,
        .offsets = lvl->agg_slot->offsets,
        .chunk_sizes = lvl->agg_slot->chunk_sizes,
      };
      struct aggregate_result ar;
      if (aggregate_cpu_batch_into(p->compressed,
                                   p->comp_sizes,
                                   lvl->batch_gather,
                                   lvl->agg_layout,
                                   active_count,
                                   &ws,
                                   &ar))
        return 1;

      if (p->metrics) {
        uint64_t batch_C =
          (uint64_t)active_count * lvl->agg_layout->covering_count;
        float ms = (float)(platform_toc(&clk) * 1000.0);
        accumulate_metric_ms(&p->metrics->aggregate, ms, ar.offsets[batch_C], 0);
      }

      // Batch deliver.
      struct platform_clock sink_clk = { 0 };
      if (p->metrics)
        platform_toc(&sink_clk);

      size_t sink_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  lvl->shard,
                                  &ar,
                                  active_count,
                                  p->sink,
                                  p->shard_alignment,
                                  &sink_bytes))
        return 1;

      if (p->metrics) {
        float ms = (float)(platform_toc(&sink_clk) * 1000.0);
        accumulate_metric_ms(&p->metrics->sink, ms, sink_bytes, 0);
      }
    } else {
      // Per-epoch fallback.
      for (uint32_t e = 0; e < n_epochs; ++e) {
        if (!(active_masks[e] & (1u << lv)))
          continue;

        uint64_t comp_base =
          (uint64_t)e * total_chunks + lvl->chunk_offset;

        struct platform_clock clk = { 0 };
        if (p->metrics)
          platform_toc(&clk);

        const void* comp_lv = (const char*)p->compressed + comp_base * max_out;
        const size_t* sizes_lv = p->comp_sizes + comp_base;

        struct aggregate_cpu_workspace ws = {
          .perm = lvl->chunk_to_shard_map,
          .permuted_sizes = p->shard_order_sizes,
          .data = lvl->agg_slot->data,
          .data_capacity = lvl->agg_slot->data_capacity,
          .offsets = lvl->agg_slot->offsets,
          .chunk_sizes = lvl->agg_slot->chunk_sizes,
        };
        struct aggregate_result ar;
        if (aggregate_cpu_into(
              comp_lv, sizes_lv, lvl->agg_layout, &ws, &ar))
          return 1;

        if (p->metrics) {
          float ms = (float)(platform_toc(&clk) * 1000.0);
          size_t agg_bytes = ar.offsets[lvl->agg_layout->covering_count];
          accumulate_metric_ms(&p->metrics->aggregate, ms, agg_bytes, 0);
        }

        // Deliver.
        struct platform_clock sink_clk = { 0 };
        if (p->metrics)
          platform_toc(&sink_clk);

        size_t sink_bytes = 0;
        if (deliver_to_shards_batch((uint8_t)lv,
                                    lvl->shard,
                                    &ar,
                                    1,
                                    p->sink,
                                    p->shard_alignment,
                                    &sink_bytes))
          return 1;

        if (p->metrics) {
          float ms = (float)(platform_toc(&sink_clk) * 1000.0);
          accumulate_metric_ms(&p->metrics->sink, ms, sink_bytes, 0);
        }
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
  const size_t bpe = dtype_bpe(p->dtype);
  const struct level_geometry* levels = &p->cl->levels;
  void* epoch_pool =
    (char*)p->chunk_pool + (uint64_t)epoch_in_batch *
                             levels->total_chunks *
                             p->cl->layouts[0].chunk_stride * bpe;

  if (!levels->enable_multiscale) {
    *out_mask = 1;
    return 0;
  }

  // Multiscale path: scatter linear → morton, reduce, dim0 fold/emit,
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
                         p->cl->layouts[0].epoch_elements * bpe,
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
                         p->cl->plan.levels.ends[p->cl->plan.nlod - 1] * bpe,
                         0);
  }

  // Dim0 fold/emit: accumulate levels 1+ across epochs.
  // Without dim0 downsample, all inner LOD levels are ready every epoch.
  uint32_t active_levels_mask =
    (levels->dim0_downsample && p->dim0_accum)
      ? 1
      : (uint32_t)((1u << levels->nlod) - 1);
  if (levels->dim0_downsample && p->dim0_accum) {
    struct platform_clock dim0_clk = { 0 };
    if (p->metrics)
      platform_toc(&dim0_clk);

    CHECK(Error,
          lod_cpu_dim0_fold(&p->cl->plan,
                            p->lod_values,
                            p->dim0_accum,
                            p->dim0_counts,
                            p->dtype,
                            p->dim0_reduce_method) == 0);

    for (int lv = 1; lv < p->cl->plan.nlod; ++lv) {
      p->dim0_counts[lv]++;
      uint32_t period = 1u << lv;
      if (p->dim0_counts[lv] >= period) {
        CHECK(Error,
              lod_cpu_dim0_emit(&p->cl->plan,
                                p->lod_values,
                                p->dim0_accum,
                                lv,
                                p->dim0_counts[lv],
                                p->dtype,
                                p->dim0_reduce_method) == 0);
        p->dim0_counts[lv] = 0;
        active_levels_mask |= (1u << lv);
      }
    }

    if (p->metrics) {
      float dim0_ms = (float)(platform_toc(&dim0_clk) * 1000.0);
      size_t dim0_bytes = 0;
      for (int lv = 1; lv < p->cl->plan.nlod; ++lv)
        dim0_bytes += p->cl->plan.batch_count * p->cl->plan.lod_nelem[lv] * bpe;
      accumulate_metric_ms(
        &p->metrics->lod_dim0_fold, dim0_ms, dim0_bytes, 0);
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
      levels->total_chunks * p->cl->layouts[0].chunk_stride * bpe,
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
    const struct aggregate_layout* al = &agg_layout[lv];
    uint64_t M_lv = al->chunks_per_epoch;

    // Single-epoch permutation.
    for (uint64_t i = 0; i < M_lv; ++i)
      out->chunk_to_shard_map[lv][i] = (uint32_t)ravel(
        al->lifted_rank, al->lifted_shape, al->lifted_strides, i);

    // Batch LUTs (K_l > 1 only).
    uint32_t K_l = batch_active_count[lv];
    if (K_l > 1) {
      uint64_t total_chunks = levels->total_chunks;
      for (uint32_t a = 0; a < K_l; ++a) {
        uint32_t period =
          (levels->dim0_downsample && lv > 0) ? (1u << lv) : 1;
        uint32_t pool_epoch = (a + 1) * period - 1;

        for (uint64_t j = 0; j < M_lv; ++j) {
          uint64_t idx = (uint64_t)a * M_lv + j;
          out->batch_gather[lv][idx] =
            (uint32_t)(pool_epoch * total_chunks +
                       levels->chunk_offset[lv] + j);
          uint32_t perm_pos = (uint32_t)ravel(
            al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
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

// ---- dim0 drain ----

int
cpu_pipeline_dim0_drain(const struct dim0_drain_params* p,
                        uint32_t* out_drain_mask)
{
  const size_t bpe = dtype_bpe(p->dtype);
  const struct lod_plan* plan = &p->cl->plan;

  struct platform_clock dim0_clk = { 0 };
  if (p->metrics)
    platform_toc(&dim0_clk);

  uint32_t drain_mask = 0;
  for (int lv = 1; lv < plan->nlod; ++lv) {
    if (p->dim0_counts[lv] > 0) {
      CHECK(Error,
            lod_cpu_dim0_emit(plan,
                              p->lod_values,
                              p->dim0_accum,
                              lv,
                              p->dim0_counts[lv],
                              p->dtype,
                              p->dim0_reduce_method) == 0);
      p->dim0_counts[lv] = 0;

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
    float dim0_ms = (float)(platform_toc(&dim0_clk) * 1000.0);
    size_t dim0_bytes = 0;
    for (int lv = 1; lv < plan->nlod; ++lv)
      dim0_bytes += plan->batch_count * plan->lod_nelem[lv] * bpe;
    accumulate_metric_ms(&p->metrics->lod_dim0_fold, dim0_ms, dim0_bytes, 0);
  }

  *out_drain_mask = drain_mask;
  return 0;

Error:
  return 1;
}

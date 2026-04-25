#include "cpu/pipeline.h"

#include "cpu/aggregate.h"
#include "cpu/compress.h"
#include "cpu/lod.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <stdlib.h>

// ---- flush_batch helpers ----

// Deliver one LOD's aggregate result to its shards, with optional metrics.
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
  if (deliver_to_shards_batch((uint8_t)lv,
                              lvl->shard,
                              ar,
                              active_count,
                              p->sink,
                              p->shard_alignment_bytes,
                              &sink_bytes))
    return 1;

  if (p->metrics) {
    float ms = (float)(platform_toc(&sink_clk) * 1000.0);
    accumulate_metric_ms(&p->metrics->sink, ms, sink_bytes, 0);
  }
  return 0;
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

    if (compress_cpu(p->codec,
                     p->chunk_pool,
                     p->chunk_stride_bytes,
                     p->compressed,
                     p->max_output_size_bytes,
                     p->comp_sizes,
                     p->chunk_bytes,
                     n_epochs * total_chunks,
                     p->bytes_per_element,
                     p->nthreads))
      return 1;

    if (p->metrics) {
      float ms = (float)(platform_toc(&clk) * 1000.0);
      accumulate_metric_ms(
        &p->metrics->compress, ms, n_epochs * total_chunks * p->chunk_bytes, 0);
    }
  }

  // Build per-LOD pool_epochs from active_masks. Each LOD's active count
  // depends on which epochs in the batch flagged that LOD. The shared
  // pool_epochs_scratch is sized [LOD_MAX_LEVELS * K]; carve a per-LOD
  // slice of length n_epochs (K is always >= n_epochs).
  uint32_t per_lod_n_active[LOD_MAX_LEVELS] = { 0 };
  const uint32_t* pool_epochs[LOD_MAX_LEVELS];
  for (int lv = 0; lv < p->nlod; ++lv) {
    uint32_t* dst = p->pool_epochs_scratch + (size_t)lv * n_epochs;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (active_masks[e] & (1u << lv))
        dst[k++] = e;
    per_lod_n_active[lv] = k;
    pool_epochs[lv] = dst;
  }

  // Build the unified batch layout for this kick.
  struct batch_aggregate_layout layout;
  if (batch_aggregate_layout_init(&layout,
                                  p->per_lod_agg_layouts,
                                  per_lod_n_active,
                                  (uint8_t)p->nlod,
                                  p->page_size))
    return 1;

  // Pick the slot to write into and wait on its previous use.
  const uint8_t cur = *p->agg_current;
  struct cpu_agg_slot* slot = &p->agg_slots[cur];

  if (slot->data_capacity_bytes < layout.total_data_bytes)
    return 1;

  if (p->sink->wait_fence) {
    struct platform_clock fence_clk = { 0 };
    if (p->metrics)
      platform_toc(&fence_clk);
    p->sink->wait_fence(p->sink, 0, p->io_done[cur]);
    if (p->metrics) {
      float fence_ms = (float)(platform_toc(&fence_clk) * 1000.0);
      accumulate_metric_ms(&p->metrics->io_fence_stall, fence_ms, 0, 0);
    }
  }

  if (p->sink->has_error && p->sink->has_error(p->sink))
    return 1;

  // Skip the rest of the batch when no LOD has active epochs (e.g. a
  // partial flush where every level was already drained).
  if (layout.total_batch_chunks == 0)
    return 0;

  // Build unified gather + perm + source_lod into the slot's scratch.
  aggregate_batch_luts_unified(&layout,
                               p->per_lod_agg_layouts,
                               p->levels_geo,
                               pool_epochs,
                               slot->gather,
                               slot->perm,
                               slot->source_lod);

  // Aggregate (single OpenMP loop spans all LODs in pass 3).
  struct platform_clock agg_clk = { 0 };
  if (p->metrics)
    platform_toc(&agg_clk);

  struct aggregate_cpu_workspace ws = {
    .perm = slot->perm,
    .permuted_sizes = slot->permuted_sizes,
    .offsets = slot->offsets,
    .chunk_sizes = slot->chunk_sizes,
    .data = slot->data,
    .data_capacity = slot->data_capacity_bytes,
  };
  struct aggregate_result per_lod_results[LOD_MAX_LEVELS];
  if (aggregate_cpu_batch_into_unified(p->compressed,
                                       p->comp_sizes,
                                       slot->gather,
                                       slot->source_lod,
                                       &layout,
                                       &ws,
                                       per_lod_results,
                                       p->nthreads))
    return 1;

  if (p->metrics) {
    float ms = (float)(platform_toc(&agg_clk) * 1000.0);
    size_t agg_bytes = layout.total_data_bytes;
    accumulate_metric_ms(&p->metrics->aggregate, ms, agg_bytes, 0);
  }

  // Per-LOD deliver. Sink dispatches by level, so this stays per-LOD.
  for (int lv = 0; lv < p->nlod; ++lv) {
    const struct flush_level_view* lvl = &p->levels[lv];
    const uint32_t active_count = per_lod_n_active[lv];
    if (active_count == 0)
      continue;
    if (deliver_aggregate(lv, p, lvl, &per_lod_results[lv], active_count))
      return 1;
  }

  // Record fence on the just-delivered slot so the next slot reuse waits
  // for the union of all LODs' IO. FS sink ignores the level argument;
  // multiscale dispatches by level — record on level 0 covers the FS case
  // and is the conservative choice for any per-level-queue sink.
  if (p->sink->record_fence)
    p->io_done[cur] = p->sink->record_fence(p->sink, 0);

  // Next batch uses the other slot.
  *p->agg_current = cur ^ 1;

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
    (char*)p->chunk_pool + (uint64_t)epoch_in_batch * levels->total_chunks *
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
                       p->scatter_fixed_dims_offsets,
                       p->dtype,
                       p->nthreads) == 0);

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
                       p->csrs,
                       p->lod_values,
                       p->dtype,
                       p->reduce_method,
                       p->nthreads) == 0);

  if (p->metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(
      &p->metrics->lod_reduce,
      ms,
      p->cl->plan.level_spans.ends[p->cl->plan.levels.nlod - 1] *
        bytes_per_element,
      0);
  }

  // Append fold/emit: accumulate levels 1+ across epochs.
  // Without append downsample, all inner LOD levels are ready every epoch.
  const int append_downsample = p->cl->dims.append_downsample;
  uint32_t active_levels_mask = (append_downsample && p->append_accum)
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
                              p->append_reduce_method,
                              p->nthreads) == 0);

    for (int lv = 1; lv < p->cl->plan.levels.nlod; ++lv) {
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
                                  p->append_reduce_method,
                                  p->nthreads) == 0);
        p->append_counts[lv] = 0;
        active_levels_mask |= (1u << lv);
      }
    }

    if (p->metrics) {
      float append_ms = (float)(platform_toc(&append_clk) * 1000.0);
      size_t append_bytes = 0;
      for (int lv = 1; lv < p->cl->plan.levels.nlod; ++lv)
        append_bytes += p->cl->plan.levels.level[lv].fixed_dims_count *
                        p->cl->plan.levels.level[lv].lod_nelem *
                        bytes_per_element;
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
                                   p->lod_fixed_dims_offsets[lv],
                                   p->dtype,
                                   p->nthreads) == 0);
  }

  if (p->metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&p->metrics->lod_morton_chunk,
                         ms,
                         levels->total_chunks * p->cl->layouts[0].chunk_stride *
                           bytes_per_element,
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
                          int nthreads,
                          struct lut_targets* out)
{
  (void)cl;

  // LOD LUTs (multiscale only).
  if (levels->enable_multiscale) {
    const struct lod_plan* plan = &cl->plan;

    lod_cpu_build_scatter_lut(plan, out->scatter_lut, nthreads);
    lod_cpu_build_scatter_fixed_dims_offsets(
      plan, out->scatter_fixed_dims_offsets, nthreads);

    for (int lv = 0; lv < levels->nlod; ++lv) {
      const struct tile_stream_layout* layout_lv = &cl->layouts[lv];
      lod_cpu_build_chunk_lut(
        plan, lv, layout_lv, out->morton_lut[lv], nthreads);

      // Convert flat batch index → lifted-space chunk pool offset.
      {
        const struct level_dims* ld = &plan->levels.level[lv];

        for (uint64_t bi = 0; bi < ld->fixed_dims_count; ++bi) {
          uint64_t remainder = bi;
          int64_t offset = 0;
          for (int k = ld->fixed_dims_ndim - 1; k >= 0; --k) {
            uint64_t coord = remainder % ld->fixed_dims_shape[k];
            remainder /= ld->fixed_dims_shape[k];
            int d = ld->fixed_dim_to_dim[k];
            uint64_t cs = layout_lv->lifted_shape[2 * d + 1];
            uint64_t ci = coord / cs;
            uint64_t wi = coord % cs;
            offset += (int64_t)ci * layout_lv->lifted_strides[2 * d];
            offset += (int64_t)wi * layout_lv->lifted_strides[2 * d + 1];
          }
          out->lod_fixed_dims_offsets[lv][bi] =
            (uint64_t)offset +
            levels->level[lv].chunk_offset * layout_lv->chunk_stride;
        }
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
  for (int lv = 1; lv < plan->levels.nlod; ++lv) {
    if (p->append_counts[lv] > 0) {
      CHECK(Error,
            lod_cpu_append_emit(plan,
                                p->lod_values,
                                p->append_accum,
                                lv,
                                p->append_counts[lv],
                                p->dtype,
                                p->append_reduce_method,
                                p->nthreads) == 0);
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
                                     p->lod_fixed_dims_offsets[lv],
                                     p->dtype,
                                     p->nthreads) == 0);
      drain_mask |= (1u << lv);
    }
  }

  if (p->metrics) {
    float append_ms = (float)(platform_toc(&append_clk) * 1000.0);
    size_t append_bytes = 0;
    for (int lv = 1; lv < plan->levels.nlod; ++lv)
      append_bytes += plan->levels.level[lv].fixed_dims_count *
                      plan->levels.level[lv].lod_nelem * bytes_per_element;
    accumulate_metric_ms(
      &p->metrics->lod_append_fold, append_ms, append_bytes, 0);
  }

  *out_drain_mask = drain_mask;
  return 0;

Error:
  return 1;
}

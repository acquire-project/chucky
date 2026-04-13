#include "gpu/flush.d2h_deliver.h"
#include "gpu/flush.helpers.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <string.h>

// --- Init / Destroy ---

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 struct level_flush_state* levels,
                 int nlod,
                 size_t shard_alignment,
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->levels = levels;
  stage->nlod = nlod;
  stage->shard_alignment = shard_alignment;

  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventCreate(&stage->t_d2h_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->offsets_ready[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->ready[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->t_d2h_start[fc], compute));
    CU(Fail, cuEventRecord(stage->offsets_ready[fc], compute));
    CU(Fail, cuEventRecord(stage->ready[fc], compute));
  }

  return 0;

Fail:
  d2h_deliver_destroy(stage);
  return 1;
}

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage)
{
  if (!stage)
    return;
  for (int fc = 0; fc < 2; ++fc) {
    cu_event_destroy(stage->t_d2h_start[fc]);
    cu_event_destroy(stage->offsets_ready[fc]);
    cu_event_destroy(stage->ready[fc]);
  }
  // levels is borrowed, not destroyed here
}

// --- Internal helpers ---

// Wait for pending IO fences on aggregate slots before reuse.
// Accumulates wall time into stage->metrics->io_fence_stall (if non-NULL).
static void
wait_io_fences(const struct d2h_deliver_stage* stage,
               int fc,
               uint32_t level_mask,
               struct shard_sink* sink)
{
  if (!sink->wait_fence)
    return;
  struct platform_clock clk = { 0 };
  platform_toc(&clk);
  for (int lv = 0; lv < stage->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;
    struct aggregate_slot* agg = &stage->levels[lv].agg[fc];
    if (agg->io_done.seq > 0)
      sink->wait_fence(sink, (uint8_t)lv, agg->io_done);
  }
  if (stage->metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&stage->metrics->io_fence_stall, ms, 0, 0);
  }
}

// Phase 1: D2H offsets only (non-blocking — no host sync).
// Records offsets_ready[fc] when offsets are on the host.
static int
kick_offset_d2h(struct d2h_deliver_stage* stage,
                const struct flush_handoff* handoff,
                const struct level_geometry* levels,
                const struct batch_state* batch,
                const struct dim_info* dims,
                CUstream d2h_stream)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;
  const uint32_t level_mask = handoff->active_levels_mask;

  for (int lv = 0; lv < levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;

    struct level_flush_state* lvl = &stage->levels[lv];

    uint32_t active_count = 1;
    if (n_epochs > 1) {
      active_count = level_actual_active_count(
        lvl, batch, dims, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;
    }

    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t covering = (uint64_t)active_count * lvl->agg_layout.covering_count;

    CU(Error,
       cuMemcpyDtoHAsync(agg->h_offsets,
                         (CUdeviceptr)agg->d_offsets,
                         (covering + 1) * sizeof(size_t),
                         d2h_stream));
  }
  CU(Error, cuEventRecord(stage->offsets_ready[fc], d2h_stream));

  return 0;

Error:
  return 1;
}

// Phase 2: sync on offsets, then D2H only the actual compressed bytes.
// Called from drain after kick has returned. Uses the stashed d2h_stream.
static int
drain_bulk_d2h(struct d2h_deliver_stage* stage,
               const struct flush_handoff* handoff,
               const struct level_geometry* levels,
               const struct batch_state* batch,
               const struct dim_info* dims,
               const struct tile_stream_configuration* config)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;
  const uint32_t level_mask = handoff->active_levels_mask;
  CUstream d2h_stream = stage->d2h_stream;

  // Wait for offset D2H to land on the host.
  {
    struct platform_clock sync_clk = { 0 };
    platform_toc(&sync_clk);
    CU(Error, cuEventSynchronize(stage->offsets_ready[fc]));
    if (stage->metrics) {
      float ms = (float)(platform_toc(&sync_clk) * 1000.0);
      accumulate_metric_ms(&stage->metrics->kick_sync_stall, ms, 0, 0);
    }
  }

  // D2H only actual compressed bytes per level.
  for (int lv = 0; lv < levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;

    struct level_flush_state* lvl = &stage->levels[lv];

    uint32_t active_count = 1;
    if (n_epochs > 1) {
      active_count = level_actual_active_count(
        lvl, batch, dims, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;
    }

    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t covering = (uint64_t)active_count * lvl->agg_layout.covering_count;

    size_t actual = agg->h_offsets[covering];
    // Add one alignment unit of slack: deliver_to_shards_batch rounds each
    // write up with align_up, so the D2H must cover at least that much.
    // The aggregate buffer was sized with this slack via agg_pool_bytes.
    actual += stage->shard_alignment;
    size_t cap =
      agg_pool_bytes((uint64_t)active_count * levels->level[lv].chunk_count,
                     handoff->max_output_size,
                     lvl->agg_layout.covering_count,
                     lvl->agg_layout.cps_inner,
                     lvl->agg_layout.page_size);
    if (actual > cap)
      actual = cap;

    CU(
      Error,
      cuMemcpyDtoHAsync(
        agg->h_aggregated, (CUdeviceptr)agg->d_aggregated, actual, d2h_stream));
    CU(Error, cuEventRecord(agg->ready, d2h_stream));
  }

  CU(Error, cuEventRecord(stage->ready[fc], d2h_stream));

  return 0;

Error:
  return 1;
}

static void
record_flush_metrics(const struct d2h_deliver_stage* stage,
                     const struct flush_handoff* handoff,
                     const struct level_geometry* levels,
                     const struct batch_state* batch,
                     const struct dim_info* dims,
                     const struct tile_stream_layout* layout,
                     const struct tile_stream_configuration* config,
                     const struct lod_state* lod,
                     struct stream_metrics* metrics)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;

  const struct lod_timing* t = &lod->timing[fc];
  if (levels->enable_multiscale && t->t_start) {
    const size_t bytes_per_element = dtype_bpe(config->dtype);
    const size_t scatter_bytes = layout->epoch_elements * bytes_per_element;
    const size_t morton_bytes =
      lod->plan.level_spans.ends[lod->plan.levels.nlod - 1] * bytes_per_element;
    const size_t unified_pool_bytes =
      levels->total_chunks * layout->chunk_stride * bytes_per_element;

    accumulate_metric_cu(&metrics->lod_gather,
                         t->t_start,
                         t->t_scatter_end,
                         scatter_bytes,
                         scatter_bytes);
    accumulate_metric_cu(&metrics->lod_reduce,
                         t->t_scatter_end,
                         t->t_reduce_end,
                         scatter_bytes,
                         morton_bytes);
    if (dims->append_downsample) {
      size_t accum_bpe = dtype_bpe(config->dtype);
      size_t accum_bytes = lod->append_accum.total_elements * accum_bpe;
      accumulate_metric_cu(&metrics->lod_append_fold,
                           t->t_reduce_end,
                           t->t_append_end,
                           accum_bytes,
                           accum_bytes);
    }
    accumulate_metric_cu(&metrics->lod_morton_chunk,
                         t->t_append_end,
                         t->t_end,
                         morton_bytes,
                         unified_pool_bytes);
  }

  {
    const size_t pool_bytes = (uint64_t)n_epochs * levels->total_chunks *
                              layout->chunk_stride * dtype_bpe(config->dtype);

    // Compute actual aggregated bytes first (available after D2H sync).
    size_t agg_bytes = 0;
    for (int lv = 0; lv < levels->nlod; ++lv) {
      if (!(handoff->active_levels_mask & (1u << lv)))
        continue;
      struct level_flush_state* lvl = &stage->levels[lv];
      uint32_t active_count = level_actual_active_count(
        lvl, batch, dims, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;
      uint64_t batch_covering =
        (uint64_t)active_count * lvl->agg_layout.covering_count;
      agg_bytes += lvl->agg[fc].h_offsets[batch_covering];
    }

    accumulate_metric_cu(&metrics->compress,
                         handoff->t_compress_start,
                         handoff->t_compress_end,
                         pool_bytes,
                         agg_bytes);
    accumulate_metric_cu(&metrics->aggregate,
                         handoff->t_compress_end,
                         handoff->t_aggregate_end,
                         agg_bytes,
                         agg_bytes);
    accumulate_metric_cu(&metrics->d2h,
                         stage->t_d2h_start[fc],
                         stage->ready[fc],
                         agg_bytes,
                         agg_bytes);
  }
}

// Wait for IO fences, issue bulk D2H (phase 2), synchronize, record metrics,
// deliver to sinks.
static struct writer_result
sync_and_deliver(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct dim_info* dims,
                 const struct tile_stream_layout* layout,
                 const struct tile_stream_configuration* config,
                 struct shard_sink* sink,
                 const struct lod_state* lod,
                 struct stream_metrics* metrics)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;

  // Wait for IO fences before overwriting h_aggregated with bulk D2H.
  // Moved from kick so that compress+aggregate can run without blocking.
  wait_io_fences(stage, fc, handoff->active_levels_mask, sink);

  // Fail fast if async IO encountered an error.
  if (sink->has_error && sink->has_error(sink))
    goto Error;

  // Phase 2: sync on offsets, issue bulk D2H, sync on bulk ready.
  CHECK(Error,
        drain_bulk_d2h(stage, handoff, levels, batch, dims, config) == 0);
  CU(Error, cuEventSynchronize(stage->ready[fc]));
  record_flush_metrics(
    stage, handoff, levels, batch, dims, layout, config, lod, metrics);

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;

    for (int lv = 0; lv < levels->nlod; ++lv) {
      if (!(handoff->active_levels_mask & (1u << lv)))
        continue;

      struct level_flush_state* lvl = &stage->levels[lv];
      uint32_t active_count = level_actual_active_count(
        lvl, batch, dims, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;

      size_t level_bytes = 0;
      struct aggregate_result ar = {
        .data = lvl->agg[fc].h_aggregated,
        .offsets = lvl->agg[fc].h_offsets,
        .chunk_sizes = lvl->agg[fc].h_permuted_sizes,
      };
      if (deliver_to_shards_batch((uint8_t)lv,
                                  &lvl->shard,
                                  &ar,
                                  active_count,
                                  sink,
                                  stage->shard_alignment,
                                  &level_bytes))
        goto Error;
      sink_bytes += level_bytes;

      if (sink->record_fence)
        lvl->agg[fc].io_done = sink->record_fence(sink, (uint8_t)lv);
    }

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&metrics->sink, sink_ms, sink_bytes, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

// Periodic metadata update (append-dim extents per level).
static int
maybe_update_metadata(const struct d2h_deliver_stage* stage,
                      const struct dim_info* dims_info,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct platform_clock* metadata_update_clock)
{
  if (!sink->update_append)
    return 0;

  struct platform_clock peek = *metadata_update_clock;
  float elapsed = platform_toc(&peek);
  if (elapsed < config->metadata_update_interval_s)
    return 0;

  *metadata_update_clock = peek;
  const uint8_t na = dim_info_n_append(dims_info);
  for (int lv = 0; lv < stage->nlod; ++lv) {
    struct shard_state* ss = &stage->levels[lv].shard;
    uint64_t flat_append_chunks =
      ss->shard_epoch * ss->chunks_per_shard_append + ss->epoch_in_shard;
    uint64_t append_sizes[HALF_MAX_RANK];
    dim_info_decompose_append_sizes(
      dims_info, flat_append_chunks, append_sizes);
    if (sink->update_append(sink, (uint8_t)lv, na, append_sizes))
      return 1;
  }
  return 0;
}

// --- Public interface ---

int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct dim_info* dims,
                 CUstream d2h_stream)
{
  const int fc = handoff->fc;

  CU(Error, cuStreamWaitEvent(d2h_stream, handoff->t_aggregate_end, 0));
  CU(Error, cuEventRecord(stage->t_d2h_start[fc], d2h_stream));
  CHECK(Error,
        kick_offset_d2h(stage, handoff, levels, batch, dims, d2h_stream) == 0);

  // Stash d2h_stream so drain can issue bulk D2H on the same stream.
  stage->d2h_stream = d2h_stream;

  return 0;

Error:
  return 1;
}

struct writer_result
d2h_deliver_drain(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  const struct dim_info* dims,
                  const struct tile_stream_layout* layout,
                  const struct tile_stream_configuration* config,
                  struct shard_sink* sink,
                  const struct lod_state* lod,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock)
{
  struct writer_result r = sync_and_deliver(
    stage, handoff, levels, batch, dims, layout, config, sink, lod, metrics);
  if (!r.error) {
    if (maybe_update_metadata(stage, dims, config, sink, metadata_update_clock))
      return writer_error();
  }
  return r;
}

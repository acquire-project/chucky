#include "flush.d2h_deliver.h"
#include "flush.helpers.h"

#include "lod.h"
#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "shard_delivery.h"

#include <string.h>

// --- Init / Destroy ---

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 struct level_flush_state* levels,
                 int nlod,
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->levels = levels;
  stage->nlod = nlod;

  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventCreate(&stage->t_d2h_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->ready[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->t_d2h_start[fc], compute));
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
    cu_event_destroy(stage->ready[fc]);
  }
  // levels is borrowed, not destroyed here
}

// --- Internal helpers ---

// Wait for pending IO fences on aggregate slots before reuse.
static void
wait_io_fences(const struct d2h_deliver_stage* stage,
               int fc,
               uint32_t level_mask,
               const struct tile_stream_configuration* config)
{
  if (!config->shard_sink->wait_fence)
    return;
  for (int lv = 0; lv < stage->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;
    struct aggregate_slot* agg = &stage->levels[lv].agg[fc];
    if (agg->io_done.seq > 0)
      config->shard_sink->wait_fence(
        config->shard_sink, (uint8_t)lv, agg->io_done);
  }
}

// Two-phase D2H: transfer offsets first (small), synchronize, then only
// actual compressed bytes.
static int
two_phase_d2h(const struct d2h_deliver_stage* stage,
              const struct flush_handoff* handoff,
              const struct level_geometry* levels,
              const struct batch_state* batch,
              const struct tile_stream_configuration* config,
              CUstream d2h_stream)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;
  const uint32_t level_mask = handoff->active_levels_mask;

  // Phase 1: D2H offsets only
  for (int lv = 0; lv < levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;

    struct level_flush_state* lvl = &stage->levels[lv];

    uint32_t active_count = 1;
    if (n_epochs > 1) {
      active_count = level_actual_active_count(
        lvl, batch, levels, handoff->batch_active_masks, lv, n_epochs);
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
  CU(Error, cuEventRecord(stage->ready[fc], d2h_stream));
  CU(Error, cuEventSynchronize(stage->ready[fc]));

  // Phase 2: D2H only actual compressed bytes per level
  for (int lv = 0; lv < levels->nlod; ++lv) {
    if (!(level_mask & (1u << lv)))
      continue;

    struct level_flush_state* lvl = &stage->levels[lv];

    uint32_t active_count = 1;
    if (n_epochs > 1) {
      active_count = level_actual_active_count(
        lvl, batch, levels, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;
    }

    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t covering = (uint64_t)active_count * lvl->agg_layout.covering_count;

    size_t actual = agg->h_offsets[covering];
    if (config->shard_alignment > 0)
      actual += config->shard_alignment;
    size_t cap =
      agg_pool_bytes((uint64_t)active_count * levels->chunk_count[lv],
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
                     const struct tile_stream_layout* layout,
                     const struct tile_stream_configuration* config,
                     const struct lod_state* lod,
                     struct stream_metrics* metrics)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;

  if (levels->enable_multiscale && lod->t_start) {
    const size_t bpe = lod_dtype_bpe(config->dtype);
    const size_t scatter_bytes = layout->epoch_elements * bpe;
    const size_t morton_bytes = lod->plan.levels.ends[lod->plan.nlod - 1] * bpe;
    const size_t unified_pool_bytes =
      levels->total_chunks * layout->chunk_stride * bpe;

    accumulate_metric_cu(
      &metrics->lod_gather, lod->t_start, lod->t_scatter_end, scatter_bytes);
    accumulate_metric_cu(&metrics->lod_reduce,
                         lod->t_scatter_end,
                         lod->t_reduce_end,
                         morton_bytes);
    if (levels->dim0_downsample) {
      size_t accum_bpe =
        lod_accum_bpe(config->dtype, config->dim0_reduce_method);
      size_t dim0_bytes = lod->dim0.total_elements * accum_bpe;
      accumulate_metric_cu(&metrics->lod_dim0_fold,
                           lod->t_reduce_end,
                           lod->t_dim0_end,
                           dim0_bytes);
    }
    accumulate_metric_cu(&metrics->lod_morton_chunk,
                         lod->t_dim0_end,
                         lod->t_end,
                         unified_pool_bytes);
  }

  {
    const size_t pool_bytes = (uint64_t)n_epochs * levels->total_chunks *
                              layout->chunk_stride *
                              lod_dtype_bpe(config->dtype);

    accumulate_metric_cu(&metrics->compress,
                         handoff->t_compress_start,
                         handoff->t_compress_end,
                         pool_bytes);

    // Use actual aggregated bytes from h_offsets (available after D2H sync).
    size_t agg_bytes = 0;
    for (int lv = 0; lv < levels->nlod; ++lv) {
      if (!(handoff->active_levels_mask & (1u << lv)))
        continue;
      struct level_flush_state* lvl = &stage->levels[lv];
      uint32_t active_count = level_actual_active_count(
        lvl, batch, levels, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;
      uint64_t batch_covering =
        (uint64_t)active_count * lvl->agg_layout.covering_count;
      agg_bytes += lvl->agg[fc].h_offsets[batch_covering];
    }

    accumulate_metric_cu(&metrics->aggregate,
                         handoff->t_compress_end,
                         handoff->t_aggregate_end,
                         agg_bytes);
    accumulate_metric_cu(
      &metrics->d2h, stage->t_d2h_start[fc], stage->ready[fc], agg_bytes);
  }
}

// Synchronize D2H, record metrics, deliver to sinks.
static struct writer_result
sync_and_deliver(const struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct tile_stream_layout* layout,
                 const struct tile_stream_configuration* config,
                 const struct lod_state* lod,
                 struct stream_metrics* metrics)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;

  CU(Error, cuEventSynchronize(stage->ready[fc]));
  record_flush_metrics(
    stage, handoff, levels, batch, layout, config, lod, metrics);

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;

    for (int lv = 0; lv < levels->nlod; ++lv) {
      if (!(handoff->active_levels_mask & (1u << lv)))
        continue;

      struct level_flush_state* lvl = &stage->levels[lv];
      uint32_t active_count = level_actual_active_count(
        lvl, batch, levels, handoff->batch_active_masks, lv, n_epochs);
      if (active_count == 0)
        continue;

      size_t level_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  &lvl->shard,
                                  &lvl->agg[fc],
                                  active_count,
                                  config->shard_sink,
                                  config->shard_alignment,
                                  &level_bytes))
        goto Error;
      sink_bytes += level_bytes;

      if (config->shard_sink->record_fence)
        lvl->agg[fc].io_done =
          config->shard_sink->record_fence(config->shard_sink, (uint8_t)lv);
    }

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&metrics->sink, sink_ms, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

// Periodic metadata update (dim0 extent per level).
static void
maybe_update_metadata(const struct d2h_deliver_stage* stage,
                      const struct tile_stream_configuration* config,
                      struct platform_clock* metadata_update_clock)
{
  if (!config->shard_sink->update_dim0)
    return;

  struct platform_clock peek = *metadata_update_clock;
  float elapsed = platform_toc(&peek);
  if (elapsed < config->metadata_update_interval_s)
    return;

  *metadata_update_clock = peek;
  const struct dimension* dims = config->dimensions;
  for (int lv = 0; lv < stage->nlod; ++lv) {
    struct shard_state* ss = &stage->levels[lv].shard;
    uint64_t dim0_chunks =
      ss->shard_epoch * ss->chunks_per_shard_0 + ss->epoch_in_shard;
    uint64_t dim0_extent = dim0_chunks * dims[0].chunk_size;
    config->shard_sink->update_dim0(
      config->shard_sink, (uint8_t)lv, dim0_extent);
  }
}

// --- Public interface ---

int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct tile_stream_configuration* config,
                 CUstream d2h_stream)
{
  const int fc = handoff->fc;

  // Wait for IO fences before reusing aggregate slots
  wait_io_fences(stage, fc, handoff->active_levels_mask, config);

  CU(Error, cuStreamWaitEvent(d2h_stream, handoff->t_aggregate_end, 0));
  CU(Error, cuEventRecord(stage->t_d2h_start[fc], d2h_stream));
  CHECK(Error,
        two_phase_d2h(stage, handoff, levels, batch, config, d2h_stream) == 0);

  return 0;

Error:
  return 1;
}

struct writer_result
d2h_deliver_drain(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  const struct tile_stream_layout* layout,
                  const struct tile_stream_configuration* config,
                  const struct lod_state* lod,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock)
{
  struct writer_result r = sync_and_deliver(
    stage, handoff, levels, batch, layout, config, lod, metrics);
  if (!r.error)
    maybe_update_metadata(stage, config, metadata_update_clock);
  return r;
}

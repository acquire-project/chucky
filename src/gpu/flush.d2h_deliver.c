#include "gpu/flush.d2h_deliver.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <string.h>

// --- Init / Destroy ---

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 enum device_aggregate_extent_kind extent_kind,
                 struct gpu_ordering* ord,
                 CUstream drain_stream,
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->shard_alignment = shard_alignment;
  return d2h_materializer_init(
    &stage->materializer, extent_kind, ord, drain_stream, compute);
}

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage)
{
  if (!stage)
    return;
  d2h_materializer_destroy(&stage->materializer);
}

// --- Internal helpers ---

static void
record_flush_metrics(const struct flush_handoff* handoff,
                     const struct host_batch* host,
                     const struct level_geometry* levels,
                     const struct tile_stream_layout* layout,
                     const struct tile_stream_configuration* config,
                     struct stream_metrics* metrics,
                     CUevent t_d2h_start,
                     CUevent t_d2h_ready)
{
  // An empty batch dispatched no kernels and moved no bytes, so it has no
  // interval to report.
  if (handoff->layout.total_batch_chunks == 0)
    return;

  const size_t pool_bytes = (uint64_t)handoff->n_epochs * levels->total_chunks *
                            layout->chunk_stride * dtype_bpe(config->dtype);

  const size_t agg_bytes = host->transfer.logical_payload_bytes;

  // Pass-through runs no codec, so it has no compress interval to report.
  if (!handoff->passthrough)
    accumulate_metric_cu_if_ready(&metrics->compress,
                                  handoff->t_compress_start,
                                  handoff->t_compress_end,
                                  pool_bytes,
                                  agg_bytes);
  // A wait, so it carries no bytes.
  accumulate_metric_cu_if_ready(&metrics->tail_gate,
                                handoff->t_compress_end,
                                handoff->t_aggregate_start,
                                0,
                                0);
  accumulate_metric_cu_if_ready(&metrics->aggregate,
                                handoff->t_aggregate_start,
                                handoff->t_aggregate_end,
                                agg_bytes,
                                agg_bytes);
  accumulate_metric_cu_if_ready(
    &metrics->d2h, t_d2h_start, t_d2h_ready, agg_bytes, agg_bytes);
}

// Deliver the drained host slot to the sink and synchronously upload the tail
// state consumed by the next page-aligned aggregation.
struct writer_result
d2h_deliver_drain_sink(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct host_batch* host,
                       CUevent payload_start,
                       struct compress_agg_array* shards,
                       const struct level_geometry* levels,
                       const struct tile_stream_layout* layout,
                       const struct tile_stream_configuration* config,
                       struct shard_sink* sink,
                       struct stream_metrics* metrics)
{
  const int fc = handoff->fc;

  record_flush_metrics(
    handoff,
    host,
    levels,
    layout,
    config,
    metrics,
    payload_start,
    gpu_ordering_event(stage->materializer.ord, GPU_EDGE_SLOT_DRAINED, fc));

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;
    const size_t page_size = shards ? shards->page_size : 0;

    if (deliver_host_batch(host,
                           handoff->shards_by_lod,
                           sink,
                           stage->shard_alignment,
                           &sink_bytes,
                           NULL))
      goto Error;

    // SYNC_MEMOPS on the destinations guarantees these copies have
    // completed at the device before they return. Returning from this function
    // is therefore the coordinator's authoritative tail-ready transition.
    if (shards && shards->total_shards > 0 && page_size > 0) {
      for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
        struct shard_state* ss = handoff->shards_by_lod[lv];
        for (uint64_t si = 0; ss && si < ss->shard_inner_count; ++si)
          shards->h_tail_bytes[shards->shards_begin[lv] + si] =
            ss->shards[si].tail_bytes;
      }
      CU(Error,
         cuMemcpyHtoD((CUdeviceptr)shards->d_tail_bytes,
                      shards->h_tail_bytes,
                      shards->total_shards * sizeof(size_t)));
      // Concatenate per-shard tail_buf slices into a single contiguous block
      // matching d_tail_carry's [total_shards * page_size] layout. We use
      // a simple loop instead of separate per-LOD HtoDs.
      for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
        struct shard_state* ss = handoff->shards_by_lod[lv];
        if (!ss || !ss->tail_buf_pool || ss->tail_buf_pool_bytes == 0)
          continue;
        const uint64_t begin = shards->shards_begin[lv];
        CUdeviceptr dst =
          shards->d_tail_carry + (CUdeviceptr)(begin * page_size);
        CU(Error,
           cuMemcpyHtoD(dst, ss->tail_buf_pool, ss->tail_buf_pool_bytes));
      }
    }

    // Record an aggregate IO fence on the unified slot; the schedule waits
    // it out before the slot's next kick.
    if (sink->record_fence)
      ((struct aggregate_slot*)host->slot_lifetime)->io_done =
        sink->record_fence(sink);

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&metrics->sink, sink_ms, sink_bytes, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

int
d2h_deliver_update_metadata(const struct flush_handoff* handoff,
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
  for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
    struct shard_state* ss = handoff->shards_by_lod[lv];
    if (ss && shard_state_publish_append(ss, sink, dims_info, lv, NULL, NULL))
      return 1;
  }
  return 0;
}

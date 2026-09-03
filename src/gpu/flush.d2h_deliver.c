#include "gpu/flush.d2h_deliver.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/shard_write_plan.h"

#include <string.h>

// --- Init / Destroy ---

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 enum aggregate_size_kind size_kind,
                 struct gpu_ordering* ordering,
                 CUstream payload_copy_stream,
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->shard_alignment = shard_alignment;
  return host_batch_copy_init(
    &stage->copy, size_kind, ordering, payload_copy_stream, compute);
}

void
d2h_deliver_destroy(struct d2h_deliver_stage* stage)
{
  if (!stage)
    return;
  host_batch_copy_destroy(&stage->copy);
}

// --- Internal helpers ---

static void
record_flush_metrics(const struct flush_handoff* handoff,
                     const struct host_batch* host,
                     const struct level_geometry* levels,
                     const struct tile_stream_layout* layout,
                     const struct tile_stream_configuration* config,
                     struct stream_metrics* metrics,
                     int variable_size,
                     CUevent t_metadata_copy_start,
                     CUevent t_metadata_copy_ready,
                     CUevent t_d2h_start,
                     CUevent t_d2h_ready)
{
  // An empty batch dispatched no kernels and moved no bytes, so it has no
  // interval to report.
  if (handoff->batch.layout.total_batch_chunks == 0)
    return;

  const size_t pool_bytes = (uint64_t)handoff->batch.epoch_count *
                            levels->total_chunks * layout->chunk_stride *
                            dtype_bpe(config->dtype);

  const size_t agg_bytes = host->transfer.payload_bytes_transferred;

  if (variable_size && host->transfer.metadata_bytes_transferred > 0) {
    accumulate_metric_cu_if_ready(&metrics->chunk_metadata_copy,
                                  t_metadata_copy_start,
                                  t_metadata_copy_ready,
                                  host->transfer.metadata_bytes_transferred,
                                  host->transfer.metadata_bytes_transferred);
  }

  // Pass-through runs no codec, so it has no compress interval to report.
  if (handoff->batch.size_kind != AGGREGATE_FIXED_SIZE)
    accumulate_metric_cu_if_ready(&metrics->compress,
                                  handoff->compress_start,
                                  handoff->compress_end,
                                  pool_bytes,
                                  agg_bytes);
  accumulate_metric_cu_if_ready(
    &metrics->aggregate,
    handoff->aggregate_start,
    gpu_ordering_event(handoff->batch.aggregate_pool->ord,
                       GPU_EDGE_AGG_DONE,
                       handoff->batch.slot_index),
    agg_bytes,
    agg_bytes);
  const size_t transferred = host->transfer.payload_bytes_transferred;
  accumulate_metric_cu_if_ready(
    &metrics->d2h, t_d2h_start, t_d2h_ready, transferred, transferred);
}

// Deliver the copied host batch to the sink. Fixed-size persistent tails
// stay entirely in shard_state; aggregation never reads or uploads them.
struct writer_result
d2h_deliver_host_batch(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct host_batch* host,
                       CUevent payload_start,
                       const struct level_geometry* levels,
                       const struct tile_stream_layout* layout,
                       const struct tile_stream_configuration* config,
                       struct shard_sink* sink,
                       struct stream_metrics* metrics)
{
  const int fc = handoff->batch.slot_index;

  record_flush_metrics(
    handoff,
    host,
    levels,
    layout,
    config,
    metrics,
    handoff->batch.size_kind == AGGREGATE_VARIABLE_SIZE,
    stage->copy.metadata_copy_start[fc],
    gpu_ordering_event(stage->copy.ordering, GPU_EDGE_CHUNK_INDEX_READY, fc),
    payload_start,
    gpu_ordering_event(stage->copy.ordering, GPU_EDGE_SLOT_COPY_DONE, fc));

  metrics->d2h_payload_bytes_transferred +=
    host->transfer.payload_bytes_transferred;
  metrics->d2h_metadata_bytes_transferred +=
    host->transfer.metadata_bytes_transferred;
  metrics->d2h_payload_copy_count += host->transfer.payload_copy_count;

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;
    const int sink_error = deliver_host_batch(
      host, handoff->shards_by_level, sink, &sink_bytes, metrics);
    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&metrics->sink, sink_ms, sink_bytes, sink_bytes);
    record_duration_ms(&metrics->delivery.payload_ready_to_writes_posted,
                       sink_ms);
    if (sink_error)
      goto Error;
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
  for (uint8_t lv = 0; lv < handoff->batch.layout.nlod; ++lv) {
    struct shard_state* ss = handoff->shards_by_level[lv];
    if (ss && shard_state_publish_append(ss, sink, dims_info, lv, NULL, NULL))
      return 1;
  }
  return 0;
}

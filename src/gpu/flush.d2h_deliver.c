#include "gpu/flush.d2h_deliver.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <string.h>

#define D2H_TRY(err_flag, name, call)                                          \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      handle_curesult(LOG_ERROR, _r, __FILE__, __LINE__, (name));              \
      (err_flag) = 1;                                                          \
    }                                                                          \
  } while (0)

// --- Init / Destroy ---

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 struct gpu_ordering* ord,
                 CUstream drain_stream,
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->ord = ord;
  stage->shard_alignment = shard_alignment;
  stage->drain_stream = drain_stream;

  // Seed timing events so the first metric reads see a valid interval.
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventCreate(&stage->t_d2h_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->t_d2h_start[fc], compute));
    CU(Fail, cuEventCreate(&stage->t_d2h_drain_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->t_d2h_drain_start[fc], compute));
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
    cu_event_destroy(stage->t_d2h_drain_start[fc]);
  }
  stage->drain_stream = NULL;
}

// --- Internal helpers ---

static void
record_flush_metrics(const struct flush_handoff* handoff,
                     const struct aggregate_slot* slot,
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

  // Aggregated bytes: sum of actual compressed chunk sizes across all LODs in
  // this batch. h_permuted_sizes carries pre-bias per-chunk sizes (with a 0
  // sentinel slot inserted per LOD); summing those gives the real D2H payload
  // regardless of the absolute/segment-relative offset semantics.
  size_t agg_bytes = 0;
  const size_t n_perm = handoff->layout.total_batch_covering + handoff->nlod;
  for (size_t i = 0; i < n_perm; ++i)
    agg_bytes += slot->h_permuted_sizes[i];

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

// `data_base` is the byte offset within h_aggregated where this LOD's first
// chunk lives. In carry-over mode that is seg->data_segment_offset; in
// contiguous mode it is the cumulative actual bytes of prior LODs.
static struct aggregate_result
lod_view(const struct flush_handoff* handoff,
         struct aggregate_slot* slot,
         uint8_t lv,
         size_t data_base)
{
  const struct lod_segment* seg = &handoff->layout.lods[lv];
  struct aggregate_result ar = {
    .data = (uint8_t*)slot->h_aggregated + data_base,
    .offsets = slot->h_offsets + seg->batch_covering_offset + lv,
    .chunk_sizes = slot->h_permuted_sizes + seg->batch_covering_offset + lv,
  };
  return ar;
}

// h_offsets must be pre-rebase (absolute, slot-relative) values here.
static int
lod_actual_bytes(const struct flush_handoff* handoff,
                 const struct aggregate_slot* slot,
                 uint8_t lv,
                 size_t* out_bytes)
{
  *out_bytes = 0;
  const struct lod_segment* seg = &handoff->layout.lods[lv];
  if (seg->n_active == 0)
    return 0;
  const uint64_t total = (uint64_t)seg->n_active * seg->covering_count;
  const size_t last = seg->batch_covering_offset + (size_t)lv + total - 1;
  const size_t end = slot->h_offsets[last] + slot->h_permuted_sizes[last];
  CHECK(Error, slot->h_offsets[last] >= seg->data_segment_offset);
  const size_t actual = end - seg->data_segment_offset;
  CHECK(Error, actual <= seg->data_segment_bytes);
  *out_bytes = actual;
  return 0;

Error:
  return 1;
}

// Sized bulk copies for the drained slot. Goes on drain_stream, never
// d2h_stream. The caller's host poll of the chunk index already proves the
// copy source is stable. Returns the dispatch error without releasing
// anything — the schedule owns the releases.
int
d2h_deliver_drain_copy(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct aggregate_slot* slot)
{
  const struct batch_aggregate_layout* alayout = &handoff->layout;
  int dispatch_err = 0;
  // Bounds the payload transfer, excluding the handoff that precedes it.
  if (cuEventRecord(stage->t_d2h_drain_start[handoff->fc],
                    stage->drain_stream) != CUDA_SUCCESS)
    return 1;
  if (alayout->page_size > 0) {
    for (uint8_t lv = 0; lv < handoff->nlod && !dispatch_err; ++lv) {
      if (handoff->per_lod_n_active[lv] == 0)
        continue;
      const struct lod_segment* seg = &alayout->lods[lv];
      size_t actual = 0;
      if (lod_actual_bytes(handoff, slot, lv, &actual))
        return 1;
      if (actual == 0)
        continue;
      D2H_TRY(dispatch_err,
              "cuMemcpyDtoHAsync",
              cuMemcpyDtoHAsync(
                (uint8_t*)slot->h_aggregated + seg->data_segment_offset,
                (CUdeviceptr)slot->d_aggregated + seg->data_segment_offset,
                actual,
                stage->drain_stream));
    }
  } else if (alayout->total_batch_covering > 0) {
    const size_t n = alayout->total_batch_covering + (size_t)handoff->nlod;
    const size_t total = slot->h_offsets[n - 1] + slot->h_permuted_sizes[n - 1];
    if (total > 0)
      D2H_TRY(dispatch_err,
              "cuMemcpyDtoHAsync",
              cuMemcpyDtoHAsync(slot->h_aggregated,
                                (CUdeviceptr)slot->d_aggregated,
                                total,
                                stage->drain_stream));
  }
  return dispatch_err;
}

// Deliver the drained host slot to the sink and synchronously upload the tail
// state consumed by the next page-aligned aggregation.
struct writer_result
d2h_deliver_drain_sink(struct d2h_deliver_stage* stage,
                       const struct flush_handoff* handoff,
                       struct aggregate_slot* slot,
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
    slot,
    levels,
    layout,
    config,
    metrics,
    handoff->passthrough ? stage->t_d2h_start[fc]
                         : stage->t_d2h_drain_start[fc],
    gpu_ordering_event(stage->ord, GPU_EDGE_SLOT_DRAINED, fc));

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;
    const size_t page_size = shards ? shards->page_size : 0;

    // In carry-over mode the bias kernel places each LOD at
    // data_segment_offset; in contiguous mode chunks pack from 0 across all
    // LODs, so LOD lv starts at h_offsets[batch_covering_offset + lv]
    // (cumulative prior-LOD bytes via the zeroed per-LOD sentinel).
    size_t data_base[LOD_MAX_LEVELS] = { 0 };
    for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
      if (handoff->per_lod_n_active[lv] == 0)
        continue;
      const struct lod_segment* seg = &handoff->layout.lods[lv];
      if (handoff->layout.page_size > 0)
        data_base[lv] = seg->data_segment_offset;
      else
        data_base[lv] = slot->h_offsets[seg->batch_covering_offset + lv];
    }

    // Rebase per-LOD offsets to be segment-relative. GPU produces absolute
    // offsets (each chunk's position in the unified d_aggregated buffer); the
    // shard_delivery contract — shared with the CPU path — pairs a
    // segment-shifted `result.data` with segment-relative offsets, so we
    // subtract the per-LOD data base from each LOD's offsets here. This
    // matches src/cpu/aggregate.c:330-338 which builds the same view shape.
    // The rebase mutates h_offsets in place; this is safe because the next
    // kick's D2H repopulates the buffer before anything reads stale values.
    for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
      if (handoff->per_lod_n_active[lv] == 0 || data_base[lv] == 0)
        continue;
      const struct lod_segment* seg = &handoff->layout.lods[lv];
      size_t* off = slot->h_offsets + seg->batch_covering_offset + lv;
      const uint64_t n = (uint64_t)seg->n_active * seg->covering_count + 1;
      for (uint64_t i = 0; i < n; ++i)
        off[i] -= data_base[lv];
    }

    for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
      if (handoff->per_lod_n_active[lv] == 0)
        continue;

      struct aggregate_result ar = lod_view(handoff, slot, lv, data_base[lv]);
      // Per-LOD slice of the unified host tail-bytes scratch.
      size_t* h_tail_lv = NULL;
      if (shards && shards->h_tail_bytes && page_size > 0)
        h_tail_lv = shards->h_tail_bytes + shards->shards_begin[lv];

      size_t level_bytes = 0;
      if (deliver_to_shards_batch((uint8_t)lv,
                                  handoff->shards_by_lod[lv],
                                  &ar,
                                  &handoff->per_lod_agg_layouts[lv],
                                  h_tail_lv,
                                  handoff->per_lod_n_active[lv],
                                  sink,
                                  stage->shard_alignment,
                                  &level_bytes))
        goto Error;
      sink_bytes += level_bytes;
    }

    // SYNC_MEMOPS on the destinations guarantees these copies have
    // completed at the device before they return. Returning from this function
    // is therefore the coordinator's authoritative tail-ready transition.
    if (shards && shards->total_shards > 0 && page_size > 0) {
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
      slot->io_done = sink->record_fence(sink);

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
    if (ss && shard_state_publish_append(ss, sink, dims_info, lv))
      return 1;
  }
  return 0;
}

int
d2h_deliver_kick(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 struct aggregate_slot* slot,
                 CUstream d2h_stream)
{
  const int fc = handoff->fc;
  const struct batch_aggregate_layout* layout = &handoff->layout;

  CU(Error, cuEventRecord(stage->t_d2h_start[fc], d2h_stream));

  // Compressed codecs use these in drain to size exact per-LOD transfers;
  // pass-through copies them too so delivery's chunk-index walk is uniform.
  int dispatch_err = 0;
  if (layout->total_batch_covering > 0) {
    const size_t n = layout->total_batch_covering + (size_t)handoff->nlod;
    D2H_TRY(dispatch_err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(slot->h_offsets,
                              (CUdeviceptr)slot->d_offsets,
                              n * sizeof(size_t),
                              d2h_stream));
    if (!dispatch_err)
      D2H_TRY(dispatch_err,
              "cuMemcpyDtoHAsync",
              cuMemcpyDtoHAsync(slot->h_permuted_sizes,
                                (CUdeviceptr)slot->d_permuted_sizes,
                                n * sizeof(size_t),
                                d2h_stream));
  }

  // Compressed defers bulk D2H to drain time, once the chunk index lands.
  if (!dispatch_err && handoff->passthrough && layout->total_data_bytes > 0)
    D2H_TRY(dispatch_err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(slot->h_aggregated,
                              (CUdeviceptr)slot->d_aggregated,
                              layout->total_data_bytes,
                              d2h_stream));

  return dispatch_err;

Error:
  return 1;
}

#include "gpu/flush.d2h_deliver.h"
#include "gpu/flush.helpers.h"

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
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->ord = ord;
  stage->shard_alignment = shard_alignment;

  // Seed timing events so the first metric reads see a valid interval.
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventCreate(&stage->t_d2h_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->t_d2h_start[fc], compute));
  }

  // Drain-time copies must not share the d2h stream (see d2h_deliver_stage).
  CU(Fail, cuStreamCreate(&stage->drain_stream, CU_STREAM_NON_BLOCKING));
  gpu_ordering_register_stream(ord, GPU_STREAM_DRAIN, stage->drain_stream);

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
  for (int fc = 0; fc < 2; ++fc)
    cu_event_destroy(stage->t_d2h_start[fc]);
  cu_stream_destroy(stage->drain_stream);
  stage->drain_stream = NULL;
}

// --- Internal helpers ---

// Wait for any pending IO fence on the unified slot (across all LODs that
// share it). Accumulates wall time into stage->metrics->io_fence_stall.
static void
wait_io_fences(struct aggregate_slot* slot,
               struct shard_sink* sink,
               struct stream_metrics* metrics)
{
  if (!sink->wait_fence)
    return;
  struct platform_clock clk = { 0 };
  platform_toc(&clk);
  if (slot->io_done.seq > 0)
    sink->wait_fence(sink, slot->io_done);
  if (metrics) {
    float ms = (float)(platform_toc(&clk) * 1000.0);
    accumulate_metric_ms(&metrics->io_fence_stall, ms, 0, 0);
  }
}

static void
record_flush_metrics(const struct flush_handoff* handoff,
                     const struct aggregate_slot* slot,
                     const struct level_geometry* levels,
                     const struct dim_info* dims,
                     const struct tile_stream_layout* layout,
                     const struct tile_stream_configuration* config,
                     const struct lod_state* lod,
                     const struct lod_shared_state* lod_shared,
                     struct stream_metrics* metrics,
                     CUevent t_d2h_start,
                     CUevent t_d2h_ready)
{
  const int fc = handoff->fc;
  const uint32_t n_epochs = handoff->n_epochs;

  const struct lod_timing* t = &lod_shared->timing[fc];
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
      size_t accum_bytes = lod->append_accum.element_capacity * accum_bpe;
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

    // Aggregated bytes: sum of actual compressed chunk sizes across all LODs
    // in this batch. h_permuted_sizes carries pre-bias per-chunk sizes (with
    // a 0 sentinel slot inserted per LOD); summing those gives the real D2H
    // payload regardless of the absolute/segment-relative offset semantics.
    size_t agg_bytes = 0;
    const size_t n_perm = handoff->layout.total_batch_covering + handoff->nlod;
    for (size_t i = 0; i < n_perm; ++i)
      agg_bytes += slot->h_permuted_sizes[i];

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
    accumulate_metric_cu(
      &metrics->d2h, t_d2h_start, t_d2h_ready, agg_bytes, agg_bytes);
  }
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

// Releases the drained kick's tail generation; must run exactly once per
// drain, on failure exits too — the gate threshold counts kicks, so a
// skipped release leaves the gate unsatisfiable and destroy's auto-flush
// hangs polling. Tail-state content is moot once the drain has failed.
static struct writer_result
finish_drain(struct gpu_pool* tail, int err)
{
  gpu_pool_release_produce_gen(tail);
  return err ? writer_error() : writer_ok();
}

static struct writer_result
sync_and_deliver(struct d2h_deliver_stage* stage,
                 const struct flush_handoff* handoff,
                 const struct level_geometry* levels,
                 const struct dim_info* dims,
                 const struct tile_stream_layout* layout,
                 const struct tile_stream_configuration* config,
                 struct shard_sink* sink,
                 const struct lod_state* lod,
                 const struct lod_shared_state* lod_shared,
                 struct stream_metrics* metrics)
{
  const int fc = handoff->fc;
  struct aggregate_slot* slot = NULL;
  const struct batch_aggregate_layout* alayout = &handoff->layout;

  if (sink->has_error && sink->has_error(sink))
    goto Error;

  {
    struct platform_clock kick_clk = { 0 };
    platform_toc(&kick_clk);

    if (handoff->passthrough) {
      struct gpu_pool_view hv;
      if (gpu_pool_host_acquire_consume(handoff->agg_host, fc, &hv))
        goto Error;
      slot = hv.p;
    } else {
      struct gpu_pool_view iv;
      if (gpu_pool_host_acquire_consume(handoff->agg_index, fc, &iv))
        goto Error;
      slot = iv.p;

      // Bulk copies go on drain_stream, never d2h_stream — sharing
      // deadlocks against the tail gate (see d2h_deliver_stage).
      int dispatch_err = 0;
      if (alayout->page_size > 0) {
        for (uint8_t lv = 0; lv < handoff->nlod && !dispatch_err; ++lv) {
          if (handoff->per_lod_n_active[lv] == 0)
            continue;
          const struct lod_segment* seg = &alayout->lods[lv];
          size_t actual = 0;
          if (lod_actual_bytes(handoff, slot, lv, &actual))
            goto Error;
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
        const size_t total =
          slot->h_offsets[n - 1] + slot->h_permuted_sizes[n - 1];
        if (total > 0)
          D2H_TRY(dispatch_err,
                  "cuMemcpyDtoHAsync",
                  cuMemcpyDtoHAsync(slot->h_aggregated,
                                    (CUdeviceptr)slot->d_aggregated,
                                    total,
                                    stage->drain_stream));
      }

      // Always release the slot (SLOT_DRAINED), even if the D2H dispatch
      // above failed: the host poll below and the next kick's acquire block
      // on it and would hang otherwise. Release-on-error is harmless
      // because the stream is already in an error state and the next op
      // will short-circuit.
      CHECK(Error,
            gpu_pool_release_consume(
              handoff->agg_pool, fc, stage->drain_stream) == 0);

      if (dispatch_err)
        goto Error;
      if (gpu_pool_host_acquire_consume(handoff->agg_host, fc, NULL))
        goto Error;
    }

    float kick_ms = platform_toc(&kick_clk) * 1000.0f;
    accumulate_metric_ms(&metrics->kick_sync_stall, kick_ms, 0, 0);
  }

  record_flush_metrics(handoff,
                       slot,
                       levels,
                       dims,
                       layout,
                       config,
                       lod,
                       lod_shared,
                       metrics,
                       stage->t_d2h_start[fc],
                       gpu_ordering_event(stage->ord, GPU_EDGE_SLOT_DRAINED,
                                          fc));

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;
    // The consumed direction is the deliver-oldest-first host rule, so no
    // device wait is queued; the acquire hands out the array whose tail
    // buffers this delivery uploads.
    struct gpu_pool_view tv = { 0 };
    if (gpu_pool_host_acquire_produce(handoff->tail, 0, &tv))
      goto Error;
    struct compress_agg_array* shards = tv.p;
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
    // completed at the device before they return, so the tail-gate publish
    // in finish_drain cannot outrun them.
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

    // Record an aggregate IO fence on the unified slot. wait_io_fences()
    // checks slot->io_done at the next kick.
    if (sink->record_fence)
      slot->io_done = sink->record_fence(sink);

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&metrics->sink, sink_ms, sink_bytes, sink_bytes);
  }

  return finish_drain(handoff->tail, 0);

Error:
  return finish_drain(handoff->tail, 1);
}

// Periodic metadata update (append-dim extents per level).
static int
maybe_update_metadata(const struct flush_handoff* handoff,
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
  for (uint8_t lv = 0; lv < handoff->nlod; ++lv) {
    struct shard_state* ss = handoff->shards_by_lod[lv];
    if (!ss)
      continue;
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
                 struct shard_sink* sink,
                 CUstream d2h_stream)
{
  const int fc = handoff->fc;
  const struct batch_aggregate_layout* layout = &handoff->layout;

  // io_done is host-owned slot bookkeeping; its fence must retire before
  // any device acquire, so peek rather than acquire here.
  wait_io_fences(gpu_pool_at(handoff->agg_host, fc, 0).p, sink, stage->metrics);

  struct gpu_pool_view v;
  CHECK(Error,
        gpu_pool_acquire_consume(handoff->agg_pool, fc, d2h_stream, &v) == 0);
  struct aggregate_slot* slot = v.p;
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
  // Passthrough never polls the chunk index (its drain waits on the
  // slot-drained edge recorded after the bulk copy below).
  if (!dispatch_err && !handoff->passthrough &&
      gpu_pool_release_produce(handoff->agg_index, fc, d2h_stream))
    dispatch_err = 1;

  // Compressed defers bulk D2H to sync_and_deliver once the chunk index lands.
  if (!dispatch_err && handoff->passthrough && layout->total_data_bytes > 0)
    D2H_TRY(dispatch_err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(slot->h_aggregated,
                              (CUdeviceptr)slot->d_aggregated,
                              layout->total_data_bytes,
                              d2h_stream));

  // Always release the passthrough slot (SLOT_DRAINED) even on dispatch
  // error: the drain's host poll blocks on it and would hang otherwise.
  if (handoff->passthrough) {
    CHECK(Error,
          gpu_pool_release_consume(handoff->agg_pool, fc, d2h_stream) == 0);
  }

  return dispatch_err;

Error:
  return 1;
}

struct writer_result
d2h_deliver_drain(struct d2h_deliver_stage* stage,
                  const struct flush_handoff* handoff,
                  const struct level_geometry* levels,
                  const struct dim_info* dims,
                  const struct tile_stream_layout* layout,
                  const struct tile_stream_configuration* config,
                  struct shard_sink* sink,
                  const struct lod_state* lod,
                  const struct lod_shared_state* lod_shared,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock,
                  CUstream d2h_stream)
{
  (void)d2h_stream; // drain-time copies use stage->drain_stream
  struct writer_result r = sync_and_deliver(stage,
                                            handoff,
                                            levels,
                                            dims,
                                            layout,
                                            config,
                                            sink,
                                            lod,
                                            lod_shared,
                                            metrics);
  if (!r.error) {
    if (maybe_update_metadata(
          handoff, dims, config, sink, metadata_update_clock))
      return writer_error();
  }
  return r;
}

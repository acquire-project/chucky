#include "gpu/flush.d2h_deliver.h"
#include "gpu/flush.helpers.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <assert.h>
#include <string.h>

// --- Init / Destroy ---

int
d2h_deliver_init(struct d2h_deliver_stage* stage,
                 size_t shard_alignment,
                 CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->shard_alignment = shard_alignment;

  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventCreate(&stage->t_d2h_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->h_chunk_index_ready[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->ready[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->t_d2h_start[fc], compute));
    CU(Fail, cuEventRecord(stage->h_chunk_index_ready[fc], compute));
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
    cu_event_destroy(stage->h_chunk_index_ready[fc]);
    cu_event_destroy(stage->ready[fc]);
  }
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
    // TODO(phase3): with batches_per_slot > 1, sum across all slot batches
    // (slot->slot_batches[0..batches_per_slot-1]) instead of just the
    // current handoff->layout. handoff carries the latest batch only.
    size_t agg_bytes = 0;
    const size_t n_perm = handoff->layout.total_batch_covering + handoff->nlod;
    for (size_t i = 0; i < n_perm; ++i)
      agg_bytes += handoff->agg->h_permuted_sizes[i];

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

// Slot-relative byte offset for batch be's LOD lv segment (carry-over mode).
// In contiguous mode the bias kernel doesn't run, so the per-LOD start must
// be read from the post-scan h_offsets — see slot_lod_data_base.
static size_t
slot_data_offset(const struct batch_slice_entry* be, uint8_t lv)
{
  return be->data_base_offset + be->per_lod_lods[lv].data_segment_offset;
}

static uint64_t
slot_desc_offset(const struct batch_slice_entry* be, uint8_t lv)
{
  return be->desc_base_offset + be->per_lod_lods[lv].batch_covering_offset +
         (uint64_t)lv;
}

// Mode-aware: carry-over uses the page-aligned per-LOD offset; contiguous
// reads h_offsets where the cumulative scan already accounts for prior LODs.
static size_t
slot_lod_data_base(const struct batch_aggregate_layout* layout,
                   const struct batch_slice_entry* be,
                   uint8_t lv,
                   const size_t* h_offsets)
{
  if (layout->page_size > 0)
    return slot_data_offset(be, lv);
  return h_offsets[slot_desc_offset(be, lv)];
}

static struct aggregate_result
batch_lod_view(struct aggregate_slot* slot,
               const struct batch_aggregate_layout* layout,
               const struct batch_slice_entry* be,
               uint8_t lv)
{
  const size_t off = slot_lod_data_base(layout, be, lv, slot->h_offsets);
  const uint64_t desc = slot_desc_offset(be, lv);
  struct aggregate_result ar = {
    .data = (uint8_t*)slot->h_aggregated + off,
    .offsets = slot->h_offsets + desc,
    .chunk_sizes = slot->h_permuted_sizes + desc,
  };
  return ar;
}

// Poll a CUDA event with a sleep loop. Returns 0 on success / deinit,
// 1 on error.
static int
poll_event(CUevent ev, int fc)
{
  for (;;) {
    CUresult r = cuEventQuery(ev);
    if (r == CUDA_SUCCESS)
      return 0;
    if (r == CUDA_ERROR_DEINITIALIZED) {
      log_debug("d2h_deliver: cuEventQuery returned DEINITIALIZED (fc=%d)", fc);
      return 0;
    }
    if (r != CUDA_ERROR_NOT_READY) {
      handle_curesult(LOG_ERROR, r, __FILE__, __LINE__, "cuEventQuery");
      return 1;
    }
    platform_sleep_ns(50000);
  }
}

// Per-batch / per-LOD actual bytes in d_aggregated. End of last chunk
// (last shard's last chunk in shard-permuted order) minus the LOD segment
// base within the slot. h_offsets here must be the pre-rebase (absolute)
// values produced by the unified scan + bias.
static size_t
batch_lod_actual_bytes(const struct aggregate_slot* slot,
                       const struct batch_slice_entry* be,
                       uint8_t lv)
{
  const struct lod_segment* seg = &be->per_lod_lods[lv];
  if (seg->n_active == 0)
    return 0;
  const uint64_t total = (uint64_t)seg->n_active * seg->covering_count;
  const uint64_t last = slot_desc_offset(be, lv) + total - 1;
  const size_t base = slot_data_offset(be, lv);
  const size_t end = slot->h_offsets[last] + slot->h_permuted_sizes[last];
  assert(slot->h_offsets[last] >= base);
  const size_t actual = end - base;
  assert(actual <= seg->data_segment_bytes);
  return actual;
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
                 struct stream_metrics* metrics,
                 CUstream d2h_stream)
{
  const int fc = handoff->fc;
  struct aggregate_slot* slot = handoff->agg;
  const struct batch_aggregate_layout* alayout = &handoff->layout;

  if (sink->has_error && sink->has_error(sink))
    goto Error;

  {
    struct platform_clock kick_clk = { 0 };
    platform_toc(&kick_clk);

    if (handoff->passthrough) {
      // Bulk D2H was queued in d2h_deliver_kick; just wait for it.
      if (poll_event(stage->ready[fc], fc))
        goto Error;
    } else {
      // Compressed: wait for the chunk index on host, then dispatch the
      // exact-size bulk D2H per batch×LOD (carry-over) or once across the
      // slot's contiguous data (contiguous mode).
      if (poll_event(stage->h_chunk_index_ready[fc], fc))
        goto Error;

      // Rewrite per-batch / per-LOD data_segment_offset to point at the
      // dense LOD start (= dense base of first shard in lv). Mutates
      // h_shard_base_offsets_dense in place to hold LOD-relative shard
      // offsets, which downstream delivery passes to shard_delivery.
      // Carry-over only; contiguous mode keeps uniform semantics.
      struct shard_tables* dense_shards = handoff->shards;
      if (alayout->page_size > 0 && dense_shards &&
          dense_shards->total_shards > 0 && slot->h_shard_base_offsets_dense) {
        volatile size_t* dense = slot->h_shard_base_offsets_dense;
        for (uint32_t b = 0; b < slot->batches_per_slot; ++b) {
          struct batch_slice_entry* be = &slot->slot_batches[b];
          for (uint8_t lv = 0; lv < be->nlod; ++lv) {
            const uint32_t lv_begin = dense_shards->shards_begin[lv];
            const uint32_t lv_n = dense_shards->n_shards[lv];
            if (lv_n == 0)
              continue;
            const size_t dense_lv_start = (size_t)dense[lv_begin];
            be->per_lod_lods[lv].data_segment_offset =
              dense_lv_start - be->data_base_offset;
            for (uint32_t si = 0; si < lv_n; ++si)
              dense[lv_begin + si] = (size_t)dense[lv_begin + si] -
                                     dense_lv_start; // now LOD-relative
          }
        }
      }

      if (alayout->page_size > 0) {
        for (uint32_t b = 0; b < slot->batches_per_slot; ++b) {
          const struct batch_slice_entry* be = &slot->slot_batches[b];
          for (uint8_t lv = 0; lv < be->nlod; ++lv) {
            if (be->per_lod_lods[lv].n_active == 0)
              continue;
            const size_t off = slot_data_offset(be, lv);
            const size_t actual = batch_lod_actual_bytes(slot, be, lv);
            if (actual == 0)
              continue;
            CU(Error,
               cuMemcpyDtoHAsync((uint8_t*)slot->h_aggregated + off,
                                 (CUdeviceptr)slot->d_aggregated + off,
                                 actual,
                                 d2h_stream));
          }
        }
      } else if (slot->batches_per_slot > 0) {
        // Cumulative-end index = last sentinel of the slot's last batch.
        // Without bias (page_size==0), h_offsets there is the running sum
        // of all chunk sizes across this batch's slice; with Phase 3
        // macro-agg the desc_base_offset advances per batch.
        const struct batch_slice_entry* last_be =
          &slot->slot_batches[slot->batches_per_slot - 1];
        uint64_t C_batch = 0;
        for (uint8_t lv = 0; lv < last_be->nlod; ++lv)
          C_batch += (uint64_t)last_be->per_lod_lods[lv].n_active *
                     last_be->per_lod_lods[lv].covering_count;
        const uint64_t last =
          last_be->desc_base_offset + C_batch + (uint64_t)last_be->nlod - 1;
        const size_t total =
          slot->h_offsets[last] + slot->h_permuted_sizes[last];
        if (total > 0) {
          CU(Error,
             cuMemcpyDtoHAsync(slot->h_aggregated,
                               (CUdeviceptr)slot->d_aggregated,
                               total,
                               d2h_stream));
        }
      }

      CU(Error, cuEventRecord(stage->ready[fc], d2h_stream));
      CU(Error, cuEventRecord(slot->ready, d2h_stream));

      if (poll_event(stage->ready[fc], fc))
        goto Error;
    }

    float kick_ms = platform_toc(&kick_clk) * 1000.0f;
    accumulate_metric_ms(&metrics->kick_sync_stall, kick_ms, 0, 0);
  }

  record_flush_metrics(handoff,
                       levels,
                       dims,
                       layout,
                       config,
                       lod,
                       lod_shared,
                       metrics,
                       stage->t_d2h_start[fc],
                       stage->ready[fc]);

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;
    struct shard_tables* shards = handoff->shards;
    const size_t page_size = shards ? shards->page_size : 0;

    // Rebase per-batch / per-LOD offsets to be segment-relative for the
    // shard_delivery contract (which pairs segment-shifted result.data with
    // segment-relative offsets). Mutates h_offsets in place. In contiguous
    // mode the base for LOD lv is read from h_offsets[slot_desc_offset(be,
    // lv)] — that value lives at the START of LOD lv's range, which the
    // current iteration is about to overwrite, but each subsequent (batch,
    // LOD) reads its OWN starting index which is outside the prior range.
    for (uint32_t b = 0; b < slot->batches_per_slot; ++b) {
      const struct batch_slice_entry* be = &slot->slot_batches[b];
      for (uint8_t lv = 0; lv < be->nlod; ++lv) {
        const struct lod_segment* seg = &be->per_lod_lods[lv];
        if (seg->n_active == 0)
          continue;
        const size_t base =
          slot_lod_data_base(&handoff->layout, be, lv, slot->h_offsets);
        if (base == 0)
          continue;
        size_t* off = slot->h_offsets + slot_desc_offset(be, lv);
        const uint64_t n = (uint64_t)seg->n_active * seg->covering_count + 1;
        for (uint64_t i = 0; i < n; ++i)
          off[i] -= base;
      }
    }

    for (uint32_t b = 0; b < slot->batches_per_slot; ++b) {
      const struct batch_slice_entry* be = &slot->slot_batches[b];
      for (uint8_t lv = 0; lv < be->nlod; ++lv) {
        if (be->per_lod_lods[lv].n_active == 0)
          continue;

        struct aggregate_result ar =
          batch_lod_view(slot, &handoff->layout, be, lv);
        size_t* h_tail_lv = NULL;
        if (shards && shards->h_tail_bytes && page_size > 0)
          h_tail_lv = shards->h_tail_bytes + shards->shards_begin[lv];

        // Per-LOD dense rel offsets (carry-over only; NULL = uniform).
        const size_t* shard_base_offsets_lv = NULL;
        if (page_size > 0 && shards && shards->total_shards > 0 &&
            slot->h_shard_base_offsets_dense) {
          shard_base_offsets_lv =
            (const size_t*)slot->h_shard_base_offsets_dense +
            shards->shards_begin[lv];
        }
        size_t level_bytes = 0;
        if (deliver_to_shards_batch((uint8_t)lv,
                                    handoff->shards_by_lod[lv],
                                    &ar,
                                    &handoff->per_lod_agg_layouts[lv],
                                    h_tail_lv,
                                    be->per_lod_lods[lv].n_active,
                                    sink,
                                    stage->shard_alignment,
                                    shard_base_offsets_lv,
                                    &level_bytes))
          goto Error;
        sink_bytes += level_bytes;
      }
    }

    // Push the post-delivery tail state back to GPU in two bulk HtoDs that
    // cover all shards across all LODs at once. Replaces the per-LOD
    // tail-state uploads from the legacy pipeline.
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
      handoff->agg->io_done = sink->record_fence(sink);

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&metrics->sink, sink_ms, sink_bytes, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

// Periodic metadata update (append-dim extents per level).
static int
maybe_update_metadata(const struct flush_handoff* handoff,
                      const struct dim_info* dims_info,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct platform_clock* metadata_update_clock)
{
  (void)config;
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
                 const struct level_geometry* levels,
                 const struct batch_state* batch,
                 const struct dim_info* dims,
                 struct shard_sink* sink,
                 CUstream d2h_stream)
{
  (void)levels;
  (void)batch;
  (void)dims;
  const int fc = handoff->fc;
  struct aggregate_slot* slot = handoff->agg;
  const struct batch_aggregate_layout* layout = &handoff->layout;

  wait_io_fences(slot, sink, stage->metrics);

  CU(Error, cuStreamWaitEvent(d2h_stream, handoff->t_aggregate_end, 0));
  CU(Error, cuEventRecord(stage->t_d2h_start[fc], d2h_stream));

  // Offsets + permuted sizes. Compressed codecs use these in drain to size
  // exact per-LOD transfers; pass-through copies them too so delivery's
  // chunk-index walk works uniformly.
  if (layout->total_batch_covering > 0) {
    const size_t n = layout->total_batch_covering + (size_t)handoff->nlod;
    CU(Error,
       cuMemcpyDtoHAsync(slot->h_offsets,
                         (CUdeviceptr)slot->d_offsets,
                         n * sizeof(size_t),
                         d2h_stream));
    CU(Error,
       cuMemcpyDtoHAsync(slot->h_permuted_sizes,
                         (CUdeviceptr)slot->d_permuted_sizes,
                         n * sizeof(size_t),
                         d2h_stream));
  }

  CU(Error, cuEventRecord(stage->h_chunk_index_ready[fc], d2h_stream));

  // Pass-through: per-LOD actual equals worst-case. Dispatch the bulk D2H
  // now so it pipelines with the next batch's compute (matches pre-Phase-1
  // behavior). Compressed path defers bulk dispatch to sync_and_deliver
  // once the chunk index has landed.
  if (handoff->passthrough) {
    if (layout->total_data_bytes > 0) {
      CU(Error,
         cuMemcpyDtoHAsync(slot->h_aggregated,
                           (CUdeviceptr)slot->d_aggregated,
                           layout->total_data_bytes,
                           d2h_stream));
    }
    // Record events unconditionally so the poll in sync_and_deliver sees a
    // current-cycle signal even when this batch carries no data (all LODs
    // inactive). Without this the poll would observe a prior-cycle signal,
    // which happens to be harmless today but invalidates the "every kick
    // signals slot->ready" invariant the cap-stacking paths depend on.
    CU(Error, cuEventRecord(stage->ready[fc], d2h_stream));
    CU(Error, cuEventRecord(slot->ready, d2h_stream));
  }

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
                  const struct lod_shared_state* lod_shared,
                  struct stream_metrics* metrics,
                  struct platform_clock* metadata_update_clock,
                  CUstream d2h_stream)
{
  (void)batch;
  struct writer_result r = sync_and_deliver(stage,
                                            handoff,
                                            levels,
                                            dims,
                                            layout,
                                            config,
                                            sink,
                                            lod,
                                            lod_shared,
                                            metrics,
                                            d2h_stream);
  if (!r.error) {
    if (maybe_update_metadata(
          handoff, dims, config, sink, metadata_update_clock))
      return writer_error();
  }
  return r;
}

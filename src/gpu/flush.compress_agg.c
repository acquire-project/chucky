#include "gpu/flush.compress_agg.h"
#include "gpu/flush.helpers.h"

#include "defs.limits.h"
#include "gpu/aggregate.h"
#include "gpu/compress.h"
#include "gpu/prelude.cuda.h"
#include "stream/layouts.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

// --- Internal helpers ---

// Record compress-start, compress, record compress-end.
static int
kick_compress(struct compress_agg_stage* stage,
              int fc,
              const void* d_input,
              uint64_t n_chunks,
              size_t chunk_bytes,
              CUstream compress_stream)
{
  CU(Error, cuEventRecord(stage->t_compress_start[fc], compress_stream));
  CHECK(Error,
        codec_compress(&stage->codec,
                       d_input,
                       chunk_bytes,
                       (void*)stage->d_compressed[fc],
                       n_chunks,
                       compress_stream) == 0);
  CU(Error, cuEventRecord(stage->t_compress_end[fc], compress_stream));
  return 0;

Error:
  return 1;
}

static uint32_t
derive_batches_per_slot_cap(int can_stack,
                            size_t slot_data_bytes,
                            uint64_t desc_entries_per_batch)
{
  if (!can_stack)
    return 1;
  const uint64_t desc_pair_bytes =
    desc_entries_per_batch * (uint64_t)(2 * sizeof(size_t));
  if (slot_data_bytes == 0 || desc_pair_bytes == 0)
    return 1;
  uint64_t cap = (uint64_t)slot_data_bytes / desc_pair_bytes;
  if (cap < 1)
    cap = 1;
  if (cap > UINT32_MAX)
    cap = UINT32_MAX;
  return (uint32_t)cap;
}

// --- Init / Destroy ---

int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  CUstream compute)
{
  memset(stage, 0, sizeof(*stage));

  const size_t bytes_per_element = dtype_bpe(config->dtype);
  const uint32_t K = cl->epochs_per_batch;
  const uint64_t total_chunks = cl->levels.total_chunks;
  const uint64_t chunk_stride = cl->layouts[0].chunk_stride;
  CHECK_MUL_OVERFLOW(Fail, K, total_chunks, UINT64_MAX);
  const uint64_t M = (uint64_t)K * total_chunks;
  const size_t chunk_bytes = chunk_stride * bytes_per_element;

  // Codec
  CHECK(Fail, codec_init(&stage->codec, config->codec.id, chunk_bytes, M) == 0);

  // Per-LOD scratch for mask-scan results, plus the previous-kick cache used
  // for steady-state LUT-cache validation. Both are sized to
  // LOD_MAX_LEVELS * K with stride K so each LOD's slice lives at a stable
  // offset and the cache comparison can match by stride.
  stage->pool_epochs_stride = K;
  stage->pool_epochs_scratch =
    (uint32_t*)malloc((size_t)LOD_MAX_LEVELS * K * sizeof(uint32_t));
  stage->cached_pool_epochs =
    (uint32_t*)malloc((size_t)LOD_MAX_LEVELS * K * sizeof(uint32_t));
  CHECK(Fail, stage->pool_epochs_scratch && stage->cached_pool_epochs);

  CHECK_MUL_OVERFLOW(Fail, M, stage->codec.max_output_size, SIZE_MAX);
  // Compressed buffers + events. CODEC_NONE aggregates directly from pool_buf,
  // so the d_compressed buffer is unused — skip its M * chunk_bytes allocation
  // per fc. Destroy is NULL-safe.
  const int need_compressed = (stage->codec.type != CODEC_NONE);
  for (int fc = 0; fc < 2; ++fc) {
    if (need_compressed)
      CU(
        Fail,
        cuMemAlloc(&stage->d_compressed[fc], M * stage->codec.max_output_size));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->d_measurement[fc],
                  sizeof(struct aggregate_append_measurement)));
    CU(Fail,
       cuMemHostAlloc((void**)&stage->h_measurement[fc],
                      sizeof(struct aggregate_append_measurement),
                      0));
    memset((void*)stage->h_measurement[fc],
           0,
           sizeof(struct aggregate_append_measurement));
    CU(Fail, cuEventCreate(&stage->measurement_ready[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->t_compress_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->t_compress_end[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->t_aggregate_end[fc], CU_EVENT_DEFAULT));
  }

  // --- Unified across-LODs aggregate state ---------------------------------
  // Mirrors the CPU pipeline (src/cpu/pipeline.c + src/cpu/aggregate.c).

  stage->nlod = (uint8_t)cl->levels.nlod;

  // Per-LOD aggregate_layouts: own copy so multiarray bind/unbind can swap
  // them per-array. Each layout's GPU-side d_lifted_shape/strides are uploaded.
  for (int lv = 0; lv < cl->levels.nlod; ++lv) {
    stage->per_lod_agg_layouts[lv] = cl->per_level[lv].agg_layout;
    CHECK(Fail, aggregate_layout_upload(&stage->per_lod_agg_layouts[lv]) == 0);
  }

  // Cached max batch layout assuming each LOD fires its worst-case active
  // count.
  {
    uint32_t per_lod_max[LOD_MAX_LEVELS] = { 0 };
    for (int lv = 0; lv < cl->levels.nlod; ++lv)
      per_lod_max[lv] = cl->per_level[lv].batch_active_count;
    const size_t page_size = cl->per_level[0].agg_layout.page_size;
    CHECK(Fail,
          batch_aggregate_layout_init(&stage->max_batch_layout,
                                      stage->per_lod_agg_layouts,
                                      per_lod_max,
                                      stage->nlod,
                                      page_size) == 0);
    stage->max_total_batch_chunks = stage->max_batch_layout.total_batch_chunks;
    stage->max_total_batch_covering =
      stage->max_batch_layout.total_batch_covering;
    stage->max_total_data_bytes = stage->max_batch_layout.total_data_bytes;
  }

  // Total shards across LODs — needed by slot init (for per-shard sums
  // buffer) before the per-shard tables block populates stage->shards.
  uint64_t total_shards_init = 0;
  for (int lv = 0; lv < cl->levels.nlod; ++lv)
    total_shards_init += cl->per_level[lv].agg_layout.num_shards;

  if (stage->max_total_batch_chunks > 0) {
    // Passthrough with paged shards (page_size > 0) sizes each batch to its
    // worst-case envelope, so only one fits in the per-batch slot capacity.
    // Compressed codecs and contiguous-mode passthrough can actually stack.
    const size_t page_size_init = cl->per_level[0].agg_layout.page_size;
    const int can_stack =
      (stage->codec.type != CODEC_NONE) || (page_size_init == 0);
    const uint64_t one_batch =
      stage->max_total_batch_covering + (uint64_t)stage->nlod;
    const uint32_t batches_per_slot_cap = derive_batches_per_slot_cap(
      can_stack, stage->max_total_data_bytes, one_batch);
    CHECK_MUL_OVERFLOW(
      Fail, (uint64_t)batches_per_slot_cap, one_batch, UINT64_MAX);
    const uint64_t slot_chunk_cap = (uint64_t)batches_per_slot_cap * one_batch;
    for (int fc = 0; fc < 2; ++fc) {
      CHECK(Fail,
            aggregate_batch_slot_init(&stage->output[fc],
                                      slot_chunk_cap,
                                      stage->max_total_data_bytes,
                                      batches_per_slot_cap,
                                      total_shards_init) == 0);
      CU(Fail, cuEventRecord(stage->output[fc].ready, compute));
      CU(Fail, cuEventRecord(stage->output[fc].host_func_done, compute));
    }
  }

  CU(Fail,
     cuMemAlloc((CUdeviceptr*)&stage->d_write_desc,
                sizeof(struct aggregate_write_desc)));
  CU(Fail,
     cuMemsetD8((CUdeviceptr)stage->d_write_desc,
                0,
                sizeof(struct aggregate_write_desc)));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&stage->d_tail_sum_bytes, sizeof(size_t)));
  CU(Fail,
     cuMemHostAlloc(
       (void**)&stage->h_write_desc, sizeof(struct aggregate_write_desc), 0));
  memset((void*)stage->h_write_desc, 0, sizeof(struct aggregate_write_desc));
  for (int i = 0; i < 4; ++i) {
    stage->cb_args_ring[i].slots[0] = &stage->output[0];
    stage->cb_args_ring[i].slots[1] = &stage->output[1];
    stage->cb_args_ring[i].h_write_desc = stage->h_write_desc;
  }

  if (stage->max_total_batch_chunks > 0) {
    const uint64_t temp_count =
      stage->max_total_batch_covering + (uint64_t)LOD_MAX_LEVELS;
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->d_temp_offsets,
                  temp_count * sizeof(size_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->d_temp_perm_sizes,
                  temp_count * sizeof(size_t)));
  }

  // Unified LUT buffers + host scratch.
  if (stage->max_total_batch_chunks > 0) {
    CU(Fail,
       cuMemAlloc(&stage->d_batch_gather,
                  stage->max_total_batch_chunks * sizeof(uint32_t)));
    CU(Fail,
       cuMemAlloc(&stage->d_batch_perm,
                  stage->max_total_batch_chunks * sizeof(uint32_t)));
    stage->h_lut_gather_scratch =
      (uint32_t*)malloc(stage->max_total_batch_chunks * sizeof(uint32_t));
    stage->h_lut_perm_scratch =
      (uint32_t*)malloc(stage->max_total_batch_chunks * sizeof(uint32_t));
    CHECK(Fail, stage->h_lut_gather_scratch && stage->h_lut_perm_scratch);
  }

  // Per-shard tables. total_shards = sum_lv num_shards[lv].
  {
    uint64_t total_shards = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      stage->shards.shards_begin[lv] = (uint32_t)total_shards;
      stage->shards.n_shards[lv] =
        (uint32_t)cl->per_level[lv].agg_layout.num_shards;
      total_shards += cl->per_level[lv].agg_layout.num_shards;
    }
    stage->shards.total_shards = total_shards;
    stage->shards.page_size = cl->per_level[0].agg_layout.page_size;

    if (total_shards > 0) {
      stage->shards.h_base_offsets =
        (size_t*)malloc(total_shards * sizeof(size_t));
      stage->shards.h_shard_capacity =
        (size_t*)malloc(total_shards * sizeof(size_t));
      stage->shards.h_tps_group =
        (uint64_t*)malloc(total_shards * sizeof(uint64_t));
      stage->shards.h_offsets_base =
        (uint64_t*)malloc(total_shards * sizeof(uint64_t));
      stage->shards.h_tail_bytes =
        (size_t*)calloc(total_shards, sizeof(size_t));
      CHECK(Fail,
            stage->shards.h_base_offsets && stage->shards.h_shard_capacity &&
              stage->shards.h_tps_group && stage->shards.h_offsets_base &&
              stage->shards.h_tail_bytes);
      for (uint64_t i = 0; i < total_shards; ++i) {
        for (int lv = 0; lv < cl->levels.nlod; ++lv) {
          if (i >= stage->shards.shards_begin[lv] &&
              i < stage->shards.shards_begin[lv] + stage->shards.n_shards[lv]) {
            stage->shards.h_shard_capacity[i] =
              cl->per_level[lv].agg_layout.shard_capacity;
            break;
          }
        }
      }

      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&stage->shards.d_base_offsets,
                    total_shards * sizeof(size_t)));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&stage->shards.d_shard_capacity,
                    total_shards * sizeof(size_t)));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&stage->shards.d_tps_group,
                    total_shards * sizeof(uint64_t)));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&stage->shards.d_offsets_base,
                    total_shards * sizeof(uint64_t)));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&stage->shards.d_tail_bytes,
                    total_shards * sizeof(size_t)));
      CU(Fail,
         cuMemsetD8((CUdeviceptr)stage->shards.d_tail_bytes,
                    0,
                    total_shards * sizeof(size_t)));

      // d_shard_capacity stays constant across batches in steady state; upload
      // it once now. d_base_offsets / d_tps_group / d_offsets_base depend on
      // per-batch active counts and are uploaded by the kick.
      CU(Fail,
         cuMemcpyHtoD((CUdeviceptr)stage->shards.d_shard_capacity,
                      stage->shards.h_shard_capacity,
                      total_shards * sizeof(size_t)));

      // Tail-carry buffer: total_shards * page_size bytes; uniform layout
      // across LODs (sink page size is uniform).
      if (stage->shards.page_size > 0) {
        stage->shards.tail_carry_bytes = total_shards * stage->shards.page_size;
        CU(Fail,
           cuMemAlloc(&stage->shards.d_tail_carry,
                      stage->shards.tail_carry_bytes));
        CU(Fail,
           cuMemsetD8(
             stage->shards.d_tail_carry, 0, stage->shards.tail_carry_bytes));
      }
    }
  }

  // Per-LOD shard_state (writers + tail/footer pools + generation
  // bookkeeping). The unified kick/D2H path iterates this directly.
  for (int lv = 0; lv < cl->levels.nlod; ++lv)
    CHECK(Fail, init_shard_state(&stage->shard[lv], &cl->per_level[lv]) == 0);

  // Seed events
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventRecord(stage->t_compress_start[fc], compute));
    CU(Fail, cuEventRecord(stage->t_compress_end[fc], compute));
    CU(Fail, cuEventRecord(stage->t_aggregate_end[fc], compute));
  }

  return 0;

Fail:
  compress_agg_destroy(stage, cl->levels.nlod);
  return 1;
}

void
compress_agg_destroy(struct compress_agg_stage* stage, int nlod)
{
  if (!stage)
    return;
  codec_free(&stage->codec);
  free(stage->pool_epochs_scratch);
  free(stage->cached_pool_epochs);
  stage->pool_epochs_scratch = NULL;
  stage->cached_pool_epochs = NULL;
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(stage->d_compressed[fc]);
    cu_mem_free((CUdeviceptr)stage->d_measurement[fc]);
    if (stage->h_measurement[fc])
      cuMemFreeHost((void*)stage->h_measurement[fc]);
    cu_event_destroy(stage->measurement_ready[fc]);
    cu_event_destroy(stage->t_compress_start[fc]);
    cu_event_destroy(stage->t_compress_end[fc]);
    cu_event_destroy(stage->t_aggregate_end[fc]);
  }
  for (int fc = 0; fc < 2; ++fc)
    aggregate_slot_destroy(&stage->output[fc]);
  cu_mem_free((CUdeviceptr)stage->d_write_desc);
  cu_mem_free((CUdeviceptr)stage->d_tail_sum_bytes);
  if (stage->h_write_desc)
    cuMemFreeHost((void*)stage->h_write_desc);
  cu_mem_free((CUdeviceptr)stage->d_temp_offsets);
  cu_mem_free((CUdeviceptr)stage->d_temp_perm_sizes);
  cu_mem_free(stage->d_batch_gather);
  cu_mem_free(stage->d_batch_perm);
  free(stage->h_lut_gather_scratch);
  free(stage->h_lut_perm_scratch);
  stage->h_lut_gather_scratch = NULL;
  stage->h_lut_perm_scratch = NULL;
  for (int lv = 0; lv < nlod; ++lv) {
    aggregate_layout_destroy(&stage->per_lod_agg_layouts[lv]);
    shard_state_destroy(&stage->shard[lv]);
  }
  free(stage->shards.h_base_offsets);
  free(stage->shards.h_shard_capacity);
  free(stage->shards.h_tps_group);
  free(stage->shards.h_offsets_base);
  free(stage->shards.h_tail_bytes);
  cu_mem_free((CUdeviceptr)stage->shards.d_base_offsets);
  cu_mem_free((CUdeviceptr)stage->shards.d_shard_capacity);
  cu_mem_free((CUdeviceptr)stage->shards.d_tps_group);
  cu_mem_free((CUdeviceptr)stage->shards.d_offsets_base);
  cu_mem_free((CUdeviceptr)stage->shards.d_tail_bytes);
  cu_mem_free(stage->shards.d_tail_carry);
  memset(&stage->shards, 0, sizeof(stage->shards));
}

// --- Kick ---

// Build per-shard tables for this batch's `layout`. Populates host shadows
// and uploads to device on `stream`. tables.h_shard_capacity is constant and
// uploaded at init; not re-uploaded here.
static int
build_and_upload_shard_tables(struct compress_agg_stage* stage,
                              const struct batch_aggregate_layout* layout,
                              CUstream stream)
{
  struct shard_tables* t = &stage->shards;
  if (t->total_shards == 0)
    return 0;

  for (uint8_t lv = 0; lv < layout->nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    const struct aggregate_layout* al = &stage->per_lod_agg_layouts[lv];
    const uint32_t begin = t->shards_begin[lv];
    const uint32_t n = t->n_shards[lv];
    const uint64_t cps_inner = al->cps_inner;
    const uint64_t tps_group_lv = (uint64_t)seg->n_active * cps_inner;
    // Each LOD's offsets range starts at seg->batch_covering_offset + lv
    // (the +lv is the per-LOD shift built into aggregate_batch_luts_unified).
    const uint64_t lod_offsets_base = seg->batch_covering_offset + (uint64_t)lv;

    for (uint32_t si = 0; si < n; ++si) {
      const uint32_t s = begin + si;
      t->h_offsets_base[s] = lod_offsets_base + (uint64_t)si * tps_group_lv;
      t->h_tps_group[s] = tps_group_lv;
      t->h_base_offsets[s] =
        seg->data_segment_offset + (size_t)si * al->shard_capacity;
    }
  }

  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)t->d_base_offsets,
                       t->h_base_offsets,
                       t->total_shards * sizeof(size_t),
                       stream));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)t->d_tps_group,
                       t->h_tps_group,
                       t->total_shards * sizeof(uint64_t),
                       stream));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)t->d_offsets_base,
                       t->h_offsets_base,
                       t->total_shards * sizeof(uint64_t),
                       stream));

  return 0;

Error:
  return 1;
}

int
compress_agg_measure(struct compress_agg_stage* stage,
                     const struct compress_agg_input* in,
                     const struct level_geometry* levels,
                     CUstream compress_stream,
                     struct flush_handoff* out,
                     struct compress_agg_work* work)
{
  const int fc = in->fc;
  const uint32_t n_epochs = in->n_epochs;
  const uint8_t nlod = stage->nlod;

  // --- Phase 1: mask scan (LOD-aware ends here) ----------------------------
  // Per-LOD pool_epochs slices from the [LOD_MAX_LEVELS * K] scratch.
  // Stride is the configured K (pool_epochs_stride), so the cache (same
  // layout) can compare each LOD's slice at a stable offset.
  const uint32_t stride = stage->pool_epochs_stride;
  uint32_t per_lod_n_active[LOD_MAX_LEVELS] = { 0 };
  const uint32_t* per_lod_pool_epochs[LOD_MAX_LEVELS] = { 0 };
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    uint32_t* dst = stage->pool_epochs_scratch + (size_t)lv * stride;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (in->batch_active_masks[e] & (1u << lv))
        dst[k++] = e;
    per_lod_n_active[lv] = k;
    per_lod_pool_epochs[lv] = dst;
  }

  // --- Phase 2: build the per-kick unified batch layout --------------------
  // Page size is uniform across LODs (sink-driven); read from LOD 0.
  struct batch_aggregate_layout layout;
  const size_t page_size = stage->per_lod_agg_layouts[0].page_size;
  CHECK(
    Error,
    batch_aggregate_layout_init(
      &layout, stage->per_lod_agg_layouts, per_lod_n_active, nlod, page_size) ==
      0);

  CHECK(Error, layout.total_data_bytes <= stage->max_total_data_bytes);
  CHECK(Error, layout.total_batch_chunks <= stage->max_total_batch_chunks);
  CHECK(Error, layout.total_batch_covering <= stage->max_total_batch_covering);

  // --- Phase 3: build & upload unified LUTs (cached in steady state) -------
  // Cache key: per-LOD active count AND each LOD's pool_epoch values. Counts
  // alone are insufficient — the gather LUT encodes the actual epoch indices,
  // so two batches with identical counts but different active-epoch positions
  // (mid-stream phase shifts when K doesn't divide an LOD's append period)
  // would mis-hit and reuse stale gather indices. See
  // [[ok-let-s-make-a-curious-prism]].
  int lut_steady =
    stage->lut_cache_valid && memcmp(stage->cached_per_lod_n_active,
                                     per_lod_n_active,
                                     (size_t)nlod * sizeof(uint32_t)) == 0;
  for (uint8_t lv = 0; lut_steady && lv < nlod; ++lv) {
    const uint32_t n_lv = per_lod_n_active[lv];
    if (n_lv == 0)
      continue;
    if (memcmp(stage->cached_pool_epochs + (size_t)lv * stride,
               per_lod_pool_epochs[lv],
               (size_t)n_lv * sizeof(uint32_t)) != 0)
      lut_steady = 0;
  }
  if (lut_steady) {
    stage->lut_steady_count++;
  } else {
    stage->lut_recompute_count++;
    if (layout.total_batch_chunks > 0) {
      aggregate_batch_luts_unified(&layout,
                                   stage->per_lod_agg_layouts,
                                   levels,
                                   per_lod_pool_epochs,
                                   stage->h_lut_gather_scratch,
                                   stage->h_lut_perm_scratch);
      CU(Error,
         cuMemcpyHtoDAsync(stage->d_batch_gather,
                           stage->h_lut_gather_scratch,
                           layout.total_batch_chunks * sizeof(uint32_t),
                           compress_stream));
      CU(Error,
         cuMemcpyHtoDAsync(stage->d_batch_perm,
                           stage->h_lut_perm_scratch,
                           layout.total_batch_chunks * sizeof(uint32_t),
                           compress_stream));
    }
    memcpy(stage->cached_per_lod_n_active,
           per_lod_n_active,
           (size_t)nlod * sizeof(uint32_t));
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      const uint32_t n_lv = per_lod_n_active[lv];
      if (n_lv > 0)
        memcpy(stage->cached_pool_epochs + (size_t)lv * stride,
               per_lod_pool_epochs[lv],
               (size_t)n_lv * sizeof(uint32_t));
    }
    stage->lut_cache_valid = 1;
  }

  // --- Phase 4: per-shard tables ------------------------------------------
  // Always rebuild + upload; the runtime cost is small (≤ a few KB) and the
  // values depend on per_lod_n_active which can shift mid-stream. We do it
  // unconditionally so any layout drift is reflected immediately.
  CHECK(Error,
        build_and_upload_shard_tables(stage, &layout, compress_stream) == 0);

  // --- Phase 5: synchronization ------------------------------------------
  CU(Error, cuStreamWaitEvent(compress_stream, in->pool_ready, 0));
  if (in->lod_done)
    CU(Error, cuStreamWaitEvent(compress_stream, in->lod_done, 0));

  // The host ledger may reserve either output slot after measurement. Wait for
  // both slots' prior D2H before producing temp offsets that the reserved write
  // will later consume on this stream.
  for (int oi = 0; oi < 2; ++oi)
    if (in->prev_d2h_done[oi])
      CU(Error, cuStreamWaitEvent(compress_stream, in->prev_d2h_done[oi], 0));

  // --- Phase 6: compress --------------------------------------------------
  const int skip_compress = (stage->codec.type == CODEC_NONE);
  const CUdeviceptr d_aggregate_src =
    skip_compress ? in->pool_buf : stage->d_compressed[fc];

  if (skip_compress) {
    CU(Error, cuEventRecord(stage->t_compress_start[fc], compress_stream));
    CU(Error, cuEventRecord(stage->t_compress_end[fc], compress_stream));
  } else {
    CHECK_MUL_OVERFLOW(Error, n_epochs, levels->total_chunks, UINT64_MAX);
    uint64_t batch_chunks = (uint64_t)n_epochs * levels->total_chunks;
    CHECK(Error,
          kick_compress(stage,
                        fc,
                        (void*)in->pool_buf,
                        batch_chunks,
                        stage->codec.chunk_bytes,
                        compress_stream) == 0);
  }

  const int output_idx = in->output_idx;
  struct aggregate_slot* out_slot = &stage->output[output_idx];
  if (layout.total_batch_chunks > 0) {
    // The gate only matters when shards have per-page tails (page_size > 0):
    // rollforward_tail_unified_k corrupts in-slot if a shard finalizes
    // mid-slot. page_size == 0 has no tail rollforward, so stacking is safe.
    int would_finalize_stay = 0;
    int would_finalize_alone = 0;
    if (page_size > 0) {
      uint64_t epoch_stay[LOD_MAX_LEVELS] = { 0 };
      uint64_t epoch_alone[LOD_MAX_LEVELS] = { 0 };
      uint64_t cps_append[LOD_MAX_LEVELS] = { 0 };
      for (uint8_t lv = 0; lv < nlod; ++lv) {
        epoch_alone[lv] = stage->shard[lv].epoch_in_shard;
        epoch_stay[lv] = epoch_alone[lv] + out_slot->stacked_n_active[lv];
        cps_append[lv] = stage->shard[lv].chunks_per_shard_append;
      }
      would_finalize_stay =
        !no_shard_finalizes(nlod, epoch_stay, per_lod_n_active, cps_append);
      would_finalize_alone =
        !no_shard_finalizes(nlod, epoch_alone, per_lod_n_active, cps_append);
    }

    const struct aggregate_batch_measure_params measure_params = {
      .indices = {
        .d_comp_sizes = stage->codec.d_comp_sizes,
        .d_batch_gather = (const uint32_t*)(uintptr_t)stage->d_batch_gather,
        .d_batch_perm = (const uint32_t*)(uintptr_t)stage->d_batch_perm,
      },
      .shape = {
        .total_batch_chunks = layout.total_batch_chunks,
        .total_batch_covering = layout.total_batch_covering,
        .nlod = nlod,
      },
      .slot = out_slot,
      .shards = {
        .d_shard_tps_group = stage->shards.d_tps_group,
        .d_shard_offsets_base = stage->shards.d_offsets_base,
        .d_tail_bytes = stage->shards.d_tail_bytes,
        .d_tail_carry = stage->shards.d_tail_carry,
        .page_size = page_size,
        .total_shards = stage->shards.total_shards,
      },
      .would_finalize_stay = would_finalize_stay,
      .would_finalize_alone = would_finalize_alone,
      .outputs = {
        .d_measurement = stage->d_measurement[fc],
        .h_measurement = stage->h_measurement[fc],
        .measurement_ready = stage->measurement_ready[fc],
        .d_tail_sum_bytes = stage->d_tail_sum_bytes,
        .d_temp_offsets = stage->d_temp_offsets,
        .d_temp_perm_sizes = stage->d_temp_perm_sizes,
      },
      .stream = compress_stream,
    };
    CHECK(Error, aggregate_batch_measure_unified_async(&measure_params) == 0);
  }

  // --- Phase 8: fill handoff ----------------------------------------------
  out->fc = fc;
  out->output_idx = output_idx;
  out->n_epochs = n_epochs;
  out->active_levels_mask = in->active_levels_mask;
  out->batch_active_masks = in->batch_active_masks;
  out->nlod = nlod;
  memcpy(
    out->per_lod_n_active, per_lod_n_active, (size_t)nlod * sizeof(uint32_t));
  out->t_aggregate_end = stage->t_aggregate_end[fc];
  out->t_compress_start = stage->t_compress_start[fc];
  out->t_compress_end = stage->t_compress_end[fc];
  out->max_output_size = stage->codec.max_output_size;
  out->passthrough = (stage->codec.type == CODEC_NONE);
  out->output = out_slot;
  out->layout = layout;
  out->slot_total_data_bytes = layout.total_data_bytes;
  out->slot_total_desc_entries = layout.total_batch_covering + (uint64_t)nlod;
  out->per_lod_agg_layouts = stage->per_lod_agg_layouts;
  out->shards = &stage->shards;
  for (uint8_t lv = 0; lv < nlod; ++lv)
    out->shards_by_lod[lv] = &stage->shard[lv];

  if (work) {
    *work = (struct compress_agg_work){
      .fc = fc,
      .active_output_idx = output_idx,
      .d_aggregate_src = (const void*)d_aggregate_src,
      .scratch_slot = out_slot,
      .layout = layout,
      .page_size = page_size,
    };
    memcpy(work->per_lod_n_active,
           per_lod_n_active,
           (size_t)nlod * sizeof(uint32_t));
  }

  return 0;

Error:
  return 1;
}

int
compress_agg_write_reserved(struct compress_agg_stage* stage,
                            const struct compress_agg_work* work,
                            const struct output_slot_reservation* reservation,
                            CUstream compress_stream)
{
  CHECK(Error, stage);
  CHECK(Error, work);
  CHECK(Error, reservation);

  const int fc = work->fc;
  const uint8_t nlod = stage->nlod;
  const struct batch_aggregate_layout* layout = &work->layout;
  struct aggregate_slot* target = &stage->output[reservation->slot];

  if (layout->total_batch_chunks > 0) {
    const uint32_t ring_idx = (uint32_t)(stage->cb_args_seq & 3);
    stage->cb_args_seq++;
    struct aggregate_write_cb_args* cb_args = &stage->cb_args_ring[ring_idx];
    cb_args->layout = *layout;
    cb_args->nlod = nlod;
    memcpy(cb_args->per_lod_n_active,
           work->per_lod_n_active,
           (size_t)nlod * sizeof(uint32_t));

    const struct aggregate_slot_reservation agg_reservation = {
      .slot = reservation->slot,
      .close_slot =
        reservation->close_before_append ? reservation->close_slot : -1,
      .data_base = reservation->data_base,
      .desc_base = reservation->desc_base,
      .batch_index = reservation->batch_index,
    };

    const struct aggregate_batch_write_reserved_params write_params = {
      .batch = {
        .indices = {
          .d_comp_sizes = stage->codec.d_comp_sizes,
          .d_batch_gather = (const uint32_t*)(uintptr_t)stage->d_batch_gather,
          .d_batch_perm = (const uint32_t*)(uintptr_t)stage->d_batch_perm,
        },
        .d_compressed = work->d_aggregate_src,
        .max_comp_chunk_bytes = stage->codec.max_output_size,
      },
      .shape = {
        .total_batch_chunks = layout->total_batch_chunks,
        .total_batch_covering = layout->total_batch_covering,
        .nlod = nlod,
      },
      .scratch_slot = work->scratch_slot,
      .target_slot = target,
      .reservation = &agg_reservation,
      .shards = {
        .d_shard_tps_group = stage->shards.d_tps_group,
        .d_shard_offsets_base = stage->shards.d_offsets_base,
        .d_tail_bytes = stage->shards.d_tail_bytes,
        .d_tail_carry = stage->shards.d_tail_carry,
        .page_size = work->page_size,
        .total_shards = stage->shards.total_shards,
      },
      .desc = stage->d_write_desc,
      .d_measurement = stage->d_measurement[fc],
      .d_temp_offsets = stage->d_temp_offsets,
      .d_temp_perm_sizes = stage->d_temp_perm_sizes,
      .cb_args = cb_args,
      .stream = compress_stream,
    };
    CHECK(Error,
          aggregate_batch_write_reserved_unified_async(&write_params) == 0);
  } else {
    CU(Error,
       cuEventRecord(work->scratch_slot->host_func_done, compress_stream));
  }

  CU(Error, cuEventRecord(stage->t_aggregate_end[fc], compress_stream));
  return 0;

Error:
  return 1;
}

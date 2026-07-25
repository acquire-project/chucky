#include "gpu/flush.compress_agg.h"

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

// --- Init / Destroy ---

int
compress_agg_init_shared(struct compress_agg_stage* stage,
                         const struct engine_limits* lim,
                         enum compression_codec codec_id,
                         struct gpu_ordering* ord,
                         CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->ord = ord;
  gpu_pool_init(
    &stage->agg_pool, ord, GPU_EDGE_AGG_DONE, GPU_EDGE_SLOT_DRAINED);
  gpu_pool_init(&stage->agg_host, ord, GPU_EDGE_D2H_DONE, GPU_EDGE_COUNT);
  gpu_pool_init(
    &stage->agg_index, ord, GPU_EDGE_CHUNK_INDEX_READY, GPU_EDGE_COUNT);
  for (int fc = 0; fc < 2; ++fc) {
    gpu_pool_bind(&stage->agg_pool, fc, &stage->agg[fc]);
    gpu_pool_bind(&stage->agg_host, fc, &stage->agg[fc]);
    gpu_pool_bind(&stage->agg_index, fc, &stage->agg[fc]);
  }
  gpu_pool_init(&stage->tail, ord, GPU_EDGE_TAIL_PUBLISHED, GPU_EDGE_COUNT);
  // &stage->ar is stable across multiarray binds (swapped by value).
  gpu_pool_bind(&stage->tail, 0, &stage->ar);

  const uint32_t K = lim->epochs_per_batch;
  const uint64_t M = lim->codec_batch;

  // Codec
  CHECK(Fail, codec_init(&stage->codec, codec_id, lim->chunk_bytes, M) == 0);

  stage->pool_epochs_stride = K;
  stage->pool_epochs_scratch =
    (uint32_t*)malloc((size_t)LOD_MAX_LEVELS * K * sizeof(uint32_t));
  stage->cached_pool_epochs =
    (uint32_t*)malloc((size_t)LOD_MAX_LEVELS * K * sizeof(uint32_t));
  CHECK(Fail, stage->pool_epochs_scratch && stage->cached_pool_epochs);

  CHECK_MUL_OVERFLOW(Fail, M, stage->codec.max_output_size, SIZE_MAX);
  // Compressed buffers + events. CODEC_NONE aggregates directly from pool_buf
  // (see compress_agg_aggregate), so the d_compressed buffer is unused — skip
  // its M * chunk_bytes allocation per fc. Destroy is NULL-safe.
  const int need_compressed = (stage->codec.type != CODEC_NONE);
  for (int fc = 0; fc < 2; ++fc) {
    if (need_compressed)
      CU(
        Fail,
        cuMemAlloc(&stage->d_compressed[fc], M * stage->codec.max_output_size));
    CU(Fail, cuEventCreate(&stage->t_compress_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->t_compress_end[fc], CU_EVENT_DEFAULT));
  }

  stage->max_total_batch_chunks = lim->max_total_batch_chunks;
  stage->max_total_batch_covering = lim->max_total_batch_covering;
  stage->max_total_data_bytes = lim->max_total_data_bytes;

  if (stage->max_total_batch_chunks > 0) {
    const uint64_t C_max =
      stage->max_total_batch_covering + (uint64_t)lim->max_nlod;
    for (int fc = 0; fc < 2; ++fc) {
      CHECK(Fail,
            aggregate_batch_slot_init(&stage->agg[fc],
                                      C_max,
                                      stage->max_total_data_bytes) == 0);
    }

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

  // Sized to the max total_shards; the active array's slice is re-uploaded
  // by every kick.
  if (lim->max_total_shards > 0) {
    const uint64_t ts = lim->max_total_shards;
    stage->shards.h_base_offsets = (size_t*)malloc(ts * sizeof(size_t));
    stage->shards.h_tps_group = (uint64_t*)malloc(ts * sizeof(uint64_t));
    stage->shards.h_offsets_base = (uint64_t*)malloc(ts * sizeof(uint64_t));
    CHECK(Fail,
          stage->shards.h_base_offsets && stage->shards.h_tps_group &&
            stage->shards.h_offsets_base);
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->shards.d_base_offsets,
                  ts * sizeof(size_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->shards.d_shard_capacity,
                  ts * sizeof(size_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->shards.d_tps_group,
                  ts * sizeof(uint64_t)));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&stage->shards.d_offsets_base,
                  ts * sizeof(uint64_t)));
  }

  // Seed timing events so the first metric reads see a valid interval.
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventRecord(stage->t_compress_start[fc], compute));
    CU(Fail, cuEventRecord(stage->t_compress_end[fc], compute));
  }

  return 0;

Fail:
  compress_agg_destroy_shared(stage);
  return 1;
}

void
compress_agg_destroy_shared(struct compress_agg_stage* stage)
{
  if (!stage)
    return;
  if (stage->ord && gpu_ordering_gate_active(stage->ord)) {
    // An undrained kick (failed flush) parks work on the compress stream;
    // the frees below can block on pending work, so release first.
    gpu_pool_release_all(&stage->tail);
    CUWARN(cuCtxSynchronize());
  }
  codec_free(&stage->codec);
  free(stage->pool_epochs_scratch);
  free(stage->cached_pool_epochs);
  stage->pool_epochs_scratch = NULL;
  stage->cached_pool_epochs = NULL;
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(stage->d_compressed[fc]);
    cu_event_destroy(stage->t_compress_start[fc]);
    cu_event_destroy(stage->t_compress_end[fc]);
    stage->d_compressed[fc] = 0;
    stage->t_compress_start[fc] = NULL;
    stage->t_compress_end[fc] = NULL;
  }
  if (stage->lut_steady_count + stage->lut_recompute_count > 0) {
    const uint64_t tot = stage->lut_steady_count + stage->lut_recompute_count;
    log_debug("compress_agg unified lut_steady=%llu lut_recompute=%llu "
              "(steady=%.1f%%)",
              (unsigned long long)stage->lut_steady_count,
              (unsigned long long)stage->lut_recompute_count,
              100.0 * (double)stage->lut_steady_count / (double)tot);
  }
  for (int fc = 0; fc < 2; ++fc)
    aggregate_slot_destroy(&stage->agg[fc]);
  cu_mem_free(stage->d_batch_gather);
  cu_mem_free(stage->d_batch_perm);
  stage->d_batch_gather = 0;
  stage->d_batch_perm = 0;
  free(stage->h_lut_gather_scratch);
  free(stage->h_lut_perm_scratch);
  stage->h_lut_gather_scratch = NULL;
  stage->h_lut_perm_scratch = NULL;
  free(stage->shards.h_base_offsets);
  free(stage->shards.h_tps_group);
  free(stage->shards.h_offsets_base);
  cu_mem_free((CUdeviceptr)stage->shards.d_base_offsets);
  cu_mem_free((CUdeviceptr)stage->shards.d_shard_capacity);
  cu_mem_free((CUdeviceptr)stage->shards.d_tps_group);
  cu_mem_free((CUdeviceptr)stage->shards.d_offsets_base);
  memset(&stage->shards, 0, sizeof(stage->shards));
}

int
compress_agg_array_init(struct compress_agg_array* ar,
                        const struct computed_stream_layouts* cl,
                        struct gpu_ordering* gate_ord,
                        CUstream gate_stream)
{
  memset(ar, 0, sizeof(*ar));

  // --- Unified across-LODs aggregate state ---------------------------------
  // Mirrors the CPU pipeline (src/cpu/pipeline.c + src/cpu/aggregate.c).

  ar->nlod = (uint8_t)cl->levels.nlod;

  // Own copy so multiarray bind/unbind can swap them per-array.
  for (int lv = 0; lv < cl->levels.nlod; ++lv) {
    ar->per_lod_agg_layouts[lv] = cl->per_level[lv].agg_layout;
    CHECK(Fail, aggregate_layout_upload(&ar->per_lod_agg_layouts[lv]) == 0);
  }

  uint64_t total_shards = 0;
  for (int lv = 0; lv < cl->levels.nlod; ++lv) {
    ar->shards_begin[lv] = (uint32_t)total_shards;
    ar->n_shards[lv] = (uint32_t)cl->per_level[lv].agg_layout.num_shards;
    total_shards += cl->per_level[lv].agg_layout.num_shards;
  }
  ar->total_shards = total_shards;
  ar->page_size = cl->per_level[0].agg_layout.page_size;

  if (total_shards > 0) {
    ar->h_shard_capacity = (size_t*)malloc(total_shards * sizeof(size_t));
    ar->h_tail_bytes = (size_t*)calloc(total_shards, sizeof(size_t));
    CHECK(Fail, ar->h_shard_capacity && ar->h_tail_bytes);
    for (uint64_t i = 0; i < total_shards; ++i) {
      for (int lv = 0; lv < cl->levels.nlod; ++lv) {
        if (i >= ar->shards_begin[lv] &&
            i < ar->shards_begin[lv] + ar->n_shards[lv]) {
          ar->h_shard_capacity[i] = cl->per_level[lv].agg_layout.shard_capacity;
          break;
        }
      }
    }

    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&ar->d_tail_bytes,
                  total_shards * sizeof(size_t)));
    CU(Fail,
       cuMemsetD8(
         (CUdeviceptr)ar->d_tail_bytes, 0, total_shards * sizeof(size_t)));

    // Delivery's tail upload is a synchronous HtoD from pageable memory,
    // which may return before the DMA reaches the device; SYNC_MEMOPS
    // makes it complete at the device first, so the tail-gate publish
    // that follows it cannot outrun the copy (flush.d2h_deliver.c).
    {
      unsigned int sync_memops = 1;
      CU(Fail,
         cuPointerSetAttribute(&sync_memops,
                               CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                               (CUdeviceptr)ar->d_tail_bytes));
    }

    // One layout across LODs: the sink page size is uniform.
    if (ar->page_size > 0) {
      ar->tail_carry_bytes = total_shards * ar->page_size;
      CU(Fail, cuMemAlloc(&ar->d_tail_carry, ar->tail_carry_bytes));
      CU(Fail, cuMemsetD8(ar->d_tail_carry, 0, ar->tail_carry_bytes));
      // Same pageable-HtoD constraint as d_tail_bytes above.
      {
        unsigned int sync_memops = 1;
        CU(Fail,
           cuPointerSetAttribute(&sync_memops,
                                 CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 ar->d_tail_carry));
      }

      // Without stream memops — or when the counter can't be allocated or
      // mapped — the lazy path host-drains instead (SCHEDULE_DRAIN_BEFORE_KICK).
      if (gate_ord) {
        CHECK(Fail, gpu_ordering_gate_init(gate_ord, gate_stream) == 0);
        if (!gpu_ordering_gate_supported(gate_ord))
          log_warn("compress_agg: tail gate unavailable; page-aligned "
                   "pipeline degrades to host-ordered tail uploads");
      }
    }
  }

  for (int lv = 0; lv < cl->levels.nlod; ++lv)
    CHECK(Fail, init_shard_state(&ar->shard[lv], &cl->per_level[lv]) == 0);

  return 0;

Fail:
  compress_agg_array_destroy(ar);
  return 1;
}

void
compress_agg_array_destroy(struct compress_agg_array* ar)
{
  if (!ar)
    return;
  for (int lv = 0; lv < ar->nlod; ++lv) {
    aggregate_layout_destroy(&ar->per_lod_agg_layouts[lv]);
    shard_state_destroy(&ar->shard[lv]);
  }
  free(ar->h_shard_capacity);
  free(ar->h_tail_bytes);
  cu_mem_free((CUdeviceptr)ar->d_tail_bytes);
  cu_mem_free(ar->d_tail_carry);
  memset(ar, 0, sizeof(*ar));
}

int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  struct gpu_ordering* ord,
                  CUstream compute)
{
  struct engine_limits lim;
  memset(&lim, 0, sizeof(lim));
  CHECK(Fail, engine_limits_accumulate(&lim, cl, config) == 0);
  CHECK(Fail,
        compress_agg_init_shared(stage, &lim, config->codec.id, ord, compute) ==
          0);
  CHECK(Fail, compress_agg_array_init(&stage->ar, cl, ord, compute) == 0);
  // d_shard_capacity is constant per array — upload once; the other shard
  // tables depend on per-batch active counts and are uploaded by the kick.
  if (stage->ar.total_shards > 0)
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)stage->shards.d_shard_capacity,
                    stage->ar.h_shard_capacity,
                    stage->ar.total_shards * sizeof(size_t)));
  return 0;

Fail:
  compress_agg_destroy(stage);
  return 1;
}

void
compress_agg_destroy(struct compress_agg_stage* stage)
{
  if (!stage)
    return;
  compress_agg_destroy_shared(stage);
  compress_agg_array_destroy(&stage->ar);
}

// --- Memory estimate ---

// Mirrors compress_agg_init_shared + compress_agg_array_init for one array,
// from the same engine_limits the real init consumes.
int
compress_agg_memory_estimate(const struct engine_limits* lim,
                             const struct computed_stream_layouts* cl,
                             enum compression_codec codec_id,
                             size_t* compressed_pool_bytes,
                             size_t* codec_bytes,
                             size_t* aggregate_device_bytes,
                             size_t* aggregate_host_bytes)
{
  // d_compressed[2]; skipped for CODEC_NONE (kick aggregates from the pool).
  *compressed_pool_bytes =
    (codec_id == CODEC_NONE)
      ? 0
      : 2 * (size_t)lim->codec_batch * cl->max_output_size;

  *codec_bytes =
    codec_device_bytes(codec_id, lim->chunk_bytes, lim->codec_batch);

  size_t dev = 0;
  size_t host = 0;

  if (lim->max_total_batch_chunks > 0) {
    const uint64_t C_max =
      lim->max_total_batch_covering + (uint64_t)lim->max_nlod;
    size_t slot_dev = 0;
    size_t slot_host = 0;
    CHECK(Error,
          aggregate_batch_slot_memory(
            C_max, lim->max_total_data_bytes, &slot_dev, &slot_host) == 0);
    dev += 2 * slot_dev;  // agg[2]
    host += 2 * slot_host;
    dev += 2 * lim->max_total_batch_chunks *
           sizeof(uint32_t); // d_batch_gather + d_batch_perm
  }

  // Shared per-shard device tables: d_base_offsets, d_shard_capacity,
  // d_tps_group, d_offsets_base.
  dev +=
    lim->max_total_shards * (2 * sizeof(size_t) + 2 * sizeof(uint64_t));

  // Per-array slice (compress_agg_array_init).
  {
    uint64_t total_shards = 0;
    for (int lv = 0; lv < cl->levels.nlod; ++lv) {
      dev += aggregate_layout_device_bytes(&cl->per_level[lv].agg_layout);
      total_shards += cl->per_level[lv].agg_layout.num_shards;
    }
    if (total_shards > 0) {
      dev += total_shards * sizeof(size_t); // d_tail_bytes
      const size_t page_size = cl->per_level[0].agg_layout.page_size;
      if (page_size > 0)
        dev += total_shards * page_size; // d_tail_carry
    }
  }

  *aggregate_device_bytes = dev;
  *aggregate_host_bytes = host;
  return 0;

Error:
  return 1;
}

// --- Kick phases (acquire/release placement lives in schedule.c) ---

// Build per-shard tables for this batch's `layout`. Populates host shadows
// and uploads to device on `stream`. tables.h_shard_capacity is constant and
// uploaded at init; not re-uploaded here. The rest is rebuilt every kick:
// it depends on per_lod_n_active, which can shift mid-stream.
static int
build_and_upload_shard_tables(struct compress_agg_stage* stage,
                              const struct batch_aggregate_layout* layout,
                              CUstream stream)
{
  struct shard_tables* t = &stage->shards;
  const struct compress_agg_array* ar = &stage->ar;
  if (ar->total_shards == 0)
    return 0;

  for (uint8_t lv = 0; lv < layout->nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    const struct aggregate_layout* al = &stage->ar.per_lod_agg_layouts[lv];
    const uint32_t begin = ar->shards_begin[lv];
    const uint32_t n = ar->n_shards[lv];
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
                       ar->total_shards * sizeof(size_t),
                       stream));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)t->d_tps_group,
                       t->h_tps_group,
                       ar->total_shards * sizeof(uint64_t),
                       stream));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)t->d_offsets_base,
                       t->h_offsets_base,
                       ar->total_shards * sizeof(uint64_t),
                       stream));

  return 0;

Error:
  return 1;
}

static void
scan_active_masks(struct compress_agg_stage* stage,
                  const struct compress_agg_input* in,
                  uint32_t* per_lod_n_active,
                  const uint32_t** per_lod_pool_epochs)
{
  const uint32_t stride = stage->pool_epochs_stride;
  for (uint8_t lv = 0; lv < stage->ar.nlod; ++lv) {
    uint32_t* dst = stage->pool_epochs_scratch + (size_t)lv * stride;
    uint32_t k = 0;
    for (uint32_t e = 0; e < in->n_epochs; ++e)
      if (in->batch_active_masks[e] & (1u << lv))
        dst[k++] = e;
    per_lod_n_active[lv] = k;
    per_lod_pool_epochs[lv] = dst;
  }
}

static int
build_batch_layout(struct compress_agg_stage* stage,
                   const uint32_t* per_lod_n_active,
                   struct batch_aggregate_layout* layout)
{
  // Page size is uniform across LODs (sink-driven); read from LOD 0.
  const size_t page_size = stage->ar.per_lod_agg_layouts[0].page_size;
  CHECK(Error,
        batch_aggregate_layout_init(layout,
                                    stage->ar.per_lod_agg_layouts,
                                    per_lod_n_active,
                                    stage->ar.nlod,
                                    page_size) == 0);

  CHECK(Error, layout->total_data_bytes <= stage->max_total_data_bytes);
  CHECK(Error, layout->total_batch_chunks <= stage->max_total_batch_chunks);
  CHECK(Error, layout->total_batch_covering <= stage->max_total_batch_covering);
  return 0;

Error:
  return 1;
}

static int
build_and_upload_luts(struct compress_agg_stage* stage,
                      const struct batch_aggregate_layout* layout,
                      const struct level_geometry* levels,
                      const uint32_t* per_lod_n_active,
                      const uint32_t* const* per_lod_pool_epochs,
                      CUstream compress_stream)
{
  const uint8_t nlod = stage->ar.nlod;
  const uint32_t stride = stage->pool_epochs_stride;

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
    return 0;
  }

  stage->lut_recompute_count++;
  if (layout->total_batch_chunks > 0) {
    aggregate_batch_luts_unified(layout,
                                 stage->ar.per_lod_agg_layouts,
                                 levels,
                                 per_lod_pool_epochs,
                                 stage->h_lut_gather_scratch,
                                 stage->h_lut_perm_scratch);
    CU(Error,
       cuMemcpyHtoDAsync(stage->d_batch_gather,
                         stage->h_lut_gather_scratch,
                         layout->total_batch_chunks * sizeof(uint32_t),
                         compress_stream));
    CU(Error,
       cuMemcpyHtoDAsync(stage->d_batch_perm,
                         stage->h_lut_perm_scratch,
                         layout->total_batch_chunks * sizeof(uint32_t),
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
  return 0;

Error:
  return 1;
}

int
compress_agg_prepare(struct compress_agg_stage* stage,
                     const struct compress_agg_input* in,
                     const struct level_geometry* levels,
                     CUstream compress_stream,
                     struct compress_agg_plan* plan)
{
  const uint32_t* per_lod_pool_epochs[LOD_MAX_LEVELS] = { 0 };
  memset(plan, 0, sizeof(*plan));
  scan_active_masks(stage, in, plan->per_lod_n_active, per_lod_pool_epochs);
  CHECK(Error,
        build_batch_layout(stage, plan->per_lod_n_active, &plan->layout) == 0);
  CHECK(Error,
        build_and_upload_luts(stage,
                              &plan->layout,
                              levels,
                              plan->per_lod_n_active,
                              per_lod_pool_epochs,
                              compress_stream) == 0);
  CHECK(Error,
        build_and_upload_shard_tables(stage, &plan->layout, compress_stream) ==
          0);
  return 0;

Error:
  return 1;
}

int
compress_agg_compress(struct compress_agg_stage* stage,
                      const struct compress_agg_input* in,
                      const struct level_geometry* levels,
                      struct gpu_pool_view pool_buf,
                      CUstream compress_stream)
{
  const int fc = in->fc;
  if (stage->codec.type == CODEC_NONE) {
    // Timing events still bracket the skipped phase so metrics stay valid.
    CU(Error, cuEventRecord(stage->t_compress_start[fc], compress_stream));
    CU(Error, cuEventRecord(stage->t_compress_end[fc], compress_stream));
  } else {
    CHECK_MUL_OVERFLOW(Error, in->n_epochs, levels->total_chunks, UINT64_MAX);
    uint64_t batch_chunks = (uint64_t)in->n_epochs * levels->total_chunks;
    CHECK(Error,
          kick_compress(stage,
                        fc,
                        pool_buf.p,
                        batch_chunks,
                        stage->codec.chunk_bytes,
                        compress_stream) == 0);
  }
  return 0;

Error:
  return 1;
}

int
compress_agg_aggregate(struct compress_agg_stage* stage,
                       const struct compress_agg_plan* plan,
                       int fc,
                       struct aggregate_slot* slot,
                       struct gpu_pool_view pool_buf,
                       CUstream compress_stream)
{
  // CODEC_NONE aggregates straight from the pool buffer, skipping compress.
  const CUdeviceptr d_aggregate_src = (stage->codec.type == CODEC_NONE)
                                        ? gpu_pool_view_d(pool_buf)
                                        : stage->d_compressed[fc];
  if (plan->layout.total_batch_chunks > 0) {
    const size_t page_size = stage->ar.per_lod_agg_layouts[0].page_size;
    CHECK(Error,
          aggregate_batch_unified_async(
            (const void*)d_aggregate_src,
            stage->codec.d_comp_sizes,
            (const uint32_t*)(uintptr_t)stage->d_batch_gather,
            (const uint32_t*)(uintptr_t)stage->d_batch_perm,
            plan->layout.total_batch_chunks,
            plan->layout.total_batch_covering,
            stage->ar.nlod,
            stage->codec.max_output_size,
            slot,
            stage->shards.d_base_offsets,
            stage->shards.d_shard_capacity,
            stage->shards.d_tps_group,
            stage->shards.d_offsets_base,
            stage->ar.d_tail_bytes,
            stage->ar.d_tail_carry,
            page_size,
            stage->ar.total_shards,
            compress_stream) == 0);
  }
  return 0;

Error:
  return 1;
}

void
compress_agg_fill_handoff(struct compress_agg_stage* stage,
                          const struct compress_agg_input* in,
                          const struct compress_agg_plan* plan,
                          struct flush_handoff* out)
{
  const int fc = in->fc;
  const uint8_t nlod = stage->ar.nlod;
  out->fc = fc;
  out->n_epochs = in->n_epochs;
  out->lod_timing_slot = in->lod_timing_slot;
  out->has_lod_timing = in->has_lod_timing;
  out->active_levels_mask = in->active_levels_mask;
  out->batch_active_masks = in->batch_active_masks;
  out->nlod = nlod;
  memcpy(out->per_lod_n_active,
         plan->per_lod_n_active,
         (size_t)nlod * sizeof(uint32_t));
  out->t_aggregate_end = gpu_ordering_event(stage->ord, GPU_EDGE_AGG_DONE, fc);
  out->t_compress_start = stage->t_compress_start[fc];
  out->t_compress_end = stage->t_compress_end[fc];
  out->max_output_size = stage->codec.max_output_size;
  out->passthrough = (stage->codec.type == CODEC_NONE);
  out->agg_pool = &stage->agg_pool;
  out->agg_host = &stage->agg_host;
  out->agg_index = &stage->agg_index;
  out->tail = &stage->tail;
  out->layout = plan->layout;
  out->per_lod_agg_layouts = stage->ar.per_lod_agg_layouts;
  for (uint8_t lv = 0; lv < nlod; ++lv)
    out->shards_by_lod[lv] = &stage->ar.shard[lv];
}

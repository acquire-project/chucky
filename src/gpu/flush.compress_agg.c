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
    CU(Fail, cuEventCreate(&stage->t_aggregate_start[fc], CU_EVENT_DEFAULT));
  }

  stage->max_total_batch_chunks = lim->max_total_batch_chunks;
  stage->max_total_batch_covering = lim->max_total_batch_covering;
  stage->max_device_data_bytes = lim->max_device_data_bytes;
  stage->max_host_data_bytes = lim->max_host_data_bytes;

  if (stage->max_total_batch_chunks > 0) {
    const uint64_t C_max =
      stage->max_total_batch_covering + (uint64_t)lim->max_nlod;
    for (int fc = 0; fc < 2; ++fc) {
      CHECK(Fail,
            aggregate_batch_slot_init(&stage->agg[fc],
                                      C_max,
                                      stage->max_device_data_bytes,
                                      stage->max_host_data_bytes) == 0);
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

  // Seed timing events so the first metric reads see a valid interval.
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventRecord(stage->t_compress_start[fc], compute));
    CU(Fail, cuEventRecord(stage->t_compress_end[fc], compute));
    CU(Fail, cuEventRecord(stage->t_aggregate_start[fc], compute));
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
  codec_free(&stage->codec);
  free(stage->pool_epochs_scratch);
  free(stage->cached_pool_epochs);
  stage->pool_epochs_scratch = NULL;
  stage->cached_pool_epochs = NULL;
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(stage->d_compressed[fc]);
    cu_event_destroy(stage->t_compress_start[fc]);
    cu_event_destroy(stage->t_compress_end[fc]);
    cu_event_destroy(stage->t_aggregate_start[fc]);
    stage->d_compressed[fc] = 0;
    stage->t_compress_start[fc] = NULL;
    stage->t_compress_end[fc] = NULL;
    stage->t_aggregate_start[fc] = NULL;
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
}

int
compress_agg_array_init(struct compress_agg_array* ar,
                        const struct computed_stream_layouts* cl)
{
  memset(ar, 0, sizeof(*ar));

  // --- Unified across-LODs aggregate state ---------------------------------
  // Mirrors the CPU pipeline (src/cpu/pipeline.c + src/cpu/aggregate.c).

  ar->nlod = (uint8_t)cl->levels.nlod;

  // Own copy so multiarray bind/unbind can swap them per-array.
  for (int lv = 0; lv < cl->levels.nlod; ++lv) {
    ar->per_lod_agg_layouts[lv] = cl->per_level[lv].agg_layout;
    ar->total_shards += cl->per_level[lv].agg_layout.num_shards;
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
  for (int lv = 0; lv < ar->nlod; ++lv)
    shard_state_destroy(&ar->shard[lv]);
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
  CHECK(Fail, compress_agg_array_init(&stage->ar, cl) == 0);
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
          aggregate_batch_slot_memory(C_max,
                                      lim->max_device_data_bytes,
                                      lim->max_host_data_bytes,
                                      &slot_dev,
                                      &slot_host) == 0);
    dev += 2 * slot_dev; // agg[2]
    host += 2 * slot_host;
    dev += 2 * lim->max_total_batch_chunks *
           sizeof(uint32_t); // d_batch_gather + d_batch_perm
  }

  *aggregate_device_bytes = dev;
  *aggregate_host_bytes = host;
  return 0;

Error:
  return 1;
}

// --- Kick phases (acquire/release placement lives in schedule.c) ---

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
  CHECK(Error,
        batch_aggregate_layout_init_compact(layout,
                                            stage->ar.per_lod_agg_layouts,
                                            per_lod_n_active,
                                            stage->ar.nlod) == 0);

  CHECK(Error, layout->total_data_bytes <= stage->max_device_data_bytes);
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
  if (lut_steady)
    return 0;

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
  CU(Error, cuEventRecord(stage->t_aggregate_start[fc], compress_stream));
  // CODEC_NONE aggregates straight from the pool buffer, skipping compress.
  const CUdeviceptr d_aggregate_src = (stage->codec.type == CODEC_NONE)
                                        ? gpu_pool_view_d(pool_buf)
                                        : stage->d_compressed[fc];
  if (plan->layout.total_batch_chunks > 0) {
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
  out->active_levels_mask = in->active_levels_mask;
  out->batch_active_masks = in->batch_active_masks;
  out->nlod = nlod;
  memcpy(out->per_lod_n_active,
         plan->per_lod_n_active,
         (size_t)nlod * sizeof(uint32_t));
  out->t_aggregate_end = gpu_ordering_event(stage->ord, GPU_EDGE_AGG_DONE, fc);
  out->t_compress_start = stage->t_compress_start[fc];
  out->t_compress_end = stage->t_compress_end[fc];
  out->t_aggregate_start = stage->t_aggregate_start[fc];
  out->max_output_size = stage->codec.max_output_size;
  out->passthrough = (stage->codec.type == CODEC_NONE);
  out->agg_pool = &stage->agg_pool;
  out->agg_host = &stage->agg_host;
  out->agg_index = &stage->agg_index;
  out->layout = plan->layout;
  out->per_lod_agg_layouts = stage->ar.per_lod_agg_layouts;
  for (uint8_t lv = 0; lv < nlod; ++lv)
    out->shards_by_lod[lv] = &stage->ar.shard[lv];

  out->device_batch = (struct device_aggregate_batch){
    .slot_index = fc,
    .extent_kind = stage->codec.type == CODEC_NONE
                     ? DEVICE_AGGREGATE_FIXED_EXTENT
                     : DEVICE_AGGREGATE_INDEXED_EXTENT,
    .layout = plan->layout,
    .nlod = nlod,
    .per_lod_layouts = stage->ar.per_lod_agg_layouts,
    .fixed_chunk_bytes = stage->codec.max_output_size,
    .aggregate_pool = &stage->agg_pool,
    .host_pool = &stage->agg_host,
    .index_pool = &stage->agg_index,
    .completion = out->t_aggregate_end,
  };
  memcpy(out->device_batch.per_lod_n_active,
         plan->per_lod_n_active,
         (size_t)nlod * sizeof(uint32_t));
}

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

// --- Init / Destroy ---

// Mirrors the old init_compression + init_aggregate_and_shards +
// init_batch_luts from stream_init.c, but scoped to this stage.

static void
destroy_level_state(struct level_flush_state* lls)
{
  aggregate_layout_destroy(&lls->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&lls->agg[i]);
  cu_mem_free(lls->d_batch_gather);
  cu_mem_free(lls->d_batch_perm);
  cu_mem_free(lls->d_tail_carry);
  cu_mem_free((CUdeviceptr)lls->d_tail_bytes);
  lls->d_tail_bytes = NULL;
  free(lls->h_tail_bytes);
  lls->h_tail_bytes = NULL;
  if (lls->shard.shards) {
    for (uint64_t i = 0; i < lls->shard.shard_inner_count; ++i) {
      free(lls->shard.shards[i].index);
      free(lls->shard.shards[i].tail_buf);
    }
    free(lls->shard.shards);
  }
}

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

  // Per-level scratch for mask-scan results. Sized at K (max active_count).
  stage->pool_epochs_scratch = (uint32_t*)malloc((size_t)K * sizeof(uint32_t));
  CHECK(Fail, stage->pool_epochs_scratch);

  CHECK_MUL_OVERFLOW(Fail, M, stage->codec.max_output_size, SIZE_MAX);
  // Compressed buffers + events
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail,
       cuMemAlloc(&stage->d_compressed[fc], M * stage->codec.max_output_size));
    CU(Fail, cuEventCreate(&stage->t_compress_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->t_compress_end[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->t_aggregate_end[fc], CU_EVENT_DEFAULT));
  }

  // Per-level aggregate + shard + LUTs
  for (int lv = 0; lv < cl->levels.nlod; ++lv) {
    const struct level_layout_info* li = &cl->per_level[lv];

    stage->levels[lv].agg_layout = li->agg_layout;
    // GPU bias kernels pad intra-batch generation boundaries up to a page,
    // matching the CPU aggregator. Deliver consumes this flag to advance
    // bytes_consumed[si] over the agg-buffer pad after each finalizing run.
    stage->levels[lv].agg_layout.requires_gen_pads = 1;
    CHECK(Fail, aggregate_layout_upload(&stage->levels[lv].agg_layout) == 0);

    stage->levels[lv].batch_active_count = li->batch_active_count;

    uint32_t slot_count =
      li->batch_active_count > 0 ? li->batch_active_count : 1;
    uint64_t chunks_lv = cl->levels.level[lv].chunk_count;
    uint64_t batch_chunks = (uint64_t)slot_count * chunks_lv;
    uint64_t batch_covering =
      (uint64_t)slot_count * li->agg_layout.covering_count;
    size_t batch_agg_bytes = agg_pool_bytes_layout(&li->agg_layout);

    const uint64_t num_shards = li->agg_layout.num_shards;
    const uint64_t max_gens = li->agg_layout.max_gens_per_batch;

    for (int i = 0; i < 2; ++i) {
      CHECK(Fail,
            aggregate_batch_slot_init(&stage->levels[lv].agg[i],
                                      batch_chunks,
                                      batch_covering,
                                      num_shards,
                                      max_gens,
                                      batch_agg_bytes) == 0);
      CU(Fail, cuEventRecord(stage->levels[lv].agg[i].ready, compute));
    }

    // Per-LOD persistent tail-carry state (GPU-resident).
    if (num_shards > 0 && li->agg_layout.page_size > 0) {
      const size_t carry_bytes = num_shards * li->agg_layout.page_size;
      CU(Fail, cuMemAlloc(&stage->levels[lv].d_tail_carry, carry_bytes));
      CU(Fail, cuMemsetD8(stage->levels[lv].d_tail_carry, 0, carry_bytes));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&stage->levels[lv].d_tail_bytes,
                    num_shards * sizeof(size_t)));
      CU(Fail,
         cuMemsetD8((CUdeviceptr)stage->levels[lv].d_tail_bytes,
                    0,
                    num_shards * sizeof(size_t)));
      stage->levels[lv].h_tail_bytes =
        (size_t*)calloc(num_shards, sizeof(size_t));
      CHECK(Fail, stage->levels[lv].h_tail_bytes);
    }

    // Shard state
    struct shard_state* ss = &stage->levels[lv].shard;
    ss->chunks_per_shard_append = li->chunks_per_shard_append;
    ss->chunks_per_shard_inner = li->chunks_per_shard_inner;
    ss->chunks_per_shard_total = li->chunks_per_shard_total;
    ss->shard_inner_count = li->shard_inner_count;

    ss->shards = (struct active_shard*)calloc(ss->shard_inner_count,
                                              sizeof(struct active_shard));
    CHECK(Fail, ss->shards);

    size_t index_bytes = 2 * ss->chunks_per_shard_total * sizeof(uint64_t);
    const size_t page = li->agg_layout.page_size;
    for (uint64_t i = 0; i < ss->shard_inner_count; ++i) {
      ss->shards[i].index = (uint64_t*)malloc(index_bytes);
      CHECK(Fail, ss->shards[i].index);
      memset(ss->shards[i].index, 0xFF, index_bytes);
      if (page > 0) {
        ss->shards[i].tail_buf = (uint8_t*)malloc(page);
        CHECK(Fail, ss->shards[i].tail_buf);
      }
    }

    ss->epoch_in_shard = 0;
    ss->shard_epoch = 0;
  }

  // Batch LUTs (gather + perm, epoch-major shard order).
  for (int lv = 0; lv < cl->levels.nlod; ++lv) {
    struct level_flush_state* lvl = &stage->levels[lv];
    uint32_t batch_count = lvl->batch_active_count;
    uint32_t slot_count = batch_count > 0 ? batch_count : 1;
    uint64_t chunks_lv = cl->levels.level[lv].chunk_count;
    uint64_t lut_len = (uint64_t)slot_count * chunks_lv;

    if (lut_len == 0)
      continue;

    uint32_t* h_gather = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
    uint32_t* h_perm = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
    CHECK(Fail, h_gather && h_perm);

    {
      uint32_t* pool_epochs = stage->pool_epochs_scratch;
      for (uint32_t a = 0; a < slot_count; ++a) {
        uint32_t period = 1;
        if (cl->dims.append_downsample && lv > 0)
          period = 1u << lv;
        pool_epochs[a] = (a + 1) * period - 1;
      }
      aggregate_batch_luts(&lvl->agg_layout,
                           &cl->levels,
                           lv,
                           slot_count,
                           pool_epochs,
                           h_gather,
                           h_perm);
    }

    CU(LutFail, cuMemAlloc(&lvl->d_batch_gather, lut_len * sizeof(uint32_t)));
    CU(LutFail,
       cuMemcpyHtoD(lvl->d_batch_gather, h_gather, lut_len * sizeof(uint32_t)));
    CU(LutFail, cuMemAlloc(&lvl->d_batch_perm, lut_len * sizeof(uint32_t)));
    CU(LutFail,
       cuMemcpyHtoD(lvl->d_batch_perm, h_perm, lut_len * sizeof(uint32_t)));

    free(h_gather);
    free(h_perm);
    continue;

  LutFail:
    free(h_gather);
    free(h_perm);
    goto Fail;
  }

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
  stage->pool_epochs_scratch = NULL;
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(stage->d_compressed[fc]);
    cu_event_destroy(stage->t_compress_start[fc]);
    cu_event_destroy(stage->t_compress_end[fc]);
    cu_event_destroy(stage->t_aggregate_end[fc]);
  }
  for (int lv = 0; lv < nlod; ++lv)
    destroy_level_state(&stage->levels[lv]);
}

// --- Kick ---

int
compress_agg_kick(struct compress_agg_stage* stage,
                  const struct compress_agg_input* in,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  const struct dim_info* dims,
                  CUstream compress_stream,
                  struct flush_handoff* out)
{
  (void)batch;
  (void)dims;
  const int fc = in->fc;
  const uint32_t n_epochs = in->n_epochs;

  // Pool work is submitted on the compute stream in order; a single batch-level
  // pool_ready event recorded after the last scatter subsumes all prior ready
  // signals.
  CU(Error, cuStreamWaitEvent(compress_stream, in->pool_ready, 0));
  if (in->lod_done)
    CU(Error, cuStreamWaitEvent(compress_stream, in->lod_done, 0));

  // Aggregate writes to agg[fc].d_aggregated; ensure the prior D2H at this
  // fc has finished reading from it before we overwrite. prev_d2h_done is
  // initialized signaled, so the first kick on each fc no-ops here.
  if (in->prev_d2h_done)
    CU(Error, cuStreamWaitEvent(compress_stream, in->prev_d2h_done, 0));

  // Compress all epochs as one batch
  {
    CHECK_MUL_OVERFLOW(Error, n_epochs, levels->total_chunks, UINT64_MAX);
    uint64_t batch_chunks = (uint64_t)n_epochs * levels->total_chunks;
    size_t real_chunk_bytes = stage->codec.chunk_bytes;
    CHECK(Error,
          kick_compress(stage,
                        fc,
                        (void*)in->pool_buf,
                        batch_chunks,
                        real_chunk_bytes,
                        compress_stream) == 0);
  }

  // Zero per-level active counts; populated below and copied into handoff so
  // d2h delivery doesn't need to re-scan masks (the slot's masks buffer can
  // be reset by a pool swap before delivery runs).
  uint32_t active_counts[LOD_MAX_LEVELS] = { 0 };

  // Per-level batch aggregate on compress stream.
  // Always use the batch aggregate path (LUT-based) and derive pool positions
  // from per-epoch masks (the steady-state pattern shifts when K doesn't
  // divide the level period, so a pre-computed LUT is not always valid).
  for (int lv = 0; lv < levels->nlod; ++lv) {
    if (!(in->active_levels_mask & (1u << lv)))
      continue;

    struct level_flush_state* lvl = &stage->levels[lv];

    uint32_t* pool_epochs_buf = stage->pool_epochs_scratch;
    uint32_t active_count = 0;
    for (uint32_t e = 0; e < n_epochs; ++e)
      if (in->batch_active_masks[e] & (1u << lv))
        pool_epochs_buf[active_count++] = e;
    if (active_count == 0)
      continue;
    active_counts[lv] = active_count;

    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t chunks_lv = levels->level[lv].chunk_count;
    uint64_t batch_chunk_count = (uint64_t)active_count * chunks_lv;
    uint64_t batch_covering =
      (uint64_t)active_count * lvl->agg_layout.covering_count;

    // LUT buffers are sized for max(batch_active_count, 1) * chunks_lv.
    uint32_t lut_cap =
      lvl->batch_active_count > 0 ? lvl->batch_active_count : 1;
    CHECK(Error, active_count <= lut_cap);

    // Rebuild batch LUTs for this batch's actual pool positions and upload.
    {
      uint64_t lut_len = batch_chunk_count;
      uint32_t* h_gather = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
      uint32_t* h_perm = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
      CHECK(LutRecompute, h_gather && h_perm);

      aggregate_batch_luts(&lvl->agg_layout,
                           levels,
                           lv,
                           active_count,
                           pool_epochs_buf,
                           h_gather,
                           h_perm);

      CU(LutRecompute,
         cuMemcpyHtoDAsync(lvl->d_batch_gather,
                           h_gather,
                           lut_len * sizeof(uint32_t),
                           compress_stream));
      CU(LutRecompute,
         cuMemcpyHtoDAsync(lvl->d_batch_perm,
                           h_perm,
                           lut_len * sizeof(uint32_t),
                           compress_stream));

      free(h_gather);
      free(h_perm);
      goto LutRecomputeDone;

    LutRecompute:
      free(h_gather);
      free(h_perm);
      goto Error;
    LutRecomputeDone:;
    }

    CHECK(Error,
          aggregate_batch_by_shard_async(
            (void*)stage->d_compressed[fc],
            stage->codec.d_comp_sizes,
            (const uint32_t*)(uintptr_t)lvl->d_batch_gather,
            (const uint32_t*)(uintptr_t)lvl->d_batch_perm,
            batch_chunk_count,
            batch_covering,
            stage->codec.max_output_size,
            &lvl->agg_layout,
            agg,
            lvl->d_tail_bytes,
            lvl->d_tail_carry,
            lvl->shard.epoch_in_shard,
            compress_stream) == 0);
  }

  CU(Error, cuEventRecord(stage->t_aggregate_end[fc], compress_stream));

  // Fill handoff. active_counts is copied (not borrowed) so delivery can
  // run after a pool swap has overwritten the slot's mask buffer.
  out->fc = fc;
  out->n_epochs = n_epochs;
  out->active_levels_mask = in->active_levels_mask;
  out->batch_active_masks = in->batch_active_masks;
  out->t_aggregate_end = stage->t_aggregate_end[fc];
  out->t_compress_start = stage->t_compress_start[fc];
  out->t_compress_end = stage->t_compress_end[fc];
  out->max_output_size = stage->codec.max_output_size;

  for (int lv = 0; lv < levels->nlod; ++lv) {
    out->agg[lv] = &stage->levels[lv].agg[fc];
    out->agg_layout[lv] = &stage->levels[lv].agg_layout;
    out->active_counts[lv] = active_counts[lv];
  }

  return 0;

Error:
  return 1;
}

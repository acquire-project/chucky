#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/flush.helpers.h"
#include "gpu/stream.ingest.h"
#include "gpu/stream.lod.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "multiarray.gpu.h"
#include "platform/platform.h"
#include "stream/config.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

// ---- Per-array descriptor ----

struct array_descriptor_gpu
{
  struct tile_stream_configuration config;
  struct computed_stream_layouts cl;
  struct tile_stream_layout layout;
  struct tile_stream_layout_gpu layout_gpu; // L0 GPU pointers
  struct level_geometry levels;
  struct dim_info dims;
  struct shard_state shard[LOD_MAX_LEVELS];
  struct aggregate_layout agg_layout[LOD_MAX_LEVELS];
  uint32_t batch_active_count[LOD_MAX_LEVELS];
  struct shard_sink* sink;
  uint64_t cursor_elements;
  uint64_t max_cursor_elements;
  uint32_t batch_accumulated;
  struct flush_slot_gpu flush_slots[2];
  size_t shard_alignment;
};

// ---- Pool maxima (computed across all arrays) ----

struct pool_maxima
{
  size_t pool_bytes;         // max K * total_chunks * chunk_stride * bpe
  size_t buffer_capacity;    // max aligned buffer_capacity_bytes
  size_t compressed_bytes;   // max K * total_chunks * max_output_size
  uint64_t codec_batch;      // max K * total_chunks
  size_t chunk_bytes;        // max chunk_stride * bpe
  uint32_t epochs_per_batch; // max K
  int max_nlod;
  size_t max_output_size;

  // Per-level aggregate maxima
  struct
  {
    uint64_t batch_chunks;
    uint64_t batch_covering;
    size_t batch_agg_bytes;
    uint64_t lut_len;
  } level[LOD_MAX_LEVELS];
};

// ---- Main struct ----

struct multiarray_tile_stream_gpu
{
  struct multiarray_writer writer;
  int n_arrays;
  int active; // -1 = none

  struct array_descriptor_gpu* arrays;

  // Shared GPU resources
  struct gpu_streams streams;
  struct pool_state pools;
  size_t pool_bytes;
  struct staging_state stage;
  struct batch_state batch;
  struct flush_pipeline flush;

  // Shared compress+aggregate stage (shard state swapped on array switch)
  struct compress_agg_stage compress_agg;
  struct d2h_deliver_stage d2h_deliver;

  // LOD (shared buffers)
  struct lod_state lod;

  int max_nlod;
  int metrics_enabled;
  struct stream_metrics metrics;
  struct platform_clock metadata_update_clock;
};

// ---- Forward declarations ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data);
static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self);

// ---- Helpers ----

static inline size_t
max_sz(size_t a, size_t b)
{
  return a > b ? a : b;
}

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
  return a > b ? a : b;
}

static inline uint32_t
max_u32(uint32_t a, uint32_t b)
{
  return a > b ? a : b;
}

static inline struct multiarray_writer_result
mw_fail_at(const void* beg, const void* end)
{
  return (struct multiarray_writer_result){
    .error = multiarray_writer_fail,
    .rest = { .beg = beg, .end = end },
  };
}

// ---- Per-array init ----

static int
init_array_descriptor(struct array_descriptor_gpu* desc,
                      const struct tile_stream_configuration* config,
                      struct shard_sink* sink,
                      struct pool_maxima* mx)
{
  if (!codec_is_gpu_supported(config->codec.id))
    return 1;

  desc->config = *config;
  desc->sink = sink;
  desc->shard_alignment = shard_sink_required_shard_alignment(sink);

  if (compute_stream_layouts(config,
                             codec_alignment(config->codec.id),
                             codec_max_output_size,
                             desc->shard_alignment,
                             &desc->cl))
    return 1;

  desc->layout = desc->cl.layouts[0];
  desc->levels = desc->cl.levels;
  desc->dims = desc->cl.dims;

  const uint32_t K = desc->cl.epochs_per_batch;
  const size_t bpe = dtype_bpe(config->dtype);
  const uint64_t total_chunks = desc->levels.total_chunks;
  const uint64_t chunk_stride = desc->layout.chunk_stride;

  // max_cursor
  {
    const struct dimension* dims = config->dimensions;
    const uint8_t na = dim_info_n_append(&desc->dims);
    if (dims[0].size > 0) {
      desc->max_cursor_elements = desc->layout.epoch_elements;
      for (int d = 0; d < na; ++d)
        desc->max_cursor_elements *= ceildiv(dims[d].size, dims[d].chunk_size);
    }
  }

  // Buffer capacity (page-aligned like GPU stream)
  size_t aligned_cap = (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // Update pool maxima
  mx->pool_bytes =
    max_sz(mx->pool_bytes, (size_t)K * total_chunks * chunk_stride * bpe);
  mx->buffer_capacity = max_sz(mx->buffer_capacity, aligned_cap);
  mx->compressed_bytes = max_sz(
    mx->compressed_bytes, (size_t)K * total_chunks * desc->cl.max_output_size);
  mx->codec_batch = max_u64(mx->codec_batch, (uint64_t)K * total_chunks);
  mx->chunk_bytes = max_sz(mx->chunk_bytes, chunk_stride * bpe);
  mx->epochs_per_batch = max_u32(mx->epochs_per_batch, K);
  mx->max_output_size = max_sz(mx->max_output_size, desc->cl.max_output_size);

  if (desc->levels.nlod > mx->max_nlod)
    mx->max_nlod = desc->levels.nlod;

  // Per-level aggregate + shard state
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    const struct level_layout_info* li = &desc->cl.per_level[lv];
    desc->agg_layout[lv] = li->agg_layout;
    desc->batch_active_count[lv] = li->batch_active_count;

    uint32_t slot_count =
      li->batch_active_count > 0 ? li->batch_active_count : 1;
    uint64_t chunks_lv = desc->levels.level[lv].chunk_count;
    uint64_t batch_chunks = (uint64_t)slot_count * chunks_lv;
    uint64_t batch_covering =
      (uint64_t)slot_count * li->agg_layout.covering_count;
    size_t batch_agg_bytes = agg_pool_bytes(batch_chunks,
                                            desc->cl.max_output_size,
                                            li->agg_layout.covering_count,
                                            li->agg_layout.cps_inner,
                                            li->agg_layout.page_size);

    mx->level[lv].batch_chunks =
      max_u64(mx->level[lv].batch_chunks, batch_chunks);
    mx->level[lv].batch_covering =
      max_u64(mx->level[lv].batch_covering, batch_covering);
    mx->level[lv].batch_agg_bytes =
      max_sz(mx->level[lv].batch_agg_bytes, batch_agg_bytes);
    mx->level[lv].lut_len = max_u64(mx->level[lv].lut_len, batch_chunks);

    // Initialize per-array shard state
    if (init_shard_state(&desc->shard[lv], li))
      return 1;
  }

  return 0;
}

static void
destroy_array_descriptor(struct array_descriptor_gpu* desc)
{
  if (!desc)
    return;
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    struct shard_state* ss = &desc->shard[lv];
    if (ss->shards) {
      for (uint64_t i = 0; i < ss->shard_inner_count; ++i)
        free(ss->shards[i].index);
      free(ss->shards);
    }
    aggregate_layout_destroy(&desc->agg_layout[lv]);
  }
  cu_mem_free((CUdeviceptr)desc->layout_gpu.d_lifted_shape);
  cu_mem_free((CUdeviceptr)desc->layout_gpu.d_lifted_strides);
  computed_stream_layouts_free(&desc->cl);
}

// ---- Shared resource allocation ----

static int
init_shared_resources(struct multiarray_tile_stream_gpu* ms,
                      const struct pool_maxima* mx)
{
  // CUDA streams
  CU(Fail, cuStreamCreate(&ms->streams.h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&ms->streams.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&ms->streams.compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&ms->streams.d2h, CU_STREAM_NON_BLOCKING));

  // Pool events
  for (int i = 0; i < 2; ++i)
    CU(Fail, cuEventCreate(&ms->pools.ready[i], CU_EVENT_DEFAULT));

  // Staging buffers
  CHECK(Fail,
        ingest_init(&ms->stage, mx->buffer_capacity, ms->streams.compute) == 0);

  // Chunk pools (double-buffered)
  ms->pool_bytes = mx->pool_bytes;
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&ms->pools.buf[i], mx->pool_bytes));
    CU(Fail,
       cuMemsetD8Async(
         ms->pools.buf[i], 0, mx->pool_bytes, ms->streams.compute));
  }

  // Batch events
  ms->batch.epochs_per_batch = mx->epochs_per_batch;
  for (uint32_t i = 0; i < mx->epochs_per_batch; ++i) {
    CU(Fail, cuEventCreate(&ms->batch.pool_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(ms->batch.pool_events[i], ms->streams.compute));
  }

  // Codec
  CHECK(Fail,
        codec_init(&ms->compress_agg.codec,
                   ms->arrays[0].config.codec.id,
                   mx->chunk_bytes,
                   mx->codec_batch) == 0);

  // Compressed buffers + timing events
  for (int fc = 0; fc < 2; ++fc) {
    size_t comp_sz = mx->codec_batch * ms->compress_agg.codec.max_output_size;
    if (comp_sz > 0)
      CU(Fail, cuMemAlloc(&ms->compress_agg.d_compressed[fc], comp_sz));
    CU(Fail,
       cuEventCreate(&ms->compress_agg.t_compress_start[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&ms->compress_agg.t_compress_end[fc], CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&ms->compress_agg.t_aggregate_end[fc], CU_EVENT_DEFAULT));
  }

  // Per-level aggregate slots and LUT buffers (shared, max-sized)
  for (int lv = 0; lv < ms->max_nlod; ++lv) {
    struct level_flush_state* lvl = &ms->compress_agg.levels[lv];

    for (int fc = 0; fc < 2; ++fc) {
      if (mx->level[lv].batch_covering > 0) {
        CHECK(Fail,
              aggregate_batch_slot_init(&lvl->agg[fc],
                                        mx->level[lv].batch_chunks,
                                        mx->level[lv].batch_covering,
                                        mx->level[lv].batch_agg_bytes) == 0);
        CU(Fail, cuEventRecord(lvl->agg[fc].ready, ms->streams.compute));
      }
    }

    if (mx->level[lv].lut_len > 0) {
      size_t lut_bytes = mx->level[lv].lut_len * sizeof(uint32_t);
      CU(Fail, cuMemAlloc(&lvl->d_batch_gather, lut_bytes));
      CU(Fail, cuMemAlloc(&lvl->d_batch_perm, lut_bytes));
    }
  }

  // D2H delivery stage
  CHECK(Fail,
        d2h_deliver_init(&ms->d2h_deliver,
                         ms->compress_agg.levels,
                         ms->max_nlod,
                         0, // shard_alignment set per-array on switch
                         ms->streams.compute) == 0);

  // Seed pool events
  CU(Fail, cuEventRecord(ms->pools.ready[0], ms->streams.compute));
  CU(Fail, cuEventRecord(ms->pools.ready[1], ms->streams.compute));

  // Seed timing events
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail,
       cuEventRecord(ms->compress_agg.t_compress_start[fc],
                     ms->streams.compute));
    CU(Fail,
       cuEventRecord(ms->compress_agg.t_compress_end[fc], ms->streams.compute));
    CU(
      Fail,
      cuEventRecord(ms->compress_agg.t_aggregate_end[fc], ms->streams.compute));
  }

  CU(Fail, cuStreamSynchronize(ms->streams.compute));

  return 0;

Fail:
  return 1;
}

// ---- Array switching ----

// Save the active array's mutable state from shared resources back into its
// descriptor.
static void
save_active_state(struct multiarray_tile_stream_gpu* ms)
{
  if (ms->active < 0)
    return;
  struct array_descriptor_gpu* desc = &ms->arrays[ms->active];
  for (int lv = 0; lv < desc->levels.nlod; ++lv)
    desc->shard[lv] = ms->compress_agg.levels[lv].shard;
}

// Load an array's per-array state into the shared resources.
static void
load_array_state(struct multiarray_tile_stream_gpu* ms, int array_index)
{
  struct array_descriptor_gpu* desc = &ms->arrays[array_index];
  for (int lv = 0; lv < ms->max_nlod; ++lv) {
    struct level_flush_state* lvl = &ms->compress_agg.levels[lv];
    if (lv < desc->levels.nlod) {
      lvl->shard = desc->shard[lv];
      lvl->agg_layout = desc->agg_layout[lv];
      lvl->batch_active_count = desc->batch_active_count[lv];
    } else {
      memset(&lvl->shard, 0, sizeof(lvl->shard));
      lvl->batch_active_count = 0;
    }
  }
  ms->d2h_deliver.nlod = desc->levels.nlod;
  ms->d2h_deliver.shard_alignment = desc->shard_alignment;
}

// Flush departing array's pending batch, then prepare for incoming array.
static int
switch_to_array(struct multiarray_tile_stream_gpu* ms, int array_index)
{
  if (ms->active >= 0) {
    struct array_descriptor_gpu* departing = &ms->arrays[ms->active];

    // Reject switch mid-epoch
    if (departing->cursor_elements % departing->layout.epoch_elements != 0)
      return multiarray_writer_not_flushable;

    // Flush departing batch
    if (departing->batch_accumulated > 0) {
      int fc = ms->pools.current;
      struct flush_slot_gpu* fs = &departing->flush_slots[fc];
      fs->batch_epoch_count = (int)departing->batch_accumulated;

      struct compress_agg_input in = {
        .fc = fc,
        .n_epochs = departing->batch_accumulated,
        .active_levels_mask = fs->active_levels_mask,
        .epochs_per_batch = ms->batch.epochs_per_batch,
        .pool_buf = ms->pools.buf[fc],
      };
      memcpy(in.batch_active_masks,
             fs->batch_active_masks,
             departing->batch_accumulated * sizeof(uint32_t));
      memcpy(in.epoch_events,
             ms->batch.pool_events,
             departing->batch_accumulated * sizeof(CUevent));

      struct flush_handoff handoff = { 0 };
      CHECK(Fail,
            compress_agg_kick(&ms->compress_agg,
                              &in,
                              &departing->levels,
                              &ms->batch,
                              &departing->dims,
                              ms->streams.compress,
                              &handoff) == 0);
      CHECK(Fail,
            d2h_deliver_kick(&ms->d2h_deliver,
                             &handoff,
                             &departing->levels,
                             &ms->batch,
                             &departing->dims,
                             ms->streams.d2h) == 0);

      struct writer_result r = d2h_deliver_drain(&ms->d2h_deliver,
                                                 &handoff,
                                                 &departing->levels,
                                                 &ms->batch,
                                                 &departing->dims,
                                                 &departing->layout,
                                                 &departing->config,
                                                 departing->sink,
                                                 &ms->lod,
                                                 &ms->metrics,
                                                 &ms->metadata_update_clock);
      CHECK(Fail, r.error == 0);

      departing->batch_accumulated = 0;
      fs->active_levels_mask = 0;
      memset(fs->batch_active_masks, 0, sizeof(fs->batch_active_masks));
    }

    save_active_state(ms);
  }

  ms->active = array_index;
  load_array_state(ms, array_index);

  // Zero the active pool for the incoming array
  CU(Fail,
     cuMemsetD8Async(ms->pools.buf[ms->pools.current],
                     0,
                     ms->pool_bytes,
                     ms->streams.compute));

  return 0;

Fail:
  return multiarray_writer_fail;
}

// ---- Ingest dispatch (adapted from gpu/stream.c) ----

static inline void*
pool_epoch_ptr(struct multiarray_tile_stream_gpu* ms,
               const struct array_descriptor_gpu* desc,
               uint32_t epoch_in_batch)
{
  const size_t bpe = dtype_bpe(desc->config.dtype);
  return (char*)ms->pools.buf[ms->pools.current] +
         (uint64_t)epoch_in_batch * desc->levels.total_chunks *
           desc->layout.chunk_stride * bpe;
}

static int
dispatch_ingest(struct multiarray_tile_stream_gpu* ms,
                struct array_descriptor_gpu* desc)
{
  if (desc->levels.enable_multiscale) {
    return ingest_dispatch_multiscale(&ms->stage,
                                      ms->lod.d_linear,
                                      desc->layout.epoch_elements,
                                      &desc->cursor_elements,
                                      dtype_bpe(desc->config.dtype),
                                      ms->streams.h2d,
                                      ms->streams.compute);
  } else {
    return ingest_dispatch_scatter(
      &ms->stage,
      &desc->layout,
      &desc->layout_gpu,
      pool_epoch_ptr(ms, desc, desc->batch_accumulated),
      ms->pools.ready[ms->pools.current],
      &desc->cursor_elements,
      dtype_bpe(desc->config.dtype),
      ms->streams.h2d,
      ms->streams.compute);
  }
}

// ---- Epoch flush helpers ----

static int
run_epoch_lod(struct multiarray_tile_stream_gpu* ms,
              struct array_descriptor_gpu* desc,
              uint32_t* out_active_mask)
{
  if (!desc->levels.enable_multiscale || !ms->lod.d_linear) {
    *out_active_mask = 1; // L0 only
    return 0;
  }
  return lod_run_epoch(&ms->lod,
                       ms->pools.current,
                       &desc->levels,
                       pool_epoch_ptr(ms, desc, desc->batch_accumulated),
                       desc->config.dtype,
                       desc->config.reduce_method,
                       desc->config.append_reduce_method,
                       &desc->dims,
                       ms->streams.compute,
                       out_active_mask);
}

// Flush a completed batch through the pipeline: compress → aggregate → D2H →
// deliver.  Synchronous: blocks until delivery completes.
static int
flush_batch_sync(struct multiarray_tile_stream_gpu* ms,
                 struct array_descriptor_gpu* desc)
{
  int fc = ms->pools.current;
  struct flush_slot_gpu* fs = &desc->flush_slots[fc];

  fs->batch_epoch_count = (int)desc->batch_accumulated;

  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = desc->batch_accumulated,
    .active_levels_mask = fs->active_levels_mask,
    .epochs_per_batch = ms->batch.epochs_per_batch,
    .pool_buf = ms->pools.buf[fc],
    .lod_done = (desc->levels.enable_multiscale && ms->lod.timing[fc].t_end)
                  ? ms->lod.timing[fc].t_end
                  : NULL,
  };
  memcpy(in.batch_active_masks,
         fs->batch_active_masks,
         desc->batch_accumulated * sizeof(uint32_t));
  memcpy(in.epoch_events,
         ms->batch.pool_events,
         desc->batch_accumulated * sizeof(CUevent));

  struct flush_handoff handoff = { 0 };
  CHECK(Fail,
        compress_agg_kick(&ms->compress_agg,
                          &in,
                          &desc->levels,
                          &ms->batch,
                          &desc->dims,
                          ms->streams.compress,
                          &handoff) == 0);

  CHECK(Fail,
        d2h_deliver_kick(&ms->d2h_deliver,
                         &handoff,
                         &desc->levels,
                         &ms->batch,
                         &desc->dims,
                         ms->streams.d2h) == 0);

  struct writer_result r = d2h_deliver_drain(&ms->d2h_deliver,
                                             &handoff,
                                             &desc->levels,
                                             &ms->batch,
                                             &desc->dims,
                                             &desc->layout,
                                             &desc->config,
                                             desc->sink,
                                             &ms->lod,
                                             &ms->metrics,
                                             &ms->metadata_update_clock);
  CHECK(Fail, r.error == 0);

  desc->batch_accumulated = 0;
  fs->active_levels_mask = 0;
  memset(fs->batch_active_masks, 0, sizeof(fs->batch_active_masks));

  // Zero pool for next batch
  size_t bpe = dtype_bpe(desc->config.dtype);
  size_t batch_pool_bytes = (uint64_t)desc->cl.epochs_per_batch *
                            desc->levels.total_chunks *
                            desc->layout.chunk_stride * bpe;
  CU(Fail,
     cuMemsetD8Async(ms->pools.buf[ms->pools.current],
                     0,
                     batch_pool_bytes,
                     ms->streams.compute));

  return 0;

Fail:
  return 1;
}

// ---- Writer: update ----

static struct multiarray_writer_result
update_impl(struct multiarray_writer* self, int array_index, struct slice data)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);

  if (array_index < 0 || array_index >= ms->n_arrays)
    return mw_fail_at(data.beg, data.end);

  struct array_descriptor_gpu* desc = &ms->arrays[array_index];

  // Switch arrays if needed
  if (array_index != ms->active) {
    int err = switch_to_array(ms, array_index);
    if (err)
      return (struct multiarray_writer_result){ .error = err, .rest = data };
  }

  const size_t bpe = dtype_bpe(desc->config.dtype);
  const size_t buf_cap =
    (desc->config.buffer_capacity_bytes + 4095) & ~(size_t)4095;
  const uint8_t* src = (const uint8_t*)data.beg;
  const uint8_t* end = (const uint8_t*)data.end;
  const uint64_t max_cursor = desc->max_cursor_elements;

  while (src < end) {
    // Check capacity
    if (max_cursor > 0 && desc->cursor_elements >= max_cursor) {
      if (desc->batch_accumulated > 0) {
        if (flush_batch_sync(ms, desc))
          return mw_fail_at(src, end);
      }
      return (struct multiarray_writer_result){
        .error = multiarray_writer_finished,
        .rest = { .beg = src, .end = end },
      };
    }

    // Elements to process
    const uint64_t epoch_remaining =
      desc->layout.epoch_elements -
      (desc->cursor_elements % desc->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    uint64_t elements =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    if (max_cursor > 0) {
      uint64_t cap = max_cursor - desc->cursor_elements;
      if (elements > cap)
        elements = cap;
    }
    const uint64_t bytes = elements * bpe;

    // Fill staging buffer, dispatch when full
    {
      uint64_t written = 0;
      while (written < bytes) {
        const size_t space = buf_cap - ms->stage.bytes_written;
        const uint64_t remaining = bytes - written;
        const size_t payload = space < remaining ? space : (size_t)remaining;

        if (ms->stage.bytes_written == 0) {
          const int si = ms->stage.current;
          struct staging_slot* ss = &ms->stage.slot[si];
          CU(Error, cuEventSynchronize(ss->t_h2d_end));

          if (desc->cursor_elements > 0 && ms->metrics_enabled) {
            accumulate_metric_cu(&ms->metrics.h2d,
                                 ss->t_h2d_start,
                                 ss->t_h2d_end,
                                 ss->dispatched_bytes,
                                 ss->dispatched_bytes);
            accumulate_metric_cu(&ms->metrics.scatter,
                                 ss->t_scatter_start,
                                 ss->t_scatter_end,
                                 ss->dispatched_bytes,
                                 ss->dispatched_bytes);
          }
        }

        {
          struct platform_clock mc = { 0 };
          platform_toc(&mc);
          memcpy((uint8_t*)ms->stage.slot[ms->stage.current].h_in +
                   ms->stage.bytes_written,
                 src + written,
                 payload);
          if (ms->metrics_enabled)
            accumulate_metric_ms(&ms->metrics.memcpy,
                                 (float)(platform_toc(&mc) * 1000.0),
                                 payload,
                                 payload);
        }
        ms->stage.bytes_written += payload;
        written += payload;

        if (ms->stage.bytes_written == buf_cap || written == bytes) {
          CHECK(Error, dispatch_ingest(ms, desc) == 0);
          ms->stage.bytes_written = 0;
        }
      }
    }
    src += bytes;

    // Epoch boundary
    if (desc->cursor_elements % desc->layout.epoch_elements == 0 &&
        desc->cursor_elements > 0) {
      uint32_t active_mask = 1;
      if (run_epoch_lod(ms, desc, &active_mask))
        return mw_fail_at(src, end);

      CU(Error,
         cuEventRecord(ms->batch.pool_events[desc->batch_accumulated],
                       ms->streams.compute));

      int fc = ms->pools.current;
      struct flush_slot_gpu* fs = &desc->flush_slots[fc];
      fs->batch_active_masks[desc->batch_accumulated] = active_mask;
      fs->active_levels_mask |= active_mask;
      desc->batch_accumulated++;

      // Flush batch if full
      if (desc->batch_accumulated >= desc->cl.epochs_per_batch) {
        if (flush_batch_sync(ms, desc))
          return mw_fail_at(src, end);
      }
    }
  }

  return (struct multiarray_writer_result){
    .error = multiarray_writer_ok,
    .rest = { .beg = src, .end = end },
  };

Error:
  return mw_fail_at(src, end);
}

// ---- Writer: flush ----

static int
flush_partial_epoch(struct multiarray_tile_stream_gpu* ms,
                    struct array_descriptor_gpu* desc)
{
  if (desc->cursor_elements % desc->layout.epoch_elements == 0)
    return 0;

  // Dispatch remaining staging data
  if (ms->stage.bytes_written > 0) {
    if (dispatch_ingest(ms, desc))
      return 1;
    ms->stage.bytes_written = 0;
  }

  uint32_t active_mask = 1;
  if (run_epoch_lod(ms, desc, &active_mask))
    return 1;

  CU(Fail,
     cuEventRecord(ms->batch.pool_events[desc->batch_accumulated],
                   ms->streams.compute));

  int fc = ms->pools.current;
  struct flush_slot_gpu* fs = &desc->flush_slots[fc];
  fs->batch_active_masks[desc->batch_accumulated] = active_mask;
  fs->active_levels_mask |= active_mask;
  desc->batch_accumulated++;

  return 0;

Fail:
  return 1;
}

static int
finalize_array(struct multiarray_tile_stream_gpu* ms,
               struct array_descriptor_gpu* desc)
{
  // Save previous array's state, then load this one
  int array_index = (int)(desc - ms->arrays);
  save_active_state(ms);
  ms->active = array_index;
  load_array_state(ms, array_index);

  // Flush partial epoch
  if (flush_partial_epoch(ms, desc))
    return 1;

  // Flush remaining batch
  if (desc->batch_accumulated > 0) {
    if (flush_batch_sync(ms, desc))
      return 1;
  }

  // Finalize partial shards
  for (int lv = 0; lv < desc->levels.nlod; ++lv) {
    struct shard_state* ss = &ms->compress_agg.levels[lv].shard;
    if (ss->epoch_in_shard > 0) {
      if (finalize_shards(ss, desc->shard_alignment))
        return 1;
    }
  }

  // Final metadata update
  if (desc->sink->update_append) {
    const uint8_t na = dim_info_n_append(&desc->dims);
    for (int lv = 0; lv < desc->levels.nlod; ++lv) {
      uint64_t append_sizes[HALF_MAX_RANK];
      dim_info_final_append_sizes(
        &desc->dims, desc->cursor_elements, lv, append_sizes);
      if (desc->sink->update_append(desc->sink, (uint8_t)lv, na, append_sizes))
        return 1;
    }
  }

  // Save back shard state
  save_active_state(ms);

  return 0;
}

static struct multiarray_writer_result
flush_impl(struct multiarray_writer* self)
{
  struct multiarray_tile_stream_gpu* ms =
    container_of(self, struct multiarray_tile_stream_gpu, writer);

  // Flush remaining staging for active array
  if (ms->active >= 0 && ms->stage.bytes_written > 0) {
    struct array_descriptor_gpu* desc = &ms->arrays[ms->active];
    if (dispatch_ingest(ms, desc))
      goto Error;
    ms->stage.bytes_written = 0;
  }

  // Finalize all arrays
  for (int a = 0; a < ms->n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    if (desc->cursor_elements == 0 && desc->batch_accumulated == 0)
      continue;
    if (finalize_array(ms, desc))
      goto Error;
  }

  return (struct multiarray_writer_result){
    .error = multiarray_writer_ok,
  };

Error:
  return (struct multiarray_writer_result){
    .error = multiarray_writer_fail,
  };
}

// ---- Create / Destroy ----

static void
sync_all(struct gpu_streams* streams)
{
  if (streams->h2d)
    cuStreamSynchronize(streams->h2d);
  if (streams->compute)
    cuStreamSynchronize(streams->compute);
  if (streams->compress)
    cuStreamSynchronize(streams->compress);
  if (streams->d2h)
    cuStreamSynchronize(streams->d2h);
}

void
multiarray_tile_stream_gpu_destroy(struct multiarray_tile_stream_gpu* ms)
{
  if (!ms)
    return;

  sync_all(&ms->streams);

  // Per-array descriptors
  if (ms->arrays) {
    for (int a = 0; a < ms->n_arrays; ++a)
      destroy_array_descriptor(&ms->arrays[a]);
    free(ms->arrays);
  }

  // Batch events
  for (uint32_t i = 0; i < ms->batch.epochs_per_batch; ++i)
    cu_event_destroy(ms->batch.pool_events[i]);

  // D2H delivery
  d2h_deliver_destroy(&ms->d2h_deliver);

  // Compress/agg (shared parts only — shard state owned by descriptors)
  codec_free(&ms->compress_agg.codec);
  for (int fc = 0; fc < 2; ++fc) {
    cu_mem_free(ms->compress_agg.d_compressed[fc]);
    cu_event_destroy(ms->compress_agg.t_compress_start[fc]);
    cu_event_destroy(ms->compress_agg.t_compress_end[fc]);
    cu_event_destroy(ms->compress_agg.t_aggregate_end[fc]);
  }
  for (int lv = 0; lv < ms->max_nlod; ++lv) {
    struct level_flush_state* lvl = &ms->compress_agg.levels[lv];
    for (int fc = 0; fc < 2; ++fc)
      aggregate_slot_destroy(&lvl->agg[fc]);
    cu_mem_free(lvl->d_batch_gather);
    cu_mem_free(lvl->d_batch_perm);
    // Note: shard state is owned by per-array descriptors, not here
  }

  // LOD
  lod_state_destroy(&ms->lod);

  // Pools
  for (int i = 0; i < 2; ++i) {
    cu_mem_free(ms->pools.buf[i]);
    cu_event_destroy(ms->pools.ready[i]);
  }

  // Staging
  ingest_destroy(&ms->stage);

  // CUDA streams
  cu_stream_destroy(ms->streams.h2d);
  cu_stream_destroy(ms->streams.compute);
  cu_stream_destroy(ms->streams.compress);
  cu_stream_destroy(ms->streams.d2h);

  free(ms);
}

static struct stream_metrics
init_metrics(void)
{
  return (struct stream_metrics){
    .memcpy = mk_stream_metric("Memcpy"),
    .h2d = mk_stream_metric("H2D"),
    .scatter = mk_stream_metric("Scatter"),
    .lod_gather = mk_stream_metric("LOD Gather"),
    .lod_reduce = mk_stream_metric("LOD Reduce"),
    .lod_append_fold = mk_stream_metric("Append Fold"),
    .lod_morton_chunk = mk_stream_metric("LOD to chunks"),
    .compress = mk_stream_metric("Compress"),
    .aggregate = mk_stream_metric("Aggregate"),
    .d2h = mk_stream_metric("D2H"),
    .sink = mk_stream_metric("Sink"),
    .flush_stall = mk_stream_metric("FlushStall"),
    .kick_sync_stall = mk_stream_metric("KickSync"),
    .io_fence_stall = mk_stream_metric("IOFence"),
    .backpressure = mk_stream_metric("Backpres"),
  };
}

struct multiarray_tile_stream_gpu*
multiarray_tile_stream_gpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[],
  int enable_metrics)
{
  if (n_arrays <= 0)
    return NULL;

  struct multiarray_tile_stream_gpu* ms =
    (struct multiarray_tile_stream_gpu*)calloc(1, sizeof(*ms));
  if (!ms)
    return NULL;

  ms->n_arrays = n_arrays;
  ms->active = -1;
  ms->metrics_enabled = enable_metrics;
  ms->writer.update = update_impl;
  ms->writer.flush = flush_impl;

  ms->arrays = (struct array_descriptor_gpu*)calloc(
    n_arrays, sizeof(struct array_descriptor_gpu));
  CHECK(Fail, ms->arrays);

  // Phase 1: compute per-array layouts and pool maxima
  struct pool_maxima mx;
  memset(&mx, 0, sizeof(mx));

  for (int a = 0; a < n_arrays; ++a) {
    CHECK(Fail,
          init_array_descriptor(&ms->arrays[a], &configs[a], sinks[a], &mx) ==
            0);
  }

  ms->max_nlod = mx.max_nlod;

  // Phase 2: upload per-array aggregate layouts to GPU
  for (int a = 0; a < n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    for (int lv = 0; lv < desc->levels.nlod; ++lv)
      CHECK(Fail, aggregate_layout_upload(&desc->agg_layout[lv]) == 0);
  }

  // Phase 3: upload per-array L0 layout to GPU (d_lifted_shape,
  // d_lifted_strides)
  for (int a = 0; a < n_arrays; ++a) {
    struct array_descriptor_gpu* desc = &ms->arrays[a];
    // Initialize LOD state for layout GPU upload (always needed for L0 scatter)
    // We reuse lod_state_init's layout_gpu upload for each array.
    // For now, do a minimal upload of the L0 layout.
    {
      uint64_t lr = desc->layout.lifted_rank;
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&desc->layout_gpu.d_lifted_shape,
                    lr * sizeof(uint64_t)));
      CU(Fail,
         cuMemcpyHtoD((CUdeviceptr)desc->layout_gpu.d_lifted_shape,
                      desc->layout.lifted_shape,
                      lr * sizeof(uint64_t)));
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&desc->layout_gpu.d_lifted_strides,
                    lr * sizeof(int64_t)));
      CU(Fail,
         cuMemcpyHtoD((CUdeviceptr)desc->layout_gpu.d_lifted_strides,
                      desc->layout.lifted_strides,
                      lr * sizeof(int64_t)));
    }
  }

  // Phase 4: allocate shared GPU resources
  CHECK(Fail, init_shared_resources(ms, &mx) == 0);

  ms->metrics = init_metrics();

  ms->metadata_update_clock = (struct platform_clock){ 0 };
  platform_toc(&ms->metadata_update_clock);

  return ms;

Fail:
  multiarray_tile_stream_gpu_destroy(ms);
  return NULL;
}

// ---- Accessors ----

struct multiarray_writer*
multiarray_tile_stream_gpu_writer(struct multiarray_tile_stream_gpu* ms)
{
  return &ms->writer;
}

struct stream_metrics
multiarray_tile_stream_gpu_get_metrics(
  const struct multiarray_tile_stream_gpu* ms)
{
  return ms->metrics;
}

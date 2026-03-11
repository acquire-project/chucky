#include "compress.h"
#include "lod.h"
#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"

#include <stdlib.h>
#include <string.h>

static inline uint32_t
next_pow2_u32(uint32_t v)
{
  if (v == 0)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

static struct writer_result
writer_ok(void)
{
  return (struct writer_result){ 0 };
}

static struct writer_result
writer_error(void)
{
  return (struct writer_result){ .error = 1 };
}

static struct writer_result
writer_error_at(const void* beg, const void* end)
{
  return (struct writer_result){ .error = 1, .rest = { beg, end } };
}

static void
buffer_free(struct buffer* buffer)
{
  if (!buffer || !buffer->data) {
    return;
  }

  if (buffer->ready) {
    CUresult res = cuEventDestroy(buffer->ready);
    if (res != CUDA_SUCCESS) {
      const char* err_str = NULL;
      cuGetErrorString(res, &err_str);
      log_warn("Failed to destroy event: %s", err_str ? err_str : "unknown");
    }
    buffer->ready = NULL;
  }

  switch (buffer->domain) {
    case host:
      cuMemFreeHost(buffer->data);
      break;
    case device:
      cuMemFree((CUdeviceptr)buffer->data);
      break;
    default:
      log_error("Invalid domain during buffer_free: %d", buffer->domain);
      return;
  }

  buffer->data = NULL;
}

static struct buffer
buffer_new(size_t capacity, enum domain domain, unsigned int host_flags)
{
  struct buffer buf = { 0 };
  buf.domain = domain;

  switch (domain) {
    case host:
      CU(Fail, cuMemHostAlloc(&buf.data, capacity, host_flags));
      break;
    case device:
      CU(Fail, cuMemAlloc((CUdeviceptr*)&buf.data, capacity));
      break;
    default:
      log_error("Invalid domain: %d", domain);
      goto Fail;
  }
  CU(Fail, cuEventCreate(&buf.ready, CU_EVENT_DEFAULT));
  return buf;

Fail:
  buffer_free(&buf);
  return (struct buffer){ 0 };
}

// --- Helpers ---

static inline void*
dbuf_current(struct double_buffer* db)
{
  return db->buf[db->current].data;
}

static inline void
dbuf_swap(struct double_buffer* db)
{
  db->current ^= 1;
}

// Return pointer to the current pool buffer (entire K-epoch super-pool).
static inline void*
current_pool_base(struct tile_stream_gpu* s)
{
  return dbuf_current(&s->pools);
}

// Return pointer to the current epoch's tile region within the super-pool.
// epoch_in_batch: 0..K-1
static inline void*
current_pool_epoch(struct tile_stream_gpu* s, uint32_t epoch_in_batch)
{
  const size_t bpe = s->config.bytes_per_element;
  return (char*)dbuf_current(&s->pools) +
         (uint64_t)epoch_in_batch * s->total_tiles * s->layout.tile_stride * bpe;
}

// H2D transfer + scatter into tile pool.
// Returns 0 on success, 1 on error.
static int
dispatch_scatter(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 1;

  const uint64_t elements = s->stage.bytes_written / bpe;
  if (elements == 0)
    return 1;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];
  void* pool = current_pool_epoch(s, s->epochs_accumulated);

  ss->dispatched_bytes = s->stage.bytes_written;

  // H2D — wait for prior scatter to finish reading d_in before overwriting
  CU(Error, cuStreamWaitEvent(s->h2d, ss->t_scatter_end, 0));
  CU(Error, cuEventRecord(ss->t_h2d_start, s->h2d));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)ss->d_in.data,
                       ss->h_in.data,
                       s->stage.bytes_written,
                       s->h2d));
  CU(Error, cuEventRecord(ss->h_in.ready, s->h2d));

  // Scatter into tile pool
  CU(Error, cuStreamWaitEvent(s->compute, ss->h_in.ready, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, s->compute));
  transpose((CUdeviceptr)pool,
            (CUdeviceptr)ss->d_in.data,
            s->stage.bytes_written,
            (uint8_t)bpe,
            s->cursor,
            s->layout.lifted_rank,
            s->layout.d_lifted_shape,
            s->layout.d_lifted_strides,
            s->compute);
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));

  CU(Error, cuEventRecord(s->pools.buf[s->pools.current].ready, s->compute));

  s->cursor += elements;
  s->stage.current ^= 1;
  return 1;

Error:
  return 0;
}

// H2D transfer + copy to linear epoch buffer for LOD.
// L0 tiling is deferred to run_lod (lod_morton_to_tiles_lut at lv=0).
// Returns 0 on success, 1 on error.
static int
dispatch_scatter_multiscale(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 1;

  const uint64_t elements = s->stage.bytes_written / bpe;
  if (elements == 0)
    return 1;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];

  ss->dispatched_bytes = s->stage.bytes_written;

  // H2D — wait for prior d_linear copy to finish reading d_in
  CU(Error, cuStreamWaitEvent(s->h2d, ss->t_scatter_end, 0));
  CU(Error, cuEventRecord(ss->t_h2d_start, s->h2d));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)ss->d_in.data,
                       ss->h_in.data,
                       s->stage.bytes_written,
                       s->h2d));
  CU(Error, cuEventRecord(ss->h_in.ready, s->h2d));

  // Copy raw input to linear epoch buffer for LOD downsampling
  CU(Error, cuStreamWaitEvent(s->compute, ss->h_in.ready, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, s->compute));
  {
    uint64_t epoch_offset = (s->cursor % s->layout.epoch_elements) * bpe;
    CU(Error,
       cuMemcpyDtoDAsync((CUdeviceptr)s->lod.d_linear.data + epoch_offset,
                         (CUdeviceptr)ss->d_in.data,
                         elements * bpe,
                         s->compute));
  }
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));

  s->cursor += elements;
  s->stage.current ^= 1;
  return 1;

Error:
  return 0;
}

struct writer_result
writer_append(struct writer* w, struct slice data)
{
  return w->append(w, data);
}

struct writer_result
writer_flush(struct writer* w)
{
  return w->flush(w);
}

struct writer_result
writer_append_wait(struct writer* w, struct slice data)
{
  int stalls = 0;
  const int max_stalls = 10;

  while (data.beg < data.end) {
    struct writer_result r = writer_append(w, data);
    if (r.error)
      return r;

    if (r.rest.beg == data.beg) {
      if (++stalls >= max_stalls) {
        log_error("writer_append_wait: no progress after %d retries", stalls);
        return writer_error_at(data.beg, data.end);
      }
      log_warn(
        "writer_append_wait: stall %d/%d, backing off", stalls, max_stalls);
      platform_sleep_ns(1000000LL << (stalls < 6 ? stalls : 6)); // 1ms..64ms
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return writer_ok();
}

// Software CRC32C (Castagnoli) computed at runtime via a generated table.
static uint32_t crc32c_table[256];
static int crc32c_table_ready;

static void
crc32c_init_table(void)
{
  if (crc32c_table_ready)
    return;
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78 & (0u - (crc & 1)));
    crc32c_table[i] = crc;
  }
  crc32c_table_ready = 1;
}

static uint32_t
crc32c(const void* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFF;
}

// --- Unified shard delivery ---

// Emit completed shards (write index block + finalize).
static int
emit_shards(struct shard_state* ss)
{
  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;

    size_t index_data_bytes = ss->tiles_per_shard_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;
    uint8_t* index_buf = (uint8_t*)malloc(index_total_bytes);
    CHECK(Error, index_buf);

    memcpy(index_buf, sh->index, index_data_bytes);

    uint32_t crc_val = crc32c(index_buf, index_data_bytes);
    memcpy(index_buf + index_data_bytes, &crc_val, 4);

    int wrc = sh->writer->write(
      sh->writer, sh->data_cursor, index_buf, index_buf + index_total_bytes);
    free(index_buf);
    CHECK(Error, wrc == 0);

    CHECK(Error, sh->writer->finalize(sh->writer) == 0);

    sh->writer = NULL;
    sh->data_cursor = 0;
    memset(sh->index, 0xFF, ss->tiles_per_shard_total * 2 * sizeof(uint64_t));
  }

  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return 1;

Error:
  return 0;
}

// Deliver compressed tile data from an aggregate slot to shards.
static int
deliver_to_shards(struct tile_stream_gpu* s,
                  uint8_t level,
                  struct shard_state* ss,
                  struct aggregate_slot* agg_slot)
{
  const uint64_t tps_inner = ss->tiles_per_shard_inner;
  const uint64_t epoch_in_shard = ss->epoch_in_shard;

  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    uint64_t j_start = si * tps_inner;
    uint64_t j_end = j_start + tps_inner;

    struct active_shard* sh = &ss->shards[si];

    if (!sh->writer) {
      uint64_t flat = ss->shard_epoch * ss->shard_inner_count + si;
      sh->writer =
        s->config.shard_sink->open(s->config.shard_sink, level, flat);
      CHECK(Error, sh->writer);
    }

    size_t shard_bytes =
      agg_slot->h_offsets[j_end] - agg_slot->h_offsets[j_start];
    if (shard_bytes > 0) {
      const void* src =
        (const char*)agg_slot->h_aggregated + agg_slot->h_offsets[j_start];
      CHECK(Error,
            sh->writer->write(sh->writer,
                              sh->data_cursor,
                              src,
                              (const char*)src + shard_bytes) == 0);
    }

    for (uint64_t j = j_start; j < j_end; ++j) {
      size_t tile_size = agg_slot->h_offsets[j + 1] - agg_slot->h_offsets[j];
      if (tile_size > 0) {
        uint64_t within_inner = j - j_start;
        uint64_t slot = epoch_in_shard * tps_inner + within_inner;
        size_t tile_off = sh->data_cursor + (agg_slot->h_offsets[j] -
                                             agg_slot->h_offsets[j_start]);
        sh->index[2 * slot] = tile_off;
        sh->index[2 * slot + 1] = tile_size;
      }
    }
    sh->data_cursor += shard_bytes;
  }

  ss->epoch_in_shard++;

  if (ss->epoch_in_shard >= ss->tiles_per_shard_0)
    return emit_shards(ss);

  return 1;

Error:
  return 0;
}

// Deliver compressed tile data from a batch aggregate slot to shards.
// Iterates over active epochs in the batch, delivering each epoch's tiles.
// n_active: number of active epochs for this level in the batch.
// In the aggregate output, tiles are interleaved by epoch within each shard:
//   [shard 0 epoch 0, shard 0 epoch 1, ..., shard 1 epoch 0, ...]
static int
deliver_to_shards_batch(struct tile_stream_gpu* s,
                        uint8_t level,
                        struct shard_state* ss,
                        struct aggregate_slot* agg_slot,
                        uint32_t n_active)
{
  const uint64_t tps_inner = ss->tiles_per_shard_inner;

  for (uint32_t a = 0; a < n_active; ++a) {
    const uint64_t epoch_in_shard = ss->epoch_in_shard;

    for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
      uint64_t j_start = si * n_active * tps_inner + a * tps_inner;
      uint64_t j_end = j_start + tps_inner;

      struct active_shard* sh = &ss->shards[si];

      if (!sh->writer) {
        uint64_t flat = ss->shard_epoch * ss->shard_inner_count + si;
        sh->writer =
          s->config.shard_sink->open(s->config.shard_sink, level, flat);
        CHECK(Error, sh->writer);
      }

      size_t shard_bytes =
        agg_slot->h_offsets[j_end] - agg_slot->h_offsets[j_start];
      if (shard_bytes > 0) {
        const void* src =
          (const char*)agg_slot->h_aggregated + agg_slot->h_offsets[j_start];
        CHECK(Error,
              sh->writer->write(sh->writer,
                                sh->data_cursor,
                                src,
                                (const char*)src + shard_bytes) == 0);
      }

      for (uint64_t j = j_start; j < j_end; ++j) {
        size_t tile_size = agg_slot->h_offsets[j + 1] - agg_slot->h_offsets[j];
        if (tile_size > 0) {
          uint64_t within_inner = j - j_start;
          uint64_t slot_idx = epoch_in_shard * tps_inner + within_inner;
          size_t tile_off = sh->data_cursor + (agg_slot->h_offsets[j] -
                                               agg_slot->h_offsets[j_start]);
          sh->index[2 * slot_idx] = tile_off;
          sh->index[2 * slot_idx + 1] = tile_size;
        }
      }
      sh->data_cursor += shard_bytes;
    }

    ss->epoch_in_shard++;

    if (ss->epoch_in_shard >= ss->tiles_per_shard_0) {
      if (!emit_shards(ss))
        goto Error;
    }
  }

  return 1;

Error:
  return 0;
}

// How many epochs fire for level `lv` within a batch of `n_epochs` epochs.
// L0 fires every epoch; dim0-downsampled level lv fires every 2^lv epochs.
// Returns 0 if the level doesn't fire at all in this batch.
static uint32_t
level_active_epochs(const struct tile_stream_gpu* s, int lv, uint32_t n_epochs)
{
  uint32_t full = s->lod_levels[lv].batch_active_count;
  if (n_epochs >= s->epochs_per_batch)
    return full;
  uint32_t period = (s->dim0_downsample && lv > 0) ? (1u << lv) : 1;
  return (n_epochs >= period) ? n_epochs / period : 0;
}

// Wait for D2H on the given flush slot, record timing, deliver to sinks.
static struct writer_result
wait_and_deliver(struct tile_stream_gpu* s, int fc)
{
  struct flush_slot_gpu* fs = &s->flush[fc];

  CU(Error, cuEventSynchronize(fs->ready));

  // LOD metrics: safe to read now — compress stream waited on t_lod_end
  if (s->enable_multiscale && s->lod.t_start) {
    const size_t bpe = s->config.bytes_per_element;
    const size_t scatter_bytes = s->layout.epoch_elements * bpe;
    const size_t morton_bytes =
      s->lod.plan.levels.ends[s->lod.plan.nlod - 1] * bpe;
    const size_t unified_pool_bytes =
      s->total_tiles * s->layout.tile_stride * bpe;

    accumulate_metric_cu(&s->metrics.lod_gather,
                         s->lod.t_start,
                         s->lod.t_scatter_end,
                         scatter_bytes);
    accumulate_metric_cu(&s->metrics.lod_reduce,
                         s->lod.t_scatter_end,
                         s->lod.t_reduce_end,
                         morton_bytes);
    if (s->dim0_downsample) {
      size_t accum_bpe =
        (s->dim0_reduce_method == lod_reduce_mean && bpe == 2) ? 4 : bpe;
      size_t dim0_bytes = s->dim0.total_elements * accum_bpe;
      accumulate_metric_cu(&s->metrics.lod_dim0_fold,
                           s->lod.t_reduce_end,
                           s->lod.t_dim0_end,
                           dim0_bytes);
    }
    accumulate_metric_cu(&s->metrics.lod_morton_tile,
                         s->lod.t_dim0_end,
                         s->lod.t_end,
                         unified_pool_bytes);
  }

  {
    const size_t pool_bytes =
      (uint64_t)fs->batch_epoch_count *
      s->total_tiles * s->layout.tile_stride * s->config.bytes_per_element;
    const size_t comp_bytes =
      (uint64_t)fs->batch_epoch_count *
      s->total_tiles * s->codec.max_output_size;

    accumulate_metric_cu(&s->metrics.compress,
                         fs->t_compress_start,
                         fs->d_compressed.ready,
                         pool_bytes);
    accumulate_metric_cu(&s->metrics.aggregate,
                         fs->d_compressed.ready,
                         fs->t_aggregate_end,
                         comp_bytes);
    accumulate_metric_cu(
      &s->metrics.d2h, fs->t_d2h_start, fs->ready, comp_bytes);
  }

  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;

    for (int lv = 0; lv < s->nlod; ++lv) {
      if (!(fs->active_levels_mask & (1u << lv)))
        continue;

      uint32_t active_count =
        level_active_epochs(s, lv, (uint32_t)fs->batch_epoch_count);
      if (active_count == 0)
        continue;

      struct lod_level_state* lvl = &s->lod_levels[lv];
      uint64_t tiles_lv = s->level_tile_count[lv];
      sink_bytes += (uint64_t)active_count * tiles_lv * s->codec.max_output_size;

      if (s->epochs_per_batch > 1 && active_count > 1) {
        if (!deliver_to_shards_batch(
              s, (uint8_t)lv, &lvl->shard, &lvl->agg[fc], active_count))
          goto Error;
      } else {
        if (!deliver_to_shards(s, (uint8_t)lv, &lvl->shard, &lvl->agg[fc]))
          goto Error;
      }
    }

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&s->metrics.sink, sink_ms, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

// Drain pending flush from the previous epoch.
static struct writer_result
drain_pending_flush(struct tile_stream_gpu* s)
{
  if (!s->flush_pending)
    return writer_ok();

  s->flush_pending = 0;
  return wait_and_deliver(s, s->flush_current);
}

// --- Batch flush pipeline ---

// Kick compress + aggregate + D2H for a batch of n_epochs epochs.
// fc: flush slot index (0 or 1, matches pools.current before swap).
// n_epochs: number of epochs in this batch (may be < K for final flush).
static int
kick_batch(struct tile_stream_gpu* s, int fc, uint32_t n_epochs)
{
  struct flush_slot_gpu* fs = &s->flush[fc];
  const size_t tile_bytes =
    s->layout.tile_stride * s->config.bytes_per_element;

  // Wait for all per-epoch pool-ready events
  for (uint32_t e = 0; e < n_epochs; ++e)
    CU(Error, cuStreamWaitEvent(s->compress, s->batch_pool_events[e], 0));
  if (s->enable_multiscale && s->lod.t_end)
    CU(Error, cuStreamWaitEvent(s->compress, s->lod.t_end, 0));

  CU(Error, cuEventRecord(fs->t_compress_start, s->compress));

  // Compress all epochs in the batch as one big batch
  size_t batch_tiles = (uint64_t)n_epochs * s->total_tiles;
  CHECK(Error,
        codec_compress(&s->codec,
                       s->pools.buf[fc].data,
                       tile_bytes,
                       fs->d_compressed.data,
                       batch_tiles,
                       s->compress));

  CU(Error, cuEventRecord(fs->d_compressed.ready, s->compress));

  // Per-level batch aggregate on compress stream
  for (int lv = 0; lv < s->nlod; ++lv) {
    if (!(fs->active_levels_mask & (1u << lv)))
      continue;

    uint32_t active_count = level_active_epochs(s, lv, n_epochs);
    if (active_count == 0)
      continue;

    struct lod_level_state* lvl = &s->lod_levels[lv];
    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t tiles_lv = s->level_tile_count[lv];
    uint64_t batch_tile_count =
      (uint64_t)active_count * tiles_lv;
    uint64_t batch_covering =
      (uint64_t)active_count * lvl->agg_layout.covering_count;

    int use_luts = (s->epochs_per_batch > 1 && lvl->d_batch_gather &&
                    active_count == lvl->batch_active_count);
    if (use_luts) {
      CHECK(Error,
            aggregate_batch_by_shard_async(
              fs->d_compressed.data,
              s->codec.d_comp_sizes,
              (const uint32_t*)(uintptr_t)lvl->d_batch_gather,
              (const uint32_t*)(uintptr_t)lvl->d_batch_perm,
              batch_tile_count,
              batch_covering,
              s->codec.max_output_size,
              agg,
              s->compress) == 0);
    } else {
      // K=1 or partial batch: per-epoch aggregate
      for (uint32_t a = 0; a < active_count; ++a) {
        uint32_t period = 1;
        if (s->dim0_downsample && lv > 0)
          period = 1u << lv;
        uint32_t pool_epoch =
          (n_epochs == 1) ? 0 : (a + 1) * period - 1;

        void* d_comp =
          (char*)fs->d_compressed.data +
          (pool_epoch * s->total_tiles + s->level_tile_offset[lv]) *
            s->codec.max_output_size;
        size_t* d_sizes =
          s->codec.d_comp_sizes +
          pool_epoch * s->total_tiles + s->level_tile_offset[lv];

        CHECK(Error,
              aggregate_by_shard_async(&lvl->agg_layout,
                                       d_comp,
                                       d_sizes,
                                       &lvl->agg[fc],
                                       s->compress) == 0);
      }
    }
    CU(Error, cuEventRecord(agg->ready, s->compress));
  }

  CU(Error, cuEventRecord(fs->t_aggregate_end, s->compress));

  // Per-level D2H on d2h stream
  CU(Error, cuStreamWaitEvent(s->d2h, fs->t_aggregate_end, 0));
  CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));

  for (int lv = 0; lv < s->nlod; ++lv) {
    if (!(fs->active_levels_mask & (1u << lv)))
      continue;

    uint32_t active_count = level_active_epochs(s, lv, n_epochs);
    if (active_count == 0)
      continue;

    struct lod_level_state* lvl = &s->lod_levels[lv];
    struct aggregate_slot* agg = &lvl->agg[fc];
    uint64_t tiles_lv = s->level_tile_count[lv];
    size_t agg_bytes =
      (uint64_t)active_count * tiles_lv * s->codec.max_output_size;
    uint64_t batch_covering =
      (uint64_t)active_count * lvl->agg_layout.covering_count;

    CU(Error,
       cuMemcpyDtoHAsync(agg->h_aggregated,
                         (CUdeviceptr)agg->d_aggregated,
                         agg_bytes,
                         s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(agg->h_offsets,
                         (CUdeviceptr)agg->d_offsets,
                         (batch_covering + 1) * sizeof(size_t),
                         s->d2h));
    CU(Error, cuEventRecord(agg->ready, s->d2h));
  }

  CU(Error, cuEventRecord(fs->ready, s->d2h));

  return 1;

Error:
  return 0;
}

// Run LOD scatter + fill_ends + reduce on the linear epoch buffer.
// Outputs to d_morton. Must be called after L0 scatter completes for the epoch.
static int
run_lod(struct tile_stream_gpu* s)
{
  if (!s->enable_multiscale || !s->lod.d_linear.data) {
    // Non-multiscale: all levels (just L0) are active
    struct flush_slot_gpu* fs = &s->flush[s->pools.current];
    fs->batch_active_masks[s->epochs_accumulated] = 1;
    fs->active_levels_mask |= 1;
    return 1;
  }

  struct lod_plan* p = &s->lod.plan;
  const size_t bpe = s->config.bytes_per_element;
  enum lod_dtype dtype = (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

  CU(Error, cuEventRecord(s->lod.t_start, s->compute));

  lod_gather_lut((CUdeviceptr)s->lod.d_morton.data,
                 (CUdeviceptr)s->lod.d_linear.data,
                 s->lod.d_gather_lut,
                 s->lod.d_batch_offsets,
                 dtype,
                 p->lod_counts[0],
                 p->batch_count,
                 s->compute);

  CU(Error, cuEventRecord(s->lod.t_scatter_end, s->compute));

  for (int l = 0; l < p->nlod - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    lod_fill_ends_gpu(s->lod.d_level_ends[l],
                      p->lod_ndim,
                      s->lod.d_child_shapes[l],
                      s->lod.d_parent_shapes[l],
                      p->lod_shapes[l],
                      p->lod_shapes[l + 1],
                      n_parents,
                      s->compute);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    lod_reduce((CUdeviceptr)s->lod.d_morton.data,
               s->lod.d_level_ends[l],
               dtype,
               s->config.reduce_method,
               src_level.beg,
               dst_level.beg,
               p->lod_counts[l],
               p->lod_counts[l + 1],
               p->batch_count,
               s->compute);
  }

  CU(Error, cuEventRecord(s->lod.t_reduce_end, s->compute));

  // Dim0: fold + emit (writes back to morton regions, no tiles yet)
  uint32_t active_levels_mask = 1; // L0 always active

  if (s->dim0_downsample && s->dim0.total_elements > 0) {
    struct dim0_state* d0 = &s->dim0;

    // Upload current counts to device before fused kernel
    CU(Error,
       cuMemcpyHtoDAsync(
         d0->d_counts, d0->counts, p->nlod * sizeof(uint32_t), s->compute));

    // Single fused fold over all levels 1+
    CUdeviceptr morton_1plus =
      (CUdeviceptr)s->lod.d_morton.data + d0->morton_offset * bpe;
    lod_accum_fold_fused((CUdeviceptr)d0->d_accum.data,
                         morton_1plus,
                         d0->d_level_ids,
                         d0->d_counts,
                         dtype,
                         s->dim0_reduce_method,
                         d0->total_elements,
                         s->compute);

    // Increment counts, emit ready levels back to morton
    for (int lv = 1; lv < p->nlod; ++lv) {
      d0->counts[lv]++;
      uint32_t period = 1u << lv;

      if (d0->counts[lv] >= period) {
        struct lod_span lev = lod_spans_at(&p->levels, lv);
        uint64_t n_elements = p->batch_count * p->lod_counts[lv];

        uint64_t accum_offset = 0;
        for (int k = 1; k < lv; ++k)
          accum_offset += p->batch_count * p->lod_counts[k];

        size_t accum_bpe =
          (s->dim0_reduce_method == lod_reduce_mean && bpe == 2) ? 4 : bpe;

        CUdeviceptr morton_lv =
          (CUdeviceptr)s->lod.d_morton.data + lev.beg * bpe;
        CUdeviceptr accum_lv =
          (CUdeviceptr)d0->d_accum.data + accum_offset * accum_bpe;

        lod_accum_emit(morton_lv,
                       accum_lv,
                       dtype,
                       s->dim0_reduce_method,
                       n_elements,
                       d0->counts[lv],
                       s->compute);

        d0->counts[lv] = 0;
        active_levels_mask |= (1u << lv);
      }
    }
  }

  CU(Error, cuEventRecord(s->lod.t_dim0_end, s->compute));

  // Morton-to-tile scatter: L0 always, levels 1+ when active
  {
    struct lod_span lev0 = lod_spans_at(&p->levels, 0);
    void* pool = current_pool_epoch(s, s->epochs_accumulated);

    lod_morton_to_tiles_lut((CUdeviceptr)pool,
                            (CUdeviceptr)s->lod.d_morton.data + lev0.beg * bpe,
                            s->lod.d_morton_tile_lut[0],
                            s->lod.d_morton_batch_tile_offsets[0],
                            dtype,
                            p->lod_counts[0],
                            p->batch_count,
                            s->compute);
  }

  for (int lv = 1; lv < p->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    struct lod_span lev = lod_spans_at(&p->levels, lv);
    CUdeviceptr morton_lv =
      (CUdeviceptr)s->lod.d_morton.data + lev.beg * bpe;

    CUdeviceptr dst =
      (CUdeviceptr)current_pool_epoch(s, s->epochs_accumulated) +
      s->level_tile_offset[lv] * s->layout.tile_stride * bpe;

    lod_morton_to_tiles_lut(dst,
                            morton_lv,
                            s->lod.d_morton_tile_lut[lv],
                            s->lod.d_morton_batch_tile_offsets[lv],
                            dtype,
                            p->lod_counts[lv],
                            p->batch_count,
                            s->compute);
  }

  // Store per-epoch mask and accumulate into union mask
  {
    struct flush_slot_gpu* fs = &s->flush[s->pools.current];
    fs->batch_active_masks[s->epochs_accumulated] = active_levels_mask;
    fs->active_levels_mask |= active_levels_mask;
  }

  // Signal pool ready AFTER all levels' morton-to-tile scatter
  CU(Error, cuEventRecord(s->pools.buf[s->pools.current].ready, s->compute));
  CU(Error, cuEventRecord(s->lod.t_end, s->compute));
  return 1;

Error:
  return 0;
}

// Accumulate one epoch into the current batch, or flush when batch is full.
// Called at each epoch boundary.
static struct writer_result
accumulate_epoch_or_flush(struct tile_stream_gpu* s)
{
  const uint32_t K = s->epochs_per_batch;

  // Run LOD pipeline: populates current epoch's pool tiles + LOD level tiles
  if (!run_lod(s))
    return writer_error();

  // Record per-epoch pool-ready event
  CU(Error,
     cuEventRecord(s->batch_pool_events[s->epochs_accumulated], s->compute));

  s->epochs_accumulated++;

  if (s->epochs_accumulated < K)
    return writer_ok();

  // Batch is full — drain previous, kick, swap
  const int fc = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush[fc];

  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  fs->batch_epoch_count = (int)s->epochs_accumulated;
  if (!kick_batch(s, fc, s->epochs_accumulated))
    return writer_error();

  // Swap pool and zero next super-pool (K epochs)
  dbuf_swap(&s->pools);
  void* next = current_pool_base(s);
  size_t total_pool_bytes =
    (uint64_t)K * s->total_tiles * s->layout.tile_stride *
    s->config.bytes_per_element;
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)next, 0, total_pool_bytes, s->compute));

  // Reset for next batch
  s->epochs_accumulated = 0;
  s->flush[s->pools.current].active_levels_mask = 0;
  memset(s->flush[s->pools.current].batch_active_masks, 0,
         sizeof(s->flush[s->pools.current].batch_active_masks));

  s->flush_pending = 1;
  s->flush_current = fc;
  return writer_ok();

Error:
  return writer_error();
}

// Kick and deliver a single epoch from the super-pool (used for partial batch).
// epoch_in_batch: which epoch within the super-pool to process.
static struct writer_result
kick_and_deliver_one_epoch(struct tile_stream_gpu* s,
                           int fc,
                           uint32_t epoch_in_batch,
                           uint32_t active_mask)
{
  struct flush_slot_gpu* fs = &s->flush[fc];
  const size_t bpe = s->config.bytes_per_element;
  const size_t tile_bytes = s->layout.tile_stride * bpe;

  // Wait for this epoch's pool-ready event
  CU(Error, cuStreamWaitEvent(s->compress, s->batch_pool_events[epoch_in_batch], 0));
  if (s->enable_multiscale && s->lod.t_end)
    CU(Error, cuStreamWaitEvent(s->compress, s->lod.t_end, 0));

  CU(Error, cuEventRecord(fs->t_compress_start, s->compress));

  // Compress one epoch's tiles from within the super-pool
  void* epoch_pool = (char*)s->pools.buf[fc].data +
    (uint64_t)epoch_in_batch * s->total_tiles * tile_bytes;
  void* epoch_comp = (char*)fs->d_compressed.data; // reuse start of compressed buf

  CHECK(Error,
        codec_compress(&s->codec,
                       epoch_pool,
                       tile_bytes,
                       epoch_comp,
                       s->total_tiles,
                       s->compress));

  CU(Error, cuEventRecord(fs->d_compressed.ready, s->compress));

  // Per-level aggregate
  for (int lv = 0; lv < s->nlod; ++lv) {
    if (!(active_mask & (1u << lv)))
      continue;

    struct lod_level_state* lvl = &s->lod_levels[lv];
    struct aggregate_slot* agg = &lvl->agg[fc];

    void* d_comp = (char*)epoch_comp +
                   s->level_tile_offset[lv] * s->codec.max_output_size;
    size_t* d_sizes = s->codec.d_comp_sizes + s->level_tile_offset[lv];

    CHECK(Error,
          aggregate_by_shard_async(&lvl->agg_layout,
                                   d_comp,
                                   d_sizes,
                                   agg,
                                   s->compress) == 0);
    CU(Error, cuEventRecord(agg->ready, s->compress));
  }

  CU(Error, cuEventRecord(fs->t_aggregate_end, s->compress));

  // D2H
  CU(Error, cuStreamWaitEvent(s->d2h, fs->t_aggregate_end, 0));
  CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));

  for (int lv = 0; lv < s->nlod; ++lv) {
    if (!(active_mask & (1u << lv)))
      continue;

    struct aggregate_slot* agg = &s->lod_levels[lv].agg[fc];
    size_t pool_bytes_lv = s->level_tile_count[lv] * s->codec.max_output_size;

    CU(Error,
       cuMemcpyDtoHAsync(agg->h_aggregated,
                         (CUdeviceptr)agg->d_aggregated,
                         pool_bytes_lv,
                         s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(agg->h_offsets,
                         (CUdeviceptr)agg->d_offsets,
                         (s->lod_levels[lv].agg_layout.covering_count + 1) *
                           sizeof(size_t),
                         s->d2h));
    CU(Error, cuEventRecord(agg->ready, s->d2h));
  }

  CU(Error, cuEventRecord(fs->ready, s->d2h));

  // Wait and deliver synchronously
  CU(Error, cuEventSynchronize(fs->ready));

  fs->batch_epoch_count = 1;
  {
    struct platform_clock sink_clock = { 0 };
    platform_toc(&sink_clock);
    size_t sink_bytes = 0;

    for (int lv = 0; lv < s->nlod; ++lv) {
      if (!(active_mask & (1u << lv)))
        continue;
      sink_bytes += s->level_tile_count[lv] * s->codec.max_output_size;
      if (!deliver_to_shards(
            s, (uint8_t)lv, &s->lod_levels[lv].shard, &s->lod_levels[lv].agg[fc]))
        goto Error;
    }

    float sink_ms = platform_toc(&sink_clock) * 1000.0f;
    accumulate_metric_ms(&s->metrics.sink, sink_ms, sink_bytes);
  }

  return writer_ok();

Error:
  return writer_error();
}

// Synchronously flush accumulated epochs (partial or full batch).
// Used at final flush when there are accumulated epochs that haven't been kicked.
static struct writer_result
flush_accumulated_sync(struct tile_stream_gpu* s)
{
  if (s->epochs_accumulated == 0)
    return writer_ok();

  const int fc = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush[fc];

  if (s->epochs_accumulated == s->epochs_per_batch) {
    // Full batch: use the batch pipeline
    fs->batch_epoch_count = (int)s->epochs_accumulated;
    if (!kick_batch(s, fc, s->epochs_accumulated))
      return writer_error();

    struct writer_result r = wait_and_deliver(s, fc);
    s->epochs_accumulated = 0;
    s->flush[s->pools.current].active_levels_mask = 0;
    memset(s->flush[s->pools.current].batch_active_masks, 0,
           sizeof(s->flush[s->pools.current].batch_active_masks));
    return r;
  }

  // Partial batch: process each epoch individually
  for (uint32_t e = 0; e < s->epochs_accumulated; ++e) {
    uint32_t mask = fs->batch_active_masks[e];
    if (!mask)
      continue;

    struct writer_result r = kick_and_deliver_one_epoch(s, fc, e, mask);
    if (r.error)
      return r;
  }

  s->epochs_accumulated = 0;
  fs->active_levels_mask = 0;
  memset(fs->batch_active_masks, 0, sizeof(fs->batch_active_masks));
  return writer_ok();
}

// Drain partial dim0 accumulators on final flush.
// Emits any level with count > 0, scatters to tiles, and delivers.
static struct writer_result
flush_partial_dim0(struct tile_stream_gpu* s)
{
  if (!s->dim0_downsample || !s->enable_multiscale)
    return writer_ok();

  struct dim0_state* d0 = &s->dim0;
  struct lod_plan* p = &s->lod.plan;
  const size_t bpe = s->config.bytes_per_element;
  enum lod_dtype dtype = (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

  // Check if any level has pending data
  uint32_t active_levels_mask = 0;
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (d0->counts[lv] > 0)
      active_levels_mask |= (1u << lv);
  }

  if (!active_levels_mask)
    return writer_ok();

  const int fc = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush[fc];
  fs->active_levels_mask = active_levels_mask;
  fs->batch_epoch_count = 1; // partial dim0 flush is always 1 epoch

  // Zero LOD level regions of pool, emit partial accums, scatter to tiles
  for (int lv = 1; lv < p->nlod; ++lv) {
    if (!(active_levels_mask & (1u << lv)))
      continue;

    uint64_t n_elements = p->batch_count * p->lod_counts[lv];

    // Compute offset of this level within the packed accumulator
    uint64_t accum_offset = 0;
    for (int k = 1; k < lv; ++k)
      accum_offset += p->batch_count * p->lod_counts[k];

    size_t accum_bpe =
      (s->dim0_reduce_method == lod_reduce_mean && bpe == 2) ? 4 : bpe;

    struct lod_span lev = lod_spans_at(&p->levels, lv);
    CUdeviceptr morton_lv =
      (CUdeviceptr)s->lod.d_morton.data + lev.beg * bpe;
    CUdeviceptr accum_lv =
      (CUdeviceptr)d0->d_accum.data + accum_offset * accum_bpe;

    // Emit with actual count (not period) — mean divides by actual count
    lod_accum_emit(morton_lv,
                   accum_lv,
                   dtype,
                   s->dim0_reduce_method,
                   n_elements,
                   d0->counts[lv],
                   s->compute);

    d0->counts[lv] = 0;

    // Zero and scatter to tile pool (epoch 0 in current pool)
    CUdeviceptr dst =
      (CUdeviceptr)current_pool_epoch(s, 0) +
      s->level_tile_offset[lv] * s->layout.tile_stride * bpe;
    size_t lv_pool_bytes = s->level_tile_count[lv] * s->layout.tile_stride * bpe;
    CU(Error, cuMemsetD8Async(dst, 0, lv_pool_bytes, s->compute));

    lod_morton_to_tiles_lut(dst,
                            morton_lv,
                            s->lod.d_morton_tile_lut[lv],
                            s->lod.d_morton_batch_tile_offsets[lv],
                            dtype,
                            p->lod_counts[lv],
                            p->batch_count,
                            s->compute);
  }

  CU(Error, cuEventRecord(s->pools.buf[fc].ready, s->compute));
  if (s->lod.t_end)
    CU(Error, cuEventRecord(s->lod.t_end, s->compute));

  CU(Error, cuEventRecord(s->batch_pool_events[0], s->compute));
  if (!kick_batch(s, fc, 1))
    return writer_error();
  return wait_and_deliver(s, fc);

Error:
  return writer_error();
}

struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s)
{
  return s->metrics;
}

// --- Destroy ---

static void
destroy_level_state(struct lod_level_state* lls)
{
  aggregate_layout_destroy(&lls->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&lls->agg[i]);
  if (lls->d_batch_gather)
    CUWARN(cuMemFree(lls->d_batch_gather));
  if (lls->d_batch_perm)
    CUWARN(cuMemFree(lls->d_batch_perm));
  if (lls->shard.shards) {
    for (uint64_t i = 0; i < lls->shard.shard_inner_count; ++i)
      free(lls->shard.shards[i].index);
    free(lls->shard.shards);
  }
}

void
tile_stream_gpu_destroy(struct tile_stream_gpu* stream)
{
  if (!stream)
    return;

  CUWARN(cuStreamDestroy(stream->h2d));
  CUWARN(cuStreamDestroy(stream->compute));
  CUWARN(cuStreamDestroy(stream->compress));
  CUWARN(cuStreamDestroy(stream->d2h));

  CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_shape));
  CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_strides));

  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stream->stage.slot[i];
    CUWARN(cuEventDestroy(ss->t_h2d_start));
    CUWARN(cuEventDestroy(ss->t_scatter_start));
    CUWARN(cuEventDestroy(ss->t_scatter_end));
    buffer_free(&ss->h_in);
    buffer_free(&ss->d_in);
  }

  // Tile pools
  buffer_free(&stream->pools.buf[0]);
  buffer_free(&stream->pools.buf[1]);

  // Unified codec
  codec_free(&stream->codec);

  // Flush slots
  for (int i = 0; i < 2; ++i) {
    struct flush_slot_gpu* fs = &stream->flush[i];
    buffer_free(&fs->d_compressed);
    CUWARN(cuEventDestroy(fs->t_compress_start));
    CUWARN(cuEventDestroy(fs->t_aggregate_end));
    CUWARN(cuEventDestroy(fs->t_d2h_start));
    CUWARN(cuEventDestroy(fs->ready));
  }

  // Batch pool events
  for (uint32_t i = 0; i < stream->epochs_per_batch; ++i) {
    if (stream->batch_pool_events[i])
      CUWARN(cuEventDestroy(stream->batch_pool_events[i]));
  }

  // Per-level aggregate + shard state
  for (int lv = 0; lv < stream->nlod; ++lv)
    destroy_level_state(&stream->lod_levels[lv]);

  // Dim0 state
  buffer_free(&stream->dim0.d_accum);
  if (stream->dim0.d_level_ids)
    CUWARN(cuMemFree(stream->dim0.d_level_ids));
  if (stream->dim0.d_counts)
    CUWARN(cuMemFree(stream->dim0.d_counts));

  // LOD cleanup
  buffer_free(&stream->lod.d_linear);
  buffer_free(&stream->lod.d_morton);
  CUWARN(cuMemFree(stream->lod.d_full_shape));
  CUWARN(cuMemFree(stream->lod.d_lod_shape));
  CUWARN(cuMemFree(stream->lod.d_ends));
  CUWARN(cuMemFree(stream->lod.d_gather_lut));
  CUWARN(cuMemFree(stream->lod.d_batch_offsets));
  for (int i = 0; i < stream->lod.plan.nlod; ++i) {
    CUWARN(cuMemFree(stream->lod.d_morton_tile_lut[i]));
    CUWARN(cuMemFree(stream->lod.d_morton_batch_tile_offsets[i]));
  }
  for (int i = 0; i < stream->lod.plan.nlod - 1; ++i) {
    CUWARN(cuMemFree(stream->lod.d_child_shapes[i]));
    CUWARN(cuMemFree(stream->lod.d_parent_shapes[i]));
    CUWARN(cuMemFree(stream->lod.d_level_ends[i]));
  }
  for (int i = 1; i < stream->lod.plan.nlod; ++i) {
    CUWARN(cuMemFree((CUdeviceptr)stream->lod.layouts[i].d_lifted_shape));
    CUWARN(cuMemFree((CUdeviceptr)stream->lod.layouts[i].d_lifted_strides));
  }
  if (stream->lod.t_start) {
    CUWARN(cuEventDestroy(stream->lod.t_start));
    CUWARN(cuEventDestroy(stream->lod.t_scatter_end));
    CUWARN(cuEventDestroy(stream->lod.t_reduce_end));
    CUWARN(cuEventDestroy(stream->lod.t_dim0_end));
    CUWARN(cuEventDestroy(stream->lod.t_end));
  }
  lod_plan_free(&stream->lod.plan);

  *stream = (struct tile_stream_gpu){ 0 };
}

// --- Create ---

// Forward declarations for vtable
static struct writer_result
tile_stream_gpu_append(struct writer* self, struct slice input);
static struct writer_result
tile_stream_gpu_flush(struct writer* self);

static int
init_cuda_streams_and_events(struct tile_stream_gpu* s)
{
  CU(Fail, cuStreamCreate(&s->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&s->stage.slot[i].t_h2d_start, CU_EVENT_DEFAULT));
    CU(Fail,
       cuEventCreate(&s->stage.slot[i].t_scatter_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->stage.slot[i].t_scatter_end, CU_EVENT_DEFAULT));
  }

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&s->flush[i].t_compress_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].t_aggregate_end, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].t_d2h_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].ready, CU_EVENT_DEFAULT));
  }

  return 1;
Fail:
  return 0;
}

static int
init_l0_layout(struct tile_stream_gpu* s)
{
  const uint8_t rank = s->config.rank;
  const size_t bpe = s->config.bytes_per_element;
  const struct dimension* dims = s->config.dimensions;

  s->layout.lifted_rank = 2 * rank;
  s->layout.tile_elements = 1;

  uint64_t tile_count[MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    s->layout.lifted_shape[2 * i] = tile_count[i];
    s->layout.lifted_shape[2 * i + 1] = dims[i].tile_size;
    s->layout.tile_elements *= dims[i].tile_size;
  }

  {
    size_t alignment = codec_alignment(s->config.codec);
    size_t tile_bytes = s->layout.tile_elements * bpe;
    size_t padded_bytes = align_up(tile_bytes, alignment);
    s->layout.tile_stride = padded_bytes / bpe;
  }

  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)s->layout.tile_stride;

    for (int i = rank - 1; i >= 0; --i) {
      s->layout.lifted_strides[2 * i + 1] = n_stride;
      n_stride *= (int64_t)dims[i].tile_size;

      s->layout.lifted_strides[2 * i] = t_stride;
      t_stride *= (int64_t)tile_count[i];
    }
  }

  s->layout.tiles_per_epoch =
    s->layout.lifted_strides[0] / s->layout.tile_stride;
  s->layout.epoch_elements =
    s->layout.tiles_per_epoch * s->layout.tile_elements;
  s->layout.lifted_strides[0] = 0; // collapse epoch dim
  s->layout.tile_pool_bytes =
    s->layout.tiles_per_epoch * s->layout.tile_stride * bpe;

  {
    const size_t shape_bytes = s->layout.lifted_rank * sizeof(uint64_t);
    const size_t strides_bytes = s->layout.lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&s->layout.d_lifted_shape, shape_bytes));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&s->layout.d_lifted_strides, strides_bytes));
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)s->layout.d_lifted_shape,
                    s->layout.lifted_shape,
                    shape_bytes));
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)s->layout.d_lifted_strides,
                    s->layout.lifted_strides,
                    strides_bytes));
  }

  return 1;
Fail:
  return 0;
}

static int
init_staging_buffers(struct tile_stream_gpu* s)
{
  for (int i = 0; i < 2; ++i) {
    CHECK(Fail,
          (s->stage.slot[i].h_in =
             buffer_new(s->config.buffer_capacity_bytes, host, 0))
            .data);
    CHECK(Fail,
          (s->stage.slot[i].d_in =
             buffer_new(s->config.buffer_capacity_bytes, device, 0))
            .data);
  }

  return 1;
Fail:
  return 0;
}

static int
init_tile_pools(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const uint32_t K = s->epochs_per_batch;

  // Compute total_tiles and level tile offsets from L0 + LOD layouts
  // These are per-epoch counts; the pool holds K epochs.
  s->level_tile_count[0] = s->layout.tiles_per_epoch;
  s->level_tile_offset[0] = 0;
  s->total_tiles = s->layout.tiles_per_epoch;

  for (int lv = 1; lv < s->nlod; ++lv) {
    s->level_tile_count[lv] = s->lod.layouts[lv].tiles_per_epoch;
    s->level_tile_offset[lv] = s->total_tiles;
    s->total_tiles += s->level_tile_count[lv];
  }

  // Pool holds K epochs worth of tiles
  const size_t pool_bytes =
    (uint64_t)K * s->total_tiles * s->layout.tile_stride * bpe;

  for (int i = 0; i < 2; ++i) {
    CHECK(Fail, (s->pools.buf[i] = buffer_new(pool_bytes, device, 0)).data);
    CU(Fail,
       cuMemsetD8Async(
         (CUdeviceptr)s->pools.buf[i].data, 0, pool_bytes, s->compute));
  }

  return 1;
Fail:
  return 0;
}

static int
init_compression(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const uint32_t K = s->epochs_per_batch;
  const uint64_t M = (uint64_t)K * s->total_tiles;
  const size_t tile_bytes = s->layout.tile_stride * bpe;

  CHECK(Fail, codec_init(&s->codec, s->config.codec, tile_bytes, M));

  for (int fc = 0; fc < 2; ++fc) {
    struct flush_slot_gpu* fs = &s->flush[fc];
    CHECK(
      Fail,
      (fs->d_compressed = buffer_new(M * s->codec.max_output_size, device, 0))
        .data);
  }

  return 1;
Fail:
  return 0;
}

static int
init_aggregate_and_shards(struct tile_stream_gpu* s)
{
  const uint8_t rank = s->config.rank;
  const struct dimension* dims = s->config.dimensions;
  const uint32_t K = s->epochs_per_batch;

  crc32c_init_table();

  for (int lv = 0; lv < s->nlod; ++lv) {
    uint64_t tile_count[MAX_RANK / 2];
    uint64_t tiles_per_shard[MAX_RANK / 2];

    if (lv == 0) {
      for (int d = 0; d < rank; ++d) {
        tile_count[d] = ceildiv(dims[d].size, dims[d].tile_size);
        uint64_t tps = dims[d].tiles_per_shard;
        tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
      }
    } else {
      const uint64_t* lv_shape = s->lod.plan.shapes[lv];
      for (int d = 0; d < rank; ++d) {
        tile_count[d] = ceildiv(lv_shape[d], dims[d].tile_size);
        uint64_t tps = dims[d].tiles_per_shard;
        tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
      }
    }

    uint64_t tiles_lv = s->level_tile_count[lv];

    // Epochs per batch for this level.
    // L0 fires every epoch; higher levels fire every 2^lv epochs.
    uint32_t batch_count = K;
    if (s->dim0_downsample && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count = (K >= period) ? K / period : 1;
    }
    s->lod_levels[lv].batch_active_count = batch_count;

    // Aggregate layout: per-epoch tile geometry (used by the single-epoch
    // aggregate path for K=1 or partial batch fallback).
    CHECK(Fail,
          aggregate_layout_init(&s->lod_levels[lv].agg_layout,
                                rank,
                                tile_count,
                                tiles_per_shard,
                                tiles_lv,
                                s->codec.max_output_size) == 0);

    // Aggregate slots sized for batch.
    uint64_t batch_tiles = (uint64_t)batch_count * tiles_lv;
    uint64_t batch_covering =
      (uint64_t)batch_count * s->lod_levels[lv].agg_layout.covering_count;
    size_t batch_agg_bytes = batch_tiles * s->codec.max_output_size;

    for (int i = 0; i < 2; ++i) {
      CHECK(Fail,
            aggregate_batch_slot_init(&s->lod_levels[lv].agg[i],
                                      batch_tiles,
                                      batch_covering,
                                      batch_agg_bytes) == 0);
      CU(Fail, cuEventRecord(s->lod_levels[lv].agg[i].ready, s->compute));
    }

    // Shard state
    struct shard_state* ss = &s->lod_levels[lv].shard;
    {
      uint64_t tps0 = tiles_per_shard[0];
      // Dim0-downsampled levels emit every 2^lv epochs, so each emission
      // covers 2^lv times more temporal range -> fewer tiles per shard.
      if (s->dim0_downsample && lv > 0) {
        uint64_t divisor = 1ull << lv;
        tps0 = (tps0 > divisor) ? tps0 / divisor : 1;
      }
      ss->tiles_per_shard_0 = tps0;
    }
    ss->tiles_per_shard_inner = 1;
    for (int d = 1; d < rank; ++d)
      ss->tiles_per_shard_inner *= tiles_per_shard[d];
    ss->tiles_per_shard_total =
      ss->tiles_per_shard_0 * ss->tiles_per_shard_inner;

    ss->shard_inner_count = 1;
    for (int d = 1; d < rank; ++d)
      ss->shard_inner_count *= ceildiv(tile_count[d], tiles_per_shard[d]);

    ss->shards = (struct active_shard*)calloc(ss->shard_inner_count,
                                              sizeof(struct active_shard));
    CHECK(Fail, ss->shards);

    size_t index_bytes = 2 * ss->tiles_per_shard_total * sizeof(uint64_t);
    for (uint64_t i = 0; i < ss->shard_inner_count; ++i) {
      ss->shards[i].index = (uint64_t*)malloc(index_bytes);
      CHECK(Fail, ss->shards[i].index);
      memset(ss->shards[i].index, 0xFF, index_bytes);
    }

    ss->epoch_in_shard = 0;
    ss->shard_epoch = 0;
  }

  return 1;
Fail:
  return 0;
}

// Precompute per-level gather and permutation LUTs for batch aggregate.
//
// gather[a * tiles_lv + j] maps batch-tile (a, j) to its position in the
// compressed buffer: pool_epoch(a) * total_tiles + level_offset + j.
//
// perm[a * tiles_lv + j] maps batch-tile (a, j) to the shard-ordered
// output position: shard_perm(j) * batch_count + a.  This interleaves
// epochs within each shard so tiles appear in epoch order.
//
// Must be called AFTER init_aggregate_and_shards (needs agg_layout).
static int
init_batch_luts(struct tile_stream_gpu* s)
{
  const uint32_t K = s->epochs_per_batch;
  if (K <= 1)
    return 1;

  for (int lv = 0; lv < s->nlod; ++lv) {
    struct lod_level_state* lvl = &s->lod_levels[lv];
    uint32_t batch_count = lvl->batch_active_count;
    uint64_t tiles_lv = s->level_tile_count[lv];
    uint64_t lut_len = (uint64_t)batch_count * tiles_lv;

    if (lut_len == 0)
      continue;

    uint32_t* h_gather = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
    uint32_t* h_perm = (uint32_t*)malloc(lut_len * sizeof(uint32_t));
    CHECK(Fail, h_gather && h_perm);

    const struct aggregate_layout* al = &lvl->agg_layout;

    for (uint32_t a = 0; a < batch_count; ++a) {
      // Level lv fires at pool epochs period-1, 2*period-1, etc.
      uint32_t period = 1;
      if (s->dim0_downsample && lv > 0)
        period = 1u << lv;
      uint32_t pool_epoch = (a + 1) * period - 1;

      for (uint64_t j = 0; j < tiles_lv; ++j) {
        uint64_t idx = (uint64_t)a * tiles_lv + j;

        // gather: map batch-tile to compressed buffer position
        h_gather[idx] = (uint32_t)(pool_epoch * s->total_tiles +
                                    s->level_tile_offset[lv] + j);

        // perm: shard-order position via lifted strides, interleaved by epoch
        uint64_t perm_pos = 0;
        uint64_t rest = j;
        for (int d = al->lifted_rank - 1; d >= 0; --d) {
          uint64_t coord = rest % al->lifted_shape[d];
          rest /= al->lifted_shape[d];
          perm_pos += coord * (uint64_t)al->lifted_strides[d];
        }
        h_perm[idx] = (uint32_t)(perm_pos * batch_count + a);
      }
    }

    CU(Fail2, cuMemAlloc(&lvl->d_batch_gather, lut_len * sizeof(uint32_t)));
    CU(Fail2,
       cuMemcpyHtoD(lvl->d_batch_gather, h_gather, lut_len * sizeof(uint32_t)));

    CU(Fail2, cuMemAlloc(&lvl->d_batch_perm, lut_len * sizeof(uint32_t)));
    CU(Fail2,
       cuMemcpyHtoD(lvl->d_batch_perm, h_perm, lut_len * sizeof(uint32_t)));

    free(h_gather);
    free(h_perm);
    continue;

  Fail2:
    free(h_gather);
    free(h_perm);
    goto Fail;
  }

  return 1;
Fail:
  return 0;
}

// Allocate and seed per-epoch batch pool events.
static int
init_batch_events(struct tile_stream_gpu* s)
{
  const uint32_t K = s->epochs_per_batch;
  for (uint32_t i = 0; i < K; ++i) {
    CU(Fail, cuEventCreate(&s->batch_pool_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(s->batch_pool_events[i], s->compute));
  }
  return 1;
Fail:
  return 0;
}

// init_lod_layouts: lod_plan, per-level layouts, device shape arrays,
// morton_tile structs. Must be called BEFORE init_tile_pools so total_tiles can
// be computed.
static int
init_lod_layouts(struct tile_stream_gpu* s)
{
  if (!s->enable_multiscale)
    return 1;

  const uint8_t rank = s->config.rank;
  const struct dimension* dims = s->config.dimensions;
  const size_t bpe = s->config.bytes_per_element;

  // Use epoch shape, not full volume shape.
  uint64_t shape[MAX_RANK / 2];
  uint64_t tile_shape[MAX_RANK / 2];
  shape[0] = dims[0].tile_size;
  for (int d = 1; d < rank; ++d)
    shape[d] = dims[d].size;
  for (int d = 0; d < rank; ++d)
    tile_shape[d] = dims[d].tile_size;

  CHECK(Fail,
        lod_plan_init(&s->lod.plan,
                      rank,
                      shape,
                      tile_shape,
                      (uint8_t)s->lod_mask,
                      LOD_MAX_LEVELS));

  // Validate LOD counts fit in uint32_t (scatter LUT uses uint32_t offsets)
  for (int k = 0; k < s->lod.plan.nlod; ++k) {
    if (s->lod.plan.lod_counts[k] > UINT32_MAX) {
      log_error("LOD level %d count %llu exceeds uint32_t limit",
                k,
                (unsigned long long)s->lod.plan.lod_counts[k]);
      goto Fail;
    }
  }

  // Upload shapes to device
  CU(Fail, cuMemAlloc(&s->lod.d_full_shape, rank * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(
       s->lod.d_full_shape, s->lod.plan.shapes[0], rank * sizeof(uint64_t)));

  if (s->lod.plan.lod_ndim > 0) {
    CU(
      Fail,
      cuMemAlloc(&s->lod.d_lod_shape, s->lod.plan.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(s->lod.d_lod_shape,
                    s->lod.plan.lod_shapes[0],
                    s->lod.plan.lod_ndim * sizeof(uint64_t)));
  }

  // Build gather LUT for L0 scatter
  if (s->lod.plan.lod_ndim > 0) {
    struct lod_plan* p = &s->lod.plan;
    uint64_t lod_count = p->lod_counts[0];

    // Validate epoch_elements fits in u32 (gather LUT uses u32 offsets)
    {
      uint64_t epoch_elements = 1;
      for (int d = 0; d < rank; ++d)
        epoch_elements *= shape[d];
      if (epoch_elements > UINT32_MAX) {
        log_error("epoch_elements %llu exceeds uint32_t limit for gather LUT",
                  (unsigned long long)epoch_elements);
        goto Fail;
      }
    }

    // Compute LOD strides on host and upload
    CUdeviceptr d_lod_strides = 0;
    {
      uint64_t full_strides[LOD_MAX_NDIM];
      full_strides[rank - 1] = 1;
      for (int d = rank - 2; d >= 0; --d)
        full_strides[d] = full_strides[d + 1] * shape[d + 1];

      uint64_t lod_strides[LOD_MAX_NDIM];
      int li = p->lod_ndim - 1;
      for (int d = rank - 1; d >= 0; --d) {
        if ((p->lod_mask >> d) & 1) {
          lod_strides[li] = full_strides[d];
          li--;
        }
      }

      CU(Fail, cuMemAlloc(&d_lod_strides, p->lod_ndim * sizeof(uint64_t)));
      CU(Fail,
         cuMemcpyHtoD(
           d_lod_strides, lod_strides, p->lod_ndim * sizeof(uint64_t)));
    }

    // Build gather (inverse) LUT
    CU(Fail, cuMemAlloc(&s->lod.d_gather_lut, lod_count * sizeof(uint32_t)));
    lod_build_gather_lut(s->lod.d_gather_lut,
                         s->lod.d_lod_shape,
                         d_lod_strides,
                         p->lod_ndim,
                         p->lod_shapes[0],
                         lod_count,
                         0);

    cuMemFree(d_lod_strides);

    // Compute batch_offsets on host and upload
    {
      uint32_t* batch_offsets =
        (uint32_t*)calloc(p->batch_count, sizeof(uint32_t));
      CHECK(Fail, batch_offsets);

      uint64_t full_strides[LOD_MAX_NDIM];
      full_strides[rank - 1] = 1;
      for (int d = rank - 2; d >= 0; --d)
        full_strides[d] = full_strides[d + 1] * shape[d + 1];

      for (uint64_t bi = 0; bi < p->batch_count; ++bi) {
        uint64_t remainder = bi;
        uint64_t offset = 0;
        for (int k = p->batch_ndim - 1; k >= 0; --k) {
          uint64_t coord = remainder % p->batch_shape[k];
          remainder /= p->batch_shape[k];
          offset += coord * full_strides[p->batch_map[k]];
        }
        batch_offsets[bi] = (uint32_t)offset;
      }

      CU(
        Fail,
        cuMemAlloc(&s->lod.d_batch_offsets, p->batch_count * sizeof(uint32_t)));
      CU(Fail,
         cuMemcpyHtoD(s->lod.d_batch_offsets,
                      batch_offsets,
                      p->batch_count * sizeof(uint32_t)));
      free(batch_offsets);
    }
  }

  // Per-level device arrays for fill_ends
  for (int l = 0; l < s->lod.plan.nlod - 1; ++l) {
    struct lod_span seg = lod_segment(&s->lod.plan, l);
    uint64_t n_parents = lod_span_len(seg);

    CU(Fail,
       cuMemAlloc(&s->lod.d_child_shapes[l],
                  s->lod.plan.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(s->lod.d_child_shapes[l],
                    s->lod.plan.lod_shapes[l],
                    s->lod.plan.lod_ndim * sizeof(uint64_t)));

    CU(Fail,
       cuMemAlloc(&s->lod.d_parent_shapes[l],
                  s->lod.plan.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(s->lod.d_parent_shapes[l],
                    s->lod.plan.lod_shapes[l + 1],
                    s->lod.plan.lod_ndim * sizeof(uint64_t)));

    CU(Fail, cuMemAlloc(&s->lod.d_level_ends[l], n_parents * sizeof(uint64_t)));
  }

  // Per-level tile layouts
  {
    size_t alignment = codec_alignment(s->config.codec);

    for (int lv = 1; lv < s->lod.plan.nlod; ++lv) {
      struct stream_layout* lay = &s->lod.layouts[lv];
      const uint64_t* lv_shape = s->lod.plan.shapes[lv];

      lay->lifted_rank = 2 * rank;
      lay->tile_elements = 1;

      uint64_t tc[MAX_RANK / 2];
      for (int d = 0; d < rank; ++d) {
        tc[d] = ceildiv(lv_shape[d], dims[d].tile_size);
        lay->lifted_shape[2 * d] = tc[d];
        lay->lifted_shape[2 * d + 1] = dims[d].tile_size;
        lay->tile_elements *= dims[d].tile_size;
      }

      {
        size_t tile_bytes = lay->tile_elements * bpe;
        size_t padded_bytes = align_up(tile_bytes, alignment);
        lay->tile_stride = padded_bytes / bpe;
      }

      {
        int64_t n_stride = 1;
        int64_t t_stride = (int64_t)lay->tile_stride;
        for (int i = rank - 1; i >= 0; --i) {
          lay->lifted_strides[2 * i + 1] = n_stride;
          n_stride *= (int64_t)dims[i].tile_size;
          lay->lifted_strides[2 * i] = t_stride;
          t_stride *= (int64_t)tc[i];
        }
      }

      lay->tiles_per_epoch = lay->lifted_strides[0] / lay->tile_stride;
      lay->epoch_elements = lay->tiles_per_epoch * lay->tile_elements;
      lay->lifted_strides[0] = 0; // collapse epoch dim
      lay->tile_pool_bytes = lay->tiles_per_epoch * lay->tile_stride * bpe;

      // Upload lifted shape/strides to device
      {
        const size_t sb = lay->lifted_rank * sizeof(uint64_t);
        const size_t stb = lay->lifted_rank * sizeof(int64_t);
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lay->d_lifted_shape, sb));
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lay->d_lifted_strides, stb));
        CU(Fail,
           cuMemcpyHtoD(
             (CUdeviceptr)lay->d_lifted_shape, lay->lifted_shape, sb));
        CU(Fail,
           cuMemcpyHtoD(
             (CUdeviceptr)lay->d_lifted_strides, lay->lifted_strides, stb));
      }
    }
  }

  // Build morton-to-tile scatter LUTs for each level
  if (s->lod.plan.lod_ndim > 0) {
    struct lod_plan* p = &s->lod.plan;

    for (int lv = 0; lv < p->nlod; ++lv) {
      struct stream_layout* lay = (lv == 0) ? &s->layout : &s->lod.layouts[lv];
      uint64_t lod_count = p->lod_counts[lv];

      // Upload LOD shape to device (temporary for LUT building)
      CUdeviceptr d_lod_shape_lv = 0;
      if (lv == 0) {
        d_lod_shape_lv = s->lod.d_lod_shape;
      } else {
        const size_t lod_shape_bytes = p->lod_ndim * sizeof(uint64_t);
        CU(Fail, cuMemAlloc(&d_lod_shape_lv, lod_shape_bytes));
        CU(Fail,
           cuMemcpyHtoD(d_lod_shape_lv, p->lod_shapes[lv], lod_shape_bytes));
      }

      // Compute lod_tile_sizes and lod_tile_strides on host
      {
        uint64_t lod_tile_sizes[LOD_MAX_NDIM];
        int64_t lod_tile_strides[2 * LOD_MAX_NDIM];

        for (int li = 0; li < p->lod_ndim; ++li) {
          int d = p->lod_map[li];                            // full dim index
          lod_tile_sizes[li] = lay->lifted_shape[2 * d + 1]; // tile_size
          lod_tile_strides[2 * li] = lay->lifted_strides[2 * d]; // grid
          lod_tile_strides[2 * li + 1] =
            lay->lifted_strides[2 * d + 1]; // within
        }

        CUdeviceptr d_tile_sizes = 0, d_tile_strides = 0;
        CU(Fail2, cuMemAlloc(&d_tile_sizes, p->lod_ndim * sizeof(uint64_t)));
        CU(Fail2,
           cuMemAlloc(&d_tile_strides, 2 * p->lod_ndim * sizeof(int64_t)));
        CU(Fail2,
           cuMemcpyHtoD(
             d_tile_sizes, lod_tile_sizes, p->lod_ndim * sizeof(uint64_t)));
        CU(Fail2,
           cuMemcpyHtoD(d_tile_strides,
                        lod_tile_strides,
                        2 * p->lod_ndim * sizeof(int64_t)));

        // Allocate and build tile LUT
        CU(Fail2,
           cuMemAlloc(&s->lod.d_morton_tile_lut[lv],
                      lod_count * sizeof(uint32_t)));
        lod_build_tile_scatter_lut(s->lod.d_morton_tile_lut[lv],
                                   d_lod_shape_lv,
                                   d_tile_sizes,
                                   d_tile_strides,
                                   p->lod_ndim,
                                   p->lod_shapes[lv],
                                   lod_count,
                                   0);

        cuMemFree(d_tile_sizes);
        cuMemFree(d_tile_strides);
        goto Built;
      Fail2:
        cuMemFree(d_tile_sizes);
        cuMemFree(d_tile_strides);
        if (lv > 0)
          cuMemFree(d_lod_shape_lv);
        goto Fail;
      Built:;
      }

      if (lv > 0)
        cuMemFree(d_lod_shape_lv);

      // Compute batch_tile_offsets on host and upload
      {
        uint32_t* batch_offsets =
          (uint32_t*)calloc(p->batch_count, sizeof(uint32_t));
        CHECK(Fail, batch_offsets);

        for (uint64_t bi = 0; bi < p->batch_count; ++bi) {
          uint64_t remainder = bi;
          int64_t offset = 0;
          for (int k = p->batch_ndim - 1; k >= 0; --k) {
            uint64_t coord = remainder % p->batch_shape[k];
            remainder /= p->batch_shape[k];
            int d = p->batch_map[k]; // full dim index
            uint64_t tile_idx = coord / lay->lifted_shape[2 * d + 1];
            uint64_t within = coord % lay->lifted_shape[2 * d + 1];
            offset += (int64_t)tile_idx * lay->lifted_strides[2 * d];
            offset += (int64_t)within * lay->lifted_strides[2 * d + 1];
          }
          batch_offsets[bi] = (uint32_t)offset;
        }

        CU(Fail,
           cuMemAlloc(&s->lod.d_morton_batch_tile_offsets[lv],
                      p->batch_count * sizeof(uint32_t)));
        CU(Fail,
           cuMemcpyHtoD(s->lod.d_morton_batch_tile_offsets[lv],
                        batch_offsets,
                        p->batch_count * sizeof(uint32_t)));
        free(batch_offsets);
      }
    }
  }

  s->nlod = s->lod.plan.nlod;
  return 1;
Fail:
  return 0;
}

// init_lod_buffers: allocate d_linear, d_morton, LOD timing events.
// Must be called AFTER init_lod_layouts.
static int
init_lod_buffers(struct tile_stream_gpu* s)
{
  if (!s->enable_multiscale)
    return 1;

  const size_t bpe = s->config.bytes_per_element;

  // Allocate linear epoch buffer
  size_t linear_bytes = s->layout.epoch_elements * bpe;
  CHECK(Fail, (s->lod.d_linear = buffer_new(linear_bytes, device, 0)).data);

  // Allocate morton buffer (all levels packed)
  uint64_t total_vals = s->lod.plan.levels.ends[s->lod.plan.nlod - 1];
  size_t morton_bytes = total_vals * bpe;
  CHECK(Fail, (s->lod.d_morton = buffer_new(morton_bytes, device, 0)).data);

  CU(Fail, cuEventCreate(&s->lod.t_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_scatter_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_reduce_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_dim0_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_end, CU_EVENT_DEFAULT));

  return 1;
Fail:
  return 0;
}

// init_dim0_accums: allocate single contiguous accumulator for all LOD levels
// 1+, plus a level-ID buffer (u8 per element) and per-level count array.
// Must be called AFTER init_lod_layouts (needs lod.plan).
static int
init_dim0_accums(struct tile_stream_gpu* s)
{
  if (!s->dim0_downsample)
    return 1;

  const size_t bpe = s->config.bytes_per_element;
  struct lod_plan* p = &s->lod.plan;
  struct dim0_state* d0 = &s->dim0;

  // Morton offset: start of level 1 data in d_morton
  d0->morton_offset = p->levels.ends[0];

  // Total elements across all levels 1+ (batch_count * sum of lod_counts[1..])
  d0->total_elements = 0;
  for (int lv = 1; lv < p->nlod; ++lv)
    d0->total_elements += p->batch_count * p->lod_counts[lv];

  if (d0->total_elements == 0)
    return 1;

  // Accumulator buffer: wider type for mean (u32 for u16), native otherwise
  size_t accum_bpe =
    (s->dim0_reduce_method == lod_reduce_mean && bpe == 2) ? 4 : bpe;
  size_t accum_bytes = d0->total_elements * accum_bpe;
  CHECK(Fail, (d0->d_accum = buffer_new(accum_bytes, device, 0)).data);

  // Level-ID buffer: u8 per element, built on host and uploaded
  {
    uint8_t* h_level_ids = (uint8_t*)malloc(d0->total_elements);
    CHECK(Fail, h_level_ids);

    uint64_t offset = 0;
    for (int lv = 1; lv < p->nlod; ++lv) {
      uint64_t n = p->batch_count * p->lod_counts[lv];
      memset(h_level_ids + offset, (uint8_t)lv, n);
      offset += n;
    }

    CU(Fail2, cuMemAlloc(&d0->d_level_ids, d0->total_elements));
    CU(Fail2,
       cuMemcpyHtoD(d0->d_level_ids, h_level_ids, d0->total_elements));
    free(h_level_ids);
    goto LevelIdsDone;
  Fail2:
    free(h_level_ids);
    goto Fail;
  LevelIdsDone:;
  }

  // Per-level counts array on device (nlod entries, u32)
  {
    CU(Fail,
       cuMemAlloc(&d0->d_counts, (uint64_t)p->nlod * sizeof(uint32_t)));
    memset(d0->counts, 0, sizeof(d0->counts));
    CU(Fail,
       cuMemcpyHtoD(
         d0->d_counts, d0->counts, (uint64_t)p->nlod * sizeof(uint32_t)));
  }

  return 1;
Fail:
  return 0;
}

static int
seed_events(struct tile_stream_gpu* s)
{
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(s->stage.slot[i].h_in.ready, s->compute));
    CU(Fail, cuEventRecord(s->stage.slot[i].t_h2d_start, s->compute));
    CU(Fail, cuEventRecord(s->stage.slot[i].t_scatter_start, s->compute));
    CU(Fail, cuEventRecord(s->stage.slot[i].t_scatter_end, s->compute));
  }
  CU(Fail, cuEventRecord(s->pools.buf[0].ready, s->compute));
  CU(Fail, cuEventRecord(s->pools.buf[1].ready, s->compute));
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(s->flush[i].t_compress_start, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].t_aggregate_end, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].t_d2h_start, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].ready, s->compute));
  }

  if (s->lod.t_start) {
    CU(Fail, cuEventRecord(s->lod.t_start, s->compute));
    CU(Fail, cuEventRecord(s->lod.t_scatter_end, s->compute));
    CU(Fail, cuEventRecord(s->lod.t_reduce_end, s->compute));
    CU(Fail, cuEventRecord(s->lod.t_dim0_end, s->compute));
    CU(Fail, cuEventRecord(s->lod.t_end, s->compute));
  }

  return 1;
Fail:
  return 0;
}

int
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct tile_stream_gpu* out)
{
  CHECK(Fail, config);
  CHECK(Fail, out);
  CHECK(Fail, config->bytes_per_element > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= MAX_RANK / 2);
  CHECK(Fail, config->dimensions);
  CHECK(Fail, config->shard_sink);

  // Compute lod_mask and enable_multiscale from dimensions.
  // Dim 0 downsample is handled separately (temporal accumulation),
  // NOT included in the spatial lod_mask.
  uint32_t lod_mask = 0;
  int dim0_downsample = 0;
  for (int d = 0; d < config->rank; ++d) {
    if (config->dimensions[d].downsample) {
      if (d == 0) {
        dim0_downsample = 1;
        // Validate dim0 reduce method: only mean/min/max supported
        enum lod_reduce_method m = config->dim0_reduce_method;
        if (m != lod_reduce_mean && m != lod_reduce_min &&
            m != lod_reduce_max) {
          log_error("dim0 reduce method must be mean, min, or max");
          goto Fail;
        }
      } else {
        lod_mask |= (1u << d);
      }
    }
  }
  // dim0 downsampling requires at least one spatial dim also downsampled
  if (dim0_downsample && lod_mask == 0) {
    log_error("dim0 downsample requires at least one spatial dim downsampled");
    goto Fail;
  }
  int enable_multiscale = lod_mask != 0;

  *out = (struct tile_stream_gpu){
    .writer = { .append = tile_stream_gpu_append,
                .flush = tile_stream_gpu_flush },
    .dispatch =
      enable_multiscale ? dispatch_scatter_multiscale : dispatch_scatter,
    .config = *config,
    .nlod = 1,
    .enable_multiscale = enable_multiscale,
    .lod_mask = lod_mask,
    .dim0_downsample = dim0_downsample,
    .dim0_reduce_method = config->dim0_reduce_method,
  };

  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  CHECK(Fail, init_cuda_streams_and_events(out));
  CHECK(Fail, init_l0_layout(out));
  CHECK(Fail, init_staging_buffers(out));
  CHECK(Fail, init_lod_layouts(out));

  // Compute epochs_per_batch (K) after lod_layouts sets total_tiles/nlod.
  // nlod is set by init_lod_layouts; total_tiles is set by init_tile_pools.
  // We need total_tiles to compute K, but init_tile_pools needs K for sizing.
  // Compute per-epoch total_tiles here (same logic as init_tile_pools).
  {
    uint64_t total_tiles_per_epoch = out->layout.tiles_per_epoch;
    for (int lv = 1; lv < out->nlod; ++lv)
      total_tiles_per_epoch += out->lod.layouts[lv].tiles_per_epoch;

    uint32_t K = config->epochs_per_batch;
    if (K == 0) {
      uint32_t target = config->target_min_tiles;
      if (target == 0)
        target = 1024;
      K = (uint32_t)ceildiv(target, total_tiles_per_epoch);
      K = next_pow2_u32(K);
    }
    if (K > MAX_BATCH_EPOCHS)
      K = MAX_BATCH_EPOCHS;
    // Ensure K is a power of 2
    CHECK(Fail, (K & (K - 1)) == 0);
    out->epochs_per_batch = K;
    out->epochs_accumulated = 0;
    log_info("epochs_per_batch K=%u (total_tiles/epoch=%llu)",
             K,
             (unsigned long long)total_tiles_per_epoch);
  }

  CHECK(Fail, init_tile_pools(out));
  CHECK(Fail, init_compression(out));
  CHECK(Fail, init_aggregate_and_shards(out));
  CHECK(Fail, init_batch_luts(out));
  CHECK(Fail, init_batch_events(out));
  CHECK(Fail, init_lod_buffers(out));
  CHECK(Fail, init_dim0_accums(out));
  CHECK(Fail, seed_events(out));

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics = (struct stream_metrics){
    .memcpy = { .name = "Memcpy", .best_ms = 1e30f },
    .h2d = { .name = "H2D", .best_ms = 1e30f },
    .scatter = { .name = out->enable_multiscale ? "Copy" : "Scatter",
                 .best_ms = 1e30f },
    .lod_gather = { .name = "LOD Gather", .best_ms = 1e30f },
    .lod_reduce = { .name = "LOD Reduce", .best_ms = 1e30f },
    .lod_dim0_fold = { .name = "Dim0 Fold", .best_ms = 1e30f },
    .lod_morton_tile = { .name = "LOD to tiles", .best_ms = 1e30f },
    .compress = { .name = "Compress", .best_ms = 1e30f },
    .aggregate = { .name = "Aggregate", .best_ms = 1e30f },
    .d2h = { .name = "D2H", .best_ms = 1e30f },
    .sink = { .name = "Sink", .best_ms = 1e30f },
  };

  return 1;

Fail:
  tile_stream_gpu_destroy(out);
  return 0;
}

static struct writer_result
tile_stream_gpu_append(struct writer* self, struct slice input)
{
  struct tile_stream_gpu* s =
    container_of(self, struct tile_stream_gpu, writer);
  const size_t bpe = s->config.bytes_per_element;
  const size_t buffer_capacity = s->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  while (src < end) {
    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    const uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    const uint64_t bytes_this_pass = elements_this_pass * bpe;

    {
      uint64_t written = 0;
      while (written < bytes_this_pass) {
        const size_t space = buffer_capacity - s->stage.bytes_written;
        const uint64_t remaining = bytes_this_pass - written;
        const size_t payload = space < remaining ? space : (size_t)remaining;

        if (s->stage.bytes_written == 0) {
          const int si = s->stage.current;
          struct staging_slot* ss = &s->stage.slot[si];
          CU(Error, cuEventSynchronize(ss->h_in.ready));

          if (s->cursor > 0) {
            accumulate_metric_cu(&s->metrics.h2d,
                                 ss->t_h2d_start,
                                 ss->h_in.ready,
                                 ss->dispatched_bytes);
            accumulate_metric_cu(&s->metrics.scatter,
                                 ss->t_scatter_start,
                                 ss->t_scatter_end,
                                 ss->dispatched_bytes);
          }
        }

        {
          struct platform_clock mc = { 0 };
          platform_toc(&mc);
          memcpy((uint8_t*)s->stage.slot[s->stage.current].h_in.data +
                   s->stage.bytes_written,
                 src + written,
                 payload);
          accumulate_metric_ms(
            &s->metrics.memcpy, (float)(platform_toc(&mc) * 1000.0), payload);
        }
        s->stage.bytes_written += payload;
        written += payload;

        if (s->stage.bytes_written == buffer_capacity ||
            written == bytes_this_pass) {
          if (!s->dispatch(s))
            goto Error;
          s->stage.bytes_written = 0;
        }
      }
    }
    src += bytes_this_pass;

    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = accumulate_epoch_or_flush(s);
      if (fr.error)
        return writer_error_at(src, end);
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

static struct writer_result
tile_stream_gpu_flush(struct writer* self)
{
  struct tile_stream_gpu* s =
    container_of(self, struct tile_stream_gpu, writer);

  if (s->stage.bytes_written > 0) {
    if (!s->dispatch(s))
      return writer_error();
    s->stage.bytes_written = 0;
  }

  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  // Flush any partial epoch first (sub-epoch data)
  if (s->cursor % s->layout.epoch_elements != 0 || s->cursor == 0) {
    // run_lod + record pool event + increment epochs_accumulated
    if (!run_lod(s))
      return writer_error();
    CU(Error,
       cuEventRecord(s->batch_pool_events[s->epochs_accumulated], s->compute));
    s->epochs_accumulated++;
  }

  // Flush any accumulated epochs (partial batch)
  r = flush_accumulated_sync(s);
  if (r.error)
    return r;

  // Drain any partial dim0 accumulators
  r = flush_partial_dim0(s);
  if (r.error)
    return r;

  // Emit partial shards for all levels
  for (int lv = 0; lv < s->nlod; ++lv) {
    if (s->lod_levels[lv].shard.epoch_in_shard > 0) {
      if (!emit_shards(&s->lod_levels[lv].shard))
        return writer_error();
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

// --- Memory estimate ---

int
tile_stream_gpu_memory_estimate(const struct tile_stream_configuration* config,
                                struct tile_stream_memory_info* info)
{
  if (!config || !info)
    return 1;
  if (config->bytes_per_element == 0)
    return 1;
  if (config->buffer_capacity_bytes == 0)
    return 1;
  if (config->rank == 0 || config->rank > MAX_RANK / 2)
    return 1;
  if (!config->dimensions)
    return 1;

  memset(info, 0, sizeof(*info));

  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const struct dimension* dims = config->dimensions;
  const size_t buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  // --- L0 layout math (mirrors init_l0_layout) ---

  uint64_t tile_elements = 1;
  uint64_t tile_count[MAX_RANK / 2];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    tile_elements *= dims[i].tile_size;
  }

  const size_t alignment = codec_alignment(config->codec);
  const size_t tile_bytes = tile_elements * bpe;
  const size_t padded_bytes = align_up(tile_bytes, alignment);
  const uint64_t tile_stride = padded_bytes / bpe;

  uint64_t tiles_per_epoch = 1;
  for (int d = 1; d < rank; ++d)
    tiles_per_epoch *= tile_count[d];

  const uint64_t epoch_elements = tiles_per_epoch * tile_elements;

  // --- LOD plan (CPU only) ---

  uint32_t lod_mask = 0;
  int dim0_ds = 0;
  for (int d = 0; d < rank; ++d) {
    if (dims[d].downsample) {
      if (d == 0)
        dim0_ds = 1;
      else
        lod_mask |= (1u << d);
    }
  }
  const int enable_multiscale = lod_mask != 0;

  struct lod_plan plan = { 0 };
  int nlod = 1;

  if (enable_multiscale) {
    uint64_t shape[MAX_RANK / 2];
    uint64_t tile_shape[MAX_RANK / 2];
    shape[0] = dims[0].tile_size;
    for (int d = 1; d < rank; ++d)
      shape[d] = dims[d].size;
    for (int d = 0; d < rank; ++d)
      tile_shape[d] = dims[d].tile_size;

    if (!lod_plan_init(
          &plan, rank, shape, tile_shape, (uint8_t)lod_mask, LOD_MAX_LEVELS))
      return 1;

    nlod = plan.nlod;
  }

  // --- Per-level tile counts (mirrors init_tile_pools) ---

  uint64_t level_tile_count[LOD_MAX_LEVELS];
  memset(level_tile_count, 0, sizeof(level_tile_count));
  level_tile_count[0] = tiles_per_epoch;
  uint64_t total_tiles = tiles_per_epoch;

  for (int lv = 1; lv < nlod; ++lv) {
    const uint64_t* lv_shape = plan.shapes[lv];
    uint64_t lv_tiles = 1;
    for (int d = 1; d < rank; ++d)
      lv_tiles *= ceildiv(lv_shape[d], dims[d].tile_size);
    level_tile_count[lv] = lv_tiles;
    total_tiles += lv_tiles;
  }

  // --- Compute K (mirrors create path) ---

  uint32_t K = config->epochs_per_batch;
  if (K == 0) {
    uint32_t target = config->target_min_tiles;
    if (target == 0)
      target = 1024;
    K = (uint32_t)ceildiv(target, total_tiles);
    K = next_pow2_u32(K);
  }
  if (K > MAX_BATCH_EPOCHS)
    K = MAX_BATCH_EPOCHS;

  // --- Codec queries (no GPU allocation) ---

  const size_t chunk_bytes = tile_stride * bpe;
  const size_t max_output_size =
    codec_max_output_size(config->codec, chunk_bytes);
  if (config->codec != CODEC_NONE && max_output_size == 0)
    goto Fail;

  const uint64_t codec_batch = (uint64_t)K * total_tiles;
  const size_t nvcomp_temp =
    codec_temp_bytes(config->codec, chunk_bytes, codec_batch);

  // --- Tally: staging (device + host pinned) ---

  const size_t staging_bytes = 2 * buffer_capacity_bytes;
  const size_t staging_host = 2 * buffer_capacity_bytes;

  // --- Tally: tile pools (device) ---

  const size_t tile_pool_bytes = 2 * (uint64_t)K * total_tiles * tile_stride * bpe;

  // --- Tally: compressed pool in flush slots (device) ---

  const size_t compressed_pool_bytes =
    2 * (uint64_t)K * total_tiles * max_output_size;

  // --- Tally: codec device arrays ---

  size_t codec_bytes = 0;
  codec_bytes += codec_batch * sizeof(size_t); // d_comp_sizes
  codec_bytes += codec_batch * sizeof(size_t); // d_uncomp_sizes
  if (config->codec != CODEC_NONE)
    codec_bytes += 2 * codec_batch * sizeof(void*); // d_ptrs
  codec_bytes += nvcomp_temp;                       // d_temp

  // --- Tally: aggregate (device + host pinned) ---

  size_t aggregate_device = 0;
  size_t aggregate_host = 0;

  for (int lv = 0; lv < nlod; ++lv) {
    uint64_t tc[MAX_RANK / 2];
    uint64_t tps[MAX_RANK / 2];

    if (lv == 0) {
      for (int d = 0; d < rank; ++d) {
        tc[d] = tile_count[d];
        tps[d] =
          (dims[d].tiles_per_shard == 0) ? tc[d] : dims[d].tiles_per_shard;
      }
    } else {
      const uint64_t* lv_shape = plan.shapes[lv];
      for (int d = 0; d < rank; ++d) {
        tc[d] = ceildiv(lv_shape[d], dims[d].tile_size);
        tps[d] =
          (dims[d].tiles_per_shard == 0) ? tc[d] : dims[d].tiles_per_shard;
      }
    }

    // covering_count = prod(ceildiv(tc[d], tps[d]) * tps[d]) for d=1..rank-1
    uint64_t covering_count = 1;
    for (int d = 1; d < rank; ++d)
      covering_count *= ceildiv(tc[d], tps[d]) * tps[d];

    uint32_t batch_count = K;
    if (dim0_ds && lv > 0) {
      uint32_t period = 1u << lv;
      batch_count = (K >= period) ? K / period : 1;
    }

    uint64_t batch_tiles = (uint64_t)batch_count * level_tile_count[lv];
    uint64_t batch_covering = (uint64_t)batch_count * covering_count;
    size_t batch_agg_bytes = batch_tiles * max_output_size;

    // aggregate_layout: device lifted shape + strides
    size_t agg_layout_dev =
      2 * (rank - 1) * sizeof(uint64_t) + 2 * (rank - 1) * sizeof(int64_t);

    // aggregate_slot x 2: device arrays (batch-sized)
    size_t slot_dev = (batch_covering + 1) * sizeof(size_t)   // d_permuted_sizes
                      + (batch_covering + 1) * sizeof(size_t) // d_offsets
                      + batch_tiles * sizeof(uint32_t)        // d_perm
                      + batch_agg_bytes;                      // d_aggregated

    // aggregate_slot x 2: host pinned (batch-sized)
    size_t slot_host = batch_agg_bytes                            // h_aggregated
                       + (batch_covering + 1) * sizeof(size_t);   // h_offsets

    // Batch LUTs (per level, if K > 1)
    size_t lut_dev = 0;
    if (K > 1)
      lut_dev = batch_tiles * sizeof(uint32_t) * 2; // gather + perm

    aggregate_device += agg_layout_dev + 2 * slot_dev + lut_dev;
    aggregate_host += 2 * slot_host;
  }

  // --- Tally: LOD buffers + shape arrays (device) ---

  size_t lod_device = 0;

  // L0 layout arrays (always present)
  lod_device += 2 * rank * sizeof(uint64_t); // d_lifted_shape
  lod_device += 2 * rank * sizeof(int64_t);  // d_lifted_strides

  if (enable_multiscale) {
    // d_linear + d_morton
    lod_device += epoch_elements * bpe;
    uint64_t total_lod_vals = plan.levels.ends[plan.nlod - 1];
    lod_device += total_lod_vals * bpe;

    // Global shape arrays
    lod_device += rank * sizeof(uint64_t); // d_full_shape
    if (plan.lod_ndim > 0)
      lod_device += plan.lod_ndim * sizeof(uint64_t); // d_lod_shape

    // Gather LUT + batch offsets
    if (plan.lod_ndim > 0) {
      lod_device += plan.lod_counts[0] * sizeof(uint32_t); // d_gather_lut
      lod_device += plan.batch_count * sizeof(uint32_t);   // d_batch_offsets
    }

    // Per reduce-level arrays (0..nlod-2)
    for (int l = 0; l < plan.nlod - 1; ++l) {
      lod_device += plan.lod_ndim * sizeof(uint64_t); // d_child_shapes
      lod_device += plan.lod_ndim * sizeof(uint64_t); // d_parent_shapes

      struct lod_span seg = lod_segment(&plan, l);
      uint64_t n_parents = lod_span_len(seg);
      lod_device += n_parents * sizeof(uint64_t); // d_level_ends
    }

    // Per LOD level (1..nlod-1): layout + shape arrays
    for (int lv = 1; lv < plan.nlod; ++lv) {
      lod_device += 2 * rank * sizeof(uint64_t); // d_lifted_shape
      lod_device += 2 * rank * sizeof(int64_t);  // d_lifted_strides
    }

    // Dim0 accumulator: single buffer + level-ID LUT + counts
    if (dim0_ds) {
      size_t accum_bpe =
        (config->dim0_reduce_method == lod_reduce_mean && bpe == 2) ? 4 : bpe;
      uint64_t total_elems = 0;
      for (int lv = 1; lv < plan.nlod; ++lv)
        total_elems += plan.batch_count * plan.lod_counts[lv];
      lod_device += total_elems * accum_bpe;          // d_accum
      lod_device += total_elems;                       // d_level_ids (u8)
      lod_device += (uint64_t)plan.nlod * sizeof(uint32_t); // d_counts
    }
  }

  // --- Sum totals ---

  info->staging_bytes = staging_bytes;
  info->tile_pool_bytes = tile_pool_bytes;
  info->compressed_pool_bytes = compressed_pool_bytes;
  info->aggregate_bytes = aggregate_device;
  info->lod_bytes = lod_device;
  info->codec_bytes = codec_bytes;

  info->device_bytes = staging_bytes + tile_pool_bytes + compressed_pool_bytes +
                       aggregate_device + lod_device + codec_bytes;
  info->host_pinned_bytes = staging_host + aggregate_host;

  info->tiles_per_epoch = tiles_per_epoch;
  info->total_tiles = total_tiles;
  info->max_output_size = max_output_size;
  info->nlod = nlod;
  info->epochs_per_batch = K;

  if (enable_multiscale)
    lod_plan_free(&plan);

  return 0;

Fail:
  if (enable_multiscale)
    lod_plan_free(&plan);
  return 1;
}

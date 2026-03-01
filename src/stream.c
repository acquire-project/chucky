#include "compress.h"
#include "lod.h"
#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"

#include <stdlib.h>
#include <string.h>

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

// FIXME: don't need this function
static void
device_free(void* ptr)
{
  CUWARN(cuMemFree((CUdeviceptr)ptr));
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

// Return pointer to the current L0 tile pool.
static inline void*
current_pool(struct tile_stream_gpu* s)
{
  return dbuf_current(&s->pools);
}

// H2D transfer + scatter into tile pool.
// Returns 0 on success, 1 on error.
static int
dispatch_scatter(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 0;

  const uint64_t elements = s->stage.bytes_written / bpe;
  if (elements == 0)
    return 0;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];
  void* pool = current_pool(s);

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
            bpe,
            s->cursor,
            s->layout.lifted_rank,
            s->layout.d_lifted_shape,
            s->layout.d_lifted_strides,
            s->compute);
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));

  CU(Error,
     cuEventRecord(s->pools.buf[s->pools.current].ready, s->compute));

  s->cursor += elements;
  s->stage.current ^= 1;
  return 0;

Error:
  return 1;
}

// H2D transfer + copy to linear epoch buffer for LOD.
// L0 tiling is deferred to run_lod (lod_morton_to_tiles at lv=0).
// Returns 0 on success, 1 on error.
static int
dispatch_scatter_multiscale(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 0;

  const uint64_t elements = s->stage.bytes_written / bpe;
  if (elements == 0)
    return 0;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];

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
       cuMemcpyDtoDAsync((CUdeviceptr)s->d_linear.data + epoch_offset,
                         (CUdeviceptr)ss->d_in.data,
                         elements * bpe,
                         s->compute));
  }
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));

  s->cursor += elements;
  s->stage.current ^= 1;
  return 0;

Error:
  return 1;
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
  return 0;

Error:
  return 1;
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

  return 0;

Error:
  return 1;
}

// Wait for D2H on the given flush slot, record timing, deliver to sinks.
static struct writer_result
wait_and_deliver(struct tile_stream_gpu* s, int fc)
{
  struct flush_slot_gpu* fs = &s->flush[fc];

  CU(Error, cuEventSynchronize(fs->ready));

  // LOD metrics: safe to read now — compress stream waited on t_lod_end
  if (s->config.enable_multiscale && s->t_lod_start) {
    accumulate_metric_cu(
      &s->metrics.lod_scatter, s->t_lod_start, s->t_lod_scatter_end);
    accumulate_metric_cu(
      &s->metrics.lod_reduce, s->t_lod_scatter_end, s->t_lod_reduce_end);
    accumulate_metric_cu(
      &s->metrics.lod_m2t, s->t_lod_reduce_end, s->t_lod_end);
  }

  accumulate_metric_cu(&s->metrics.compress,
                       fs->t_compress_start,
                       s->flush[fc].d_compressed.ready);
  accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start, fs->ready);
  s->metrics.d2h.total_bytes += s->codec.pool_bytes;

  struct aggregate_slot* agg = &s->agg[fc];
  if (deliver_to_shards(s, 0, &s->shard, agg))
    goto Error;

  // Deliver LOD level shards
  if (s->config.enable_multiscale && s->d_lod_tiles.data) {
    for (int lv = 1; lv < s->lod.nlev; ++lv) {
      struct aggregate_slot* lod_agg = &s->lod_levels[lv].agg_slot;
      CU(Error, cuEventSynchronize(lod_agg->ready));
      if (deliver_to_shards(
            s, (uint8_t)lv, &s->lod_levels[lv].shard, lod_agg))
        goto Error;
    }
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

// --- Epoch flush pipeline ---

// Kick compress + aggregate + D2H for the current epoch.
// fc: flush slot index (0 or 1, matches pools.current before swap).
static int
kick_epoch(struct tile_stream_gpu* s, int fc)
{
  struct flush_slot_gpu* fs = &s->flush[fc];
  const uint64_t M0 = s->layout.tiles_per_epoch;

  // Wait for scatter to finish
  CU(Error,
     cuStreamWaitEvent(s->compress, s->pools.buf[fc].ready, 0));

  CU(Error, cuEventRecord(fs->t_compress_start, s->compress));

  // Compress (or copy for CODEC_NONE)
  if (s->config.codec == CODEC_ZSTD) {
    CHECK(
      Error,
      compress_batch_async((const void* const*)fs->d_uncomp_ptrs,
                           s->codec.d_uncomp_sizes,
                           s->layout.tile_stride * s->config.bytes_per_element,
                           M0,
                           s->codec.d_comp_temp,
                           s->codec.comp_temp_bytes,
                           (void**)fs->d_comp_ptrs,
                           s->codec.d_comp_sizes,
                           s->compress) == 0);
  } else {
    // CODEC_NONE: DtoD copy tiles into compressed buffer, set sizes = tile_bytes
    const size_t tile_bytes = s->layout.tile_stride * s->config.bytes_per_element;
    CU(Error,
       cuMemcpyDtoDAsync((CUdeviceptr)fs->d_compressed.data,
                         (CUdeviceptr)s->pools.buf[fc].data,
                         M0 * tile_bytes,
                         s->compress));
    // d_comp_sizes already set to tile_bytes during init
  }

  CU(Error, cuEventRecord(fs->d_compressed.ready, s->compress));

  // Aggregate + D2H
  CU(Error, cuStreamWaitEvent(s->d2h, fs->d_compressed.ready, 0));
  CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));

  struct aggregate_slot* agg = &s->agg[fc];
  CHECK(Error,
        aggregate_by_shard_async(&s->agg_layout,
                                 fs->d_compressed.data,
                                 s->codec.d_comp_sizes,
                                 agg,
                                 s->compress) == 0);
  CU(Error, cuEventRecord(agg->ready, s->compress));

  CU(Error, cuStreamWaitEvent(s->d2h, agg->ready, 0));
  CU(Error,
     cuMemcpyDtoHAsync(agg->h_aggregated,
                       (CUdeviceptr)agg->d_aggregated,
                       s->codec.pool_bytes,
                       s->d2h));
  CU(Error,
     cuMemcpyDtoHAsync(agg->h_offsets,
                       (CUdeviceptr)agg->d_offsets,
                       (s->agg_layout.covering_count + 1) * sizeof(size_t),
                       s->d2h));
  CU(Error, cuEventRecord(agg->ready, s->d2h));

  // LOD levels: compress + aggregate + D2H per level
  if (s->config.enable_multiscale && s->d_lod_tiles.data) {
    CU(Error, cuStreamWaitEvent(s->compress, s->t_lod_end, 0));

    for (int lv = 1; lv < s->lod.nlev; ++lv) {
      struct stream_layout* lay = &s->lod_layouts[lv];
      uint64_t M_lv = lay->tiles_per_epoch;
      size_t tile_bytes_lv = lay->tile_stride * s->config.bytes_per_element;

      if (s->config.codec == CODEC_ZSTD) {
        CHECK(Error,
              compress_batch_async(
                (const void* const*)s->lod_levels[lv].d_uncomp_ptrs,
                s->lod_levels[lv].codec.d_uncomp_sizes,
                tile_bytes_lv,
                M_lv,
                s->lod_levels[lv].codec.d_comp_temp,
                s->lod_levels[lv].codec.comp_temp_bytes,
                (void**)s->lod_levels[lv].d_comp_ptrs,
                s->lod_levels[lv].codec.d_comp_sizes,
                s->compress) == 0);
      } else {
        // CODEC_NONE: DtoD copy
        uint64_t tile_offset = 0;
        for (int prev = 1; prev < lv; ++prev)
          tile_offset +=
            s->lod_layouts[prev].tiles_per_epoch * s->lod_layouts[prev].tile_stride;
        CU(Error,
           cuMemcpyDtoDAsync(
             (CUdeviceptr)s->lod_levels[lv].d_compressed.data,
             (CUdeviceptr)s->d_lod_tiles.data +
               tile_offset * s->config.bytes_per_element,
             M_lv * tile_bytes_lv,
             s->compress));
      }

      CU(Error,
         cuEventRecord(s->lod_levels[lv].d_compressed.ready, s->compress));

      CU(Error,
         cuStreamWaitEvent(s->d2h, s->lod_levels[lv].d_compressed.ready, 0));

      struct aggregate_slot* lod_agg = &s->lod_levels[lv].agg_slot;
      CHECK(Error,
            aggregate_by_shard_async(&s->lod_levels[lv].agg_layout,
                                     s->lod_levels[lv].d_compressed.data,
                                     s->lod_levels[lv].codec.d_comp_sizes,
                                     lod_agg,
                                     s->compress) == 0);
      CU(Error, cuEventRecord(lod_agg->ready, s->compress));

      CU(Error, cuStreamWaitEvent(s->d2h, lod_agg->ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(lod_agg->h_aggregated,
                           (CUdeviceptr)lod_agg->d_aggregated,
                           s->lod_levels[lv].codec.pool_bytes,
                           s->d2h));
      CU(Error,
         cuMemcpyDtoHAsync(lod_agg->h_offsets,
                           (CUdeviceptr)lod_agg->d_offsets,
                           (s->lod_levels[lv].agg_layout.covering_count + 1) *
                             sizeof(size_t),
                           s->d2h));
      CU(Error, cuEventRecord(lod_agg->ready, s->d2h));
    }
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
  if (!s->config.enable_multiscale || !s->d_linear.data)
    return 1;

  struct lod_plan* p = &s->lod;
  const size_t bpe = s->config.bytes_per_element;
  enum lod_dtype dtype = (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

  uint64_t n_elements = lod_span_len(lod_spans_at(&p->levels, 0));

  CU(Error, cuEventRecord(s->t_lod_start, s->compute));

  lod_scatter((CUdeviceptr)s->d_morton.data,
              (CUdeviceptr)s->d_linear.data,
              dtype,
              p->ndim,
              n_elements,
              s->d_lod_full_shape,
              s->d_lod_shape,
              p->lod_ndim,
              p->lod_shapes[0],
              p->lod_mask,
              p->lod_counts[0],
              s->compute);

  CU(Error, cuEventRecord(s->t_lod_scatter_end, s->compute));

  for (int l = 0; l < p->nlev - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t n_parents = lod_span_len(seg);

    lod_fill_ends_gpu(s->d_lod_level_ends[l],
                      p->lod_ndim,
                      s->d_lod_child_shapes[l],
                      s->d_lod_parent_shapes[l],
                      p->lod_shapes[l],
                      p->lod_shapes[l + 1],
                      n_parents,
                      s->compute);

    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    lod_reduce((CUdeviceptr)s->d_morton.data,
               s->d_lod_level_ends[l],
               dtype,
               src_level.beg,
               dst_level.beg,
               p->lod_counts[l],
               p->lod_counts[l + 1],
               p->batch_count,
               s->compute);
  }

  CU(Error, cuEventRecord(s->t_lod_reduce_end, s->compute));

  // L0: morton-to-tile scatter into the tile pool
  {
    struct lod_span lev0 = lod_spans_at(&p->levels, 0);
    void* pool = current_pool(s);

    lod_morton_to_tiles((CUdeviceptr)pool,
                        (CUdeviceptr)s->d_morton.data + lev0.beg * bpe,
                        &s->m2t_l0,
                        s->compute);
  }

  // Signal L0 pool is ready for compress
  CU(Error,
     cuEventRecord(s->pools.buf[s->pools.current].ready, s->compute));

  // LOD levels 1+: morton-to-tile scatter into unified tile buffer
  if (s->d_lod_tiles.data) {
    uint64_t tile_offset = 0;

    for (int lv = 1; lv < p->nlev; ++lv) {
      struct stream_layout* lay = &s->lod_layouts[lv];
      struct lod_span lev = lod_spans_at(&p->levels, lv);

      lod_morton_to_tiles((CUdeviceptr)s->d_lod_tiles.data + tile_offset * bpe,
                          (CUdeviceptr)s->d_morton.data + lev.beg * bpe,
                          &s->m2t_levels[lv],
                          s->compute);

      tile_offset += lay->tiles_per_epoch * lay->tile_stride;
    }
  }

  CU(Error, cuEventRecord(s->t_lod_end, s->compute));
  return 1;

Error:
  return 0;
}

// Flush the current epoch's tile pool: compress, D2H, swap.
static struct writer_result
flush_epoch(struct tile_stream_gpu* s)
{
  const int fc = s->pools.current;

  // Deliver the previous epoch if its D2H is still in flight
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  // Run LOD pipeline: populates L0 pool tiles + LOD level tiles
  if (!run_lod(s))
    return writer_error();

  if (!kick_epoch(s, fc))
    return writer_error();

  // Swap pool and zero next L0 region
  dbuf_swap(&s->pools);
  void* next = current_pool(s);
  CU(Error,
     cuMemsetD8Async(
       (CUdeviceptr)next, 0, s->layout.tile_pool_bytes, s->compute));

  s->flush_pending = 1;
  s->flush_current = fc;
  return writer_ok();

Error:
  return writer_error();
}

// Synchronously flush the current tile pool (used for the final partial epoch).
static struct writer_result
flush_epoch_sync(struct tile_stream_gpu* s)
{
  const int fc = s->pools.current;
  if (!run_lod(s))
    return writer_error();
  if (!kick_epoch(s, fc))
    return writer_error();
  return wait_and_deliver(s, fc);
}

struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s)
{
  return s->metrics;
}

// --- Destroy ---

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

  // Flush slots
  for (int i = 0; i < 2; ++i) {
    struct flush_slot_gpu* fs = &stream->flush[i];
    buffer_free(&fs->d_compressed);
    device_free(fs->d_uncomp_ptrs);
    device_free(fs->d_comp_ptrs);
    CUWARN(cuEventDestroy(fs->t_compress_start));
    CUWARN(cuEventDestroy(fs->t_d2h_start));
    CUWARN(cuEventDestroy(fs->ready));
  }

  // Shared compress state
  device_free(stream->codec.d_comp_sizes);
  device_free(stream->codec.d_uncomp_sizes);
  device_free(stream->codec.d_comp_temp);

  // L0 aggregate + shard
  aggregate_layout_destroy(&stream->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&stream->agg[i]);

  if (stream->shard.shards) {
    for (uint64_t i = 0; i < stream->shard.shard_inner_count; ++i)
      free(stream->shard.shards[i].index);
    free(stream->shard.shards);
  }

  // LOD cleanup
  buffer_free(&stream->d_linear);
  buffer_free(&stream->d_morton);
  buffer_free(&stream->d_lod_tiles);
  CUWARN(cuMemFree(stream->d_lod_full_shape));
  CUWARN(cuMemFree(stream->d_lod_shape));
  CUWARN(cuMemFree(stream->d_lod_ends));
  for (int i = 0; i < stream->lod.nlev - 1; ++i) {
    CUWARN(cuMemFree(stream->d_lod_child_shapes[i]));
    CUWARN(cuMemFree(stream->d_lod_parent_shapes[i]));
    CUWARN(cuMemFree(stream->d_lod_level_ends[i]));
  }
  for (int i = 1; i < stream->lod.nlev; ++i) {
    CUWARN(cuMemFree(stream->d_lod_lv_full_shapes[i]));
    CUWARN(cuMemFree(stream->d_lod_lv_lod_shapes[i]));
    CUWARN(cuMemFree((CUdeviceptr)stream->lod_layouts[i].d_lifted_shape));
    CUWARN(cuMemFree((CUdeviceptr)stream->lod_layouts[i].d_lifted_strides));

    // Per-level compress + aggregate + shard cleanup
    buffer_free(&stream->lod_levels[i].d_compressed);
    device_free(stream->lod_levels[i].d_uncomp_ptrs);
    device_free(stream->lod_levels[i].d_comp_ptrs);
    device_free(stream->lod_levels[i].codec.d_comp_sizes);
    device_free(stream->lod_levels[i].codec.d_uncomp_sizes);
    device_free(stream->lod_levels[i].codec.d_comp_temp);
    aggregate_layout_destroy(&stream->lod_levels[i].agg_layout);
    aggregate_slot_destroy(&stream->lod_levels[i].agg_slot);
    if (stream->lod_levels[i].shard.shards) {
      for (uint64_t j = 0; j < stream->lod_levels[i].shard.shard_inner_count;
           ++j)
        free(stream->lod_levels[i].shard.shards[j].index);
      free(stream->lod_levels[i].shard.shards);
    }
  }
  if (stream->t_lod_start) {
    CUWARN(cuEventDestroy(stream->t_lod_start));
    CUWARN(cuEventDestroy(stream->t_lod_scatter_end));
    CUWARN(cuEventDestroy(stream->t_lod_reduce_end));
    CUWARN(cuEventDestroy(stream->t_lod_end));
  }
  lod_plan_free(&stream->lod);

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
    CU(Fail, cuEventCreate(&s->flush[i].t_d2h_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&s->flush[i].ready, CU_EVENT_DEFAULT));
  }

  return 0;
Fail:
  return 1;
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
    size_t alignment = s->config.codec == CODEC_ZSTD ? compress_get_input_alignment() : 1;
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

  s->layout.tiles_per_epoch = s->layout.lifted_strides[0] / s->layout.tile_stride;
  s->layout.epoch_elements = s->layout.tiles_per_epoch * s->layout.tile_elements;
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

  return 0;
Fail:
  return 1;
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

  return 0;
Fail:
  return 1;
}

static int
init_tile_pools(struct tile_stream_gpu* s)
{
  const size_t pool_bytes = s->layout.tile_pool_bytes;

  for (int i = 0; i < 2; ++i) {
    CHECK(Fail, (s->pools.buf[i] = buffer_new(pool_bytes, device, 0)).data);
    CU(Fail,
       cuMemsetD8Async(
         (CUdeviceptr)s->pools.buf[i].data, 0, pool_bytes, s->compute));
  }

  return 0;
Fail:
  return 1;
}

static int
init_compression(struct tile_stream_gpu* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const uint64_t M0 = s->layout.tiles_per_epoch;
  const size_t tile_bytes = s->layout.tile_stride * bpe;

  if (s->config.codec == CODEC_ZSTD) {
    s->codec.max_chunk_bytes = align_up(compress_get_max_output_size(tile_bytes),
                                       compress_get_input_alignment());
    CHECK(Fail, s->codec.max_chunk_bytes > 0);
    s->codec.comp_temp_bytes = compress_get_temp_size(M0, tile_bytes);
    if (s->codec.comp_temp_bytes > 0)
      CU(Fail, cuMemAlloc((CUdeviceptr*)&s->codec.d_comp_temp, s->codec.comp_temp_bytes));
  } else {
    // CODEC_NONE: no expansion, no temp buffer
    s->codec.max_chunk_bytes = tile_bytes;
  }
  s->codec.pool_bytes = M0 * s->codec.max_chunk_bytes;

  CU(Fail, cuMemAlloc((CUdeviceptr*)&s->codec.d_comp_sizes, M0 * sizeof(size_t)));

  {
    size_t* h_sizes = (size_t*)malloc(M0 * sizeof(size_t));
    CHECK(Fail, h_sizes);
    for (uint64_t k = 0; k < M0; ++k)
      h_sizes[k] = tile_bytes;
    CU(Fail, cuMemAlloc((CUdeviceptr*)&s->codec.d_uncomp_sizes, M0 * sizeof(size_t)));
    CUresult rc = cuMemcpyHtoD(
      (CUdeviceptr)s->codec.d_uncomp_sizes, h_sizes, M0 * sizeof(size_t));
    free(h_sizes);
    CU(Fail, rc);

    // For CODEC_NONE, pre-fill d_comp_sizes with tile_bytes
    if (s->config.codec == CODEC_NONE) {
      size_t* h_comp = (size_t*)malloc(M0 * sizeof(size_t));
      CHECK(Fail, h_comp);
      for (uint64_t k = 0; k < M0; ++k)
        h_comp[k] = tile_bytes;
      rc = cuMemcpyHtoD(
        (CUdeviceptr)s->codec.d_comp_sizes, h_comp, M0 * sizeof(size_t));
      free(h_comp);
      CU(Fail, rc);
    }
  }

  for (int fc = 0; fc < 2; ++fc) {
    struct flush_slot_gpu* fs = &s->flush[fc];

    CHECK(
      Fail,
      (fs->d_compressed = buffer_new(M0 * s->codec.max_chunk_bytes, device, 0))
        .data);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&fs->d_uncomp_ptrs, M0 * sizeof(void*)));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&fs->d_comp_ptrs, M0 * sizeof(void*)));

    void** h_ptrs = (void**)malloc(M0 * sizeof(void*));
    CHECK(Fail, h_ptrs);

    void* pool = s->pools.buf[fc].data;
    for (uint64_t k = 0; k < M0; ++k)
      h_ptrs[k] = (char*)pool + k * tile_bytes;
    CU(
      Fail,
      cuMemcpyHtoD((CUdeviceptr)fs->d_uncomp_ptrs, h_ptrs, M0 * sizeof(void*)));

    for (uint64_t k = 0; k < M0; ++k)
      h_ptrs[k] = (char*)fs->d_compressed.data + k * s->codec.max_chunk_bytes;
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)fs->d_comp_ptrs, h_ptrs, M0 * sizeof(void*)));
    free(h_ptrs);
  }

  return 0;
Fail:
  return 1;
}

static int
init_l0_aggregate_and_shards(struct tile_stream_gpu* s)
{

  const uint8_t rank = s->config.rank;
  const struct dimension* dims = s->config.dimensions;

  crc32c_init_table();

  uint64_t tile_count[MAX_RANK];
  uint64_t tiles_per_shard[MAX_RANK];
  for (int d = 0; d < rank; ++d) {
    tile_count[d] = ceildiv(dims[d].size, dims[d].tile_size);
    uint64_t tps = dims[d].tiles_per_shard;
    tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
  }

  CHECK(Fail,
        aggregate_layout_init(&s->agg_layout,
                              rank,
                              tile_count,
                              tiles_per_shard,
                              s->layout.tiles_per_epoch,
                              s->codec.max_chunk_bytes) == 0);

  for (int i = 0; i < 2; ++i)
    CHECK(Fail,
          aggregate_slot_init(&s->agg[i], &s->agg_layout, s->codec.pool_bytes) ==
            0);

  s->shard.tiles_per_shard_0 = tiles_per_shard[0];
  s->shard.tiles_per_shard_inner = 1;
  for (int d = 1; d < rank; ++d)
    s->shard.tiles_per_shard_inner *= tiles_per_shard[d];
  s->shard.tiles_per_shard_total =
    s->shard.tiles_per_shard_0 * s->shard.tiles_per_shard_inner;

  s->shard.shard_inner_count = 1;
  for (int d = 1; d < rank; ++d)
    s->shard.shard_inner_count *= ceildiv(tile_count[d], tiles_per_shard[d]);

  s->shard.shards = (struct active_shard*)calloc(s->shard.shard_inner_count,
                                                 sizeof(struct active_shard));
  CHECK(Fail, s->shard.shards);

  size_t index_bytes = 2 * s->shard.tiles_per_shard_total * sizeof(uint64_t);
  for (uint64_t i = 0; i < s->shard.shard_inner_count; ++i) {
    s->shard.shards[i].index = (uint64_t*)malloc(index_bytes);
    CHECK(Fail, s->shard.shards[i].index);
    memset(s->shard.shards[i].index, 0xFF, index_bytes);
  }

  s->shard.epoch_in_shard = 0;
  s->shard.shard_epoch = 0;

  for (int i = 0; i < 2; ++i)
    CU(Fail, cuEventRecord(s->agg[i].ready, s->compute));

  return 0;
Fail:
  return 1;
}

static int
init_lod(struct tile_stream_gpu* s)
{
  if (!s->config.enable_multiscale)
    return 0;

  const uint8_t rank = s->config.rank;
  const struct dimension* dims = s->config.dimensions;
  const size_t bpe = s->config.bytes_per_element;

  // Use epoch shape, not full volume shape.
  // Dim 0's lifted stride is collapsed (strides[0]=0), so one epoch covers
  // tile_size[0] along dim 0 and the full extent along all other dims.
  uint64_t shape[MAX_RANK / 2];
  shape[0] = dims[0].tile_size;
  for (int d = 1; d < rank; ++d)
    shape[d] = dims[d].size;

  CHECK(Fail,
        lod_plan_init(
          &s->lod, rank, shape, (uint8_t)s->config.lod_mask, LOD_MAX_LEVELS));

  // Allocate linear epoch buffer
  size_t linear_bytes = s->layout.epoch_elements * bpe;
  CHECK(Fail, (s->d_linear = buffer_new(linear_bytes, device, 0)).data);

  // Allocate morton buffer (all levels packed)
  uint64_t total_vals = s->lod.levels.ends[s->lod.nlev - 1];
  size_t morton_bytes = total_vals * bpe;
  CHECK(Fail, (s->d_morton = buffer_new(morton_bytes, device, 0)).data);

  // Upload shapes to device
  CU(Fail, cuMemAlloc(&s->d_lod_full_shape, rank * sizeof(uint64_t)));
  CU(Fail,
     cuMemcpyHtoD(
       s->d_lod_full_shape, s->lod.shapes[0], rank * sizeof(uint64_t)));

  if (s->lod.lod_ndim > 0) {
    CU(Fail, cuMemAlloc(&s->d_lod_shape, s->lod.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(s->d_lod_shape,
                    s->lod.lod_shapes[0],
                    s->lod.lod_ndim * sizeof(uint64_t)));
  }

  // Per-level device arrays for fill_ends
  for (int l = 0; l < s->lod.nlev - 1; ++l) {
    struct lod_span seg = lod_segment(&s->lod, l);
    uint64_t n_parents = lod_span_len(seg);

    CU(Fail,
       cuMemAlloc(&s->d_lod_child_shapes[l],
                  s->lod.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(s->d_lod_child_shapes[l],
                    s->lod.lod_shapes[l],
                    s->lod.lod_ndim * sizeof(uint64_t)));

    CU(Fail,
       cuMemAlloc(&s->d_lod_parent_shapes[l],
                  s->lod.lod_ndim * sizeof(uint64_t)));
    CU(Fail,
       cuMemcpyHtoD(s->d_lod_parent_shapes[l],
                    s->lod.lod_shapes[l + 1],
                    s->lod.lod_ndim * sizeof(uint64_t)));

    CU(Fail, cuMemAlloc(&s->d_lod_level_ends[l], n_parents * sizeof(uint64_t)));
  }

  CU(Fail, cuEventCreate(&s->t_lod_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->t_lod_scatter_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->t_lod_reduce_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->t_lod_end, CU_EVENT_DEFAULT));

  // Per-level tile layouts and unified tile buffer
  {
    size_t alignment = s->config.codec == CODEC_ZSTD ? compress_get_input_alignment() : 1;
    size_t total_tile_bytes = 0;

    for (int lv = 1; lv < s->lod.nlev; ++lv) {
      struct stream_layout* lay = &s->lod_layouts[lv];
      const uint64_t* lv_shape = s->lod.shapes[lv];

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

      // Upload per-level full shape and LOD shape for morton-to-tile
      {
        CU(Fail,
           cuMemAlloc(&s->d_lod_lv_full_shapes[lv], rank * sizeof(uint64_t)));
        CU(Fail,
           cuMemcpyHtoD(s->d_lod_lv_full_shapes[lv],
                        s->lod.shapes[lv],
                        rank * sizeof(uint64_t)));

        CU(Fail,
           cuMemAlloc(&s->d_lod_lv_lod_shapes[lv],
                      s->lod.lod_ndim * sizeof(uint64_t)));
        CU(Fail,
           cuMemcpyHtoD(s->d_lod_lv_lod_shapes[lv],
                        s->lod.lod_shapes[lv],
                        s->lod.lod_ndim * sizeof(uint64_t)));
      }

      s->lod_tile_ends[lv] =
        total_tile_bytes / bpe + lay->tiles_per_epoch * lay->tile_stride;
      total_tile_bytes += lay->tile_pool_bytes;

      // Per-level compress + aggregate + shard init
      {
        uint64_t M_lv = lay->tiles_per_epoch;
        size_t tile_bytes = lay->tile_stride * bpe;

        if (s->config.codec == CODEC_ZSTD) {
          s->lod_levels[lv].codec.max_chunk_bytes =
            align_up(compress_get_max_output_size(tile_bytes),
                     compress_get_input_alignment());
          CHECK(Fail, s->lod_levels[lv].codec.max_chunk_bytes > 0);
          s->lod_levels[lv].codec.comp_temp_bytes =
            compress_get_temp_size(M_lv, tile_bytes);
          if (s->lod_levels[lv].codec.comp_temp_bytes > 0)
            CU(Fail,
               cuMemAlloc((CUdeviceptr*)&s->lod_levels[lv].codec.d_comp_temp,
                          s->lod_levels[lv].codec.comp_temp_bytes));
        } else {
          s->lod_levels[lv].codec.max_chunk_bytes = tile_bytes;
        }
        s->lod_levels[lv].codec.pool_bytes =
          M_lv * s->lod_levels[lv].codec.max_chunk_bytes;

        CU(Fail,
           cuMemAlloc((CUdeviceptr*)&s->lod_levels[lv].codec.d_comp_sizes,
                      M_lv * sizeof(size_t)));

        {
          size_t* h_sizes = (size_t*)malloc(M_lv * sizeof(size_t));
          CHECK(Fail, h_sizes);
          for (uint64_t k = 0; k < M_lv; ++k)
            h_sizes[k] = tile_bytes;
          CU(Fail,
             cuMemAlloc((CUdeviceptr*)&s->lod_levels[lv].codec.d_uncomp_sizes,
                        M_lv * sizeof(size_t)));
          CUresult rc =
            cuMemcpyHtoD((CUdeviceptr)s->lod_levels[lv].codec.d_uncomp_sizes,
                         h_sizes,
                         M_lv * sizeof(size_t));
          if (s->config.codec == CODEC_NONE) {
            // Pre-fill d_comp_sizes for CODEC_NONE
            CUresult rc2 =
              cuMemcpyHtoD((CUdeviceptr)s->lod_levels[lv].codec.d_comp_sizes,
                           h_sizes,
                           M_lv * sizeof(size_t));
            free(h_sizes);
            CU(Fail, rc);
            CU(Fail, rc2);
          } else {
            free(h_sizes);
            CU(Fail, rc);
          }
        }

        CHECK(Fail,
              (s->lod_levels[lv].d_compressed =
                 buffer_new(s->lod_levels[lv].codec.pool_bytes, device, 0))
                .data);
        CU(Fail,
           cuMemAlloc((CUdeviceptr*)&s->lod_levels[lv].d_uncomp_ptrs,
                      M_lv * sizeof(void*)));
        CU(Fail,
           cuMemAlloc((CUdeviceptr*)&s->lod_levels[lv].d_comp_ptrs,
                      M_lv * sizeof(void*)));

        // Aggregate layout for this level
        {
          uint64_t lv_tile_count[MAX_RANK / 2];
          uint64_t lv_tps[MAX_RANK / 2];
          for (int d = 0; d < rank; ++d) {
            lv_tile_count[d] = tc[d];
            uint64_t tps = dims[d].tiles_per_shard;
            lv_tps[d] = (tps == 0) ? lv_tile_count[d] : tps;
          }
          CHECK(Fail,
                aggregate_layout_init(&s->lod_levels[lv].agg_layout,
                                      rank,
                                      lv_tile_count,
                                      lv_tps,
                                      M_lv,
                                      s->lod_levels[lv].codec.max_chunk_bytes) ==
                  0);
          CHECK(Fail,
                aggregate_slot_init(&s->lod_levels[lv].agg_slot,
                                    &s->lod_levels[lv].agg_layout,
                                    s->lod_levels[lv].codec.pool_bytes) == 0);
          CU(Fail, cuEventRecord(s->lod_levels[lv].agg_slot.ready, s->compute));
        }

        // Shard state for this level
        {
          uint64_t lv_tps[MAX_RANK / 2];
          for (int d = 0; d < rank; ++d) {
            uint64_t tps = dims[d].tiles_per_shard;
            lv_tps[d] = (tps == 0) ? tc[d] : tps;
          }

          s->lod_levels[lv].shard.tiles_per_shard_0 = lv_tps[0];
          s->lod_levels[lv].shard.tiles_per_shard_inner = 1;
          for (int d = 1; d < rank; ++d)
            s->lod_levels[lv].shard.tiles_per_shard_inner *= lv_tps[d];
          s->lod_levels[lv].shard.tiles_per_shard_total =
            s->lod_levels[lv].shard.tiles_per_shard_0 *
            s->lod_levels[lv].shard.tiles_per_shard_inner;

          s->lod_levels[lv].shard.shard_inner_count = 1;
          for (int d = 1; d < rank; ++d)
            s->lod_levels[lv].shard.shard_inner_count *=
              ceildiv(tc[d], lv_tps[d]);

          s->lod_levels[lv].shard.shards = (struct active_shard*)calloc(
            s->lod_levels[lv].shard.shard_inner_count,
            sizeof(struct active_shard));
          CHECK(Fail, s->lod_levels[lv].shard.shards);

          size_t idx_bytes = 2 * s->lod_levels[lv].shard.tiles_per_shard_total *
                             sizeof(uint64_t);
          for (uint64_t i = 0; i < s->lod_levels[lv].shard.shard_inner_count;
               ++i) {
            s->lod_levels[lv].shard.shards[i].index =
              (uint64_t*)malloc(idx_bytes);
            CHECK(Fail, s->lod_levels[lv].shard.shards[i].index);
            memset(s->lod_levels[lv].shard.shards[i].index, 0xFF, idx_bytes);
          }
        }
      }
    }

    if (total_tile_bytes > 0) {
      CHECK(Fail,
            (s->d_lod_tiles = buffer_new(total_tile_bytes, device, 0)).data);
      CU(Fail,
         cuMemsetD8Async(
           (CUdeviceptr)s->d_lod_tiles.data, 0, total_tile_bytes, s->compute));
    }

    // Build uncomp/comp pointer arrays now that tile buffer is allocated
    if (s->d_lod_tiles.data) {
      uint64_t tile_el_offset = 0;
      for (int lv = 1; lv < s->lod.nlev; ++lv) {
        struct stream_layout* lay = &s->lod_layouts[lv];
        uint64_t M_lv = lay->tiles_per_epoch;

        void** h_ptrs = (void**)malloc(M_lv * sizeof(void*));
        CHECK(Fail, h_ptrs);

        for (uint64_t k = 0; k < M_lv; ++k)
          h_ptrs[k] = (char*)s->d_lod_tiles.data +
                      (tile_el_offset + k * lay->tile_stride) * bpe;
        CU(Fail,
           cuMemcpyHtoD((CUdeviceptr)s->lod_levels[lv].d_uncomp_ptrs,
                        h_ptrs,
                        M_lv * sizeof(void*)));

        for (uint64_t k = 0; k < M_lv; ++k)
          h_ptrs[k] = (char*)s->lod_levels[lv].d_compressed.data +
                      k * s->lod_levels[lv].codec.max_chunk_bytes;
        CU(Fail,
           cuMemcpyHtoD((CUdeviceptr)s->lod_levels[lv].d_comp_ptrs,
                        h_ptrs,
                        M_lv * sizeof(void*)));
        free(h_ptrs);

        tile_el_offset += M_lv * lay->tile_stride;
      }
    }
  }

  // Populate m2t_layout structs
  {
    enum lod_dtype dtype =
      (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

    // L0
    {
      uint64_t n_el = 1;
      for (int d = 0; d < rank; ++d)
        n_el *= s->lod.shapes[0][d];

      s->m2t_l0 = (struct m2t_layout){
        .dtype = dtype,
        .ndim = rank,
        .d_full_shape = s->d_lod_full_shape,
        .lod_ndim = s->lod.lod_ndim,
        .lod_mask = s->lod.lod_mask,
        .d_lod_shape = s->d_lod_shape,
        .lod_count = s->lod.lod_counts[0],
        .n_elements = n_el,
        .lod_nlod = lod_m2t_lod_nlod(s->lod.lod_ndim, s->lod.lod_shapes[0]),
        .d_lifted_shape = (CUdeviceptr)s->layout.d_lifted_shape,
        .d_lifted_strides = (CUdeviceptr)s->layout.d_lifted_strides,
      };
    }

    // Levels 1+
    for (int lv = 1; lv < s->lod.nlev; ++lv) {
      struct stream_layout* lay = &s->lod_layouts[lv];
      uint64_t n_el = 1;
      for (int d = 0; d < rank; ++d)
        n_el *= s->lod.shapes[lv][d];

      s->m2t_levels[lv] = (struct m2t_layout){
        .dtype = dtype,
        .ndim = rank,
        .d_full_shape = s->d_lod_lv_full_shapes[lv],
        .lod_ndim = s->lod.lod_ndim,
        .lod_mask = s->lod.lod_mask,
        .d_lod_shape = s->d_lod_lv_lod_shapes[lv],
        .lod_count = s->lod.lod_counts[lv],
        .n_elements = n_el,
        .lod_nlod = lod_m2t_lod_nlod(s->lod.lod_ndim, s->lod.lod_shapes[lv]),
        .d_lifted_shape = (CUdeviceptr)lay->d_lifted_shape,
        .d_lifted_strides = (CUdeviceptr)lay->d_lifted_strides,
      };
    }
  }

  return 0;
Fail:
  return 1;
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
    CU(Fail, cuEventRecord(s->flush[i].t_d2h_start, s->compute));
    CU(Fail, cuEventRecord(s->flush[i].ready, s->compute));
  }

  if (s->t_lod_start) {
    CU(Fail, cuEventRecord(s->t_lod_start, s->compute));
    CU(Fail, cuEventRecord(s->t_lod_scatter_end, s->compute));
    CU(Fail, cuEventRecord(s->t_lod_reduce_end, s->compute));
    CU(Fail, cuEventRecord(s->t_lod_end, s->compute));
  }

  return 0;
Fail:
  return 1;
}

int
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct tile_stream_gpu* out)
{
  CHECK(Fail, config);
  CHECK(Fail, out);

  *out = (struct tile_stream_gpu){
    .writer = { .append = tile_stream_gpu_append,
                .flush = tile_stream_gpu_flush },
    .dispatch = config->enable_multiscale ? dispatch_scatter_multiscale
                                          : dispatch_scatter,
    .config = *config,
  };

  CHECK(Fail, config->bytes_per_element > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= MAX_RANK / 2);
  CHECK(Fail, config->dimensions);
  CHECK(Fail, config->shard_sink);

  CHECK(Fail, init_cuda_streams_and_events(out) == 0);
  CHECK(Fail, init_l0_layout(out) == 0);
  CHECK(Fail, init_staging_buffers(out) == 0);
  CHECK(Fail, init_tile_pools(out) == 0);
  CHECK(Fail, init_compression(out) == 0);
  CHECK(Fail, init_l0_aggregate_and_shards(out) == 0);
  CHECK(Fail, init_lod(out) == 0);
  CHECK(Fail, seed_events(out) == 0);

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics = (struct stream_metrics){
    .memcpy = { .name = "Memcpy", .best_ms = 1e30f },
    .h2d = { .name = "H2D", .best_ms = 1e30f },
    .scatter = { .name = "Scatter", .best_ms = 1e30f },
    .compress = { .name = "Compress", .best_ms = 1e30f },
    .aggregate = { .name = "Aggregate", .best_ms = 1e30f },
    .d2h = { .name = "D2H", .best_ms = 1e30f },
    .lod_scatter = { .name = "LOD Scat", .best_ms = 1e30f },
    .lod_reduce = { .name = "LOD Red", .best_ms = 1e30f },
    .lod_m2t = { .name = "LOD M2T", .best_ms = 1e30f },
  };

  return 0;

Fail:
  tile_stream_gpu_destroy(out);
  return 1;
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
            accumulate_metric_cu(
              &s->metrics.h2d, ss->t_h2d_start, ss->h_in.ready);
            accumulate_metric_cu(
              &s->metrics.scatter, ss->t_scatter_start, ss->t_scatter_end);
          }
        }

        {
          struct platform_clock mc = { 0 };
          platform_toc(&mc);
          memcpy((uint8_t*)s->stage.slot[s->stage.current].h_in.data +
                   s->stage.bytes_written,
                 src + written,
                 payload);
          accumulate_metric_ms(&s->metrics.memcpy,
                               (float)(platform_toc(&mc) * 1000.0));
        }
        s->stage.bytes_written += payload;
        written += payload;

        if (s->stage.bytes_written == buffer_capacity ||
            written == bytes_this_pass) {
          if (s->dispatch(s))
            goto Error;
          s->stage.bytes_written = 0;
        }
      }
    }
    src += bytes_this_pass;

    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = flush_epoch(s);
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
    if (s->dispatch(s))
      return writer_error();
    s->stage.bytes_written = 0;
  }

  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  if (s->cursor % s->layout.epoch_elements != 0 || s->cursor == 0) {
    r = flush_epoch_sync(s);
    if (r.error)
      return r;
  }

  if (s->shard.epoch_in_shard > 0) {
    if (emit_shards(&s->shard))
      return writer_error();
  }
  // Emit partial LOD shards
  if (s->config.enable_multiscale && s->d_lod_tiles.data) {
    for (int lv = 1; lv < s->lod.nlev; ++lv) {
      if (s->lod_levels[lv].shard.epoch_in_shard > 0) {
        if (emit_shards(&s->lod_levels[lv].shard))
          return writer_error();
      }
    }
  }

  return writer_ok();
}

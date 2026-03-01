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
    return 1;

  const uint64_t elements = s->stage.bytes_written / bpe;
  if (elements == 0)
    return 1;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];
  void* pool = current_pool(s);

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
// L0 tiling is deferred to run_lod (lod_morton_to_tiles at lv=0).
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
      s->lod.plan.levels.ends[s->lod.plan.nlev - 1] * bpe;
    const size_t unified_pool_bytes =
      s->total_tiles * s->layout.tile_stride * bpe;

    accumulate_metric_cu(&s->metrics.lod_scatter,
                         s->lod.t_start,
                         s->lod.t_scatter_end,
                         scatter_bytes);
    accumulate_metric_cu(&s->metrics.lod_reduce,
                         s->lod.t_scatter_end,
                         s->lod.t_reduce_end,
                         morton_bytes);
    accumulate_metric_cu(&s->metrics.lod_morton_tile,
                         s->lod.t_reduce_end,
                         s->lod.t_end,
                         unified_pool_bytes);
  }

  {
    const size_t pool_bytes =
      s->total_tiles * s->layout.tile_stride * s->config.bytes_per_element;
    const size_t comp_bytes =
      s->codec.batch_size * s->codec.max_output_size;

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

  for (int lv = 0; lv < s->nlev; ++lv) {
    if (!deliver_to_shards(
          s, (uint8_t)lv, &s->lod_levels[lv].shard, &s->lod_levels[lv].agg[fc]))
      goto Error;
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
// Single compress batch for all levels, then per-level aggregate + D2H.
// fc: flush slot index (0 or 1, matches pools.current before swap).
static int
kick_epoch(struct tile_stream_gpu* s, int fc)
{
  struct flush_slot_gpu* fs = &s->flush[fc];
  const size_t bpe = s->config.bytes_per_element;
  const size_t tile_bytes = s->layout.tile_stride * bpe;

  // Wait for all tile writes to finish
  CU(Error, cuStreamWaitEvent(s->compress, s->pools.buf[fc].ready, 0));
  if (s->enable_multiscale && s->lod.t_end)
    CU(Error, cuStreamWaitEvent(s->compress, s->lod.t_end, 0));

  CU(Error, cuEventRecord(fs->t_compress_start, s->compress));

  // Single compress batch for all levels
  CHECK(Error,
        codec_compress(&s->codec,
                       s->pools.buf[fc].data,
                       tile_bytes,
                       fs->d_compressed.data,
                       s->compress));

  CU(Error, cuEventRecord(fs->d_compressed.ready, s->compress));

  // Per-level aggregate on compress stream
  for (int lv = 0; lv < s->nlev; ++lv) {
    void* d_comp_lv = (char*)fs->d_compressed.data +
                      s->level_tile_offset[lv] * s->codec.max_output_size;
    size_t* d_sizes_lv = s->codec.d_comp_sizes + s->level_tile_offset[lv];
    struct aggregate_slot* agg = &s->lod_levels[lv].agg[fc];

    CHECK(Error,
          aggregate_by_shard_async(&s->lod_levels[lv].agg_layout,
                                   d_comp_lv,
                                   d_sizes_lv,
                                   agg,
                                   s->compress) == 0);
    CU(Error, cuEventRecord(agg->ready, s->compress));
  }

  CU(Error, cuEventRecord(fs->t_aggregate_end, s->compress));

  // Per-level D2H on d2h stream
  CU(Error, cuStreamWaitEvent(s->d2h, fs->t_aggregate_end, 0));
  CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));

  for (int lv = 0; lv < s->nlev; ++lv) {
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

  return 1;

Error:
  return 0;
}

// Run LOD scatter + fill_ends + reduce on the linear epoch buffer.
// Outputs to d_morton. Must be called after L0 scatter completes for the epoch.
static int
run_lod(struct tile_stream_gpu* s)
{
  if (!s->enable_multiscale || !s->lod.d_linear.data)
    return 1;

  struct lod_plan* p = &s->lod.plan;
  const size_t bpe = s->config.bytes_per_element;
  enum lod_dtype dtype = (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

  uint64_t n_elements = lod_span_len(lod_spans_at(&p->levels, 0));

  CU(Error, cuEventRecord(s->lod.t_start, s->compute));

  lod_scatter((CUdeviceptr)s->lod.d_morton.data,
              (CUdeviceptr)s->lod.d_linear.data,
              dtype,
              p->ndim,
              n_elements,
              s->lod.d_full_shape,
              s->lod.d_lod_shape,
              p->lod_ndim,
              p->lod_shapes[0],
              p->lod_mask,
              p->lod_counts[0],
              s->compute);

  CU(Error, cuEventRecord(s->lod.t_scatter_end, s->compute));

  for (int l = 0; l < p->nlev - 1; ++l) {
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
               src_level.beg,
               dst_level.beg,
               p->lod_counts[l],
               p->lod_counts[l + 1],
               p->batch_count,
               s->compute);
  }

  CU(Error, cuEventRecord(s->lod.t_reduce_end, s->compute));

  // L0: morton-to-tile scatter into unified pool at offset 0
  {
    struct lod_span lev0 = lod_spans_at(&p->levels, 0);
    void* pool = current_pool(s);

    lod_morton_to_tiles((CUdeviceptr)pool,
                        (CUdeviceptr)s->lod.d_morton.data + lev0.beg * bpe,
                        &s->lod.morton_tile[0],
                        s->compute);
  }

  // LOD levels 1+: morton-to-tile scatter into unified pool at level offsets
  for (int lv = 1; lv < p->nlev; ++lv) {
    struct lod_span lev = lod_spans_at(&p->levels, lv);
    CUdeviceptr dst = (CUdeviceptr)current_pool(s) +
                      s->level_tile_offset[lv] * s->layout.tile_stride * bpe;

    lod_morton_to_tiles(dst,
                        (CUdeviceptr)s->lod.d_morton.data + lev.beg * bpe,
                        &s->lod.morton_tile[lv],
                        s->compute);
  }

  // Signal pool ready AFTER all levels' morton-to-tile scatter
  CU(Error, cuEventRecord(s->pools.buf[s->pools.current].ready, s->compute));
  CU(Error, cuEventRecord(s->lod.t_end, s->compute));
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

  // Swap pool and zero next unified region
  dbuf_swap(&s->pools);
  void* next = current_pool(s);
  size_t total_pool_bytes =
    s->total_tiles * s->layout.tile_stride * s->config.bytes_per_element;
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)next, 0, total_pool_bytes, s->compute));

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

static void
destroy_level_state(struct lod_level_state* lls)
{
  aggregate_layout_destroy(&lls->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&lls->agg[i]);
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

  // Per-level aggregate + shard state
  for (int lv = 0; lv < stream->nlev; ++lv)
    destroy_level_state(&stream->lod_levels[lv]);

  // LOD cleanup
  buffer_free(&stream->lod.d_linear);
  buffer_free(&stream->lod.d_morton);
  CUWARN(cuMemFree(stream->lod.d_full_shape));
  CUWARN(cuMemFree(stream->lod.d_lod_shape));
  CUWARN(cuMemFree(stream->lod.d_ends));
  for (int i = 0; i < stream->lod.plan.nlev - 1; ++i) {
    CUWARN(cuMemFree(stream->lod.d_child_shapes[i]));
    CUWARN(cuMemFree(stream->lod.d_parent_shapes[i]));
    CUWARN(cuMemFree(stream->lod.d_level_ends[i]));
  }
  for (int i = 1; i < stream->lod.plan.nlev; ++i) {
    CUWARN(cuMemFree(stream->lod.d_lv_full_shapes[i]));
    CUWARN(cuMemFree(stream->lod.d_lv_lod_shapes[i]));
    CUWARN(cuMemFree((CUdeviceptr)stream->lod.layouts[i].d_lifted_shape));
    CUWARN(cuMemFree((CUdeviceptr)stream->lod.layouts[i].d_lifted_strides));
  }
  if (stream->lod.t_start) {
    CUWARN(cuEventDestroy(stream->lod.t_start));
    CUWARN(cuEventDestroy(stream->lod.t_scatter_end));
    CUWARN(cuEventDestroy(stream->lod.t_reduce_end));
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

  // Compute total_tiles and level tile offsets from L0 + LOD layouts
  s->level_tile_count[0] = s->layout.tiles_per_epoch;
  s->level_tile_offset[0] = 0;
  s->total_tiles = s->layout.tiles_per_epoch;

  for (int lv = 1; lv < s->nlev; ++lv) {
    s->level_tile_count[lv] = s->lod.layouts[lv].tiles_per_epoch;
    s->level_tile_offset[lv] = s->total_tiles;
    s->total_tiles += s->level_tile_count[lv];
  }

  const size_t pool_bytes = s->total_tiles * s->layout.tile_stride * bpe;

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
  const uint64_t M = s->total_tiles;
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

  crc32c_init_table();

  for (int lv = 0; lv < s->nlev; ++lv) {
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

    uint64_t M_lv = s->level_tile_count[lv];
    size_t pool_bytes_lv = M_lv * s->codec.max_output_size;

    CHECK(Fail,
          aggregate_layout_init(&s->lod_levels[lv].agg_layout,
                                rank,
                                tile_count,
                                tiles_per_shard,
                                M_lv,
                                s->codec.max_output_size) == 0);

    for (int i = 0; i < 2; ++i) {
      CHECK(Fail,
            aggregate_slot_init(&s->lod_levels[lv].agg[i],
                                &s->lod_levels[lv].agg_layout,
                                pool_bytes_lv) == 0);
      CU(Fail, cuEventRecord(s->lod_levels[lv].agg[i].ready, s->compute));
    }

    // Shard state
    struct shard_state* ss = &s->lod_levels[lv].shard;
    ss->tiles_per_shard_0 = tiles_per_shard[0];
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
  shape[0] = dims[0].tile_size;
  for (int d = 1; d < rank; ++d)
    shape[d] = dims[d].size;

  CHECK(
    Fail,
    lod_plan_init(
      &s->lod.plan, rank, shape, (uint8_t)s->lod_mask, LOD_MAX_LEVELS));

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

  // Per-level device arrays for fill_ends
  for (int l = 0; l < s->lod.plan.nlev - 1; ++l) {
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

    for (int lv = 1; lv < s->lod.plan.nlev; ++lv) {
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

      // Upload per-level full shape and LOD shape for morton-to-tile
      {
        CU(Fail,
           cuMemAlloc(&s->lod.d_lv_full_shapes[lv], rank * sizeof(uint64_t)));
        CU(Fail,
           cuMemcpyHtoD(s->lod.d_lv_full_shapes[lv],
                        s->lod.plan.shapes[lv],
                        rank * sizeof(uint64_t)));

        CU(Fail,
           cuMemAlloc(&s->lod.d_lv_lod_shapes[lv],
                      s->lod.plan.lod_ndim * sizeof(uint64_t)));
        CU(Fail,
           cuMemcpyHtoD(s->lod.d_lv_lod_shapes[lv],
                        s->lod.plan.lod_shapes[lv],
                        s->lod.plan.lod_ndim * sizeof(uint64_t)));
      }
    }
  }

  // Populate morton_tile_layout structs
  {
    enum lod_dtype dtype = (bpe == 2) ? lod_dtype_u16 : lod_dtype_f32;

    // L0
    {
      uint64_t n_el = 1;
      for (int d = 0; d < rank; ++d)
        n_el *= s->lod.plan.shapes[0][d];

      s->lod.morton_tile[0] = (struct morton_tile_layout){
        .dtype = dtype,
        .ndim = rank,
        .d_full_shape = s->lod.d_full_shape,
        .lod_ndim = s->lod.plan.lod_ndim,
        .lod_mask = s->lod.plan.lod_mask,
        .d_lod_shape = s->lod.d_lod_shape,
        .lod_count = s->lod.plan.lod_counts[0],
        .n_elements = n_el,
        .lod_nlod =
          lod_morton_tile_nlod(s->lod.plan.lod_ndim, s->lod.plan.lod_shapes[0]),
        .d_lifted_shape = (CUdeviceptr)s->layout.d_lifted_shape,
        .d_lifted_strides = (CUdeviceptr)s->layout.d_lifted_strides,
      };
    }

    // Levels 1+
    for (int lv = 1; lv < s->lod.plan.nlev; ++lv) {
      struct stream_layout* lay = &s->lod.layouts[lv];
      uint64_t n_el = 1;
      for (int d = 0; d < rank; ++d)
        n_el *= s->lod.plan.shapes[lv][d];

      s->lod.morton_tile[lv] = (struct morton_tile_layout){
        .dtype = dtype,
        .ndim = rank,
        .d_full_shape = s->lod.d_lv_full_shapes[lv],
        .lod_ndim = s->lod.plan.lod_ndim,
        .lod_mask = s->lod.plan.lod_mask,
        .d_lod_shape = s->lod.d_lv_lod_shapes[lv],
        .lod_count = s->lod.plan.lod_counts[lv],
        .n_elements = n_el,
        .lod_nlod = lod_morton_tile_nlod(s->lod.plan.lod_ndim,
                                         s->lod.plan.lod_shapes[lv]),
        .d_lifted_shape = (CUdeviceptr)lay->d_lifted_shape,
        .d_lifted_strides = (CUdeviceptr)lay->d_lifted_strides,
      };
    }
  }

  s->nlev = s->lod.plan.nlev;
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
  uint64_t total_vals = s->lod.plan.levels.ends[s->lod.plan.nlev - 1];
  size_t morton_bytes = total_vals * bpe;
  CHECK(Fail, (s->lod.d_morton = buffer_new(morton_bytes, device, 0)).data);

  CU(Fail, cuEventCreate(&s->lod.t_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_scatter_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_reduce_end, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&s->lod.t_end, CU_EVENT_DEFAULT));

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

  // Compute lod_mask and enable_multiscale from dimensions
  uint32_t lod_mask = 0;
  for (int d = 0; d < config->rank; ++d) {
    if (config->dimensions[d].downsample) {
      if (d == 0) {
        log_error("Downsample on dim 0 is not supported");
        goto Fail;
      }
      lod_mask |= (1u << d);
    }
  }
  int enable_multiscale = lod_mask != 0;

  *out = (struct tile_stream_gpu){
    .writer = { .append = tile_stream_gpu_append,
                .flush = tile_stream_gpu_flush },
    .dispatch = enable_multiscale ? dispatch_scatter_multiscale
                                  : dispatch_scatter,
    .config = *config,
    .nlev = 1,
    .enable_multiscale = enable_multiscale,
    .lod_mask = lod_mask,
  };

  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;

  CHECK(Fail, init_cuda_streams_and_events(out));
  CHECK(Fail, init_l0_layout(out));
  CHECK(Fail, init_staging_buffers(out));
  CHECK(Fail, init_lod_layouts(out));
  CHECK(Fail, init_tile_pools(out));
  CHECK(Fail, init_compression(out));
  CHECK(Fail, init_aggregate_and_shards(out));
  CHECK(Fail, init_lod_buffers(out));
  CHECK(Fail, seed_events(out));

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics = (struct stream_metrics){
    .memcpy = { .name = "Memcpy", .best_ms = 1e30f },
    .h2d = { .name = "H2D", .best_ms = 1e30f },
    .scatter = { .name = out->enable_multiscale ? "Copy" : "Scatter",
                 .best_ms = 1e30f },
    .lod_scatter = { .name = "LOD Gather", .best_ms = 1e30f },
    .lod_reduce = { .name = "LOD Reduce", .best_ms = 1e30f },
    .lod_morton_tile = { .name = "LOD Scatter", .best_ms = 1e30f },
    .compress = { .name = "Compress", .best_ms = 1e30f },
    .aggregate = { .name = "Aggregate", .best_ms = 1e30f },
    .d2h = { .name = "D2H", .best_ms = 1e30f },
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
    if (!s->dispatch(s))
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

  // Emit partial shards for all levels
  for (int lv = 0; lv < s->nlev; ++lv) {
    if (s->lod_levels[lv].shard.epoch_in_shard > 0) {
      if (!emit_shards(&s->lod_levels[lv].shard))
        return writer_error();
    }
  }

  return writer_ok();
}

#include "stream_internal.h"

#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "shard_delivery.h"

#include <string.h>

// Forward declarations
static struct writer_result
tile_stream_gpu_append(struct writer* self, struct slice input);
static struct writer_result
tile_stream_gpu_flush(struct writer* self);

// --- Helpers ---

// Return pointer to the current pool buffer (entire K-epoch super-pool).
static inline void*
current_pool_base(struct tile_stream_gpu* s)
{
  return (void*)s->pools.buf[s->pools.current];
}

// Return pointer to the current epoch's tile region within the super-pool.
// epoch_in_batch: 0..K-1
static inline void*
current_pool_epoch(struct tile_stream_gpu* s, uint32_t epoch_in_batch)
{
  const size_t bpe = s->config.bytes_per_element;
  return (char*)s->pools.buf[s->pools.current] +
         (uint64_t)epoch_in_batch * s->levels.total_tiles *
           s->layout.tile_stride * bpe;
}

static struct flush_context
make_flush_context(struct tile_stream_gpu* s)
{
  return (struct flush_context){
    .flush = &s->flush,
    .levels = &s->levels,
    .codec = &s->codec,
    .batch = &s->batch,
    .pools = &s->pools,
    .lod = &s->lod,
    .metrics = &s->metrics,
    .config = &s->config,
    .layout = &s->layout,
    .streams = s->streams,
    .metadata_update_clock = &s->metadata_update_clock,
  };
}

// --- Ingest dispatch ---

static int
dispatch_ingest(struct tile_stream_gpu* s)
{
  if (s->levels.enable_multiscale) {
    return ingest_dispatch_multiscale(&s->stage,
                                      s->lod.d_linear,
                                      s->layout.epoch_elements,
                                      &s->cursor,
                                      s->config.bytes_per_element,
                                      s->streams.h2d,
                                      s->streams.compute);
  } else {
    return ingest_dispatch_scatter(&s->stage,
                                   &s->layout,
                                   current_pool_epoch(s, s->batch.accumulated),
                                   s->pools.ready[s->pools.current],
                                   &s->cursor,
                                   s->config.bytes_per_element,
                                   s->streams.h2d,
                                   s->streams.compute);
  }
}

// --- LOD ---

// Run LOD pipeline for the current epoch, or handle non-multiscale case.
// Updates flush slot batch_active_masks and active_levels_mask.
static int
run_lod_for_epoch(struct tile_stream_gpu* s)
{
  struct flush_slot_gpu* fs = &s->flush.slot[s->pools.current];
  uint32_t active_mask;

  if (!s->levels.enable_multiscale || !s->lod.d_linear) {
    // Non-multiscale: all levels (just L0) are active
    active_mask = 1;
  } else {
    CHECK(Error,
          lod_run_epoch(&s->lod,
                        &s->levels,
                        &s->layout,
                        current_pool_epoch(s, s->batch.accumulated),
                        s->config.bytes_per_element,
                        s->config.reduce_method,
                        s->config.dim0_reduce_method,
                        s->streams.compute,
                        &active_mask) == 0);
  }

  fs->batch_active_masks[s->batch.accumulated] = active_mask;
  fs->active_levels_mask |= active_mask;
  return 0;

Error:
  return 1;
}

// --- Orchestrator ---

// Accumulate one epoch into the current batch, or flush when batch is full.
// Called at each epoch boundary.
static struct writer_result
accumulate_epoch_or_flush(struct tile_stream_gpu* s)
{
  const uint32_t K = s->batch.epochs_per_batch;

  // Run LOD pipeline: populates current epoch's pool tiles + LOD level tiles
  if (run_lod_for_epoch(s))
    return writer_error();

  // Record per-epoch pool-ready event
  CU(Error,
     cuEventRecord(s->batch.pool_events[s->batch.accumulated], s->streams.compute));

  s->batch.accumulated++;

  if (s->batch.accumulated < K)
    return writer_ok();

  // Batch is full — drain previous, kick, swap.
  // old_pool is the pool index before swap; it becomes the flush slot index.
  const int old_pool = s->pools.current;
  struct flush_slot_gpu* fs = &s->flush.slot[old_pool];

  struct flush_context fctx = make_flush_context(s);
  struct writer_result r = flush_drain_pending(&fctx);
  if (r.error)
    return r;

  fs->batch_epoch_count = (int)s->batch.accumulated;
  if (flush_kick_batch(&fctx, old_pool, s->batch.accumulated))
    return writer_error();

  // Swap pool and zero next super-pool (K epochs)
  s->pools.current ^= 1;
  void* next = current_pool_base(s);
  size_t total_pool_bytes = (uint64_t)K * s->levels.total_tiles *
                            s->layout.tile_stride * s->config.bytes_per_element;
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)next, 0, total_pool_bytes, s->streams.compute));

  // Reset for next batch
  s->batch.accumulated = 0;
  s->flush.slot[s->pools.current].active_levels_mask = 0;
  memset(s->flush.slot[s->pools.current].batch_active_masks,
         0,
         sizeof(s->flush.slot[s->pools.current].batch_active_masks));

  s->flush.pending = 1;
  s->flush.current = old_pool;
  return writer_ok();

Error:
  return writer_error();
}

struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s)
{
  return s->metrics;
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

  const uint64_t dim0_size = s->config.dimensions[0].size;
  const uint64_t max_cursor =
    (dim0_size > 0) ? ceildiv(dim0_size, s->config.dimensions[0].tile_size) *
                        s->layout.epoch_elements
                    : 0; // 0 = unbounded (never checked)

  while (src < end) {
    // Bounded dim0: check capacity
    if (dim0_size > 0 && s->cursor >= max_cursor) {
      struct writer_result fr = tile_stream_gpu_flush(&s->writer);
      if (fr.error)
        return writer_error_at(src, end);
      return writer_finished_at(src, end);
    }

    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    // Bounded dim0: clamp to remaining capacity
    if (dim0_size > 0) {
      const uint64_t remaining_capacity = max_cursor - s->cursor;
      if (elements_this_pass > remaining_capacity)
        elements_this_pass = remaining_capacity;
    }

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
          CU(Error, cuEventSynchronize(ss->t_h2d_end));

          if (s->cursor > 0) {
            accumulate_metric_cu(&s->metrics.h2d,
                                 ss->t_h2d_start,
                                 ss->t_h2d_end,
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
          memcpy((uint8_t*)s->stage.slot[s->stage.current].h_in +
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
          if (dispatch_ingest(s))
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
    if (dispatch_ingest(s))
      return writer_error();
    s->stage.bytes_written = 0;
  }

  struct flush_context fctx = make_flush_context(s);

  struct writer_result r = flush_drain_pending(&fctx);
  if (r.error)
    return r;

  // Flush any partial epoch first (sub-epoch data)
  if (s->cursor % s->layout.epoch_elements != 0) {
    // run_lod + record pool event + increment epochs_accumulated
    if (run_lod_for_epoch(s))
      return writer_error();
    CU(Error,
       cuEventRecord(s->batch.pool_events[s->batch.accumulated], s->streams.compute));
    s->batch.accumulated++;
  }

  // Flush any accumulated epochs (partial batch)
  r = flush_accumulated_sync(&fctx);
  if (r.error)
    return r;

  // Drain any partial dim0 accumulators
  r = flush_partial_dim0(&fctx);
  if (r.error)
    return r;

  // Capture actual dim0 tile counts before partial shard emission,
  // since emit_shards resets epoch_in_shard and increments shard_epoch.
  uint64_t dim0_tiles[LOD_MAX_LEVELS];
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->flush.levels[lv].shard;
    dim0_tiles[lv] =
      ss->shard_epoch * ss->tiles_per_shard_0 + ss->epoch_in_shard;
  }

  // Emit partial shards for all levels
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (s->flush.levels[lv].shard.epoch_in_shard > 0) {
      if (emit_shards(&s->flush.levels[lv].shard, s->config.shard_alignment))
        return writer_error();
    }
  }

  // Final metadata update using pre-emit tile counts.
  if (s->config.shard_sink->update_dim0) {
    const struct dimension* dims = s->config.dimensions;
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      uint64_t dim0_extent = dim0_tiles[lv] * dims[0].tile_size;
      s->config.shard_sink->update_dim0(
        s->config.shard_sink, (uint8_t)lv, dim0_extent);
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

void
tile_stream_gpu_init_vtable(struct tile_stream_gpu* s)
{
  s->writer.append = tile_stream_gpu_append;
  s->writer.flush = tile_stream_gpu_flush;
}

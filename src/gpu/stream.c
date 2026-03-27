#include "gpu/stream.flush.h"
#include "gpu/stream.ingest.h"

#include "gpu/metric.cuda.h"
#include "platform/platform.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <string.h>

// Forward declarations
static struct writer_result
tile_stream_gpu_append(struct writer* self, struct slice input);
static struct writer_result
tile_stream_gpu_flush(struct writer* self);

// --- Helpers ---

// Return pointer to the current epoch's chunk region within the super-pool.
// epoch_in_batch: 0..K-1
static inline void*
current_pool_epoch(struct tile_stream_gpu* s, uint32_t epoch_in_batch)
{
  const size_t bytes_per_element = dtype_bpe(s->config.dtype);
  return (char*)s->pools.buf[s->pools.current] + (uint64_t)epoch_in_batch *
                                                   s->levels.total_chunks *
                                                   s->layout.chunk_stride * bytes_per_element;
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
                                      dtype_bpe(s->config.dtype),
                                      s->streams.h2d,
                                      s->streams.compute);
  } else {
    return ingest_dispatch_scatter(&s->stage,
                                   &s->layout,
                                   &s->lod.layout_gpu[0],
                                   current_pool_epoch(s, s->batch.accumulated),
                                   s->pools.ready[s->pools.current],
                                   &s->cursor,
                                   dtype_bpe(s->config.dtype),
                                   s->streams.h2d,
                                   s->streams.compute);
  }
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
  const size_t bytes_per_element = dtype_bpe(s->config.dtype);
  const size_t buffer_capacity = s->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  const uint64_t max_cursor = s->max_cursor;

  while (src < end) {
    // Bounded append dims: check capacity
    if (max_cursor > 0 && s->cursor >= max_cursor) {
      struct writer_result fr = tile_stream_gpu_flush(&s->writer);
      if (fr.error)
        return writer_error_at(src, end);
      return writer_finished_at(src, end);
    }

    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bytes_per_element;
    uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    // Bounded append dims: clamp to remaining capacity
    if (max_cursor > 0) {
      const uint64_t remaining_capacity = max_cursor - s->cursor;
      if (elements_this_pass > remaining_capacity)
        elements_this_pass = remaining_capacity;
    }

    const uint64_t bytes_this_pass = elements_this_pass * bytes_per_element;

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
                                 ss->dispatched_bytes,
                                 ss->dispatched_bytes);
            accumulate_metric_cu(&s->metrics.scatter,
                                 ss->t_scatter_start,
                                 ss->t_scatter_end,
                                 ss->dispatched_bytes,
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
            &s->metrics.memcpy, (float)(platform_toc(&mc) * 1000.0),
            payload, payload);
        }
        s->stage.bytes_written += payload;
        written += payload;

        if (s->stage.bytes_written == buffer_capacity ||
            written == bytes_this_pass) {
          CHECK(Error, dispatch_ingest(s) == 0);
          s->stage.bytes_written = 0;
        }
      }
    }
    src += bytes_this_pass;

    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = flush_accumulate_epoch(s);
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

  struct writer_result r = flush_drain_pending(s);
  if (r.error)
    return r;

  // Flush any partial epoch first (sub-epoch data)
  if (s->cursor % s->layout.epoch_elements != 0) {
    // run_lod + record pool event + increment epochs_accumulated
    if (flush_run_epoch_lod(s))
      return writer_error();
    CU(Error,
       cuEventRecord(s->batch.pool_events[s->batch.accumulated],
                     s->streams.compute));
    s->batch.accumulated++;
  }

  // Flush any accumulated epochs (partial batch)
  r = flush_accumulated_sync(s);
  if (r.error)
    return r;

  // Drain any partial append accumulators
  r = flush_partial_append(s);
  if (r.error)
    return r;

  // Capture actual append chunk counts before partial shard emission,
  // since finalize_shards resets epoch_in_shard and increments shard_epoch.
  uint64_t append_chunks[LOD_MAX_LEVELS];
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    struct shard_state* ss = &s->compress_agg.levels[lv].shard;
    append_chunks[lv] =
      ss->shard_epoch * ss->chunks_per_shard_append + ss->epoch_in_shard;
  }

  // Emit partial shards for all levels
  for (int lv = 0; lv < s->levels.nlod; ++lv) {
    if (s->compress_agg.levels[lv].shard.epoch_in_shard > 0) {
      if (finalize_shards(&s->compress_agg.levels[lv].shard,
                      s->config.shard_alignment))
        return writer_error();
    }
  }

  // Final metadata update using pre-emit chunk counts.
  if (s->shard_sink->update_append) {
    const uint8_t na = dim_info_n_append(&s->dims);
    for (int lv = 0; lv < s->levels.nlod; ++lv) {
      uint64_t append_sizes[HALF_MAX_RANK];
      dim_info_decompose_append_sizes(&s->dims, append_chunks[lv], append_sizes);
      if (s->shard_sink->update_append(
            s->shard_sink, (uint8_t)lv, na, append_sizes))
        return writer_error();
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s)
{
  s->writer.append = tile_stream_gpu_append;
  s->writer.flush = tile_stream_gpu_flush;
}

#include "gpu/stream.flush.h"
#include "gpu/stream.ingest.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "log/log.h"
#include "platform/platform.h"
#include "stream/config.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <string.h>

// Forward declarations for tile_stream_gpu wrappers
static struct writer_result
tile_stream_gpu_append(struct writer* self, struct slice input);
static struct writer_result
tile_stream_gpu_flush_final(struct writer* self);

// --- Shared helpers (engine + context) ---

struct stream_metrics
stream_engine_init_metrics(int enable_multiscale)
{
  return (struct stream_metrics){
    .memcpy = mk_stream_metric("Memcpy"),
    .h2d = mk_stream_metric("H2D"),
    .scatter = mk_stream_metric(enable_multiscale ? "Copy" : "Scatter"),
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

void*
stream_engine_pool_epoch(struct stream_engine* e,
                         struct stream_context* ctx,
                         uint32_t epoch_in_batch)
{
  const size_t bpe = dtype_bpe(ctx->config.dtype);
  return (char*)e->pools.buf[e->pools.current] +
         (uint64_t)epoch_in_batch * ctx->levels.total_chunks *
           ctx->layout.chunk_stride * bpe;
}

static int
engine_dispatch_ingest(struct stream_engine* e, struct stream_context* ctx)
{
  if (ctx->levels.enable_multiscale) {
    return ingest_dispatch_multiscale(&e->stage,
                                      e->lod_shared.d_linear,
                                      ctx->layout.epoch_elements,
                                      &ctx->cursor_elements,
                                      dtype_bpe(ctx->config.dtype),
                                      e->streams.h2d,
                                      e->streams.compute);
  } else {
    return ingest_dispatch_scatter(
      &e->stage,
      &ctx->layout,
      &ctx->layout_gpu,
      stream_engine_pool_epoch(e, ctx, e->batch.accumulated),
      e->pools.ready[e->pools.current],
      &ctx->cursor_elements,
      dtype_bpe(ctx->config.dtype),
      e->streams.h2d,
      e->streams.compute);
  }
}

// --- Shared append body ---

struct writer_result
stream_append_body(struct stream_engine* e,
                   struct stream_context* ctx,
                   struct slice input)
{
  const size_t bytes_per_element = dtype_bpe(ctx->config.dtype);
  const size_t buffer_capacity = ctx->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  const uint64_t max_cursor_elements = ctx->max_cursor_elements;

  while (src < end) {
    // Bounded append dims: check capacity
    if (max_cursor_elements > 0 &&
        ctx->cursor_elements >= max_cursor_elements) {
      struct writer_result fr = stream_flush_body(e, ctx);
      if (fr.error)
        return writer_error_at(src, end);
      return writer_finished_at(src, end);
    }

    const uint64_t epoch_remaining =
      ctx->layout.epoch_elements -
      (ctx->cursor_elements % ctx->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bytes_per_element;
    uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;

    // Bounded append dims: clamp to remaining capacity
    if (max_cursor_elements > 0) {
      const uint64_t remaining_capacity =
        max_cursor_elements - ctx->cursor_elements;
      if (elements_this_pass > remaining_capacity)
        elements_this_pass = remaining_capacity;
    }

    const uint64_t bytes_this_pass = elements_this_pass * bytes_per_element;

    {
      uint64_t written = 0;
      while (written < bytes_this_pass) {
        const size_t space = buffer_capacity - e->stage.bytes_written;
        const uint64_t remaining = bytes_this_pass - written;
        const size_t payload = space < remaining ? space : (size_t)remaining;

        if (e->stage.bytes_written == 0) {
          const int si = e->stage.current;
          struct staging_slot* ss = &e->stage.slot[si];
          CU(Error, cuEventSynchronize(ss->t_h2d_end));

          if (ctx->cursor_elements > 0) {
            accumulate_metric_cu(&e->metrics.h2d,
                                 ss->t_h2d_start,
                                 ss->t_h2d_end,
                                 ss->dispatched_bytes,
                                 ss->dispatched_bytes);
            accumulate_metric_cu(&e->metrics.scatter,
                                 ss->t_scatter_start,
                                 ss->t_scatter_end,
                                 ss->dispatched_bytes,
                                 ss->dispatched_bytes);
          }
        }

        {
          struct platform_clock mc = { 0 };
          platform_toc(&mc);
          memcpy((uint8_t*)e->stage.slot[e->stage.current].h_in +
                   e->stage.bytes_written,
                 src + written,
                 payload);
          accumulate_metric_ms(&e->metrics.memcpy,
                               (float)(platform_toc(&mc) * 1000.0),
                               payload,
                               payload);
        }
        e->stage.bytes_written += payload;
        written += payload;

        if (e->stage.bytes_written == buffer_capacity ||
            written == bytes_this_pass) {
          CHECK(Error, engine_dispatch_ingest(e, ctx) == 0);
          e->stage.bytes_written = 0;
        }
      }
    }
    src += bytes_this_pass;

    if (ctx->cursor_elements % ctx->layout.epoch_elements == 0 &&
        ctx->cursor_elements > 0) {
      struct writer_result fr = flush_accumulate_epoch(e, ctx);
      if (fr.error)
        return writer_error_at(src, end);
      // Sample sink backpressure at epoch boundaries.
      size_t pend = shard_sink_pending_bytes(ctx->sink);
      if (pend > e->metrics.peak_pending_bytes)
        e->metrics.peak_pending_bytes = pend;
      // Backpressure: if the sink's IO queue exceeds the watermark,
      // poll here until it drains below the threshold.
      if (ctx->config.backpressure_bytes > 0 &&
          pend > ctx->config.backpressure_bytes) {
        struct platform_clock bp_clk = { 0 };
        platform_toc(&bp_clk);
        int64_t start_ns = bp_clk.last_ns;
        const double timeout_s = 30.0;
        int drained = 0;
        for (;;) {
          if (shard_sink_pending_bytes(ctx->sink) <=
              ctx->config.backpressure_bytes) {
            drained = 1;
            break;
          }
          platform_toc(&bp_clk);
          if ((bp_clk.last_ns - start_ns) / 1e9 >= timeout_s)
            break;
          platform_sleep_ns(100000); // 100 µs
        }
        platform_toc(&bp_clk);
        float bp_ms = (float)((bp_clk.last_ns - start_ns) / 1e6);
        accumulate_metric_ms(&e->metrics.backpressure, bp_ms, 0, 0);
        if (!drained)
          log_warn("backpressure timeout after %.1fs (pending %zu bytes)",
                   timeout_s,
                   shard_sink_pending_bytes(ctx->sink));
      }
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

// --- Shared flush body ---

struct writer_result
stream_flush_body(struct stream_engine* e, struct stream_context* ctx)
{
  if (e->stage.bytes_written > 0) {
    if (engine_dispatch_ingest(e, ctx))
      return writer_error();
    e->stage.bytes_written = 0;
  }

  struct writer_result r = flush_drain_pending(e, ctx);
  if (r.error)
    return r;

  // Flush any partial epoch first (sub-epoch data)
  if (ctx->cursor_elements % ctx->layout.epoch_elements != 0) {
    if (flush_run_epoch_lod(e, ctx))
      return writer_error();
    CU(Error,
       cuEventRecord(e->batch.pool_events[e->batch.accumulated],
                     e->streams.compute));
    e->batch.accumulated++;
  }

  // Flush any accumulated epochs (partial batch)
  r = flush_accumulated_sync(e, ctx);
  if (r.error)
    return r;

  // Drain any partial append accumulators
  r = flush_partial_append(e, ctx);
  if (r.error)
    return r;

  // Emit partial shards for all levels
  for (int lv = 0; lv < ctx->levels.nlod; ++lv) {
    if (e->compress_agg.levels[lv].shard.epoch_in_shard > 0) {
      if (finalize_shards(&e->compress_agg.levels[lv].shard,
                          ctx->shard_alignment))
        return writer_error();
    }
  }

  // Final metadata update using pre-emit chunk counts.
  if (ctx->sink->update_append) {
    const uint8_t na = dim_info_n_append(&ctx->dims);
    for (int lv = 0; lv < ctx->levels.nlod; ++lv) {
      uint64_t append_sizes[HALF_MAX_RANK];
      dim_info_final_append_sizes(
        &ctx->dims, ctx->cursor_elements, lv, append_sizes);
      if (ctx->sink->update_append(ctx->sink, (uint8_t)lv, na, append_sizes))
        return writer_error();
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

// --- Accessor ---

struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s)
{
  return s->engine.metrics;
}

// --- tile_stream_gpu writer wrappers ---

static struct writer_result
tile_stream_gpu_append(struct writer* self, struct slice input)
{
  struct tile_stream_gpu* s =
    container_of(self, struct tile_stream_gpu, writer);

  if (s->flushed)
    return writer_finished_at(input.beg, input.end);

  struct platform_clock clk = { 0 };
  platform_toc(&clk);
  struct writer_result r = stream_append_body(&s->engine, &s->ctx, input);
  float ms = (float)(platform_toc(&clk) * 1000.0);
  if (ms > s->engine.metrics.max_append_ms)
    s->engine.metrics.max_append_ms = ms;
  return r;
}

static struct writer_result
tile_stream_gpu_flush_final(struct writer* self)
{
  struct tile_stream_gpu* s =
    container_of(self, struct tile_stream_gpu, writer);
  struct writer_result r = stream_flush_body(&s->engine, &s->ctx);
  s->flushed = 1;
  return r;
}

void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s)
{
  s->writer.append = tile_stream_gpu_append;
  s->writer.flush = tile_stream_gpu_flush_final;
}

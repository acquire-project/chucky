#include "gpu/schedule.h"
#include "gpu/stream.ingest.h"
#include "gpu/stream.lod.h"

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
    .memcpy = mk_stream_metric("Memcpy", METRIC_OWNER_PRODUCER),
    .h2d = mk_stream_metric("H2D", METRIC_OWNER_H2D),
    .scatter = mk_stream_metric(enable_multiscale ? "Copy" : "Scatter",
                                METRIC_OWNER_COMPUTE),
    .lod_gather = mk_stream_metric("LOD Gather", METRIC_OWNER_COMPUTE),
    .lod_reduce = mk_stream_metric("LOD Reduce", METRIC_OWNER_COMPUTE),
    .lod_append_fold = mk_stream_metric("Append Fold", METRIC_OWNER_COMPUTE),
    .lod_morton_chunk = mk_stream_metric("LOD to chunks", METRIC_OWNER_COMPUTE),
    .compress = mk_stream_metric("Compress", METRIC_OWNER_COMPRESS),
    .aggregate = mk_stream_metric("Aggregate", METRIC_OWNER_COMPRESS),
    .d2h = mk_stream_metric("D2H", METRIC_OWNER_D2H),
    .sink = mk_stream_metric("Sink", METRIC_OWNER_DRAIN),
    .flush_stall = mk_stream_metric("FlushStall", METRIC_OWNER_PRODUCER),
    .drain_dispatch = mk_stream_metric("DrainDisp", METRIC_OWNER_DRAIN),
    .io_fence_stall = mk_stream_metric("IOFence", METRIC_OWNER_PRODUCER),
    .backpressure = mk_stream_metric("Backpres", METRIC_OWNER_PRODUCER),
    .tail_gate = mk_stream_metric("TailGate", METRIC_OWNER_COMPRESS),
  };
}

void
stream_engine_attach_edge_stalls(struct stream_engine* e)
{
  e->metrics.edge_stall[0] =
    mk_stream_metric("StagingFree", METRIC_OWNER_PRODUCER);
  e->metrics.edge_stall[1] = mk_stream_metric("ChunkIndex", METRIC_OWNER_DRAIN);
  e->metrics.edge_stall[2] = mk_stream_metric("D2HDone", METRIC_OWNER_DRAIN);
  gpu_ordering_attach_stall_metric(
    &e->ord, GPU_EDGE_STAGING_FREE, &e->metrics.edge_stall[0]);
  gpu_ordering_attach_stall_metric(
    &e->ord, GPU_EDGE_CHUNK_INDEX_READY, &e->metrics.edge_stall[1]);
  gpu_ordering_attach_stall_metric(
    &e->ord, GPU_EDGE_D2H_DONE, &e->metrics.edge_stall[2]);
}

static size_t
pool_epoch_bytes(const struct stream_context* ctx)
{
  return (size_t)ctx->levels.total_chunks * ctx->layout.chunk_stride *
         dtype_bpe(ctx->config.dtype);
}

struct gpu_pool_view
stream_engine_pool_epoch(struct stream_engine* e,
                         struct stream_context* ctx,
                         uint32_t epoch_in_batch)
{
  return gpu_pool_at(
    &e->pools.p, e->sched.fill, (size_t)epoch_in_batch * pool_epoch_bytes(ctx));
}

// The append cursor counts everything the caller has passed in, including what
// is still sitting in staging.
static uint64_t
dispatched_elements(const struct stream_engine* e,
                    const struct stream_context* ctx)
{
  return ctx->cursor_elements -
         e->stage.bytes_written / dtype_bpe(ctx->config.dtype);
}

static int
engine_dispatch_ingest(struct stream_engine* e,
                       struct stream_context* ctx,
                       uint64_t first_element)
{
  if (ctx->levels.enable_multiscale) {
    return ingest_dispatch_multiscale(&e->stage,
                                      e->lod_shared.d_linear,
                                      ctx->layout.epoch_elements,
                                      first_element,
                                      dtype_bpe(ctx->config.dtype),
                                      e->streams.h2d,
                                      e->streams.compute);
  } else {
    const struct scatter_destination dst = {
      .first_epoch = stream_engine_pool_epoch(e, ctx, e->sched.accumulated),
      .epoch_bytes = pool_epoch_bytes(ctx),
      .epoch_elements = ctx->layout.epoch_elements,
      .epochs = e->sched.epochs_per_batch - e->sched.accumulated,
    };
    return ingest_dispatch_scatter(&e->stage,
                                   &ctx->layout,
                                   &ctx->layout_gpu,
                                   dst,
                                   first_element,
                                   dtype_bpe(ctx->config.dtype),
                                   e->streams.h2d,
                                   e->streams.compute);
  }
}

struct writer_result
stream_dispatch_staged(struct stream_engine* e, struct stream_context* ctx)
{
  if (ctx->append_failed)
    return writer_error();
  if (e->stage.bytes_written == 0)
    return writer_ok();

  const uint64_t first = dispatched_elements(e, ctx);
  if (engine_dispatch_ingest(e, ctx, first)) {
    ctx->append_failed = 1;
    // Staging is shared with the stream's other arrays, and these bytes are
    // never going to reach this one's pool, so let go of them.
    e->stage.bytes_written = 0;
    return writer_error();
  }
  e->stage.bytes_written = 0;

  const uint64_t epoch = ctx->layout.epoch_elements;
  for (uint64_t at = first - first % epoch + epoch; at <= ctx->cursor_elements;
       at += epoch) {
    struct writer_result r = schedule_accumulate_epoch(e, ctx);
    if (r.error) {
      ctx->append_failed = 1;
      return r;
    }
  }
  return writer_ok();
}

// The scatter writes into the epochs of the batch being filled, so a dispatch
// cannot hold more than the room left in that batch. Multiscale takes one epoch
// at a time, since its linear buffer holds exactly one.
static size_t
bytes_before_dispatch(const struct stream_engine* e,
                      const struct stream_context* ctx)
{
  const uint64_t epoch = ctx->layout.epoch_elements;
  const uint32_t epochs = ctx->levels.enable_multiscale
                            ? 1
                            : e->sched.epochs_per_batch - e->sched.accumulated;
  const uint64_t room =
    ((uint64_t)epochs * epoch - dispatched_elements(e, ctx) % epoch) *
    dtype_bpe(ctx->config.dtype);
  const size_t capacity = ctx->config.buffer_capacity_bytes;
  return room < capacity ? (size_t)room : capacity;
}

static void
apply_backpressure(struct stream_engine* e, struct stream_context* ctx)
{
  size_t pend = shard_sink_pending_bytes(ctx->sink);
  if (pend > e->metrics.peak_pending_bytes)
    e->metrics.peak_pending_bytes = pend;
  if (ctx->config.backpressure_bytes == 0 ||
      pend <= ctx->config.backpressure_bytes)
    return;

  struct platform_clock bp_clk = { 0 };
  platform_toc(&bp_clk);
  int64_t start_ns = bp_clk.last_ns;
  const double timeout_s = 30.0;
  int drained = 0;
  for (;;) {
    if (shard_sink_pending_bytes(ctx->sink) <= ctx->config.backpressure_bytes) {
      drained = 1;
      break;
    }
    platform_toc(&bp_clk);
    if ((bp_clk.last_ns - start_ns) / 1e9 >= timeout_s)
      break;
    platform_sleep_ns(100000); // 100 µs
  }
  platform_toc(&bp_clk);
  accumulate_metric_ms(
    &e->metrics.backpressure, (float)((bp_clk.last_ns - start_ns) / 1e6), 0, 0);
  if (!drained)
    log_warn("backpressure timeout after %.1fs (pending %zu bytes)",
             timeout_s,
             shard_sink_pending_bytes(ctx->sink));
}

// --- Shared append body ---

struct writer_result
stream_append_body(struct stream_engine* e,
                   struct stream_context* ctx,
                   struct slice input)
{
  const size_t bpe = dtype_bpe(ctx->config.dtype);
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  const uint64_t total_limit = ctx->total_element_limit;

  if (ctx->append_failed)
    return writer_error_at(src, end);
  // A failed kick leaves the batch full without swapping the slot, so the
  // epochs the pool holds are all accounted for and there is nowhere to put
  // more. Only a flush that succeeds can clear it.
  if (e->sched.accumulated >= e->sched.epochs_per_batch)
    return writer_error_at(src, end);

  while (src < end) {
    // Capacity reached: refuse further writes and report `finished` with the
    // remaining input unconsumed. Sink finalization is NOT run here — it
    // happens on explicit `writer_flush` or on stream destroy.
    if (total_limit > 0 && ctx->cursor_elements >= total_limit)
      return writer_finished_at(src, end);

    const size_t target = bytes_before_dispatch(e, ctx);
    uint64_t elements = (target - e->stage.bytes_written) / bpe;
    const uint64_t offered = (uint64_t)(end - src) / bpe;
    if (elements > offered)
      elements = offered;
    if (total_limit > 0 && elements > total_limit - ctx->cursor_elements)
      elements = total_limit - ctx->cursor_elements;
    if (elements == 0)
      break;

    const size_t payload = (size_t)(elements * bpe);

    if (e->stage.bytes_written == 0) {
      // Poll instead of cuEventSynchronize to keep the producer thread hot —
      // it has memcpy work queued up immediately after.
      if (gpu_pool_host_acquire_produce(
            &e->stage.h_pool, e->stage.current, NULL))
        return writer_error_at(src, end);

      // The acquire above waited on this slot's H2D, so its interval is
      // ready. The scatter runs afterwards on the compute stream, so its
      // samples come from the ring whenever they finish.
      ingest_collect_h2d_timing(&e->stage, &e->metrics.h2d);
      ingest_collect_scatter_timing(&e->stage, &e->metrics.scatter);
    }

    {
      struct platform_clock mc = { 0 };
      platform_toc(&mc);
      // Same h_in generation as the bytes_written==0 acquire above.
      ingest_copy(
        e->copy_pool,
        gpu_pool_at(&e->stage.h_pool, e->stage.current, e->stage.bytes_written)
          .p,
        src,
        payload);
      accumulate_metric_ms(&e->metrics.memcpy,
                           (float)(platform_toc(&mc) * 1000.0),
                           payload,
                           payload);
    }
    e->stage.bytes_written += payload;
    ctx->cursor_elements += elements;
    src += payload;

    if (e->stage.bytes_written < target)
      continue;

    struct writer_result r = stream_dispatch_staged(e, ctx);
    if (r.error)
      return writer_error_at(src, end);
    apply_backpressure(e, ctx);
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };
}

// --- Shared flush body ---

// The epochs the schedule has counted but not yet handed to compression.
static struct writer_result
finish_accumulated(struct stream_engine* e, struct stream_context* ctx)
{
  if (ctx->cursor_elements % ctx->layout.epoch_elements != 0 &&
      schedule_add_partial_epoch(e, ctx)) {
    ctx->append_failed = 1;
    return writer_error();
  }

  struct writer_result r = schedule_flush_accumulated(e, ctx);
  if (r.error) {
    // The batch stays counted and unkicked, and nothing can say which of its
    // epochs reached the sink.
    ctx->append_failed = 1;
    return r;
  }

  r = schedule_flush_partial_append(e, ctx);
  if (r.error)
    ctx->append_failed = 1;
  return r;
}

// A shard's index claims every chunk it names is present, and the array's shape
// claims every element up to the append cursor is. Both only hold once
// everything before them succeeded: a reader cannot tell a complete array from
// one whose tail never arrived, so after a failure the output is left short
// rather than made to look whole.
static struct writer_result
finalize_output_shards(struct stream_engine* e, struct stream_context* ctx)
{
  struct writer_result r = writer_ok();
  for (int lv = 0; lv < ctx->levels.nlod; ++lv) {
    if (e->compress_agg.ar.shard[lv].epoch_in_shard > 0 &&
        finalize_shards(
          &e->compress_agg.ar.shard[lv], ctx->sink, ctx->shard_alignment))
      r = writer_error();
  }
  return r;
}

static struct writer_result
publish_array_shape(struct stream_context* ctx)
{
  if (!ctx->sink->update_append)
    return writer_ok();

  struct writer_result r = writer_ok();
  const uint8_t na = dim_info_n_append(&ctx->dims);
  for (int lv = 0; lv < ctx->levels.nlod; ++lv) {
    uint64_t append_sizes[HALF_MAX_RANK];
    dim_info_final_append_sizes(
      &ctx->dims, ctx->cursor_elements, lv, append_sizes);
    if (ctx->sink->update_append(ctx->sink, (uint8_t)lv, na, append_sizes))
      r = writer_error();
  }
  return r;
}

struct writer_result
stream_flush_body(struct stream_engine* e, struct stream_context* ctx)
{
  // A create that fails before sizing the layout leaves epoch_elements at 0;
  // the divisions below would then fault. Nothing was ever sized, so there is
  // nothing to flush.
  if (ctx->layout.epoch_elements == 0)
    return writer_ok();

  // Returns the failure without touching staging once the array has failed.
  struct writer_result r = stream_dispatch_staged(e, ctx);

  // Whatever happened above, batches already handed to the delivery worker have
  // to be joined: the worker writes the shard state this function goes on to
  // read, and its queued writes point into buffers destroy frees.
  {
    struct writer_result d = schedule_drain_kicked(e, ctx);
    if (d.error)
      ctx->append_failed = 1;
    if (!r.error)
      r = d;
  }

  if (!r.error)
    r = finish_accumulated(e, ctx);

  // Last chance to read the outstanding ingest intervals; flush is already a
  // sync point, so waiting here costs nothing and leaves no sample unread.
  cuStreamSynchronize(e->streams.compute);
  ingest_collect_h2d_timing(&e->stage, &e->metrics.h2d);
  ingest_collect_scatter_timing(&e->stage, &e->metrics.scatter);
  lod_collect_timing(&e->lod_shared, &e->metrics);
  e->metrics.scatter_samples_lost = e->stage.scatter_samples_lost;
  e->metrics.lod_samples_lost = e->lod_shared.timing_samples_lost;
  if (e->lod_shared.timing_samples_lost > 0)
    log_debug("lod timing ring wrapped %llu times",
              (unsigned long long)e->lod_shared.timing_samples_lost);
  if (e->stage.scatter_samples_lost > 0)
    log_debug("scatter timing ring wrapped %llu times",
              (unsigned long long)e->stage.scatter_samples_lost);

  if (!r.error)
    r = finalize_output_shards(e, ctx);

  // Queued writes point into buffers destroy frees, so drain on every path. The
  // shape below is written after it, so it never names data that is still
  // queued.
  if (shard_sink_drain(ctx->sink) && !r.error)
    r = writer_error();

  if (!r.error)
    r = publish_array_shape(ctx);

  if (!r.error && ctx->sink->flush && ctx->sink->flush(ctx->sink))
    r = writer_error();

  return r;
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

  struct platform_clock clk = { 0 };
  platform_toc(&clk);
  struct writer_result r = stream_append_body(&s->engine, &s->ctx, input);
  float ms = (float)(platform_toc(&clk) * 1000.0);
  if (ms > s->engine.metrics.max_append_ms)
    s->engine.metrics.max_append_ms = ms;
  if (r.rest.beg != input.beg)
    record_append_ms(&s->engine.metrics, ms);
  if (s->flushed && r.rest.beg != input.beg)
    s->flushed = 0;
  return r;
}

static struct writer_result
tile_stream_gpu_flush_final(struct writer* self)
{
  struct tile_stream_gpu* s =
    container_of(self, struct tile_stream_gpu, writer);
  // Idempotency: mirrors the CPU guard. Reset by tile_stream_gpu_append when
  // new data arrives.
  if (s->flushed)
    return writer_ok();
  struct writer_result r = stream_flush_body(&s->engine, &s->ctx);
  if (r.error == 0)
    s->flushed = 1;
  return r;
}

void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s)
{
  s->writer.append = tile_stream_gpu_append;
  s->writer.flush = tile_stream_gpu_flush_final;
}

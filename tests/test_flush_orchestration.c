#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/schedule.h"
#include "platform/platform.h"
#include "stream/config.h"
#include "stream/dim_info.h"

#include "test_gpu_helpers.h"
#include "test_shard_sink.h"

#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#include "test_runner.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Orchestration test context: assembles tile_stream_gpu from individual parts
// ---------------------------------------------------------------------------

struct orch_ctx
{
  struct computed_stream_layouts cl;
  struct tile_stream_gpu* s;
};

static void
orch_ctx_init(struct orch_ctx* c)
{
  memset(c, 0, sizeof(*c));
}

static void
orch_ctx_destroy(struct orch_ctx* c)
{
  if (c->s) {
    for (int fc = 0; fc < 2; ++fc) {
      free(c->s->engine.sched.slot[fc].batch_active_masks);
      c->s->engine.sched.slot[fc].batch_active_masks = NULL;
    }

    d2h_deliver_destroy(&c->s->engine.d2h_deliver);
    compress_agg_destroy(&c->s->engine.compress_agg);

    // Pools
    for (int i = 0; i < 2; ++i)
      cu_mem_free(c->s->engine.pools.buf[i]);

    gpu_ordering_destroy(&c->s->engine.ord);
    gpu_streams_destroy(&c->s->engine.streams);

    free(c->s);
    c->s = NULL;
  }

  computed_stream_layouts_free(&c->cl);
}

// Set up all components for the flush orchestration test. shard_alignment 0
// models a sink with no alignment requirement; both aligned and unaligned
// single-stream fixtures retain the depth-two compact-aggregate schedule.
static int
orch_ctx_setup_aligned(struct orch_ctx* c,
                       struct tile_stream_configuration* config,
                       struct shard_sink* sink,
                       size_t shard_alignment)
{
  CHECK(Fail,
        compute_stream_layouts(config,
                               codec_alignment(config->codec.id),
                               codec_max_output_size,
                               shard_alignment,
                               &c->cl) == 0);

  c->s = (struct tile_stream_gpu*)calloc(1, sizeof(*c->s));
  CHECK(Fail, c->s);

  c->s->ctx.config = *config;
  c->s->ctx.sink = sink;
  c->s->ctx.levels = c->cl.levels;
  c->s->ctx.layout = c->cl.layouts[0];
  CHECK(Fail,
        dim_info_init(&c->s->ctx.dims, config->dimensions, config->rank) == 0);

  const uint32_t K = c->cl.epochs_per_batch;
  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  const size_t pool_bytes =
    (uint64_t)K * total_chunks * chunk_stride * bytes_per_element;

  CHECK(Fail, gpu_streams_init(&c->s->engine.streams) == 0);
  CHECK(Fail,
        gpu_ordering_init(&c->s->engine.ord, c->s->engine.streams.compute) ==
          0);
  gpu_streams_register(&c->s->engine.streams, &c->s->engine.ord);

  // Compress+aggregate stage
  CHECK(Fail,
        compress_agg_init(&c->s->engine.compress_agg,
                          &c->cl,
                          config,
                          &c->s->engine.ord,
                          c->s->engine.streams.compute) == 0);

  // D2H+deliver stage
  CHECK(Fail,
        d2h_deliver_init(&c->s->engine.d2h_deliver,
                         platform_page_alignment(),
                         config->codec.id == CODEC_NONE
                           ? DEVICE_AGGREGATE_FIXED_EXTENT
                           : DEVICE_AGGREGATE_INDEXED_EXTENT,
                         &c->s->engine.ord,
                         c->s->engine.streams.drain,
                         c->s->engine.streams.compute) == 0);

  // Double-buffered chunk pools
  gpu_pool_init(&c->s->engine.pools.p,
                &c->s->engine.ord,
                GPU_EDGE_POOL_FILLED,
                GPU_EDGE_POOL_CONSUMED);
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&c->s->engine.pools.buf[i], pool_bytes));
    gpu_pool_bind(
      &c->s->engine.pools.p, i, (void*)(uintptr_t)c->s->engine.pools.buf[i]);
    CU(Fail,
       cuMemsetD8Async(c->s->engine.pools.buf[i],
                       0,
                       pool_bytes,
                       c->s->engine.streams.compute));
  }
  // This fixture has no delivery worker; single-array scheduling still uses
  // depth two and drains the oldest slot inline on reuse.
  c->s->engine.sched.epochs_per_batch = K;
  for (int fc = 0; fc < 2; ++fc) {
    c->s->engine.sched.slot[fc].batch_active_masks =
      (uint32_t*)calloc(K, sizeof(uint32_t));
    CHECK(Fail, c->s->engine.sched.slot[fc].batch_active_masks);
  }
  schedule_select(&c->s->engine.sched, &c->s->engine.delivery);

  // Non-multiscale: zeroed lod
  memset(&c->s->engine.lod, 0, sizeof(c->s->engine.lod));

  memset(&c->s->engine.metrics, 0, sizeof(c->s->engine.metrics));
  c->s->engine.metrics.compress =
    mk_stream_metric("Compress", METRIC_OWNER_COMPRESS);
  c->s->engine.metrics.aggregate =
    mk_stream_metric("Aggregate", METRIC_OWNER_COMPRESS);
  c->s->engine.metrics.d2h = mk_stream_metric("D2H", METRIC_OWNER_D2H);
  c->s->engine.metrics.sink = mk_stream_metric("Sink", METRIC_OWNER_DRAIN);
  c->s->engine.metrics.lod_gather =
    mk_stream_metric("LOD Gather", METRIC_OWNER_COMPUTE);

  memset(&c->s->engine.metadata_update_clock,
         0,
         sizeof(c->s->engine.metadata_update_clock));

  CU(Fail, cuStreamSynchronize(c->s->engine.streams.compute));
  return 0;

Fail:
  return 1;
}

static int
orch_ctx_setup(struct orch_ctx* c,
               struct tile_stream_configuration* config,
               struct shard_sink* sink)
{
  return orch_ctx_setup_aligned(c, config, sink, platform_page_alignment());
}

// Fill one epoch in the current pool. Syncs compute stream first to ensure
// any pending pool zeroing is complete.
static int
orch_ctx_fill_epoch(struct orch_ctx* c,
                    uint32_t epoch_in_batch,
                    const struct tile_stream_configuration* config,
                    uint16_t (*fill_fn)(uint64_t))
{
  CU(Fail, cuStreamSynchronize(c->s->engine.streams.compute));

  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  CUdeviceptr epoch_ptr =
    c->s->engine.pools.buf[c->s->engine.sched.fill] +
    (uint64_t)epoch_in_batch * total_chunks * chunk_stride * bytes_per_element;
  return fill_pool_epoch(
    epoch_ptr, total_chunks, chunk_stride, bytes_per_element, fill_fn);

Fail:
  return 1;
}

// ---------------------------------------------------------------------------
// Test 1: Accumulate one epoch into K=2 batch — no flush triggered
// ---------------------------------------------------------------------------
static int
test_accumulate_one_epoch(void)
{
  log_info("=== test_accumulate_one_epoch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  // Fill epoch 0 in current pool
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);

  // Accumulate epoch
  struct writer_result r = schedule_accumulate_epoch(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);

  // Verify: mid-batch, no flush triggered
  CHECK(Fail, c.s->engine.sched.accumulated == 1);
  CHECK(Fail, c.s->engine.sched.fill == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 0);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);

  // Epoch mask recorded
  CHECK(Fail, c.s->engine.sched.slot[0].batch_active_masks[0] == 0x1);
  CHECK(Fail, c.s->engine.sched.slot[0].active_levels_mask == 0x1);

  // Sink not touched
  CHECK(Fail, sink.open_count == 0);
  CHECK(Fail, sink.finalize_count == 0);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: Full batch triggers auto-flush + pool swap
// ---------------------------------------------------------------------------
static int
test_full_batch_auto_flush(void)
{
  log_info("=== test_full_batch_auto_flush ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // Fill and accumulate 2 epochs
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.sched.accumulated == 1);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  // After full batch: drain_kick_and_swap fired
  CHECK(Fail, c.s->engine.sched.accumulated == 0);
  CHECK(Fail, c.s->engine.sched.fill == 1);           // swapped to pool 1
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1); // batch 1 pending at fc=0
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);

  // Fresh pool slot is reset
  CHECK(Fail, c.s->engine.sched.slot[1].active_levels_mask == 0);

  // Sink not yet written (D2H kicked but not drained)
  CHECK(Fail, sink.open_count == 0);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 3: Full batch + drain → data arrives in sink
// ---------------------------------------------------------------------------
static int
test_drain_delivers_data(void)
{
  log_info("=== test_drain_delivers_data ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // Fill and accumulate 2 epochs (full batch → auto-kick)
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1);

  // Drain the pending batch
  struct writer_result r = schedule_drain_kicked(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 0);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);

  // Data delivered: shard opened and finalized (tps_0=2, 2 epochs → complete)
  CHECK(Fail, sink.open_count >= 1);
  CHECK(Fail, sink.finalize_count == 1);
  CHECK(Fail, sink.writers[0][0].size > 0);

  // Sink metric always recorded (uses platform_toc, not CUDA events)
  CHECK(Fail, c.s->engine.metrics.sink.count == 1);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 4: schedule_flush_accumulated with partial batch (1 epoch, K=2)
// ---------------------------------------------------------------------------
static int
test_accumulated_sync_partial(void)
{
  log_info("=== test_accumulated_sync_partial ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // Fill and accumulate 1 epoch (partial batch)
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.sched.accumulated == 1);

  // Sync flush: processes the partial batch (per-epoch path)
  struct writer_result r = schedule_flush_accumulated(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);

  // Batch drained
  CHECK(Fail, c.s->engine.sched.accumulated == 0);

  // Data delivered into the pipeline. With tail-carryover delivery, a partial
  // batch's data is staged in the in-memory tail (writes to disk only happen
  // at page-aligned boundaries / on shard finalize), so verify shard state
  // rather than on-disk size.
  CHECK(Fail, sink.open_count >= 1);
  {
    struct shard_state* ss = &c.s->engine.compress_agg.ar.shard[0];
    struct active_shard* sh = &ss->shards[0];
    CHECK(Fail, sh->tail_bytes > 0);
  }

  // Sink metric recorded (platform_toc, not CUDA events — always fires)
  CHECK(Fail, c.s->engine.metrics.sink.count == 1);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 5: Two full batch cycles — verifies pool swap dance
// ---------------------------------------------------------------------------
static int
test_two_batch_cycle(void)
{
  log_info("=== test_two_batch_cycle ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // --- Batch 1: epochs 0,1 on pool 0 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  // Batch 1 kicked, pool swapped to 1, batch 1 pending at fc=0
  CHECK(Fail, c.s->engine.sched.fill == 1);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);
  CHECK(Fail, c.s->engine.sched.accumulated == 0);

  // --- Batch 2: epochs 2,3 on pool 1 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch2) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch3) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  // Page alignment no longer reduces schedule depth. With the worker absent,
  // both compact aggregates remain outstanding until the producer performs
  // the ordered materialization drain.
  CHECK(Fail, c.s->engine.sched.fill == 0);           // swapped back to pool 0
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1); // batch 1 pending
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 1); // batch 2 pending
  CHECK(Fail, c.s->engine.sched.accumulated == 0);
  CHECK(Fail, c.s->engine.metrics.sink.count == 0);

  // Drain both batches oldest-first; committed tail state from batch 1 places
  // batch 2's payload before either aggregate slot is reused.
  struct writer_result r = schedule_drain_kicked(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 0);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);

  // Both shards finalized (tps_0=2, 2 epochs each → 2 shards)
  CHECK(Fail, sink.finalize_count >= 2);

  // Sink metric: 2 batch drains
  CHECK(Fail, c.s->engine.metrics.sink.count == 2);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Depth two: a kicked batch stays pending across a slot swap and is drained
// oldest-first later, rather than being drained before the next kick.
static int
test_two_batch_cycle_pipelined(void)
{
  log_info("=== test_two_batch_cycle_pipelined ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup_aligned(&c, &config, &sink.base, 0) == 0);
  CHECK(Fail, c.s->engine.sched.mode == SCHEDULE_PIPELINED_DIRECT);

  // --- Batch 1: epochs 0,1 on pool 0 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  CHECK(Fail, c.s->engine.sched.fill == 1);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);

  // --- Batch 2: epochs 2,3 on pool 1 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch2) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch3) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  // Both batches are outstanding at once; nothing was delivered yet.
  CHECK(Fail, c.s->engine.sched.fill == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 1);
  CHECK(Fail, c.s->engine.sched.accumulated == 0);
  CHECK(Fail, c.s->engine.metrics.sink.count == 0);
  CHECK(Fail,
        c.s->engine.sched.slot[0].generation <
          c.s->engine.sched.slot[1].generation);

  struct writer_result r = schedule_drain_kicked(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 0);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);
  CHECK(Fail, sink.finalize_count >= 2);
  CHECK(Fail, c.s->engine.metrics.sink.count == 2);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// A direct-mode stream can lose its delivery worker after construction.  A
// partial flush must still drain an older full batch before resolving the
// partial batch's committed tail placement.
static int
test_partial_flush_inline_oldest_first(void)
{
  log_info("=== test_partial_flush_inline_oldest_first ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup_aligned(&c, &config, &sink.base, 0) == 0);
  gpu_delivery_stop_join(&c.s->engine.delivery);
  schedule_select(&c.s->engine.sched, &c.s->engine.delivery);
  CHECK(Fail, c.s->engine.sched.mode == SCHEDULE_PIPELINED_DIRECT);

  // Full generation 1 remains queued in slot 0.
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 1);

  // Generation 2 is partial in slot 1. The flush kicks it, then drains both
  // generations oldest-first even though no worker is available.
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch2) == 0);
  CHECK(Fail, schedule_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, schedule_flush_accumulated(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.sched.slot[0].kicked == 0);
  CHECK(Fail, c.s->engine.sched.slot[1].kicked == 0);
  CHECK(Fail, c.s->engine.metrics.sink.count == 2);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "accumulate_one_epoch", test_accumulate_one_epoch },
              { "full_batch_auto_flush", test_full_batch_auto_flush },
              { "drain_delivers_data", test_drain_delivers_data },
              { "accumulated_sync_partial", test_accumulated_sync_partial },
              { "two_batch_cycle", test_two_batch_cycle },
              { "two_batch_cycle_pipelined", test_two_batch_cycle_pipelined },
              { "partial_flush_inline_oldest_first",
                test_partial_flush_inline_oldest_first }, )

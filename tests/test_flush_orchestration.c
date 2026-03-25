#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "stream/config.h"
#include "gpu/stream.flush.h"

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
    // Batch events
    for (uint32_t i = 0; i < c->s->batch.epochs_per_batch; ++i)
      cu_event_destroy(c->s->batch.pool_events[i]);

    d2h_deliver_destroy(&c->s->d2h_deliver);
    compress_agg_destroy(&c->s->compress_agg, c->cl.levels.nlod);

    // Pools
    for (int i = 0; i < 2; ++i) {
      cu_mem_free(c->s->pools.buf[i]);
      cu_event_destroy(c->s->pools.ready[i]);
    }

    cu_stream_destroy(c->s->streams.compute);
    cu_stream_destroy(c->s->streams.compress);
    cu_stream_destroy(c->s->streams.d2h);

    free(c->s);
    c->s = NULL;
  }

  computed_stream_layouts_free(&c->cl);
}

// Set up all components for the flush orchestration test.
static int
orch_ctx_setup(struct orch_ctx* c,
               struct tile_stream_configuration* config,
               struct shard_sink* sink)
{
  CHECK(Fail,
        compute_stream_layouts(config,
                               codec_alignment(config->codec),
                               codec_max_output_size,
                               &c->cl) == 0);

  c->s = (struct tile_stream_gpu*)calloc(1, sizeof(*c->s));
  CHECK(Fail, c->s);

  c->s->config = *config;
  c->s->shard_sink = sink;
  c->s->levels = c->cl.levels;
  c->s->layout = c->cl.layouts[0];

  const uint32_t K = c->cl.epochs_per_batch;
  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t bpe = dtype_bpe(config->dtype);
  const size_t pool_bytes = (uint64_t)K * total_chunks * chunk_stride * bpe;

  // GPU streams
  CU(Fail, cuStreamCreate(&c->s->streams.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->s->streams.compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->s->streams.d2h, CU_STREAM_NON_BLOCKING));

  // Compress+aggregate stage
  CHECK(Fail,
        compress_agg_init(
          &c->s->compress_agg, &c->cl, config, c->s->streams.compute) == 0);

  // D2H+deliver stage
  CHECK(Fail,
        d2h_deliver_init(&c->s->d2h_deliver,
                         c->s->compress_agg.levels,
                         c->cl.levels.nlod,
                         c->s->streams.compute) == 0);

  // Double-buffered chunk pools
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&c->s->pools.buf[i], pool_bytes));
    CU(Fail,
       cuMemsetD8Async(
         c->s->pools.buf[i], 0, pool_bytes, c->s->streams.compute));
    CU(Fail, cuEventCreate(&c->s->pools.ready[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(c->s->pools.ready[i], c->s->streams.compute));
  }
  c->s->pools.current = 0;

  // Batch state + pool events
  c->s->batch.epochs_per_batch = K;
  c->s->batch.accumulated = 0;
  for (uint32_t i = 0; i < K; ++i) {
    CU(Fail, cuEventCreate(&c->s->batch.pool_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(c->s->batch.pool_events[i], c->s->streams.compute));
  }

  // Flush pipeline state
  memset(&c->s->flush, 0, sizeof(c->s->flush));

  // Non-multiscale: zeroed lod
  memset(&c->s->lod, 0, sizeof(c->s->lod));

  memset(&c->s->metrics, 0, sizeof(c->s->metrics));
  c->s->metrics.compress = mk_stream_metric("Compress");
  c->s->metrics.aggregate = mk_stream_metric("Aggregate");
  c->s->metrics.d2h = mk_stream_metric("D2H");
  c->s->metrics.sink = mk_stream_metric("Sink");
  c->s->metrics.lod_gather = mk_stream_metric("LOD Gather");

  memset(&c->s->metadata_update_clock, 0, sizeof(c->s->metadata_update_clock));

  CU(Fail, cuStreamSynchronize(c->s->streams.compute));
  return 0;

Fail:
  return 1;
}

// Fill one epoch in the current pool. Syncs compute stream first to ensure
// any pending pool zeroing is complete.
static int
orch_ctx_fill_epoch(struct orch_ctx* c,
                    uint32_t epoch_in_batch,
                    const struct tile_stream_configuration* config,
                    uint16_t (*fill_fn)(uint64_t))
{
  CU(Fail, cuStreamSynchronize(c->s->streams.compute));

  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t bpe = dtype_bpe(config->dtype);
  CUdeviceptr epoch_ptr =
    c->s->pools.buf[c->s->pools.current] +
    (uint64_t)epoch_in_batch * total_chunks * chunk_stride * bpe;
  return fill_pool_epoch(epoch_ptr, total_chunks, chunk_stride, bpe, fill_fn);

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
  make_test_config(&config, dims, CODEC_NONE, 2);

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
  struct writer_result r = flush_accumulate_epoch(c.s);
  CHECK(Fail, r.error == 0);

  // Verify: mid-batch, no flush triggered
  CHECK(Fail, c.s->batch.accumulated == 1);
  CHECK(Fail, c.s->pools.current == 0);
  CHECK(Fail, c.s->flush.pending == 0);

  // Epoch mask recorded
  CHECK(Fail, c.s->flush.slot[0].batch_active_masks[0] == 0x1);
  CHECK(Fail, c.s->flush.slot[0].active_levels_mask == 0x1);

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
  make_test_config(&config, dims, CODEC_NONE, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // Fill and accumulate 2 epochs
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);
  CHECK(Fail, c.s->batch.accumulated == 1);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);

  // After full batch: drain_kick_and_swap fired
  CHECK(Fail, c.s->batch.accumulated == 0);
  CHECK(Fail, c.s->pools.current == 1); // swapped to pool 1
  CHECK(Fail, c.s->flush.pending == 1); // batch 1 pending delivery
  CHECK(Fail, c.s->flush.current == 0); // batch 1 was on pool 0

  // Fresh pool slot is reset
  CHECK(Fail, c.s->flush.slot[1].active_levels_mask == 0);

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
  make_test_config(&config, dims, CODEC_NONE, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // Fill and accumulate 2 epochs (full batch → auto-kick)
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);
  CHECK(Fail, c.s->flush.pending == 1);

  // Drain the pending batch
  struct writer_result r = flush_drain_pending(c.s);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.s->flush.pending == 0);

  // Data delivered: shard opened and finalized (tps_0=2, 2 epochs → complete)
  CHECK(Fail, sink.open_count >= 1);
  CHECK(Fail, sink.finalize_count == 1);
  CHECK(Fail, sink.writers[0][0].size > 0);

  // Sink metric always recorded (uses platform_toc, not CUDA events)
  CHECK(Fail, c.s->metrics.sink.count == 1);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 4: flush_accumulated_sync with partial batch (1 epoch, K=2)
// ---------------------------------------------------------------------------
static int
test_accumulated_sync_partial(void)
{
  log_info("=== test_accumulated_sync_partial ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_NONE, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // Fill and accumulate 1 epoch (partial batch)
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);
  CHECK(Fail, c.s->batch.accumulated == 1);

  // Sync flush: processes the partial batch (per-epoch path)
  struct writer_result r = flush_accumulated_sync(c.s);
  CHECK(Fail, r.error == 0);

  // Batch drained
  CHECK(Fail, c.s->batch.accumulated == 0);

  // Data delivered to sink (1 epoch, shard not finalized since tps_0=2)
  CHECK(Fail, sink.open_count >= 1);
  CHECK(Fail, sink.writers[0][0].size > 0);

  // Sink metric recorded (platform_toc, not CUDA events — always fires)
  CHECK(Fail, c.s->metrics.sink.count == 1);

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
  make_test_config(&config, dims, CODEC_NONE, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);

  // --- Batch 1: epochs 0,1 on pool 0 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);

  // Batch 1 kicked, pool swapped to 1
  CHECK(Fail, c.s->pools.current == 1);
  CHECK(Fail, c.s->flush.pending == 1);
  CHECK(Fail, c.s->flush.current == 0);
  CHECK(Fail, c.s->batch.accumulated == 0);

  // --- Batch 2: epochs 2,3 on pool 1 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch2) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch3) == 0);
  CHECK(Fail, flush_accumulate_epoch(c.s).error == 0);

  // Batch 2 auto-drained batch 1 (pending was set), then kicked batch 2
  CHECK(Fail, c.s->pools.current == 0); // swapped back to pool 0
  CHECK(Fail, c.s->flush.pending == 1); // batch 2 now pending
  CHECK(Fail, c.s->flush.current == 1); // batch 2 was on pool 1
  CHECK(Fail, c.s->batch.accumulated == 0);

  // Batch 1 was already drained (by drain_kick_and_swap during batch 2)
  // tps_0=2, so batch 1 (2 epochs) finalized one shard
  CHECK(Fail, sink.finalize_count >= 1);

  // Drain batch 2
  struct writer_result r = flush_drain_pending(c.s);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.s->flush.pending == 0);

  // Second shard finalized
  CHECK(Fail, sink.finalize_count >= 2);

  // Sink metric: 2 batch drains
  CHECK(Fail, c.s->metrics.sink.count == 2);

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
              { "two_batch_cycle", test_two_batch_cycle }, )

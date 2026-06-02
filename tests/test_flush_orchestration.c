#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/stream.flush.h"
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

// "Slot has unfinished work": d2h queued OR data buffered (cap>1).
static int
slot_has_work(struct stream_engine* e, int oi)
{
  return e->flush.output.slot[oi].state != OUTPUT_LEDGER_EMPTY;
}

static void
orch_ctx_init(struct orch_ctx* c)
{
  memset(c, 0, sizeof(*c));
}

static void
orch_ctx_destroy(struct orch_ctx* c)
{
  if (c->s) {
    cu_event_destroy(c->s->engine.batch.pool_ready);

    for (int fc = 0; fc < 2; ++fc) {
      free(c->s->engine.flush.slot[fc].batch_active_masks);
      c->s->engine.flush.slot[fc].batch_active_masks = NULL;
    }

    d2h_deliver_destroy(&c->s->engine.d2h_deliver);
    compress_agg_destroy(&c->s->engine.compress_agg, c->cl.levels.nlod);

    // Pools
    for (int i = 0; i < 2; ++i) {
      cu_mem_free(c->s->engine.pools.buf[i]);
      cu_event_destroy(c->s->engine.pools.ready[i]);
    }

    cu_stream_destroy(c->s->engine.streams.compute);
    cu_stream_destroy(c->s->engine.streams.compress);
    cu_stream_destroy(c->s->engine.streams.d2h);

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
                               codec_alignment(config->codec.id),
                               codec_max_output_size,
                               platform_page_alignment(),
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

  // GPU streams
  CU(Fail,
     cuStreamCreate(&c->s->engine.streams.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail,
     cuStreamCreate(&c->s->engine.streams.compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->s->engine.streams.d2h, CU_STREAM_NON_BLOCKING));

  // Compress+aggregate stage
  CHECK(Fail,
        compress_agg_init(&c->s->engine.compress_agg,
                          &c->cl,
                          config,
                          c->s->engine.streams.compute) == 0);

  // D2H+deliver stage
  CHECK(Fail,
        d2h_deliver_init(&c->s->engine.d2h_deliver,
                         platform_page_alignment(),
                         c->s->engine.streams.compute) == 0);

  // Double-buffered chunk pools
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&c->s->engine.pools.buf[i], pool_bytes));
    CU(Fail,
       cuMemsetD8Async(c->s->engine.pools.buf[i],
                       0,
                       pool_bytes,
                       c->s->engine.streams.compute));
    CU(Fail, cuEventCreate(&c->s->engine.pools.ready[i], CU_EVENT_DEFAULT));
    CU(
      Fail,
      cuEventRecord(c->s->engine.pools.ready[i], c->s->engine.streams.compute));
  }
  c->s->engine.pools.current = 0;

  // Batch state + pool_ready event
  c->s->engine.batch.epochs_per_batch = K;
  c->s->engine.batch.accumulated = 0;
  CU(Fail, cuEventCreate(&c->s->engine.batch.pool_ready, CU_EVENT_DEFAULT));
  CU(
    Fail,
    cuEventRecord(c->s->engine.batch.pool_ready, c->s->engine.streams.compute));

  // Flush pipeline state
  memset(&c->s->engine.flush, 0, sizeof(c->s->engine.flush));
  for (int fc = 0; fc < 2; ++fc) {
    c->s->engine.flush.slot[fc].batch_active_masks =
      (uint32_t*)calloc(K, sizeof(uint32_t));
    CHECK(Fail, c->s->engine.flush.slot[fc].batch_active_masks);
  }
  {
    const struct aggregate_slot* slot = &c->s->engine.compress_agg.output[0];
    const struct output_slot_capacity capacity = {
      .data_bytes = slot->slot_capacity_bytes,
      .desc_entries = slot->slot_desc_capacity,
      .batch_records = slot->batches_per_slot_cap,
    };
    CHECK(Fail,
          output_slot_ledger_init(&c->s->engine.flush.output, capacity) ==
            OUTPUT_LEDGER_OK);
  }

  // Non-multiscale: zeroed lod
  memset(&c->s->engine.lod, 0, sizeof(c->s->engine.lod));

  memset(&c->s->engine.metrics, 0, sizeof(c->s->engine.metrics));
  c->s->engine.metrics.compress = mk_stream_metric("Compress");
  c->s->engine.metrics.aggregate = mk_stream_metric("Aggregate");
  c->s->engine.metrics.d2h = mk_stream_metric("D2H");
  c->s->engine.metrics.sink = mk_stream_metric("Sink");
  c->s->engine.metrics.lod_gather = mk_stream_metric("LOD Gather");

  memset(&c->s->engine.metadata_update_clock,
         0,
         sizeof(c->s->engine.metadata_update_clock));

  CU(Fail, cuStreamSynchronize(c->s->engine.streams.compute));
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
  CU(Fail, cuStreamSynchronize(c->s->engine.streams.compute));

  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config->dtype);
  CUdeviceptr epoch_ptr =
    c->s->engine.pools.buf[c->s->engine.pools.current] +
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
  struct writer_result r = flush_accumulate_epoch(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);

  // Verify: mid-batch, no flush triggered
  CHECK(Fail, c.s->engine.batch.accumulated == 1);
  CHECK(Fail, c.s->engine.pools.current == 0);
  CHECK(Fail, c.s->engine.flush.output.slot[0].state == OUTPUT_LEDGER_EMPTY);
  CHECK(Fail, c.s->engine.flush.output.slot[1].state == OUTPUT_LEDGER_EMPTY);

  // Epoch mask recorded
  CHECK(Fail, c.s->engine.flush.slot[0].batch_active_masks[0] == 0x1);
  CHECK(Fail, c.s->engine.flush.slot[0].active_levels_mask == 0x1);

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
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.batch.accumulated == 1);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  // After full batch: drain_kick_and_swap fired
  CHECK(Fail, c.s->engine.batch.accumulated == 0);
  CHECK(Fail, c.s->engine.pools.current == 1); // swapped to pool 1
  CHECK(Fail, slot_has_work(&c.s->engine, 0)); // batch 1 in-flight at slot 0
  CHECK(Fail, !slot_has_work(&c.s->engine, 1));

  // Fresh pool slot is reset
  CHECK(Fail, c.s->engine.flush.slot[1].active_levels_mask == 0);

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
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, slot_has_work(&c.s->engine, 0));

  // Drain the pending batch
  struct writer_result r = flush_drain_pending(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, !slot_has_work(&c.s->engine, 0));
  CHECK(Fail, !slot_has_work(&c.s->engine, 1));

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
// Test 4: flush_accumulated_sync with partial batch (1 epoch, K=2)
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
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, c.s->engine.batch.accumulated == 1);

  // Sync flush: processes the partial batch (per-epoch path)
  struct writer_result r = flush_accumulated_sync(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);

  // Batch drained
  CHECK(Fail, c.s->engine.batch.accumulated == 0);

  // Data delivered into the pipeline. With tail-carryover delivery, a partial
  // batch's data is staged in the in-memory tail (writes to disk only happen
  // at page-aligned boundaries / on shard finalize), so verify shard state
  // rather than on-disk size.
  CHECK(Fail, sink.open_count >= 1);
  {
    struct shard_state* ss = &c.s->engine.compress_agg.shard[0];
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
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  // Batch 1 kicked, pool swapped to 1, batch 1 in-flight at slot 0
  CHECK(Fail, c.s->engine.pools.current == 1);
  CHECK(Fail, slot_has_work(&c.s->engine, 0));
  CHECK(Fail, !slot_has_work(&c.s->engine, 1));
  CHECK(Fail, c.s->engine.batch.accumulated == 0);

  // --- Batch 2: epochs 2,3 on pool 1 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch2) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch3) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  CHECK(Fail, c.s->engine.pools.current == 0);
  CHECK(Fail, slot_has_work(&c.s->engine, 0) || slot_has_work(&c.s->engine, 1));
  CHECK(Fail, c.s->engine.batch.accumulated == 0);
  CHECK(Fail, sink.finalize_count == 0);

  struct writer_result r = flush_drain_pending(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, !slot_has_work(&c.s->engine, 0));
  CHECK(Fail, !slot_has_work(&c.s->engine, 1));

  CHECK(Fail, sink.finalize_count >= 2);

  // At cap=1 the two batches drain separately; at cap>=2 they stack into one
  // slot and drain together. Pick the expected count from the live cap.
  const uint32_t cap = c.s->engine.compress_agg.output[0].batches_per_slot_cap;
  const int expected_drains = (cap >= 2) ? 1 : 2;
  CHECK(Fail, c.s->engine.metrics.sink.count == expected_drains);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 6: compressed auto-flush defers callback finalization
// ---------------------------------------------------------------------------
static int
test_compressed_auto_flush_defers_aggregate_callback(void)
{
  log_info("=== test_compressed_auto_flush_defers_aggregate_callback ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_ZSTD }, 1);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);
  CHECK(Fail, c.s->engine.compress_agg.output[0].batches_per_slot_cap > 1);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  CHECK(Fail, c.s->engine.batch.accumulated == 0);
  CHECK(Fail, c.s->engine.pools.current == 1);
  CHECK(Fail, c.s->engine.flush.aggregate_pending.active);
  CHECK(Fail, c.s->engine.flush.output.slot[0].state == OUTPUT_LEDGER_OPEN);
  CHECK(Fail, c.s->engine.flush.pending_handoff[0].output == NULL);

  struct writer_result r = flush_drain_pending(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, !c.s->engine.flush.aggregate_pending.active);
  CHECK(Fail, !slot_has_work(&c.s->engine, 0));
  CHECK(Fail, !slot_has_work(&c.s->engine, 1));

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 7: compressed cap-stacking must retire the alternate slot before reuse
// ---------------------------------------------------------------------------
static int
test_compressed_alt_slot_retired_before_reuse(void)
{
  log_info("=== test_compressed_alt_slot_retired_before_reuse ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_ZSTD }, 1);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config, &sink.base) == 0);
  CHECK(Fail, c.s->engine.compress_agg.output[0].batches_per_slot_cap > 1);

  uint16_t (*fills[4])(uint64_t) = {
    fill_epoch0,
    fill_epoch1,
    fill_epoch2,
    fill_epoch3,
  };

  int next_fill = 0;
  const int max_kicks = 8;
  for (; next_fill < max_kicks; ++next_fill) {
    CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fills[next_fill % 4]) == 0);
    CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);
    if (c.s->engine.flush.output.slot[0].state == OUTPUT_LEDGER_D2H_IN_FLIGHT)
      break;
  }

  // Slot 0 is now immutable pending delivery; the next kick must retire it
  // before any possible swap can target it.
  CHECK(Fail,
        c.s->engine.flush.output.slot[0].state == OUTPUT_LEDGER_D2H_IN_FLIGHT);
  CU(Fail, cuEventSynchronize(c.s->engine.d2h_deliver.ready[0]));

  CHECK(Fail, next_fill + 1 < max_kicks);
  CHECK(Fail,
        orch_ctx_fill_epoch(&c, 0, &config, fills[(next_fill + 1) % 4]) == 0);
  CHECK(Fail, flush_accumulate_epoch(&c.s->engine, &c.s->ctx).error == 0);

  CHECK(Fail,
        c.s->engine.flush.output.slot[0].state != OUTPUT_LEDGER_D2H_IN_FLIGHT);
  CHECK(Fail,
        c.s->engine.flush.output.slot[0].state != OUTPUT_LEDGER_DELIVERING);

  struct writer_result r = flush_drain_pending(&c.s->engine, &c.s->ctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, !slot_has_work(&c.s->engine, 0));
  CHECK(Fail, !slot_has_work(&c.s->engine, 1));

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
              { "compressed_auto_flush_defers_aggregate_callback",
                test_compressed_auto_flush_defers_aggregate_callback },
              { "compressed_alt_slot_retired_before_reuse",
                test_compressed_alt_slot_retired_before_reuse }, )

#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "gpu/schedule.h"
#include "gpu/stream.engine.h"
#include "platform/platform.h"
#include "stream/config.h"

#include "index.ops.util.h"
#include "test_gpu_helpers.h"
#include "test_metric_check.h"
#include "test_runner.h"
#include "test_shard_sink.h"

#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// ---------------------------------------------------------------------------
// Common setup: compress_agg + d2h_deliver stages, chunk pool, fill, kick both.
// Returns 0 on success, populates out_handoff.
// ---------------------------------------------------------------------------

struct test_ctx
{
  struct computed_stream_layouts cl;
  struct compress_agg_stage ca;
  struct d2h_deliver_stage d2h;
  CUstream compute;
  CUstream d2h_stream;
  CUstream drain_stream;
  CUdeviceptr d_pool;
  struct gpu_ordering ord; // edge registry the harness pools draw from
  struct gpu_pool pool;    // chunk pool; slots rebound per kick's pool_buf
  uint32_t* batch_active_masks;
  struct stream_metrics metrics;
  struct lod_state lod;
  struct lod_shared_state lod_shared;
  struct platform_clock metadata_clock;
  int ord_inited;
  int ca_inited;
  int d2h_inited;
};

static void
test_ctx_init(struct test_ctx* c)
{
  memset(c, 0, sizeof(*c));
}

static void
test_ctx_destroy(struct test_ctx* c)
{
  if (c->d2h_inited)
    d2h_deliver_destroy(&c->d2h);
  if (c->ca_inited)
    compress_agg_destroy(&c->ca);
  if (c->ord_inited)
    gpu_ordering_destroy(&c->ord);
  computed_stream_layouts_free(&c->cl);
  cu_mem_free(c->d_pool);
  free(c->batch_active_masks);
  cu_stream_destroy(c->compute);
  cu_stream_destroy(c->d2h_stream);
  cu_stream_destroy(c->drain_stream);
}

// Setup: compute layouts, init compress_agg + d2h_deliver, allocate pool.
// n_pool_epochs: how many epochs of chunk pool to allocate.
static int
test_ctx_setup(struct test_ctx* c,
               struct tile_stream_configuration* config,
               int n_pool_epochs)
{
  CHECK(Fail,
        compute_stream_layouts(config,
                               codec_alignment(config->codec.id),
                               codec_max_output_size,
                               platform_page_alignment(),
                               &c->cl) == 0);

  CU(Fail, cuStreamCreate(&c->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->d2h_stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->drain_stream, CU_STREAM_NON_BLOCKING));

  CHECK(Fail, gpu_ordering_init(&c->ord, c->compute) == 0);
  c->ord_inited = 1;
  gpu_ordering_register_stream(&c->ord, GPU_STREAM_COMPUTE, c->compute);
  gpu_ordering_register_stream(&c->ord, GPU_STREAM_D2H, c->d2h_stream);
  gpu_ordering_register_stream(&c->ord, GPU_STREAM_DRAIN, c->drain_stream);

  CHECK(Fail,
        compress_agg_init(&c->ca, &c->cl, config, &c->ord, c->compute) == 0);
  c->ca_inited = 1;

  CHECK(Fail,
        d2h_deliver_init(&c->d2h,
                         platform_page_alignment(),
                         &c->ord,
                         c->drain_stream,
                         c->compute) == 0);
  c->d2h_inited = 1;

  size_t pool_bytes = (uint64_t)n_pool_epochs * c->cl.levels.total_chunks *
                      c->cl.layouts[0].chunk_stride * dtype_bpe(config->dtype);
  CU(Fail, cuMemAlloc(&c->d_pool, pool_bytes));
  gpu_pool_init(
    &c->pool, &c->ord, GPU_EDGE_POOL_FILLED, GPU_EDGE_POOL_CONSUMED);

  c->batch_active_masks =
    (uint32_t*)calloc(c->cl.epochs_per_batch, sizeof(uint32_t));
  CHECK(Fail, c->batch_active_masks);

  c->metrics = stream_engine_init_metrics(0);

  memset(&c->lod, 0, sizeof(c->lod));
  memset(&c->lod_shared, 0, sizeof(c->lod_shared));
  memset(&c->metadata_clock, 0, sizeof(c->metadata_clock));

  return 0;

Fail:
  return 1;
}

// Run the compress_agg kick + d2h_deliver kick + drain for a batch.
static int
test_ctx_kick_and_drain(struct test_ctx* c,
                        const struct tile_stream_configuration* config,
                        struct shard_sink* sink,
                        int fc,
                        uint32_t n_epochs,
                        CUdeviceptr pool_buf,
                        struct flush_handoff* handoff)
{
  for (uint32_t i = 0; i < n_epochs; ++i)
    c->batch_active_masks[i] = 0x1;

  gpu_pool_bind(&c->pool, fc, (void*)(uintptr_t)pool_buf);
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = 0x1,
    .batch_active_masks = c->batch_active_masks,
    .epochs_per_batch = c->cl.epochs_per_batch,
  };

  memset(handoff, 0, sizeof(*handoff));

  CHECK(Fail,
        schedule_compress_agg_kick(
          &c->ca, &in, &c->cl.levels, &c->pool, 0, c->compute, handoff) == 0);

  CHECK(Fail, schedule_d2h_kick(&c->d2h, handoff, sink, c->d2h_stream) == 0);

  struct writer_result r = schedule_d2h_drain(&c->d2h,
                                              handoff,
                                              &c->cl.levels,
                                              &c->cl.dims,
                                              &c->cl.layouts[0],
                                              config,
                                              sink,
                                              &c->metrics,
                                              &c->metadata_clock);
  CHECK(Fail, r.error == 0);

  return 0;

Fail:
  return 1;
}

// ---------------------------------------------------------------------------
// Test 1: CODEC_NONE, K=1, single epoch — data arrives in sink
// ---------------------------------------------------------------------------
static int
test_d2h_single_epoch_none(void)
{
  log_info("=== test_d2h_single_epoch_none ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 1);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct test_ctx c;
  test_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, test_ctx_setup(&c, &config, 1) == 0);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config.dtype);
  const size_t chunk_bytes = chunk_stride * bytes_per_element;

  log_info("  total_chunks=%lu chunk_stride=%lu chunk_bytes=%zu",
           (unsigned long)total_chunks,
           (unsigned long)chunk_stride,
           chunk_bytes);

  // Fill pool with epoch 0 data
  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch0) ==
      0);

  // Record pool-filled after the fill
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  // Kick compress_agg + D2H + drain
  struct flush_handoff handoff;
  CHECK(Fail,
        test_ctx_kick_and_drain(
          &c, &config, &sink.base, 0, 1, c.d_pool, &handoff) == 0);

  // Verify sink state
  CHECK(Fail, sink.open_count == 1);     // shard_inner_count=1
  CHECK(Fail, sink.finalize_count == 0); // tps_0=2, need 2 epochs

  // Sink is host-clocked, so it fires without CUDA events.
  CHECK(Fail, metric_arrived_timed(&c.metrics.sink, 1));
  CHECK(Fail, metric_arrived(&c.metrics.lod_gather, 0));

  // Pass-through runs no codec, so a missing compress row is the truth here.
  CHECK(Fail, metric_arrived(&c.metrics.compress, 0));
  CHECK(Fail, metric_arrived_timed(&c.metrics.aggregate, 1));
  CHECK(Fail, metric_arrived_timed(&c.metrics.d2h, 1));
  CHECK(Fail, metric_arrived(&c.metrics.tail_gate, 1));

  // Tile data correctness verified by test_compress_agg

  ok = 1;

Fail:
  test_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: CODEC_NONE, K=2, full batch → shard finalized
// ---------------------------------------------------------------------------
static int
test_d2h_batch_none(void)
{
  log_info("=== test_d2h_batch_none ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct test_ctx c;
  test_ctx_init(&c);
  int ok = 0;
  uint32_t* inv_perm = NULL;

  CHECK(Fail, test_ctx_setup(&c, &config, 2) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config.dtype);
  const size_t chunk_bytes = chunk_stride * bytes_per_element;

  // Fill pool: epoch 0 and epoch 1
  size_t epoch_pool_bytes = total_chunks * chunk_stride * bytes_per_element;
  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch0) ==
      0);
  CHECK(Fail,
        fill_pool_epoch(c.d_pool + epoch_pool_bytes,
                        total_chunks,
                        chunk_stride,
                        bytes_per_element,
                        fill_epoch1) == 0);

  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  // Kick with 2 epochs
  struct flush_handoff handoff;
  CHECK(Fail,
        test_ctx_kick_and_drain(
          &c, &config, &sink.base, 0, 2, c.d_pool, &handoff) == 0);

  // tps_0=2, 2 epochs → shard complete
  CHECK(Fail, sink.finalize_count == 1);

  // Parse finalized shard: index block at end
  // chunks_per_shard_total = 8, index = 8 * 16 bytes + 4 byte CRC
  {
    struct shard_state* ss = &c.ca.ar.shard[0];
    uint64_t tps_total = ss->chunks_per_shard_total;
    size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    CHECK(Fail, sink.writers[0][0].size >= index_total_bytes);
    size_t index_start = sink.writers[0][0].size - index_total_bytes;

    const uint64_t* idx =
      (const uint64_t*)(sink.writers[0][0].buf + index_start);

    // Shard output layout: [num_shards, batch_count, cps_inner] row-major.
    // Slot → (si, epoch, ci) via unravel, then perm_pos = si * cps_inner + ci.
    const struct aggregate_layout* al = &c.ca.ar.per_lod_agg_layouts[0];
    uint32_t batch_count = c.ca.ar.per_lod_agg_layouts[0].active_count_max;
    uint32_t cps_inner = (uint32_t)al->cps_inner;
    uint32_t num_shards = (uint32_t)(al->covering_count / cps_inner);
    uint64_t chunks_lv = c.cl.levels.level[0].chunk_count;
    // unravel uses column-major (d=0 fastest), so reverse for row-major order.
    const uint64_t slot_shape[3] = { cps_inner, batch_count, num_shards };

    // Build inverse perm: inv_perm[perm_pos] = original chunk j
    inv_perm = (uint32_t*)malloc(chunks_lv * sizeof(uint32_t));
    CHECK(Fail, inv_perm);
    for (uint64_t j = 0; j < chunks_lv; ++j) {
      uint32_t pp =
        cpu_perm(j, al->lifted_rank, al->lifted_shape, al->lifted_strides);
      inv_perm[pp] = (uint32_t)j;
    }

    int errors = 0;
    for (uint64_t slot = 0; slot < tps_total; ++slot) {
      uint64_t coords[3];
      unravel(3, slot_shape, slot, coords);
      uint32_t perm_pos = (uint32_t)(coords[2] * cps_inner + coords[0]);
      uint32_t epoch = (uint32_t)coords[1];
      uint16_t (*fill_fn)(uint64_t) = (epoch == 0) ? fill_epoch0 : fill_epoch1;
      uint32_t orig_tile = inv_perm[perm_pos];

      uint64_t tile_off = idx[2 * slot];
      uint64_t tile_sz = idx[2 * slot + 1];

      if (tile_sz != chunk_bytes) {
        if (errors < 5)
          log_error("  slot %lu (epoch %u chunk %u): size=%lu expected=%zu",
                    (unsigned long)slot,
                    epoch,
                    orig_tile,
                    (unsigned long)tile_sz,
                    chunk_bytes);
        errors++;
        continue;
      }

      uint16_t expected_val = fill_fn(orig_tile);
      const uint16_t* got =
        (const uint16_t*)(sink.writers[0][0].buf + tile_off);
      for (uint64_t e = 0; e < chunk_stride; ++e) {
        if (got[e] != expected_val) {
          if (errors < 5)
            log_error(
              "  slot %lu (epoch %u chunk %u) elem %lu: expected %u got %u",
              (unsigned long)slot,
              epoch,
              orig_tile,
              (unsigned long)e,
              expected_val,
              got[e]);
          errors++;
        }
      }
    }
    free(inv_perm);
    inv_perm = NULL;
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(inv_perm);
  test_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 3: CODEC_ZSTD, K=2 batch (full shard) — compressed data arrives
// Verifies ZSTD round-trip end-to-end: with tail-carryover delivery, partial
// batches stay staged in the tail buffer and only land on disk at finalize.
// ---------------------------------------------------------------------------
static int
test_d2h_zstd_single_epoch(void)
{
  log_info("=== test_d2h_zstd_single_epoch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_ZSTD }, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct test_ctx c;
  test_ctx_init(&c);
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail, test_ctx_setup(&c, &config, 2) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config.dtype);
  const size_t chunk_bytes = chunk_stride * bytes_per_element;
  size_t epoch_pool_bytes = total_chunks * chunk_stride * bytes_per_element;

  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch0) ==
      0);
  CHECK(Fail,
        fill_pool_epoch(c.d_pool + epoch_pool_bytes,
                        total_chunks,
                        chunk_stride,
                        bytes_per_element,
                        fill_epoch0) == 0);

  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  struct flush_handoff handoff;
  CHECK(Fail,
        test_ctx_kick_and_drain(
          &c, &config, &sink.base, 0, 2, c.d_pool, &handoff) == 0);

  CHECK(Fail, sink.finalize_count == 1);
  CHECK(Fail, sink.writers[0][0].size > 0);

  CHECK(Fail, metric_arrived_timed(&c.metrics.compress, 1));
  CHECK(Fail, metric_arrived_timed(&c.metrics.aggregate, 1));
  CHECK(Fail, metric_arrived_timed(&c.metrics.d2h, 1));
  CHECK(Fail, metric_arrived(&c.metrics.tail_gate, 1));

  // Decompress and verify chunk data via the on-disk index.
  {
    struct shard_state* ss = &c.ca.ar.shard[0];
    uint64_t tps_total = ss->chunks_per_shard_total;
    size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    CHECK(Fail, sink.writers[0][0].size >= index_total_bytes);
    size_t index_start = sink.writers[0][0].size - index_total_bytes;
    const uint64_t* idx =
      (const uint64_t*)(sink.writers[0][0].buf + index_start);

    const struct aggregate_layout* al = &c.ca.ar.per_lod_agg_layouts[0];
    uint64_t cps_inner = ss->chunks_per_shard_inner;

    decomp_buf = (uint8_t*)malloc(chunk_bytes);
    CHECK(Fail, decomp_buf);

    int errors = 0;
    for (int epoch = 0; epoch < 2; ++epoch) {
      for (uint64_t t = 0; t < total_chunks; ++t) {
        uint32_t pi =
          cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
        uint64_t slot_idx = (uint64_t)epoch * cps_inner + pi;
        uint64_t tile_off = idx[2 * slot_idx];
        uint64_t tile_sz = idx[2 * slot_idx + 1];

        CHECK(Fail, tile_sz > 0);
        CHECK(Fail, tile_off + tile_sz <= sink.writers[0][0].size);

        size_t result = ZSTD_decompress(
          decomp_buf, chunk_bytes, sink.writers[0][0].buf + tile_off, tile_sz);
        if (ZSTD_isError(result)) {
          log_error("  chunk %lu: ZSTD_decompress failed: %s",
                    (unsigned long)t,
                    ZSTD_getErrorName(result));
          errors++;
          continue;
        }
        CHECK(Fail, result == chunk_bytes);

        uint16_t expected_val = fill_epoch0(t);
        const uint16_t* got = (const uint16_t*)decomp_buf;
        for (uint64_t e = 0; e < chunk_stride; ++e) {
          if (got[e] != expected_val) {
            if (errors < 5)
              log_error("  chunk %lu elem %lu: expected %u got %u",
                        (unsigned long)t,
                        (unsigned long)e,
                        expected_val,
                        got[e]);
            errors++;
          }
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(decomp_buf);
  test_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 4: Two consecutive kick+drain cycles (double buffer, fc=0 then fc=1)
// ---------------------------------------------------------------------------
static int
test_d2h_double_buffer(void)
{
  log_info("=== test_d2h_double_buffer ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_NONE }, 1);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct test_ctx c;
  test_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, test_ctx_setup(&c, &config, 2) == 0);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config.dtype);
  const size_t chunk_bytes = chunk_stride * bytes_per_element;
  size_t epoch_pool_bytes = total_chunks * chunk_stride * bytes_per_element;

  // Iteration 1: fc=0, fill with epoch0
  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch0) ==
      0);
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  {
    struct flush_handoff handoff;
    CHECK(Fail,
          test_ctx_kick_and_drain(
            &c, &config, &sink.base, 0, 1, c.d_pool, &handoff) == 0);
  }

  CHECK(Fail, sink.finalize_count == 0); // 1 of 2 epochs

  // Iteration 2: fc=1, fill with epoch1
  CHECK(Fail,
        fill_pool_epoch(c.d_pool + epoch_pool_bytes,
                        total_chunks,
                        chunk_stride,
                        bytes_per_element,
                        fill_epoch1) == 0);
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  {
    struct flush_handoff handoff;
    CHECK(
      Fail,
      test_ctx_kick_and_drain(
        &c, &config, &sink.base, 1, 1, c.d_pool + epoch_pool_bytes, &handoff) ==
        0);
  }

  CHECK(Fail, sink.finalize_count == 1); // shard complete

  // Parse finalized shard and verify both epochs' data
  {
    struct shard_state* ss = &c.ca.ar.shard[0];
    uint64_t tps_total = ss->chunks_per_shard_total;
    size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    CHECK(Fail, sink.writers[0][0].size >= index_total_bytes);
    size_t index_start = sink.writers[0][0].size - index_total_bytes;
    const uint64_t* idx =
      (const uint64_t*)(sink.writers[0][0].buf + index_start);

    const struct aggregate_layout* al = &c.ca.ar.per_lod_agg_layouts[0];
    uint64_t cps_inner = ss->chunks_per_shard_inner;

    int errors = 0;
    for (int epoch = 0; epoch < 2; ++epoch) {
      uint16_t (*fill_fn)(uint64_t) = (epoch == 0) ? fill_epoch0 : fill_epoch1;
      for (uint64_t j = 0; j < total_chunks; ++j) {
        uint32_t pi =
          cpu_perm(j, al->lifted_rank, al->lifted_shape, al->lifted_strides);
        uint64_t slot_idx = (uint64_t)epoch * cps_inner + pi;
        uint64_t tile_off = idx[2 * slot_idx];
        uint64_t tile_sz = idx[2 * slot_idx + 1];

        if (tile_sz != chunk_bytes) {
          if (errors < 5)
            log_error("  epoch %d chunk %lu: size=%lu expected=%zu",
                      epoch,
                      (unsigned long)j,
                      (unsigned long)tile_sz,
                      chunk_bytes);
          errors++;
          continue;
        }

        uint16_t expected_val = fill_fn(j);
        const uint16_t* got =
          (const uint16_t*)(sink.writers[0][0].buf + tile_off);
        for (uint64_t e = 0; e < chunk_stride; ++e) {
          if (got[e] != expected_val) {
            if (errors < 5)
              log_error("  epoch %d chunk %lu elem %lu: expected %u got %u",
                        epoch,
                        (unsigned long)j,
                        (unsigned long)e,
                        expected_val,
                        got[e]);
            errors++;
          }
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  test_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Three-cycle compressed run: cycle 3 reuses fc=0, so the slot-drained
// edge recorded in cycle 1 must survive into cycle 3's compress wait.
// Two cycles would never reuse a slot.
static int
test_d2h_zstd_double_buffer(void)
{
  log_info("=== test_d2h_zstd_double_buffer ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, (struct codec_config){ .id = CODEC_ZSTD }, 1);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct test_ctx c;
  test_ctx_init(&c);
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail, test_ctx_setup(&c, &config, 2) == 0);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config.dtype);
  const size_t chunk_bytes = chunk_stride * bytes_per_element;
  size_t epoch_pool_bytes = total_chunks * chunk_stride * bytes_per_element;

  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch0) ==
      0);
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  {
    struct flush_handoff handoff;
    CHECK(Fail,
          test_ctx_kick_and_drain(
            &c, &config, &sink.base, 0, 1, c.d_pool, &handoff) == 0);
  }

  CHECK(Fail, sink.finalize_count == 0);

  CHECK(Fail,
        fill_pool_epoch(c.d_pool + epoch_pool_bytes,
                        total_chunks,
                        chunk_stride,
                        bytes_per_element,
                        fill_epoch1) == 0);
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  {
    struct flush_handoff handoff;
    CHECK(
      Fail,
      test_ctx_kick_and_drain(
        &c, &config, &sink.base, 1, 1, c.d_pool + epoch_pool_bytes, &handoff) ==
        0);
  }

  CHECK(Fail, sink.finalize_count == 1);

  // Cycle 3: reuse fc=0. compress_agg's GPU_EDGE_SLOT_DRAINED wait depends
  // on the drain having recorded the edge in cycle 1.
  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch2) ==
      0);
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  {
    struct flush_handoff handoff;
    CHECK(Fail,
          test_ctx_kick_and_drain(
            &c, &config, &sink.base, 0, 1, c.d_pool, &handoff) == 0);
  }

  CHECK(Fail, sink.finalize_count == 1);

  // Cycle 4 finalizes shard 1 so cycle 3's data lands on disk and can be
  // verified — without this, cycle 3 corruption would pass silently.
  CHECK(Fail,
        fill_pool_epoch(c.d_pool + epoch_pool_bytes,
                        total_chunks,
                        chunk_stride,
                        bytes_per_element,
                        fill_epoch3) == 0);
  CHECK(Fail, gpu_pool_release_produce(&c.pool, 0, c.compute) == 0);

  {
    struct flush_handoff handoff;
    CHECK(
      Fail,
      test_ctx_kick_and_drain(
        &c, &config, &sink.base, 1, 1, c.d_pool + epoch_pool_bytes, &handoff) ==
        0);
  }

  CHECK(Fail, sink.finalize_count == 2);

  // These events are seeded at setup, so a cycle that skipped its record
  // still reports a plausible interval. Counting alone cannot see that.
  CHECK(Fail, metric_arrived_timed(&c.metrics.compress, 4));
  CHECK(Fail, metric_arrived_timed(&c.metrics.aggregate, 4));
  CHECK(Fail, metric_arrived_timed(&c.metrics.d2h, 4));
  CHECK(Fail, metric_arrived(&c.metrics.tail_gate, 4));

  {
    struct shard_state* ss = &c.ca.ar.shard[0];
    uint64_t tps_total = ss->chunks_per_shard_total;
    size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    const struct aggregate_layout* al = &c.ca.ar.per_lod_agg_layouts[0];
    uint64_t cps_inner = ss->chunks_per_shard_inner;

    decomp_buf = (uint8_t*)malloc(chunk_bytes);
    CHECK(Fail, decomp_buf);

    uint16_t (*fills[4])(
      uint64_t) = { fill_epoch0, fill_epoch1, fill_epoch2, fill_epoch3 };
    int errors = 0;
    for (int shard = 0; shard < 2; ++shard) {
      CHECK(Fail, sink.writers[0][shard].size >= index_total_bytes);
      size_t index_start = sink.writers[0][shard].size - index_total_bytes;
      const uint64_t* idx =
        (const uint64_t*)(sink.writers[0][shard].buf + index_start);

      for (int local_epoch = 0; local_epoch < 2; ++local_epoch) {
        const int global_epoch = shard * 2 + local_epoch;
        uint16_t (*fill_fn)(uint64_t) = fills[global_epoch];
        for (uint64_t t = 0; t < total_chunks; ++t) {
          uint32_t pi =
            cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
          uint64_t slot_idx = (uint64_t)local_epoch * cps_inner + pi;
          uint64_t tile_off = idx[2 * slot_idx];
          uint64_t tile_sz = idx[2 * slot_idx + 1];

          CHECK(Fail, tile_sz > 0);
          CHECK(Fail, tile_off + tile_sz <= sink.writers[0][shard].size);

          size_t result = ZSTD_decompress(decomp_buf,
                                          chunk_bytes,
                                          sink.writers[0][shard].buf + tile_off,
                                          tile_sz);
          if (ZSTD_isError(result)) {
            log_error("  shard %d epoch %d chunk %lu: ZSTD_decompress: %s",
                      shard,
                      global_epoch,
                      (unsigned long)t,
                      ZSTD_getErrorName(result));
            errors++;
            continue;
          }
          CHECK(Fail, result == chunk_bytes);

          uint16_t expected_val = fill_fn(t);
          const uint16_t* got = (const uint16_t*)decomp_buf;
          for (uint64_t e = 0; e < chunk_stride; ++e) {
            if (got[e] != expected_val) {
              if (errors < 5)
                log_error("  shard %d epoch %d chunk %lu elem %lu: "
                          "expected %u got %u",
                          shard,
                          global_epoch,
                          (unsigned long)t,
                          (unsigned long)e,
                          expected_val,
                          got[e]);
              errors++;
            }
          }
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(decomp_buf);
  test_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "d2h_single_epoch_none", test_d2h_single_epoch_none },
              { "d2h_batch_none", test_d2h_batch_none },
              { "d2h_zstd_single_epoch", test_d2h_zstd_single_epoch },
              { "d2h_double_buffer", test_d2h_double_buffer },
              { "d2h_zstd_double_buffer", test_d2h_zstd_double_buffer }, )

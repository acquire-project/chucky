#include "gpu/flush.compress_agg.h"
#include "gpu/flush.d2h_deliver.h"
#include "stream/config.h"

#include "index.ops.util.h"
#include "test_gpu_helpers.h"
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
  CUdeviceptr d_pool;
  CUevent epoch_events[2];
  struct batch_state batch;
  struct stream_metrics metrics;
  struct lod_state lod;
  struct platform_clock metadata_clock;
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
    compress_agg_destroy(&c->ca, c->cl.levels.nlod);
  computed_stream_layouts_free(&c->cl);
  cu_mem_free(c->d_pool);
  for (int i = 0; i < 2; ++i)
    cu_event_destroy(c->epoch_events[i]);
  cu_stream_destroy(c->compute);
  cu_stream_destroy(c->d2h_stream);
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
                               codec_alignment(config->codec),
                               codec_max_output_size,
                               &c->cl) == 0);

  CU(Fail, cuStreamCreate(&c->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->d2h_stream, CU_STREAM_NON_BLOCKING));

  CHECK(Fail, compress_agg_init(&c->ca, &c->cl, config, c->compute) == 0);
  c->ca_inited = 1;

  CHECK(Fail,
        d2h_deliver_init(
          &c->d2h, c->ca.levels, c->cl.levels.nlod, c->compute) == 0);
  c->d2h_inited = 1;

  size_t pool_bytes = (uint64_t)n_pool_epochs * c->cl.levels.total_chunks *
                      c->cl.layouts[0].chunk_stride * dtype_bpe(config->dtype);
  CU(Fail, cuMemAlloc(&c->d_pool, pool_bytes));

  c->batch = (struct batch_state){
    .epochs_per_batch = c->cl.epochs_per_batch,
    .accumulated = 0,
  };

  memset(&c->metrics, 0, sizeof(c->metrics));
  c->metrics.compress = mk_stream_metric("Compress");
  c->metrics.aggregate = mk_stream_metric("Aggregate");
  c->metrics.d2h = mk_stream_metric("D2H");
  c->metrics.sink = mk_stream_metric("Sink");
  c->metrics.lod_gather = mk_stream_metric("LOD Gather");

  memset(&c->lod, 0, sizeof(c->lod));
  memset(&c->metadata_clock, 0, sizeof(c->metadata_clock));

  return 0;

Fail:
  return 1;
}

// Run compress_agg_kick + d2h_deliver_kick + d2h_deliver_drain for a batch.
static int
test_ctx_kick_and_drain(struct test_ctx* c,
                        const struct tile_stream_configuration* config,
                        struct shard_sink* sink,
                        int fc,
                        uint32_t n_epochs,
                        CUdeviceptr pool_buf,
                        const CUevent* epoch_events,
                        struct flush_handoff* handoff)
{
  struct compress_agg_input in = {
    .fc = fc,
    .n_epochs = n_epochs,
    .active_levels_mask = 0x1,
    .pool_buf = pool_buf,
    .epochs_per_batch = c->cl.epochs_per_batch,
    .lod_done = 0,
  };
  for (uint32_t i = 0; i < n_epochs; ++i) {
    in.batch_active_masks[i] = 0x1;
    in.epoch_events[i] = epoch_events[i];
  }

  memset(handoff, 0, sizeof(*handoff));

  CHECK(Fail,
        compress_agg_kick(&c->ca,
                          &in,
                          &c->cl.levels,
                          &c->batch,
                          &c->cl.dims,
                          c->compute,
                          handoff) == 0);

  CHECK(Fail,
        d2h_deliver_kick(&c->d2h,
                         handoff,
                         &c->cl.levels,
                         &c->batch,
                         &c->cl.dims,
                         config,
                         sink,
                         c->d2h_stream) == 0);

  struct writer_result r = d2h_deliver_drain(&c->d2h,
                                             handoff,
                                             &c->cl.levels,
                                             &c->batch,
                                             &c->cl.dims,
                                             &c->cl.layouts[0],
                                             config,
                                             sink,
                                             &c->lod,
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
  make_test_config(&config, dims, CODEC_NONE, 1);

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

  // Create and record epoch event
  CU(Fail, cuEventCreate(&c.epoch_events[0], CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(c.epoch_events[0], c.compute));

  // Kick compress_agg + D2H + drain
  struct flush_handoff handoff;
  CHECK(Fail,
        test_ctx_kick_and_drain(
          &c, &config, &sink.base, 0, 1, c.d_pool, c.epoch_events, &handoff) ==
          0);

  // Verify sink state
  CHECK(Fail, sink.open_count == 1);     // shard_inner_count=1
  CHECK(Fail, sink.finalize_count == 0); // tps_0=2, need 2 epochs

  // Verify metrics (sink uses platform_toc, always fires)
  CHECK(Fail, c.metrics.sink.count == 1);
  CHECK(Fail, c.metrics.lod_gather.count == 0);

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
  make_test_config(&config, dims, CODEC_NONE, 2);

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

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&c.epoch_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(c.epoch_events[i], c.compute));
  }

  // Kick with 2 epochs
  struct flush_handoff handoff;
  CHECK(Fail,
        test_ctx_kick_and_drain(
          &c, &config, &sink.base, 0, 2, c.d_pool, c.epoch_events, &handoff) ==
          0);

  // tps_0=2, 2 epochs → shard complete
  CHECK(Fail, sink.finalize_count == 1);

  // Parse finalized shard: index block at end
  // chunks_per_shard_total = 8, index = 8 * 16 bytes + 4 byte CRC
  {
    struct shard_state* ss = &c.ca.levels[0].shard;
    uint64_t tps_total = ss->chunks_per_shard_total;
    size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    CHECK(Fail, sink.writers[0][0].size >= index_total_bytes);
    size_t index_start = sink.writers[0][0].size - index_total_bytes;

    const uint64_t* idx =
      (const uint64_t*)(sink.writers[0][0].buf + index_start);

    // The batch LUT aggregate interleaves epochs: output position =
    // perm_pos * batch_count + epoch. deliver_to_shards_batch reads
    // them linearly, so shard index slot_idx maps as:
    //   perm_pos = slot_idx / batch_count
    //   epoch    = slot_idx % batch_count
    const struct aggregate_layout* al = &c.ca.levels[0].agg_layout;
    uint32_t batch_count = c.ca.levels[0].batch_active_count;
    uint64_t chunks_lv = c.cl.levels.chunk_count[0];

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
      uint64_t perm_pos = slot / batch_count;
      uint32_t epoch = (uint32_t)(slot % batch_count);
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
// Test 3: CODEC_ZSTD, K=1, single epoch — compressed data arrives
// ---------------------------------------------------------------------------
static int
test_d2h_zstd_single_epoch(void)
{
  log_info("=== test_d2h_zstd_single_epoch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_ZSTD, 1);

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct test_ctx c;
  test_ctx_init(&c);
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail, test_ctx_setup(&c, &config, 1) == 0);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(config.dtype);
  const size_t chunk_bytes = chunk_stride * bytes_per_element;

  CHECK(
    Fail,
    fill_pool_epoch(
      c.d_pool, total_chunks, chunk_stride, bytes_per_element, fill_epoch0) ==
      0);

  CU(Fail, cuEventCreate(&c.epoch_events[0], CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(c.epoch_events[0], c.compute));

  struct flush_handoff handoff;
  CHECK(Fail,
        test_ctx_kick_and_drain(
          &c, &config, &sink.base, 0, 1, c.d_pool, c.epoch_events, &handoff) ==
          0);

  CHECK(Fail, sink.writers[0][0].size > 0);

  // Decompress and verify chunk data
  {
    struct shard_state* ss = &c.ca.levels[0].shard;
    struct active_shard* sh = &ss->shards[0];
    const struct aggregate_layout* al = &c.ca.levels[0].agg_layout;

    decomp_buf = (uint8_t*)malloc(chunk_bytes);
    CHECK(Fail, decomp_buf);

    int errors = 0;
    for (uint64_t t = 0; t < total_chunks; ++t) {
      uint32_t pi =
        cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
      uint64_t tile_off = sh->index[2 * pi];
      uint64_t tile_sz = sh->index[2 * pi + 1];

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
  make_test_config(&config, dims, CODEC_NONE, 1);

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
  CU(Fail, cuEventCreate(&c.epoch_events[0], CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(c.epoch_events[0], c.compute));

  {
    struct flush_handoff handoff;
    CHECK(
      Fail,
      test_ctx_kick_and_drain(
        &c, &config, &sink.base, 0, 1, c.d_pool, c.epoch_events, &handoff) ==
        0);
  }

  CHECK(Fail, sink.finalize_count == 0); // 1 of 2 epochs

  // Iteration 2: fc=1, fill with epoch1
  CHECK(Fail,
        fill_pool_epoch(c.d_pool + epoch_pool_bytes,
                        total_chunks,
                        chunk_stride,
                        bytes_per_element,
                        fill_epoch1) == 0);
  CU(Fail, cuEventCreate(&c.epoch_events[1], CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(c.epoch_events[1], c.compute));

  {
    struct flush_handoff handoff;
    CHECK(Fail,
          test_ctx_kick_and_drain(&c,
                                  &config,
                                  &sink.base,
                                  1,
                                  1,
                                  c.d_pool + epoch_pool_bytes,
                                  &c.epoch_events[1],
                                  &handoff) == 0);
  }

  CHECK(Fail, sink.finalize_count == 1); // shard complete

  // Parse finalized shard and verify both epochs' data
  {
    struct shard_state* ss = &c.ca.levels[0].shard;
    uint64_t tps_total = ss->chunks_per_shard_total;
    size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    CHECK(Fail, sink.writers[0][0].size >= index_total_bytes);
    size_t index_start = sink.writers[0][0].size - index_total_bytes;
    const uint64_t* idx =
      (const uint64_t*)(sink.writers[0][0].buf + index_start);

    const struct aggregate_layout* al = &c.ca.levels[0].agg_layout;
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

RUN_GPU_TESTS({ "d2h_single_epoch_none", test_d2h_single_epoch_none },
              { "d2h_batch_none", test_d2h_batch_none },
              { "d2h_zstd_single_epoch", test_d2h_zstd_single_epoch },
              { "d2h_double_buffer", test_d2h_double_buffer }, )

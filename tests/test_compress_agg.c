#include "gpu/flush.compress_agg.h"
#include "stream/config.h"

#include "index.ops.util.h"
#include "test_gpu_helpers.h"
#include "test_runner.h"
#include "test_shard_verify.h"

#include "gpu/prelude.cuda.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// ---------------------------------------------------------------------------
// Shared test context: setup, kick, teardown
// ---------------------------------------------------------------------------

struct ca_test_ctx
{
  struct tile_stream_configuration config;
  struct dimension dims[3];
  struct computed_stream_layouts cl;
  struct compress_agg_stage stage;
  CUstream compute;
  CUdeviceptr d_pool;
  CUevent epoch_events[2];
  struct batch_state batch;
  int stage_inited;
};

static void
ca_ctx_init(struct ca_test_ctx* c)
{
  memset(c, 0, sizeof(*c));
}

static void
ca_ctx_destroy(struct ca_test_ctx* c)
{
  if (c->stage_inited)
    compress_agg_destroy(&c->stage, c->cl.levels.nlod);
  computed_stream_layouts_free(&c->cl);
  cu_mem_free(c->d_pool);
  for (int i = 0; i < 2; ++i)
    cu_event_destroy(c->epoch_events[i]);
  cu_stream_destroy(c->compute);
}

// Setup: compute layouts, init compress_agg, allocate pool for n_pool_epochs.
static int
ca_ctx_setup(struct ca_test_ctx* c,
             struct codec_config codec,
             uint8_t epochs_per_batch,
             int n_pool_epochs)
{
  make_test_config(&c->config, c->dims, codec, epochs_per_batch);
  CHECK(Fail,
        compute_stream_layouts(&c->config,
                               codec_alignment(c->config.codec.id),
                               codec_max_output_size,
                               &c->cl) == 0);

  CU(Fail, cuStreamCreate(&c->compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail,
        compress_agg_init(&c->stage, &c->cl, &c->config, c->compute) == 0);
  c->stage_inited = 1;

  size_t pool_bytes = (uint64_t)n_pool_epochs * c->cl.levels.total_chunks *
                      c->cl.layouts[0].chunk_stride *
                      dtype_bpe(c->config.dtype);
  CU(Fail, cuMemAlloc(&c->d_pool, pool_bytes));

  c->batch = (struct batch_state){
    .epochs_per_batch = c->cl.epochs_per_batch,
    .accumulated = 0,
  };
  return 0;

Fail:
  return 1;
}

// Fill epoch n in pool, create/record event.
static int
ca_ctx_fill_epoch(struct ca_test_ctx* c,
                  int epoch_idx,
                  uint16_t (*fill_fn)(uint64_t))
{
  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t bytes_per_element = dtype_bpe(c->config.dtype);
  CUdeviceptr epoch_ptr = c->d_pool + (uint64_t)epoch_idx * total_chunks *
                                        chunk_stride * bytes_per_element;
  CHECK(Fail,
        fill_pool_epoch(
          epoch_ptr, total_chunks, chunk_stride, bytes_per_element, fill_fn) ==
          0);

  CU(Fail, cuEventCreate(&c->epoch_events[epoch_idx], CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(c->epoch_events[epoch_idx], c->compute));
  return 0;

Fail:
  return 1;
}

// Build input, kick compress_agg, sync.
static int
ca_ctx_kick(struct ca_test_ctx* c,
            uint32_t n_epochs,
            struct flush_handoff* handoff)
{
  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = n_epochs,
    .active_levels_mask = 0x1,
    .pool_buf = c->d_pool,
    .epochs_per_batch = c->cl.epochs_per_batch,
    .lod_done = 0,
  };
  for (uint32_t i = 0; i < n_epochs; ++i) {
    in.batch_active_masks[i] = 0x1;
    in.epoch_events[i] = c->epoch_events[i];
  }

  memset(handoff, 0, sizeof(*handoff));

  CHECK(Fail,
        compress_agg_kick(&c->stage,
                          &in,
                          &c->cl.levels,
                          &c->batch,
                          &c->cl.dims,
                          c->compute,
                          handoff) == 0);
  CU(Fail, cuStreamSynchronize(c->compute));
  return 0;

Fail:
  return 1;
}

// D2H aggregate offsets and data for level 0. Caller frees *out_agg_data.
static int
ca_ctx_fetch_agg(struct flush_handoff* handoff,
                 uint64_t n_covering,
                 void** out_agg_data)
{
  struct aggregate_slot* agg = handoff->agg[0];

  CU(Fail,
     cuMemcpyDtoH(agg->h_offsets,
                  (CUdeviceptr)agg->d_offsets,
                  (n_covering + 1) * sizeof(size_t)));

  size_t total_data = agg->h_offsets[n_covering];
  void* h_agg = malloc(total_data);
  CHECK(Fail, h_agg);
  CU(Fail, cuMemcpyDtoH(h_agg, (CUdeviceptr)agg->d_aggregated, total_data));

  *out_agg_data = h_agg;
  return 0;

Fail:
  return 1;
}

// Verify uncompressed chunk data for a single-epoch (non-batch) aggregate.
static int
verify_tiles_none(const struct flush_handoff* handoff,
                  const struct ca_test_ctx* c,
                  const void* h_agg,
                  uint16_t (*fill_fn)(uint64_t))
{
  const struct aggregate_layout* al = handoff->agg_layout[0];
  const struct aggregate_slot* agg = handoff->agg[0];
  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t chunk_bytes = chunk_stride * dtype_bpe(c->config.dtype);

  int errors = 0;
  for (uint64_t t = 0; t < total_chunks; ++t) {
    uint32_t pi =
      cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
    size_t off = agg->h_offsets[pi];
    size_t sz = agg->h_offsets[pi + 1] - off;
    if (sz != chunk_bytes) {
      if (errors < 5)
        log_error("  chunk %lu: size=%zu expected=%zu",
                  (unsigned long)t,
                  sz,
                  chunk_bytes);
      errors++;
      continue;
    }

    uint16_t expected_val = fill_fn(t);
    const uint16_t* got = (const uint16_t*)((const char*)h_agg + off);
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
  return errors;
}

// ---------------------------------------------------------------------------
// Test 1: CODEC_NONE, K=1, single epoch
// ---------------------------------------------------------------------------
static int
test_compress_agg_single_epoch(void)
{
  log_info("=== test_compress_agg_single_epoch ===");

  struct ca_test_ctx c;
  ca_ctx_init(&c);
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail,
        ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_NONE }, 1, 1) == 0);
  CHECK(Fail, ca_ctx_fill_epoch(&c, 0, fill_epoch0) == 0);

  struct flush_handoff handoff;
  CHECK(Fail, ca_ctx_kick(&c, 1, &handoff) == 0);

  // Verify handoff
  const size_t chunk_bytes =
    c.cl.layouts[0].chunk_stride * dtype_bpe(c.config.dtype);
  CHECK(Fail, handoff.fc == 0);
  CHECK(Fail, handoff.n_epochs == 1);
  CHECK(Fail, handoff.active_levels_mask == 0x1);
  CHECK(Fail, handoff.t_aggregate_end != 0);
  CHECK(Fail, handoff.t_compress_start != 0);
  CHECK(Fail, handoff.t_compress_end != 0);
  CHECK(Fail, handoff.max_output_size == chunk_bytes);
  CHECK(Fail, handoff.agg[0] != NULL);
  CHECK(Fail, handoff.agg_layout[0] != NULL);

  // D2H and verify
  uint64_t C = handoff.agg_layout[0]->covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff, C, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(handoff.agg[0]->h_offsets, C) == 0);

  size_t expected_total = c.cl.levels.total_chunks * chunk_bytes;
  CHECK(Fail, handoff.agg[0]->h_offsets[C] == expected_total);
  CHECK(Fail, verify_tiles_none(&handoff, &c, h_agg, fill_epoch0) == 0);

  ok = 1;

Fail:
  free(h_agg);
  ca_ctx_destroy(&c);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: CODEC_NONE, K=2, batch LUT path
// ---------------------------------------------------------------------------
static int
test_compress_agg_batch(void)
{
  log_info("=== test_compress_agg_batch ===");

  struct ca_test_ctx c;
  ca_ctx_init(&c);
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail,
        ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_NONE }, 2, 2) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  CHECK(Fail, ca_ctx_fill_epoch(&c, 0, fill_epoch0) == 0);
  CHECK(Fail, ca_ctx_fill_epoch(&c, 1, fill_epoch1) == 0);

  struct flush_handoff handoff;
  CHECK(Fail, ca_ctx_kick(&c, 2, &handoff) == 0);

  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t chunk_bytes = chunk_stride * dtype_bpe(c.config.dtype);
  CHECK(Fail, handoff.n_epochs == 2);
  CHECK(Fail, handoff.max_output_size == chunk_bytes);

  // D2H aggregate
  const struct aggregate_layout* al = handoff.agg_layout[0];
  uint64_t C = al->covering_count;
  uint32_t batch_count = c.stage.levels[0].batch_active_count;
  uint64_t batch_covering = (uint64_t)batch_count * C;

  CHECK(Fail, ca_ctx_fetch_agg(&handoff, batch_covering, &h_agg) == 0);
  CHECK(Fail,
        verify_offsets_monotonic(handoff.agg[0]->h_offsets, batch_covering) ==
          0);

  size_t expected_total = 2 * c.cl.levels.total_chunks * chunk_bytes;
  CHECK(Fail, handoff.agg[0]->h_offsets[batch_covering] == expected_total);

  // Verify data per epoch
  uint64_t chunks_lv = c.cl.levels.chunk_count[0];
  uint32_t cps_inner = (uint32_t)al->cps_inner;
  uint32_t num_shards = (uint32_t)(al->covering_count / cps_inner);
  const uint64_t shard_shape[2] = { num_shards, cps_inner };
  const int64_t shard_strides[2] = { (int64_t)(batch_count * cps_inner), 1 };
  int errors = 0;
  for (uint32_t a = 0; a < batch_count; ++a) {
    uint16_t (*fill_fn)(uint64_t) = (a == 0) ? fill_epoch0 : fill_epoch1;
    for (uint64_t j = 0; j < chunks_lv; ++j) {
      uint64_t perm_pos =
        ravel(al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
      uint64_t out_idx =
        ravel(2, shard_shape, shard_strides, perm_pos) + a * cps_inner;
      size_t off = handoff.agg[0]->h_offsets[out_idx];
      size_t sz = handoff.agg[0]->h_offsets[out_idx + 1] - off;
      if (sz != chunk_bytes) {
        if (errors < 5)
          log_error("  epoch %u chunk %lu: size=%zu expected=%zu",
                    a,
                    (unsigned long)j,
                    sz,
                    chunk_bytes);
        errors++;
        continue;
      }
      uint16_t expected_val = fill_fn(j);
      const uint16_t* got = (const uint16_t*)((char*)h_agg + off);
      for (uint64_t e = 0; e < chunk_stride; ++e) {
        if (got[e] != expected_val) {
          if (errors < 5)
            log_error("  epoch %u chunk %lu elem %lu: expected %u got %u",
                      a,
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

  ok = 1;

Fail:
  free(h_agg);
  ca_ctx_destroy(&c);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 3: CODEC_NONE, K=2 but n_epochs=1 (partial batch, no LUTs)
// ---------------------------------------------------------------------------
static int
test_compress_agg_partial_batch(void)
{
  log_info("=== test_compress_agg_partial_batch ===");

  struct ca_test_ctx c;
  ca_ctx_init(&c);
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail,
        ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_NONE }, 2, 2) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  CHECK(Fail, ca_ctx_fill_epoch(&c, 0, fill_epoch0) == 0);

  // Kick with n_epochs=1 even though K=2 -> partial batch
  struct flush_handoff handoff;
  CHECK(Fail, ca_ctx_kick(&c, 1, &handoff) == 0);
  CHECK(Fail, handoff.n_epochs == 1);

  const size_t chunk_bytes =
    c.cl.layouts[0].chunk_stride * dtype_bpe(c.config.dtype);
  uint64_t C = handoff.agg_layout[0]->covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff, C, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(handoff.agg[0]->h_offsets, C) == 0);

  size_t expected_total = c.cl.levels.total_chunks * chunk_bytes;
  CHECK(Fail, handoff.agg[0]->h_offsets[C] == expected_total);
  CHECK(Fail, verify_tiles_none(&handoff, &c, h_agg, fill_epoch0) == 0);

  ok = 1;

Fail:
  free(h_agg);
  ca_ctx_destroy(&c);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 4: CODEC_ZSTD, K=1, single epoch
// ---------------------------------------------------------------------------
static int
test_compress_agg_zstd_single_epoch(void)
{
  log_info("=== test_compress_agg_zstd_single_epoch ===");

  struct ca_test_ctx c;
  ca_ctx_init(&c);
  void* h_agg = NULL;
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail,
        ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_ZSTD }, 1, 1) == 0);
  CHECK(Fail, ca_ctx_fill_epoch(&c, 0, fill_epoch0) == 0);

  struct flush_handoff handoff;
  CHECK(Fail, ca_ctx_kick(&c, 1, &handoff) == 0);

  const uint64_t total_chunks = c.cl.levels.total_chunks;
  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t chunk_bytes = chunk_stride * dtype_bpe(c.config.dtype);

  const struct aggregate_layout* al = handoff.agg_layout[0];
  uint64_t C = al->covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff, C, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(handoff.agg[0]->h_offsets, C) == 0);

  decomp_buf = (uint8_t*)malloc(chunk_bytes);
  CHECK(Fail, decomp_buf);

  int errors = 0;
  for (uint64_t t = 0; t < total_chunks; ++t) {
    uint32_t pi =
      cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
    size_t off = handoff.agg[0]->h_offsets[pi];
    size_t comp_sz = handoff.agg[0]->h_offsets[pi + 1] - off;
    CHECK(Fail, comp_sz > 0);
    CHECK(Fail, comp_sz <= handoff.max_output_size);

    size_t result =
      ZSTD_decompress(decomp_buf, chunk_bytes, (char*)h_agg + off, comp_sz);
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

  ok = 1;

Fail:
  free(h_agg);
  free(decomp_buf);
  ca_ctx_destroy(&c);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 5: CODEC_ZSTD, K=2, batch LUT path
// ---------------------------------------------------------------------------
static int
test_compress_agg_zstd_batch(void)
{
  log_info("=== test_compress_agg_zstd_batch ===");

  struct ca_test_ctx c;
  ca_ctx_init(&c);
  void* h_agg = NULL;
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail,
        ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_ZSTD }, 2, 2) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  CHECK(Fail, ca_ctx_fill_epoch(&c, 0, fill_epoch0) == 0);
  CHECK(Fail, ca_ctx_fill_epoch(&c, 1, fill_epoch1) == 0);

  struct flush_handoff handoff;
  CHECK(Fail, ca_ctx_kick(&c, 2, &handoff) == 0);

  const uint64_t chunk_stride = c.cl.layouts[0].chunk_stride;
  const size_t chunk_bytes = chunk_stride * dtype_bpe(c.config.dtype);
  const struct aggregate_layout* al = handoff.agg_layout[0];
  uint64_t C = al->covering_count;
  uint32_t batch_count = c.stage.levels[0].batch_active_count;
  uint64_t batch_covering = (uint64_t)batch_count * C;

  CHECK(Fail, ca_ctx_fetch_agg(&handoff, batch_covering, &h_agg) == 0);
  CHECK(Fail,
        verify_offsets_monotonic(handoff.agg[0]->h_offsets, batch_covering) ==
          0);

  decomp_buf = (uint8_t*)malloc(chunk_bytes);
  CHECK(Fail, decomp_buf);

  uint64_t chunks_lv = c.cl.levels.chunk_count[0];
  uint32_t cps_inner = (uint32_t)al->cps_inner;
  uint32_t num_shards = (uint32_t)(al->covering_count / cps_inner);
  const uint64_t shard_shape[2] = { num_shards, cps_inner };
  const int64_t shard_strides[2] = { (int64_t)(batch_count * cps_inner), 1 };
  int errors = 0;
  for (uint32_t a = 0; a < batch_count; ++a) {
    uint16_t (*fill_fn)(uint64_t) = (a == 0) ? fill_epoch0 : fill_epoch1;
    for (uint64_t j = 0; j < chunks_lv; ++j) {
      uint64_t perm_pos =
        ravel(al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
      uint64_t out_idx =
        ravel(2, shard_shape, shard_strides, perm_pos) + a * cps_inner;
      size_t off = handoff.agg[0]->h_offsets[out_idx];
      size_t comp_sz = handoff.agg[0]->h_offsets[out_idx + 1] - off;

      CHECK(Fail, comp_sz > 0);
      CHECK(Fail, comp_sz <= handoff.max_output_size);

      size_t result =
        ZSTD_decompress(decomp_buf, chunk_bytes, (char*)h_agg + off, comp_sz);
      if (ZSTD_isError(result)) {
        log_error("  epoch %u chunk %lu: ZSTD_decompress failed: %s",
                  a,
                  (unsigned long)j,
                  ZSTD_getErrorName(result));
        errors++;
        continue;
      }
      CHECK(Fail, result == chunk_bytes);

      uint16_t expected_val = fill_fn(j);
      const uint16_t* got = (const uint16_t*)decomp_buf;
      for (uint64_t e = 0; e < chunk_stride; ++e) {
        if (got[e] != expected_val) {
          if (errors < 5)
            log_error("  epoch %u chunk %lu elem %lu: expected %u got %u",
                      a,
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

  ok = 1;

Fail:
  free(h_agg);
  free(decomp_buf);
  ca_ctx_destroy(&c);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "compress_agg_single_epoch", test_compress_agg_single_epoch },
              { "compress_agg_batch", test_compress_agg_batch },
              { "compress_agg_partial_batch", test_compress_agg_partial_batch },
              { "compress_agg_zstd_single_epoch",
                test_compress_agg_zstd_single_epoch },
              { "compress_agg_zstd_batch", test_compress_agg_zstd_batch }, )

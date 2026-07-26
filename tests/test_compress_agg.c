#include "gpu/flush.compress_agg.h"
#include "gpu/schedule.h"
#include "stream/config.h"
#include "stream/types.aggregate.h"

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
  struct gpu_ordering ord;      // edge registry the harness pools draw from
  struct gpu_pool pool;         // chunk pool faked over d_pool (both slots)
  uint32_t* batch_active_masks; // [K] owned scratch for tests
  int ord_inited;
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
    compress_agg_destroy(&c->stage);
  if (c->ord_inited)
    gpu_ordering_destroy(&c->ord);
  computed_stream_layouts_free(&c->cl);
  cu_mem_free(c->d_pool);
  free(c->batch_active_masks);
  cu_stream_destroy(c->compute);
}

// Setup: compute layouts, init compress_agg, allocate pool for n_pool_epochs.
static int
ca_ctx_setup(struct ca_test_ctx* c,
             struct codec_config codec,
             uint32_t epochs_per_batch,
             int n_pool_epochs)
{
  make_test_config(&c->config, c->dims, codec, epochs_per_batch);
  CHECK(Fail,
        compute_stream_layouts(&c->config,
                               codec_alignment(c->config.codec.id),
                               codec_max_output_size,
                               platform_page_alignment(),
                               &c->cl) == 0);

  CU(Fail, cuStreamCreate(&c->compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, gpu_ordering_init(&c->ord, c->compute) == 0);
  c->ord_inited = 1;
  CHECK(Fail,
        compress_agg_init(&c->stage, &c->cl, &c->config, &c->ord, c->compute) ==
          0);
  c->stage_inited = 1;

  size_t pool_bytes = (uint64_t)n_pool_epochs * c->cl.levels.total_chunks *
                      c->cl.layouts[0].chunk_stride *
                      dtype_bpe(c->config.dtype);
  CU(Fail, cuMemAlloc(&c->d_pool, pool_bytes));
  gpu_pool_init(
    &c->pool, &c->ord, GPU_EDGE_POOL_FILLED, GPU_EDGE_POOL_CONSUMED);
  for (int fc = 0; fc < 2; ++fc)
    gpu_pool_bind(&c->pool, fc, (void*)(uintptr_t)c->d_pool);

  c->batch_active_masks =
    (uint32_t*)calloc(c->cl.epochs_per_batch, sizeof(uint32_t));
  CHECK(Fail, c->batch_active_masks);
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

  // Re-release pool-filled on the compute stream after every fill so the
  // kick's acquire sees batch-level readiness.
  CHECK(Fail, gpu_pool_release_produce(&c->pool, 0, c->compute) == 0);
  return 0;

Fail:
  return 1;
}

// Stand in for delivery's tail release (flush.d2h_deliver.c): tests drive
// the kick without the delivery loop, and an unreleased generation parks
// the next kick's tail gate forever.
static void
ca_ctx_publish_tail(struct ca_test_ctx* c)
{
  gpu_pool_release_produce_gen(&c->stage.tail);
}

// Build input, kick compress_agg, sync.
static int
ca_ctx_kick(struct ca_test_ctx* c,
            uint32_t n_epochs,
            struct flush_handoff* handoff)
{
  for (uint32_t i = 0; i < n_epochs; ++i)
    c->batch_active_masks[i] = 0x1;

  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = n_epochs,
    .active_levels_mask = 0x1,
    .batch_active_masks = c->batch_active_masks,
    .epochs_per_batch = c->cl.epochs_per_batch,
  };

  memset(handoff, 0, sizeof(*handoff));

  CHECK(Fail,
        schedule_compress_agg_kick(
          &c->stage, &in, &c->cl.levels, &c->pool, 0, c->compute, handoff) ==
          0);
  CU(Fail, cuStreamSynchronize(c->compute));
  ca_ctx_publish_tail(c);
  return 0;

Fail:
  return 1;
}

// Tests host-sync the compute stream before reading slot contents, so the
// host-ordered peek is the right acquire.
static struct aggregate_slot*
handoff_slot(const struct flush_handoff* handoff)
{
  return gpu_pool_at(handoff->agg_host, handoff->fc, 0).p;
}

// D2H aggregate offsets and data for level 0. Caller frees *out_agg_data.
// Copies the full d_aggregated pool because the new tail-carryover layout
// places each shard's chunks at fixed `s * shard_capacity` offsets — not
// contiguously packed at the start as the old pad-to-page layout did.
//
// Unified handoff: the slot holds all LODs' offsets/sizes/data in a single
// buffer. For single-LOD tests (lv=0), seg->batch_covering_offset == 0 and
// seg->data_segment_offset == 0, so the +lv shift accounts for the per-LOD
// disjoint-span offset built into aggregate_batch_luts_unified.
static int
ca_ctx_fetch_agg(struct flush_handoff* handoff,
                 uint64_t n_covering,
                 void** out_agg_data)
{
  struct aggregate_slot* agg = handoff_slot(handoff);
  const struct lod_segment* seg = &handoff->layout.lods[0];
  const uint64_t base = seg->batch_covering_offset + 0; // lv=0

  // (n_covering + 1) entries for LOD 0's offsets, shifted by base.
  CU(Fail,
     cuMemcpyDtoH(agg->h_offsets + base,
                  (CUdeviceptr)(agg->d_offsets + base),
                  (n_covering + 1) * sizeof(size_t)));
  // Production fetches h_permuted_sizes via d2h_deliver_kick on d2h_stream;
  // tests that drive the kick directly do the D2H here.
  CU(Fail,
     cuMemcpyDtoH(agg->h_permuted_sizes + base,
                  (CUdeviceptr)(agg->d_permuted_sizes + base),
                  n_covering * sizeof(size_t)));

  size_t pool_bytes = agg_pool_bytes_layout(&handoff->per_lod_agg_layouts[0]);
  void* h_agg = malloc(pool_bytes);
  CHECK(Fail, h_agg);
  CU(Fail,
     cuMemcpyDtoH(h_agg,
                  (CUdeviceptr)((const uint8_t*)agg->d_aggregated +
                                seg->data_segment_offset),
                  pool_bytes));

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
  const struct aggregate_layout* al = &handoff->per_lod_agg_layouts[0];
  const struct aggregate_slot* agg = handoff_slot(handoff);
  const struct lod_segment* seg = &handoff->layout.lods[0];
  const size_t* lv_offsets = agg->h_offsets + seg->batch_covering_offset + 0;
  const uint64_t total_chunks = c->cl.levels.total_chunks;
  const uint64_t chunk_stride = c->cl.layouts[0].chunk_stride;
  const size_t chunk_bytes = chunk_stride * dtype_bpe(c->config.dtype);

  int errors = 0;
  for (uint64_t t = 0; t < total_chunks; ++t) {
    uint32_t pi =
      cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
    size_t off = lv_offsets[pi];
    size_t sz = lv_offsets[pi + 1] - off;
    // Last chunk per shard may include alignment padding; check >= instead.
    if (sz < chunk_bytes) {
      if (errors < 5)
        log_error("  chunk %lu: size=%zu expected>=%zu",
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
  CHECK(Fail, handoff.agg_pool != NULL);
  CHECK(Fail, handoff.per_lod_agg_layouts != NULL);

  // D2H and verify
  const struct lod_segment* seg0 = &handoff.layout.lods[0];
  const size_t* lv_offsets =
    handoff_slot(&handoff)->h_offsets + seg0->batch_covering_offset + 0;
  uint64_t C = handoff.per_lod_agg_layouts[0].covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff, C, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(lv_offsets, C) == 0);

  {
    // h_offsets[C] is the un-biased prefix-sum sentinel: sum of all real
    // permuted_sizes — chunk_bytes per real chunk for CODEC_NONE.
    // Each shard's data starts at s * shard_capacity in h_agg.
    const struct aggregate_layout* al = &handoff.per_lod_agg_layouts[0];
    uint64_t num_shards = C / al->cps_inner;
    uint64_t N = (uint64_t)c.stage.ar.per_lod_agg_layouts[0].active_count_max *
                 c.cl.levels.level[0].chunk_count;
    CHECK(Fail, lv_offsets[C] == N * chunk_bytes);
    for (uint64_t s = 0; s < num_shards; ++s)
      CHECK(Fail, lv_offsets[s * al->cps_inner] == s * al->shard_capacity);
  }
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
  const struct aggregate_layout* al = &handoff.per_lod_agg_layouts[0];
  const struct lod_segment* seg = &handoff.layout.lods[0];
  const size_t* lv_offsets =
    handoff_slot(&handoff)->h_offsets + seg->batch_covering_offset + 0;
  uint64_t C = al->covering_count;
  uint32_t batch_count = c.stage.ar.per_lod_agg_layouts[0].active_count_max;
  uint64_t batch_covering = (uint64_t)batch_count * C;

  CHECK(Fail, ca_ctx_fetch_agg(&handoff, batch_covering, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(lv_offsets, batch_covering) == 0);

  {
    // h_offsets[batch_covering] is the un-biased prefix-sum sentinel: real
    // chunk bytes summed across all chunks. Each shard's data starts at
    // s * shard_capacity in h_agg.
    uint64_t num_shards = C / al->cps_inner;
    uint64_t tps_group = batch_covering / num_shards;
    uint64_t N = (uint64_t)batch_count * c.cl.levels.level[0].chunk_count;
    CHECK(Fail, lv_offsets[batch_covering] == N * chunk_bytes);
    for (uint64_t s = 0; s < num_shards; ++s)
      CHECK(Fail, lv_offsets[s * tps_group] == s * al->shard_capacity);
  }

  // Verify data per epoch
  uint64_t chunks_lv = c.cl.levels.level[0].chunk_count;
  uint32_t cps_inner = (uint32_t)al->cps_inner;
  uint32_t num_shards = (uint32_t)(al->covering_count / cps_inner);
  const uint64_t shard_shape[2] = { num_shards, cps_inner };
  const int64_t shard_strides[2] = { (int64_t)batch_count * cps_inner, 1 };
  int errors = 0;
  for (uint32_t a = 0; a < batch_count; ++a) {
    uint16_t (*fill_fn)(uint64_t) = (a == 0) ? fill_epoch0 : fill_epoch1;
    for (uint64_t j = 0; j < chunks_lv; ++j) {
      uint64_t perm_pos =
        ravel(al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
      uint64_t out_idx =
        ravel(2, shard_shape, shard_strides, perm_pos) + a * cps_inner;
      size_t off = lv_offsets[out_idx];
      size_t sz = lv_offsets[out_idx + 1] - off;
      // Last chunk per shard may include alignment padding.
      if (sz < chunk_bytes) {
        if (errors < 5)
          log_error("  epoch %u chunk %lu: size=%zu expected>=%zu",
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
  const struct lod_segment* seg = &handoff.layout.lods[0];
  const size_t* lv_offsets =
    handoff_slot(&handoff)->h_offsets + seg->batch_covering_offset + 0;
  uint64_t C = handoff.per_lod_agg_layouts[0].covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff, C, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(lv_offsets, C) == 0);

  {
    // Partial batch: only the actually-active 1 epoch's worth of chunks are
    // filled by permute_sizes_batch_k; the rest of d_permuted_sizes stays 0
    // (from cuMemset). Sentinel h_offsets[C] therefore equals N*chunk_bytes
    // where N is the active-epoch chunk count, not K * chunks_lv.
    const struct aggregate_layout* al = &handoff.per_lod_agg_layouts[0];
    uint64_t num_shards = C / al->cps_inner;
    uint64_t N = (uint64_t)1 * c.cl.levels.level[0].chunk_count; // n_epochs=1
    CHECK(Fail, lv_offsets[C] == N * chunk_bytes);
    for (uint64_t s = 0; s < num_shards; ++s)
      CHECK(Fail, lv_offsets[s * al->cps_inner] == s * al->shard_capacity);
  }
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

  const struct aggregate_layout* al = &handoff.per_lod_agg_layouts[0];
  const struct lod_segment* seg = &handoff.layout.lods[0];
  const size_t* lv_offsets =
    handoff_slot(&handoff)->h_offsets + seg->batch_covering_offset + 0;
  const size_t* lv_sizes =
    handoff_slot(&handoff)->h_permuted_sizes + seg->batch_covering_offset + 0;
  uint64_t C = al->covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff, C, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(lv_offsets, C) == 0);

  decomp_buf = (uint8_t*)malloc(chunk_bytes);
  CHECK(Fail, decomp_buf);

  int errors = 0;
  for (uint64_t t = 0; t < total_chunks; ++t) {
    uint32_t pi =
      cpu_perm(t, al->lifted_rank, al->lifted_shape, al->lifted_strides);
    size_t off = lv_offsets[pi];
    size_t comp_sz = lv_sizes[pi];
    CHECK(Fail, comp_sz > 0);
    // Last chunk per shard may include alignment padding.
    CHECK(Fail, comp_sz <= handoff.max_output_size + al->page_size);

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
  const struct aggregate_layout* al = &handoff.per_lod_agg_layouts[0];
  const struct lod_segment* seg = &handoff.layout.lods[0];
  const size_t* lv_offsets =
    handoff_slot(&handoff)->h_offsets + seg->batch_covering_offset + 0;
  uint64_t C = al->covering_count;
  uint32_t batch_count = c.stage.ar.per_lod_agg_layouts[0].active_count_max;
  uint64_t batch_covering = (uint64_t)batch_count * C;

  CHECK(Fail, ca_ctx_fetch_agg(&handoff, batch_covering, &h_agg) == 0);
  CHECK(Fail, verify_offsets_monotonic(lv_offsets, batch_covering) == 0);

  decomp_buf = (uint8_t*)malloc(chunk_bytes);
  CHECK(Fail, decomp_buf);

  uint64_t chunks_lv = c.cl.levels.level[0].chunk_count;
  uint32_t cps_inner = (uint32_t)al->cps_inner;
  uint32_t num_shards = (uint32_t)(al->covering_count / cps_inner);
  const uint64_t shard_shape[2] = { num_shards, cps_inner };
  const int64_t shard_strides[2] = { (int64_t)batch_count * cps_inner, 1 };
  int errors = 0;
  for (uint32_t a = 0; a < batch_count; ++a) {
    uint16_t (*fill_fn)(uint64_t) = (a == 0) ? fill_epoch0 : fill_epoch1;
    for (uint64_t j = 0; j < chunks_lv; ++j) {
      uint64_t perm_pos =
        ravel(al->lifted_rank, al->lifted_shape, al->lifted_strides, j);
      uint64_t out_idx =
        ravel(2, shard_shape, shard_strides, perm_pos) + a * cps_inner;
      size_t off = lv_offsets[out_idx];
      size_t slot_sz = lv_offsets[out_idx + 1] - off;
      CHECK(Fail, slot_sz > 0);

      // Slot may include shard-boundary padding; find the actual frame size.
      size_t comp_sz =
        ZSTD_findFrameCompressedSize((char*)h_agg + off, slot_sz);
      CHECK(Fail, !ZSTD_isError(comp_sz));
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

// ---------------------------------------------------------------------------
// Test 6: CODEC_NONE skips d_compressed alloc; CODEC_ZSTD allocates it.
// Pins the "aggregate reads pool directly" invariant — without the
// allocation, any future regression that tries to read/write d_compressed
// on the codec=none path will crash on a NULL device pointer instead of
// silently waste M * chunk_bytes of VRAM.
// ---------------------------------------------------------------------------
static int
test_compress_agg_none_no_compressed_buffer(void)
{
  log_info("=== test_compress_agg_none_no_compressed_buffer ===");
  int ok = 0;

  // CODEC_NONE: d_compressed[fc] must be unallocated (0).
  {
    struct ca_test_ctx c;
    ca_ctx_init(&c);
    CHECK(NoneFail,
          ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_NONE }, 2, 2) ==
            0);
    CHECK(NoneFail, c.stage.d_compressed[0] == (CUdeviceptr)0);
    CHECK(NoneFail, c.stage.d_compressed[1] == (CUdeviceptr)0);
    ca_ctx_destroy(&c);
    goto ZstdCheck;
  NoneFail:
    ca_ctx_destroy(&c);
    goto Fail;
  }

ZstdCheck:
  // CODEC_ZSTD: d_compressed[fc] must be allocated.
  {
    struct ca_test_ctx c;
    ca_ctx_init(&c);
    CHECK(ZstdFail,
          ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_ZSTD }, 2, 2) ==
            0);
    CHECK(ZstdFail, c.stage.d_compressed[0] != (CUdeviceptr)0);
    CHECK(ZstdFail, c.stage.d_compressed[1] != (CUdeviceptr)0);
    ca_ctx_destroy(&c);
    goto Done;
  ZstdFail:
    ca_ctx_destroy(&c);
    goto Fail;
  }

Done:
  ok = 1;
Fail:
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 7: LUT cache must invalidate when pool_epoch positions shift even
// though per-LOD active counts stay the same.
//
// This pins the regression for the "counts match but positions differ" case
// (e.g. masks [1,0] then [0,1] both with n_active=1): the unified gather LUT
// encodes the actual pool_epoch values, so a cache hit keyed only on counts
// would replay the prior kick's gather indices and produce the prior kick's
// data. Without the fix, kick 2 below would gather epoch 0 data while
// reporting it as epoch 1 — the byte-exact check below catches that.
// ---------------------------------------------------------------------------
static int
test_compress_agg_lut_cache_position_shift(void)
{
  log_info("=== test_compress_agg_lut_cache_position_shift ===");

  struct ca_test_ctx c;
  ca_ctx_init(&c);
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail,
        ca_ctx_setup(&c, (struct codec_config){ .id = CODEC_NONE }, 2, 2) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  // Fill epoch 0 with one signature, epoch 1 with another.
  CHECK(Fail, ca_ctx_fill_epoch(&c, 0, fill_epoch0) == 0);
  CHECK(Fail, ca_ctx_fill_epoch(&c, 1, fill_epoch1) == 0);

  // Kick 1: only epoch 0 active (mask [1, 0]).
  // n_active=1, pool_epochs=[0]. Populates the LUT cache.
  c.batch_active_masks[0] = 0x1;
  c.batch_active_masks[1] = 0x0;
  {
    struct compress_agg_input in = {
      .fc = 0,
      .n_epochs = 2,
      .active_levels_mask = 0x1,
      .batch_active_masks = c.batch_active_masks,
      .epochs_per_batch = c.cl.epochs_per_batch,
    };
    struct flush_handoff handoff;
    memset(&handoff, 0, sizeof(handoff));
    CHECK(Fail,
          schedule_compress_agg_kick(
            &c.stage, &in, &c.cl.levels, &c.pool, 0, c.compute, &handoff) == 0);
    CU(Fail, cuStreamSynchronize(c.compute));
    ca_ctx_publish_tail(&c);
  }

  // Kick 2: only epoch 1 active (mask [0, 1]).
  // Same n_active=1 as kick 1, but pool_epochs=[1] — a steady-state cache
  // keyed on counts alone would mis-hit and gather from epoch 0 instead.
  uint64_t lut_recompute_before = c.stage.lut_recompute_count;
  c.batch_active_masks[0] = 0x0;
  c.batch_active_masks[1] = 0x1;
  struct flush_handoff handoff2;
  memset(&handoff2, 0, sizeof(handoff2));
  {
    struct compress_agg_input in = {
      .fc = 1,
      .n_epochs = 2,
      .active_levels_mask = 0x1,
      .batch_active_masks = c.batch_active_masks,
      .epochs_per_batch = c.cl.epochs_per_batch,
    };
    CHECK(Fail,
          schedule_compress_agg_kick(
            &c.stage, &in, &c.cl.levels, &c.pool, 0, c.compute, &handoff2) ==
            0);
    CU(Fail, cuStreamSynchronize(c.compute));
    ca_ctx_publish_tail(&c);
  }

  // The cache must have missed (recompute count incremented) because the
  // pool_epoch values changed even though n_active didn't.
  CHECK(Fail, c.stage.lut_recompute_count > lut_recompute_before);

  // The aggregated data must reflect epoch 1, not epoch 0.
  uint64_t C = handoff2.per_lod_agg_layouts[0].covering_count;
  CHECK(Fail, ca_ctx_fetch_agg(&handoff2, C, &h_agg) == 0);
  CHECK(Fail, verify_tiles_none(&handoff2, &c, h_agg, fill_epoch1) == 0);

  ok = 1;

Fail:
  free(h_agg);
  ca_ctx_destroy(&c);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "compress_agg_single_epoch", test_compress_agg_single_epoch },
              { "compress_agg_batch", test_compress_agg_batch },
              { "compress_agg_partial_batch", test_compress_agg_partial_batch },
              { "compress_agg_zstd_single_epoch",
                test_compress_agg_zstd_single_epoch },
              { "compress_agg_zstd_batch", test_compress_agg_zstd_batch },
              { "compress_agg_none_no_compressed_buffer",
                test_compress_agg_none_no_compressed_buffer },
              { "compress_agg_lut_cache_position_shift",
                test_compress_agg_lut_cache_position_shift }, )

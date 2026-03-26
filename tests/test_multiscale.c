#include "lod/lod_plan.h"
#include "gpu/prelude.cuda.h"
#include "util/prelude.h"
#include "stream.gpu.h"
#include "test_data.h"
#include "test_runner.h"
#include "test_shard_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- L0 correctness: multiscale vs non-multiscale ---

static int
test_multiscale_l0_correctness(void)
{
  log_info("=== test_multiscale_l0_correctness ===");

  // 5D: t, z, y, x, c. LOD on z, y, x.
  // Small enough for fast testing, large enough to exercise multiple epochs.
  struct dimension dims[] = {
    { .size = 8,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "z",
      .storage_position = 1 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 2 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 3 },
    { .size = 1,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .name = "c",
      .storage_position = 4 },
  };
  struct dimension dims_ms[] = {
    { .size = 8,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .name = "c",
      .storage_position = 4 },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims, rank);
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  // Compute number of L0 shards
  int num_shards = 1;
  for (int d = 0; d < rank; ++d) {
    int chunk_count = (int)(dims[d].size / dims[d].chunk_size);
    int shard_count = (int)(chunk_count / dims[d].chunks_per_shard);
    num_shards *= shard_count;
  }
  const size_t shard_cap = total_bytes + 4096; // generous per-shard capacity

  log_info("  total: %zu elements, %d L0 shards", total_elements, num_shards);

  // --- Run 1: non-multiscale (baseline) ---
  struct test_shard_sink baseline_sink;
  test_sink_init(&baseline_sink, num_shards, shard_cap);

  {
    struct tile_stream_gpu* s = NULL;
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .dtype = dtype_u16,
      .rank = rank,
      .dimensions = dims,
      .codec = CODEC_ZSTD,
    };
    CHECK(Fail1, (s = tile_stream_gpu_create(&config, &baseline_sink.base)) != NULL);
    xor_pattern_init(dims, rank, 2);
    CHECK(Fail1b,
          pump_data(tile_stream_gpu_writer(s), total_elements, fill_xor) == 0);
    CHECK(Fail1b, tile_stream_gpu_cursor(s) == total_elements);
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Run2;
  Fail1b:
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Fail1;
  }

Run2:
  // --- Run 2: multiscale ---
  ;
  struct test_shard_sink ms_sink;
  test_sink_init(&ms_sink, num_shards, shard_cap);

  {
    struct tile_stream_gpu* s = NULL;
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .dtype = dtype_u16,
      .rank = rank,
      .dimensions = dims_ms,
      .codec = CODEC_ZSTD,
    };
    CHECK(Fail2b, (s = tile_stream_gpu_create(&config, &ms_sink.base)) != NULL);
    xor_pattern_init(dims_ms, rank, 2);
    CHECK(Fail2c,
          pump_data(tile_stream_gpu_writer(s), total_elements, fill_xor) == 0);
    CHECK(Fail2c, tile_stream_gpu_cursor(s) == total_elements);
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Compare;
  Fail2c:
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Fail2b;
  }

Compare:
  // --- Compare L0 shard output ---
  {
    int errors = 0;
    for (int i = 0; i < num_shards; ++i) {
      struct test_shard_writer* b = &baseline_sink.writers[0][i];
      struct test_shard_writer* m = &ms_sink.writers[0][i];

      if (!b->finalized) {
        log_error("  shard %d: baseline not finalized", i);
        errors++;
        continue;
      }
      if (!m->finalized) {
        log_error("  shard %d: multiscale not finalized", i);
        errors++;
        continue;
      }
      if (b->size != m->size) {
        log_error("  shard %d: size mismatch (baseline=%zu, multiscale=%zu)",
                  i,
                  b->size,
                  m->size);
        errors++;
        continue;
      }
      if (memcmp(b->buf, m->buf, b->size) != 0) {
        // Find first difference
        size_t diff_off = 0;
        for (size_t j = 0; j < b->size; ++j) {
          if (b->buf[j] != m->buf[j]) {
            diff_off = j;
            break;
          }
        }
        log_error("  shard %d: data mismatch at byte %zu (of %zu)",
                  i,
                  diff_off,
                  b->size);
        errors++;
      }
    }

    if (errors > 0) {
      log_error("  %d shard comparison errors", errors);
      goto Fail2b;
    }
  }

  log_info("  PASS");
  test_sink_free(&ms_sink);
  test_sink_free(&baseline_sink);
  return 0;

Fail2b:
  test_sink_free(&ms_sink);
Fail1:
  test_sink_free(&baseline_sink);
  xor_pattern_free();
  log_error("  FAIL");
  return 1;
}

// --- Dim0 LOD correctness: L0 must match inner-only multiscale ---

static int
test_dim0_l0_correctness(void)
{
  log_info("=== test_dim0_l0_correctness ===");

  // 5D: t, z, y, x, c. Inner LOD on z, y, x.
  // 8 epochs along t so dim0 levels 1+ accumulate and emit.
  struct dimension dims_inner[] = {
    { .size = 8,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "t",
      .storage_position = 0 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .name = "c",
      .storage_position = 4 },
  };
  struct dimension dims_dim0[] = {
    { .size = 8,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "t",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .name = "c",
      .storage_position = 4 },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims_inner, rank);

  // Compute number of L0 shards
  int num_shards = 1;
  for (int d = 0; d < rank; ++d) {
    int chunk_count = (int)(dims_inner[d].size / dims_inner[d].chunk_size);
    int shard_count = (int)(chunk_count / dims_inner[d].chunks_per_shard);
    num_shards *= shard_count;
  }
  const size_t shard_cap = total_elements * sizeof(uint16_t) + 4096;
  log_info("  total: %zu elements, %d L0 shards", total_elements, num_shards);

  // --- Run 1: inner-only multiscale (baseline) ---
  struct test_shard_sink baseline_sink;
  test_sink_init(&baseline_sink, num_shards, shard_cap);

  {
    struct tile_stream_gpu* s = NULL;
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .dtype = dtype_u16,
      .rank = rank,
      .dimensions = dims_inner,
      .codec = CODEC_ZSTD,
      .reduce_method = lod_reduce_mean,
    };
    CHECK(Fail1, (s = tile_stream_gpu_create(&config, &baseline_sink.base)) != NULL);
    xor_pattern_init(dims_inner, rank, 2);
    CHECK(Fail1b,
          pump_data(tile_stream_gpu_writer(s), total_elements, fill_xor) == 0);
    CHECK(Fail1b, tile_stream_gpu_cursor(s) == total_elements);
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Run2d;
  Fail1b:
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Fail1;
  }

Run2d:
  // --- Run 2: inner + dim0 multiscale ---
  ;
  struct test_shard_sink dim0_sink;
  test_sink_init(&dim0_sink, num_shards, shard_cap);

  {
    struct tile_stream_gpu* s = NULL;
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .dtype = dtype_u16,
      .rank = rank,
      .dimensions = dims_dim0,
      .codec = CODEC_ZSTD,
      .reduce_method = lod_reduce_mean,
      .append_reduce_method = lod_reduce_mean,
    };
    CHECK(Fail2b, (s = tile_stream_gpu_create(&config, &dim0_sink.base)) != NULL);
    {
      struct tile_stream_status st = tile_stream_gpu_status(s);
      log_info("  dim0 enabled: nlod=%d, append_downsample=%d",
               st.nlod,
               st.append_downsample);
    }
    xor_pattern_init(dims_dim0, rank, 2);
    CHECK(Fail2c,
          pump_data(tile_stream_gpu_writer(s), total_elements, fill_xor) == 0);
    CHECK(Fail2c, tile_stream_gpu_cursor(s) == total_elements);
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Compared;
  Fail2c:
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    goto Fail2b;
  }

Compared:
  // --- Compare L0 shard output (must be identical) ---
  {
    int errors = 0;
    for (int i = 0; i < num_shards; ++i) {
      struct test_shard_writer* b = &baseline_sink.writers[0][i];
      struct test_shard_writer* m = &dim0_sink.writers[0][i];

      if (!b->finalized || !m->finalized) {
        log_error("  shard %d: not finalized (baseline=%d, dim0=%d)",
                  i,
                  b->finalized,
                  m->finalized);
        errors++;
        continue;
      }
      if (b->size != m->size) {
        log_error("  shard %d: size mismatch (baseline=%zu, dim0=%zu)",
                  i,
                  b->size,
                  m->size);
        errors++;
        continue;
      }
      if (memcmp(b->buf, m->buf, b->size) != 0) {
        size_t diff_off = 0;
        for (size_t j = 0; j < b->size; ++j) {
          if (b->buf[j] != m->buf[j]) {
            diff_off = j;
            break;
          }
        }
        log_error("  shard %d: data mismatch at byte %zu (of %zu)",
                  i,
                  diff_off,
                  b->size);
        errors++;
      }
    }

    if (errors > 0) {
      log_error("  %d shard comparison errors", errors);
      goto Fail2b;
    }
  }

  log_info("  PASS");
  test_sink_free(&dim0_sink);
  test_sink_free(&baseline_sink);
  return 0;

Fail2b:
  test_sink_free(&dim0_sink);
Fail1:
  test_sink_free(&baseline_sink);
  xor_pattern_free();
  log_error("  FAIL");
  return 1;
}

// --- Dim0 multi-epoch: verify higher LOD levels are populated ---

static int
test_dim0_multi_epoch_levels(void)
{
  log_info("=== test_dim0_multi_epoch_levels ===");

  // 5D: t, z, y, x, c. Dim0 + inner downsample.
  // 16 epochs along t to trigger multiple dim0 emissions.
  // Inner dims 32 with chunk 8 → 4 chunks → level 1 has 16 (>= chunk 8) →
  // nlod>=2
  struct dimension dims[] = {
    { .size = 32,
      .chunk_size = 2,
      .chunks_per_shard = 4,
      .name = "t",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "z",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 2 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 3 },
    { .size = 1,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .name = "c",
      .storage_position = 4 },
  };
  const uint8_t rank = 5;
  const size_t total_elements = dim_total_elements(dims, rank);

  // We need nlod — compute from lod_plan_init_from_dims
  struct lod_plan plan = { 0 };
  CHECK(Fail, lod_plan_init_from_dims(&plan, dims, rank, LOD_MAX_LEVELS) == 0);

  int nlod = plan.nlod;
  log_info("  nlod=%d total_elements=%zu", nlod, total_elements);
  CHECK(Fail, nlod >= 2); // need at least 2 levels for this test

  // Use generous shard allocation — higher levels may have different shard
  // layouts than what we can compute from just shapes. L0 gets a careful count,
  // higher levels get TEST_SHARD_SINK_MAX_SHARDS.
  int num_shards_per_level[TEST_SHARD_SINK_MAX_LEVELS];
  for (int lv = 0; lv < nlod; ++lv) {
    if (lv == 0) {
      int ns = 1;
      for (int d = 0; d < rank; ++d) {
        uint64_t tc = ceildiv(dims[d].size, dims[d].chunk_size);
        uint64_t tps = dims[d].chunks_per_shard;
        if (tps == 0)
          tps = tc;
        uint64_t sc = ceildiv(tc, tps);
        ns *= (int)sc;
      }
      num_shards_per_level[lv] =
        ns < TEST_SHARD_SINK_MAX_SHARDS ? ns : TEST_SHARD_SINK_MAX_SHARDS;
    } else {
      num_shards_per_level[lv] = TEST_SHARD_SINK_MAX_SHARDS;
    }
    log_info("  level %d: shape=(%lu,%lu,%lu,%lu,%lu) shards=%d",
             lv,
             (unsigned long)plan.shapes[lv][0],
             (unsigned long)plan.shapes[lv][1],
             (unsigned long)plan.shapes[lv][2],
             (unsigned long)plan.shapes[lv][3],
             (unsigned long)plan.shapes[lv][4],
             num_shards_per_level[lv]);
  }

  lod_plan_free(&plan);

  struct test_shard_sink sink;
  test_sink_init_multi(&sink, nlod, num_shards_per_level, 256 * 1024);

  {
    struct tile_stream_gpu* s = NULL;
    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4 << 20,
      .dtype = dtype_u16,
      .rank = rank,
      .dimensions = dims,
      .codec = CODEC_ZSTD,
      .reduce_method = lod_reduce_mean,
      .append_reduce_method = lod_reduce_mean,
    };

    CHECK(Fail2, (s = tile_stream_gpu_create(&config, &sink.base)) != NULL);
    {
      struct tile_stream_status st = tile_stream_gpu_status(s);
      log_info("  stream nlod=%d append_downsample=%d epochs_per_batch=%u",
               st.nlod,
               st.append_downsample,
               st.epochs_per_batch);
    }

    xor_pattern_init(dims, rank, 2);
    int pump_ok =
      (pump_data(tile_stream_gpu_writer(s), total_elements, fill_xor) == 0);
    int cursor_ok = pump_ok && (tile_stream_gpu_cursor(s) == total_elements);
    tile_stream_gpu_destroy(s);
    xor_pattern_free();
    CHECK(Fail2, pump_ok);
    CHECK(Fail2, cursor_ok);
  }

  // Verify: L0 shards should be finalized and non-empty
  {
    int l0_finalized = 0;
    for (int i = 0; i < num_shards_per_level[0]; ++i) {
      if (sink.writers[0][i].finalized && sink.writers[0][i].size > 0)
        l0_finalized++;
    }
    log_info(
      "  L0: %d/%d shards finalized", l0_finalized, num_shards_per_level[0]);
    CHECK(Fail2, l0_finalized > 0);
  }

  // Verify: level 1+ shards should have at least some finalized non-empty
  for (int lv = 1; lv < nlod; ++lv) {
    int finalized = 0;
    size_t total_bytes = 0;
    for (int i = 0; i < num_shards_per_level[lv]; ++i) {
      if (sink.writers[lv][i].finalized) {
        finalized++;
        total_bytes += sink.writers[lv][i].size;
      }
    }
    log_info("  L%d: %d/%d shards finalized, %zu total bytes",
             lv,
             finalized,
             num_shards_per_level[lv],
             total_bytes);
    CHECK(Fail2, finalized > 0);
    CHECK(Fail2, total_bytes > 0);
  }

  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail2:
  test_sink_free(&sink);
Fail:
  log_error("  FAIL");
  return 1;
}

RUN_GPU_TESTS({ "multiscale_l0_correctness", test_multiscale_l0_correctness },
              { "dim0_l0_correctness", test_dim0_l0_correctness },
              { "dim0_multi_epoch_levels", test_dim0_multi_epoch_levels }, )

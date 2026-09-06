#include "defs.limits.h"
#include "stream.cpu.h"
#include "stream/layouts.h"
#include "test_shard_sink.h"
#include "util/prelude.h"
#include "zarr/crc32c.h"

#include <stdlib.h>
#include <string.h>

#define SHARD_CAP (1 << 20)

// ---- Test: write 2 epochs of u16 data, verify shards are non-empty ----

static int
test_basic_pipeline(void)
{
  log_info("=== test_stream_cpu_basic ===");

  struct test_shard_sink sink;
  test_sink_init(&sink, 16, SHARD_CAP);

  // 3D: 4×4×6, chunk 2×2×3, chunks_per_shard = 1×2×2
  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);
  log_info("  epoch_elements=%lu chunks_per_epoch=%lu",
           (unsigned long)lay->epoch_elements,
           (unsigned long)lay->chunks_per_epoch);

  // Write 2 epochs of sequential u16 data.
  uint64_t total_elements = 2 * lay->epoch_elements;
  size_t total_bytes = total_elements * sizeof(uint16_t);
  uint16_t* data = (uint16_t*)malloc(total_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);

  struct writer* w = tile_stream_cpu_writer(s);
  struct slice sl = { .beg = data, .end = (const char*)data + total_bytes };
  struct writer_result r = writer_append(w, sl);
  CHECK(Fail, r.error == 0);

  r = writer_flush(w);
  CHECK(Fail, r.error == 0);

  CHECK(Fail, tile_stream_cpu_cursor(s) == total_elements);

  // Verify at least one shard was written.
  int found = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    if (sink.writers[0][i].buf && sink.writers[0][i].size > 0) {
      found = 1;
      log_info(
        "  shard %d: %lu bytes", i, (unsigned long)sink.writers[0][i].size);
    }
  }
  CHECK(Fail, found);

  // Metrics sanity.
  struct stream_metrics m = tile_stream_cpu_get_metrics(s);
  CHECK(Fail, m.scatter.count > 0);
  CHECK(Fail, m.compress.count > 0);
  CHECK(Fail, m.sink.count > 0);

  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// Test f16 rejection.
static int
test_f16_rejected(void)
{
  log_info("=== test_stream_cpu_f16 ===");

  struct test_shard_sink sink;
  test_sink_init(&sink, 16, SHARD_CAP);

  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 1 },
  };
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_f16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s == NULL); // should be rejected

  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// Flush finalizes: it writes out what was appended and then takes no more
// input, and calling it again does nothing.
static int
test_no_append_after_flush(void)
{
  log_info("=== test_stream_cpu_no_append_after_flush ===");

  struct test_shard_sink sink;
  test_sink_init(&sink, 16, SHARD_CAP);

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  uint16_t* data = NULL;
  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);
  uint64_t epoch_elems = lay->epoch_elements;
  size_t epoch_bytes = epoch_elems * sizeof(uint16_t);
  data = (uint16_t*)malloc(epoch_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < epoch_elems; ++i)
    data[i] = (uint16_t)i;

  struct writer* w = tile_stream_cpu_writer(s);

  // First batch: append + flush.
  {
    struct slice sl = { .beg = data, .end = (const char*)data + epoch_bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }

  int shards_after_first_flush = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    if (sink.writers[0][i].buf && sink.writers[0][i].size > 0)
      shards_after_first_flush++;
  }
  CHECK(Fail, shards_after_first_flush > 0);

  // The flush finalized the stream, so it takes no more input and consumes
  // none of what it is offered.
  {
    struct slice sl = { .beg = data, .end = (const char*)data + epoch_bytes };
    struct writer_result r = writer_append(w, sl);
    CHECK(Fail, r.error == writer_error_finished);
    CHECK(Fail, r.rest.beg == sl.beg && r.rest.end == sl.end);
  }
  CHECK(Fail, tile_stream_cpu_cursor(s) == epoch_elems);

  // Finalizing again is a no-op rather than a second close-out.
  {
    struct writer_result r = writer_flush(w);
    CHECK(Fail, r.error == 0);
  }
  int shards_after_second_flush = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    if (sink.writers[0][i].buf && sink.writers[0][i].size > 0)
      shards_after_second_flush++;
  }
  CHECK(Fail, shards_after_second_flush == shards_after_first_flush);

  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// --- tile_stream_cpu_advise_layout tests ---

static int
test_advise_basic_fit(void)
{
  log_info("=== test_advise_basic_fit ===");
  struct dimension dims[3];
  uint64_t sizes[] = { 256, 128, 128 };
  dims_create(dims, "tyx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .target_batch_bytes = 8u << 20,
  };

  int ratios[] = { 1, 1, 1 };
  struct advise_layout_diagnostic diag = { 0 };
  CHECK(
    Fail,
    tile_stream_cpu_advise_layout(
      &config, 1 << 15, 1024, ratios, 1ull << 30, 1 << 20, 1, 0, 0, &diag) ==
      0);
  CHECK(Fail, diag.reason == ADVISE_OK);
  CHECK(Fail, config.epochs_per_batch >= 1);
  for (int d = 0; d < 3; ++d) {
    CHECK(Fail, dims[d].chunk_size >= 1);
    CHECK(Fail, dims[d].chunks_per_shard >= 1);
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_blosc_requires_block_size(void)
{
  struct dimension dims[2];
  dims_create(dims, "ty", (uint64_t[]){ 4, 64 });
  dims_set_chunk_sizes(dims, 2, (uint64_t[]){ 1, 64 });
  dims_set_shard_counts(dims, 2, (uint64_t[]){ 1, 1 });
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 1,
  };
  struct tile_stream_cpu_memory_info info;
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 0, &info) == 0);
  config.codec.id = CODEC_BLOSC_ZSTD;
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 0, &info) != 0);
  config.codec.level = 5;
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 0, &info) != 0);
  config.codec.id = CODEC_BLOSC_LZ4;
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 0, &info) != 0);
  return 0;
Fail:
  return 1;
}

static int
test_advise_invalid_config(void)
{
  log_info("=== test_advise_invalid_config ===");
  struct dimension dims[2];
  uint64_t sizes[] = { 100, 64 };
  dims_create(dims, "yx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  int ratios[] = { 1, 1 };
  struct advise_layout_diagnostic diag = { 0 };

  // budget=0 -> INVALID_CONFIG.
  CHECK(Fail,
        tile_stream_cpu_advise_layout(
          &config, 1 << 14, 1024, ratios, 0, 1 << 20, 1, 0, 0, &diag) != 0);
  CHECK(Fail, diag.reason == ADVISE_INVALID_CONFIG);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_advise_chunk_budget_infeasible(void)
{
  log_info("=== test_advise_chunk_budget_infeasible ===");
  // Pinned dims whose element product exceeds the target budget → solver bails
  // immediately with ADVISE_CHUNK_BUDGET_INFEASIBLE (halving target makes it
  // worse, not better).
  struct dimension dims[3];
  uint64_t sizes[] = { 100, 512, 512 };
  dims_create(dims, "tyx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  // Pinned y*x = 262144 elements * 2 bytes = 512 KiB. Target = 64 KiB → pinned
  // dims alone exceed the budget.
  int ratios[] = { 1, -1, -1 };
  struct advise_layout_diagnostic diag = { 0 };

  CHECK(Fail,
        tile_stream_cpu_advise_layout(
          &config, 1 << 16, 1 << 16, ratios, 1ull << 30, 0, 0, 0, 0, &diag) !=
          0);
  CHECK(Fail, diag.reason == ADVISE_CHUNK_BUDGET_INFEASIBLE);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_advise_min_append_shards_overrides_floor(void)
{
  log_info("=== test_advise_min_append_shards_overrides_floor ===");
  // Both min_append_shards and min_shard_bytes set → diagnostic flag is raised.
  struct dimension dims[3];
  uint64_t sizes[] = { 100, 64, 64 };
  dims_create(dims, "tyx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  int ratios[] = { 0, 1, 1 };
  struct advise_layout_diagnostic diag = { 0 };

  CHECK(
    Fail,
    tile_stream_cpu_advise_layout(
      &config, 1 << 15, 1024, ratios, 1ull << 30, 1 << 20, 1, 4, 0, &diag) ==
      0);
  CHECK(Fail, diag.reason == ADVISE_OK);
  CHECK(Fail, diag.min_append_shards_overrode_min_shard_bytes == 1);

  // Sanity: min_append_shards not set → flag stays 0.
  struct advise_layout_diagnostic diag2 = { 0 };
  CHECK(
    Fail,
    tile_stream_cpu_advise_layout(
      &config, 1 << 15, 1024, ratios, 1ull << 30, 1 << 20, 1, 0, 0, &diag2) ==
      0);
  CHECK(Fail, diag2.min_append_shards_overrode_min_shard_bytes == 0);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_advise_min_shard_too_small(void)
{
  log_info("=== test_advise_min_shard_too_small ===");
  struct dimension dims[3];
  uint64_t sizes[] = { 100, 64, 64 };
  dims_create(dims, "tyx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .target_batch_bytes = 8u << 20,
  };

  int ratios[] = { 1, 1, 1 };
  struct advise_layout_diagnostic diag = { 0 };

  // target=1 MiB chunks but min_shard=512 B < chunk_bytes ->
  // MIN_SHARD_TOO_SMALL
  CHECK(Fail,
        tile_stream_cpu_advise_layout(
          &config, 1 << 20, 1 << 20, ratios, 1ull << 30, 512, 1, 0, 0, &diag) !=
          0);
  CHECK(Fail, diag.reason == ADVISE_MIN_SHARD_TOO_SMALL);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_advise_parts_limit(void)
{
  log_info("=== test_advise_parts_limit ===");
  // Same config as the old smallepoch-style case that used to fail with
  // PARTS_LIMIT_EXCEEDED. Phase B in dims_set_shard_geometry now splits inner
  // dims past target_concurrent_shards to keep chunks_per_shard_total within
  // MAX_PARTS_PER_SHARD. Solver should succeed; min_shard_bytes may be met
  // less than fully (it's soft).
  struct dimension dims[3];
  uint64_t sizes[] = { 1 << 20, 16, 16 };
  dims_create(dims, "tyx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .target_batch_bytes = 8u << 20,
  };

  int ratios[] = { 1, 0, 0 };
  struct advise_layout_diagnostic diag = { 0 };

  CHECK(Fail,
        tile_stream_cpu_advise_layout(&config,
                                      1 << 16,
                                      1 << 16,
                                      ratios,
                                      1ull << 30,
                                      1ull << 30,
                                      1,
                                      0,
                                      0,
                                      &diag) == 0);
  uint64_t cps_total = 1;
  for (uint8_t d = 0; d < config.rank; ++d) {
    uint64_t cps = config.dimensions[d].chunks_per_shard;
    if (cps == 0)
      cps = 1;
    cps_total *= cps;
  }
  CHECK(Fail, cps_total <= MAX_PARTS_PER_SHARD);

  // Diagnostic populated on success.
  CHECK(Fail, diag.reason == ADVISE_OK);
  CHECK(Fail, diag.actual_concurrent_shards >= 1);
  CHECK(Fail, diag.actual_shard_bytes == diag.chunk_bytes * cps_total);
  // The bounded dim 0 (1M rows) caps cps_append below cps_floor — actual shard
  // is smaller than the 1 GiB min_shard_bytes floor (soft loss).
  CHECK(Fail, diag.actual_shard_bytes < (1ull << 30));

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_advise_halves_k(void)
{
  log_info("=== test_advise_halves_k ===");
  // Pin min_chunk_bytes == target_chunk_bytes so the outer chunk-size loop
  // has exactly one iteration; K halving is the only adaptation path. Under
  // a tight budget, advise must choose a K strictly below the auto K.
  struct dimension dims[3];
  uint64_t sizes[] = { 1024, 64, 64 };
  dims_create(dims, "tyx", sizes);

  const size_t target = 1 << 16; // 64 KiB chunk
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .target_batch_bytes = 8u << 20,
  };

  int ratios[] = { 1, 1, 1 };
  struct advise_layout_diagnostic diag = { 0 };

  // Step 1: huge budget -> auto K. Preserves the advised chunk geometry.
  CHECK(
    Fail,
    tile_stream_cpu_advise_layout(
      &config, target, target, ratios, 1ull << 40, 1 << 20, 1, 0, 0, &diag) ==
      0);
  const uint32_t auto_k = config.epochs_per_batch;
  CHECK(Fail, auto_k > 1);

  // Step 2: probe heap for the just-advised geometry at auto K.
  struct tile_stream_cpu_memory_info mi;
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 0, &mi) == 0);
  const size_t heap_at_auto_k = mi.heap_bytes;

  // Step 3: reset dims and advise again with a tight budget. With
  // min_chunk==target, the outer loop runs only once — K halving is the only
  // way to fit.
  dims_create(dims, "tyx", sizes);
  config.epochs_per_batch = 0;
  const size_t tight_budget = heap_at_auto_k / 4;
  CHECK(
    Fail,
    tile_stream_cpu_advise_layout(
      &config, target, target, ratios, tight_budget, 1 << 20, 1, 0, 0, &diag) ==
      0);
  CHECK(Fail, diag.reason == ADVISE_OK);
  CHECK(Fail, config.epochs_per_batch < auto_k);
  CHECK(Fail, config.epochs_per_batch >= 1);

  // Verify the chosen configuration actually fits within the tight budget.
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 0, &mi) == 0);
  CHECK(Fail, mi.heap_bytes <= tight_budget);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_advise_user_k_respected(void)
{
  log_info("=== test_advise_user_k_respected ===");
  // Non-zero config.epochs_per_batch on entry is authoritative and isn't
  // reduced by advise even when the budget is tight.
  struct dimension dims[3];
  uint64_t sizes[] = { 256, 128, 128 };
  dims_create(dims, "tyx", sizes);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .target_batch_bytes = 16u << 20,
    .epochs_per_batch = 4, // user-pinned
  };

  int ratios[] = { 1, 1, 1 };
  struct advise_layout_diagnostic diag = { 0 };
  CHECK(
    Fail,
    tile_stream_cpu_advise_layout(
      &config, 1 << 15, 1024, ratios, 1ull << 30, 1 << 20, 1, 0, 0, &diag) ==
      0);
  CHECK(Fail, config.epochs_per_batch == 4);
  CHECK(Fail, diag.reason == ADVISE_OK);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_host_output_memory_estimate(void)
{
  log_info("=== test_host_output_memory_estimate ===");
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 2,
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .storage_position = 2 },
  };
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 2,
  };

  struct tile_stream_cpu_memory_info info;
  CHECK(Fail, tile_stream_cpu_memory_estimate(&config, 4096, &info) == 0);
  CHECK(Fail, info.host_output_bytes > 0);
  CHECK(Fail, info.host_output_pool_bytes == 2 * info.host_output_bytes);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_basic_pipeline();
  rc |= test_f16_rejected();
  rc |= test_no_append_after_flush();
  rc |= test_advise_basic_fit();
  rc |= test_advise_invalid_config();
  rc |= test_blosc_requires_block_size();
  rc |= test_advise_chunk_budget_infeasible();
  rc |= test_advise_min_append_shards_overrides_floor();
  rc |= test_advise_min_shard_too_small();
  rc |= test_advise_parts_limit();
  rc |= test_advise_halves_k();
  rc |= test_advise_user_k_respected();
  rc |= test_host_output_memory_estimate();
  return rc;
}

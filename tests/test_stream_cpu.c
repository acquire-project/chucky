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
    .codec = CODEC_NONE,
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
    .codec = CODEC_NONE,
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

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_basic_pipeline();
  rc |= test_f16_rejected();
  return rc;
}

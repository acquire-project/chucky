// End-to-end CPU stream test with n_append=2.
// Verifies that a 4D array "tzyx" with chunk (1,1,64,64) streams correctly
// with two append dimensions (t unbounded, z bounded).

#include "stream/dim_info.h"
#include "stream/layouts.h"
#include "stream.cpu.h"
#include "test_shard_sink.h"
#include "defs.limits.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define SHARD_CAP (1 << 20)

// --- Test: basic n_append=2 pipeline ---

static int
test_two_append_basic(void)
{
  log_info("=== test_two_append_basic ===");

  struct tile_stream_cpu* s = NULL;
  struct test_shard_sink sink;
  test_sink_init(&sink, 16, SHARD_CAP);

  // 4D: t=unbounded, z=4, y=64, x=64
  // chunk: (1, 1, 32, 32)
  // n_append should be 2 (t and z both have chunk_size=1)
  struct dimension dims[4];
  uint64_t sizes[] = { 0, 4, 64, 64 };
  uint8_t rank = dims_create(dims, "tzyx", sizes);
  CHECK(Fail, rank == 4);

  uint64_t cs[] = { 1, 1, 32, 32 };
  dims_set_chunk_sizes(dims, rank, cs);
  dims[0].chunks_per_shard = 2;
  dims[1].chunks_per_shard = 4;
  dims[2].chunks_per_shard = 2;
  dims[3].chunks_per_shard = 2;

  // Verify dim_info partition
  struct dim_info info;
  CHECK(Fail, dim_info_init(&info, dims, rank) == 0);
  CHECK(Fail, dim_info_n_append(&info) == 2);
  CHECK(Fail, info.append_downsample == 0);
  CHECK(Fail, info.lod_mask == 0); // no downsample
  // inner_append_count = chunk_count for z = ceil(4/1) = 4
  CHECK(Fail, info.inner_append_count == 4);

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 64 * 1024,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);

  // chunks_per_epoch = chunk_count_y * chunk_count_x = 2 * 2 = 4
  CHECK(Fail, lay->chunks_per_epoch == 4);
  // epoch_elements = chunks_per_epoch * chunk_elements
  // chunk_elements = 1 * 1 * 32 * 32 = 1024
  CHECK(Fail, lay->chunk_elements == 1024);
  CHECK(Fail, lay->epoch_elements == 4096);

  // lifted_strides: append dims (0,1) should be collapsed (stride=0)
  CHECK(Fail, lay->lifted_strides[0] == 0); // t chunk stride
  CHECK(Fail, lay->lifted_strides[2] == 0); // z chunk stride

  log_info("  epoch_elements=%lu chunks_per_epoch=%lu",
           (unsigned long)lay->epoch_elements,
           (unsigned long)lay->chunks_per_epoch);

  // Write 2 t-frames x 4 z-slices = 8 epochs of sequential u16 data.
  uint64_t n_epochs = 2 * 4; // 2 t * 4 z
  uint64_t total_elements = n_epochs * lay->epoch_elements;
  size_t total_bytes = total_elements * sizeof(uint16_t);
  uint16_t* data = (uint16_t*)malloc(total_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);

  struct writer* w = tile_stream_cpu_writer(s);
  struct slice sl = { .beg = data, .end = (const char*)data + total_bytes };
  struct writer_result r = writer_append(w, sl);
  CHECK(Fail_data, r.error == 0);

  r = writer_flush(w);
  CHECK(Fail_data, r.error == 0);

  CHECK(Fail_data, tile_stream_cpu_cursor(s) == total_elements);

  // Verify at least one shard was written.
  int found = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    if (sink.writers[0][i].buf && sink.writers[0][i].size > 0) {
      found = 1;
      log_info("  shard %d: %lu bytes", i,
               (unsigned long)sink.writers[0][i].size);
    }
  }
  CHECK(Fail_data, found);

  // Metrics sanity.
  struct stream_metrics m = tile_stream_cpu_get_metrics(s);
  CHECK(Fail_data, m.scatter.count > 0);
  CHECK(Fail_data, m.compress.count > 0);
  CHECK(Fail_data, m.sink.count > 0);

  log_info("  PASS");

  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  return 0;

Fail_data:
  free(data);
Fail:
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// --- Test: n_append=2 with bounded dim0 ---

static int
test_two_append_bounded(void)
{
  log_info("=== test_two_append_bounded ===");
  struct tile_stream_cpu* s = NULL;

  struct test_shard_sink sink;
  test_sink_init(&sink, 16, SHARD_CAP);

  // All dims bounded: t=6, z=4, y=64, x=64
  struct dimension dims[4];
  uint64_t sizes[] = { 6, 4, 64, 64 };
  uint8_t rank = dims_create(dims, "tzyx", sizes);

  uint64_t cs[] = { 1, 1, 32, 32 };
  dims_set_chunk_sizes(dims, rank, cs);
  dims[0].chunks_per_shard = 6;
  dims[1].chunks_per_shard = 4;
  dims[2].chunks_per_shard = 2;
  dims[3].chunks_per_shard = 2;

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 64 * 1024,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);

  // Write exactly all data: 6 * 4 = 24 epochs
  uint64_t n_epochs = 6 * 4;
  uint64_t total_elements = n_epochs * lay->epoch_elements;
  size_t total_bytes = total_elements * sizeof(uint16_t);
  uint16_t* data = (uint16_t*)malloc(total_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);

  struct writer* w = tile_stream_cpu_writer(s);
  struct slice sl = { .beg = data, .end = (const char*)data + total_bytes };
  struct writer_result r = writer_append(w, sl);
  CHECK(Fail_data, r.error == 0);

  r = writer_flush(w);
  CHECK(Fail_data, r.error == 0);

  CHECK(Fail_data, tile_stream_cpu_cursor(s) == total_elements);

  // Verify shards written.
  int found = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    if (sink.writers[0][i].buf && sink.writers[0][i].size > 0)
      found++;
  }
  CHECK(Fail_data, found > 0);
  log_info("  %d shards written", found);

  // Write one more element — should get writer_error_finished.
  {
    uint16_t extra = 42;
    struct slice extra_sl = { .beg = &extra, .end = &extra + 1 };
    struct writer_result r2 = writer_append(w, extra_sl);
    CHECK(Fail_data, r2.error == writer_error_finished);
  }

  log_info("  PASS");

  free(data);
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  return 0;

Fail_data:
  free(data);
Fail:
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// --- Test: verify layout geometry for n_append=2 ---

static int
test_two_append_layout_geometry(void)
{
  log_info("=== test_two_append_layout_geometry ===");
  struct tile_stream_cpu* s = NULL;

  struct test_shard_sink sink;
  test_sink_init(&sink, 64, SHARD_CAP);

  // 4D: t=0(unbounded), z=5, y=128, x=96, chunk (1,1,64,32)
  struct dimension dims[4];
  uint64_t sizes[] = { 0, 5, 128, 96 };
  uint8_t rank = dims_create(dims, "tzyx", sizes);
  uint64_t cs[] = { 1, 1, 64, 32 };
  dims_set_chunk_sizes(dims, rank, cs);
  dims[0].chunks_per_shard = 4;
  dims[1].chunks_per_shard = 5;

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 64 * 1024,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);

  // chunk_count_y = ceil(128/64) = 2
  // chunk_count_x = ceil(96/32) = 3
  // chunks_per_epoch = 2 * 3 = 6
  CHECK(Fail, lay->chunks_per_epoch == 6);

  // chunk_elements = 1 * 1 * 64 * 32 = 2048
  CHECK(Fail, lay->chunk_elements == 2048);

  // epoch_elements = 6 * 2048 = 12288
  CHECK(Fail, lay->epoch_elements == 12288);

  // Both append dims collapsed
  CHECK(Fail, lay->lifted_strides[0] == 0);
  CHECK(Fail, lay->lifted_strides[2] == 0);

  // Inner dim strides should be non-zero
  CHECK(Fail, lay->lifted_strides[4] != 0); // y chunk stride
  CHECK(Fail, lay->lifted_strides[6] != 0); // x chunk stride

  log_info("  PASS");

  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  return 0;

Fail:
  tile_stream_cpu_destroy(s);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  int rc = 0;
  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "two_append_basic", test_two_append_basic },
    { "two_append_bounded", test_two_append_bounded },
    { "two_append_layout_geometry", test_two_append_layout_geometry },
  };
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    int r = tests[i].fn();
    if (r)
      rc = 1;
  }
  return rc;
}

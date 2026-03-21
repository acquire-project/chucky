#include "crc32c.h"
#include "index.ops.util.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_gpu_helpers.h"
#include "test_runner.h"
#include "test_shard_sink.h"
#include "test_shard_verify.h"
#include "test_voxel_encode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- chunk pool verification helpers ---

// Build expected chunk pool for one epoch.
static uint16_t*
make_expected_tiles(uint64_t epoch_start,
                    uint64_t epoch_elements,
                    uint64_t chunks_per_epoch,
                    uint64_t chunk_elements,
                    uint8_t lifted_rank,
                    const uint64_t* lifted_shape,
                    const int64_t* lifted_strides)
{
  uint64_t pool_size = chunks_per_epoch * chunk_elements;
  uint16_t* expected = (uint16_t*)calloc(pool_size, sizeof(uint16_t));
  if (!expected)
    return NULL;

  for (uint64_t i = 0; i < epoch_elements; ++i) {
    uint64_t idx = epoch_start + i;
    uint64_t off = ravel(lifted_rank, lifted_shape, lifted_strides, idx);
    expected[off] = (uint16_t)(idx % 65536);
  }
  return expected;
}

// Verify chunk data against expected chunks for all epochs.
// If use_zstd, decompress each chunk before comparing.
// Returns 0 on success, 1 on failure.
static int
verify_tiles(const struct tile_stream_gpu* s,
             const uint8_t* shard_buf,
             const uint64_t* chunk_offsets,
             const uint64_t* chunk_sizes,
             int use_zstd)
{
  const struct tile_stream_layout* lay = tile_stream_gpu_layout(s);
  const size_t chunk_bytes = lay->chunk_stride * sizeof(uint16_t);
  int n_epochs = 2;

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    uint16_t* expected =
      make_expected_tiles((uint64_t)epoch * lay->epoch_elements,
                          lay->epoch_elements,
                          lay->chunks_per_epoch,
                          lay->chunk_elements,
                          lay->lifted_rank,
                          lay->lifted_shape,
                          lay->lifted_strides);
    if (!expected)
      return 1;

    int err = 0;
    for (uint64_t t = 0; t < lay->chunks_per_epoch; ++t) {
      size_t slot = (size_t)epoch * lay->chunks_per_epoch + t;
      const uint16_t* chunk_data = NULL;
      uint8_t* decomp = NULL;

      if (use_zstd) {
        if (chunk_sizes[slot] == 0) {
          err = 1;
          break;
        }
        decomp = (uint8_t*)calloc(1, chunk_bytes);
        if (!decomp) {
          err = 1;
          break;
        }
        size_t result = ZSTD_decompress(decomp,
                                        chunk_bytes,
                                        shard_buf + chunk_offsets[slot],
                                        chunk_sizes[slot]);
        if (ZSTD_isError(result) || result != chunk_bytes) {
          log_error("  ZSTD_decompress failed for chunk %lu epoch %d",
                    (unsigned long)t,
                    epoch);
          free(decomp);
          err = 1;
          break;
        }
        chunk_data = (const uint16_t*)decomp;
      } else {
        if (chunk_sizes[slot] != chunk_bytes) {
          err = 1;
          break;
        }
        chunk_data = (const uint16_t*)(shard_buf + chunk_offsets[slot]);
      }

      const uint16_t* expected_tile = expected + t * lay->chunk_elements;
      for (uint64_t e = 0; e < lay->chunk_elements; ++e) {
        if (chunk_data[e] != expected_tile[e]) {
          log_error("  epoch %d chunk %lu elem %lu: expected %u, got %u",
                    epoch,
                    (unsigned long)t,
                    (unsigned long)e,
                    expected_tile[e],
                    chunk_data[e]);
          err = 1;
        }
      }
      free(decomp);
    }
    free(expected);
    if (err) {
      log_error("  FAIL: epoch %d verification", epoch);
      return 1;
    }
    log_info("  epoch %d: OK", epoch);
  }
  return 0;
}

// Test: feed all data in one append call.
// Shape (4,4,6), chunk (2,2,3) -> 2 epochs, 4 chunks/epoch, 12 elements/chunk.
// Total 96 elements. Uses CODEC_NONE shard path.
static int
test_stream_single_append(void)
{
  log_info("=== test_stream_single_append ===");

  struct dimension dims[3];
  make_test_dims_3d(dims);

  // chunk_count = (2, 2, 2), chunks_per_shard = (2, 2, 2), total = 8.
  const size_t chunks_per_shard_total = 8;

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // Verify computed layout
  log_info("  chunk_elements=%lu  chunks_per_epoch=%lu  epoch_elements=%lu",
           (unsigned long)tile_stream_gpu_layout(s)->chunk_elements,
           (unsigned long)tile_stream_gpu_layout(s)->chunks_per_epoch,
           (unsigned long)tile_stream_gpu_layout(s)->epoch_elements);
  CHECK(Fail, tile_stream_gpu_layout(s)->chunk_elements == 12);
  CHECK(Fail, tile_stream_gpu_layout(s)->chunks_per_epoch == 4);
  CHECK(Fail, tile_stream_gpu_layout(s)->epoch_elements == 48);

  {
    printf("  lifted_shape: ");
    println_vu64(tile_stream_gpu_layout(s)->lifted_rank,
                 tile_stream_gpu_layout(s)->lifted_shape);
    printf("  lifted_strides: ");
    println_vi64(tile_stream_gpu_layout(s)->lifted_rank,
                 tile_stream_gpu_layout(s)->lifted_strides);
  }

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  // Append all data
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail, r.error == 0);

  // Flush to get all data
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail, r.error == 0);
  CHECK(Fail, mss.writers[0][0].size > 0);

  {
    uint64_t chunk_offsets[8], chunk_sizes[8];
    CHECK(Fail,
          shard_index_parse(mss.writers[0][0].buf,
                            mss.writers[0][0].size,
                            chunks_per_shard_total,
                            chunk_offsets,
                            chunk_sizes) == 0);
    CHECK(Fail,
          verify_tiles(
            s, mss.writers[0][0].buf, chunk_offsets, chunk_sizes, 0) == 0);
  }

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: feed data in small pieces (e.g., 7 elements at a time)
// to exercise buffer-fill + dispatch + epoch-crossing logic.
// Uses CODEC_NONE shard path.
static int
test_stream_incremental_append(void)
{
  log_info("=== test_stream_incremental_append ===");

  struct dimension dims[3];
  make_test_dims_3d(dims);

  const size_t chunks_per_shard_total = 8;

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  // Small buffer: 10 elements worth (rounded up to 4KB internally)
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 10 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  const int total = 96;
  uint16_t src[96];
  for (int i = 0; i < total; ++i)
    src[i] = (uint16_t)i;

  // Feed in pieces of 7 elements
  const int step_size = 7;

  for (int off = 0; off < total; off += step_size) {
    int n = step_size;
    if (off + n > total)
      n = total - off;

    struct slice input = { .beg = src + off, .end = src + off + n };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail, r.error == 0);
  }

  // Flush remaining data
  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    CHECK(Fail, r.error == 0);
  }

  CHECK(Fail, mss.writers[0][0].size > 0);

  {
    uint64_t chunk_offsets[8], chunk_sizes[8];
    CHECK(Fail,
          shard_index_parse(mss.writers[0][0].buf,
                            mss.writers[0][0].size,
                            chunks_per_shard_total,
                            chunk_offsets,
                            chunk_sizes) == 0);
    CHECK(Fail,
          verify_tiles(
            s, mss.writers[0][0].buf, chunk_offsets, chunk_sizes, 0) == 0);
  }

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: compressed roundtrip via shard path — compress chunks with nvcomp,
// collect shard data, parse index, decompress with libzstd, verify contents
// match expected chunk pool.
static int
test_stream_compressed_roundtrip(void)
{
  log_info("=== test_stream_compressed_roundtrip ===");

  struct dimension dims[3];
  make_test_dims_3d(dims);

  const size_t chunks_per_shard_total = 8;

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  log_info("  chunk_elements=%lu  chunk_stride=%lu  chunks_per_epoch=%lu  "
           "epoch_elements=%lu",
           (unsigned long)tile_stream_gpu_layout(s)->chunk_elements,
           (unsigned long)tile_stream_gpu_layout(s)->chunk_stride,
           (unsigned long)tile_stream_gpu_layout(s)->chunks_per_epoch,
           (unsigned long)tile_stream_gpu_layout(s)->epoch_elements);
  log_info("  max_output_size=%zu  chunk_pool_bytes=%zu",
           tile_stream_gpu_status(s).max_compressed_size,
           tile_stream_gpu_layout(s)->chunk_pool_bytes);

  CHECK(Fail, tile_stream_gpu_layout(s)->chunk_elements == 12);
  CHECK(Fail, tile_stream_gpu_layout(s)->chunks_per_epoch == 4);
  CHECK(Fail, tile_stream_gpu_layout(s)->epoch_elements == 48);

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  // Append all data
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail, r.error == 0);
  CHECK(Fail, mss.writers[0][0].size > 0);

  {
    uint64_t chunk_offsets[8], chunk_sizes[8];
    CHECK(Fail,
          shard_index_parse(mss.writers[0][0].buf,
                            mss.writers[0][0].size,
                            chunks_per_shard_total,
                            chunk_offsets,
                            chunk_sizes) == 0);
    CHECK(Fail,
          verify_tiles(
            s, mss.writers[0][0].buf, chunk_offsets, chunk_sizes, 1) == 0);
  }

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: LZ4 compressed stream — verify shard structural integrity.
// No CPU LZ4 decompression, so we check structural properties only:
// shard size > 0, all chunk sizes > 0 and ≤ max_output_size, valid offsets.
static int
test_stream_lz4_roundtrip(void)
{
  log_info("=== test_stream_lz4_roundtrip ===");

  struct dimension dims[3];
  make_test_dims_3d(dims);

  const size_t chunks_per_shard_total = 8;

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_LZ4,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail, r.error == 0);

  const size_t shard_size = mss.writers[0][0].size;
  CHECK(Fail, shard_size > 0);
  log_info("  shard_size=%zu", shard_size);

  // Parse shard index
  uint64_t chunk_offsets[8], chunk_sizes[8];
  CHECK(Fail,
        shard_index_parse(mss.writers[0][0].buf,
                          shard_size,
                          chunks_per_shard_total,
                          chunk_offsets,
                          chunk_sizes) == 0);

  // Verify structural properties
  const size_t index_data_bytes = chunks_per_shard_total * 2 * sizeof(uint64_t);
  size_t chunk_data_total = 0;
  for (size_t i = 0; i < chunks_per_shard_total; ++i) {
    CHECK(Fail, chunk_sizes[i] > 0);
    CHECK(Fail,
          chunk_sizes[i] <= tile_stream_gpu_status(s).max_compressed_size);
    CHECK(Fail, chunk_offsets[i] + chunk_sizes[i] <= shard_size);
    chunk_data_total += chunk_sizes[i];
    log_info("  chunk %zu: offset=%lu size=%lu",
             i,
             (unsigned long)chunk_offsets[i],
             (unsigned long)chunk_sizes[i]);
  }

  // Total chunk data + index block + CRC should equal shard size
  CHECK(Fail, chunk_data_total + index_data_bytes + 4 == shard_size);
  log_info("  chunk_data_total=%zu  expected_shard_size=%zu",
           chunk_data_total,
           chunk_data_total + index_data_bytes + 4);

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// --- Error path tests ---

static int
test_stream_zero_length_append(void)
{
  log_info("=== test_stream_zero_length_append ===");

  struct dimension dims[3];
  make_test_dims_3d(dims);

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // Append empty slice
  uint16_t dummy;
  struct slice empty = { .beg = &dummy, .end = &dummy };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), empty);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, tile_stream_gpu_cursor(s) == 0);

  // Now append real data and verify it still works
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  struct slice input = { .beg = src, .end = src + countof(src) };
  r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail, r.error == 0);
  CHECK(Fail, mss.writers[0][0].size > 0);

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

static int
test_stream_null_config_fields(void)
{
  log_info("=== test_stream_null_config_fields ===");

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .storage_position = 0 },
    { .size = 6, .chunk_size = 3, .storage_position = 1 },
  };

  // NULL shard_sink should cause create to fail
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 24 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, NULL);
  if (!s) {
    log_info("  create correctly returned NULL for NULL shard_sink");
    log_info("  PASS");
    return 0;
  }

  // If it didn't fail, clean up and report
  log_error("  create succeeded with NULL shard_sink — expected failure");
  tile_stream_gpu_destroy(s);
  log_error("  FAIL");
  return 1;
}

static int
test_stream_rank_1_dim(void)
{
  log_info("=== test_stream_rank_1_dim ===");

  struct dimension dims[] = {
    { .size = 12, .chunk_size = 4 },
  };

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 12 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 1,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // Verify we can push data through
  uint16_t src[12];
  for (int i = 0; i < 12; ++i)
    src[i] = (uint16_t)i;

  struct slice input = { .beg = src, .end = src + 12 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail, r.error == 0);
  CHECK(Fail, mss.writers[0][0].size > 0);
  log_info("  rank=1 pipeline produced %zu bytes", mss.writers[0][0].size);

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

static int
test_stream_flush_empty(void)
{
  log_info("=== test_stream_flush_empty ===");

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .storage_position = 0 },
    { .size = 6, .chunk_size = 3, .storage_position = 1 },
  };

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 24 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // Flush with no data appended — should be a no-op
  struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail, r.error == 0);
  CHECK(Fail, tile_stream_gpu_cursor(s) == 0);

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: unbounded dim0 (size=0) — stream multiple epochs without crashing.
static int
test_stream_unbounded_dim0(void)
{
  log_info("=== test_stream_unbounded_dim0 ===");

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);

  struct test_shard_sink mss;
  test_sink_init(&mss, 2, 1024 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // chunks_per_epoch should be prod(chunk_count[d] for d>0) = 2*2 = 4
  CHECK(Fail, tile_stream_gpu_layout(s)->chunks_per_epoch == 4);
  CHECK(Fail, tile_stream_gpu_layout(s)->epoch_elements == 48);
  log_info("  chunks_per_epoch=%lu  epoch_elements=%lu",
           (unsigned long)tile_stream_gpu_layout(s)->chunks_per_epoch,
           (unsigned long)tile_stream_gpu_layout(s)->epoch_elements);

  // Stream 4 epochs worth of data (192 elements)
  const int total = 4 * 48;
  uint16_t* src = (uint16_t*)malloc(total * sizeof(uint16_t));
  CHECK(Fail, src);
  for (int i = 0; i < total; ++i)
    src[i] = (uint16_t)(i % 65536);

  struct slice input = { .beg = src, .end = src + total };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);
  CHECK(Fail2, mss.writers[0][0].size > 0);
  log_info(
    "  streamed %d elements, shard bytes=%zu", total, mss.writers[0][0].size);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: unbounded dim0 requires chunks_per_shard > 0.
static int
test_stream_unbounded_requires_tps(void)
{
  log_info("=== test_stream_unbounded_requires_tps ===");

  // size=0, chunks_per_shard=0 → should fail validation
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 2,
      .chunks_per_shard = 0,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .storage_position = 1 },
    { .size = 6, .chunk_size = 3, .storage_position = 2 },
  };

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &mss.base);
  if (!s) {
    log_info("  create correctly rejected unbounded dim0 with cps=0");
    test_sink_free(&mss);
    log_info("  PASS");
    return 0;
  }

  log_error("  create should have failed for unbounded dim0 with cps=0");
  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: bounded dim0 — append more data than capacity, expect auto-flush.
static int
test_stream_bounded_dim0(void)
{
  log_info("=== test_stream_bounded_dim0 ===");

  // dim0.size=4, chunk_size=2 → 2 epochs max → 96 elements capacity
  struct dimension dims[3];
  make_test_dims_3d(dims);

  struct test_shard_sink mss;
  test_sink_init(&mss, 1, 256 * 1024);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail0, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

  // Try to feed 150 elements (more than 96 capacity)
  const int total = 150;
  uint16_t src[150];
  for (int i = 0; i < total; ++i)
    src[i] = (uint16_t)i;

  struct slice input = { .beg = src, .end = src + total };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);

  // Should get writer_error_finished (auto-flushed at capacity)
  CHECK(Fail, r.error == writer_error_finished);
  log_info("  got writer_error_finished as expected");

  // rest should point to unconsumed data
  size_t consumed = (size_t)((const uint16_t*)r.rest.beg - src);
  size_t unconsumed =
    (size_t)((const uint16_t*)r.rest.end - (const uint16_t*)r.rest.beg);
  log_info(
    "  consumed=%zu elements, unconsumed=%zu elements", consumed, unconsumed);
  CHECK(Fail, consumed == 96);
  CHECK(Fail, unconsumed == 54);

  // Shard data should have been written
  CHECK(Fail, mss.writers[0][0].size > 0);
  log_info("  shard bytes=%zu", mss.writers[0][0].size);

  tile_stream_gpu_destroy(s);
  test_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Verify shard index structure: monotonic offsets, size sum, CRC.
// Tests with both even (multi-shard u32) and single (u16) cases.
// Moved from test_shard_contents.c.
static int
test_shard_index_structure(void)
{
  log_info("=== test_shard_index_structure ===");

  crc32c_init();

  // --- Case 1: Even tiling (all chunks fill shard evenly) ---
  {
    const int size[3] = { 12, 8, 12 };
    const int chunk_size[3] = { 2, 4, 3 };
    const int cps[3] = { 3, 2, 2 };

    const int chunk_count[3] = {
      size[0] / chunk_size[0],
      size[1] / chunk_size[1],
      size[2] / chunk_size[2],
    };
    const int shard_count[3] = {
      chunk_count[0] / cps[0],
      chunk_count[1] / cps[1],
      chunk_count[2] / cps[2],
    };

    const int total_elements = size[0] * size[1] * size[2];
    const int num_shards = shard_count[0] * shard_count[1] * shard_count[2];
    const int chunks_per_shard_total = cps[0] * cps[1] * cps[2];

    uint32_t* src = generate_encoded_volume(size, chunk_size, cps);
    CHECK(Fail0, src);

    struct test_shard_sink mss;
    test_sink_init(&mss, num_shards, 256 * 1024);

    struct dimension dims[] = {
      { .size = 12,
        .chunk_size = 2,
        .chunks_per_shard = 3,
        .storage_position = 0 },
      { .size = 8,
        .chunk_size = 4,
        .chunks_per_shard = 2,
        .storage_position = 1 },
      { .size = 12,
        .chunk_size = 3,
        .chunks_per_shard = 2,
        .storage_position = 2 },
    };

    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = total_elements * sizeof(uint32_t),
      .dtype = dtype_u32,
      .rank = 3,
      .dimensions = dims,
      .codec = CODEC_ZSTD,
    };

    struct tile_stream_gpu* s = NULL;
    CHECK(Fail2, (s = tile_stream_gpu_create(&config, &mss.base)) != NULL);

    {
      struct slice input = { .beg = src, .end = src + total_elements };
      struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
      CHECK(Fail3, r.error == 0);
    }
    {
      struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
      CHECK(Fail3, r.error == 0);
    }

    // Verify index structure for each shard
    const size_t index_data_bytes =
      (size_t)chunks_per_shard_total * 2 * sizeof(uint64_t);
    const size_t index_total_bytes = index_data_bytes + 4;
    int errors = 0;

    for (int si = 0; si < num_shards; ++si) {
      struct test_shard_writer* w = &mss.writers[0][si];
      CHECK(Fail3, w->finalized);
      CHECK(Fail3, w->size > index_total_bytes);

      uint64_t chunk_offsets[12], tile_nbytes[12];
      CHECK(Fail3,
            shard_index_parse(w->buf,
                              w->size,
                              (size_t)chunks_per_shard_total,
                              chunk_offsets,
                              tile_nbytes) == 0);

      // 1. Offsets must be monotonically non-decreasing
      for (int i = 1; i < chunks_per_shard_total; ++i) {
        if (chunk_offsets[i] < chunk_offsets[i - 1]) {
          log_error("  shard %d: offset[%d]=%lu < offset[%d]=%lu",
                    si,
                    i,
                    (unsigned long)chunk_offsets[i],
                    i - 1,
                    (unsigned long)chunk_offsets[i - 1]);
          errors++;
        }
      }

      // 2. Sum of chunk sizes + index block + CRC = shard size
      size_t chunk_data_sum = 0;
      for (int i = 0; i < chunks_per_shard_total; ++i)
        chunk_data_sum += tile_nbytes[i];

      if (chunk_data_sum + index_total_bytes != w->size) {
        log_error(
          "  shard %d: chunk_data_sum=%zu + index=%zu != shard_size=%zu",
          si,
          chunk_data_sum,
          index_total_bytes,
          w->size);
        errors++;
      }

      // 3. Verify CRC32C
      const uint8_t* index_ptr = w->buf + w->size - index_total_bytes;
      uint32_t stored_crc;
      memcpy(&stored_crc, index_ptr + index_data_bytes, 4);
      uint32_t computed_crc = crc32c(index_ptr, index_data_bytes);
      if (stored_crc != computed_crc) {
        log_error("  shard %d: CRC mismatch (stored=0x%08x computed=0x%08x)",
                  si,
                  stored_crc,
                  computed_crc);
        errors++;
      }
    }

    if (errors > 0) {
      log_error("  %d index structure errors", errors);
      goto Fail3;
    }

    log_info("  even tiling: %d shards verified", num_shards);
    tile_stream_gpu_destroy(s);
    test_sink_free(&mss);
    free(src);
    goto Case2;

  Fail3:
    tile_stream_gpu_destroy(s);
  Fail2:
    test_sink_free(&mss);
  Fail0:
    free(src);
    log_error("  FAIL");
    return 1;
  }

Case2:
  // --- Case 2: Single shard (u16 data, smaller shape) ---
  {
    struct dimension dims2[] = {
      { .size = 4, .chunk_size = 2, .storage_position = 0 },
      { .size = 4, .chunk_size = 2, .storage_position = 1 },
      { .size = 6, .chunk_size = 3, .storage_position = 2 },
    };
    const size_t chunks_per_shard_total2 = 8;
    const int total2 = 96;

    struct test_shard_sink mss2;
    test_sink_init(&mss2, 1, 256 * 1024);

    const struct tile_stream_configuration config2 = {
      .buffer_capacity_bytes = total2 * sizeof(uint16_t),
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = dims2,
      .codec = CODEC_ZSTD,
    };

    struct tile_stream_gpu* s2 = NULL;
    CHECK(FailB1, (s2 = tile_stream_gpu_create(&config2, &mss2.base)) != NULL);

    uint16_t src2[96];
    for (int i = 0; i < 96; ++i)
      src2[i] = (uint16_t)i;

    {
      struct slice input = { .beg = src2, .end = src2 + total2 };
      struct writer_result r = writer_append(tile_stream_gpu_writer(s2), input);
      CHECK(FailB2, r.error == 0);
    }
    {
      struct writer_result r = writer_flush(tile_stream_gpu_writer(s2));
      CHECK(FailB2, r.error == 0);
    }

    struct test_shard_writer* w = &mss2.writers[0][0];
    CHECK(FailB2, w->finalized);

    uint64_t offs[8], sizes[8];
    CHECK(FailB2,
          shard_index_parse(
            w->buf, w->size, chunks_per_shard_total2, offs, sizes) == 0);

    // Monotonic offsets
    for (size_t i = 1; i < chunks_per_shard_total2; ++i)
      CHECK(FailB2, offs[i] >= offs[i - 1]);

    // Size sum
    size_t sum = 0;
    for (size_t i = 0; i < chunks_per_shard_total2; ++i)
      sum += sizes[i];
    const size_t index_data_bytes2 =
      chunks_per_shard_total2 * 2 * sizeof(uint64_t);
    const size_t index_total_bytes2 = index_data_bytes2 + 4;
    CHECK(FailB2, sum + index_total_bytes2 == w->size);

    // CRC
    const uint8_t* index_ptr2 = w->buf + w->size - index_total_bytes2;
    uint32_t stored, computed;
    memcpy(&stored, index_ptr2 + index_data_bytes2, 4);
    computed = crc32c(index_ptr2, index_data_bytes2);
    CHECK(FailB2, stored == computed);

    log_info("  single shard (u16): verified");
    tile_stream_gpu_destroy(s2);
    test_sink_free(&mss2);
    goto Done;

  FailB2:
    tile_stream_gpu_destroy(s2);
  FailB1:
    test_sink_free(&mss2);
    log_error("  FAIL");
    return 1;
  }

Done:
  log_info("  PASS");
  return 0;
}

RUN_GPU_TESTS({ "single_append", test_stream_single_append },
              { "incremental_append", test_stream_incremental_append },
              { "compressed_roundtrip", test_stream_compressed_roundtrip },
              { "lz4_roundtrip", test_stream_lz4_roundtrip },
              { "zero_length_append", test_stream_zero_length_append },
              { "null_config_fields", test_stream_null_config_fields },
              { "rank_1_dim", test_stream_rank_1_dim },
              { "flush_empty", test_stream_flush_empty },
              { "unbounded_dim0", test_stream_unbounded_dim0 },
              { "unbounded_requires_tps", test_stream_unbounded_requires_tps },
              { "bounded_dim0", test_stream_bounded_dim0 },
              { "shard_index_structure", test_shard_index_structure }, )

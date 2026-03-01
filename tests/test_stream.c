#include "index.ops.util.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- tile pool verification helpers ---

// Build expected tile pool for one epoch.
static uint16_t*
make_expected_tiles(uint64_t epoch_start,
                    uint64_t epoch_elements,
                    uint64_t tiles_per_epoch,
                    uint64_t tile_elements,
                    uint8_t lifted_rank,
                    const uint64_t* lifted_shape,
                    const int64_t* lifted_strides)
{
  uint64_t pool_size = tiles_per_epoch * tile_elements;
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

// --- Collecting shard writer ---

struct mem_shard_writer
{
  struct shard_writer base;
  uint8_t* buf;
  size_t capacity;
  size_t size; // high water mark
};

struct mem_shard_sink
{
  struct shard_sink base;
  struct mem_shard_writer writer;
};

static int
mem_shard_write(struct shard_writer* self,
                uint64_t offset,
                const void* beg,
                const void* end)
{
  struct mem_shard_writer* w = (struct mem_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > w->capacity) {
    log_error("mem_shard_write: overflow");
    return 1;
  }
  memcpy(w->buf + offset, beg, nbytes);
  if (offset + nbytes > w->size)
    w->size = offset + nbytes;
  return 0;
}

static int
mem_shard_finalize(struct shard_writer* self)
{
  (void)self;
  return 0;
}

static struct shard_writer*
mem_shard_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct mem_shard_sink* s = (struct mem_shard_sink*)self;
  return &s->writer.base;
}

static void
mem_shard_sink_init(struct mem_shard_sink* s, size_t capacity)
{
  *s = (struct mem_shard_sink){
    .base = { .open = mem_shard_open },
  };
  s->writer = (struct mem_shard_writer){
    .base = { .write = mem_shard_write, .finalize = mem_shard_finalize },
    .buf = (uint8_t*)calloc(1, capacity),
    .capacity = capacity,
  };
}

static void
mem_shard_sink_free(struct mem_shard_sink* s)
{
  free(s->writer.buf);
  *s = (struct mem_shard_sink){ 0 };
}

// Test: feed all data in one append call.
// Shape (4,4,6), tile (2,2,3) -> 2 epochs, 4 tiles/epoch, 12 elements/tile.
// Total 96 elements. Uses CODEC_NONE shard path.
static int
test_stream_single_append(void)
{
  log_info("=== test_stream_single_append ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 }, // slowest (dim 0)
    { .size = 4, .tile_size = 2 }, // dim 1
    { .size = 6, .tile_size = 3 }, // fastest (dim 2)
  };

  // tiles_per_shard defaults to tile_count → single shard containing all tiles.
  // tile_count = (2, 2, 2), tiles_per_shard_total = 8.
  const size_t tiles_per_shard_total = 8;

  struct mem_shard_sink mss;
  mem_shard_sink_init(&mss, 256 * 1024);
  CHECK(Fail0, mss.writer.buf);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .shard_sink = &mss.base,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu s;
  CHECK(Fail0, tile_stream_gpu_create(&config, &s) == 0);

  // Verify computed layout
  log_info("  tile_elements=%lu  tiles_per_epoch=%lu  epoch_elements=%lu",
           (unsigned long)s.layout.tile_elements,
           (unsigned long)s.layout.tiles_per_epoch,
           (unsigned long)s.layout.epoch_elements);
  CHECK(Fail, s.layout.tile_elements == 12);
  CHECK(Fail, s.layout.tiles_per_epoch == 4);
  CHECK(Fail, s.layout.epoch_elements == 48);

  {
    printf("  lifted_shape: ");
    println_vu64(s.layout.lifted_rank, s.layout.lifted_shape);
    printf("  lifted_strides: ");
    println_vi64(s.layout.lifted_rank, s.layout.lifted_strides);
  }

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  // Append all data
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(&s.writer, input);
  CHECK(Fail, r.error == 0);

  // Flush to get all data
  r = writer_flush(&s.writer);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, mss.writer.size > 0);

  // Parse shard index from the end of the buffer.
  const size_t tile_bytes = s.layout.tile_stride * sizeof(uint16_t);
  const size_t index_data_bytes = tiles_per_shard_total * 2 * sizeof(uint64_t);
  const size_t shard_size = mss.writer.size;
  CHECK(Fail, shard_size > index_data_bytes + 4);
  const uint8_t* index_ptr = mss.writer.buf + shard_size - index_data_bytes - 4;

  uint64_t tile_offsets[8], tile_sizes[8];
  for (size_t i = 0; i < tiles_per_shard_total; ++i) {
    memcpy(&tile_offsets[i], index_ptr + i * 16, sizeof(uint64_t));
    memcpy(&tile_sizes[i], index_ptr + i * 16 + 8, sizeof(uint64_t));
  }

  for (int epoch = 0; epoch < 2; ++epoch) {
    uint16_t* expected =
      make_expected_tiles((uint64_t)epoch * s.layout.epoch_elements,
                          s.layout.epoch_elements,
                          s.layout.tiles_per_epoch,
                          s.layout.tile_elements,
                          s.layout.lifted_rank,
                          s.layout.lifted_shape,
                          s.layout.lifted_strides);
    CHECK(Fail, expected);

    int err = 0;
    for (uint64_t t = 0; t < s.layout.tiles_per_epoch; ++t) {
      size_t slot = (size_t)epoch * s.layout.tiles_per_epoch + t;
      CHECK(Fail, tile_sizes[slot] == tile_bytes);

      const uint16_t* tile_data =
        (const uint16_t*)(mss.writer.buf + tile_offsets[slot]);
      const uint16_t* expected_tile = expected + t * s.layout.tile_elements;
      for (uint64_t e = 0; e < s.layout.tile_elements; ++e) {
        if (tile_data[e] != expected_tile[e]) {
          log_error("  epoch %d tile %lu elem %lu: expected %u, got %u",
                    epoch,
                    (unsigned long)t,
                    (unsigned long)e,
                    expected_tile[e],
                    tile_data[e]);
          err = 1;
        }
      }
    }
    free(expected);
    if (err) {
      log_error("  FAIL: epoch %d verification", epoch);
      goto Fail;
    }
    log_info("  epoch %d: OK", epoch);
  }

  tile_stream_gpu_destroy(&s);
  mem_shard_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(&s);
Fail0:
  mem_shard_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: feed data in small chunks (e.g., 7 elements at a time)
// to exercise buffer-fill + dispatch + epoch-crossing logic.
// Uses CODEC_NONE shard path.
static int
test_stream_chunked_append(void)
{
  log_info("=== test_stream_chunked_append ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 },
    { .size = 4, .tile_size = 2 },
    { .size = 6, .tile_size = 3 },
  };

  const size_t tiles_per_shard_total = 8;

  struct mem_shard_sink mss;
  mem_shard_sink_init(&mss, 256 * 1024);
  CHECK(Fail0, mss.writer.buf);

  // Small buffer: 10 elements worth (rounded up to 4KB internally)
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 10 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .shard_sink = &mss.base,
    .codec = CODEC_NONE,
  };

  struct tile_stream_gpu s;
  CHECK(Fail0, tile_stream_gpu_create(&config, &s) == 0);

  const int total = 96;
  uint16_t src[96];
  for (int i = 0; i < total; ++i)
    src[i] = (uint16_t)i;

  // Feed in chunks of 7 elements
  const int chunk_elements = 7;

  for (int off = 0; off < total; off += chunk_elements) {
    int n = chunk_elements;
    if (off + n > total)
      n = total - off;

    struct slice input = { .beg = src + off, .end = src + off + n };
    struct writer_result r = writer_append(&s.writer, input);
    CHECK(Fail, r.error == 0);
  }

  // Flush remaining data
  {
    struct writer_result r = writer_flush(&s.writer);
    CHECK(Fail, r.error == 0);
  }

  CHECK(Fail, mss.writer.size > 0);

  // Parse shard index
  const size_t tile_bytes = s.layout.tile_stride * sizeof(uint16_t);
  const size_t index_data_bytes = tiles_per_shard_total * 2 * sizeof(uint64_t);
  const size_t shard_size = mss.writer.size;
  CHECK(Fail, shard_size > index_data_bytes + 4);
  const uint8_t* index_ptr = mss.writer.buf + shard_size - index_data_bytes - 4;

  uint64_t tile_offsets[8], tile_sizes[8];
  for (size_t i = 0; i < tiles_per_shard_total; ++i) {
    memcpy(&tile_offsets[i], index_ptr + i * 16, sizeof(uint64_t));
    memcpy(&tile_sizes[i], index_ptr + i * 16 + 8, sizeof(uint64_t));
  }

  for (int epoch = 0; epoch < 2; ++epoch) {
    uint16_t* expected =
      make_expected_tiles((uint64_t)epoch * s.layout.epoch_elements,
                          s.layout.epoch_elements,
                          s.layout.tiles_per_epoch,
                          s.layout.tile_elements,
                          s.layout.lifted_rank,
                          s.layout.lifted_shape,
                          s.layout.lifted_strides);
    CHECK(Fail, expected);

    int err = 0;
    for (uint64_t t = 0; t < s.layout.tiles_per_epoch; ++t) {
      size_t slot = (size_t)epoch * s.layout.tiles_per_epoch + t;
      CHECK(Fail, tile_sizes[slot] == tile_bytes);

      const uint16_t* tile_data =
        (const uint16_t*)(mss.writer.buf + tile_offsets[slot]);
      const uint16_t* expected_tile = expected + t * s.layout.tile_elements;
      for (uint64_t e = 0; e < s.layout.tile_elements; ++e) {
        if (tile_data[e] != expected_tile[e]) {
          log_error("  epoch %d tile %lu elem %lu: expected %u, got %u",
                    epoch,
                    (unsigned long)t,
                    (unsigned long)e,
                    expected_tile[e],
                    tile_data[e]);
          err = 1;
        }
      }
    }
    free(expected);
    if (err) {
      log_error("  FAIL: epoch %d verification", epoch);
      goto Fail;
    }
    log_info("  epoch %d: OK", epoch);
  }

  tile_stream_gpu_destroy(&s);
  mem_shard_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(&s);
Fail0:
  mem_shard_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// Test: compressed roundtrip via shard path — compress tiles with nvcomp,
// collect shard data, parse index, decompress with libzstd, verify contents
// match expected tile pool.
static int
test_stream_compressed_roundtrip(void)
{
  log_info("=== test_stream_compressed_roundtrip ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 },
    { .size = 4, .tile_size = 2 },
    { .size = 6, .tile_size = 3 },
  };

  // tiles_per_shard defaults to tile_count → single shard containing all tiles.
  // tile_count = (2, 2, 2), tiles_per_shard_total = 8.
  const size_t tiles_per_shard_total = 8;

  // Generous buffer for compressed shard data + index
  struct mem_shard_sink mss;
  mem_shard_sink_init(&mss, 256 * 1024);
  CHECK(Fail0, mss.writer.buf);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = &mss.base,
  };

  struct tile_stream_gpu s;
  CHECK(Fail0, tile_stream_gpu_create(&config, &s) == 0);

  log_info("  tile_elements=%lu  tile_stride=%lu  tiles_per_epoch=%lu  "
           "epoch_elements=%lu",
           (unsigned long)s.layout.tile_elements,
           (unsigned long)s.layout.tile_stride,
           (unsigned long)s.layout.tiles_per_epoch,
           (unsigned long)s.layout.epoch_elements);
  log_info("  max_comp_chunk_bytes=%zu  tile_pool_bytes=%zu",
           s.codec.max_chunk_bytes,
           s.layout.tile_pool_bytes);

  CHECK(Fail, s.layout.tile_elements == 12);
  CHECK(Fail, s.layout.tiles_per_epoch == 4);
  CHECK(Fail, s.layout.epoch_elements == 48);

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  // Append all data
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(&s.writer, input);
  CHECK(Fail, r.error == 0);

  r = writer_flush(&s.writer);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, mss.writer.size > 0);

  // Parse shard index from the end of the buffer.
  // Index layout: tiles_per_shard_total * 2 * uint64_t + 4-byte CRC32C
  const size_t index_data_bytes = tiles_per_shard_total * 2 * sizeof(uint64_t);
  const size_t tile_bytes = s.layout.tile_stride * sizeof(uint16_t);

  // The index is the last write. Read tile offsets/sizes from it.
  // Shard data layout: [compressed tile data...] [index block + crc]
  // The index block starts at (shard_size - index_data_bytes - 4).
  const size_t shard_size = mss.writer.size;
  CHECK(Fail, shard_size > index_data_bytes + 4);
  const uint8_t* index_ptr = mss.writer.buf + shard_size - index_data_bytes - 4;

  // Parse index: tiles_per_shard_total pairs of (offset, nbytes) as uint64
  uint64_t tile_offsets[8], tile_sizes[8];
  for (size_t i = 0; i < tiles_per_shard_total; ++i) {
    memcpy(&tile_offsets[i], index_ptr + i * 16, sizeof(uint64_t));
    memcpy(&tile_sizes[i], index_ptr + i * 16 + 8, sizeof(uint64_t));
  }

  // With single shard + identity permutation, shard index slots map:
  //   slot = epoch * tiles_per_shard_inner + within_inner
  // where tiles_per_shard_inner = 4, within_inner = tile pool index.
  // So slot 0..3 = epoch 0 tiles 0..3, slot 4..7 = epoch 1 tiles 0..3.
  for (int epoch = 0; epoch < 2; ++epoch) {
    uint16_t* expected =
      make_expected_tiles((uint64_t)epoch * s.layout.epoch_elements,
                          s.layout.epoch_elements,
                          s.layout.tiles_per_epoch,
                          s.layout.tile_elements,
                          s.layout.lifted_rank,
                          s.layout.lifted_shape,
                          s.layout.lifted_strides);
    CHECK(Fail, expected);

    int err = 0;
    for (uint64_t t = 0; t < s.layout.tiles_per_epoch; ++t) {
      size_t slot = (size_t)epoch * s.layout.tiles_per_epoch + t;
      CHECK(Fail, tile_sizes[slot] > 0);

      const uint8_t* comp_data = mss.writer.buf + tile_offsets[slot];

      uint8_t* decomp = (uint8_t*)calloc(1, tile_bytes);
      CHECK(Fail, decomp);

      size_t result =
        ZSTD_decompress(decomp, tile_bytes, comp_data, tile_sizes[slot]);
      if (ZSTD_isError(result)) {
        log_error("  ZSTD_decompress failed for tile %lu epoch %d: %s",
                  (unsigned long)t,
                  epoch,
                  ZSTD_getErrorName(result));
        free(decomp);
        free(expected);
        goto Fail;
      }
      CHECK(Fail, result == tile_bytes);

      const uint16_t* decomp_u16 = (const uint16_t*)decomp;
      const uint16_t* expected_tile = expected + t * s.layout.tile_elements;
      for (uint64_t e = 0; e < s.layout.tile_elements; ++e) {
        if (decomp_u16[e] != expected_tile[e]) {
          log_error("  epoch %d tile %lu elem %lu: expected %u, got %u",
                    epoch,
                    (unsigned long)t,
                    (unsigned long)e,
                    expected_tile[e],
                    decomp_u16[e]);
          err = 1;
        }
      }
      free(decomp);
    }
    free(expected);
    if (err) {
      log_error("  FAIL: epoch %d verification", epoch);
      goto Fail;
    }
    log_info("  epoch %d: OK", epoch);
  }

  tile_stream_gpu_destroy(&s);
  mem_shard_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(&s);
Fail0:
  mem_shard_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_stream_single_append();
  log_info("");
  ecode |= test_stream_chunked_append();
  log_info("");
  ecode |= test_stream_compressed_roundtrip();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}

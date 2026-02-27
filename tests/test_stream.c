#include "index.ops.util.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "writer.mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// Decompose flat index into C row-major coords: dim 0 slowest, dim rank-1
// fastest.
static void
unravel(int rank, const uint64_t* shape, size_t idx, int* coords)
{
  size_t rem = idx;
  for (int d = rank - 1; d >= 0; --d) {
    coords[d] = (int)(rem % shape[d]);
    rem /= shape[d];
  }
}

// --- tile pool verification helpers ---

// Build expected tile pool for one epoch.
static uint16_t*
make_expected_tiles(uint64_t epoch_start,
                    uint64_t epoch_elements,
                    uint64_t slot_count,
                    uint64_t tile_elements,
                    uint8_t lifted_rank,
                    const uint64_t* lifted_shape,
                    const int64_t* lifted_strides)
{
  uint64_t pool_size = slot_count * tile_elements;
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

// Compare tile pool contents against expected.
static int
verify_tiles(const uint16_t* actual,
             const uint16_t* expected,
             uint64_t pool_size,
             int epoch,
             const char* label)
{
  int ecode = 0;
  int mismatch_count = 0;
  for (uint64_t i = 0; i < pool_size; ++i) {
    if (actual[i] != expected[i]) {
      if (mismatch_count < 10) {
        log_error("%s epoch %d: mismatch at pool[%lu]: expected %u, got %u",
                  label,
                  epoch,
                  (unsigned long)i,
                  expected[i],
                  actual[i]);
      }
      mismatch_count++;
      ecode = 1;
    }
  }
  if (mismatch_count > 10) {
    log_warn(
      "%s epoch %d: ... %d total mismatches", label, epoch, mismatch_count);
  }
  return ecode;
}

// Test: feed all data in one append call.
// Shape (4,4,6), tile (2,2,3) -> 2 epochs, 4 tiles/epoch, 12 elements/tile.
// Total 96 elements.
static int
test_stream_single_append(void)
{
  log_info("=== test_stream_single_append ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 }, // slowest (dim 0)
    { .size = 4, .tile_size = 2 }, // dim 1
    { .size = 6, .tile_size = 3 }, // fastest (dim 2)
  };

  // 2 epochs * 4 slots * 12 elements * 2 bytes = 192 bytes
  const size_t pool_bytes = 4 * 12 * sizeof(uint16_t);
  struct mem_writer mw = mem_writer_new(2 * pool_bytes);
  CHECK(Fail0, mw.buf);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .sink = &mw.base,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

  // Verify computed layout
  log_info("  tile_elements=%lu  slot_count=%lu  epoch_elements=%lu",
           (unsigned long)s.layout.tile_elements,
           (unsigned long)s.layout.slot_count,
           (unsigned long)s.layout.epoch_elements);
  CHECK(Fail, s.layout.tile_elements == 12);
  CHECK(Fail, s.layout.slot_count == 4);
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

  // Append all data — this processes epoch 0, flushes it to the writer,
  // then processes epoch 1's data.
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(&s.writer, input);
  CHECK(Fail, r.error == 0);

  // Epoch 0 was flushed to mw during append.
  CHECK(Fail, mw.cursor >= pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(0,
                                             s.layout.epoch_elements,
                                             s.layout.slot_count,
                                             s.layout.tile_elements,
                                             s.layout.lifted_rank,
                                             s.layout.lifted_shape,
                                             s.layout.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)mw.buf,
                           expected,
                           s.layout.slot_count * s.layout.tile_elements,
                           0,
                           "single_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 0 verification");
      goto Fail;
    }
    log_info("  epoch 0: OK");
  }

  // Flush to get epoch 1
  r = writer_flush(&s.writer);
  CHECK(Fail, r.error == 0);

  CHECK(Fail, mw.cursor == 2 * pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(s.layout.epoch_elements,
                                             s.layout.epoch_elements,
                                             s.layout.slot_count,
                                             s.layout.tile_elements,
                                             s.layout.lifted_rank,
                                             s.layout.lifted_shape,
                                             s.layout.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)(mw.buf + pool_bytes),
                           expected,
                           s.layout.slot_count * s.layout.tile_elements,
                           1,
                           "single_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 1 verification");
      goto Fail;
    }
    log_info("  epoch 1: OK");
  }

  transpose_stream_destroy(&s);
  mem_writer_free(&mw);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  mem_writer_free(&mw);
  log_error("  FAIL");
  return 1;
}

// Test: feed data in small chunks (e.g., 7 elements at a time)
// to exercise buffer-fill + dispatch + epoch-crossing logic.
static int
test_stream_chunked_append(void)
{
  log_info("=== test_stream_chunked_append ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 },
    { .size = 4, .tile_size = 2 },
    { .size = 6, .tile_size = 3 },
  };

  const size_t pool_bytes = 4 * 12 * sizeof(uint16_t);
  struct mem_writer mw = mem_writer_new(2 * pool_bytes);
  CHECK(Fail0, mw.buf);

  // Small buffer: 10 elements worth (rounded up to 4KB internally)
  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 10 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .sink = &mw.base,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

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

  // Epoch 0 was flushed during append.
  CHECK(Fail, mw.cursor >= pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(0,
                                             s.layout.epoch_elements,
                                             s.layout.slot_count,
                                             s.layout.tile_elements,
                                             s.layout.lifted_rank,
                                             s.layout.lifted_shape,
                                             s.layout.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)mw.buf,
                           expected,
                           s.layout.slot_count * s.layout.tile_elements,
                           0,
                           "chunked_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 0 verification");
      goto Fail;
    }
    log_info("  epoch 0: OK");
  }

  // Flush remaining data (epoch 1)
  {
    struct writer_result r = writer_flush(&s.writer);
    CHECK(Fail, r.error == 0);
  }

  CHECK(Fail, mw.cursor == 2 * pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(s.layout.epoch_elements,
                                             s.layout.epoch_elements,
                                             s.layout.slot_count,
                                             s.layout.tile_elements,
                                             s.layout.lifted_rank,
                                             s.layout.lifted_shape,
                                             s.layout.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)(mw.buf + pool_bytes),
                           expected,
                           s.layout.slot_count * s.layout.tile_elements,
                           1,
                           "chunked_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 1 verification");
      goto Fail;
    }
    log_info("  epoch 1: OK");
  }

  transpose_stream_destroy(&s);
  mem_writer_free(&mw);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  mem_writer_free(&mw);
  log_error("  FAIL");
  return 1;
}

// --- Collecting shard writer for compressed roundtrip test ---

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

// --- Multi-level, multi-shard sink for LOD tests ---

#define MAX_SHARD_LEVELS 5
#define MAX_SHARDS_PER_LEVEL 16

struct multilevel_shard_sink
{
  struct shard_sink base;
  struct mem_shard_writer writers[MAX_SHARD_LEVELS][MAX_SHARDS_PER_LEVEL];
  int num_levels;
  int max_shards;
};

static struct shard_writer*
multilevel_shard_open(struct shard_sink* self,
                      uint8_t level,
                      uint64_t shard_index)
{
  struct multilevel_shard_sink* s = (struct multilevel_shard_sink*)self;
  if (level >= s->num_levels || (int)shard_index >= s->max_shards) {
    log_error("multilevel_shard_open: level %u shard %lu out of range",
              level,
              (unsigned long)shard_index);
    return NULL;
  }
  return &s->writers[level][shard_index].base;
}

static void
multilevel_shard_sink_init(struct multilevel_shard_sink* s,
                           int num_levels,
                           int max_shards,
                           size_t capacity)
{
  *s = (struct multilevel_shard_sink){
    .base = { .open = multilevel_shard_open },
    .num_levels = num_levels,
    .max_shards = max_shards,
  };
  for (int lv = 0; lv < num_levels; ++lv) {
    for (int si = 0; si < max_shards; ++si) {
      s->writers[lv][si] = (struct mem_shard_writer){
        .base = { .write = mem_shard_write, .finalize = mem_shard_finalize },
        .buf = (uint8_t*)calloc(1, capacity),
        .capacity = capacity,
      };
    }
  }
}

static void
multilevel_shard_sink_free(struct multilevel_shard_sink* s)
{
  for (int lv = 0; lv < s->num_levels; ++lv)
    for (int si = 0; si < s->max_shards; ++si)
      free(s->writers[lv][si].buf);
  *s = (struct multilevel_shard_sink){ 0 };
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

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &mss.base,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

  log_info("  tile_elements=%lu  tile_stride=%lu  slot_count=%lu  "
           "epoch_elements=%lu",
           (unsigned long)s.layout.tile_elements,
           (unsigned long)s.layout.tile_stride,
           (unsigned long)s.layout.slot_count,
           (unsigned long)s.layout.epoch_elements);
  log_info("  max_comp_chunk_bytes=%zu  tile_pool_bytes=%zu",
           s.max_comp_chunk_bytes,
           s.layout.tile_pool_bytes);

  CHECK(Fail, s.layout.tile_elements == 12);
  CHECK(Fail, s.layout.slot_count == 4);
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
                          s.layout.slot_count,
                          s.layout.tile_elements,
                          s.layout.lifted_rank,
                          s.layout.lifted_shape,
                          s.layout.lifted_strides);
    CHECK(Fail, expected);

    int err = 0;
    for (uint64_t t = 0; t < s.layout.slot_count; ++t) {
      size_t slot = (size_t)epoch * s.layout.slot_count + t;
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

  transpose_stream_destroy(&s);
  mem_shard_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  mem_shard_sink_free(&mss);
  log_error("  FAIL");
  return 1;
}

// --- LOD roundtrip test ---
//
// Feed sequential u16 data through the stream with enable_lod=1, collect
// shard output at level 0 and LOD level 1, decompress, and verify tile
// contents match expected values (original and downsampled means).
//
// Geometry: rank=3, shape=(8,8,8), tile_size=(2,4,4), all dims downsampled.
//   Level 0: tile_count=(4,2,2), slot_count=4, epochs=4, tile_elements=32
//   LOD 1:   size=(4,4,4), tile_count=(2,1,1), slot_count=1, needs_two_epochs=1
//            Gets 2 epochs from 4 level-0 epochs (pairs).
//            tiles_per_shard_total = 2 (one shard covering everything).

// Compute expected value at global coordinate c[] at a given LOD level.
// lv=0 is full resolution (source data), lv=1..N are successive 2x
// downsamplings. src_sizes = full resolution extent per dim.
// ds = downsample flag per dim.
static uint16_t
expected_lod_value(int rank,
                   int lv,
                   const int* c,
                   const int* src_sizes,
                   const int* ds)
{
  if (lv == 0) {
    int idx = 0;
    int stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
      idx += c[d] * stride;
      stride *= src_sizes[d];
    }
    return (uint16_t)(idx % 65536);
  }

  // Parent level sizes
  int parent_sizes[8];
  for (int d = 0; d < rank; ++d) {
    parent_sizes[d] = src_sizes[d];
    for (int l = 0; l < lv - 1; ++l)
      if (ds[d])
        parent_sizes[d] = (parent_sizes[d] + 1) / 2;
  }

  int num_ds = 0;
  for (int d = 0; d < rank; ++d)
    if (ds[d])
      num_ds++;
  int combos = 1 << num_ds;

  uint32_t sum = 0;
  for (int m = 0; m < combos; ++m) {
    int pc[8];
    int bit = 0;
    for (int d = 0; d < rank; ++d) {
      if (ds[d]) {
        int off = (m >> bit) & 1;
        bit++;
        pc[d] = 2 * c[d] + off;
        if (pc[d] >= parent_sizes[d])
          pc[d] = parent_sizes[d] - 1;
      } else {
        pc[d] = c[d];
      }
    }
    sum += (uint32_t)expected_lod_value(rank, lv - 1, pc, src_sizes, ds);
  }
  return (uint16_t)((sum + (uint32_t)(combos / 2)) / (uint32_t)combos);
}

// Decompress and verify all tiles in a shard against expected values.
// Works for both L0 (lv=0) and LOD levels (lv>0).
// shard_idx is the shard index along dim 0. tps0 = tiles_per_shard[0].
// Returns 0 on success.
static int
verify_shard(const char* label,
             int rank,
             int lv,
             int shard_idx,
             const struct mem_shard_writer* w,
             const uint64_t* tile_size,
             const int* level_sizes,
             const int* src_sizes,
             const int* ds,
             int tps0)
{
  // Compute tiles_per_shard along each dim
  uint64_t tps[8];
  tps[0] = (uint64_t)tps0;
  for (int d = 1; d < rank; ++d)
    tps[d] = ((uint64_t)level_sizes[d] + tile_size[d] - 1) / tile_size[d];

  size_t tps_total = 1;
  for (int d = 0; d < rank; ++d)
    tps_total *= (size_t)tps[d];

  size_t te = 1;
  for (int d = 0; d < rank; ++d)
    te *= (size_t)tile_size[d];

  size_t tile_bytes = te * sizeof(uint16_t);
  size_t padded_bytes = (tile_bytes + 511) & ~(size_t)511;

  // Parse shard index
  size_t index_data_bytes = tps_total * 2 * sizeof(uint64_t);
  if (w->size <= index_data_bytes + 4) {
    log_error("  %s shard %d: shard too small (%zu bytes)",
              label, shard_idx, w->size);
    return 1;
  }

  const uint8_t* index_ptr = w->buf + w->size - index_data_bytes - 4;

  uint8_t* decomp = (uint8_t*)calloc(1, padded_bytes);
  if (!decomp)
    return 1;

  int err = 0;

  for (size_t ti = 0; ti < tps_total; ++ti) {
    uint64_t tile_offset, tile_comp_size;
    memcpy(&tile_offset, index_ptr + ti * 16, sizeof(uint64_t));
    memcpy(&tile_comp_size, index_ptr + ti * 16 + 8, sizeof(uint64_t));

    if (tile_comp_size == 0) {
      log_error("  %s shard %d tile %zu: zero compressed size",
                label, shard_idx, ti);
      err++;
      continue;
    }

    size_t result = ZSTD_decompress(
      decomp, padded_bytes, w->buf + tile_offset, tile_comp_size);
    if (ZSTD_isError(result)) {
      log_error("  %s shard %d tile %zu: ZSTD_decompress failed: %s",
                label, shard_idx, ti, ZSTD_getErrorName(result));
      err++;
      continue;
    }

    int td[8];
    unravel(rank, tps, ti, td);

    const uint16_t* data = (const uint16_t*)decomp;
    for (size_t e = 0; e < te; ++e) {
      int ec[8];
      unravel(rank, tile_size, e, ec);

      // Global coordinate
      int gc[8];
      gc[0] = (shard_idx * tps[0] + td[0]) * (int)tile_size[0] + ec[0];
      for (int d = 1; d < rank; ++d)
        gc[d] = td[d] * (int)tile_size[d] + ec[d];

      // Skip elements beyond the level's extent (tile padding)
      int oob = 0;
      for (int d = 0; d < rank; ++d)
        if (gc[d] >= level_sizes[d])
          oob = 1;
      if (oob)
        continue;

      uint16_t expected = expected_lod_value(rank, lv, gc, src_sizes, ds);
      if (data[e] != expected) {
        if (err < 10)
          log_error("  %s shard %d tile %zu elem %zu: expected %u, got %u",
                    label, shard_idx, ti, e, expected, data[e]);
        err++;
      }
    }
  }

  free(decomp);
  if (err) {
    log_error("  FAIL: %s shard %d: %d mismatches", label, shard_idx, err);
    return 1;
  }
  log_info("  %s shard %d: OK", label, shard_idx);
  return 0;
}

static int
test_stream_lod_roundtrip(void)
{
  log_info("=== test_stream_lod_roundtrip ===");

  // Rank 4. Dims 0 and 3 are downsampled.
  // dim 0: size=16, tile_size=2, tiles_per_shard=1, ds=1 → tc=8
  //   LOD: 16→8→4→2 (stop: 2<=2)
  // dim 1: size=6, tile_size=6 → tc=1
  // dim 2: size=2, tile_size=2 → tc=1
  // dim 3: size=48, tile_size=6, ds=1 → tc=8
  //   LOD: 48→24→12→6 (stop: 6<=6)
  //
  // Volume = 16*6*2*48 = 9216 elements
  // L0:   shape=(16,6,2,48), tc=(8,1,1,8), slot_count=8, tile_elements=144
  //       epoch_elements=1152, epochs=8, 8 shards (8 tiles each)
  // LOD1: shape=(8,6,2,24),  tc=(4,1,1,4), slot_count=4, needs_two_epochs=1
  //       4 shards, 4 tiles each
  // LOD2: shape=(4,6,2,12),  tc=(2,1,1,2), slot_count=2
  //       2 shards, 2 tiles each
  // LOD3: shape=(2,6,2,6),   tc=(1,1,1,1), slot_count=1
  //       1 shard, 1 tile

  const struct dimension dims[] = {
    { .size = 16, .tile_size = 2, .tiles_per_shard = 1, .downsample = 1 },
    { .size = 6, .tile_size = 6 },
    { .size = 2, .tile_size = 2 },
    { .size = 48, .tile_size = 6, .downsample = 1 },
  };

  const int rank = 4;
  const int src_sizes[] = { 16, 6, 2, 48 };
  const int ds[] = { 1, 0, 0, 1 };

  // 4 resolution levels (L0 + 3 LOD), up to 8 shards per level
  struct multilevel_shard_sink mss;
  multilevel_shard_sink_init(&mss, 4, 8, 128 * 1024);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 9216 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = rank,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &mss.base,
    .enable_lod = 1,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

  log_info("  num_lod_levels=%d", s.num_levels);
  CHECK(Fail, s.num_levels == 3);

  log_info("  L0: tile_elements=%lu  slot_count=%lu  epoch_elements=%lu",
           (unsigned long)s.layout.tile_elements,
           (unsigned long)s.layout.slot_count,
           (unsigned long)s.layout.epoch_elements);
  CHECK(Fail, s.layout.tile_elements == 144);
  CHECK(Fail, s.layout.slot_count == 8);
  CHECK(Fail, s.layout.epoch_elements == 1152);

  for (int lv = 0; lv < 3; ++lv) {
    struct level_state* lod = &s.levels[lv];
    log_info("  LOD%d: tile_elements=%lu  slot_count=%lu  tps_total=%lu",
             lv + 1,
             (unsigned long)lod->layout.tile_elements,
             (unsigned long)lod->layout.slot_count,
             (unsigned long)lod->shard.tiles_per_shard_total);
  }

  // Fill source with sequential u16
  uint16_t src[9216];
  for (int i = 0; i < 9216; ++i)
    src[i] = (uint16_t)i;

  struct slice input = { .beg = src, .end = src + 9216 };
  struct writer_result r = writer_append(&s.writer, input);
  CHECK(Fail, r.error == 0);
  r = writer_flush(&s.writer);
  CHECK(Fail, r.error == 0);

  // --- Verify all levels (L0 + LOD1..LOD3) ---
  {
    const uint64_t tile_size[] = { 2, 6, 2, 6 };

    for (int lv = 0; lv <= 3; ++lv) {
      int lod_sz[4];
      for (int d = 0; d < rank; ++d) {
        lod_sz[d] = src_sizes[d];
        for (int l = 0; l < lv; ++l)
          if (ds[d])
            lod_sz[d] = (lod_sz[d] + 1) / 2;
      }

      int shard_count =
        (lod_sz[0] + (int)tile_size[0] - 1) / (int)tile_size[0];
      char label[16];
      snprintf(label, sizeof(label), lv == 0 ? "L0" : "LOD%d", lv);

      for (int si = 0; si < shard_count; ++si) {
        const struct mem_shard_writer* w = &mss.writers[lv][si];
        if (verify_shard(label, rank, lv, si, w,
                         tile_size, lod_sz, src_sizes, ds, 1))
          goto Fail;
      }
    }
  }

  transpose_stream_destroy(&s);
  multilevel_shard_sink_free(&mss);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  multilevel_shard_sink_free(&mss);
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
  log_info("");
  ecode |= test_stream_lod_roundtrip();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}

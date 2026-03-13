#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- Coordinate encoding ---
//
// 3D volume: dim 0 (slowest), dim 1, dim 2 (fastest)
//
//   dim | size | tile_size | tiles_per_shard | tile_count | shard_count
//   ----|------|-----------|-----------------|------------|------------
//    0  |  12  |     2     |       3         |     6      |     2
//    1  |   8  |     4     |       2         |     2      |     1
//    2  |  12  |     3     |       2         |     4      |     2
//
// Total elements: 12*8*12 = 1152
// Total bytes (u32): 4608
// Shards: 2*1*2 = 4 (flat index = s0 * (1*2) + s1 * 2 + s2)
// Tiles per shard: 3*2*2 = 12
// Voxels per tile: 2*4*3 = 24
// Tiles per epoch = tile_count[1]*tile_count[2] = 2*4 = 8
// Epoch elements = 8 * 24 = 192
// Epochs = tile_count[0] = 6
// Shard epochs = shard_count[0] = 2, tiles_per_shard[0] = 3 epochs/shard-epoch
//
// Encoding (27 bits of 32):
//   bits [26:24] = shard_coord[0]
//   bits [23:21] = shard_coord[1]
//   bits [20:18] = shard_coord[2]
//   bits [17:15] = tile_in_shard[0]
//   bits [14:12] = tile_in_shard[1]
//   bits [11:9]  = tile_in_shard[2]
//   bits [8:6]   = voxel_in_tile[0]
//   bits [5:3]   = voxel_in_tile[1]
//   bits [2:0]   = voxel_in_tile[2]

static uint32_t
encode_voxel(int s0,
             int s1,
             int s2,
             int t0,
             int t1,
             int t2,
             int v0,
             int v1,
             int v2)
{
  return ((uint32_t)s0 << 24) | ((uint32_t)s1 << 21) | ((uint32_t)s2 << 18) |
         ((uint32_t)t0 << 15) | ((uint32_t)t1 << 12) | ((uint32_t)t2 << 9) |
         ((uint32_t)v0 << 6) | ((uint32_t)v1 << 3) | ((uint32_t)v2);
}

// --- Multi-shard collecting sink ---

#define MAX_SHARDS 16

struct collecting_shard_writer
{
  struct shard_writer base;
  uint8_t* buf;
  size_t capacity;
  size_t size; // high water mark
  uint64_t shard_index;
  int finalized;
};

struct collecting_shard_sink
{
  struct shard_sink base;
  struct collecting_shard_writer writers[MAX_SHARDS];
  int num_shards;
  size_t per_shard_capacity;
};

static int
collecting_write(struct shard_writer* self,
                 uint64_t offset,
                 const void* beg,
                 const void* end)
{
  struct collecting_shard_writer* w = (struct collecting_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > w->capacity) {
    log_error("collecting_write: overflow shard %lu",
              (unsigned long)w->shard_index);
    return 1;
  }
  memcpy(w->buf + offset, beg, nbytes);
  if (offset + nbytes > w->size)
    w->size = offset + nbytes;
  return 0;
}

static int
collecting_finalize(struct shard_writer* self)
{
  struct collecting_shard_writer* w = (struct collecting_shard_writer*)self;
  w->finalized = 1;
  return 0;
}

static struct shard_writer*
collecting_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct collecting_shard_sink* s = (struct collecting_shard_sink*)self;
  if ((int)shard_index >= s->num_shards) {
    log_error("collecting_open: shard_index %lu >= num_shards %d",
              (unsigned long)shard_index,
              s->num_shards);
    return NULL;
  }
  struct collecting_shard_writer* w = &s->writers[shard_index];
  w->shard_index = shard_index;
  w->finalized = 0;
  w->size = 0;
  return &w->base;
}

static void
collecting_sink_free(struct collecting_shard_sink* s);

static int
collecting_sink_init(struct collecting_shard_sink* s,
                     int num_shards,
                     size_t per_shard_capacity)
{
  *s = (struct collecting_shard_sink){
    .base = { .open = collecting_open },
    .num_shards = num_shards,
    .per_shard_capacity = per_shard_capacity,
  };
  for (int i = 0; i < num_shards; ++i) {
    s->writers[i] = (struct collecting_shard_writer){
      .base = { .write = collecting_write, .finalize = collecting_finalize },
      .buf = (uint8_t*)calloc(1, per_shard_capacity),
      .capacity = per_shard_capacity,
    };
    if (!s->writers[i].buf) {
      collecting_sink_free(s);
      return 1;
    }
  }
  return 0;
}

static void
collecting_sink_free(struct collecting_shard_sink* s)
{
  for (int i = 0; i < s->num_shards; ++i)
    free(s->writers[i].buf);
  *s = (struct collecting_shard_sink){ 0 };
}

// --- Test ---

static int
test_shard_contents(void)
{
  log_info("=== test_shard_contents ===");

  // Dimensions
  const int size[3] = { 12, 8, 12 };
  const int tile_size[3] = { 2, 4, 3 };
  const int tiles_per_shard[3] = { 3, 2, 2 };

  const int tile_count[3] = {
    size[0] / tile_size[0], // 6
    size[1] / tile_size[1], // 2
    size[2] / tile_size[2], // 4
  };
  const int shard_count[3] = {
    tile_count[0] / tiles_per_shard[0], // 2
    tile_count[1] / tiles_per_shard[1], // 1
    tile_count[2] / tiles_per_shard[2], // 2
  };

  const int total_elements = size[0] * size[1] * size[2]; // 1152
  const int num_shards = shard_count[0] * shard_count[1] * shard_count[2]; // 4
  const int tiles_per_shard_total =
    tiles_per_shard[0] * tiles_per_shard[1] * tiles_per_shard[2];         // 12
  const int voxels_per_tile = tile_size[0] * tile_size[1] * tile_size[2]; // 24

  log_info("  total_elements=%d  num_shards=%d  tiles_per_shard=%d  "
           "voxels_per_tile=%d",
           total_elements,
           num_shards,
           tiles_per_shard_total,
           voxels_per_tile);

  // Generate source data: raster-order u32 with encoded coordinates
  uint32_t* src = (uint32_t*)malloc(total_elements * sizeof(uint32_t));
  CHECK(Fail0, src);

  for (int x0 = 0; x0 < size[0]; ++x0) {
    for (int x1 = 0; x1 < size[1]; ++x1) {
      for (int x2 = 0; x2 < size[2]; ++x2) {
        int gi = x0 * size[1] * size[2] + x1 * size[2] + x2;
        int s0 = x0 / (tile_size[0] * tiles_per_shard[0]);
        int s1 = x1 / (tile_size[1] * tiles_per_shard[1]);
        int s2 = x2 / (tile_size[2] * tiles_per_shard[2]);
        int t0 = (x0 / tile_size[0]) % tiles_per_shard[0];
        int t1 = (x1 / tile_size[1]) % tiles_per_shard[1];
        int t2 = (x2 / tile_size[2]) % tiles_per_shard[2];
        int v0 = x0 % tile_size[0];
        int v1 = x1 % tile_size[1];
        int v2 = x2 % tile_size[2];
        src[gi] = encode_voxel(s0, s1, s2, t0, t1, t2, v0, v1, v2);
      }
    }
  }

  // Set up collecting shard sink
  struct collecting_shard_sink css;
  CHECK(Fail1, collecting_sink_init(&css, num_shards, 256 * 1024) == 0);

  // Configure stream
  const struct dimension dims[] = {
    { .size = 12, .tile_size = 2, .tiles_per_shard = 3, .storage_position = 0 },
    { .size = 8, .tile_size = 4, .tiles_per_shard = 2, .storage_position = 1 },
    { .size = 12, .tile_size = 3, .tiles_per_shard = 2, .storage_position = 2 },
  };

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = total_elements * sizeof(uint32_t),
    .bytes_per_element = sizeof(uint32_t),
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = &css.base,
  };

  struct tile_stream_gpu s;
  CHECK(Fail1, tile_stream_gpu_create(&config, &s) == 0);

  log_info("  tile_elements=%lu  tile_stride=%lu  tiles_per_epoch=%lu  "
           "epoch_elements=%lu",
           (unsigned long)s.layout.tile_elements,
           (unsigned long)s.layout.tile_stride,
           (unsigned long)s.layout.tiles_per_epoch,
           (unsigned long)s.layout.epoch_elements);

  CHECK(Fail2, s.layout.tile_elements == (uint64_t)voxels_per_tile);
  CHECK(Fail2, s.layout.tiles_per_epoch == 8);
  CHECK(Fail2, s.layout.epoch_elements == 192);

  // Feed all data
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(&s.writer, input);
    CHECK(Fail2, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(&s.writer);
    CHECK(Fail2, r.error == 0);
  }

  CHECK(Fail2, s.cursor == (uint64_t)total_elements);

  // Verify all shards were finalized
  for (int si = 0; si < num_shards; ++si) {
    CHECK(Fail2, css.writers[si].finalized);
    CHECK(Fail2, css.writers[si].size > 0);
  }

  // Verify each shard
  {
    const size_t tile_bytes = s.layout.tile_stride * sizeof(uint32_t);
    const size_t index_data_bytes =
      (size_t)tiles_per_shard_total * 2 * sizeof(uint64_t);
    const size_t index_total_bytes = index_data_bytes + 4;

    // shard_inner_count and tps_inner for flat → coordinate mapping
    const int shard_inner_count = shard_count[1] * shard_count[2];
    const int tps_inner = tiles_per_shard[1] * tiles_per_shard[2];

    int errors = 0;

    for (int i_shard = 0; i_shard < num_shards; ++i_shard) {
      struct collecting_shard_writer* w = &css.writers[i_shard];
      CHECK(Fail2, w->size > index_total_bytes);

      // Expected shard coordinates from flat index
      // flat = s0 * shard_inner_count + (s1 * shard_count[2] + s2)
      int exp_s0 = i_shard / shard_inner_count;
      int exp_s1 = (i_shard % shard_inner_count) / shard_count[2];
      int exp_s2 = (i_shard % shard_inner_count) % shard_count[2];

      // Parse index block from end of shard
      const uint8_t* index_ptr = w->buf + w->size - index_total_bytes;

      uint64_t tile_offsets[12], tile_nbytes[12];
      for (int i = 0; i < tiles_per_shard_total; ++i) {
        memcpy(&tile_offsets[i], index_ptr + i * 16, sizeof(uint64_t));
        memcpy(&tile_nbytes[i], index_ptr + i * 16 + 8, sizeof(uint64_t));
      }

      int tiles_verified = 0;
      for (int i_tile = 0; i_tile < tiles_per_shard_total; ++i_tile) {
        if (tile_nbytes[i_tile] == 0 ||
            tile_nbytes[i_tile] > ZSTD_compressBound(tile_bytes)) {
          log_error(
            "  shard %d tile %d: unexpected nbytes=%lu (expected 1..%zu)",
            i_shard,
            i_tile,
            (unsigned long)tile_nbytes[i_tile],
            ZSTD_compressBound(tile_bytes));
          errors++;
          continue;
        }

        // Expected tile-in-shard coordinates from slot
        // slot = t0 * tps_inner + (t1 * tiles_per_shard[2] + t2)
        int exp_t0 = i_tile / tps_inner;
        int exp_t1 = (i_tile % tps_inner) / tiles_per_shard[2];
        int exp_t2 = (i_tile % tps_inner) % tiles_per_shard[2];

        // Decompress tile
        const uint8_t* comp_data = w->buf + tile_offsets[i_tile];
        uint8_t* decomp = (uint8_t*)calloc(1, tile_bytes);
        CHECK(Fail2, decomp);

        size_t result =
          ZSTD_decompress(decomp, tile_bytes, comp_data, tile_nbytes[i_tile]);
        if (ZSTD_isError(result)) {
          log_error("  shard %d tile %d: ZSTD_decompress failed: %s",
                    i_shard,
                    i_tile,
                    ZSTD_getErrorName(result));
          free(decomp);
          errors++;
          continue;
        }
        if (result != tile_bytes) {
          log_error("  shard %d tile %d: decompressed size %zu, expected %zu",
                    i_shard,
                    i_tile,
                    result,
                    tile_bytes);
          free(decomp);
          errors++;
          continue;
        }

        const uint32_t* voxels = (const uint32_t*)decomp;

        for (int e = 0; e < voxels_per_tile; ++e) {
          uint32_t val = voxels[e];

          // Expected voxel-in-tile coordinates from position e
          // e = v0 * (tile_size[1] * tile_size[2]) + v1 * tile_size[2] + v2
          int exp_v0 = e / (tile_size[1] * tile_size[2]);
          int exp_v1 = (e / tile_size[2]) % tile_size[1];
          int exp_v2 = e % tile_size[2];

          uint32_t expected = encode_voxel(exp_s0,
                                           exp_s1,
                                           exp_s2,
                                           exp_t0,
                                           exp_t1,
                                           exp_t2,
                                           exp_v0,
                                           exp_v1,
                                           exp_v2);

          if (val != expected) {
            int got_s0 = (val >> 24) & 7;
            int got_s1 = (val >> 21) & 7;
            int got_s2 = (val >> 18) & 7;
            int got_t0 = (val >> 15) & 7;
            int got_t1 = (val >> 12) & 7;
            int got_t2 = (val >> 9) & 7;
            int got_v0 = (val >> 6) & 7;
            int got_v1 = (val >> 3) & 7;
            int got_v2 = val & 7;

            if (errors < 10) {
              log_error("  shard %d slot %d elem %d: "
                        "expected s(%d,%d,%d) t(%d,%d,%d) v(%d,%d,%d) "
                        "got s(%d,%d,%d) t(%d,%d,%d) v(%d,%d,%d)",
                        i_shard,
                        i_tile,
                        e,
                        exp_s0,
                        exp_s1,
                        exp_s2,
                        exp_t0,
                        exp_t1,
                        exp_t2,
                        exp_v0,
                        exp_v1,
                        exp_v2,
                        got_s0,
                        got_s1,
                        got_s2,
                        got_t0,
                        got_t1,
                        got_t2,
                        got_v0,
                        got_v1,
                        got_v2);
            }
            errors++;
          }
        }
        free(decomp);
        tiles_verified++;
      }

      if (tiles_verified != tiles_per_shard_total) {
        log_error("  shard %d: verified %d/%d tiles",
                  i_shard,
                  tiles_verified,
                  tiles_per_shard_total);
      }
    }

    if (errors > 0) {
      log_error("  %d total errors", errors);
      goto Fail2;
    }
  }

  tile_stream_gpu_destroy(&s);
  collecting_sink_free(&css);
  free(src);
  log_info("  PASS");
  return 0;

Fail2:
  tile_stream_gpu_destroy(&s);
Fail1:
  collecting_sink_free(&css);
Fail0:
  free(src);
  log_error("  FAIL");
  return 1;
}

// Software CRC32C (Castagnoli) — same algorithm as stream.c.
static uint32_t crc32c_test_table[256];
static int crc32c_test_table_ready;

static void
crc32c_test_init_table(void)
{
  if (crc32c_test_table_ready)
    return;
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78 & (0u - (crc & 1)));
    crc32c_test_table[i] = crc;
  }
  crc32c_test_table_ready = 1;
}

static uint32_t
crc32c_test(const void* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_test_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFF;
}

// Verify shard index structure: monotonic offsets, size sum, CRC.
// Tests with both even (all tiles present) and uneven (padded) cases.
static int
test_shard_index_structure(void)
{
  log_info("=== test_shard_index_structure ===");

  crc32c_test_init_table();

  // --- Case 1: Even tiling (all tiles fill shard evenly) ---
  {
    const int size[3] = { 12, 8, 12 };
    const int tile_size[3] = { 2, 4, 3 };
    const int tiles_per_shard[3] = { 3, 2, 2 };

    const int tile_count[3] = {
      size[0] / tile_size[0],
      size[1] / tile_size[1],
      size[2] / tile_size[2],
    };
    const int shard_count[3] = {
      tile_count[0] / tiles_per_shard[0],
      tile_count[1] / tiles_per_shard[1],
      tile_count[2] / tiles_per_shard[2],
    };

    const int total_elements = size[0] * size[1] * size[2];
    const int num_shards =
      shard_count[0] * shard_count[1] * shard_count[2];
    const int tiles_per_shard_total =
      tiles_per_shard[0] * tiles_per_shard[1] * tiles_per_shard[2];

    uint32_t* src = (uint32_t*)malloc(total_elements * sizeof(uint32_t));
    CHECK(Fail0, src);

    // Fill with coordinate-encoded data (same as test_shard_contents)
    for (int x0 = 0; x0 < size[0]; ++x0)
      for (int x1 = 0; x1 < size[1]; ++x1)
        for (int x2 = 0; x2 < size[2]; ++x2) {
          int gi = x0 * size[1] * size[2] + x1 * size[2] + x2;
          src[gi] = encode_voxel(
            x0 / (tile_size[0] * tiles_per_shard[0]),
            x1 / (tile_size[1] * tiles_per_shard[1]),
            x2 / (tile_size[2] * tiles_per_shard[2]),
            (x0 / tile_size[0]) % tiles_per_shard[0],
            (x1 / tile_size[1]) % tiles_per_shard[1],
            (x2 / tile_size[2]) % tiles_per_shard[2],
            x0 % tile_size[0],
            x1 % tile_size[1],
            x2 % tile_size[2]);
        }

    struct collecting_shard_sink css;
    CHECK(Fail1, collecting_sink_init(&css, num_shards, 256 * 1024) == 0);

    const struct dimension dims[] = {
      { .size = 12, .tile_size = 2, .tiles_per_shard = 3, .storage_position = 0 },
      { .size = 8, .tile_size = 4, .tiles_per_shard = 2, .storage_position = 1 },
      { .size = 12, .tile_size = 3, .tiles_per_shard = 2, .storage_position = 2 },
    };

    const struct tile_stream_configuration config = {
      .buffer_capacity_bytes = total_elements * sizeof(uint32_t),
      .bytes_per_element = sizeof(uint32_t),
      .rank = 3,
      .dimensions = dims,
      .codec = CODEC_ZSTD,
      .shard_sink = &css.base,
    };

    struct tile_stream_gpu s;
    CHECK(Fail2, tile_stream_gpu_create(&config, &s) == 0);

    {
      struct slice input = { .beg = src, .end = src + total_elements };
      struct writer_result r = writer_append(&s.writer, input);
      CHECK(Fail3, r.error == 0);
    }
    {
      struct writer_result r = writer_flush(&s.writer);
      CHECK(Fail3, r.error == 0);
    }

    // Verify index structure for each shard
    const size_t index_data_bytes =
      (size_t)tiles_per_shard_total * 2 * sizeof(uint64_t);
    const size_t index_total_bytes = index_data_bytes + 4;
    int errors = 0;

    for (int si = 0; si < num_shards; ++si) {
      struct collecting_shard_writer* w = &css.writers[si];
      CHECK(Fail3, w->finalized);
      CHECK(Fail3, w->size > index_total_bytes);

      const uint8_t* index_ptr = w->buf + w->size - index_total_bytes;

      uint64_t tile_offsets[12], tile_nbytes[12];
      for (int i = 0; i < tiles_per_shard_total; ++i) {
        memcpy(&tile_offsets[i], index_ptr + i * 16, sizeof(uint64_t));
        memcpy(&tile_nbytes[i], index_ptr + i * 16 + 8, sizeof(uint64_t));
      }

      // 1. Offsets must be monotonically non-decreasing
      for (int i = 1; i < tiles_per_shard_total; ++i) {
        if (tile_offsets[i] < tile_offsets[i - 1]) {
          log_error("  shard %d: offset[%d]=%lu < offset[%d]=%lu",
                    si,
                    i,
                    (unsigned long)tile_offsets[i],
                    i - 1,
                    (unsigned long)tile_offsets[i - 1]);
          errors++;
        }
      }

      // 2. Sum of tile sizes + index block + CRC = shard size
      size_t tile_data_sum = 0;
      for (int i = 0; i < tiles_per_shard_total; ++i)
        tile_data_sum += tile_nbytes[i];

      if (tile_data_sum + index_total_bytes != w->size) {
        log_error(
          "  shard %d: tile_data_sum=%zu + index=%zu != shard_size=%zu",
          si,
          tile_data_sum,
          index_total_bytes,
          w->size);
        errors++;
      }

      // 3. Verify CRC32C at end of shard matches computed CRC of index data
      uint32_t stored_crc;
      memcpy(&stored_crc, index_ptr + index_data_bytes, 4);
      uint32_t computed_crc = crc32c_test(index_ptr, index_data_bytes);
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
    tile_stream_gpu_destroy(&s);
    collecting_sink_free(&css);
    free(src);
    goto Case2;

  Fail3:
    tile_stream_gpu_destroy(&s);
  Fail2:
    collecting_sink_free(&css);
  Fail1:
    free(src);
  Fail0:
    log_error("  FAIL");
    return 1;
  }

Case2:
  // --- Case 2: Single shard (u16 data, smaller shape) ---
  {
    const struct dimension dims2[] = {
      { .size = 4, .tile_size = 2, .storage_position = 0 },
      { .size = 4, .tile_size = 2, .storage_position = 1 },
      { .size = 6, .tile_size = 3, .storage_position = 2 },
    };
    const size_t tiles_per_shard_total2 = 8; // 2*2*2
    const int total2 = 96;

    struct collecting_shard_sink css2;
    CHECK(FailB0, collecting_sink_init(&css2, 1, 256 * 1024) == 0);

    const struct tile_stream_configuration config2 = {
      .buffer_capacity_bytes = total2 * sizeof(uint16_t),
      .bytes_per_element = sizeof(uint16_t),
      .rank = 3,
      .dimensions = dims2,
      .codec = CODEC_ZSTD,
      .shard_sink = &css2.base,
    };

    struct tile_stream_gpu s2;
    CHECK(FailB1, tile_stream_gpu_create(&config2, &s2) == 0);

    uint16_t src2[96];
    for (int i = 0; i < 96; ++i)
      src2[i] = (uint16_t)i;

    {
      struct slice input = { .beg = src2, .end = src2 + total2 };
      struct writer_result r = writer_append(&s2.writer, input);
      CHECK(FailB2, r.error == 0);
    }
    {
      struct writer_result r = writer_flush(&s2.writer);
      CHECK(FailB2, r.error == 0);
    }

    struct collecting_shard_writer* w = &css2.writers[0];
    CHECK(FailB2, w->finalized);

    const size_t index_data_bytes2 =
      tiles_per_shard_total2 * 2 * sizeof(uint64_t);
    const size_t index_total_bytes2 = index_data_bytes2 + 4;
    CHECK(FailB2, w->size > index_total_bytes2);

    const uint8_t* index_ptr2 = w->buf + w->size - index_total_bytes2;
    uint64_t offs[8], sizes[8];
    for (size_t i = 0; i < tiles_per_shard_total2; ++i) {
      memcpy(&offs[i], index_ptr2 + i * 16, sizeof(uint64_t));
      memcpy(&sizes[i], index_ptr2 + i * 16 + 8, sizeof(uint64_t));
    }

    // Monotonic offsets
    for (size_t i = 1; i < tiles_per_shard_total2; ++i)
      CHECK(FailB2, offs[i] >= offs[i - 1]);

    // Size sum
    size_t sum = 0;
    for (size_t i = 0; i < tiles_per_shard_total2; ++i)
      sum += sizes[i];
    CHECK(FailB2, sum + index_total_bytes2 == w->size);

    // CRC
    uint32_t stored, computed;
    memcpy(&stored, index_ptr2 + index_data_bytes2, 4);
    computed = crc32c_test(index_ptr2, index_data_bytes2);
    CHECK(FailB2, stored == computed);

    log_info("  single shard (u16): verified");
    tile_stream_gpu_destroy(&s2);
    collecting_sink_free(&css2);
    goto Done;

  FailB2:
    tile_stream_gpu_destroy(&s2);
  FailB1:
    collecting_sink_free(&css2);
  FailB0:
    log_error("  FAIL");
    return 1;
  }

Done:
  log_info("  PASS");
  return 0;
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

  ecode |= test_shard_contents();
  ecode |= test_shard_index_structure();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}

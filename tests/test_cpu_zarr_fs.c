// Test tile_stream_cpu + zarr_fs_sink integration.
// Exercises the write_direct -> io_queue async path that requires fencing.

#include "stream.cpu.h"
#include "stream/layouts.h"
#include "test_platform.h"
#include "test_shard_verify.h"
#include "util/prelude.h"
#include "zarr_fs_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

static int
read_file_all(const char* path, uint8_t** out, size_t* out_len)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  *out = (uint8_t*)malloc((size_t)len + 1);
  if (!*out) {
    fclose(f);
    return -1;
  }
  size_t rd = fread(*out, 1, (size_t)len, f);
  fclose(f);
  if ((long)rd != len) {
    free(*out);
    *out = NULL;
    return -1;
  }
  *out_len = (size_t)len;
  return 0;
}

// Verify a shard file: parse index, decompress each present chunk.
// Returns number of valid (non-empty) chunks, or -1 on error.
static int
verify_shard(const char* path,
             size_t chunks_per_shard,
             size_t chunk_stride_bytes)
{
  uint8_t* shard_data;
  size_t shard_len;
  if (read_file_all(path, &shard_data, &shard_len))
    return -1;

  uint64_t* offsets = (uint64_t*)malloc(chunks_per_shard * sizeof(uint64_t));
  uint64_t* nbytes = (uint64_t*)malloc(chunks_per_shard * sizeof(uint64_t));
  if (!offsets || !nbytes) {
    free(offsets);
    free(nbytes);
    free(shard_data);
    return -1;
  }

  if (shard_index_parse(shard_data, shard_len, chunks_per_shard, offsets,
                        nbytes)) {
    log_error("  shard_index_parse failed for %s", path);
    free(offsets);
    free(nbytes);
    free(shard_data);
    return -1;
  }

  int valid = 0;
  int errors = 0;
  for (size_t i = 0; i < chunks_per_shard; ++i) {
    if (nbytes[i] == (uint64_t)-1)
      continue; // empty slot (partial shard)
    if (nbytes[i] == 0) {
      log_error("  %s chunk %zu: zero-length data", path, i);
      errors++;
      continue;
    }
    uint8_t* decomp = (uint8_t*)calloc(1, chunk_stride_bytes);
    if (!decomp) {
      errors++;
      continue;
    }
    if (chunk_decompress(shard_data + offsets[i], (size_t)nbytes[i], decomp,
                         chunk_stride_bytes)) {
      log_error("  %s chunk %zu: decompress failed", path, i);
      errors++;
    } else {
      valid++;
    }
    free(decomp);
  }

  free(offsets);
  free(nbytes);
  free(shard_data);
  return errors > 0 ? -1 : valid;
}

// ---- Tests ----

// Full pipeline: tile_stream_cpu -> zarr_fs_sink -> verify on disk.
// Feeds all data at once (single append call).
static int
test_pipeline(const char* tmpdir)
{
  log_info("=== test_cpu_zarr_pipeline ===");

  const int inner_size[2] = { 8, 12 };
  const int n_epochs = 6;
  const int chunks_per_shard_append = 3;
  const int epoch_elements = inner_size[0] * inner_size[1];
  const int total_elements = n_epochs * epoch_elements;
  const int chunks_per_shard_total = 3 * 2 * 2; // t * y * x
  const int chunks_per_epoch = (inner_size[0] / 4) * (inner_size[1] / 3);
  const int num_shards_inner = 1 * 2; // y: 2/2=1, x: 4/2=2

  uint16_t* src = (uint16_t*)malloc((size_t)total_elements * sizeof(uint16_t));
  CHECK(Fail, src);
  for (int i = 0; i < total_elements; ++i)
    src[i] = (uint16_t)(i & 0xFFFF);

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = chunks_per_shard_append,
      .name = "t",
      .storage_position = 0 },
    { .size = inner_size[0],
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = inner_size[1],
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail2, zs);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct tile_stream_cpu* s =
    tile_stream_cpu_create(&config, zarr_fs_sink_as_shard_sink(zs));
  CHECK(Fail3, s);

  size_t chunk_stride_bytes =
    tile_stream_cpu_layout(s)->chunk_stride * sizeof(uint16_t);

  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_cpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(tile_stream_cpu_writer(s));
    CHECK(Fail4, r.error == 0);
  }

  zarr_fs_sink_flush(zs);

  // Verify: correct number of shards, all chunks decompress.
  {
    int shard_count_append = ceildiv(n_epochs, chunks_per_shard_append);
    int total_valid_chunks = 0;

    for (int sa = 0; sa < shard_count_append; ++sa) {
      for (int si = 0; si < num_shards_inner; ++si) {
        int sy = si / 2; // shard_count_x = 2
        int sx = si % 2;
        char path[4096];
        snprintf(path, sizeof(path), "%s/0/c/%d/%d/%d", tmpdir, sa, sy, sx);
        CHECK(Fail4, test_file_exists(path));

        int valid = verify_shard(path, (size_t)chunks_per_shard_total,
                                 chunk_stride_bytes);
        CHECK(Fail4, valid >= 0);
        total_valid_chunks += valid;
      }
    }

    int expected_chunks = n_epochs * chunks_per_epoch;
    if (total_valid_chunks != expected_chunks) {
      log_error("  expected %d valid chunks, got %d", expected_chunks,
                total_valid_chunks);
      goto Fail4;
    }
  }

  tile_stream_cpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  free(src);
  log_info("  PASS");
  return 0;

Fail4:
  tile_stream_cpu_destroy(s);
Fail3:
  zarr_fs_sink_destroy(zs);
Fail2:
  free(src);
Fail:
  log_error("  FAIL");
  return 1;
}

// Streaming append: feed data one epoch at a time to exercise multi-batch
// fencing (aggregate buffer reuse across batches).
static int
test_streaming_append(const char* tmpdir)
{
  log_info("=== test_cpu_zarr_streaming ===");

  const int inner_size[2] = { 8, 12 };
  const int n_epochs = 12; // enough to cross multiple shard boundaries
  const int chunks_per_shard_append = 3;
  const int epoch_elements = inner_size[0] * inner_size[1];
  const int total_elements = n_epochs * epoch_elements;
  const int chunks_per_shard_total = 3 * 2 * 2;
  const int chunks_per_epoch = (inner_size[0] / 4) * (inner_size[1] / 3);
  const int num_shards_inner = 1 * 2;

  uint16_t* src = (uint16_t*)malloc((size_t)total_elements * sizeof(uint16_t));
  CHECK(Fail, src);
  for (int i = 0; i < total_elements; ++i)
    src[i] = (uint16_t)(i & 0xFFFF);

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = chunks_per_shard_append,
      .name = "t",
      .storage_position = 0 },
    { .size = inner_size[0],
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = inner_size[1],
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail2, zs);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct tile_stream_cpu* s =
    tile_stream_cpu_create(&config, zarr_fs_sink_as_shard_sink(zs));
  CHECK(Fail3, s);

  size_t chunk_stride_bytes =
    tile_stream_cpu_layout(s)->chunk_stride * sizeof(uint16_t);

  // Feed one epoch at a time
  for (int t = 0; t < n_epochs; ++t) {
    const uint16_t* epoch = src + t * epoch_elements;
    struct slice input = { .beg = epoch, .end = epoch + epoch_elements };
    struct writer_result r = writer_append(tile_stream_cpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(tile_stream_cpu_writer(s));
    CHECK(Fail4, r.error == 0);
  }

  zarr_fs_sink_flush(zs);

  // Verify all shards
  {
    int shard_count_append = ceildiv(n_epochs, chunks_per_shard_append);
    int total_valid_chunks = 0;

    for (int sa = 0; sa < shard_count_append; ++sa) {
      for (int si = 0; si < num_shards_inner; ++si) {
        int sy = si / 2;
        int sx = si % 2;
        char path[4096];
        snprintf(path, sizeof(path), "%s/0/c/%d/%d/%d", tmpdir, sa, sy, sx);
        CHECK(Fail4, test_file_exists(path));

        int valid = verify_shard(path, (size_t)chunks_per_shard_total,
                                 chunk_stride_bytes);
        CHECK(Fail4, valid >= 0);
        total_valid_chunks += valid;
      }
    }

    int expected_chunks = n_epochs * chunks_per_epoch;
    if (total_valid_chunks != expected_chunks) {
      log_error("  expected %d valid chunks, got %d", expected_chunks,
                total_valid_chunks);
      goto Fail4;
    }
  }

  tile_stream_cpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  free(src);
  log_info("  PASS");
  return 0;

Fail4:
  tile_stream_cpu_destroy(s);
Fail3:
  zarr_fs_sink_destroy(zs);
Fail2:
  free(src);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  int err = 0;

  char tmpdir1[4096], tmpdir2[4096];
  CHECK(Fail, test_tmpdir_create(tmpdir1, sizeof(tmpdir1)) == 0);
  CHECK(Fail, test_tmpdir_create(tmpdir2, sizeof(tmpdir2)) == 0);

  err |= test_pipeline(tmpdir1);
  err |= test_streaming_append(tmpdir2);

  test_tmpdir_remove(tmpdir1);
  test_tmpdir_remove(tmpdir2);
  return err;

Fail:
  return 1;
}

#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_platform.h"
#include "zarr_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// --- Coordinate encoding (same as test_shard_contents) ---
// 3D: dim0=12, dim1=8, dim2=12
// tile: 2, 4, 3. tps: 3, 2, 2.

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

static int
read_file_all(const char* path, uint8_t** out, size_t* out_len)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  *out = (uint8_t*)malloc((size_t)len);
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

// --- Test: metadata files ---

static int
test_metadata(const char* tmpdir)
{
  log_info("=== test_metadata ===");

  struct dimension dims[] = {
    { .size = 12, .tile_size = 2, .tiles_per_shard = 3, .name = "z" },
    { .size = 8, .tile_size = 4, .tiles_per_shard = 2, .name = "y" },
    { .size = 12, .tile_size = 3, .tiles_per_shard = 2, .name = "x" },
  };

  struct zarr_config cfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = zarr_dtype_uint32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
  };

  struct zarr_sink* zs = zarr_sink_create(&cfg);
  CHECK(Fail, zs);

  // Check root zarr.json
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    CHECK(Fail2, strstr((char*)data, "\"zarr_format\":3"));
    CHECK(Fail2, strstr((char*)data, "\"node_type\":\"group\""));
    free(data);
  }

  // Check array zarr.json
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';

    CHECK(Fail2, strstr((char*)data, "\"zarr_format\":3"));
    CHECK(Fail2, strstr((char*)data, "\"node_type\":\"array\""));
    CHECK(Fail2, strstr((char*)data, "\"data_type\":\"uint32\""));
    CHECK(Fail2, strstr((char*)data, "\"shape\":[12,8,12]"));
    // chunk_shape (shard shape) = tile_size * tiles_per_shard = [6,8,6]
    CHECK(Fail2, strstr((char*)data, "\"chunk_shape\":[6,8,6]"));
    CHECK(Fail2, strstr((char*)data, "\"sharding_indexed\""));
    CHECK(Fail2,
          strstr((char*)data, "\"dimension_names\":[\"z\",\"y\",\"x\"]"));
    free(data);
  }

  zarr_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_sink_destroy(zs);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: full pipeline → zarr → verify ---

static int
test_pipeline(const char* tmpdir)
{
  log_info("=== test_pipeline ===");

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
  const int num_shards = shard_count[0] * shard_count[1] * shard_count[2];
  const int tiles_per_shard_total =
    tiles_per_shard[0] * tiles_per_shard[1] * tiles_per_shard[2];
  const int voxels_per_tile = tile_size[0] * tile_size[1] * tile_size[2];

  uint32_t* src = NULL;

  // Generate source data
  src = (uint32_t*)malloc((size_t)total_elements * sizeof(uint32_t));
  CHECK(Fail, src);

  for (int x0 = 0; x0 < size[0]; ++x0)
    for (int x1 = 0; x1 < size[1]; ++x1)
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

  // Create zarr sink
  const struct dimension dims[] = {
    { .size = 12, .tile_size = 2, .tiles_per_shard = 3, .name = "z" },
    { .size = 8, .tile_size = 4, .tiles_per_shard = 2, .name = "y" },
    { .size = 12, .tile_size = 3, .tiles_per_shard = 2, .name = "x" },
  };

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = zarr_dtype_uint32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
  };

  struct zarr_sink* zs = zarr_sink_create(&zcfg);
  CHECK(Fail2, zs);

  // Configure tile stream
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint32_t),
    .bytes_per_element = sizeof(uint32_t),
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = zarr_sink_as_shard_sink(zs),
  };

  struct tile_stream_gpu s;
  CHECK(Fail3, tile_stream_gpu_create(&config, &s));

  // Feed data
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(&s.writer, input);
    CHECK(Fail4, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(&s.writer);
    CHECK(Fail4, r.error == 0);
  }

  zarr_sink_flush(zs);

  // Verify shard files exist and contents are correct
  {
    const int tps_inner = tiles_per_shard[1] * tiles_per_shard[2];
    int errors = 0;

    for (int i_shard = 0; i_shard < num_shards; ++i_shard) {
      // Compute shard coordinates
      int sc[3];
      {
        int rem = i_shard;
        for (int d = 2; d >= 0; --d) {
          sc[d] = rem % shard_count[d];
          rem /= shard_count[d];
        }
      }

      char path[4096];
      snprintf(
        path, sizeof(path), "%s/0/c/%d/%d/%d", tmpdir, sc[0], sc[1], sc[2]);
      CHECK(Fail4, test_file_exists(path));

      uint8_t* shard_data;
      size_t shard_len;
      CHECK(Fail4, read_file_all(path, &shard_data, &shard_len) == 0);

      // Parse index from end: tiles_per_shard_total * 2 * 8 bytes + 4 crc
      size_t index_data_bytes =
        (size_t)tiles_per_shard_total * 2 * sizeof(uint64_t);
      size_t index_total_bytes = index_data_bytes + 4;
      CHECK(Fail4, shard_len > index_total_bytes);

      const uint8_t* index_ptr = shard_data + shard_len - index_total_bytes;

      uint64_t tile_offsets[12], tile_nbytes[12];
      for (int i = 0; i < tiles_per_shard_total; ++i) {
        memcpy(&tile_offsets[i], index_ptr + (size_t)i * 16, sizeof(uint64_t));
        memcpy(
          &tile_nbytes[i], index_ptr + (size_t)i * 16 + 8, sizeof(uint64_t));
      }

      // Decompress and verify each tile
      // tile_stride may be padded; use actual tile_stride for decompression
      size_t tile_stride_bytes = s.layout.tile_stride * sizeof(uint32_t);

      for (int i_tile = 0; i_tile < tiles_per_shard_total; ++i_tile) {
        if (tile_nbytes[i_tile] == 0 ||
            tile_nbytes[i_tile] > ZSTD_compressBound(tile_stride_bytes)) {
          log_error("shard %d tile %d: bad nbytes=%llu",
                    i_shard,
                    i_tile,
                    (unsigned long long)tile_nbytes[i_tile]);
          errors++;
          continue;
        }

        uint8_t* decomp = (uint8_t*)calloc(1, tile_stride_bytes);
        CHECK(Fail4, decomp);

        size_t result = ZSTD_decompress(decomp,
                                        tile_stride_bytes,
                                        shard_data + tile_offsets[i_tile],
                                        (size_t)tile_nbytes[i_tile]);
        if (ZSTD_isError(result)) {
          log_error("shard %d tile %d: ZSTD error: %s",
                    i_shard,
                    i_tile,
                    ZSTD_getErrorName(result));
          free(decomp);
          errors++;
          continue;
        }

        // Tile-in-shard coordinates
        int exp_t0 = i_tile / tps_inner;
        int exp_t1 = (i_tile % tps_inner) / tiles_per_shard[2];
        int exp_t2 = (i_tile % tps_inner) % tiles_per_shard[2];

        const uint32_t* voxels = (const uint32_t*)decomp;
        for (int e = 0; e < voxels_per_tile; ++e) {
          int exp_v0 = e / (tile_size[1] * tile_size[2]);
          int exp_v1 = (e / tile_size[2]) % tile_size[1];
          int exp_v2 = e % tile_size[2];

          uint32_t expected = encode_voxel(sc[0],
                                           sc[1],
                                           sc[2],
                                           exp_t0,
                                           exp_t1,
                                           exp_t2,
                                           exp_v0,
                                           exp_v1,
                                           exp_v2);

          if (voxels[e] != expected) {
            if (errors < 10) {
              log_error("shard %d tile %d elem %d: got 0x%08x expected 0x%08x",
                        i_shard,
                        i_tile,
                        e,
                        voxels[e],
                        expected);
            }
            errors++;
          }
        }
        free(decomp);
      }
      free(shard_data);
    }

    if (errors > 0) {
      log_error("  %d total errors", errors);
      goto Fail4;
    }
  }

  tile_stream_gpu_destroy(&s);
  zarr_sink_destroy(zs);
  free(src);
  log_info("  PASS");
  return 0;

Fail4:
  tile_stream_gpu_destroy(&s);
Fail3:
  zarr_sink_destroy(zs);
Fail2:
  free(src);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: multiscale metadata ---

static int
test_multiscale_metadata(const char* tmpdir)
{
  log_info("=== test_multiscale_metadata ===");

  struct dimension dims[] = {
    { .size = 64, .tile_size = 8, .tiles_per_shard = 4, .name = "z", .downsample = 1 },
    { .size = 32, .tile_size = 8, .tiles_per_shard = 2, .name = "y" },
    { .size = 64, .tile_size = 8, .tiles_per_shard = 4, .name = "x", .downsample = 1 },
  };

  struct zarr_multiscale_config cfg = {
    .store_path = tmpdir,
    .data_type = zarr_dtype_uint16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .nlod = 0, // auto
  };

  struct zarr_multiscale_sink* ms = zarr_multiscale_sink_create(&cfg);
  CHECK(Fail, ms);

  // Check root zarr.json has multiscales attribute
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 8191 ? len : 8191] = '\0';

    CHECK(Fail2, strstr((char*)data, "\"zarr_format\":3"));
    CHECK(Fail2, strstr((char*)data, "\"node_type\":\"group\""));
    CHECK(Fail2, strstr((char*)data, "\"ome\""));
    CHECK(Fail2, strstr((char*)data, "\"multiscales\""));
    CHECK(Fail2, strstr((char*)data, "\"version\":\"0.5\""));
    CHECK(Fail2, strstr((char*)data, "\"path\":\"0\""));
    CHECK(Fail2, strstr((char*)data, "\"path\":\"1\""));
    CHECK(Fail2, strstr((char*)data, "\"coordinateTransformations\""));
    free(data);
  }

  // Check L0 array zarr.json
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';

    CHECK(Fail2, strstr((char*)data, "\"shape\":[64,32,64]"));
    CHECK(Fail2, strstr((char*)data, "\"data_type\":\"uint16\""));
    free(data);
  }

  // Check L1 array zarr.json (dims 0 and 2 halved)
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/1/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';

    CHECK(Fail2, strstr((char*)data, "\"shape\":[32,32,32]"));
    free(data);
  }

  zarr_multiscale_sink_destroy(ms);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_multiscale_sink_destroy(ms);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;

  // Create temp directory
  char tmpdir[4096];
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  log_info("temp dir: %s", tmpdir);

  // Metadata tests (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/meta", tmpdir);
    test_mkdir(sub);
    ecode |= test_metadata(sub);
  }

  // Multiscale metadata test (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/msmeta", tmpdir);
    test_mkdir(sub);
    ecode |= test_multiscale_metadata(sub);
  }

  // Pipeline tests (need CUDA)
  {
    CUcontext ctx = 0;
    CUdevice dev;

    CU(Cleanup, cuInit(0));
    CU(Cleanup, cuDeviceGet(&dev, 0));
    CU(Cleanup, cuCtxCreate(&ctx, 0, dev));

    {
      char sub[4200];
      snprintf(sub, sizeof(sub), "%s/pipe", tmpdir);
      test_mkdir(sub);
      ecode |= test_pipeline(sub);
    }

    cuCtxDestroy(ctx);
  }

Cleanup:
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}

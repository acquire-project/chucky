#include "gpu/prelude.cuda.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "test_platform.h"
#include "test_shard_verify.h"
#include "test_voxel_encode.h"
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

// Read a group zarr.json at `dir`/zarr.json and validate common fields.
// On success returns 0 and sets *out (caller must free).
// On failure returns non-zero.
static int
check_group_zarr_json(const char* dir, char** out, size_t bufsz)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", dir);
  if (!test_file_exists(path))
    return 1;

  uint8_t* data;
  size_t len;
  if (read_file_all(path, &data, &len))
    return 1;
  data[len < bufsz - 1 ? len : bufsz - 1] = '\0';

  int ok = strstr((char*)data, "\"zarr_format\":3") &&
           strstr((char*)data, "\"node_type\":\"group\"") &&
           strstr((char*)data, "\"consolidated_metadata\":null");
  if (!ok) {
    free(data);
    return 1;
  }
  *out = (char*)data;
  return 0;
}

// --- Test: metadata files ---

static int
test_metadata(const char* tmpdir)
{
  log_info("=== test_metadata ===");

  struct dimension dims[] = {
    { .size = 12,
      .chunk_size = 2,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 12,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config cfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&cfg);
  CHECK(Fail, zs);

  {
    char* data;
    CHECK(Fail2, check_group_zarr_json(tmpdir, &data, 4096) == 0);
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
    // chunk_shape (shard shape) = chunk_size * chunks_per_shard = [6,8,6]
    CHECK(Fail2, strstr((char*)data, "\"chunk_shape\":[6,8,6]"));
    CHECK(Fail2, strstr((char*)data, "\"sharding_indexed\""));
    CHECK(Fail2,
          strstr((char*)data, "\"dimension_names\":[\"z\",\"y\",\"x\"]"));
    free(data);
  }

  zarr_fs_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_fs_sink_destroy(zs);
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
  const int chunk_size[3] = { 2, 4, 3 };
  const int chunks_per_shard[3] = { 3, 2, 2 };

  const int chunk_count[3] = {
    size[0] / chunk_size[0],
    size[1] / chunk_size[1],
    size[2] / chunk_size[2],
  };
  const int shard_count[3] = {
    chunk_count[0] / chunks_per_shard[0],
    chunk_count[1] / chunks_per_shard[1],
    chunk_count[2] / chunks_per_shard[2],
  };

  const int total_elements = size[0] * size[1] * size[2];
  const int num_shards = shard_count[0] * shard_count[1] * shard_count[2];
  const int chunks_per_shard_total =
    chunks_per_shard[0] * chunks_per_shard[1] * chunks_per_shard[2];
  const int voxels_per_chunk = chunk_size[0] * chunk_size[1] * chunk_size[2];

  uint32_t* src = generate_encoded_volume(size, chunk_size, chunks_per_shard);
  CHECK(Fail, src);

  // Create zarr sink
  struct dimension dims[] = {
    { .size = 12,
      .chunk_size = 2,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 12,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail2, zs);

  // Configure stream
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail3,
        (s = tile_stream_gpu_create(&config, zarr_fs_sink_as_shard_sink(zs))) !=
          NULL);

  // Feed data
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    CHECK(Fail4, r.error == 0);
  }

  zarr_fs_sink_flush(zs);

  // Verify shard files exist and contents are correct
  {
    const int cps_inner = chunks_per_shard[1] * chunks_per_shard[2];
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

      // Parse shard index
      uint64_t chunk_offsets[12], chunk_nbytes[12];
      CHECK(Fail4,
            shard_index_parse(shard_data,
                              shard_len,
                              (size_t)chunks_per_shard_total,
                              chunk_offsets,
                              chunk_nbytes) == 0);

      // Decompress and verify each chunk
      // chunk_stride may be padded; use actual chunk_stride for decompression
      size_t chunk_stride_bytes =
        tile_stream_gpu_layout(s)->chunk_stride * sizeof(uint32_t);

      for (int i_chunk = 0; i_chunk < chunks_per_shard_total; ++i_chunk) {
        if (chunk_nbytes[i_chunk] == 0 ||
            chunk_nbytes[i_chunk] > ZSTD_compressBound(chunk_stride_bytes)) {
          log_error("shard %d chunk %d: bad nbytes=%llu",
                    i_shard,
                    i_chunk,
                    (unsigned long long)chunk_nbytes[i_chunk]);
          errors++;
          continue;
        }

        uint8_t* decomp = (uint8_t*)calloc(1, chunk_stride_bytes);
        CHECK(Fail4, decomp);

        size_t result = ZSTD_decompress(decomp,
                                        chunk_stride_bytes,
                                        shard_data + chunk_offsets[i_chunk],
                                        (size_t)chunk_nbytes[i_chunk]);
        if (ZSTD_isError(result)) {
          log_error("shard %d chunk %d: ZSTD error: %s",
                    i_shard,
                    i_chunk,
                    ZSTD_getErrorName(result));
          free(decomp);
          errors++;
          continue;
        }

        // Tile-in-shard coordinates
        int exp_t0 = i_chunk / cps_inner;
        int exp_t1 = (i_chunk % cps_inner) / chunks_per_shard[2];
        int exp_t2 = (i_chunk % cps_inner) % chunks_per_shard[2];

        const uint32_t* voxels = (const uint32_t*)decomp;
        for (int e = 0; e < voxels_per_chunk; ++e) {
          int exp_v0 = e / (chunk_size[1] * chunk_size[2]);
          int exp_v1 = (e / chunk_size[2]) % chunk_size[1];
          int exp_v2 = e % chunk_size[2];

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
              log_error("shard %d chunk %d elem %d: got 0x%08x expected 0x%08x",
                        i_shard,
                        i_chunk,
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

  tile_stream_gpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  free(src);
  log_info("  PASS");
  return 0;

Fail4:
  tile_stream_gpu_destroy(s);
Fail3:
  zarr_fs_sink_destroy(zs);
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
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "z",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "x",
      .downsample = 1,
      .storage_position = 2 },
  };

  struct zarr_multiscale_config cfg = {
    .store_path = tmpdir,
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .nlod = 0, // auto
  };

  struct zarr_fs_multiscale_sink* ms = zarr_fs_multiscale_sink_create(&cfg);
  CHECK(Fail, ms);

  // Check root zarr.json has multiscales attribute
  {
    char* data;
    CHECK(Fail2, check_group_zarr_json(tmpdir, &data, 8192) == 0);
    CHECK(Fail2, strstr(data, "\"ome\""));
    CHECK(Fail2, strstr(data, "\"multiscales\""));
    CHECK(Fail2, strstr(data, "\"version\":\"0.5\""));
    CHECK(Fail2, strstr(data, "\"path\":\"0\""));
    CHECK(Fail2, strstr(data, "\"path\":\"1\""));
    CHECK(Fail2, strstr(data, "\"coordinateTransformations\""));
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

  // Check L1 array zarr.json (dim 2 halved; dim 0 excluded from LOD mask)
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/1/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';

    CHECK(Fail2, strstr((char*)data, "\"shape\":[64,32,32]"));
    free(data);
  }

  zarr_fs_multiscale_sink_destroy(ms);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_fs_multiscale_sink_destroy(ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: non-power-of-2 sizes get pow(2,level) scale, not shape ratio ---

static int
test_multiscale_scale_non_pow2(const char* tmpdir)
{
  log_info("=== test_multiscale_scale_non_pow2 ===");

  // x=6 (non-power-of-2): L0 x=6, L1 x=3, L2 x=2.
  // Old code gave scale ratio 6/3=2, 6/2=3. Correct: 2, 4.
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .name = "z",
      .storage_position = 0 },
    { .size = 6,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "y",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "x",
      .downsample = 1,
      .storage_position = 2 },
  };

  struct zarr_multiscale_config cfg = {
    .store_path = tmpdir,
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .nlod = 0,
  };

  struct zarr_fs_multiscale_sink* ms = zarr_fs_multiscale_sink_create(&cfg);
  CHECK(Fail, ms);

  {
    char* data;
    CHECK(Fail2, check_group_zarr_json(tmpdir, &data, 8192) == 0);

    // L0: scale [1,1,1]
    CHECK(Fail2, strstr(data, "\"scale\":[1.0,1.0,1.0]"));
    // L1: z=1, y=2, x=2
    CHECK(Fail2, strstr(data, "\"scale\":[1.0,2.0,2.0]"));
    // L2: z=1, y=4 (not 3!), x=4 (not 3!)
    CHECK(Fail2, strstr(data, "\"scale\":[1.0,4.0,4.0]"));

    free(data);
  }

  zarr_fs_multiscale_sink_destroy(ms);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_fs_multiscale_sink_destroy(ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: unit and scale in multiscale metadata ---

static int
test_multiscale_unit_scale(const char* tmpdir)
{
  log_info("=== test_multiscale_unit_scale ===");

  struct dimension dims[] = {
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "z",
      .downsample = 1,
      .storage_position = 0,
      .ngff = { .unit = "micrometer", .scale = 0.5 } },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1,
      .ngff = { .unit = "micrometer", .scale = 0.3 } },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "x",
      .downsample = 1,
      .storage_position = 2,
      .ngff = { .unit = NULL, .scale = 0.0 } }, // NULL unit → omitted
  };

  struct zarr_multiscale_config cfg = {
    .store_path = tmpdir,
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .nlod = 0,
  };

  struct zarr_fs_multiscale_sink* ms = zarr_fs_multiscale_sink_create(&cfg);
  CHECK(Fail, ms);

  {
    char* data;
    CHECK(Fail2, check_group_zarr_json(tmpdir, &data, 8192) == 0);

    // L0 scale: z=0.5*1=0.5, y=0.3*1=0.3, x=1.0*1=1.0
    CHECK(Fail2, strstr(data, "\"scale\":[0.5,0.3,1.0]"));

    free(data);
  }

  zarr_fs_multiscale_sink_destroy(ms);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_fs_multiscale_sink_destroy(ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: metadata with n_append=2 ---

static int
test_metadata_two_append(const char* tmpdir)
{
  log_info("=== test_metadata_two_append ===");

  // 4D: t=unbounded, z=4, y=64, x=64
  // chunk (1,1,32,32) → n_append=2 (t and z both have chunk_size=1)
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .name = "t",
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .name = "z",
      .storage_position = 1 },
    { .size = 64,
      .chunk_size = 32,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 2 },
    { .size = 64,
      .chunk_size = 32,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 3 },
  };

  struct zarr_config cfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 4,
    .dimensions = dims,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&cfg);
  CHECK(Fail, zs);

  {
    char* data;
    CHECK(Fail2, check_group_zarr_json(tmpdir, &data, 4096) == 0);
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
    CHECK(Fail2, strstr((char*)data, "\"data_type\":\"uint16\""));
    CHECK(Fail2, strstr((char*)data, "\"shape\":[0,4,64,64]"));
    // chunk_shape (shard shape) = chunk_size * chunks_per_shard = [4,4,64,64]
    CHECK(Fail2, strstr((char*)data, "\"chunk_shape\":[4,4,64,64]"));
    CHECK(Fail2, strstr((char*)data, "\"sharding_indexed\""));
    CHECK(Fail2,
          strstr((char*)data, "\"dimension_names\":[\"t\",\"z\",\"y\",\"x\"]"));
    free(data);
  }

  // Check that the shard chunk directory exists
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/c", tmpdir);
    CHECK(Fail2, test_file_exists(path));
  }

  zarr_fs_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_fs_sink_destroy(zs);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: unbounded dim0 metadata update ---

static int
test_unbounded_metadata_update(const char* tmpdir)
{
  log_info("=== test_unbounded_metadata_update ===");

  // dim0 unbounded (size=0), chunks_per_shard must be > 0
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 2,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 12,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config cfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&cfg);
  CHECK(Fail, zs);

  // Initial zarr.json should have shape[0] = 0
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    int has_shape = strstr((char*)data, "\"shape\":[0,8,12]") != NULL;
    free(data);
    CHECK(Fail2, has_shape);
    log_info("  initial shape: [0,8,12] OK");
  }

  // Stream some data through the pipeline
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail2,
        (s = tile_stream_gpu_create(&config, zarr_fs_sink_as_shard_sink(zs))) !=
          NULL);

  // epoch_elements = chunks_per_epoch * chunk_elements
  // chunks_per_epoch = chunk_count[1] * chunk_count[2] = 2 * 4 = 8
  // chunk_elements = 2 * 4 * 3 = 24
  // epoch_elements = 8 * 24 = 192
  log_info("  epoch_elements=%lu",
           (unsigned long)tile_stream_gpu_layout(s)->epoch_elements);

  // Feed 4 epochs of data
  const size_t total = 4 * tile_stream_gpu_layout(s)->epoch_elements;
  uint16_t* src = (uint16_t*)malloc(total * sizeof(uint16_t));
  CHECK(Fail3, src);
  for (size_t i = 0; i < total; ++i)
    src[i] = (uint16_t)(i % 65536);

  struct slice input = { .beg = src, .end = src + total };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail4, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail4, r.error == 0);

  zarr_fs_sink_flush(zs);

  // After flush, zarr.json shape[0] should reflect data written.
  // 4 epochs of chunk_size=2 → shape[0] = 8.
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);

    uint8_t* data;
    size_t len;
    CHECK(Fail4, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    log_info("  final metadata: %s", (char*)data);

    int is_array = strstr((char*)data, "\"node_type\":\"array\"") != NULL;
    int has_shape = strstr((char*)data, "\"shape\":[8,8,12]") != NULL;
    free(data);

    CHECK(Fail4, is_array);
    CHECK(Fail4, has_shape); // 4 epochs * chunk_size 2 = 8
  }

  free(src);
  tile_stream_gpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail4:
  free(src);
Fail3:
  tile_stream_gpu_destroy(s);
Fail2:
  zarr_fs_sink_destroy(zs);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: multiscale unbounded dim0 creation ---

static int
test_multiscale_unbounded(const char* tmpdir)
{
  log_info("=== test_multiscale_unbounded ===");

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "z",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "x",
      .downsample = 1,
      .storage_position = 2 },
  };

  struct zarr_multiscale_config cfg = {
    .store_path = tmpdir,
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .nlod = 0, // auto
  };

  struct zarr_fs_multiscale_sink* ms = zarr_fs_multiscale_sink_create(&cfg);
  CHECK(Fail, ms);

  // Check root zarr.json exists with multiscales
  {
    char* data;
    CHECK(Fail2, check_group_zarr_json(tmpdir, &data, 8192) == 0);
    int has_ms = strstr(data, "\"multiscales\"") != NULL;
    int has_p0 = strstr(data, "\"path\":\"0\"") != NULL;
    free(data);
    CHECK(Fail2, has_ms);
    CHECK(Fail2, has_p0);
  }

  // Check L0 array zarr.json has shape[0] = 0
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    int has_shape = strstr((char*)data, "\"shape\":[0,32,64]") != NULL;
    free(data);
    CHECK(Fail2, has_shape);
    log_info("  L0 shape: [0,32,64] OK");
  }

  // Check L1 array zarr.json has shape[0] = 0 (unbounded propagates)
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/1/zarr.json", tmpdir);
    CHECK(Fail2, test_file_exists(path));

    uint8_t* data;
    size_t len;
    CHECK(Fail2, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    // dim0 unbounded (0), dim1 unchanged (32), dim2 halved (32)
    int has_shape = strstr((char*)data, "\"shape\":[0,32,32]") != NULL;
    free(data);
    CHECK(Fail2, has_shape);
    log_info("  L1 shape: [0,32,32] OK");
  }

  zarr_fs_multiscale_sink_destroy(ms);
  log_info("  PASS");
  return 0;

Fail2:
  zarr_fs_multiscale_sink_destroy(ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: midstream metadata update via timer ---

static int
test_midstream_metadata_update(const char* tmpdir)
{
  log_info("=== test_midstream_metadata_update ===");

  // dim0 unbounded (size=0)
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 2,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 12,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config cfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&cfg);
  CHECK(Fail, zs);

  // Enable periodic metadata updates with a tiny interval.
  // Force epochs_per_batch=1 so each epoch triggers a flush (and timer check).
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .metadata_update_interval_s = 0.0f, // always fire
    .epochs_per_batch = 1,
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail2,
        (s = tile_stream_gpu_create(&config, zarr_fs_sink_as_shard_sink(zs))) !=
          NULL);

  // Feed several epochs of data (enough to trigger timer-based update)
  const size_t total = 6 * tile_stream_gpu_layout(s)->epoch_elements;
  uint16_t* src = (uint16_t*)malloc(total * sizeof(uint16_t));
  CHECK(Fail3, src);
  for (size_t i = 0; i < total; ++i)
    src[i] = (uint16_t)(i % 65536);

  // Feed in two batches; the 1e-9 interval fires the timer on every
  // wait_and_deliver
  size_t half = 3 * tile_stream_gpu_layout(s)->epoch_elements;
  {
    struct slice input = { .beg = src, .end = src + half };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }

  {
    struct slice input = { .beg = src + half, .end = src + total };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }

  // The timer-based update_append fired during the second append batch,
  // writing zarr.json synchronously. Verify shape[0] > 0 before writer_flush.
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);

    uint8_t* data;
    size_t len;
    CHECK(Fail4, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    log_info("  midstream metadata: %s", (char*)data);

    int not_zero = strstr((char*)data, "\"shape\":[0,") == NULL;
    // 5 epochs delivered (epoch 5 still pending), dim0_extent = 5 * 2 = 10
    int has_shape = strstr((char*)data, "\"shape\":[10,8,12]") != NULL;
    int is_array = strstr((char*)data, "\"node_type\":\"array\"") != NULL;
    free(data);

    CHECK(Fail4, not_zero);
    CHECK(Fail4, has_shape);
    CHECK(Fail4, is_array);
  }

  // Now flush and verify final state
  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    CHECK(Fail4, r.error == 0);
  }
  zarr_fs_sink_flush(zs);

  // Verify final shape after flush: 6 epochs * chunk_size 2 = 12
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s/0/zarr.json", tmpdir);

    uint8_t* data;
    size_t len;
    CHECK(Fail4, read_file_all(path, &data, &len) == 0);
    data[len < 4095 ? len : 4095] = '\0';
    int has_shape = strstr((char*)data, "\"shape\":[12,8,12]") != NULL;
    free(data);
    CHECK(Fail4, has_shape);
    log_info("  final metadata shape: [12,8,12] OK");
  }

  free(src);
  tile_stream_gpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail4:
  free(src);
Fail3:
  tile_stream_gpu_destroy(s);
Fail2:
  zarr_fs_sink_destroy(zs);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: unbuffered IO pipeline (single shard) ---

static int
test_unbuffered_pipeline(const char* tmpdir)
{
  log_info("=== test_unbuffered_pipeline ===");

  // Simple 3D: 2 epochs, 1 shard
  // dim0=4 (epoch dim), dim1=4, dim2=4, chunk=2x2x2, cps=2x2x2 → 1 shard
  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "z",
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  const int total_elements = 4 * 4 * 4;
  const int voxels_per_chunk = 2 * 2 * 2;       // 8
  const int chunks_per_shard_total = 2 * 2 * 2; // 8

  uint32_t src[64];
  for (int i = 0; i < total_elements; ++i)
    src[i] = (uint32_t)i;

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .unbuffered = 1,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail, zs);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .epochs_per_batch = 1,
    .shard_alignment = platform_page_size(),
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail2,
        (s = tile_stream_gpu_create(&config, zarr_fs_sink_as_shard_sink(zs))) !=
          NULL);

  // Feed data
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail3, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    CHECK(Fail3, r.error == 0);
  }

  zarr_fs_sink_flush(zs);

  // Verify shard file exists
  char path[4096];
  snprintf(path, sizeof(path), "%s/0/c/0/0/0", tmpdir);
  CHECK(Fail3, test_file_exists(path));

  // Read and verify contents
  {
    uint8_t* shard_data;
    size_t shard_len;
    CHECK(Fail3, read_file_all(path, &shard_data, &shard_len) == 0);

    size_t index_data_bytes =
      (size_t)chunks_per_shard_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;
    CHECK(Fail4, shard_len > index_total_bytes);

    const uint8_t* index_ptr = shard_data + shard_len - index_total_bytes;

    uint64_t chunk_offsets[8], chunk_nbytes[8];
    for (int i = 0; i < chunks_per_shard_total; ++i) {
      memcpy(&chunk_offsets[i], index_ptr + (size_t)i * 16, sizeof(uint64_t));
      memcpy(
        &chunk_nbytes[i], index_ptr + (size_t)i * 16 + 8, sizeof(uint64_t));
    }

    size_t chunk_stride_bytes =
      tile_stream_gpu_layout(s)->chunk_stride * sizeof(uint32_t);
    int errors = 0;

    for (int i_chunk = 0; i_chunk < chunks_per_shard_total; ++i_chunk) {
      if (chunk_nbytes[i_chunk] == 0 ||
          chunk_nbytes[i_chunk] > ZSTD_compressBound(chunk_stride_bytes)) {
        log_error("chunk %d: bad nbytes=%llu",
                  i_chunk,
                  (unsigned long long)chunk_nbytes[i_chunk]);
        errors++;
        continue;
      }

      uint8_t* decomp = (uint8_t*)calloc(1, chunk_stride_bytes);
      CHECK(Fail4, decomp);

      size_t result = ZSTD_decompress(decomp,
                                      chunk_stride_bytes,
                                      shard_data + chunk_offsets[i_chunk],
                                      (size_t)chunk_nbytes[i_chunk]);
      if (ZSTD_isError(result)) {
        log_error(
          "chunk %d: ZSTD error: %s", i_chunk, ZSTD_getErrorName(result));
        free(decomp);
        errors++;
        continue;
      }

      // Verify decompressed data is non-zero (basic sanity)
      const uint32_t* voxels = (const uint32_t*)decomp;
      int nonzero = 0;
      for (int e = 0; e < voxels_per_chunk; ++e)
        if (voxels[e] != 0)
          nonzero++;

      // At least some chunks should have non-zero data (only chunk 0 could be
      // all-zero since src starts at 0, but even then voxels from offset > 0
      // won't be zero).
      if (i_chunk > 0 && nonzero == 0) {
        log_error("chunk %d: all zeros after decompression", i_chunk);
        errors++;
      }

      free(decomp);
    }

    if (errors > 0) {
      log_error("  %d errors in unbuffered pipeline", errors);
      free(shard_data);
      goto Fail3;
    }

    free(shard_data);
  }

  tile_stream_gpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail4:
Fail3:
  tile_stream_gpu_destroy(s);
Fail2:
  zarr_fs_sink_destroy(zs);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: unbuffered pipeline with multiple shards ---
// Uses the same geometry as test_pipeline (12x8x12, chunk 2x4x3, cps 3x2x2
// → 4 shards) but with unbuffered IO and shard alignment.

static int
test_unbuffered_pipeline_multishard(const char* tmpdir)
{
  log_info("=== test_unbuffered_pipeline_multishard ===");

  const int size[3] = { 12, 8, 12 };
  const int chunk_size[3] = { 2, 4, 3 };
  const int chunks_per_shard[3] = { 3, 2, 2 };

  const int chunk_count[3] = {
    size[0] / chunk_size[0],
    size[1] / chunk_size[1],
    size[2] / chunk_size[2],
  };
  const int shard_count[3] = {
    chunk_count[0] / chunks_per_shard[0],
    chunk_count[1] / chunks_per_shard[1],
    chunk_count[2] / chunks_per_shard[2],
  };

  const int total_elements = size[0] * size[1] * size[2];
  const int num_shards = shard_count[0] * shard_count[1] * shard_count[2];
  const int chunks_per_shard_total =
    chunks_per_shard[0] * chunks_per_shard[1] * chunks_per_shard[2];
  const int voxels_per_chunk = chunk_size[0] * chunk_size[1] * chunk_size[2];

  uint32_t* src = NULL;

  // Generate source data (same encoding as test_pipeline)
  src = (uint32_t*)malloc((size_t)total_elements * sizeof(uint32_t));
  CHECK(Fail, src);

  for (int x0 = 0; x0 < size[0]; ++x0)
    for (int x1 = 0; x1 < size[1]; ++x1)
      for (int x2 = 0; x2 < size[2]; ++x2) {
        int gi = x0 * size[1] * size[2] + x1 * size[2] + x2;
        int s0 = x0 / (chunk_size[0] * chunks_per_shard[0]);
        int s1 = x1 / (chunk_size[1] * chunks_per_shard[1]);
        int s2 = x2 / (chunk_size[2] * chunks_per_shard[2]);
        int t0 = (x0 / chunk_size[0]) % chunks_per_shard[0];
        int t1 = (x1 / chunk_size[1]) % chunks_per_shard[1];
        int t2 = (x2 / chunk_size[2]) % chunks_per_shard[2];
        int v0 = x0 % chunk_size[0];
        int v1 = x1 % chunk_size[1];
        int v2 = x2 % chunk_size[2];
        src[gi] = encode_voxel(s0, s1, s2, t0, t1, t2, v0, v1, v2);
      }

  // Create zarr sink with unbuffered IO
  struct dimension dims[] = {
    { .size = 12,
      .chunk_size = 2,
      .chunks_per_shard = 3,
      .name = "z",
      .storage_position = 0 },
    { .size = 8,
      .chunk_size = 4,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 1 },
    { .size = 12,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 2 },
  };

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .unbuffered = 1,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail2, zs);

  // Configure stream with shard alignment for unbuffered IO
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .shard_alignment = platform_page_size(),
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail3,
        (s = tile_stream_gpu_create(&config, zarr_fs_sink_as_shard_sink(zs))) !=
          NULL);

  // Feed data
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    CHECK(Fail4, r.error == 0);
  }

  zarr_fs_sink_flush(zs);

  // Verify shard files: same loop as test_pipeline
  {
    const int cps_inner = chunks_per_shard[1] * chunks_per_shard[2];
    int errors = 0;

    for (int i_shard = 0; i_shard < num_shards; ++i_shard) {
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

      size_t index_data_bytes =
        (size_t)chunks_per_shard_total * 2 * sizeof(uint64_t);
      size_t index_total_bytes = index_data_bytes + 4;
      CHECK(Fail4, shard_len > index_total_bytes);

      const uint8_t* index_ptr = shard_data + shard_len - index_total_bytes;

      uint64_t chunk_offsets[12], chunk_nbytes[12];
      for (int i = 0; i < chunks_per_shard_total; ++i) {
        memcpy(&chunk_offsets[i], index_ptr + (size_t)i * 16, sizeof(uint64_t));
        memcpy(
          &chunk_nbytes[i], index_ptr + (size_t)i * 16 + 8, sizeof(uint64_t));
      }

      size_t chunk_stride_bytes =
        tile_stream_gpu_layout(s)->chunk_stride * sizeof(uint32_t);

      for (int i_chunk = 0; i_chunk < chunks_per_shard_total; ++i_chunk) {
        if (chunk_nbytes[i_chunk] == 0 ||
            chunk_nbytes[i_chunk] > ZSTD_compressBound(chunk_stride_bytes)) {
          log_error("shard %d chunk %d: bad nbytes=%llu",
                    i_shard,
                    i_chunk,
                    (unsigned long long)chunk_nbytes[i_chunk]);
          errors++;
          continue;
        }

        uint8_t* decomp = (uint8_t*)calloc(1, chunk_stride_bytes);
        CHECK(Fail4, decomp);

        size_t result = ZSTD_decompress(decomp,
                                        chunk_stride_bytes,
                                        shard_data + chunk_offsets[i_chunk],
                                        (size_t)chunk_nbytes[i_chunk]);
        if (ZSTD_isError(result)) {
          log_error("shard %d chunk %d: ZSTD error: %s",
                    i_shard,
                    i_chunk,
                    ZSTD_getErrorName(result));
          free(decomp);
          errors++;
          continue;
        }

        int exp_t0 = i_chunk / cps_inner;
        int exp_t1 = (i_chunk % cps_inner) / chunks_per_shard[2];
        int exp_t2 = (i_chunk % cps_inner) % chunks_per_shard[2];

        const uint32_t* voxels = (const uint32_t*)decomp;
        for (int e = 0; e < voxels_per_chunk; ++e) {
          int exp_v0 = e / (chunk_size[1] * chunk_size[2]);
          int exp_v1 = (e / chunk_size[2]) % chunk_size[1];
          int exp_v2 = e % chunk_size[2];

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
              log_error("shard %d chunk %d elem %d: got 0x%08x expected 0x%08x",
                        i_shard,
                        i_chunk,
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

  tile_stream_gpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  free(src);
  log_info("  PASS");
  return 0;

Fail4:
  tile_stream_gpu_destroy(s);
Fail3:
  zarr_fs_sink_destroy(zs);
Fail2:
  free(src);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: storage_order validation (no CUDA needed) ---

static int
test_storage_order_validation(const char* tmpdir)
{
  (void)tmpdir;
  log_info("=== test_storage_order_validation ===");

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2, .name = "z" },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2, .name = "y" },
    { .size = 6, .chunk_size = 3, .chunks_per_shard = 2, .name = "x" },
  };

  // storage_position[0] != 0 → should fail
  {
    struct dimension bad_dims[] = {
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "z",
        .storage_position = 1 },
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "y",
        .storage_position = 0 },
      { .size = 6,
        .chunk_size = 3,
        .chunks_per_shard = 2,
        .name = "x",
        .storage_position = 2 },
    };
    struct tile_stream_memory_info info;
    struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4096,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = bad_dims,
    };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, &info) != 0);
    log_info("  storage_position[0]!=0 rejected OK");
  }

  // Duplicate values → should fail
  {
    struct dimension bad_dims[] = {
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "z",
        .storage_position = 0 },
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "y",
        .storage_position = 2 },
      { .size = 6,
        .chunk_size = 3,
        .chunks_per_shard = 2,
        .name = "x",
        .storage_position = 2 },
    };
    struct tile_stream_memory_info info;
    struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4096,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = bad_dims,
    };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, &info) != 0);
    log_info("  duplicate values rejected OK");
  }

  // Out-of-range values → should fail
  {
    struct dimension bad_dims[] = {
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "z",
        .storage_position = 0 },
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "y",
        .storage_position = 5 },
      { .size = 6,
        .chunk_size = 3,
        .chunks_per_shard = 2,
        .name = "x",
        .storage_position = 1 },
    };
    struct tile_stream_memory_info info;
    struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4096,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = bad_dims,
    };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, &info) != 0);
    log_info("  out-of-range rejected OK");
  }

  // All-zero (repeated indices) → should fail
  {
    struct tile_stream_memory_info info;
    struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4096,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = dims,
    };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, &info) != 0);
    log_info("  all-zero rejected OK");
  }

  // Explicit identity → should succeed
  {
    struct dimension id_dims[] = {
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "z",
        .storage_position = 0 },
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "y",
        .storage_position = 1 },
      { .size = 6,
        .chunk_size = 3,
        .chunks_per_shard = 2,
        .name = "x",
        .storage_position = 2 },
    };
    struct tile_stream_memory_info info;
    struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4096,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = id_dims,
    };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, &info) == 0);
    log_info("  explicit identity accepted OK");
  }

  // Valid permutation: dim1→pos2, dim2→pos1 (forward={0,2,1}) → should succeed
  {
    struct dimension perm_dims[] = {
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "z",
        .storage_position = 0 },
      { .size = 4,
        .chunk_size = 2,
        .chunks_per_shard = 2,
        .name = "y",
        .storage_position = 2 },
      { .size = 6,
        .chunk_size = 3,
        .chunks_per_shard = 2,
        .name = "x",
        .storage_position = 1 },
    };
    struct tile_stream_memory_info info;
    struct tile_stream_configuration config = {
      .buffer_capacity_bytes = 4096,
      .dtype = dtype_u16,
      .rank = 3,
      .dimensions = perm_dims,
    };
    CHECK(Fail, tile_stream_gpu_memory_estimate(&config, &info) == 0);
    log_info("  valid {0,2,1} accepted OK");
  }

  log_info("  PASS");
  return 0;

Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: full pipeline with non-identity storage_order ---

static int
test_pipeline_storage_order(const char* tmpdir)
{
  log_info("=== test_pipeline_storage_order ===");

  // Acquisition order: z=4, y=4, x=6
  // Tiles:             2,   2,   3
  // Tiles per shard:   1,   2,   2
  // storage_position: z→0, y→2, x→1 → storage dims: [z, x, y]
  //
  // ONE dims array is used for both tile_stream and zarr_fs_sink.

  const int acq_size[3] = { 4, 4, 6 };
  const int acq_tile[3] = { 2, 2, 3 };

  // Storage-ordered sizes/chunks/cps (for verification)
  const int sto_size[3] = { 4, 6, 4 };
  const int sto_tile[3] = { 2, 3, 2 };
  const int sto_cps[3] = { 1, 2, 2 };

  const int sto_chunk_count[3] = {
    sto_size[0] / sto_tile[0],
    sto_size[1] / sto_tile[1],
    sto_size[2] / sto_tile[2],
  };
  const int sto_shard_count[3] = {
    sto_chunk_count[0] / sto_cps[0],
    sto_chunk_count[1] / sto_cps[1],
    sto_chunk_count[2] / sto_cps[2],
  };

  const int total_elements = acq_size[0] * acq_size[1] * acq_size[2];
  const int num_shards =
    sto_shard_count[0] * sto_shard_count[1] * sto_shard_count[2];
  const int chunks_per_shard_total = sto_cps[0] * sto_cps[1] * sto_cps[2];
  const int voxels_per_chunk = acq_tile[0] * acq_tile[1] * acq_tile[2];

  uint32_t* src = NULL;

  // Generate source data in acquisition order (z slow, y medium, x fast).
  // Each voxel encodes its global acquisition coordinates.
  src = (uint32_t*)malloc((size_t)total_elements * sizeof(uint32_t));
  CHECK(Fail, src);

  for (int z = 0; z < acq_size[0]; ++z)
    for (int y = 0; y < acq_size[1]; ++y)
      for (int x = 0; x < acq_size[2]; ++x) {
        int gi = z * acq_size[1] * acq_size[2] + y * acq_size[2] + x;
        // Encode as (z << 16 | y << 8 | x) for easy debugging
        src[gi] = ((uint32_t)z << 16) | ((uint32_t)y << 8) | (uint32_t)x;
      }

  // Single dims array: acquisition order with storage_position.
  // z→pos0, y→pos2, x→pos1  ⇒  storage order is [z, x, y]
  struct dimension dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "z",
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 2 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 1 },
  };

  // Zarr_sink receives storage-ordered dims (permuted from above).
  // forward = {0, 2, 1}: storage pos 0→dim0(z), pos 1→dim2(x), pos 2→dim1(y)
  struct dimension sto_dims[] = {
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "z",
      .storage_position = 0 },
    { .size = 6,
      .chunk_size = 3,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 1 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "y",
      .storage_position = 2 },
  };

  struct zarr_config zcfg = {
    .store_path = tmpdir,
    .array_name = "0",
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = sto_dims,
  };

  struct zarr_fs_sink* zs = zarr_fs_sink_create(&zcfg);
  CHECK(Fail2, zs);

  // tile_stream uses acquisition-order dims (same array, with storage_position)
  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = (size_t)total_elements * sizeof(uint32_t),
    .dtype = dtype_u32,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
  };

  struct tile_stream_gpu* s = NULL;
  CHECK(Fail3,
        (s = tile_stream_gpu_create(&config, zarr_fs_sink_as_shard_sink(zs))) !=
          NULL);

  // Feed data
  {
    struct slice input = { .beg = src, .end = src + total_elements };
    struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
    CHECK(Fail4, r.error == 0);
  }
  {
    struct writer_result r = writer_flush(tile_stream_gpu_writer(s));
    CHECK(Fail4, r.error == 0);
  }

  zarr_fs_sink_flush(zs);

  // Verify shard files and chunk contents
  {
    const int cps_inner = sto_cps[1] * sto_cps[2];
    int errors = 0;

    for (int i_shard = 0; i_shard < num_shards; ++i_shard) {
      // Compute shard coordinates in storage order
      int sc[3];
      {
        int rem = i_shard;
        for (int d = 2; d >= 0; --d) {
          sc[d] = rem % sto_shard_count[d];
          rem /= sto_shard_count[d];
        }
      }

      char path[4096];
      snprintf(
        path, sizeof(path), "%s/0/c/%d/%d/%d", tmpdir, sc[0], sc[1], sc[2]);
      CHECK(Fail4, test_file_exists(path));

      uint8_t* shard_data;
      size_t shard_len;
      CHECK(Fail4, read_file_all(path, &shard_data, &shard_len) == 0);

      size_t index_data_bytes =
        (size_t)chunks_per_shard_total * 2 * sizeof(uint64_t);
      size_t index_total_bytes = index_data_bytes + 4;
      CHECK(Fail4, shard_len > index_total_bytes);

      const uint8_t* index_ptr = shard_data + shard_len - index_total_bytes;

      uint64_t chunk_offsets[4], chunk_nbytes[4];
      for (int i = 0; i < chunks_per_shard_total; ++i) {
        memcpy(&chunk_offsets[i], index_ptr + (size_t)i * 16, sizeof(uint64_t));
        memcpy(
          &chunk_nbytes[i], index_ptr + (size_t)i * 16 + 8, sizeof(uint64_t));
      }

      size_t chunk_stride_bytes =
        tile_stream_gpu_layout(s)->chunk_stride * sizeof(uint32_t);

      for (int i_chunk = 0; i_chunk < chunks_per_shard_total; ++i_chunk) {
        if (chunk_nbytes[i_chunk] == 0 ||
            chunk_nbytes[i_chunk] > ZSTD_compressBound(chunk_stride_bytes)) {
          log_error("shard %d chunk %d: bad nbytes=%llu",
                    i_shard,
                    i_chunk,
                    (unsigned long long)chunk_nbytes[i_chunk]);
          errors++;
          continue;
        }

        uint8_t* decomp = (uint8_t*)calloc(1, chunk_stride_bytes);
        CHECK(Fail4, decomp);

        size_t result = ZSTD_decompress(decomp,
                                        chunk_stride_bytes,
                                        shard_data + chunk_offsets[i_chunk],
                                        (size_t)chunk_nbytes[i_chunk]);
        if (ZSTD_isError(result)) {
          log_error("shard %d chunk %d: ZSTD error: %s",
                    i_shard,
                    i_chunk,
                    ZSTD_getErrorName(result));
          free(decomp);
          errors++;
          continue;
        }

        // Tile-in-shard coordinates (storage order: z, x, y)
        int st_z = i_chunk / cps_inner;
        int st_x = (i_chunk % cps_inner) / sto_cps[2];
        int st_y = (i_chunk % cps_inner) % sto_cps[2];

        // Global chunk coords in storage order
        int gt_z = sc[0] * sto_cps[0] + st_z;
        int gt_x = sc[1] * sto_cps[1] + st_x;
        int gt_y = sc[2] * sto_cps[2] + st_y;

        const uint32_t* voxels = (const uint32_t*)decomp;
        for (int e = 0; e < voxels_per_chunk; ++e) {
          // Within-chunk coords in storage order (z, x, y)
          int vt_z = e / (sto_tile[1] * sto_tile[2]);
          int vt_x = (e / sto_tile[2]) % sto_tile[1];
          int vt_y = e % sto_tile[2];

          // Global coords in storage order → acquisition order
          int gz = gt_z * sto_tile[0] + vt_z;
          int gx = gt_x * sto_tile[1] + vt_x;
          int gy = gt_y * sto_tile[2] + vt_y;

          // Expected value: encode(z, y, x) in acquisition order
          uint32_t expected =
            ((uint32_t)gz << 16) | ((uint32_t)gy << 8) | (uint32_t)gx;

          if (voxels[e] != expected) {
            if (errors < 10) {
              log_error(
                "shard (%d,%d,%d) chunk %d elem %d: got 0x%08x expected 0x%08x"
                " (gz=%d gy=%d gx=%d)",
                sc[0],
                sc[1],
                sc[2],
                i_chunk,
                e,
                voxels[e],
                expected,
                gz,
                gy,
                gx);
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

  tile_stream_gpu_destroy(s);
  zarr_fs_sink_destroy(zs);
  free(src);
  log_info("  PASS");
  return 0;

Fail4:
  tile_stream_gpu_destroy(s);
Fail3:
  zarr_fs_sink_destroy(zs);
Fail2:
  free(src);
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

  // Metadata test with n_append=2 (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/meta2app", tmpdir);
    test_mkdir(sub);
    ecode |= test_metadata_two_append(sub);
  }

  // Multiscale metadata test (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/msmeta", tmpdir);
    test_mkdir(sub);
    ecode |= test_multiscale_metadata(sub);
  }

  // Multiscale scale with non-power-of-2 sizes (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/msscalenp2", tmpdir);
    test_mkdir(sub);
    ecode |= test_multiscale_scale_non_pow2(sub);
  }

  // Multiscale unit/scale metadata test (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/msunitscale", tmpdir);
    test_mkdir(sub);
    ecode |= test_multiscale_unit_scale(sub);
  }

  // Multiscale unbounded metadata test (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/msunbounded", tmpdir);
    test_mkdir(sub);
    ecode |= test_multiscale_unbounded(sub);
  }

  // Storage order validation test (no CUDA needed)
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/so_valid", tmpdir);
    test_mkdir(sub);
    ecode |= test_storage_order_validation(sub);
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

    {
      char sub[4200];
      snprintf(sub, sizeof(sub), "%s/unbounded", tmpdir);
      test_mkdir(sub);
      ecode |= test_unbounded_metadata_update(sub);
    }

    {
      char sub[4200];
      snprintf(sub, sizeof(sub), "%s/midstream", tmpdir);
      test_mkdir(sub);
      ecode |= test_midstream_metadata_update(sub);
    }

    {
      char sub[4200];
      snprintf(sub, sizeof(sub), "%s/unbuf", tmpdir);
      test_mkdir(sub);
      ecode |= test_unbuffered_pipeline(sub);
    }

    {
      char sub[4200];
      snprintf(sub, sizeof(sub), "%s/unbuf_ms", tmpdir);
      test_mkdir(sub);
      ecode |= test_unbuffered_pipeline_multishard(sub);
    }

    {
      char sub[4200];
      snprintf(sub, sizeof(sub), "%s/so_pipe", tmpdir);
      test_mkdir(sub);
      ecode |= test_pipeline_storage_order(sub);
    }

    cuCtxDestroy(ctx);
  }

Cleanup:
  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}

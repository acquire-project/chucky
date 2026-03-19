#include "platform_cmd.h"
#include "prelude.h"
#include "zarr_s3_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Minio / Docker helpers ---

#define MINIO_ENDPOINT "http://localhost:9000"
#define MINIO_BUCKET   "chucky-test"
#define MINIO_USER     "minioadmin"
#define MINIO_PASS     "minioadmin"

static int
minio_setup(void)
{
  CHECK(Fail,
        platform_cmd_run("docker exec minio mc alias set local "
                         MINIO_ENDPOINT " " MINIO_USER " " MINIO_PASS
                         " > /dev/null 2>&1") == 0);
  // Recreate bucket
  platform_cmd_run("docker exec minio mc rb --force local/" MINIO_BUCKET
                   " > /dev/null 2>&1");
  CHECK(Fail,
        platform_cmd_run("docker exec minio mc mb local/" MINIO_BUCKET
                         " > /dev/null 2>&1") == 0);
  return 0;
Fail:
  return 1;
}

// Read an object from minio. Caller must free(*out).
static int
minio_get(const char* key, uint8_t** out, size_t* out_len)
{
  char cmd[4096];
  snprintf(cmd,
           sizeof(cmd),
           "docker exec minio mc cat local/%s/%s 2>/dev/null",
           MINIO_BUCKET,
           key);
  return platform_cmd_capture(cmd, out, out_len);
}

static int
minio_exists(const char* key)
{
  char cmd[4096];
  snprintf(cmd,
           sizeof(cmd),
           "docker exec minio mc stat local/%s/%s > /dev/null 2>&1",
           MINIO_BUCKET,
           key);
  return platform_cmd_run(cmd) == 0;
}

// --- Tests ---

static void
set_minio_creds(void)
{
  setenv("AWS_ACCESS_KEY_ID", MINIO_USER, 1);
  setenv("AWS_SECRET_ACCESS_KEY", MINIO_PASS, 1);
}

static int
test_metadata(void)
{
  log_info("=== test_s3_metadata ===");

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

  set_minio_creds();

  struct zarr_s3_config cfg = {
    .bucket = MINIO_BUCKET,
    .prefix = "test-meta",
    .array_name = "0",
    .region = "us-east-1",
    .endpoint = MINIO_ENDPOINT,
    .data_type = zarr_dtype_uint32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct zarr_s3_sink* sink = zarr_s3_sink_create(&cfg);
  CHECK(Fail, sink);

  // Verify root zarr.json
  CHECK(Fail_sink, minio_exists("test-meta/zarr.json"));

  // Verify array zarr.json
  CHECK(Fail_sink, minio_exists("test-meta/0/zarr.json"));

  // Read and check array metadata
  uint8_t* data = NULL;
  size_t len = 0;
  CHECK(Fail_sink, minio_get("test-meta/0/zarr.json", &data, &len) == 0);

  // Null-terminate for string search
  uint8_t* tmp = (uint8_t*)realloc(data, len + 1);
  CHECK(Fail_data, tmp);
  data = tmp;
  data[len] = '\0';

  CHECK(Fail_data, strstr((char*)data, "\"zarr_format\":3"));
  CHECK(Fail_data, strstr((char*)data, "\"node_type\":\"array\""));
  CHECK(Fail_data, strstr((char*)data, "\"data_type\":\"uint32\""));
  CHECK(Fail_data, strstr((char*)data, "\"sharding_indexed\""));
  CHECK(Fail_data, strstr((char*)data, "\"zstd\""));

  log_info("  array zarr.json OK (%zu bytes)", len);
  free(data);
  zarr_s3_sink_destroy(sink);
  log_info("  PASS");
  return 0;

Fail_data:
  free(data);
Fail_sink:
  zarr_s3_sink_destroy(sink);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_write(void)
{
  log_info("=== test_s3_shard_write ===");

  struct dimension dims[] = {
    { .size = 8,
      .chunk_size = 2,
      .chunks_per_shard = 2,
      .name = "x",
      .storage_position = 0 },
  };

  set_minio_creds();

  struct zarr_s3_config cfg = {
    .bucket = MINIO_BUCKET,
    .prefix = "test-shard",
    .array_name = "0",
    .region = "us-east-1",
    .endpoint = MINIO_ENDPOINT,
    .data_type = zarr_dtype_uint16,
    .fill_value = 0,
    .rank = 1,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct zarr_s3_sink* sink = zarr_s3_sink_create(&cfg);
  CHECK(Fail, sink);

  struct shard_sink* ss = zarr_s3_sink_as_shard_sink(sink);

  // shard_index=0 → key "test-shard/0/c/0"
  struct shard_writer* w = ss->open(ss, 0, 0);
  CHECK(Fail_sink, w);

  // Write 8 bytes of test data (4 uint16 values = 2 chunks)
  uint16_t chunk_data[4] = { 0x1234, 0x5678, 0xABCD, 0xEF01 };
  CHECK(Fail_sink,
        w->write(w, 0, chunk_data, (char*)chunk_data + sizeof(chunk_data)) ==
          0);
  CHECK(Fail_sink, w->finalize(w) == 0);

  // Read back the shard object
  uint8_t* data = NULL;
  size_t len = 0;
  CHECK(Fail_sink, minio_get("test-shard/0/c/0", &data, &len) == 0);
  CHECK(Fail_data, len >= sizeof(chunk_data));
  CHECK(Fail_data, memcmp(data, chunk_data, sizeof(chunk_data)) == 0);

  log_info("  shard 0 OK (%zu bytes)", len);
  free(data);
  zarr_s3_sink_destroy(sink);
  log_info("  PASS");
  return 0;

Fail_data:
  free(data);
Fail_sink:
  zarr_s3_sink_destroy(sink);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Main ---

int
main(void)
{
  if (minio_setup() != 0) {
    log_error("minio not available — is the container running?");
    log_error("  docker run -d --name minio -p 9000:9000 "
              "minio/minio server /data");
    return 1;
  }

  int rc = 0;
  rc |= test_metadata();
  rc |= test_shard_write();
  return rc;
}

#include "platform_cmd.h"
#include "prelude.h"
#include "zarr_s3_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- S3 test helpers (via aws cli) ---
//
// Uses AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars.
// Defaults target a local minio on localhost:9000.

#define S3_BUCKET "chucky-test"
#define S3_USER   "minioadmin"
#define S3_PASS   "minioadmin"

#ifdef _WIN32
#define DEVNULL "NUL"
#else
#define DEVNULL "/dev/null"
#endif

static const char*
s3_endpoint(void)
{
  const char* p = getenv("AWS_ENDPOINT_URL");
  return p ? p : "http://localhost:9000";
}

static int
s3_setup(void)
{
  char cmd[4096];
  // Recreate bucket
  snprintf(cmd,
           sizeof(cmd),
           "aws --endpoint-url %s s3 rb s3://%s --force > " DEVNULL " 2>&1",
           s3_endpoint(),
           S3_BUCKET);
  platform_cmd_run(cmd);
  snprintf(cmd,
           sizeof(cmd),
           "aws --endpoint-url %s s3 mb s3://%s > " DEVNULL " 2>&1",
           s3_endpoint(),
           S3_BUCKET);
  CHECK(Fail, platform_cmd_run(cmd) == 0);
  return 0;
Fail:
  return 1;
}

// Read an object from S3. Caller must free(*out).
static int
s3_get(const char* key, uint8_t** out, size_t* out_len)
{
  char cmd[4096];
  snprintf(cmd,
           sizeof(cmd),
           "aws --endpoint-url %s s3 cp s3://%s/%s - 2>" DEVNULL,
           s3_endpoint(),
           S3_BUCKET,
           key);
  return platform_cmd_capture(cmd, out, out_len);
}

static int
s3_exists(const char* key)
{
  char cmd[4096];
  snprintf(cmd,
           sizeof(cmd),
           "aws --endpoint-url %s s3api head-object --bucket %s --key %s"
           " > " DEVNULL " 2>&1",
           s3_endpoint(),
           S3_BUCKET,
           key);
  return platform_cmd_run(cmd) == 0;
}

// --- Tests ---

static void
set_s3_creds(void)
{
#ifdef _WIN32
  _putenv_s("AWS_ACCESS_KEY_ID", S3_USER);
  _putenv_s("AWS_SECRET_ACCESS_KEY", S3_PASS);
#else
  setenv("AWS_ACCESS_KEY_ID", S3_USER, 1);
  setenv("AWS_SECRET_ACCESS_KEY", S3_PASS, 1);
#endif
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

  set_s3_creds();

  struct zarr_s3_config cfg = {
    .bucket = S3_BUCKET,
    .prefix = "test-meta",
    .array_name = "0",
    .region = "us-east-1",
    .endpoint = s3_endpoint(),
    .data_type = dtype_u32,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
  };

  struct zarr_s3_sink* sink = zarr_s3_sink_create(&cfg);
  CHECK(Fail, sink);

  // Verify root zarr.json
  CHECK(Fail_sink, s3_exists("test-meta/zarr.json"));

  // Verify array zarr.json
  CHECK(Fail_sink, s3_exists("test-meta/0/zarr.json"));

  // Read and check array metadata
  uint8_t* data = NULL;
  size_t len = 0;
  CHECK(Fail_sink, s3_get("test-meta/0/zarr.json", &data, &len) == 0);

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

  set_s3_creds();

  struct zarr_s3_config cfg = {
    .bucket = S3_BUCKET,
    .prefix = "test-shard",
    .array_name = "0",
    .region = "us-east-1",
    .endpoint = s3_endpoint(),
    .data_type = dtype_u16,
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
  CHECK(Fail_sink, s3_get("test-shard/0/c/0", &data, &len) == 0);
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

static int
test_concurrent_finalize(void)
{
  log_info("=== test_s3_concurrent_finalize ===");

  // 1D, 8 elements, chunk=2, cps=2, so shard_inner_count=1 (only 1 dim).
  // Use 2D to get multiple inner shards: 2x4, chunk 1x2, cps 1x2.
  // shard_count = [2, 1], shard_inner_count = 1.
  // Instead, use a shape that gives multiple inner shards:
  // 3D: t=0(streaming), y=4, x=4. chunk 1x2x2, cps 1x1x1.
  // shard_count = [0, 2, 2], shard_inner_count = 4.
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .name = "t",
      .storage_position = 0 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "y",
      .storage_position = 1 },
    { .size = 4,
      .chunk_size = 2,
      .chunks_per_shard = 1,
      .name = "x",
      .storage_position = 2 },
  };

  set_s3_creds();

  struct zarr_s3_config cfg = {
    .bucket = S3_BUCKET,
    .prefix = "test-concurrent",
    .array_name = "0",
    .region = "us-east-1",
    .endpoint = s3_endpoint(),
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct zarr_s3_sink* sink = zarr_s3_sink_create(&cfg);
  CHECK(Fail, sink);
  struct shard_sink* ss = zarr_s3_sink_as_shard_sink(sink);

  // Write epoch 0: open all 4 inner shards, write, finalize (async)
  uint16_t data[2] = { 0xAAAA, 0xBBBB };
  for (int i = 0; i < 4; ++i) {
    struct shard_writer* w = ss->open(ss, 0, (uint64_t)i);
    CHECK(Fail_sink, w);
    CHECK(Fail_sink,
          w->write(w, 0, data, (char*)data + sizeof(data)) == 0);
    CHECK(Fail_sink, w->finalize(w) == 0);
  }

  // Record fence after first epoch
  struct io_event fence = ss->record_fence(ss, 0);
  CHECK(Fail_sink, fence.seq > 0);

  // Write epoch 1: open triggers wait on pending, then new uploads
  uint16_t data2[2] = { 0xCCCC, 0xDDDD };
  for (int i = 0; i < 4; ++i) {
    struct shard_writer* w = ss->open(ss, 0, (uint64_t)(4 + i));
    CHECK(Fail_sink, w);
    CHECK(Fail_sink,
          w->write(w, 0, data2, (char*)data2 + sizeof(data2)) == 0);
    CHECK(Fail_sink, w->finalize(w) == 0);
  }

  // Wait fence drains all pending
  ss->wait_fence(ss, 0, fence);

  // Flush to drain epoch 1's pending uploads
  zarr_s3_sink_flush(sink);

  // Verify epoch 0 shards arrived
  for (int i = 0; i < 4; ++i) {
    char key[256];
    snprintf(key, sizeof(key), "test-concurrent/0/c/0/%d/%d", i / 2, i % 2);
    uint8_t* obj = NULL;
    size_t len = 0;
    CHECK(Fail_sink, s3_get(key, &obj, &len) == 0);
    CHECK(Fail_obj, len >= sizeof(data));
    CHECK(Fail_obj, memcmp(obj, data, sizeof(data)) == 0);
    free(obj);
    continue;
  Fail_obj:
    free(obj);
    goto Fail_sink;
  }

  log_info("  concurrent finalize + fence OK");
  zarr_s3_sink_destroy(sink);
  log_info("  PASS");
  return 0;

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
  if (s3_setup() != 0) {
    log_error("S3 not available — is minio running?");
    log_error("  docker compose up minio");
    return 1;
  }

  int rc = 0;
  rc |= test_metadata();
  rc |= test_shard_write();
  rc |= test_concurrent_finalize();
  return rc;
}

#include "defs.limits.h"
#include "dimension.h"
#include "lod/lod_plan.h"
#include "ngff/ngff_multiscale.h"
#include "platform/platform_cmd.h"
#include "util/prelude.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"
#include "zarr/store_s3.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_group.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- S3 test helpers (via aws cli) ---
//
// Uses AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars.
// Defaults target a local minio on localhost:9000.

#define S3_BUCKET "chucky-test"
#define S3_USER "minioadmin"
#define S3_PASS "minioadmin"

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

// Read an S3 object and check that it contains required substrings.
// needle2 may be NULL to skip the second check.
// Returns 0 on success.
static int
check_array_json(const char* key, const char* needle1, const char* needle2)
{
  uint8_t* data = NULL;
  size_t len = 0;
  CHECK(Fail, s3_get(key, &data, &len) == 0);

  uint8_t* tmp = (uint8_t*)realloc(data, len + 1);
  CHECK(Fail_data, tmp);
  data = tmp;
  data[len] = '\0';

  CHECK(Fail_data, strstr((char*)data, needle1));
  if (needle2)
    CHECK(Fail_data, strstr((char*)data, needle2));

  free(data);
  return 0;

Fail_data:
  free(data);
Fail:
  return 1;
}

// Read a group zarr.json at `prefix`/zarr.json and validate common fields.
// On success returns 0 and sets *out (caller must free).
// On failure returns non-zero.
static int
check_group_zarr_json(const char* prefix, char** out)
{
  char key[4096];
  snprintf(key, sizeof(key), "%s/zarr.json", prefix);

  uint8_t* data = NULL;
  size_t len = 0;
  CHECK(Fail, s3_get(key, &data, &len) == 0);

  // Null-terminate
  uint8_t* tmp = (uint8_t*)realloc(data, len + 1);
  CHECK(Fail_data, tmp);
  data = tmp;
  data[len] = '\0';

  int ok = strstr((char*)data, "\"zarr_format\":3") &&
           strstr((char*)data, "\"node_type\":\"group\"") &&
           strstr((char*)data, "\"consolidated_metadata\":null") &&
           strstr((char*)data, "\"attributes\":{");
  if (!ok)
    goto Fail_data;

  *out = (char*)data;
  return 0;

Fail_data:
  free(data);
Fail:
  return 1;
}

// --- S3 store helpers ---

struct s3_test_sink
{
  struct store* store;
  struct shard_pool* pool;
  struct zarr_array* array;
};

static int
s3_test_sink_open(struct s3_test_sink* z,
                  const char* prefix,
                  const char* array_name,
                  const struct dimension* dims,
                  uint8_t rank,
                  enum dtype data_type,
                  double fill_value,
                  struct codec_config codec)
{
  *z = (struct s3_test_sink){ 0 };

  struct store_s3_config scfg = {
    .bucket = S3_BUCKET,
    .prefix = prefix,
    .region = "us-east-1",
    .endpoint = s3_endpoint(),
  };
  store_s3_config_set_defaults(&scfg);

  z->store = store_s3_create(&scfg);
  CHECK(Fail, z->store);
  z->store->mkdirs(z->store, ".");

  uint64_t sc[MAX_ZARR_RANK], cps[MAX_ZARR_RANK];
  uint64_t sic = dims_compute_shard_geometry(dims, rank, sc, cps);

  z->pool = z->store->create_pool(z->store, sic);
  CHECK(Fail_store, z->pool);

  // Write root group
  CHECK(Fail_pool, zarr_write_group(z->store, "zarr.json", NULL) == 0);

  // Write intermediate groups
  if (array_name && array_name[0]) {
    CHECK(Fail_pool, z->store->mkdirs(z->store, array_name) == 0);
  }

  struct zarr_array_config acfg = {
    .data_type = data_type,
    .fill_value = fill_value,
    .rank = rank,
    .dimensions = dims,
    .codec = codec,
    .shard_counts = sc,
    .chunks_per_shard = cps,
    .shard_inner_count = sic,
  };
  z->array =
    zarr_array_create(z->store, z->pool, array_name ? array_name : "", &acfg);
  CHECK(Fail_pool, z->array);
  return 0;

Fail_pool:
  z->pool->destroy(z->pool);
  z->pool = NULL;
Fail_store:
  z->store->destroy(z->store);
  z->store = NULL;
Fail:
  return 1;
}

static struct shard_sink*
s3_test_sink_as_shard_sink(struct s3_test_sink* z)
{
  return zarr_array_as_shard_sink(z->array);
}

static void
s3_test_sink_flush(struct s3_test_sink* z)
{
  if (z->pool)
    z->pool->flush(z->pool);
}

static void
s3_test_sink_close(struct s3_test_sink* z)
{
  zarr_array_destroy(z->array);
  if (z->pool)
    z->pool->destroy(z->pool);
  if (z->store)
    z->store->destroy(z->store);
  *z = (struct s3_test_sink){ 0 };
}

// --- S3 multiscale helpers ---

struct s3_test_multiscale
{
  struct store* store;
  struct shard_pool* pool;
  struct ngff_multiscale* ms;
};

static int
s3_test_multiscale_open(struct s3_test_multiscale* z,
                        const char* prefix,
                        const char* array_name,
                        const struct dimension* dims,
                        uint8_t rank,
                        enum dtype data_type,
                        double fill_value,
                        int nlod)
{
  *z = (struct s3_test_multiscale){ 0 };

  struct store_s3_config scfg = {
    .bucket = S3_BUCKET,
    .prefix = prefix,
    .region = "us-east-1",
    .endpoint = s3_endpoint(),
  };
  store_s3_config_set_defaults(&scfg);

  z->store = store_s3_create(&scfg);
  CHECK(Fail, z->store);
  z->store->mkdirs(z->store, ".");

  uint64_t sc[MAX_ZARR_RANK], cps[MAX_ZARR_RANK];
  uint64_t sic = dims_compute_shard_geometry(dims, rank, sc, cps);

  z->pool = z->store->create_pool(z->store, sic);
  CHECK(Fail_store, z->pool);

  // Write root group
  CHECK(Fail_pool, zarr_write_group(z->store, "zarr.json", NULL) == 0);

  // Write intermediate groups for array_name
  if (array_name && array_name[0]) {
    CHECK(Fail_pool, z->store->mkdirs(z->store, array_name) == 0);
  }

  struct ngff_multiscale_config mscfg = {
    .data_type = data_type,
    .fill_value = fill_value,
    .rank = rank,
    .dimensions = dims,
    .nlod = nlod,
  };
  z->ms = ngff_multiscale_create(
    z->store, z->pool, array_name ? array_name : "", &mscfg);
  CHECK(Fail_pool, z->ms);
  return 0;

Fail_pool:
  z->pool->destroy(z->pool);
  z->pool = NULL;
Fail_store:
  z->store->destroy(z->store);
  z->store = NULL;
Fail:
  return 1;
}

static void
s3_test_multiscale_close(struct s3_test_multiscale* z)
{
  ngff_multiscale_destroy(z->ms);
  if (z->pool)
    z->pool->destroy(z->pool);
  if (z->store)
    z->store->destroy(z->store);
  *z = (struct s3_test_multiscale){ 0 };
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

  struct s3_test_sink sink;
  CHECK(Fail,
        s3_test_sink_open(&sink,
                          "test-meta",
                          "0",
                          dims,
                          3,
                          dtype_u32,
                          0,
                          (struct codec_config){ .id = CODEC_ZSTD }) == 0);

  // Verify root zarr.json (group metadata with "attributes")
  {
    char* gdata;
    CHECK(Fail_sink, check_group_zarr_json("test-meta", &gdata) == 0);
    free(gdata);
  }

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
  s3_test_sink_close(&sink);
  log_info("  PASS");
  return 0;

Fail_data:
  free(data);
Fail_sink:
  s3_test_sink_close(&sink);
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

  struct s3_test_sink sink;
  CHECK(Fail,
        s3_test_sink_open(&sink,
                          "test-shard",
                          "0",
                          dims,
                          1,
                          dtype_u16,
                          0,
                          (struct codec_config){ .id = CODEC_NONE }) == 0);

  struct shard_sink* ss = s3_test_sink_as_shard_sink(&sink);

  // shard_index=0 -> key "test-shard/0/c/0"
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
  s3_test_sink_close(&sink);
  log_info("  PASS");
  return 0;

Fail_data:
  free(data);
Fail_sink:
  s3_test_sink_close(&sink);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_concurrent_finalize(void)
{
  log_info("=== test_s3_concurrent_finalize ===");

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

  struct s3_test_sink sink;
  CHECK(Fail,
        s3_test_sink_open(&sink,
                          "test-concurrent",
                          "0",
                          dims,
                          3,
                          dtype_u16,
                          0,
                          (struct codec_config){ .id = CODEC_NONE }) == 0);

  struct shard_sink* ss = s3_test_sink_as_shard_sink(&sink);

  // Write epoch 0: open all 4 inner shards, write, finalize (async)
  uint16_t data[2] = { 0xAAAA, 0xBBBB };
  for (int i = 0; i < 4; ++i) {
    struct shard_writer* w = ss->open(ss, 0, (uint64_t)i);
    CHECK(Fail_sink, w);
    CHECK(Fail_sink, w->write(w, 0, data, (char*)data + sizeof(data)) == 0);
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
    CHECK(Fail_sink, w->write(w, 0, data2, (char*)data2 + sizeof(data2)) == 0);
    CHECK(Fail_sink, w->finalize(w) == 0);
  }

  // Wait fence drains all pending
  ss->wait_fence(ss, 0, fence);

  // Flush to drain epoch 1's pending uploads
  s3_test_sink_flush(&sink);

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
  s3_test_sink_close(&sink);
  log_info("  PASS");
  return 0;

Fail_sink:
  s3_test_sink_close(&sink);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_multiscale_metadata(void)
{
  log_info("=== test_s3_multiscale_metadata ===");

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

  set_s3_creds();

  struct s3_test_multiscale ms;
  CHECK(Fail,
        s3_test_multiscale_open(
          &ms, "test-multiscale", "", dims, 3, dtype_u16, 0, 0) == 0);

  // Check root zarr.json has multiscales attribute
  {
    char* data;
    CHECK(Fail2, check_group_zarr_json("test-multiscale", &data) == 0);
    int ok = strstr(data, "\"ome\"") && strstr(data, "\"multiscales\"") &&
             strstr(data, "\"version\":\"0.5\"") &&
             strstr(data, "\"path\":\"0\"") && strstr(data, "\"path\":\"1\"") &&
             strstr(data, "\"coordinateTransformations\"");
    free(data);
    CHECK(Fail2, ok);
  }

  // Check L0 array zarr.json
  CHECK(Fail2,
        check_array_json("test-multiscale/0/zarr.json",
                         "\"shape\":[64,32,64]",
                         "\"data_type\":\"uint16\"") == 0);

  // Check L1 array zarr.json (x halved; z excluded from LOD mask by downsample)
  CHECK(Fail2,
        check_array_json(
          "test-multiscale/1/zarr.json", "\"shape\":[64,32,32]", NULL) == 0);
  s3_test_multiscale_close(&ms);
  log_info("  PASS");
  return 0;

Fail2:
  s3_test_multiscale_close(&ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: multiscale metadata with array_name ---

static int
test_multiscale_metadata_named(void)
{
  log_info("=== test_s3_multiscale_metadata_named ===");

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

  set_s3_creds();

  struct s3_test_multiscale ms;
  CHECK(Fail,
        s3_test_multiscale_open(
          &ms, "test-ms-named", "ms", dims, 3, dtype_u16, 0, 0) == 0);

  // Root zarr.json should be a plain group (attributes:{})
  {
    char* data;
    CHECK(Fail_sink, check_group_zarr_json("test-ms-named", &data) == 0);
    int ok = strstr(data, "\"attributes\":{}") != NULL;
    free(data);
    CHECK(Fail_sink, ok);
  }

  // Sub-group zarr.json should have OME multiscales
  {
    char* data;
    CHECK(Fail_sink, check_group_zarr_json("test-ms-named/ms", &data) == 0);
    int ok = strstr(data, "\"attributes\":{\"ome\"") != NULL &&
             strstr(data, "\"multiscales\"") != NULL;
    free(data);
    CHECK(Fail_sink, ok);
  }

  // L0 array zarr.json
  CHECK(Fail_sink, s3_exists("test-ms-named/ms/0/zarr.json"));

  s3_test_multiscale_close(&ms);
  log_info("  PASS");
  return 0;

Fail_sink:
  s3_test_multiscale_close(&ms);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Main ---

// --- Test: S3 part count validation (no network needed) ---

static int
test_s3_validate_part_count(void)
{
  log_info("=== test_s3_validate_part_count ===");

  // Small shard: should pass
  struct dimension small_dims[] = {
    { .size = 64, .chunk_size = 64, .chunks_per_shard = 1 },
  };
  CHECK(Fail,
        store_s3_validate_part_count(
          1, small_dims, dtype_u16, 8 * 1024 * 1024) == 0);

  // Huge shard with tiny part size: should fail (too many parts)
  struct dimension big_dims[] = {
    { .size = 65536, .chunk_size = 65536, .chunks_per_shard = 1 },
    { .size = 65536, .chunk_size = 65536, .chunks_per_shard = 1 },
  };
  // 65536^2 * 2 bytes = 8 GiB shard, 1 KiB parts → 8M parts > 10000
  CHECK(Fail, store_s3_validate_part_count(2, big_dims, dtype_u16, 1024) != 0);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: S3 config defaults ---

static int
test_s3_config_defaults(void)
{
  log_info("=== test_s3_config_defaults ===");

  struct store_s3_config cfg = { 0 };
  store_s3_config_set_defaults(&cfg);
  CHECK(Fail, cfg.part_size == 8 * 1024 * 1024);
  CHECK(Fail, cfg.throughput_gbps == 10.0);

  // Already-set values should not be overwritten
  struct store_s3_config cfg2 = { .part_size = 42, .throughput_gbps = 1.0 };
  store_s3_config_set_defaults(&cfg2);
  CHECK(Fail, cfg2.part_size == 42);
  CHECK(Fail, cfg2.throughput_gbps == 1.0);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  // Tests that don't need S3
  int rc = 0;
  rc |= test_s3_validate_part_count();
  rc |= test_s3_config_defaults();

  // Tests that need minio
  if (s3_setup() != 0) {
    log_error("S3 not available — is minio running?");
    log_error("  docker compose up minio");
    return rc ? rc : 1;
  }

  rc |= test_metadata();
  rc |= test_shard_write();
  rc |= test_concurrent_finalize();
  rc |= test_multiscale_metadata();
  rc |= test_multiscale_metadata_named();
  return rc;
}

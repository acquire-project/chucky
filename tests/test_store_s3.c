#include "platform/platform_cmd.h"
#include "store.h"
#include "util/prelude.h"
#include "zarr/store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define S3_BUCKET "chucky-test-store"
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
s3_setup(void)
{
  char cmd[4096];
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

static struct store*
make_store(const char* prefix)
{
  struct store_s3_config scfg = {
    .bucket = S3_BUCKET,
    .prefix = prefix,
    .region = "us-east-1",
    .endpoint = s3_endpoint(),
  };
  store_s3_config_set_defaults(&scfg);
  return store_s3_create(&scfg);
}

static int
test_s3_has_existing_data_empty(void)
{
  log_info("=== test_s3_has_existing_data_empty ===");

  struct store* s = make_store(NULL);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_s3_has_existing_data_after_put(void)
{
  log_info("=== test_s3_has_existing_data_after_put ===");

  struct store* s = make_store(NULL);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);

  const char* meta = "{}";
  CHECK(Fail2, s->put(s, "zarr.json", meta, strlen(meta)) == 0);
  CHECK(Fail2, store_has_existing_data(s) == 1);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_s3_has_existing_data_with_prefix(void)
{
  log_info("=== test_s3_has_existing_data_with_prefix ===");

  const char* mine = "tenantA/dataset";
  const char* other = "tenantB/dataset";

  // Put zarr.json at the OTHER prefix; the store at `mine` must still see 0.
  struct store* other_store = make_store(other);
  CHECK(Fail, other_store);
  const char* meta = "{}";
  CHECK(Fail_other, other_store->put(other_store, "zarr.json", meta,
                                     strlen(meta)) == 0);
  store_destroy(other_store);

  struct store* s = make_store(mine);
  CHECK(Fail, s);
  CHECK(Fail_s, store_has_existing_data(s) == 0);

  // Now put at `mine`; it must flip to 1.
  CHECK(Fail_s, s->put(s, "zarr.json", meta, strlen(meta)) == 0);
  CHECK(Fail_s, store_has_existing_data(s) == 1);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail_s:
  store_destroy(s);
  return 1;
Fail_other:
  store_destroy(other_store);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  set_s3_creds();

  if (s3_setup() != 0) {
    log_error("S3 not available — is minio running?");
    log_error("  docker compose up minio");
    return 1;
  }

  int rc = 0;
  rc |= test_s3_has_existing_data_empty();
  rc |= test_s3_has_existing_data_after_put();
  rc |= test_s3_has_existing_data_with_prefix();
  return rc;
}

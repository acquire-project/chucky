#include "dimension.h"
#include "ngff.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[4096];

static int
make_tmpdir(void)
{
  return test_tmpdir_create(tmpdir, sizeof(tmpdir));
}

static char*
read_file(const char* key)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, key);
  FILE* f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  char* buf = (char*)malloc((size_t)sz + 1);
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[n] = '\0';
  return buf;
}

static int
contains(const char* h, const char* n)
{
  return strstr(h, n) != NULL;
}

static int
test_ngff_set_and_flush(void)
{
  log_info("=== test_ngff_set_and_flush ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "y",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 4,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };
  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .nlod = 2,
  };

  CHECK(Fail2, s->mkdirs(s, "ms") == 0);
  struct ngff_multiscale* ms = ngff_multiscale_create(s, "ms", &cfg);
  CHECK(Fail2, ms);

  CHECK(Fail3, ngff_multiscale_set_attribute(ms, "custom", "{\"a\":1}") == 0);
  CHECK(Fail3,
        ngff_multiscale_set_attribute(ms, "tag", "[\"alpha\",\"beta\"]") == 0);
  CHECK(Fail3, ngff_multiscale_flush_metadata(ms) == 0);

  char* out = read_file("ms/zarr.json");
  CHECK(Fail3, out);

  // OME block still present.
  CHECK(Fail_out, contains(out, "\"ome\""));
  CHECK(Fail_out, contains(out, "\"multiscales\""));
  // Custom attrs visible alongside the OME block (inside attributes).
  CHECK(Fail_out, contains(out, "\"custom\":{\"a\":1}"));
  CHECK(Fail_out, contains(out, "\"tag\":[\"alpha\",\"beta\"]"));

  free(out);
  ngff_multiscale_destroy(ms);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  ngff_multiscale_destroy(ms);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_ngff_destroy_flushes(void)
{
  log_info("=== test_ngff_destroy_flushes ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };
  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u8,
    .rank = 2,
    .dimensions = dims,
    .nlod = 1,
  };

  CHECK(Fail2, s->mkdirs(s, "ms2") == 0);
  struct ngff_multiscale* ms = ngff_multiscale_create(s, "ms2", &cfg);
  CHECK(Fail2, ms);

  // set, but don't flush explicitly — destroy should flush.
  CHECK(Fail3, ngff_multiscale_set_attribute(ms, "note", "\"x\"") == 0);
  ngff_multiscale_destroy(ms);
  ms = NULL;

  char* out = read_file("ms2/zarr.json");
  CHECK(Fail2, out);
  CHECK(Fail_out, contains(out, "\"note\":\"x\""));

  free(out);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
  goto Fail2;
Fail3:
  ngff_multiscale_destroy(ms);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_ngff_reject_bad(void)
{
  log_info("=== test_ngff_reject_bad ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct dimension dims[2] = {
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };
  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u8, .rank = 2, .dimensions = dims, .nlod = 1,
  };
  CHECK(Fail2, s->mkdirs(s, "ms3") == 0);
  struct ngff_multiscale* ms = ngff_multiscale_create(s, "ms3", &cfg);
  CHECK(Fail2, ms);

  CHECK(Fail3, ngff_multiscale_set_attribute(ms, "bad", "{oops") != 0);
  CHECK(Fail3, ngff_multiscale_set_attribute(ms, "", "1") != 0);
  CHECK(Fail3, ngff_multiscale_set_attribute(ms, "a\"b", "1") != 0);

  ngff_multiscale_destroy(ms);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  ngff_multiscale_destroy(ms);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  if (make_tmpdir())
    return 1;
  log_info("tmpdir: %s", tmpdir);

  int err = 0;
  err |= test_ngff_set_and_flush();
  err |= test_ngff_destroy_flushes();
  err |= test_ngff_reject_bad();

  test_tmpdir_remove(tmpdir);
  return err;
}

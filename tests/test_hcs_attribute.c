#include "dimension.h"
#include "hcs.h"
#include "ngff.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"

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

static struct hcs_plate*
make_plate(struct store* s)
{
  static struct dimension dims[2] = {
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .name = "y",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 16,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };
  static struct hcs_plate_config cfg = {
    .name = "p",
    .rows = 2,
    .cols = 2,
    .field_count = 2,
    .fov = { .data_type = dtype_u8,
             .rank = 2,
             .dimensions = dims,
             .nlod = 1 },
  };
  return hcs_plate_create(s, &cfg);
}

static int
test_hcs_plate_attr(void)
{
  log_info("=== test_hcs_plate_attr ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  struct hcs_plate* p = make_plate(s);
  CHECK(Fail2, p);

  CHECK(Fail3, hcs_plate_set_attribute(p, "lab", "\"ACME\"") == 0);
  CHECK(Fail3, hcs_plate_flush_metadata(p) == 0);

  char* out = read_file("p/zarr.json");
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"plate\""));
  CHECK(Fail_out, contains(out, "\"lab\":\"ACME\""));

  free(out);
  hcs_plate_destroy(p);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_out:
  free(out);
Fail3:
  hcs_plate_destroy(p);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_hcs_well_attr(void)
{
  log_info("=== test_hcs_well_attr ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  struct hcs_plate* p = make_plate(s);
  CHECK(Fail2, p);

  CHECK(Fail3,
        hcs_plate_set_well_attribute(p, 0, 1, "barcode", "\"W1\"") == 0);
  CHECK(Fail3, hcs_plate_flush_metadata(p) == 0);

  // Well group for row 0, col 1 = A/2
  char* out = read_file("p/A/2/zarr.json");
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"well\""));
  CHECK(Fail_out, contains(out, "\"barcode\":\"W1\""));

  // Other wells should NOT have it.
  char* other = read_file("p/A/1/zarr.json");
  CHECK(Fail_out, other);
  CHECK(Fail_other, !contains(other, "\"barcode\""));

  free(other);
  free(out);
  hcs_plate_destroy(p);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_other:
  free(other);
Fail_out:
  free(out);
Fail3:
  hcs_plate_destroy(p);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_hcs_fov_attr(void)
{
  log_info("=== test_hcs_fov_attr ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  struct hcs_plate* p = make_plate(s);
  CHECK(Fail2, p);

  CHECK(Fail3,
        hcs_plate_set_fov_attribute(p, 1, 1, 0, "exposure_ms", "42") == 0);
  CHECK(Fail3, hcs_plate_flush_metadata(p) == 0);

  // FOV group for row 1, col 1, fov 0 = B/2/0
  char* out = read_file("p/B/2/0/zarr.json");
  CHECK(Fail3, out);
  CHECK(Fail_out, contains(out, "\"multiscales\""));
  CHECK(Fail_out, contains(out, "\"exposure_ms\":42"));

  // Other fov (B/2/1) should not have it.
  char* other = read_file("p/B/2/1/zarr.json");
  CHECK(Fail_out, other);
  CHECK(Fail_other, !contains(other, "\"exposure_ms\""));

  free(other);
  free(out);
  hcs_plate_destroy(p);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_other:
  free(other);
Fail_out:
  free(out);
Fail3:
  hcs_plate_destroy(p);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_hcs_destroy_flushes(void)
{
  log_info("=== test_hcs_destroy_flushes ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  struct hcs_plate* p = make_plate(s);
  CHECK(Fail2, p);

  CHECK(Fail3, hcs_plate_set_attribute(p, "k", "1") == 0);
  CHECK(Fail3, hcs_plate_set_well_attribute(p, 0, 0, "k", "2") == 0);
  // No flush — let destroy do it.
  hcs_plate_destroy(p);
  p = NULL;

  char* plate_j = read_file("p/zarr.json");
  CHECK(Fail2, plate_j);
  CHECK(Fail_pj, contains(plate_j, "\"k\":1"));
  free(plate_j);

  char* well_j = read_file("p/A/1/zarr.json");
  CHECK(Fail2, well_j);
  CHECK(Fail_wj, contains(well_j, "\"k\":2"));
  free(well_j);

  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail_wj:
  free(well_j);
  goto Fail2;
Fail_pj:
  free(plate_j);
  goto Fail2;
Fail3:
  hcs_plate_destroy(p);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_hcs_reject_bad(void)
{
  log_info("=== test_hcs_reject_bad ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  struct hcs_plate* p = make_plate(s);
  CHECK(Fail2, p);

  // Malformed.
  CHECK(Fail3, hcs_plate_set_attribute(p, "bad", "{oops") != 0);
  // Bad row/col.
  CHECK(Fail3, hcs_plate_set_well_attribute(p, -1, 0, "k", "1") != 0);
  CHECK(Fail3, hcs_plate_set_well_attribute(p, 0, 99, "k", "1") != 0);
  CHECK(Fail3, hcs_plate_set_fov_attribute(p, 0, 0, 99, "k", "1") != 0);

  hcs_plate_destroy(p);
  store_destroy(s);
  log_info("  PASS");
  return 0;
Fail3:
  hcs_plate_destroy(p);
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
  err |= test_hcs_plate_attr();
  err |= test_hcs_well_attr();
  err |= test_hcs_fov_attr();
  err |= test_hcs_destroy_flushes();
  err |= test_hcs_reject_bad();

  test_tmpdir_remove(tmpdir);
  return err;
}

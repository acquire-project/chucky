// Validate OME-NGFF metadata using ome-zarr-models.
// Writes multiscale and HCS stores, then runs a Python validator.

#include "dimension.h"
#include "hcs.h"
#include "ngff.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[4096];

static int
make_tmpdir(void)
{
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  return 0;
Fail:
  return 1;
}

// --- Write a multiscale store ---

static int
write_multiscale_store(const char* path)
{
  struct store* store = store_fs_create(path, 0);
  CHECK(Fail, store);

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .name = "t",
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 16,
      .chunks_per_shard = 2,
      .name = "y",
      .downsample = 1,
      .storage_position = 1 },
    { .size = 64,
      .chunk_size = 16,
      .chunks_per_shard = 2,
      .name = "x",
      .downsample = 1,
      .storage_position = 2 },
  };

  struct ngff_axis axes[] = {
    { .type = ngff_axis_time, .unit = "second", .scale = 1.0 },
    { .type = ngff_axis_space, .unit = "micrometer", .scale = 0.5 },
    { .type = ngff_axis_space, .unit = "micrometer", .scale = 0.5 },
  };

  // Ensure store root exists
  CHECK(Fail2, store->mkdirs(store, ".") == 0);

  // Write root group
  CHECK(Fail2, zarr_write_group(store, "zarr.json", NULL) == 0);

  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .axes = axes,
  };

  struct ngff_multiscale* ms =
    ngff_multiscale_create(store, "multiscale", &cfg);
  CHECK(Fail2, ms);

  ngff_multiscale_destroy(ms);
  store_destroy(store);
  return 0;

Fail2:
  store_destroy(store);
Fail:
  return 1;
}

// --- Write an HCS store ---

static int
write_hcs_store(const char* path)
{
  struct store* store = store_fs_create(path, 0);
  CHECK(Fail, store);

  struct dimension dims[] = {
    { .size = 32,
      .chunk_size = 16,
      .chunks_per_shard = 1,
      .name = "y",
      .downsample = 1,
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 16,
      .chunks_per_shard = 1,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };

  struct ngff_axis axes[] = {
    { .type = ngff_axis_space, .unit = "micrometer", .scale = 0.5 },
    { .type = ngff_axis_space, .unit = "micrometer", .scale = 0.5 },
  };

  CHECK(Fail2, store->mkdirs(store, ".") == 0);

  struct hcs_plate_config cfg = {
    .name = "plate",
    .rows = 2,
    .cols = 3,
    .field_count = 1,
    .fov =
      {
        .data_type = dtype_u16,
        .rank = 2,
        .dimensions = dims,
        .axes = axes,
      },
  };

  struct hcs_plate* plate = hcs_plate_create(store, &cfg);
  CHECK(Fail2, plate);

  hcs_plate_destroy(plate);
  store_destroy(store);
  return 0;

Fail2:
  store_destroy(store);
Fail:
  return 1;
}

// --- Main ---

int
main(void)
{
  if (system("uv --version > /dev/null 2>&1") != 0) {
    log_error("uv not found — install it: https://docs.astral.sh/uv/");
    return 1;
  }

  if (make_tmpdir())
    return 1;
  log_info("tmpdir: %s", tmpdir);

  // Write stores
  char ms_path[4096], hcs_path[4096];
  snprintf(ms_path, sizeof(ms_path), "%s/multiscale.zarr", tmpdir);
  snprintf(hcs_path, sizeof(hcs_path), "%s/hcs.zarr", tmpdir);

  CHECK(Fail, write_multiscale_store(ms_path) == 0);
  log_info("wrote multiscale store: %s", ms_path);

  CHECK(Fail, write_hcs_store(hcs_path) == 0);
  log_info("wrote HCS store: %s", hcs_path);

  // Validate with ome-zarr-models
  char cmd[8192];
  // Validate: point at the NGFF group (multiscale/) and HCS plate (plate/)
  snprintf(cmd,
           sizeof(cmd),
           "uv run " SOURCE_DIR
           "/tests/validate_ome_ngff.py %s:multiscale %s:plate",
           ms_path,
           hcs_path);
  log_info("running: %s", cmd);
  int rc = system(cmd);
  CHECK(Fail, rc == 0);

  log_info("PASS: all stores validated");

  // Cleanup
  test_tmpdir_remove(tmpdir);
  return 0;

Fail:
  log_error("FAIL");
  test_tmpdir_remove(tmpdir);
  return 1;
}

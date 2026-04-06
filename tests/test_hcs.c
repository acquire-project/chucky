#include "dimension.h"
#include "hcs/hcs.h"
#include "hcs/hcs_metadata.h"
#include "ngff/ngff_axis.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"

#include <stdio.h>
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

static int
read_file(const char* path, char* buf, size_t cap, size_t* out_len)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return 1;
  *out_len = fread(buf, 1, cap - 1, f);
  fclose(f);
  buf[*out_len] = '\0';
  return 0;
}

// --- Test: plate metadata JSON ---

static int
test_plate_metadata(void)
{
  log_info("=== test_plate_metadata ===");

  char buf[8192];
  int len =
    hcs_plate_attributes_json(buf, sizeof(buf), "myplate", 2, 3, NULL, 2, NULL);
  CHECK(Fail, len > 0);
  buf[len] = '\0';

  CHECK(Fail, strstr(buf, "\"name\":\"myplate\""));
  CHECK(Fail, strstr(buf, "\"field_count\":2"));
  CHECK(Fail, strstr(buf, "\"version\":\"0.5\""));
  // Check rows A, B
  CHECK(Fail, strstr(buf, "{\"name\":\"A\"}"));
  CHECK(Fail, strstr(buf, "{\"name\":\"B\"}"));
  // Check columns 1, 2, 3
  CHECK(Fail, strstr(buf, "{\"name\":\"1\"}"));
  CHECK(Fail, strstr(buf, "{\"name\":\"3\"}"));
  // Check a well path
  CHECK(Fail, strstr(buf, "\"path\":\"A/1\""));
  CHECK(Fail, strstr(buf, "\"path\":\"B/3\""));

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL: %.*s", len > 0 ? len : 0, buf);
  return 1;
}

// --- Test: well metadata JSON ---

static int
test_well_metadata(void)
{
  log_info("=== test_well_metadata ===");

  char buf[4096];
  int len = hcs_well_attributes_json(buf, sizeof(buf), 3);
  CHECK(Fail, len > 0);
  buf[len] = '\0';

  CHECK(Fail, strstr(buf, "\"well\""));
  CHECK(Fail, strstr(buf, "\"images\""));
  CHECK(Fail, strstr(buf, "{\"path\":\"0\",\"acquisition\":0}"));
  CHECK(Fail, strstr(buf, "{\"path\":\"2\",\"acquisition\":0}"));

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL: %.*s", len > 0 ? len : 0, buf);
  return 1;
}

// --- Test: full HCS plate creation ---

static int
test_hcs_plate_create(void)
{
  log_info("=== test_hcs_plate_create ===");

  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);

  // Simple 2D array: 32x32, chunk 8, 2 shards per dim
  struct dimension dims[] = {
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

  struct ngff_axis axes[] = {
    { .unit = "micrometer", .scale = 0.5 },
    { .unit = "micrometer", .scale = 0.5 },
  };

  // Pool sized for L0: 2 shards per dim = 4 inner shards
  struct shard_pool* pool = store->create_pool(store, 4);
  CHECK(Fail2, pool);

  struct hcs_plate_config cfg = {
    .name = "plate1",
    .rows = 2,
    .cols = 2,
    .field_count = 1,
    .fov =
      {
        .data_type = dtype_u16,
        .fill_value = 0,
        .rank = 2,
        .dimensions = dims,
        .nlod = 0,
        .axes = axes,
      },
  };

  struct hcs_plate* plate = hcs_plate_create(store, pool, &cfg);
  CHECK(Fail3, plate);

  // Verify hierarchy exists
  char path[4096], buf[8192];
  size_t len;

  // Root group
  snprintf(path, sizeof(path), "%s/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"node_type\":\"group\""));

  // Plate group with OME attrs
  snprintf(path, sizeof(path), "%s/plate1/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"plate\""));
  CHECK(Fail4, strstr(buf, "\"name\":\"plate1\""));
  CHECK(Fail4, strstr(buf, "\"field_count\":1"));

  // Row group
  snprintf(path, sizeof(path), "%s/plate1/A/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"node_type\":\"group\""));

  // Well group with OME attrs
  snprintf(path, sizeof(path), "%s/plate1/A/1/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"well\""));
  CHECK(Fail4, strstr(buf, "\"images\""));

  // FOV multiscale group
  snprintf(path, sizeof(path), "%s/plate1/A/1/0/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"multiscales\""));

  // FOV L0 array
  snprintf(path, sizeof(path), "%s/plate1/A/1/0/0/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"node_type\":\"array\""));
  CHECK(Fail4, strstr(buf, "\"shape\":[32,32]"));

  // Get shard_sink for FOV A/1/0
  struct shard_sink* fov = hcs_plate_fov_sink(plate, 0, 0, 0);
  CHECK(Fail4, fov);

  // Another FOV
  struct shard_sink* fov_b2 = hcs_plate_fov_sink(plate, 1, 1, 0);
  CHECK(Fail4, fov_b2);
  CHECK(Fail4, fov != fov_b2); // different FOVs

  // Out of range returns NULL
  CHECK(Fail4, hcs_plate_fov_sink(plate, 5, 0, 0) == NULL);

  hcs_plate_destroy(plate);
  pool->destroy(pool);
  store->destroy(store);
  log_info("  PASS");
  return 0;

Fail4:
  hcs_plate_destroy(plate);
Fail3:
  pool->destroy(pool);
Fail2:
  store->destroy(store);
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
  err |= test_plate_metadata();
  err |= test_well_metadata();
  err |= test_hcs_plate_create();

  test_tmpdir_remove(tmpdir);

  return err;
}

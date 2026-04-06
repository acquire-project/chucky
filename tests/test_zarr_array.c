#include "dimension.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/store.h"

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

// --- Test: create writes zarr.json ---

static int
test_zarr_array_metadata(void)
{
  log_info("=== test_zarr_array_metadata ===");

  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  CHECK(Fail2, store->mkdirs(store, "myarray") == 0);

  struct dimension dims[2] = {
    { .size = 64, .chunk_size = 16, .name = "y" },
    { .size = 128, .chunk_size = 32, .name = "x" },
  };
  struct zarr_array_config cfg = {
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 2,
    .dimensions = dims,
  };

  struct zarr_array* a = zarr_array_create(store, "myarray", &cfg);
  CHECK(Fail2, a);

  // Verify zarr.json exists and contains expected fields
  char path[4096];
  snprintf(path, sizeof(path), "%s/myarray/zarr.json", tmpdir);
  char buf[4096];
  size_t len;
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"node_type\":\"array\""));
  CHECK(Fail3, strstr(buf, "\"zarr_format\":3"));
  CHECK(Fail3, strstr(buf, "\"shape\":[64,128]"));

  zarr_array_destroy(a);
  store_destroy(store);
  log_info("  PASS");
  return 0;

Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(store);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: open + write shards ---

static int
test_zarr_array_shard_write(void)
{
  log_info("=== test_zarr_array_shard_write ===");

  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);

  CHECK(Fail2, store->mkdirs(store, "arr1d") == 0);

  // 1D array: 8 elements, chunk_size=4, 1 chunk per shard, 2 shards
  struct dimension dims[1] = {
    { .size = 8, .chunk_size = 4, .chunks_per_shard = 1, .name = "x" },
  };

  struct zarr_array_config cfg = {
    .data_type = dtype_u8,
    .fill_value = 0,
    .rank = 1,
    .dimensions = dims,
  };

  struct zarr_array* a = zarr_array_create(store, "arr1d", &cfg);
  CHECK(Fail2, a);

  struct shard_sink* sink = zarr_array_as_shard_sink(a);

  // Write shard 0
  struct shard_writer* w = sink->open(sink, 0, 0);
  CHECK(Fail3, w);
  uint8_t data0[4] = { 1, 2, 3, 4 };
  CHECK(Fail3, w->write(w, 0, data0, data0 + 4) == 0);
  CHECK(Fail3, w->finalize(w) == 0);

  // Write shard 1
  w = sink->open(sink, 0, 1);
  CHECK(Fail3, w);
  uint8_t data1[4] = { 5, 6, 7, 8 };
  CHECK(Fail3, w->write(w, 0, data1, data1 + 4) == 0);
  CHECK(Fail3, w->finalize(w) == 0);

  CHECK(Fail3, zarr_array_flush(a) == 0);
  CHECK(Fail3, zarr_array_has_error(a) == 0);

  // Verify shard files exist
  char path[4096];
  snprintf(path, sizeof(path), "%s/arr1d/c/0", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail3, f);
  fclose(f);

  snprintf(path, sizeof(path), "%s/arr1d/c/1", tmpdir);
  f = fopen(path, "rb");
  CHECK(Fail3, f);
  fclose(f);

  zarr_array_destroy(a);
  store_destroy(store);
  log_info("  PASS");
  return 0;

Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(store);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: update_append rewrites metadata ---

static int
test_zarr_array_update_append(void)
{
  log_info("=== test_zarr_array_update_append ===");

  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);

  CHECK(Fail2, store->mkdirs(store, "stream") == 0);

  // Unbounded dim 0
  struct dimension dims[2] = {
    { .size = 0, .chunk_size = 1, .chunks_per_shard = 4, .name = "t" },
    { .size = 64, .chunk_size = 16, .name = "x" },
  };

  struct zarr_array_config cfg = {
    .data_type = dtype_f32,
    .fill_value = 0,
    .rank = 2,
    .dimensions = dims,
  };

  struct zarr_array* a = zarr_array_create(store, "stream", &cfg);
  CHECK(Fail2, a);

  // Initial zarr.json should have shape [0, 64]
  char path[4096];
  snprintf(path, sizeof(path), "%s/stream/zarr.json", tmpdir);
  char buf[4096];
  size_t len;
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"shape\":[0,64]"));

  // Update append dim
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  uint64_t new_sizes[1] = { 10 };
  CHECK(Fail3, sink->update_append(sink, 0, 1, new_sizes) == 0);

  // Re-read and verify shape changed
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"shape\":[10,64]"));

  zarr_array_destroy(a);
  store_destroy(store);
  log_info("  PASS");
  return 0;

Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(store);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: root array (empty prefix) ---

static int
test_zarr_array_root(void)
{
  log_info("=== test_zarr_array_root ===");

  // Create a subdirectory for this test
  char root[4096];
  snprintf(root, sizeof(root), "%s/rootarr", tmpdir);

  struct store* store = store_fs_create(root, 0);
  CHECK(Fail, store);
  CHECK(Fail2, store->mkdirs(store, ".") == 0);

  struct dimension dims[1] = {
    { .size = 4, .chunk_size = 4, .chunks_per_shard = 1, .name = "x" },
  };

  struct zarr_array_config cfg = {
    .data_type = dtype_u8,
    .rank = 1,
    .dimensions = dims,
  };

  // Empty prefix: writes zarr.json at store root
  struct zarr_array* a = zarr_array_create(store, "", &cfg);
  CHECK(Fail2, a);

  char path[4096];
  snprintf(path, sizeof(path), "%s/rootarr/zarr.json", tmpdir);
  char buf[4096];
  size_t len;
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"node_type\":\"array\""));

  // Open a shard with empty prefix — exercises the no-prefix key path
  struct shard_sink* sink = zarr_array_as_shard_sink(a);
  struct shard_writer* w = sink->open(sink, 0, 0);
  CHECK(Fail3, w);
  uint8_t byte = 0x42;
  CHECK(Fail3, w->write(w, 0, &byte, &byte + 1) == 0);
  CHECK(Fail3, w->finalize(w) == 0);
  CHECK(Fail3, zarr_array_flush(a) == 0);

  zarr_array_destroy(a);
  store_destroy(store);
  log_info("  PASS");
  return 0;

Fail3:
  zarr_array_destroy(a);
Fail2:
  store_destroy(store);
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
  err |= test_zarr_array_metadata();
  err |= test_zarr_array_shard_write();
  err |= test_zarr_array_update_append();
  err |= test_zarr_array_root();

  test_tmpdir_remove(tmpdir);

  return err;
}

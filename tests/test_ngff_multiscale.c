#include "dimension.h"
#include "ngff.h"
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

// --- Test: basic multiscale creation ---

static int
test_multiscale_create(void)
{
  log_info("=== test_multiscale_create ===");

  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);

  struct dimension dims[] = {
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

  struct ngff_axis axes[] = {
    { .unit = "micrometer", .scale = 0.5 },
    { .unit = "micrometer", .scale = 0.5 },
  };

  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 2,
    .dimensions = dims,
    .nlod = 0, // auto
    .axes = axes,
  };

  // Parent writes root group
  CHECK(Fail2, zarr_write_group(store, "zarr.json", NULL) == 0);

  struct ngff_multiscale* ms = ngff_multiscale_create(store, "ms", &cfg);
  CHECK(Fail2, ms);

  // Verify group zarr.json has multiscales attribute
  char path[4096];
  snprintf(path, sizeof(path), "%s/ms/zarr.json", tmpdir);
  char buf[8192];
  size_t len;
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"multiscales\""));
  CHECK(Fail3, strstr(buf, "\"version\":\"0.5\""));
  CHECK(Fail3, strstr(buf, "\"unit\":\"micrometer\""));
  CHECK(Fail3, strstr(buf, "\"scale\":[0.5,0.5]"));

  // Verify per-level array metadata exists
  snprintf(path, sizeof(path), "%s/ms/0/zarr.json", tmpdir);
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"node_type\":\"array\""));
  CHECK(Fail3, strstr(buf, "\"shape\":[64,64]"));

  // L1: only x is downsampled (y is append dim), so shape=[64,32]
  snprintf(path, sizeof(path), "%s/ms/1/zarr.json", tmpdir);
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"shape\":[64,32]"));

  // Verify root group exists
  snprintf(path, sizeof(path), "%s/zarr.json", tmpdir);
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"node_type\":\"group\""));

  ngff_multiscale_destroy(ms);
  store_destroy(store);
  log_info("  PASS");
  return 0;

Fail3:
  ngff_multiscale_destroy(ms);
Fail2:
  store_destroy(store);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Test: open shards and update_append through shard_sink ---

static int
test_multiscale_shard_sink(void)
{
  log_info("=== test_multiscale_shard_sink ===");

  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  store->mkdirs(store, ".");

  // Unbounded dim 0 so update_append is meaningful
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 4,
      .name = "t",
      .storage_position = 0 },
    { .size = 32,
      .chunk_size = 16,
      .chunks_per_shard = 1,
      .name = "x",
      .downsample = 1,
      .storage_position = 1 },
  };

  CHECK(Fail2, zarr_write_group(store, "zarr.json", NULL) == 0);

  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
  };

  struct ngff_multiscale* ms = ngff_multiscale_create(store, "ms", &cfg);
  CHECK(Fail2, ms);

  struct shard_sink* sink = ngff_multiscale_as_shard_sink(ms);
  CHECK(Fail3, sink);

  // Open a shard on level 0, write some data, finalize
  struct shard_writer* w = sink->open(sink, 0, 0);
  CHECK(Fail3, w);
  uint8_t data[32];
  memset(data, 0xAA, sizeof(data));
  CHECK(Fail3, w->write(w, 0, data, data + sizeof(data)) == 0);
  CHECK(Fail3, w->finalize(w) == 0);

  // update_append: extend dim 0 from 0 to 4
  uint64_t new_sizes[1] = { 4 };
  CHECK(Fail3, sink->update_append(sink, 0, 1, new_sizes) == 0);

  // Verify L0 array zarr.json was updated
  char path[4096];
  snprintf(path, sizeof(path), "%s/ms/0/zarr.json", tmpdir);
  char buf[4096];
  size_t len;
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"shape\":[4,32]"));

  // Verify group zarr.json was updated (multiscales metadata)
  snprintf(path, sizeof(path), "%s/ms/zarr.json", tmpdir);
  CHECK(Fail3, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail3, strstr(buf, "\"multiscales\""));

  // Flush pending I/O
  CHECK(Fail3, ngff_multiscale_flush(ms) == 0);

  ngff_multiscale_destroy(ms);
  store_destroy(store);
  log_info("  PASS");
  return 0;

Fail3:
  ngff_multiscale_destroy(ms);
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
  err |= test_multiscale_create();
  err |= test_multiscale_shard_sink();

  test_tmpdir_remove(tmpdir);

  return err;
}

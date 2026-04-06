#include "dimension.h"
#include "ngff/ngff_axis.h"
#include "ngff/ngff_multiscale.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"
#include "zarr/zarr_group.h"

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

  // Compute L0 shard_inner_count for pool sizing
  // 2 dims, 8 chunks each, 4 cps → 2 shards each → 4 inner shards
  struct shard_pool* pool = store->create_pool(store, 4);
  CHECK(Fail2, pool);

  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .fill_value = 0,
    .rank = 2,
    .dimensions = dims,
    .nlod = 0, // auto
    .axes = axes,
  };

  // Parent writes root group
  CHECK(Fail3, zarr_write_group(store, "zarr.json", NULL) == 0);

  struct ngff_multiscale* ms = ngff_multiscale_create(store, pool, "ms", &cfg);
  CHECK(Fail3, ms);

  // Verify group zarr.json has multiscales attribute
  char path[4096];
  snprintf(path, sizeof(path), "%s/ms/zarr.json", tmpdir);
  char buf[8192];
  size_t len;
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"multiscales\""));
  CHECK(Fail4, strstr(buf, "\"version\":\"0.5\""));
  CHECK(Fail4, strstr(buf, "\"unit\":\"micrometer\""));
  CHECK(Fail4, strstr(buf, "\"scale\":[0.5,0.5]"));

  // Verify per-level array metadata exists
  snprintf(path, sizeof(path), "%s/ms/0/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"node_type\":\"array\""));
  CHECK(Fail4, strstr(buf, "\"shape\":[64,64]"));

  // L1: only x is downsampled (y is append dim), so shape=[64,32]
  snprintf(path, sizeof(path), "%s/ms/1/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"shape\":[64,32]"));

  // Verify root group exists
  snprintf(path, sizeof(path), "%s/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"node_type\":\"group\""));

  ngff_multiscale_destroy(ms);
  pool->destroy(pool);
  store->destroy(store);
  log_info("  PASS");
  return 0;

Fail4:
  ngff_multiscale_destroy(ms);
Fail3:
  pool->destroy(pool);
Fail2:
  store->destroy(store);
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

  struct shard_pool* pool = store->create_pool(store, 2);
  CHECK(Fail2, pool);

  CHECK(Fail3, zarr_write_group(store, "zarr.json", NULL) == 0);

  struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
  };

  struct ngff_multiscale* ms = ngff_multiscale_create(store, pool, "ms", &cfg);
  CHECK(Fail3, ms);

  struct shard_sink* sink = ngff_multiscale_as_shard_sink(ms);
  CHECK(Fail4, sink);

  // Open a shard on level 0, write some data, finalize
  struct shard_writer* w = sink->open(sink, 0, 0);
  CHECK(Fail4, w);
  uint8_t data[32];
  memset(data, 0xAA, sizeof(data));
  CHECK(Fail4, w->write(w, 0, data, data + sizeof(data)) == 0);
  CHECK(Fail4, w->finalize(w) == 0);

  // update_append: extend dim 0 from 0 to 4
  uint64_t new_sizes[1] = { 4 };
  CHECK(Fail4, sink->update_append(sink, 0, 1, new_sizes) == 0);

  // Verify L0 array zarr.json was updated
  char path[4096];
  snprintf(path, sizeof(path), "%s/ms/0/zarr.json", tmpdir);
  char buf[4096];
  size_t len;
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"shape\":[4,32]"));

  // Verify group zarr.json was updated (multiscales metadata)
  snprintf(path, sizeof(path), "%s/ms/zarr.json", tmpdir);
  CHECK(Fail4, read_file(path, buf, sizeof(buf), &len) == 0);
  CHECK(Fail4, strstr(buf, "\"multiscales\""));

  // Fence / flush
  struct io_event ev = sink->record_fence(sink, 0);
  sink->wait_fence(sink, 0, ev);

  ngff_multiscale_destroy(ms);
  pool->destroy(pool);
  store->destroy(store);
  log_info("  PASS");
  return 0;

Fail4:
  ngff_multiscale_destroy(ms);
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
  err |= test_multiscale_create();
  err |= test_multiscale_shard_sink();

  test_tmpdir_remove(tmpdir);

  return err;
}

#include "dimension.h"
#include "ngff/ngff_axis.h"
#include "ngff/ngff_multiscale.h"
#include "util/prelude.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[4096];

static int
make_tmpdir(void)
{
  snprintf(tmpdir, sizeof(tmpdir), "/tmp/test_ngff_ms_XXXXXX");
  CHECK(Fail, mkdtemp(tmpdir));
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

int
main(void)
{
  if (make_tmpdir())
    return 1;
  log_info("tmpdir: %s", tmpdir);

  int err = 0;
  err |= test_multiscale_create();

  char cmd[4096];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", tmpdir);
  (void)system(cmd);

  return err;
}

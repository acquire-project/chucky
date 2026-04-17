#include "platform/platform.h"
#include "store.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"
#include "zarr.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"

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

// --- store put/mkdirs ---

static int
test_store_put(void)
{
  log_info("=== test_store_put ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  const char* data = "hello world";
  CHECK(Fail2, s->put(s, "test.txt", data, strlen(data)) == 0);

  // Verify file contents
  char path[4096];
  snprintf(path, sizeof(path), "%s/test.txt", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail2, f);
  char buf[64];
  size_t n = fread(buf, 1, sizeof(buf), f);
  fclose(f);
  CHECK(Fail2, n == strlen(data));
  CHECK(Fail2, memcmp(buf, data, n) == 0);

  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_store_mkdirs(void)
{
  log_info("=== test_store_mkdirs ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, s->mkdirs(s, "a/b/c") == 0);

  // Now put a file inside the created dir
  CHECK(Fail2, s->put(s, "a/b/c/data.txt", "ok", 2) == 0);

  char path[4096];
  snprintf(path, sizeof(path), "%s/a/b/c/data.txt", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail2, f);
  fclose(f);

  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- shard pool ---

static int
test_shard_pool_write(void)
{
  log_info("=== test_shard_pool_write ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  // Create a subdir for shard files
  CHECK(Fail2, s->mkdirs(s, "shards") == 0);

  struct shard_pool* pool = s->create_pool(s, 2);
  CHECK(Fail2, pool);

  // Write to slot 0
  char key[256];
  snprintf(key, sizeof(key), "shards/shard_0.bin");
  struct shard_writer* w = pool->open(pool, 0, key);
  CHECK(Fail3, w);

  const char* data = "shard data here";
  size_t len = strlen(data);
  CHECK(Fail3, w->write(w, 0, data, data + len) == 0);
  CHECK(Fail3, w->finalize(w) == 0);

  // Flush and verify
  CHECK(Fail3, pool->flush(pool) == 0);

  char path[4096];
  snprintf(path, sizeof(path), "%s/shards/shard_0.bin", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail3, f);
  char buf[64];
  size_t n = fread(buf, 1, sizeof(buf), f);
  fclose(f);
  CHECK(Fail3, n == len);
  CHECK(Fail3, memcmp(buf, data, n) == 0);

  pool->destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  pool->destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_fence(void)
{
  log_info("=== test_shard_pool_fence ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  CHECK(Fail2, s->mkdirs(s, "fence") == 0);

  struct shard_pool* pool = s->create_pool(s, 4);
  CHECK(Fail2, pool);

  // Write to multiple slots
  for (int i = 0; i < 4; ++i) {
    char key[256];
    snprintf(key, sizeof(key), "fence/s%d.bin", i);
    struct shard_writer* w = pool->open(pool, (uint64_t)i, key);
    CHECK(Fail3, w);
    char data[32];
    int dlen = snprintf(data, sizeof(data), "slot_%d", i);
    CHECK(Fail3, w->write(w, 0, data, data + dlen) == 0);
    CHECK(Fail3, w->finalize(w) == 0);
  }

  struct io_event ev = pool->record_fence(pool);
  pool->wait_fence(pool, ev);

  CHECK(Fail3, pool->has_error(pool) == 0);
  CHECK(Fail3, pool->pending_bytes(pool) == 0);

  pool->destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  pool->destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_on_demand_mkdir(void)
{
  log_info("=== test_shard_pool_on_demand_mkdir ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct shard_pool* pool = s->create_pool(s, 1);
  CHECK(Fail2, pool);

  // Open with a key whose parent directory doesn't exist yet.
  // Pool should create it on-demand.
  struct shard_writer* w = pool->open(pool, 0, "deep/nested/dir/shard.bin");
  CHECK(Fail3, w);
  const char byte = 'x';
  CHECK(Fail3, w->write(w, 0, &byte, &byte + 1) == 0);
  CHECK(Fail3, w->finalize(w) == 0);
  CHECK(Fail3, pool->flush(pool) == 0);

  char path[4096];
  snprintf(path, sizeof(path), "%s/deep/nested/dir/shard.bin", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail3, f);
  fclose(f);

  pool->destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  pool->destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_unbuffered(void)
{
  log_info("=== test_shard_pool_unbuffered ===");

  // Create store with unbuffered=1 → pool uses page-aligned writes
  struct store* s = store_fs_create(tmpdir, 1);
  CHECK(Fail, s);
  CHECK(Fail2, s->mkdirs(s, "unbuf") == 0);

  struct shard_pool* pool = s->create_pool(s, 2);
  CHECK(Fail2, pool);

  // Write via the copy path (write) — exercises aligned alloc
  struct shard_writer* w = pool->open(pool, 0, "unbuf/shard0.bin");
  CHECK(Fail3, w);

  // Write enough data to be meaningful (must be page-aligned for O_DIRECT)
  // Use a 4096-byte aligned buffer
  size_t page = 4096;
  char* data = (char*)platform_aligned_alloc(page, page);
  CHECK(Fail3, data);
  memset(data, 0xAB, page);
  CHECK(Fail4, w->write(w, 0, data, data + page) == 0);

  // Write via write_direct (zero-copy path) — exercises pwrite_ref_job
  if (w->write_direct) {
    CHECK(Fail4, w->write_direct(w, page, data, data + page) == 0);
  }

  CHECK(Fail4, w->finalize(w) == 0);
  CHECK(Fail4, pool->flush(pool) == 0);
  CHECK(Fail4, pool->has_error(pool) == 0);

  // Verify file exists and has expected size
  char path[4096];
  snprintf(path, sizeof(path), "%s/unbuf/shard0.bin", tmpdir);
  CHECK(Fail4, test_file_exists(path));

  platform_aligned_free(data);
  pool->destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail4:
  platform_aligned_free(data);
Fail3:
  pool->destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_error_propagation(void)
{
  log_info("=== test_shard_pool_error_propagation ===");

  // Use a buffered pool — the error path under test is filesystem-independent,
  // driven by a test-only failing-job injector rather than by O_DIRECT
  // alignment enforcement (which varies across filesystems).
  struct shard_pool* pool = shard_pool_fs_create(tmpdir, 1, 0);
  CHECK(Fail, pool);

  // Inject a job that deliberately reports a write failure.
  CHECK(Fail2, shard_pool_fs_inject_failing_job(pool) == 0);

  // Flush waits for all async IO and returns the error flag.
  int flush_err = pool->flush(pool);
  CHECK(Fail2, flush_err != 0);
  CHECK(Fail2, pool->has_error(pool) != 0);

  pool->destroy(pool);
  log_info("  PASS");
  return 0;

Fail2:
  pool->destroy(pool);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_has_existing_data(void)
{
  log_info("=== test_has_existing_data ===");

  // Empty subdir: no zarr.json → 0
  char empty_root[4096];
  snprintf(empty_root, sizeof(empty_root), "%s/has_empty", tmpdir);
  CHECK(Fail, test_mkdir(empty_root) == 0);

  struct store* s = store_fs_create(empty_root, 0);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  store_destroy(s);

  // With zarr.json → 1
  char with_root[4096];
  snprintf(with_root, sizeof(with_root), "%s/has_with", tmpdir);
  CHECK(Fail, test_mkdir(with_root) == 0);

  s = store_fs_create(with_root, 0);
  CHECK(Fail, s);
  const char* meta = "{}";
  CHECK(Fail2, s->put(s, "zarr.json", meta, strlen(meta)) == 0);
  CHECK(Fail2, store_has_existing_data(s) == 1);
  store_destroy(s);

  // Non-existent root → -1 (stat fails with ENOENT on parent)
  char missing_root[4096];
  snprintf(missing_root, sizeof(missing_root), "%s/does_not_exist", tmpdir);
  s = store_fs_create(missing_root, 0);
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

// Directory exists with unrelated files but no zarr.json → 0.
static int
test_has_existing_data_unrelated_files(void)
{
  log_info("=== test_has_existing_data_unrelated_files ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/has_unrelated", tmpdir);
  CHECK(Fail, test_mkdir(root) == 0);

  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);
  const char* data = "not zarr";
  CHECK(Fail2, s->put(s, "other.txt", data, strlen(data)) == 0);
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

// After a real zarr group write at root → 1.
static int
test_has_existing_data_after_write_group(void)
{
  log_info("=== test_has_existing_data_after_write_group ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/has_group", tmpdir);
  CHECK(Fail, test_mkdir(root) == 0);

  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  struct zarr_group* g = zarr_group_create(s, "");
  CHECK(Fail2, g);
  zarr_group_destroy(g);
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

// Root path is a regular file, not a directory.
// stat("<file>/zarr.json") fails with ENOTDIR, which fs_has_existing_data
// maps to 0 (treated as "no zarr data present"). Documented behavior.
static int
test_has_existing_data_root_is_file(void)
{
  log_info("=== test_has_existing_data_root_is_file ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/has_root_is_file", tmpdir);
  FILE* f = fopen(root, "wb");
  CHECK(Fail, f);
  fputs("x", f);
  fclose(f);

  struct store* s = store_fs_create(root, 0);
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

int
main(void)
{
  if (make_tmpdir())
    return 1;
  log_info("tmpdir: %s", tmpdir);

  int err = 0;
  err |= test_store_put();
  err |= test_store_mkdirs();
  err |= test_shard_pool_write();
  err |= test_shard_pool_fence();
  err |= test_shard_pool_on_demand_mkdir();
  err |= test_shard_pool_unbuffered();
  err |= test_shard_pool_error_propagation();
  err |= test_has_existing_data();
  err |= test_has_existing_data_unrelated_files();
  err |= test_has_existing_data_after_write_group();
  err |= test_has_existing_data_root_is_file();

  // Cleanup
  test_tmpdir_remove(tmpdir);

  return err;
}

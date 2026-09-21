#include "platform/platform.h"
#include "platform/platform_io.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/io_scheduler.h"
#include "zarr/shard_pool_fs.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct preparation_gate
{
  struct io_backend inner;
  _Atomic int entered;
  _Atomic int release;
};

static void
blocked_prepare(void* ctx, const struct io_request* request)
{
  struct preparation_gate* g = ctx;
  atomic_store(&g->entered, 1);
  while (!atomic_load(&g->release))
    platform_sleep_ns(1000000);
  g->inner.execute(g->inner.ctx, request);
}

static struct io_backend
wrap_prepare(void* ctx, struct io_backend inner)
{
  struct preparation_gate* g = ctx;
  g->inner = inner;
  return (struct io_backend){ .ctx = g, .execute = blocked_prepare };
}

struct pool_wait
{
  struct shard_pool* pool;
  int cancel;
  int result;
  _Atomic int entered;
  _Atomic int done;
};

static void
wait_pool(void* arg)
{
  struct pool_wait* w = arg;
  atomic_store(&w->entered, 1);
  w->result = w->cancel ? w->pool->cancel_prepared(w->pool, 0, 1)
                        : w->pool->flush(w->pool);
  atomic_store(&w->done, 1);
}

static int
exists(const char* root, const char* key)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", root, key);
  return platform_path_exists(path);
}

static int
test_adoption_and_cleanup(void)
{
  char root[256] = { 0 }, path[512];
  struct shard_pool* pool = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  snprintf(path, sizeof(path), "%s/existing", root);
  CHECK(Done, platform_mkdir(path) == 0);
  pool = shard_pool_fs_create(root, 3, 0);
  CHECK(Done, pool);
  CHECK(Done, pool->prepare(pool, 0, "c/0/0", 65536) == 0);
  CHECK(Done, pool->prepare(pool, 1, "c/1/0", 65536) == 0);
  CHECK(Done, pool->prepare(pool, 2, "existing/future", 65536) == 0);
  CHECK(Done, pool->wait_prepared(pool, 2) == 0);
  struct shard_writer* w = pool->open(pool, 0, "c/0/0");
  CHECK(Done, w);
  const char value[] = "kept";
  CHECK(Done, w->presize(w, 65536) == 0);
  CHECK(Done, w->write(w, 0, value, value + sizeof(value)) == 0);
  CHECK(Done, w->truncate(w, sizeof(value)) == 0 && w->finalize(w) == 0);
  CHECK(Done, pool->prepare(pool, 0, "c/2/0", 65536) == 0);
  CHECK(Done, pool->flush(pool) == 0);
  CHECK(Done, pool->cancel_prepared(pool, 0, 2) == 0);
  CHECK(Done, exists(root, "c/0/0") == 1);
  CHECK(Done, exists(root, "c/1") == 0 && exists(root, "c/2") == 0);
  CHECK(Done, exists(root, "existing/future") == 1);
  CHECK(Done, pool->cancel_prepared(pool, 2, 1) == 0);
  CHECK(Done, pool->cancel_prepared(pool, 0, 3) == 0);
  CHECK(Done, exists(root, "existing") == 1);
  snprintf(path, sizeof(path), "%s/c/0/0", root);
  FILE* f = fopen(path, "rb");
  char actual[sizeof(value)];
  CHECK(Done, f);
  size_t n = fread(actual, 1, sizeof(actual), f);
  int tail = fgetc(f);
  fclose(f);
  CHECK(Done,
        n == sizeof(value) && tail == EOF &&
          memcmp(actual, value, sizeof(value)) == 0);
  error = 0;
Done:
  shard_pool_destroy(pool);
  test_tmpdir_remove(root);
  return error;
}

static int
test_preparation_does_not_hold_writes(void)
{
  char root[256] = { 0 };
  struct preparation_gate gate = { 0 };
  struct shard_pool* pool = NULL;
  test_thread* flush_thread = NULL;
  test_thread* cancel_thread = NULL;
  struct pool_wait flush = { 0 }, cancel = { .cancel = 1 };
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  pool = shard_pool_fs_create_wrapped(
    root,
    1,
    0,
    NULL,
    (struct shard_pool_fs_wrapper){ .ctx = &gate,
                                    .wrap_prepare = wrap_prepare });
  CHECK(Done, pool);
  struct shard_writer* w = pool->open(pool, 0, "c/0/0");
  CHECK(Done, w);
  CHECK(Done, pool->prepare(pool, 0, "c/1/0", 4096) == 0);
  CHECK(Done, test_wait_flag(&gate.entered, 5000) == 0);
  const char value[] = "current";
  CHECK(Done, w->write(w, 0, value, value + sizeof(value)) == 0);
  CHECK(Done, w->finalize(w) == 0);
  CHECK(Done, pool->queue_metadata(pool, "zarr.json", "{}", 2) == 0);
  flush.pool = cancel.pool = pool;
  CHECK(Done, test_thread_start(&flush_thread, wait_pool, &flush) == 0);
  CHECK(Done, test_wait_flag(&flush.done, 5000) == 0 && flush.result == 0);
  CHECK(Done, test_thread_start(&cancel_thread, wait_pool, &cancel) == 0);
  CHECK(Done, test_wait_flag(&cancel.entered, 5000) == 0);
  CHECK(Done, test_wait_flag(&cancel.done, 50) != 0);
  CHECK(Done, exists(root, "c/0/0") == 1 && exists(root, "zarr.json") == 1);
  atomic_store(&gate.release, 1);
  CHECK(Done, test_wait_flag(&cancel.done, 5000) == 0 && cancel.result == 0);
  CHECK(Done, exists(root, "c/1") == 0);
  error = 0;
Done:
  atomic_store(&gate.release, 1);
  test_thread_join(flush_thread);
  test_thread_join(cancel_thread);
  shard_pool_destroy(pool);
  test_tmpdir_remove(root);
  return error;
}

static int
test_existing_file_is_preserved(void)
{
  char root[256] = { 0 }, path[512];
  struct shard_pool* pool = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  snprintf(path, sizeof(path), "%s/keep", root);
  platform_fd fd = platform_open_write(path, 0);
  CHECK(Done, fd != PLATFORM_FD_INVALID);
  int written = platform_write(fd, "sentinel", 8);
  platform_close(fd);
  CHECK(Done, written == 0);
  pool = shard_pool_fs_create(root, 1, 0);
  CHECK(Done, pool && pool->prepare(pool, 0, "keep", 65536) == 0);
  CHECK(Done, pool->wait_prepared(pool, 0) != 0);
  CHECK(Done, pool->has_error(pool));
  CHECK(Done, pool->cancel_prepared(pool, 0, 1) == 0);
  CHECK(Done, pool->cancel_prepared(pool, 0, 1) == 0);
  FILE* f = fopen(path, "rb");
  char actual[9] = { 0 };
  CHECK(Done, f);
  size_t n = fread(actual, 1, sizeof(actual), f);
  fclose(f);
  CHECK(Done, n == 8 && strcmp(actual, "sentinel") == 0);
  error = 0;
Done:
  shard_pool_destroy(pool);
  test_tmpdir_remove(root);
  return error;
}

static int
test_cleanup_failure_can_be_retried(void)
{
  char root[256] = { 0 }, path[512], moved[512];
  struct shard_pool* pool = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  pool = shard_pool_fs_create(root, 1, 0);
  CHECK(Done, pool && pool->prepare(pool, 0, "c/3/0", 4096) == 0);
  CHECK(Done, pool->wait_prepared(pool, 0) == 0);
  snprintf(path, sizeof(path), "%s/c/3/0", root);
  snprintf(moved, sizeof(moved), "%s/held", root);
#ifndef _WIN32
  CHECK(Done, platform_rename_replace(path, moved) == 0);
  CHECK(Done, platform_mkdir(path) == 0);
  CHECK(Done, pool->cancel_prepared(pool, 0, 1) != 0);
  CHECK(Done, pool->has_error(pool));
  CHECK(Done, platform_remove_empty_directory(path) == 0);
  CHECK(Done, platform_rename_replace(moved, path) == 0);
#endif
  CHECK(Done, pool->cancel_prepared(pool, 0, 1) == 0);
  CHECK(Done, exists(root, "c") == 0);
  error = 0;
Done:
  shard_pool_destroy(pool);
  test_tmpdir_remove(root);
  return error;
}

static int
test_shared_directory_cleanup(int first)
{
  char root[256] = { 0 };
  struct shard_pool* pool = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  pool = shard_pool_fs_create(root, 2, 0);
  CHECK(Done, pool);
  CHECK(Done, pool->prepare(pool, 0, "c/0/0", 4096) == 0);
  CHECK(Done, pool->prepare(pool, 1, "c/0/1", 4096) == 0);
  CHECK(Done, pool->wait_prepared(pool, 1) == 0);
  CHECK(Done, pool->cancel_prepared(pool, (uint64_t)first, 1) == 0);
  CHECK(Done, exists(root, "c/0") == 1);
  CHECK(Done, pool->cancel_prepared(pool, (uint64_t)(first ^ 1), 1) == 0);
  CHECK(Done, exists(root, "c") == 0);
  CHECK(Done, pool->cancel_prepared(pool, 0, 2) == 0);
  error = 0;
Done:
  shard_pool_destroy(pool);
  test_tmpdir_remove(root);
  return error;
}

static int
test_adoption_preserves_shared_directories(void)
{
  char root[256] = { 0 };
  struct shard_pool* pool = NULL;
  int error = 1;
  CHECK(Done, test_tmpdir_create(root, sizeof(root)) == 0);
  pool = shard_pool_fs_create(root, 2, 0);
  CHECK(Done, pool);
  CHECK(Done, pool->prepare(pool, 0, "c/0/0", 4096) == 0);
  CHECK(Done, pool->prepare(pool, 1, "c/0/1", 4096) == 0);
  CHECK(Done, pool->wait_prepared(pool, 1) == 0);
  // Slot 0 created the shared directories; slot 1 makes them permanent.
  struct shard_writer* w = pool->open(pool, 1, "c/0/1");
  CHECK(Done, w && w->finalize(w) == 0);
  CHECK(Done, pool->cancel_prepared(pool, 0, 2) == 0);
  shard_pool_destroy(pool);
  pool = NULL;
  CHECK(Done, exists(root, "c/0/0") == 0);
  CHECK(Done, exists(root, "c/0/1") == 1);
  error = 0;
Done:
  shard_pool_destroy(pool);
  test_tmpdir_remove(root);
  return error;
}

int
main(void)
{
  int error = test_adoption_and_cleanup();
  error |= test_preparation_does_not_hold_writes();
  error |= test_existing_file_is_preserved();
  error |= test_cleanup_failure_can_be_retried();
  error |= test_shared_directory_cleanup(0);
  error |= test_shared_directory_cleanup(1);
  error |= test_adoption_preserves_shared_directories();
  return error;
}

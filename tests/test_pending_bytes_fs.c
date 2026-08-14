// The pool reports the bytes its queued writes are still carrying. The count
// itself lives in the io queue, which raises it inside the lock that hands a job
// to the worker; tests/test_io_queue.c covers that. This checks the pool passes
// the figure through, over both write paths.

#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WRITE_BYTES 4096
#define BATCH_WRITES 8

// With the worker held, every queued write must show up in the figure, and it
// must fall back to zero once they land.
static int
test_counts_every_write(const char* tmpdir, int direct)
{
  const char* label = direct ? "zero-copy" : "copy";
  log_info("=== test_counts_every_write (%s) ===", label);

  struct shard_pool* pool = NULL;
  uint8_t* src = NULL;
  int rc = 1;
  _Atomic int gate;
  atomic_store(&gate, 0);

  pool = shard_pool_fs_create(tmpdir, /*nslots=*/1, /*unbuffered=*/0);
  CHECK(Cleanup, pool);
  CHECK(Cleanup, shard_pool_pending_bytes(pool) == 0);

  struct shard_writer* w = pool->open(pool, 0, "shard");
  CHECK(Cleanup, w);

  int (*write)(struct shard_writer*, uint64_t, const void*, const void*) =
    direct ? w->write_direct : w->write;
  CHECK(Cleanup, write);

  src = (uint8_t*)calloc(1, WRITE_BYTES);
  CHECK(Cleanup, src);

  CHECK(Cleanup, shard_pool_fs_inject_blocking_job(pool, &gate) == 0);

  for (int i = 0; i < BATCH_WRITES; ++i) {
    uint64_t offset = (uint64_t)i * WRITE_BYTES;
    CHECK(Cleanup, write(w, offset, src, src + WRITE_BYTES) == 0);
    CHECK(Cleanup,
          shard_pool_pending_bytes(pool) == (size_t)(i + 1) * WRITE_BYTES);
  }

  log_info("  pending with worker held: %llu bytes",
           (unsigned long long)shard_pool_pending_bytes(pool));

  atomic_store(&gate, 1);
  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, shard_pool_pending_bytes(pool) == 0);

  rc = 0;
  log_info("  PASS");

Cleanup:
  atomic_store(&gate, 1);
  shard_pool_destroy(pool);
  free(src);
  if (rc)
    log_error("  FAIL");
  return rc;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 1;
  char tmpdir[4096];
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  ecode = 0;
  log_info("temp dir: %s", tmpdir);

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/copy", tmpdir);
    test_mkdir(sub);
    ecode |= test_counts_every_write(sub, 0);
  }
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/zero-copy", tmpdir);
    test_mkdir(sub);
    ecode |= test_counts_every_write(sub, 1);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}

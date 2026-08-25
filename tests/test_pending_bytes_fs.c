// The pool reports the bytes its queued writes still carry. The count lives in
// the io queue, and tests/test_io_queue.c covers it there. This checks that the
// pool passes the figure through, over both write paths.

#include "test_io_faults.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "writer.h"
#include "zarr/shard_pool.h"

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
  struct io_faults faults;
  uint8_t* src = NULL;
  int rc = 1;
  _Atomic int gate;
  atomic_store(&gate, 0);

  // One worker and one write at a time, so the injected blocking job holds
  // up every write behind it and the count can be read between posts.
  const struct io_scheduling io = { .workers = 1,
                                    .writes_in_flight = 1,
                                    .writes_in_flight_per_file = 1 };
  pool =
    io_faults_pool_create(&faults, tmpdir, /*nslots=*/1, /*unbuffered=*/0, &io);
  CHECK(Cleanup, pool);
  CHECK(Cleanup, shard_pool_pending_bytes(pool) == 0);

  struct shard_writer* w = pool->open(pool, 0, "shard");
  CHECK(Cleanup, w);

  int (*post_write)(struct shard_writer*, uint64_t, const void*, const void*) =
    direct ? w->write_direct : w->write;
  CHECK(Cleanup, post_write);

  src = (uint8_t*)calloc(1, WRITE_BYTES);
  CHECK(Cleanup, src);

  CHECK(Cleanup, io_faults_inject_blocking_job(&faults, &gate) == 0);

  for (int i = 0; i < BATCH_WRITES; ++i) {
    uint64_t offset = (uint64_t)i * WRITE_BYTES;
    CHECK(Cleanup, post_write(w, offset, src, src + WRITE_BYTES) == 0);
    CHECK(Cleanup,
          shard_pool_pending_bytes(pool) == (uint64_t)(i + 1) * WRITE_BYTES);
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

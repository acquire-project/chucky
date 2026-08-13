// Regression test for #201: pending_bytes must never report less than the work
// outstanding.
//
// The pool adds a write to its pending count, then hands the job to the io
// worker, which subtracts once the write lands. Hand off before counting and the
// worker can subtract a write nobody has added yet, driving the count negative.
//
// That window is a few nanoseconds wide, so waiting for a thread to lose the
// race does not work. The pool has a test hook that parks a writer inside it.

#include "platform/platform.h"
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
#define POLL_STEP_MS 10
#define PARK_TIMEOUT_MS 2000

typedef int (*write_fn)(struct shard_writer*,
                        uint64_t,
                        const void*,
                        const void*);

struct one_write_args
{
  struct shard_writer* w;
  write_fn write;
  const uint8_t* src;
  int error;
};

static void
one_write_fn(void* arg)
{
  struct one_write_args* oa = (struct one_write_args*)arg;
  oa->error = oa->write(oa->w, 0, oa->src, oa->src + WRITE_BYTES);
}

// Wait for the writer to reach the window rather than guess how long that takes.
// A build that counts after the handoff never reports the write, so this waits
// out the timeout and the caller's check fails, as it should.
static void
wait_for_pending(struct shard_pool* pool, int timeout_ms)
{
  int waited_ms = 0;
  while (shard_pool_pending_bytes(pool) == 0 && waited_ms < timeout_ms) {
    platform_sleep_ns((int64_t)POLL_STEP_MS * 1000000LL);
    waited_ms += POLL_STEP_MS;
  }
}

// A parked writer must report exactly the write it is queueing. Equality
// matters: counting after the handoff drives the count negative, which clamps to
// 0, and a check for "not too high" would accept that.
static int
test_parked_writer(const char* tmpdir, int direct)
{
  const char* label = direct ? "zero-copy" : "copy";
  log_info("=== test_parked_writer (%s) ===", label);

  struct shard_pool* pool = NULL;
  uint8_t* src = NULL;
  test_thread* thr = NULL;
  int rc = 1;
  struct one_write_args oa = { 0 };

  pool = shard_pool_fs_create(tmpdir, /*nslots=*/1, /*unbuffered=*/0);
  CHECK(Cleanup, pool);

  struct shard_writer* w = pool->open(pool, 0, "shard");
  CHECK(Cleanup, w);

  oa.w = w;
  oa.write = direct ? w->write_direct : w->write;
  CHECK(Cleanup, oa.write);

  src = (uint8_t*)calloc(1, WRITE_BYTES);
  CHECK(Cleanup, src);
  oa.src = src;

  shard_pool_fs_pause_mid_write(pool, 1);
  CHECK(Cleanup, test_thread_start(&thr, one_write_fn, &oa) == 0);

  wait_for_pending(pool, PARK_TIMEOUT_MS);

  // Settle anything already queued. A build that hands off before counting has a
  // write in flight whose worker drives the count too low; a correct build has
  // posted nothing and this returns at once.
  pool->wait_fence(pool, pool->record_fence(pool));

  size_t parked = shard_pool_pending_bytes(pool);
  log_info("  pending while parked: %llu bytes (queueing %d)",
           (unsigned long long)parked,
           WRITE_BYTES);
  CHECK(Cleanup, parked == (size_t)WRITE_BYTES);

  shard_pool_fs_pause_mid_write(pool, 0);
  CHECK(Cleanup, test_thread_join(thr) == 0);
  thr = NULL;
  CHECK(Cleanup, oa.error == 0);

  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, shard_pool_pending_bytes(pool) == 0);

  rc = 0;
  log_info("  PASS");

Cleanup:
  if (pool)
    shard_pool_fs_pause_mid_write(pool, 0);
  test_thread_join(thr);
  shard_pool_destroy(pool);
  free(src);
  if (rc)
    log_error("  FAIL");
  return rc;
}

// With the worker held, the count must equal every queued write, and fall back
// to zero once they land.
static int
test_counts_every_write(const char* tmpdir)
{
  log_info("=== test_counts_every_write ===");

  struct shard_pool* pool = NULL;
  uint8_t* src = NULL;
  int rc = 1;
  _Atomic int gate;
  atomic_store(&gate, 0);

  pool = shard_pool_fs_create(tmpdir, /*nslots=*/1, /*unbuffered=*/0);
  CHECK(Cleanup, pool);

  struct shard_writer* w = pool->open(pool, 0, "shard");
  CHECK(Cleanup, w);

  src = (uint8_t*)calloc(1, WRITE_BYTES);
  CHECK(Cleanup, src);

  CHECK(Cleanup, shard_pool_fs_inject_blocking_job(pool, &gate) == 0);

  for (int i = 0; i < BATCH_WRITES; ++i) {
    uint64_t offset = (uint64_t)i * WRITE_BYTES;
    CHECK(Cleanup, w->write(w, offset, src, src + WRITE_BYTES) == 0);
  }

  size_t pending = shard_pool_pending_bytes(pool);
  log_info("  pending with worker held: %llu bytes",
           (unsigned long long)pending);
  CHECK(Cleanup, pending == (size_t)BATCH_WRITES * WRITE_BYTES);

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
    ecode |= test_parked_writer(sub, 0);
  }
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/zero-copy", tmpdir);
    test_mkdir(sub);
    ecode |= test_parked_writer(sub, 1);
  }
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/counts", tmpdir);
    test_mkdir(sub);
    ecode |= test_counts_every_write(sub);
  }

  test_tmpdir_remove(tmpdir);

Fail:
  return ecode;
}

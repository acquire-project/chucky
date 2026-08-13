// Regression test for #201: pending_bytes must never report more bytes than
// were queued.
//
// The pool adds a write to its pending count and hands the job to the io
// worker, which subtracts once the write lands. Between those two steps the
// count and the queued work disagree, and the order decides which way: count
// first and pending_bytes reads high by that write, hand off first and the
// worker can subtract a write nobody has added yet, leaving the count negative.
//
// The window is a few nanoseconds wide, so waiting for a real thread to lose
// the race does not work. The pool has a test hook that parks a writer in the
// window instead, which makes both orders show their result every run.

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
#define WORKER_SETTLE_MS 200
#define POLL_STEP_MS 5
#define JOIN_TIMEOUT_MS 5000

struct one_write_args
{
  struct shard_writer* w;
  const uint8_t* src;
  int direct;
  _Atomic int done;
  int error;
};

static void
one_write_fn(void* arg)
{
  struct one_write_args* oa = (struct one_write_args*)arg;
  const uint8_t* beg = oa->src;
  oa->error = oa->direct
                ? oa->w->write_direct(oa->w, 0, beg, beg + WRITE_BYTES)
                : oa->w->write(oa->w, 0, beg, beg + WRITE_BYTES);
  atomic_store(&oa->done, 1);
}

static int
wait_for_done(_Atomic int* done, int timeout_ms)
{
  int waited_ms = 0;
  while (atomic_load(done) == 0 && waited_ms < timeout_ms) {
    platform_sleep_ns((int64_t)POLL_STEP_MS * 1000000LL);
    waited_ms += POLL_STEP_MS;
  }
  return atomic_load(done) != 0 ? 0 : -1;
}

// A writer parked in the window must leave pending_bytes reporting exactly the
// write it is queueing. Equality matters: counting after the handoff drives the
// count negative, and the clamp in pending_bytes would report that as 0, which
// an upper-bound check would accept.
static int
test_parked_writer(const char* tmpdir, const char* label, int direct)
{
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
  CHECK(Cleanup, !direct || w->write_direct);

  src = (uint8_t*)calloc(1, WRITE_BYTES);
  CHECK(Cleanup, src);

  shard_pool_fs_pause_mid_write(pool, 1);

  oa.w = w;
  oa.src = src;
  oa.direct = direct;
  atomic_store(&oa.done, 0);
  CHECK(Cleanup, test_thread_start(&thr, one_write_fn, &oa) == 0);

  // Long enough for the worker to have retired the write, if this build handed
  // it over before counting it.
  platform_sleep_ns((int64_t)WORKER_SETTLE_MS * 1000000LL);

  size_t parked = shard_pool_pending_bytes(pool);
  log_info("  pending while parked: %llu bytes (queueing %d)",
           (unsigned long long)parked,
           WRITE_BYTES);

  shard_pool_fs_pause_mid_write(pool, 0);
  CHECK(Cleanup, wait_for_done(&oa.done, JOIN_TIMEOUT_MS) == 0);
  test_thread_join(thr);
  thr = NULL;
  CHECK(Cleanup, oa.error == 0);

  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, parked == (size_t)WRITE_BYTES);
  CHECK(Cleanup, shard_pool_pending_bytes(pool) == 0);

  rc = 0;
  log_info("  PASS");

Cleanup:
  if (pool)
    shard_pool_fs_pause_mid_write(pool, 0);
  test_thread_join(thr);
  if (pool)
    pool->flush(pool);
  free(src);
  shard_pool_destroy(pool);
  if (rc)
    log_error("  FAIL");
  return rc;
}

// With the worker held, pending_bytes must account for every queued write, and
// must fall back to zero once they land.
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
  if (pool)
    pool->flush(pool);
  free(src);
  shard_pool_destroy(pool);
  if (rc)
    log_error("  FAIL");
  return rc;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  char tmpdir[4096];
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  log_info("temp dir: %s", tmpdir);

  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/copy", tmpdir);
    test_mkdir(sub);
    ecode |= test_parked_writer(sub, "copy", 0);
  }
  {
    char sub[4200];
    snprintf(sub, sizeof(sub), "%s/direct", tmpdir);
    test_mkdir(sub);
    ecode |= test_parked_writer(sub, "zero-copy", 1);
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

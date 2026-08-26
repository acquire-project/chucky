// The ring is put through the write path the worker threads take, and its
// files are compared with theirs. Where no ring can be had, nothing here runs.

#include "platform/platform.h"
#include "platform/platform_io.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_backend.uring.h"
#include "zarr/io_queue.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <signal.h>
#include <sys/resource.h>
#endif

#define SKIP_EXIT_CODE 77

#define WRITE_BYTES 8192u
#define WRITES_PER_FILE 6u
#define SHARD_BYTES (WRITES_PER_FILE * WRITE_BYTES)
#define POOL_SLOTS 3u
#define ROUNDS 2u
#define SOURCE_BYTES (SHARD_BYTES * POOL_SLOTS * ROUNDS)

// These are big enough that a run of writes overlaps and small enough to stay
// quick.
#define WORKERS 3u
#define WRITES_IN_FLIGHT 6u
#define WRITES_PER_FILE_IN_FLIGHT 3u

// A path is built by appending a key to a root, and a root by appending to the
// temporary directory, so each holds less than the one it is built from.
#define KEY_CHARS 64
#define PATH_CHARS 512
#define ROOT_CHARS (PATH_CHARS - KEY_CHARS - 1)
#define TMPDIR_CHARS (ROOT_CHARS - KEY_CHARS - 1)

static void
fill_source(uint8_t* src, uint64_t nbytes)
{
  for (uint64_t i = 0; i < nbytes; ++i)
    src[i] = (uint8_t)(i * 31u + (i >> 8) * 7u + 1u);
}

static void
shard_key(char* out, uint64_t round, uint64_t slot)
{
  snprintf(out,
           KEY_CHARS,
           "shard_%llu_%llu.bin",
           (unsigned long long)round,
           (unsigned long long)slot);
}

// Offsets are visited out of order so a file has more than one write ready.
static uint64_t
write_turn(uint64_t i)
{
  return (i * 5u) % WRITES_PER_FILE;
}

// The shard files are written twice over, so a writer slot and the file index
// behind it are both reused while an earlier close is still retiring.
static int
write_the_shards(const char* root,
                 enum io_backend_choice backend,
                 int unbuffered,
                 const uint8_t* src)
{
  const struct io_scheduling io = {
    .workers = WORKERS,
    .writes_in_flight = WRITES_IN_FLIGHT,
    .writes_in_flight_per_file = WRITES_PER_FILE_IN_FLIGHT,
    .backend = backend,
  };

  int rc = 1;
  struct shard_pool* pool =
    shard_pool_fs_create(root, POOL_SLOTS, unbuffered, &io);
  CHECK(Fail, pool);

  const uint8_t* next = src;
  for (uint64_t round = 0; round < ROUNDS; ++round) {
    for (uint64_t slot = 0; slot < POOL_SLOTS; ++slot) {
      char key[KEY_CHARS];
      shard_key(key, round, slot);

      struct shard_writer* w = pool->open(pool, slot, key);
      CHECK(Cleanup, w);
      CHECK(Cleanup, w->presize(w, SHARD_BYTES) == 0);

      for (uint64_t i = 0; i < WRITES_PER_FILE; ++i) {
        const uint64_t offset = write_turn(i) * WRITE_BYTES;
        const uint8_t* beg = next + offset;
        CHECK(Cleanup, w->write(w, offset, beg, beg + WRITE_BYTES) == 0);
      }

      CHECK(Cleanup, w->truncate(w, SHARD_BYTES) == 0);
      CHECK(Cleanup, w->finalize(w) == 0);
      next += SHARD_BYTES;
    }
  }

  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, shard_pool_pending_bytes(pool) == 0);
  rc = 0;

Cleanup:
  shard_pool_destroy(pool);
Fail:
  return rc;
}

// Read the file. The count of bytes read is returned, or -1 when it could not
// be opened.
static int64_t
read_whole_file(const char* path, uint8_t* buf, uint64_t cap)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return -1;
  const size_t got = fread(buf, 1, (size_t)cap, f);
  fclose(f);
  return (int64_t)got;
}

static int
files_hold_the_source(const char* root, const uint8_t* src)
{
  int rc = 1;
  uint8_t* landed = (uint8_t*)malloc(SHARD_BYTES + 1);
  CHECK(Cleanup, landed);

  const uint8_t* next = src;
  for (uint64_t round = 0; round < ROUNDS; ++round) {
    for (uint64_t slot = 0; slot < POOL_SLOTS; ++slot) {
      char key[KEY_CHARS];
      char path[PATH_CHARS];
      shard_key(key, round, slot);
      snprintf(path, sizeof(path), "%s/%s", root, key);

      CHECK(Cleanup,
            read_whole_file(path, landed, SHARD_BYTES + 1) ==
              (int64_t)SHARD_BYTES);
      CHECK(Cleanup, memcmp(landed, next, SHARD_BYTES) == 0);
      next += SHARD_BYTES;
    }
  }
  rc = 0;

Cleanup:
  free(landed);
  return rc;
}

static int
test_the_ring_writes_what_the_threads_write(const char* tmpdir, int unbuffered)
{
  log_info("=== test_the_ring_writes_what_the_threads_write (%s) ===",
           unbuffered ? "unbuffered" : "buffered");

  int rc = 1;
  char threads_root[ROOT_CHARS];
  char ring_root[ROOT_CHARS];
  snprintf(
    threads_root, sizeof(threads_root), "%s/threads%d", tmpdir, unbuffered);
  snprintf(ring_root, sizeof(ring_root), "%s/ring%d", tmpdir, unbuffered);
  CHECK(Fail, test_mkdir(threads_root) == 0);
  CHECK(Fail, test_mkdir(ring_root) == 0);

  uint8_t* src = (uint8_t*)malloc(SOURCE_BYTES);
  CHECK(Fail, src);
  fill_source(src, SOURCE_BYTES);

  CHECK(Cleanup,
        write_the_shards(threads_root, IO_BACKEND_THREADS, unbuffered, src) ==
          0);
  CHECK(Cleanup,
        write_the_shards(ring_root, IO_BACKEND_URING, unbuffered, src) == 0);
  CHECK(Cleanup, files_hold_the_source(threads_root, src) == 0);
  CHECK(Cleanup, files_hold_the_source(ring_root, src) == 0);
  rc = 0;

Cleanup:
  free(src);
Fail:
  if (rc == 0)
    log_info("  PASS");
  return rc;
}

// --- The ring on its own, behind the queue ---

struct ring_under_test
{
  struct io_backend_fs* files;
  struct io_backend_uring* ring;
  struct io_queue* queue;
  _Atomic int io_error;
};

// A queue ceiling above the ring's depth is what makes the ring say it has no
// room.
static int
ring_open(struct ring_under_test* r, uint64_t depth, uint64_t in_flight)
{
  memset(r, 0, sizeof(*r));
  atomic_store(&r->io_error, 0);

  r->files = io_backend_fs_create(&r->io_error);
  CHECK(Fail, r->files);
  r->ring = io_backend_uring_create(r->files, &r->io_error, depth);
  CHECK(Fail, r->ring);
  r->queue = io_queue_create(io_backend_uring_as_backend(r->ring),
                             (struct io_queue_limits){
                               .workers = WORKERS,
                               .writes_in_flight = in_flight,
                               .writes_in_flight_per_file = in_flight,
                             });
  CHECK(Fail, r->queue);
  CHECK(Fail, io_backend_uring_start(r->ring, r->queue) == 0);
  return 0;

Fail:
  return 1;
}

static void
ring_close(struct ring_under_test* r)
{
  io_queue_destroy(r->queue);
  io_backend_uring_destroy(r->ring);
  io_backend_fs_destroy(r->files);
}

static struct io_file_token
ring_add_file(struct ring_under_test* r, const char* path)
{
  platform_fd fd = platform_open_write(path, 0);
  if (fd == PLATFORM_FD_INVALID)
    return (struct io_file_token){ 0 };

  struct io_file_token token = io_backend_fs_add_file(r->files, fd);
  if (token.generation == 0)
    platform_close(fd);
  return token;
}

// A ring holding one write at a time is offered more than it can take, and
// every refused write still has to land, in the place it was posted in.
static int
test_a_full_ring_takes_every_write_in_the_end(const char* tmpdir)
{
  log_info("=== test_a_full_ring_takes_every_write_in_the_end ===");

  int rc = 1;
  struct ring_under_test r;
  uint8_t* src = NULL;
  uint8_t* landed = NULL;
  char path[PATH_CHARS];
  snprintf(path, sizeof(path), "%s/full_ring.bin", tmpdir);

  CHECK(Cleanup,
        ring_open(&r, /*depth=*/1, /*in_flight=*/WRITES_IN_FLIGHT) == 0);

  src = (uint8_t*)malloc(SHARD_BYTES);
  landed = (uint8_t*)malloc(SHARD_BYTES + 1);
  CHECK(Cleanup, src && landed);
  fill_source(src, SHARD_BYTES);

  const struct io_file_token token = ring_add_file(&r, path);
  CHECK(Cleanup, token.generation != 0);

  for (uint64_t i = 0; i < WRITES_PER_FILE; ++i) {
    const uint64_t offset = write_turn(i) * WRITE_BYTES;
    CHECK(Cleanup,
          io_queue_post(r.queue,
                        (struct io_request){ .op = IO_OP_WRITE,
                                             .borrowed = 1,
                                             .file = token,
                                             .payload = src + offset,
                                             .nbytes = WRITE_BYTES,
                                             .offset = offset }) == 0);
  }
  CHECK(Cleanup,
        io_queue_post(
          r.queue, (struct io_request){ .op = IO_OP_CLOSE, .file = token }) ==
          0);
  io_event_wait(r.queue, io_queue_record(r.queue));

  CHECK(Cleanup, atomic_load(&r.io_error) == 0);
  CHECK(Cleanup, io_queue_pending_bytes(r.queue) == 0);
  CHECK(Cleanup,
        read_whole_file(path, landed, SHARD_BYTES + 1) == (int64_t)SHARD_BYTES);
  CHECK(Cleanup, memcmp(src, landed, SHARD_BYTES) == 0);
  rc = 0;

Cleanup:
  ring_close(&r);
  free(landed);
  free(src);
  if (rc == 0)
    log_info("  PASS");
  return rc;
}

// A write naming a file that was never opened has to raise the pool's flag:
// the queue reads no status of its own.
static int
test_a_stale_token_raises_the_error_flag(const char* tmpdir)
{
  log_info("=== test_a_stale_token_raises_the_error_flag ===");
  (void)tmpdir;

  int rc = 1;
  struct ring_under_test r;
  uint8_t payload[WRITE_BYTES] = { 0 };

  CHECK(Cleanup, ring_open(&r, /*depth=*/2, /*in_flight=*/2) == 0);

  CHECK(
    Cleanup,
    io_queue_post(r.queue,
                  (struct io_request){ .op = IO_OP_WRITE,
                                       .borrowed = 1,
                                       .file = { .generation = 99, .index = 0 },
                                       .payload = payload,
                                       .nbytes = WRITE_BYTES }) == 0);
  io_event_wait(r.queue, io_queue_record(r.queue));

  CHECK(Cleanup, atomic_load(&r.io_error) == 1);
  CHECK(Cleanup, io_queue_pending_bytes(r.queue) == 0);
  rc = 0;

Cleanup:
  ring_close(&r);
  if (rc == 0)
    log_info("  PASS");
  return rc;
}

// A write the kernel turns down has to raise the pool's flag too. A negative
// offset is turned down whatever the file is.
static int
test_a_write_the_kernel_turns_down_raises_the_error_flag(const char* tmpdir)
{
  log_info("=== test_a_write_the_kernel_turns_down_raises_the_error_flag ===");

  int rc = 1;
  struct ring_under_test r;
  uint8_t payload[WRITE_BYTES] = { 0 };
  char path[PATH_CHARS];
  snprintf(path, sizeof(path), "%s/turned_down.bin", tmpdir);

  CHECK(Cleanup, ring_open(&r, /*depth=*/2, /*in_flight=*/2) == 0);

  const struct io_file_token token = ring_add_file(&r, path);
  CHECK(Cleanup, token.generation != 0);

  CHECK(Cleanup,
        io_queue_post(r.queue,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .borrowed = 1,
                                           .file = token,
                                           .payload = payload,
                                           .nbytes = WRITE_BYTES,
                                           .offset = (uint64_t)1 << 63 }) == 0);
  io_event_wait(r.queue, io_queue_record(r.queue));

  CHECK(Cleanup, atomic_load(&r.io_error) == 1);
  CHECK(Cleanup, io_queue_pending_bytes(r.queue) == 0);
  rc = 0;

Cleanup:
  ring_close(&r);
  if (rc == 0)
    log_info("  PASS");
  return rc;
}

#ifndef _WIN32
// A write the kernel only partly carries out is the backend's to finish. A
// file-size limit stops it halfway and turns the rest down, so a backend that
// carries on raises the flag where one that gave up would not.
static int
test_a_short_write_is_not_reported_as_done(const char* tmpdir)
{
  log_info("=== test_a_short_write_is_not_reported_as_done ===");

  int rc = 1;
  struct ring_under_test r;
  uint8_t payload[WRITE_BYTES];
  uint8_t landed[WRITE_BYTES + 1];
  char path[PATH_CHARS];
  struct rlimit was;
  int limited = 0;
  snprintf(path, sizeof(path), "%s/short_write.bin", tmpdir);
  memset(payload, 'w', sizeof(payload));

  // The retry is over the limit, and a write over it raises this signal.
  void (*was_ignored)(int) = signal(SIGXFSZ, SIG_IGN);

  CHECK(Cleanup, ring_open(&r, /*depth=*/2, /*in_flight=*/2) == 0);
  const struct io_file_token token = ring_add_file(&r, path);
  CHECK(Cleanup, token.generation != 0);

  CHECK(Cleanup, getrlimit(RLIMIT_FSIZE, &was) == 0);
  const struct rlimit half = { .rlim_cur = WRITE_BYTES / 2,
                               .rlim_max = was.rlim_max };
  CHECK(Cleanup, setrlimit(RLIMIT_FSIZE, &half) == 0);
  limited = 1;

  CHECK(Cleanup,
        io_queue_post(r.queue,
                      (struct io_request){ .op = IO_OP_WRITE,
                                           .borrowed = 1,
                                           .file = token,
                                           .payload = payload,
                                           .nbytes = WRITE_BYTES }) == 0);
  CHECK(Cleanup,
        io_queue_post(
          r.queue, (struct io_request){ .op = IO_OP_CLOSE, .file = token }) ==
          0);
  io_event_wait(r.queue, io_queue_record(r.queue));

  CHECK(Cleanup, atomic_load(&r.io_error) == 1);
  CHECK(Cleanup,
        read_whole_file(path, landed, sizeof(landed)) ==
          (int64_t)(WRITE_BYTES / 2));
  rc = 0;

Cleanup:
  if (limited)
    setrlimit(RLIMIT_FSIZE, &was);
  signal(SIGXFSZ, was_ignored);
  ring_close(&r);
  if (rc == 0)
    log_info("  PASS");
  return rc;
}
#endif

// Nothing is posted, so only teardown can wake the thread reading the ring.
static int
test_an_idle_ring_is_torn_down(const char* tmpdir)
{
  log_info("=== test_an_idle_ring_is_torn_down ===");
  (void)tmpdir;

  struct ring_under_test r;
  const int opened = ring_open(&r, /*depth=*/2, /*in_flight=*/2);
  ring_close(&r);
  if (opened)
    return 1;
  log_info("  PASS");
  return 0;
}

int
main(void)
{
  if (!io_backend_uring_supported()) {
    log_info("no ring on this machine — skipping");
    return SKIP_EXIT_CODE;
  }

  char tmpdir[TMPDIR_CHARS];
  if (test_tmpdir_create(tmpdir, sizeof(tmpdir)) != 0) {
    log_error("could not make a temporary directory");
    return 1;
  }

  int rc = 0;
  rc |= test_the_ring_writes_what_the_threads_write(tmpdir, 0);
  // An unbuffered write has to be a whole number of pages long.
  if (WRITE_BYTES % platform_page_alignment() == 0)
    rc |= test_the_ring_writes_what_the_threads_write(tmpdir, 1);
  rc |= test_a_full_ring_takes_every_write_in_the_end(tmpdir);
  rc |= test_a_stale_token_raises_the_error_flag(tmpdir);
  rc |= test_a_write_the_kernel_turns_down_raises_the_error_flag(tmpdir);
#ifndef _WIN32
  rc |= test_a_short_write_is_not_reported_as_done(tmpdir);
#endif
  rc |= test_an_idle_ring_is_torn_down(tmpdir);

  test_tmpdir_remove(tmpdir);
  return rc;
}

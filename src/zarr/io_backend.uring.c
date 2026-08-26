#include "zarr/io_backend.uring.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_queue.h"

#include <errno.h>
#include <liburing.h>
#include <stdlib.h>
#include <string.h>

#define FREE_LIST_END UINT32_MAX
#define PROBE_ENTRIES 8u
#define DEFAULT_DEPTH 8u

// A ring with nothing left in it still has to wake the thread reading it, so
// teardown posts a request carrying this in place of a slot number, which is
// too small to reach it.
#define STOP_TAG UINT64_MAX

// A write's byte count is 32 bits wide in the ring, and a longer write has to
// be split where an unbuffered file allows it.
#define MAX_WRITE_BYTES (1u << 30)

// A ring that cannot be read is a broken kernel rather than a busy one.
#define READ_RETRY_NS 100000000LL

// A submission the kernel turns down is transient.
#define SUBMIT_TRIES 3
#define SUBMIT_RETRY_NS 1000000LL

struct ring_slot
{
  struct io_request rest; // the part of the write still to be done
  uint64_t seq;
  uint64_t asked; // the size the queue was told
  platform_fd fd;
  uint32_t next_free;
  uint32_t claims;   // raised each time the slot is taken
  uint8_t in_flight; // the ring may still report it
};

// A completion is named by its slot and by the count of times that slot has
// been taken, so one for a write that is long gone can be told apart.
static uint64_t
slot_tag(uint32_t index, uint32_t claims)
{
  return ((uint64_t)claims << 32) | index;
}

struct io_backend_uring
{
  struct io_uring ring;
  struct io_backend_fs* files;
  struct io_backend blocking; // everything that is not a write goes here
  // The queue is built from the backend, so this is set after its workers are
  // already running.
  struct io_queue* _Atomic queue;
  _Atomic int* io_error;

  struct platform_mutex* mutex; // guards submission and the free list
  struct ring_slot* slots;
  uint32_t nslots;
  uint32_t free_head;

  struct platform_thread* reader;
};

static platform_once ring_probe = PLATFORM_ONCE_INIT;
static int ring_can_be_had;

static void
probe_for_a_ring(void)
{
  struct io_uring ring;
  const int rc = io_uring_queue_init(PROBE_ENTRIES, &ring, 0);
  if (rc == 0)
    io_uring_queue_exit(&ring);
  else
    log_error("io_backend_uring: no ring here (%s); writing on the workers",
              strerror(-rc));

  ring_can_be_had = rc == 0;
}

int
io_backend_uring_supported(void)
{
  platform_call_once(&ring_probe, probe_for_a_ring);
  return ring_can_be_had;
}

// Non-zero when the slot still holds the write the tag names. A completion for
// one it no longer holds is dropped, so it cannot retire a later write.
static int
slot_holds_the_write(struct io_backend_uring* b, uint64_t tag)
{
  const uint32_t index = (uint32_t)tag;
  if (index >= b->nslots)
    return 0;

  struct ring_slot* s = &b->slots[index];
  platform_mutex_lock(b->mutex);
  const int holds = s->in_flight && s->claims == (uint32_t)(tag >> 32);
  platform_mutex_unlock(b->mutex);
  return holds;
}

// Give the ring the write the slot holds. Non-zero when there was no room and
// the slot is still the caller's; otherwise the write is in flight, because a
// prepared entry cannot be taken back.
static int
submit(struct io_backend_uring* b, uint32_t index)
{
  struct ring_slot* s = &b->slots[index];

  platform_mutex_lock(b->mutex);
  const uint32_t claims = s->claims;
  struct io_uring_sqe* sqe = io_uring_get_sqe(&b->ring);
  if (!sqe) {
    io_uring_submit(&b->ring);
    sqe = io_uring_get_sqe(&b->ring);
  }
  int rc = 0;
  if (sqe) {
    const uint64_t nbytes =
      s->rest.nbytes < MAX_WRITE_BYTES ? s->rest.nbytes : MAX_WRITE_BYTES;
    io_uring_prep_write(
      sqe, s->fd, s->rest.payload, (unsigned)nbytes, s->rest.offset);
    io_uring_sqe_set_data64(sqe, slot_tag(index, claims));
    s->in_flight = 1;
    rc = io_uring_submit(&b->ring);
  }
  platform_mutex_unlock(b->mutex);

  if (!sqe)
    return 1;
  if (rc >= 0)
    return 0;

  // Holding the lock over the wait would stop the completions being read,
  // which is what lets a turned-down submission through.
  for (int attempt = 1; attempt < SUBMIT_TRIES; ++attempt) {
    platform_sleep_ns(SUBMIT_RETRY_NS);
    platform_mutex_lock(b->mutex);
    rc = io_uring_submit(&b->ring);
    platform_mutex_unlock(b->mutex);
    if (rc >= 0)
      return 0;
  }

  // Another thread's submission may have carried the entry, and a write that
  // has already run is not one to call failed.
  if (!slot_holds_the_write(b, slot_tag(index, claims)))
    return 0;

  // The entry is still in the ring for the next submission to carry.
  log_error("io_backend_uring: cannot hand the ring a write: %s",
            strerror(-rc));
  atomic_store(b->io_error, 1);
  return 0;
}

static void
release(struct io_backend_uring* b, uint32_t index)
{
  platform_mutex_lock(b->mutex);
  b->slots[index].in_flight = 0;
  b->slots[index].next_free = b->free_head;
  b->free_head = index;
  platform_mutex_unlock(b->mutex);
}

static void
report(struct io_backend_uring* b, uint32_t index, uint64_t nbytes, int status)
{
  const uint64_t seq = b->slots[index].seq;
  if (status != IO_OK)
    atomic_store(b->io_error, 1);
  release(b, index);
  io_queue_complete(
    b->queue,
    (struct io_completion){ .seq = seq, .nbytes = nbytes, .status = status });
}

// The thread that would make room in the ring is the one running this, so the
// rest of a write is offered once rather than waited on.
static void
hand_over(struct io_backend_uring* b, uint32_t index)
{
  if (!submit(b, index))
    return;
  log_error("io_backend_uring: the ring had no room for the rest of a write");
  report(b, index, 0, IO_FAILED);
}

static void
finish(struct io_backend_uring* b, uint32_t index, int32_t res)
{
  struct ring_slot* s = &b->slots[index];

  if (res == -EINTR || res == -EAGAIN) {
    hand_over(b, index);
    return;
  }
  if (res <= 0) {
    log_error("io_backend_uring: write failed: %s",
              res ? strerror(-res) : "no bytes moved");
    report(b, index, 0, IO_FAILED);
    return;
  }

  s->rest = io_write_remaining(&s->rest, (uint64_t)res);
  if (s->rest.nbytes > 0) {
    hand_over(b, index);
    return;
  }
  report(b, index, s->asked, IO_OK);
}

static void
read_completions(void* arg)
{
  struct io_backend_uring* b = (struct io_backend_uring*)arg;
  int reported = 0;

  for (;;) {
    struct io_uring_cqe* cqe = NULL;
    const int rc = io_uring_wait_cqe(&b->ring, &cqe);
    if (rc == -EINTR)
      continue;
    if (rc < 0) {
      // A write the ring holds cannot be called failed while the kernel may
      // still be reading its payload, so the thread stays and tries again.
      if (!reported++)
        log_error("io_backend_uring: cannot read the ring: %s", strerror(-rc));
      atomic_store(b->io_error, 1);
      platform_sleep_ns(READ_RETRY_NS);
      continue;
    }

    const uint64_t tag = io_uring_cqe_get_data64(cqe);
    const int32_t res = cqe->res;
    io_uring_cqe_seen(&b->ring, cqe);

    if (tag == STOP_TAG)
      return;
    if (slot_holds_the_write(b, tag))
      finish(b, (uint32_t)tag, res);
  }
}

static int
uring_execute(void* ctx,
              const struct io_request* req,
              uint64_t seq,
              struct io_completion* out)
{
  struct io_backend_uring* b = (struct io_backend_uring*)ctx;

  // A ring reports a write of no bytes the way it reports a failure, and a
  // stale token has to be refused the same way either backend refuses it, so
  // anything the ring cannot carry goes to the filesystem backend behind it.
  platform_fd fd = PLATFORM_FD_INVALID;
  if (req->op != IO_OP_WRITE || req->nbytes == 0 || !atomic_load(&b->queue) ||
      !io_backend_fs_resolve(b->files, req->file, &fd))
    return b->blocking.execute(b->blocking.ctx, req, seq, out);

  platform_mutex_lock(b->mutex);

  const uint32_t index = b->free_head;
  if (index == FREE_LIST_END) {
    platform_mutex_unlock(b->mutex);
    return IO_BUSY;
  }

  // The slot is taken before the write is submitted, so it cannot be handed
  // out twice.
  struct ring_slot* s = &b->slots[index];
  b->free_head = s->next_free;
  *s = (struct ring_slot){
    .rest = io_write_remaining(req, 0),
    .seq = seq,
    .asked = req->nbytes,
    .fd = fd,
    .next_free = FREE_LIST_END,
    .claims = s->claims + 1,
  };
  platform_mutex_unlock(b->mutex);

  if (!submit(b, index))
    return IO_SUBMITTED;

  release(b, index);
  return IO_BUSY;
}

static void
uring_stop(void* ctx)
{
  struct io_backend_uring* b = (struct io_backend_uring*)ctx;
  if (!b->reader)
    return;

  platform_mutex_lock(b->mutex);
  struct io_uring_sqe* sqe = NULL;
  while (!(sqe = io_uring_get_sqe(&b->ring)))
    io_uring_submit(&b->ring);
  io_uring_prep_nop(sqe);
  io_uring_sqe_set_data64(sqe, STOP_TAG);
  io_uring_submit(&b->ring);
  platform_mutex_unlock(b->mutex);

  platform_thread_join(b->reader);
  b->reader = NULL;
}

struct io_backend_uring*
io_backend_uring_create(struct io_backend_fs* files,
                        _Atomic int* io_error,
                        uint64_t depth)
{
  CHECK(Fail, files);
  CHECK(Fail, io_error);
  if (!io_backend_uring_supported())
    return NULL;
  if (depth == 0)
    depth = DEFAULT_DEPTH;
  if (depth > IO_BACKEND_URING_MAX_DEPTH)
    depth = IO_BACKEND_URING_MAX_DEPTH;

  struct io_backend_uring* b =
    (struct io_backend_uring*)calloc(1, sizeof(struct io_backend_uring));
  CHECK(Fail, b);

  b->files = files;
  b->blocking = io_backend_fs_as_backend(files);
  b->io_error = io_error;
  b->nslots = (uint32_t)depth;
  b->free_head = FREE_LIST_END;

  b->mutex = platform_mutex_new();
  CHECK(Fail_alloc, b->mutex);

  b->slots =
    (struct ring_slot*)calloc((size_t)b->nslots, sizeof(struct ring_slot));
  CHECK(Fail_mutex, b->slots);
  for (uint32_t i = b->nslots; i > 0; --i) {
    b->slots[i - 1].next_free = b->free_head;
    b->free_head = i - 1;
  }

  const int rc = io_uring_queue_init(b->nslots, &b->ring, 0);
  if (rc < 0) {
    log_error("io_backend_uring: could not open a ring: %s", strerror(-rc));
    goto Fail_slots;
  }

  return b;

Fail_slots:
  free(b->slots);
Fail_mutex:
  platform_mutex_free(b->mutex);
Fail_alloc:
  free(b);
Fail:
  return NULL;
}

int
io_backend_uring_start(struct io_backend_uring* b, struct io_queue* queue)
{
  CHECK(Fail, b);
  CHECK(Fail, queue);
  CHECK(Fail, !b->reader);

  b->queue = queue;
  b->reader = platform_thread_start(read_completions, b);
  CHECK(Fail, b->reader);
  return 0;

Fail:
  return 1;
}

void
io_backend_uring_destroy(struct io_backend_uring* b)
{
  if (!b)
    return;

  uring_stop(b);
  io_uring_queue_exit(&b->ring);
  platform_mutex_free(b->mutex);
  free(b->slots);
  free(b);
}

struct io_backend
io_backend_uring_as_backend(struct io_backend_uring* b)
{
  return (struct io_backend){ .ctx = b,
                              .execute = uring_execute,
                              .stop = uring_stop };
}

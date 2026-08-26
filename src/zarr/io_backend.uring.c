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
// teardown posts one request carrying this instead of a slot number. No slot
// reaches it: a slot number is small and sits in the low half.
#define STOP_TAG UINT64_MAX

// The ring counts a write's bytes in 32 bits, and a run of them has to stay
// aligned for an unbuffered file, so a longer write is split on a page
// boundary and finished the way a short one is.
#define MAX_WRITE_BYTES (1u << 30)

// A ring that cannot be read is a broken kernel rather than a busy one, so it
// is tried again on a slow timer instead of spun on.
#define READ_RETRY_NS 100000000LL

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

// Which write a completion belongs to. The claim count tells a completion for
// the write in the slot now from one for a write that is long gone.
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
  // The queue is built from the backend, so the workers exist before this is
  // set and read it from their own threads.
  struct io_queue* _Atomic queue;
  _Atomic int* io_error;

  struct platform_mutex* mutex; // guards submission and the free list
  struct ring_slot* slots;
  uint32_t nslots;
  uint32_t free_head;

  struct platform_thread* reader;
};

// -1 until a ring has been asked for, then 0 or 1.
static _Atomic int ring_probed = -1;
static _Atomic int ring_probe_reported = 0;

int
io_backend_uring_supported(void)
{
  const int known = atomic_load(&ring_probed);
  if (known >= 0)
    return known;

  struct io_uring ring;
  const int rc = io_uring_queue_init(PROBE_ENTRIES, &ring, 0);
  if (rc == 0)
    io_uring_queue_exit(&ring);
  else if (atomic_exchange(&ring_probe_reported, 1) == 0)
    log_error("io_backend_uring: no ring here (%s); writing on the workers",
              strerror(-rc));

  atomic_store(&ring_probed, rc == 0);
  return rc == 0;
}

// 0 when the write was handed to the ring, 1 when the ring had no room for
// it, and -1 when the ring refused it.
static int
submit_locked(struct io_backend_uring* b, uint32_t index)
{
  struct ring_slot* s = &b->slots[index];

  struct io_uring_sqe* sqe = io_uring_get_sqe(&b->ring);
  if (!sqe) {
    io_uring_submit(&b->ring);
    sqe = io_uring_get_sqe(&b->ring);
    if (!sqe)
      return 1;
  }

  const uint64_t nbytes =
    s->rest.nbytes < MAX_WRITE_BYTES ? s->rest.nbytes : MAX_WRITE_BYTES;
  io_uring_prep_write(
    sqe, s->fd, s->rest.payload, (unsigned)nbytes, s->rest.offset);
  io_uring_sqe_set_data64(sqe, slot_tag(index, s->claims));
  s->in_flight = 1;

  if (io_uring_submit(&b->ring) >= 0)
    return 0;

  // The entry stays in the ring whether or not the call went through, so a
  // completion carrying this slot can still arrive. It is dropped, because
  // the slot is no longer waiting for one.
  s->in_flight = 0;
  return -1;
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

// The queue does not read a status, so the pool's flag is raised here.
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

// The thread reading the ring is the only one that empties it, so the rest of
// a write is offered once rather than waited on here.
static void
hand_over(struct io_backend_uring* b, uint32_t index)
{
  platform_mutex_lock(b->mutex);
  const int answer = submit_locked(b, index);
  platform_mutex_unlock(b->mutex);

  if (answer == 0)
    return;
  log_error("io_backend_uring: the ring would not take the rest of a write");
  report(b, index, 0, IO_FAILED);
}

// The write a completion belongs to, or NULL when it belongs to none: an entry
// left in the ring by a submission that did not go through still completes,
// and by then its slot may be free or holding another write.
static struct ring_slot*
slot_of(struct io_backend_uring* b, uint64_t tag)
{
  const uint32_t index = (uint32_t)tag;
  if (index >= b->nslots)
    return NULL;

  struct ring_slot* s = &b->slots[index];
  platform_mutex_lock(b->mutex);
  const int ours = s->in_flight && s->claims == (uint32_t)(tag >> 32);
  platform_mutex_unlock(b->mutex);
  return ours ? s : NULL;
}

static void
finish(struct io_backend_uring* b, uint32_t index, int32_t res)
{
  struct ring_slot* s = &b->slots[index];

  if (res == -EINTR || res == -EAGAIN) {
    hand_over(b, index);
    return;
  }
  if (res < 0) {
    log_error("io_backend_uring: write failed: %s", strerror(-res));
    report(b, index, 0, IO_FAILED);
    return;
  }
  if (res == 0) {
    log_error("io_backend_uring: a write of %llu bytes moved none",
              (unsigned long long)s->rest.nbytes);
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
    if (slot_of(b, tag))
      finish(b, (uint32_t)tag, res);
  }
}

static void
record_failure(struct io_backend_uring* b,
               struct io_completion* out,
               const char* message)
{
  log_error("%s", message);
  atomic_store(b->io_error, 1);
  out->nbytes = 0;
  out->status = IO_FAILED;
}

static int
uring_execute(void* ctx,
              const struct io_request* req,
              uint64_t seq,
              struct io_completion* out)
{
  struct io_backend_uring* b = (struct io_backend_uring*)ctx;

  if (req->op != IO_OP_WRITE)
    return b->blocking.execute(b->blocking.ctx, req, seq, out);

  out->seq = seq;
  if (!atomic_load(&b->queue)) {
    record_failure(b, out, "io_backend_uring: no queue to report writes to");
    return IO_DONE;
  }

  platform_fd fd = PLATFORM_FD_INVALID;
  if (!io_backend_fs_resolve(b->files, req->file, &fd)) {
    record_failure(b, out, "io_backend_uring: stale file token");
    return IO_DONE;
  }

  // A ring reports a write of no bytes as one that moved none, which is how
  // it reports a failure too.
  if (req->nbytes == 0) {
    out->nbytes = 0;
    return IO_DONE;
  }

  platform_mutex_lock(b->mutex);

  const uint32_t index = b->free_head;
  if (index == FREE_LIST_END) {
    platform_mutex_unlock(b->mutex);
    return IO_BUSY;
  }

  struct ring_slot* s = &b->slots[index];
  *s = (struct ring_slot){
    .rest = io_write_remaining(req, 0),
    .seq = seq,
    .asked = req->nbytes,
    .fd = fd,
    .next_free = s->next_free,
    .claims = s->claims + 1,
  };

  const uint32_t next_free = s->next_free;
  const int answer = submit_locked(b, index);
  if (answer == 0)
    b->free_head = next_free;
  platform_mutex_unlock(b->mutex);

  if (answer == 0)
    return IO_SUBMITTED;
  if (answer > 0)
    return IO_BUSY;

  record_failure(b, out, "io_backend_uring: the ring turned a write down");
  return IO_DONE;
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

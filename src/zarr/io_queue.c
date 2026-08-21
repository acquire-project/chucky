#include "zarr/io_queue.h"
#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/io_queue_stats.h"

#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_REQUESTS 1024u

enum io_job_state
{
  IO_JOB_WAITING = 0,
  IO_JOB_RUNNING,
  IO_JOB_RETIRED,
};

struct io_job
{
  struct io_request req;
  uint64_t seq;
  int64_t post_ns;
  int64_t started_ns;
  uint8_t state;
};

// One open file with requests in the queue.
struct file_pending
{
  uint64_t generation;
  uint64_t outstanding;
  uint8_t closing;
};

struct io_queue
{
  struct platform_thread* thread;
  struct platform_mutex* mutex;
  struct platform_cond* cond_not_empty;
  struct platform_cond* cond_retired;
  struct platform_cond* cond_slot_free;

  struct io_job* ring;
  uint64_t ring_cap; // power of two, fixed for the queue's life
  uint64_t head;     // next sequence number to hand out
  uint64_t tail;     // oldest sequence number that has not retired

  struct file_pending* files;
  uint64_t nfiles;
  uint64_t files_cap;

  uint64_t max_requests;
  uint64_t max_bytes;

  uint64_t pending_bytes; // raised on commit, lowered when the job finishes
  uint64_t jobs_waiting;  // committed and not yet taken by a worker

  // Room claimed by a reserve that has not yet committed or been released.
  uint64_t reserved_requests;
  uint64_t reserved_bytes;

  // Threads parked on one of the condition variables. Teardown cannot free
  // them until every one of those threads has let go of the mutex.
  uint64_t waiters;

  struct io_queue_counters counters;

  struct io_backend backend;

  int shutdown;
};

// Teardown frees the mutex and the condition variables, so it has to know
// when a parked thread is still on its way out of one.
static void
queue_wait(struct io_queue* q, struct platform_cond* cond)
{
  q->waiters++;
  platform_cond_wait(cond, q->mutex);
  if (--q->waiters == 0)
    platform_cond_broadcast(q->cond_retired);
}

static struct file_pending*
file_find(struct io_queue* q, uint64_t generation)
{
  for (uint64_t i = 0; i < q->nfiles; ++i) {
    if (q->files[i].generation == generation)
      return &q->files[i];
  }
  return NULL;
}

static void
file_request_added(struct io_queue* q, const struct io_request* req)
{
  if (req->file.generation == 0)
    return;

  struct file_pending* f = file_find(q, req->file.generation);
  if (!f) {
    if (q->nfiles == q->files_cap) {
      uint64_t cap = q->files_cap ? q->files_cap * 2 : 16;
      struct file_pending* grown =
        (struct file_pending*)realloc(q->files, cap * sizeof(*grown));
      // A write must not fail over the table, and a missing entry only makes
      // a barrier look ready.
      if (!grown) {
        log_error("io_queue: could not grow the open file table");
        return;
      }
      q->files = grown;
      q->files_cap = cap;
    }
    f = &q->files[q->nfiles++];
    *f = (struct file_pending){ .generation = req->file.generation };
  }

  f->outstanding++;
  if (req->op == IO_OP_CLOSE)
    f->closing = 1;
}

static void
file_request_retired(struct io_queue* q, const struct io_request* req)
{
  if (req->file.generation == 0)
    return;

  struct file_pending* f = file_find(q, req->file.generation);
  if (!f)
    return;

  // An entry left at zero would never be removed, and every later barrier on
  // the file would wait on it forever.
  CHECK(Done, f->outstanding > 0);

  if (--f->outstanding == 0) {
    *f = q->files[q->nfiles - 1];
    q->nfiles--;
  }

Done:;
}

static int
is_barrier(const struct io_request* req)
{
  return req->op == IO_OP_TRUNCATE || req->op == IO_OP_CLOSE;
}

// Requests naming one file run in the order they were posted, and a barrier
// is held back until every other request naming that file has retired.
static int
job_is_ready(struct io_queue* q, const struct io_job* job)
{
  const uint64_t generation = job->req.file.generation;
  if (generation == 0)
    return 1;

  const struct file_pending* f = file_find(q, generation);
  if (!f || f->outstanding == 1)
    return 1;

  const int barrier = is_barrier(&job->req);
  for (uint64_t s = q->tail; s < job->seq; ++s) {
    const struct io_job* earlier = &q->ring[s & (q->ring_cap - 1)];
    if (earlier->state == IO_JOB_RETIRED)
      continue;
    if (earlier->req.file.generation != generation)
      continue;
    if (barrier || is_barrier(&earlier->req))
      return 0;
  }
  return 1;
}

// Zero when nothing can run right now. A barrier that is not ready is passed
// over, so one file's barrier never holds up another file's writes.
static uint64_t
next_ready_seq(struct io_queue* q)
{
  for (uint64_t s = q->tail; s < q->head; ++s) {
    const struct io_job* candidate = &q->ring[s & (q->ring_cap - 1)];
    if (candidate->state != IO_JOB_WAITING)
      continue;
    if (job_is_ready(q, candidate))
      return s;
  }
  return 0;
}

// Room is given back against what the request asked for, not what the
// completion reports: a partial write would otherwise leak the difference.
static void
retire_job(struct io_queue* q, struct io_completion c, int64_t finished_ns)
{
  void* owned = NULL;
  void (*owned_free)(void*) = NULL;

  platform_mutex_lock(q->mutex);

  struct io_job* slot = &q->ring[c.seq & (q->ring_cap - 1)];
  CHECK(Unlock, slot->seq == c.seq && slot->state == IO_JOB_RUNNING);

  const struct io_job job = *slot;
  owned = job.req.owned;
  owned_free = job.req.owned_free;

  slot->state = IO_JOB_RETIRED;
  while (q->tail < q->head &&
         q->ring[q->tail & (q->ring_cap - 1)].state == IO_JOB_RETIRED)
    q->tail++;
  q->pending_bytes -= job.req.nbytes;
  file_request_retired(q, &job.req);
  io_queue_counters_finished(
    &q->counters, &job.req, job.post_ns, job.started_ns, finished_ns);
  // Without this, a barrier that was passed over is never looked at again.
  platform_cond_broadcast(q->cond_not_empty);
  platform_cond_broadcast(q->cond_retired);
  platform_cond_broadcast(q->cond_slot_free);

Unlock:
  platform_mutex_unlock(q->mutex);
  // Freeing under the lock would hold every poster behind the allocator.
  if (owned)
    owned_free(owned);
}

static void
worker_thread(void* arg)
{
  struct io_queue* q = (struct io_queue*)arg;

  for (;;) {
    struct io_job job;

    platform_mutex_lock(q->mutex);
    uint64_t seq;
    // Only an empty window means everything drained; a submitted request is
    // still running, and freeing the ring under its completion would tear.
    while ((seq = next_ready_seq(q)) == 0) {
      if (q->shutdown && q->tail == q->head)
        break;
      platform_cond_wait(q->cond_not_empty, q->mutex);
    }

    if (seq == 0) {
      platform_mutex_unlock(q->mutex);
      break;
    }

    struct io_job* slot = &q->ring[seq & (q->ring_cap - 1)];
    slot->state = IO_JOB_RUNNING;
    // Untimed for truncate and close: no payload, so no clock reads.
    const int timed = slot->req.nbytes > 0;
    slot->started_ns = timed ? platform_monotonic_ns() : 0;
    q->jobs_waiting--;
    job = *slot;
    platform_mutex_unlock(q->mutex);

    struct io_completion done = {
      .seq = job.seq,
      .nbytes = job.req.nbytes,
      .status = IO_OK,
    };
    int dispatch = IO_DONE;
    if (q->backend.execute)
      dispatch = q->backend.execute(q->backend.ctx, &job.req, job.seq, &done);

    if (dispatch == IO_DONE)
      retire_job(q, done, timed ? platform_monotonic_ns() : 0);
  }
}

static uint64_t
round_up_pow2(uint64_t v)
{
  uint64_t p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

struct io_queue*
io_queue_create(struct io_backend backend, struct io_queue_limits limits)
{
  struct io_queue* q = (struct io_queue*)calloc(1, sizeof(*q));
  if (!q)
    return NULL;

  q->backend = backend;
  q->max_requests =
    limits.max_requests ? limits.max_requests : DEFAULT_MAX_REQUESTS;
  q->max_bytes = limits.max_bytes;
  // Masking the sequence number picks a slot, so the ring must be a power of
  // two even when the request limit is not.
  q->ring_cap = round_up_pow2(q->max_requests);
  // Sequence number zero means nothing was ever posted, so counting starts
  // at one and an event recorded before the first post waits on nothing.
  q->head = 1;
  q->tail = 1;
  q->ring = (struct io_job*)calloc(q->ring_cap, sizeof(struct io_job));
  q->mutex = platform_mutex_new();
  q->cond_not_empty = platform_cond_new();
  q->cond_retired = platform_cond_new();
  q->cond_slot_free = platform_cond_new();
  if (q->ring && q->mutex && q->cond_not_empty && q->cond_retired &&
      q->cond_slot_free)
    q->thread = platform_thread_start(worker_thread, q);

  if (!q->thread) {
    platform_cond_free(q->cond_slot_free);
    platform_cond_free(q->cond_retired);
    platform_cond_free(q->cond_not_empty);
    platform_mutex_free(q->mutex);
    free(q->ring);
    free(q);
    return NULL;
  }

  return q;
}

void
io_queue_destroy(struct io_queue* q)
{
  if (!q)
    return;

  platform_mutex_lock(q->mutex);
  q->shutdown = 1;
  platform_cond_broadcast(q->cond_not_empty);
  platform_cond_broadcast(q->cond_slot_free);
  platform_cond_broadcast(q->cond_retired);
  platform_mutex_unlock(q->mutex);

  platform_thread_join(q->thread);

  // Shutdown is set, so nothing parks from here on. Freeing the mutex while a
  // parked thread is still on its way out would pull it out from under them.
  platform_mutex_lock(q->mutex);
  while (q->waiters)
    platform_cond_wait(q->cond_retired, q->mutex);
  platform_mutex_unlock(q->mutex);

  free(q->ring);
  free(q->files);
  io_queue_counters_free(&q->counters);
  platform_cond_free(q->cond_slot_free);
  platform_cond_free(q->cond_retired);
  platform_cond_free(q->cond_not_empty);
  platform_mutex_free(q->mutex);
  free(q);
}

uint64_t
io_queue_parked_threads(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  platform_mutex_lock(mq->mutex);
  uint64_t parked = mq->waiters;
  platform_mutex_unlock(mq->mutex);
  return parked;
}

static int
has_room(const struct io_queue* q, uint64_t nbytes)
{
  if ((q->head - q->tail) + q->reserved_requests >= q->max_requests)
    return 0;
  if (q->max_bytes == 0)
    return 1;
  // A write larger than the ceiling still has to go through, so an empty
  // queue admits it.
  if (q->head == q->tail && q->reserved_requests == 0)
    return 1;
  return q->pending_bytes + q->reserved_bytes + nbytes <= q->max_bytes;
}

static int
names_a_closing_file(struct io_queue* q, const struct io_request* req)
{
  if (req->file.generation == 0)
    return 0;
  if (req->op != IO_OP_WRITE && req->op != IO_OP_TRUNCATE &&
      req->op != IO_OP_CLOSE)
    return 0;

  const struct file_pending* f = file_find(q, req->file.generation);
  return f && f->closing;
}

int
io_queue_reserve(struct io_queue* q, struct io_request req)
{
  const uint64_t nbytes = req.nbytes;

  platform_mutex_lock(q->mutex);

  while (!has_room(q, nbytes) && !q->shutdown)
    queue_wait(q, q->cond_slot_free);

  int refused = 0;
  if (q->shutdown) {
    log_error("io_queue: refused a post during shutdown");
    refused = 1;
  } else if (names_a_closing_file(q, &req)) {
    // Checked after the wait: the file can start closing while room is short.
    log_error("io_queue: refused a request naming a file that is closing");
    refused = 1;
  }

  if (!refused) {
    q->reserved_requests++;
    q->reserved_bytes += nbytes;
  }
  platform_mutex_unlock(q->mutex);
  return refused;
}

void
io_queue_commit(struct io_queue* q, struct io_request req)
{
  platform_mutex_lock(q->mutex);

  CHECK(Unlock, q->head - q->tail < q->ring_cap);
  CHECK(Unlock, q->reserved_requests > 0);
  CHECK(Unlock, q->reserved_bytes >= req.nbytes);

  q->reserved_requests--;
  q->reserved_bytes -= req.nbytes;

  const int64_t now = platform_monotonic_ns();
  const uint64_t seq = q->head;
  q->ring[seq & (q->ring_cap - 1)] = (struct io_job){
    .req = req,
    .seq = seq,
    .post_ns = now,
  };
  q->head++;
  q->pending_bytes += req.nbytes;
  q->jobs_waiting++;
  file_request_added(q, &req);

  io_queue_counters_posted(
    &q->counters, &req, q->jobs_waiting, q->pending_bytes, now);

  platform_cond_broadcast(q->cond_not_empty);

Unlock:
  platform_mutex_unlock(q->mutex);
}

void
io_queue_release(struct io_queue* q, uint64_t nbytes)
{
  platform_mutex_lock(q->mutex);
  CHECK(Unlock, q->reserved_requests > 0);
  CHECK(Unlock, q->reserved_bytes >= nbytes);
  q->reserved_requests--;
  q->reserved_bytes -= nbytes;
  platform_cond_broadcast(q->cond_slot_free);

Unlock:
  platform_mutex_unlock(q->mutex);
}

void
io_queue_complete(struct io_queue* q, struct io_completion c)
{
  retire_job(q, c, platform_monotonic_ns());
}

int
io_queue_post(struct io_queue* q, struct io_request req)
{
  if (io_queue_reserve(q, req))
    return 1;
  io_queue_commit(q, req);
  return 0;
}

uint64_t
io_queue_pending_bytes(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  platform_mutex_lock(mq->mutex);
  uint64_t pending = mq->pending_bytes;
  platform_mutex_unlock(mq->mutex);
  return pending;
}

void
io_queue_get_stats(const struct io_queue* q, struct io_queue_stats* out)
{
  struct io_queue* mq = (struct io_queue*)q;
  platform_mutex_lock(mq->mutex);
  io_queue_counters_read(&mq->counters, out, platform_monotonic_ns());
  platform_mutex_unlock(mq->mutex);
}

struct io_event
io_queue_record(struct io_queue* q)
{
  platform_mutex_lock(q->mutex);
  struct io_event ev = { .seq = q->head - 1 };
  platform_mutex_unlock(q->mutex);
  return ev;
}

void
io_event_wait(const struct io_queue* q, struct io_event ev)
{
  // Cast away const for mutex operations — the mutable sync state is logically
  // separate from the queue's public identity.
  struct io_queue* mq = (struct io_queue*)q;

  platform_mutex_lock(mq->mutex);
  while (mq->tail - 1 < ev.seq && !mq->shutdown)
    queue_wait(mq, mq->cond_retired);
  platform_mutex_unlock(mq->mutex);
}

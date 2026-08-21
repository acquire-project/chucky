#include "zarr/io_queue.h"
#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/io_queue_stats.h"

#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_REQUESTS 1024u

struct io_job
{
  struct io_request req;
  uint64_t seq;
  int64_t post_ns;
  uint8_t retired;
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
  uint64_t dispatch; // next sequence number the worker runs
  uint64_t tail;     // oldest sequence number that has not retired

  uint64_t max_requests;
  uint64_t max_bytes;

  uint64_t pending_bytes; // raised on commit, lowered when the job finishes

  // Room claimed by a reserve that has not yet committed or been released.
  uint64_t reserved_requests;
  uint64_t reserved_bytes;

  struct io_queue_counters counters;

  struct io_backend backend;

  int shutdown;
};

static void
worker_thread(void* arg)
{
  struct io_queue* q = (struct io_queue*)arg;

  for (;;) {
    struct io_job job;

    platform_mutex_lock(q->mutex);
    while (q->dispatch == q->head && !q->shutdown)
      platform_cond_wait(q->cond_not_empty, q->mutex);

    if (q->dispatch == q->head && q->shutdown) {
      platform_mutex_unlock(q->mutex);
      break;
    }

    job = q->ring[q->dispatch & (q->ring_cap - 1)];
    q->dispatch++;
    platform_mutex_unlock(q->mutex);

    // Untimed for truncate and close: no payload, so no clock reads.
    const int timed = job.req.nbytes > 0;
    const int64_t started_ns = timed ? platform_monotonic_ns() : 0;
    struct io_completion done = {
      .seq = job.seq,
      .nbytes = job.req.nbytes,
      .status = IO_OK,
    };
    if (job.req.op == IO_OP_CALL) {
      job.req.fn(job.req.ctx);
      if (job.req.ctx_free)
        job.req.ctx_free(job.req.ctx);
    } else if (q->backend.execute) {
      q->backend.execute(q->backend.ctx, &job.req, job.seq, &done);
    }
    if (job.req.owned)
      job.req.owned_free(job.req.owned);
    const int64_t finished_ns = timed ? platform_monotonic_ns() : 0;

    platform_mutex_lock(q->mutex);
    q->ring[job.seq & (q->ring_cap - 1)].retired = 1;
    while (q->tail < q->head && q->ring[q->tail & (q->ring_cap - 1)].retired) {
      q->ring[q->tail & (q->ring_cap - 1)].retired = 0;
      q->tail++;
    }
    q->pending_bytes -= job.req.nbytes;
    io_queue_counters_finished(
      &q->counters, &job.req, job.post_ns, started_ns, finished_ns);
    platform_cond_broadcast(q->cond_retired);
    platform_cond_broadcast(q->cond_slot_free);
    platform_mutex_unlock(q->mutex);
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
  q->dispatch = 1;
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
  platform_mutex_unlock(q->mutex);

  platform_thread_join(q->thread);

  free(q->ring);
  io_queue_counters_free(&q->counters);
  platform_cond_free(q->cond_slot_free);
  platform_cond_free(q->cond_retired);
  platform_cond_free(q->cond_not_empty);
  platform_mutex_free(q->mutex);
  free(q);
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

int
io_queue_reserve(struct io_queue* q, uint64_t nbytes)
{
  platform_mutex_lock(q->mutex);

  while (!has_room(q, nbytes) && !q->shutdown)
    platform_cond_wait(q->cond_slot_free, q->mutex);

  if (!has_room(q, nbytes)) {
    log_error("io_queue: refused a post during shutdown");
    platform_mutex_unlock(q->mutex);
    return 1;
  }

  q->reserved_requests++;
  q->reserved_bytes += nbytes;
  platform_mutex_unlock(q->mutex);
  return 0;
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

  io_queue_counters_posted(
    &q->counters, &req, q->head - q->dispatch, q->pending_bytes, now);

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

int
io_queue_post(struct io_queue* q, struct io_request req)
{
  if (io_queue_reserve(q, req.nbytes))
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
    platform_cond_wait(mq->cond_retired, mq->mutex);
  platform_mutex_unlock(mq->mutex);
}

int
io_queue_is_shutdown(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  platform_mutex_lock(mq->mutex);
  int r = mq->shutdown;
  platform_mutex_unlock(mq->mutex);
  return r;
}

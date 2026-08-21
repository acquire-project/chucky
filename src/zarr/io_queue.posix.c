#include "zarr/io_queue.h"
#include "log/log.h"
#include "platform/platform.h"
#include "zarr/io_queue_stats.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct io_job
{
  struct io_work work;
  uint64_t seq;
  int64_t post_ns;
};

struct io_queue
{
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t cond_not_empty;
  pthread_cond_t cond_retired;

  struct io_job* ring;
  uint64_t ring_cap; // power of 2
  uint64_t head;     // next write position (post)
  uint64_t tail;     // next read position (worker)

  uint64_t next_seq;      // incremented on post
  uint64_t retired_seq;   // updated after each job completes
  uint64_t pending_bytes; // raised on post, lowered when the job finishes

  struct io_queue_counters counters;

  int shutdown;
  int started;
};

static void*
worker_thread(void* arg)
{
  struct io_queue* q = (struct io_queue*)arg;

  for (;;) {
    struct io_job job;

    pthread_mutex_lock(&q->mutex);
    while (q->head == q->tail && !q->shutdown)
      pthread_cond_wait(&q->cond_not_empty, &q->mutex);

    if (q->head == q->tail && q->shutdown) {
      pthread_mutex_unlock(&q->mutex);
      break;
    }

    job = q->ring[q->tail & (q->ring_cap - 1)];
    q->tail++;
    pthread_mutex_unlock(&q->mutex);

    // Only work carrying a payload is timed, so a truncate or close does not
    // pay for two clock reads nobody looks at.
    const int timed = job.work.nbytes > 0;
    const int64_t started_ns = timed ? platform_monotonic_ns() : 0;
    job.work.fn(job.work.ctx);
    if (job.work.ctx_free)
      job.work.ctx_free(job.work.ctx);
    const int64_t finished_ns = timed ? platform_monotonic_ns() : 0;

    pthread_mutex_lock(&q->mutex);
    q->retired_seq = job.seq;
    q->pending_bytes -= job.work.nbytes;
    io_queue_counters_finished(
      &q->counters, &job.work, job.post_ns, started_ns, finished_ns);
    pthread_cond_broadcast(&q->cond_retired);
    pthread_mutex_unlock(&q->mutex);
  }

  return NULL;
}

struct io_queue*
io_queue_create(void)
{
  struct io_queue* q = (struct io_queue*)calloc(1, sizeof(*q));
  if (!q)
    return NULL;

  q->ring_cap = 64;
  q->ring = (struct io_job*)calloc(q->ring_cap, sizeof(struct io_job));
  if (!q->ring) {
    free(q);
    return NULL;
  }

  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->cond_not_empty, NULL);
  pthread_cond_init(&q->cond_retired, NULL);

  if (pthread_create(&q->thread, NULL, worker_thread, q) != 0) {
    free(q->ring);
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond_not_empty);
    pthread_cond_destroy(&q->cond_retired);
    free(q);
    return NULL;
  }
  q->started = 1;

  return q;
}

void
io_queue_destroy(struct io_queue* q)
{
  if (!q)
    return;

  pthread_mutex_lock(&q->mutex);
  q->shutdown = 1;
  pthread_cond_signal(&q->cond_not_empty);
  pthread_mutex_unlock(&q->mutex);

  if (q->started)
    pthread_join(q->thread, NULL);

  free(q->ring);
  io_queue_counters_free(&q->counters);
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->cond_not_empty);
  pthread_cond_destroy(&q->cond_retired);
  free(q);
}

static int
ring_grow(struct io_queue* q)
{
  uint64_t new_cap = q->ring_cap * 2;
  struct io_job* new_ring =
    (struct io_job*)calloc(new_cap, sizeof(struct io_job));
  if (!new_ring) {
    log_error("io_queue: failed to grow ring buffer");
    return 1;
  }

  // Copy existing jobs preserving order
  uint64_t count = q->head - q->tail;
  for (uint64_t i = 0; i < count; ++i)
    new_ring[i] = q->ring[(q->tail + i) & (q->ring_cap - 1)];

  free(q->ring);
  q->ring = new_ring;
  q->ring_cap = new_cap;
  q->head = count;
  q->tail = 0;
  return 0;
}

int
io_queue_post(struct io_queue* q, struct io_work work)
{
  pthread_mutex_lock(&q->mutex);

  if (q->head - q->tail == q->ring_cap) {
    if (ring_grow(q)) {
      pthread_mutex_unlock(&q->mutex);
      return 1;
    }
  }

  q->next_seq++;
  const int64_t now = platform_monotonic_ns();
  q->ring[q->head & (q->ring_cap - 1)] = (struct io_job){
    .work = work,
    .seq = q->next_seq,
    .post_ns = now,
  };
  q->head++;
  q->pending_bytes += work.nbytes;

  io_queue_counters_posted(
    &q->counters, &work, q->head - q->tail, q->pending_bytes, now);

  pthread_cond_signal(&q->cond_not_empty);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

uint64_t
io_queue_pending_bytes(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  pthread_mutex_lock(&mq->mutex);
  uint64_t pending = mq->pending_bytes;
  pthread_mutex_unlock(&mq->mutex);
  return pending;
}

void
io_queue_get_stats(const struct io_queue* q, struct io_queue_stats* out)
{
  struct io_queue* mq = (struct io_queue*)q;
  pthread_mutex_lock(&mq->mutex);
  io_queue_counters_read(&mq->counters, out, platform_monotonic_ns());
  pthread_mutex_unlock(&mq->mutex);
}

struct io_event
io_queue_record(struct io_queue* q)
{
  pthread_mutex_lock(&q->mutex);
  struct io_event ev = { .seq = q->next_seq };
  pthread_mutex_unlock(&q->mutex);
  return ev;
}

void
io_event_wait(const struct io_queue* q, struct io_event ev)
{
  // Cast away const for mutex operations — the mutable sync state is logically
  // separate from the queue's public identity.
  struct io_queue* mq = (struct io_queue*)q;

  pthread_mutex_lock(&mq->mutex);
  while (mq->retired_seq < ev.seq && !mq->shutdown)
    pthread_cond_wait(&mq->cond_retired, &mq->mutex);
  pthread_mutex_unlock(&mq->mutex);
}

int
io_queue_is_shutdown(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  pthread_mutex_lock(&mq->mutex);
  int r = mq->shutdown;
  pthread_mutex_unlock(&mq->mutex);
  return r;
}

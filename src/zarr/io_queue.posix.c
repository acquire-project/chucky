#include "zarr/io_queue.h"
#include "log/log.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct io_job
{
  void (*fn)(void*);
  void* ctx;
  void (*ctx_free)(void*);
  uint64_t seq;
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

  uint64_t next_seq;    // incremented on post
  uint64_t retired_seq; // updated after each job completes

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

    job.fn(job.ctx);
    if (job.ctx_free)
      job.ctx_free(job.ctx);

    pthread_mutex_lock(&q->mutex);
    q->retired_seq = job.seq;
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
io_queue_post(struct io_queue* q,
              void (*fn)(void*),
              void* ctx,
              void (*ctx_free)(void*))
{
  pthread_mutex_lock(&q->mutex);

  if (q->head - q->tail == q->ring_cap) {
    if (ring_grow(q)) {
      pthread_mutex_unlock(&q->mutex);
      return 1;
    }
  }

  q->next_seq++;
  q->ring[q->head & (q->ring_cap - 1)] = (struct io_job){
    .fn = fn,
    .ctx = ctx,
    .ctx_free = ctx_free,
    .seq = q->next_seq,
  };
  q->head++;

  pthread_cond_signal(&q->cond_not_empty);
  pthread_mutex_unlock(&q->mutex);
  return 0;
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

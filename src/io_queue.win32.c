#include "io_queue.h"
#include "log/log.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

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
  HANDLE thread;
  SRWLOCK srw;
  CONDITION_VARIABLE cond_not_empty;
  CONDITION_VARIABLE cond_retired;

  struct io_job* ring;
  uint64_t ring_cap; // power of 2
  uint64_t head;     // next write position (post)
  uint64_t tail;     // next read position (worker)

  uint64_t next_seq;    // incremented on post
  uint64_t retired_seq; // updated after each job completes

  int shutdown;
  int started;
};

static DWORD WINAPI
worker_thread(LPVOID arg)
{
  struct io_queue* q = (struct io_queue*)arg;

  for (;;) {
    struct io_job job;

    AcquireSRWLockExclusive(&q->srw);
    while (q->head == q->tail && !q->shutdown)
      SleepConditionVariableSRW(&q->cond_not_empty, &q->srw, INFINITE, 0);

    if (q->head == q->tail && q->shutdown) {
      ReleaseSRWLockExclusive(&q->srw);
      break;
    }

    job = q->ring[q->tail & (q->ring_cap - 1)];
    q->tail++;
    ReleaseSRWLockExclusive(&q->srw);

    job.fn(job.ctx);
    if (job.ctx_free)
      job.ctx_free(job.ctx);

    AcquireSRWLockExclusive(&q->srw);
    q->retired_seq = job.seq;
    WakeAllConditionVariable(&q->cond_retired);
    ReleaseSRWLockExclusive(&q->srw);
  }

  return 0;
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

  q->srw = (SRWLOCK)SRWLOCK_INIT;
  InitializeConditionVariable(&q->cond_not_empty);
  InitializeConditionVariable(&q->cond_retired);

  q->thread = CreateThread(NULL, 0, worker_thread, q, 0, NULL);
  if (!q->thread) {
    free(q->ring);
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

  AcquireSRWLockExclusive(&q->srw);
  q->shutdown = 1;
  WakeConditionVariable(&q->cond_not_empty);
  ReleaseSRWLockExclusive(&q->srw);

  if (q->started)
    WaitForSingleObject(q->thread, INFINITE);

  if (q->thread)
    CloseHandle(q->thread);

  free(q->ring);
  free(q);
}

static void
ring_grow(struct io_queue* q)
{
  uint64_t new_cap = q->ring_cap * 2;
  struct io_job* new_ring =
    (struct io_job*)calloc(new_cap, sizeof(struct io_job));
  if (!new_ring) {
    log_error("io_queue: failed to grow ring buffer");
    return;
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
}

uint64_t
io_queue_post(struct io_queue* q,
              void (*fn)(void*),
              void* ctx,
              void (*ctx_free)(void*))
{
  AcquireSRWLockExclusive(&q->srw);

  q->next_seq++;
  uint64_t seq = q->next_seq;

  if (q->head - q->tail == q->ring_cap)
    ring_grow(q);

  q->ring[q->head & (q->ring_cap - 1)] = (struct io_job){
    .fn = fn,
    .ctx = ctx,
    .ctx_free = ctx_free,
    .seq = seq,
  };
  q->head++;

  WakeConditionVariable(&q->cond_not_empty);
  ReleaseSRWLockExclusive(&q->srw);

  return seq;
}

struct io_event
io_queue_record(struct io_queue* q)
{
  AcquireSRWLockExclusive(&q->srw);
  struct io_event ev = { .seq = q->next_seq };
  ReleaseSRWLockExclusive(&q->srw);
  return ev;
}

void
io_event_wait(const struct io_queue* q, struct io_event ev)
{
  struct io_queue* mq = (struct io_queue*)q;

  AcquireSRWLockExclusive(&mq->srw);
  while (mq->retired_seq < ev.seq && !mq->shutdown)
    SleepConditionVariableSRW(&mq->cond_retired, &mq->srw, INFINITE, 0);
  ReleaseSRWLockExclusive(&mq->srw);
}

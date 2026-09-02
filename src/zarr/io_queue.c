#include "zarr/io_queue.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>

#define DEFAULT_MAX_REQUESTS 1024u

struct io_ready
{
  const struct io_request* request;
  uint64_t seq;
};

struct io_queue
{
  struct platform_thread** threads;
  uint64_t thread_count;
  struct platform_mutex* mutex;
  struct platform_cond* ready;
  struct platform_cond* room;
  struct platform_cond* drained;
  struct io_ready* ring;
  uint64_t ring_capacity;
  uint64_t head;
  uint64_t tail;
  uint64_t active;
  struct io_backend backend;
  struct io_queue_observer observer;
  int shutdown;
};

static void
queue_finish(struct io_queue* queue, struct io_completion completion, int busy)
{
  if (busy) {
    if (queue->observer.busy)
      queue->observer.busy(queue->observer.ctx, completion.seq);
  } else if (queue->observer.finished) {
    queue->observer.finished(queue->observer.ctx, completion);
  }

  platform_mutex_lock(queue->mutex);
  CHECK(Unlock, queue->active > 0);
  queue->active--;
  platform_cond_broadcast(queue->drained);
Unlock:
  platform_mutex_unlock(queue->mutex);
}

static void
worker_main(void* arg)
{
  struct io_queue* queue = (struct io_queue*)arg;

  for (;;) {
    platform_mutex_lock(queue->mutex);
    while (queue->head == queue->tail && !queue->shutdown)
      platform_cond_wait(queue->ready, queue->mutex);
    if (queue->head == queue->tail) {
      platform_mutex_unlock(queue->mutex);
      return;
    }

    const struct io_ready ready =
      queue->ring[queue->tail & (queue->ring_capacity - 1)];
    queue->tail++;
    queue->active++;
    platform_cond_broadcast(queue->room);
    platform_mutex_unlock(queue->mutex);

    if (queue->observer.started)
      queue->observer.started(queue->observer.ctx, ready.seq);

    struct io_completion completion = {
      .seq = ready.seq,
      .nbytes = ready.request->nbytes,
      .status = IO_OK,
    };
    const int dispatch = queue->backend.execute(
      queue->backend.ctx, ready.request, ready.seq, &completion);
    if (dispatch == IO_DONE)
      queue_finish(queue, completion, 0);
    else if (dispatch == IO_BUSY)
      queue_finish(queue, completion, 1);
  }
}

static void
queue_free(struct io_queue* queue)
{
  if (!queue)
    return;
  free(queue->threads);
  free(queue->ring);
  platform_cond_free(queue->drained);
  platform_cond_free(queue->room);
  platform_cond_free(queue->ready);
  platform_mutex_free(queue->mutex);
  free(queue);
}

struct io_queue*
io_queue_create(struct io_backend backend,
                struct io_queue_limits limits,
                struct io_queue_observer observer)
{
  if (!backend.execute) {
    log_error("io_queue: a backend must carry an execute");
    return NULL;
  }

  struct io_queue* queue = (struct io_queue*)calloc(1, sizeof(*queue));
  if (!queue)
    return NULL;

  const uint64_t max_requests =
    limits.max_requests ? limits.max_requests : DEFAULT_MAX_REQUESTS;
  const uint64_t workers = limits.workers ? limits.workers : 1;
  queue->ring_capacity = 1ull << ceil_log2(max_requests);
  queue->backend = backend;
  queue->observer = observer;
  queue->ring =
    (struct io_ready*)calloc(queue->ring_capacity, sizeof(*queue->ring));
  queue->threads =
    (struct platform_thread**)calloc(workers, sizeof(*queue->threads));
  queue->mutex = platform_mutex_new();
  queue->ready = platform_cond_new();
  queue->room = platform_cond_new();
  queue->drained = platform_cond_new();
  if (!queue->ring || !queue->threads || !queue->mutex || !queue->ready ||
      !queue->room || !queue->drained) {
    queue_free(queue);
    return NULL;
  }

  while (queue->thread_count < workers) {
    struct platform_thread* thread = platform_thread_start(worker_main, queue);
    if (!thread)
      break;
    queue->threads[queue->thread_count++] = thread;
  }
  if (queue->thread_count != workers) {
    log_error("io_queue: could only start %llu of %llu workers",
              (unsigned long long)queue->thread_count,
              (unsigned long long)workers);
    io_queue_destroy(queue);
    return NULL;
  }
  return queue;
}

void
io_queue_destroy(struct io_queue* queue)
{
  if (!queue)
    return;

  platform_mutex_lock(queue->mutex);
  queue->shutdown = 1;
  platform_cond_broadcast(queue->ready);
  platform_cond_broadcast(queue->room);
  platform_mutex_unlock(queue->mutex);

  for (uint64_t i = 0; i < queue->thread_count; ++i)
    platform_thread_join(queue->threads[i]);

  platform_mutex_lock(queue->mutex);
  while (queue->active > 0)
    platform_cond_wait(queue->drained, queue->mutex);
  platform_mutex_unlock(queue->mutex);

  if (queue->backend.stop)
    queue->backend.stop(queue->backend.ctx);
  queue_free(queue);
}

int
io_queue_post(struct io_queue* queue,
              const struct io_request* request,
              uint64_t seq)
{
  platform_mutex_lock(queue->mutex);
  while (queue->head - queue->tail == queue->ring_capacity && !queue->shutdown)
    platform_cond_wait(queue->room, queue->mutex);
  if (queue->shutdown) {
    platform_mutex_unlock(queue->mutex);
    return 1;
  }
  queue->ring[queue->head & (queue->ring_capacity - 1)] =
    (struct io_ready){ .request = request, .seq = seq };
  queue->head++;
  platform_cond_broadcast(queue->ready);
  platform_mutex_unlock(queue->mutex);
  return 0;
}

void
io_queue_complete(struct io_queue* queue, struct io_completion completion)
{
  queue_finish(queue, completion, 0);
}

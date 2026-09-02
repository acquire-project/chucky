#include "zarr/io_scheduler.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_REQUESTS 1024u
#define NO_SEQ 0u

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
  uint64_t older_on_file;
  uint64_t newer_on_file;
  uint8_t state;
};

struct file_pending
{
  uint64_t generation;
  uint64_t outstanding;
  uint64_t in_flight;
  uint64_t oldest_seq;
  uint64_t newest_seq;
  uint64_t turn;
  uint8_t barrier_running;
  uint8_t closing;
};

struct io_scheduler
{
  struct platform_thread** workers;
  uint64_t worker_count;
  struct io_backend backend;

  struct platform_mutex* mutex;
  struct platform_cond* ready;
  struct platform_cond* retired;
  struct platform_cond* room;

  struct io_job* ring;
  uint64_t ring_cap;
  uint64_t head;
  uint64_t tail;

  struct file_pending* files;
  uint64_t files_cap;
  uint32_t* order;
  uint64_t norder;
  uint64_t next_turn;
  uint64_t nofile_oldest;
  uint64_t nofile_newest;

  uint64_t max_requests;
  uint64_t max_bytes;
  uint64_t max_in_flight_per_file;
  uint64_t pending_bytes;
  uint64_t waiters;
  int shutdown;
};

static struct io_job*
job_at(struct io_scheduler* q, uint64_t seq)
{
  return &q->ring[seq & (q->ring_cap - 1)];
}

static void
queue_wait(struct io_scheduler* q, struct platform_cond* cond)
{
  q->waiters++;
  platform_cond_wait(cond, q->mutex);
  if (--q->waiters == 0)
    platform_cond_broadcast(q->retired);
}

static int
is_barrier(const struct io_request* req)
{
  return req->op == IO_OP_TRUNCATE || req->op == IO_OP_CLOSE;
}

static void
list_append(struct io_scheduler* q,
            uint64_t* oldest,
            uint64_t* newest,
            uint64_t seq)
{
  struct io_job* job = job_at(q, seq);
  job->older_on_file = *newest;
  job->newer_on_file = NO_SEQ;
  if (*newest != NO_SEQ)
    job_at(q, *newest)->newer_on_file = seq;
  else
    *oldest = seq;
  *newest = seq;
}

static void
list_remove(struct io_scheduler* q,
            uint64_t* oldest,
            uint64_t* newest,
            uint64_t seq)
{
  struct io_job* job = job_at(q, seq);
  if (job->older_on_file != NO_SEQ)
    job_at(q, job->older_on_file)->newer_on_file = job->newer_on_file;
  else
    *oldest = job->newer_on_file;
  if (job->newer_on_file != NO_SEQ)
    job_at(q, job->newer_on_file)->older_on_file = job->older_on_file;
  else
    *newest = job->older_on_file;
}

static void
order_add(struct io_scheduler* q, uint32_t index)
{
  q->files[index].turn = q->norder;
  q->order[q->norder++] = index;
}

static void
order_remove(struct io_scheduler* q, uint32_t index)
{
  const uint64_t turn = q->files[index].turn;
  const uint32_t moved = q->order[--q->norder];
  if (turn < q->norder) {
    q->order[turn] = moved;
    q->files[moved].turn = turn;
  }
}

static int
files_make_room(struct io_scheduler* q, uint32_t index)
{
  if (index < q->files_cap)
    return 0;

  uint64_t cap = q->files_cap ? q->files_cap : 16;
  while (cap <= index)
    cap *= 2;

  if (cap > SIZE_MAX / sizeof(*q->files) || cap > SIZE_MAX / sizeof(*q->order))
    return 1;

  struct file_pending* files =
    (struct file_pending*)calloc((size_t)cap, sizeof(*files));
  uint32_t* order = (uint32_t*)malloc((size_t)cap * sizeof(*order));
  if (!files || !order) {
    free(files);
    free(order);
    return 1;
  }
  if (q->files_cap > 0)
    memcpy(files, q->files, (size_t)q->files_cap * sizeof(*files));
  if (q->norder > 0)
    memcpy(order, q->order, (size_t)q->norder * sizeof(*order));

  free(q->files);
  free(q->order);
  q->files = files;
  q->order = order;
  q->files_cap = cap;
  return 0;
}

static struct file_pending*
file_find(struct io_scheduler* q, struct io_file_token file)
{
  if (file.generation == 0 || file.index >= q->files_cap)
    return NULL;
  struct file_pending* found = &q->files[file.index];
  return found->generation == file.generation ? found : NULL;
}

static void
file_request_added(struct io_scheduler* q,
                   uint64_t seq,
                   const struct io_request* req)
{
  if (req->file.generation == 0) {
    list_append(q, &q->nofile_oldest, &q->nofile_newest, seq);
    return;
  }

  struct file_pending* file = &q->files[req->file.index];
  if (file->generation != req->file.generation) {
    if (file->outstanding == 0)
      order_add(q, req->file.index);
    *file = (struct file_pending){ .generation = req->file.generation,
                                   .oldest_seq = NO_SEQ,
                                   .newest_seq = NO_SEQ,
                                   .turn = file->turn };
  }

  file->outstanding++;
  list_append(q, &file->oldest_seq, &file->newest_seq, seq);
  if (req->op == IO_OP_CLOSE)
    file->closing = 1;
}

static void
file_request_retired(struct io_scheduler* q,
                     uint64_t seq,
                     const struct io_request* req)
{
  if (req->file.generation == 0) {
    list_remove(q, &q->nofile_oldest, &q->nofile_newest, seq);
    return;
  }

  struct file_pending* file = file_find(q, req->file);
  if (!file)
    return;
  if (file->outstanding == 0) {
    log_error("io_scheduler: an open file entry was already at zero");
    return;
  }

  list_remove(q, &file->oldest_seq, &file->newest_seq, seq);
  file->in_flight--;
  if (is_barrier(req))
    file->barrier_running = 0;
  if (--file->outstanding == 0) {
    order_remove(q, req->file.index);
    file->generation = 0;
  }
}

static uint64_t
file_next_ready(struct io_scheduler* q, const struct file_pending* file)
{
  if (file->in_flight >= q->max_in_flight_per_file)
    return NO_SEQ;
  for (uint64_t seq = file->oldest_seq; seq != NO_SEQ;) {
    const struct io_job* job = job_at(q, seq);
    if (job->state == IO_JOB_WAITING) {
      if (is_barrier(&job->req))
        return file->oldest_seq == seq ? seq : NO_SEQ;
      return file->barrier_running ? NO_SEQ : seq;
    }
    seq = job->newer_on_file;
  }
  return NO_SEQ;
}

static uint64_t
next_ready_seq(struct io_scheduler* q)
{
  for (uint64_t seq = q->nofile_oldest; seq != NO_SEQ;) {
    const struct io_job* job = job_at(q, seq);
    if (job->state == IO_JOB_WAITING)
      return seq;
    seq = job->newer_on_file;
  }

  for (uint64_t i = 0; i < q->norder; ++i) {
    const uint64_t turn = (q->next_turn + i) % q->norder;
    const uint64_t seq = file_next_ready(q, &q->files[q->order[turn]]);
    if (seq != NO_SEQ) {
      q->next_turn = turn + 1;
      return seq;
    }
  }
  return NO_SEQ;
}

static void
retire_job(struct io_scheduler* q, uint64_t seq, const struct io_request* req)
{
  if (req->owned)
    req->owned_free(req->owned);
  if (req->finished)
    req->finished(req->finished_ctx);

  platform_mutex_lock(q->mutex);
  struct io_job* slot = job_at(q, seq);
  CHECK(Unlock, slot->seq == seq && slot->state == IO_JOB_RUNNING);

  slot->state = IO_JOB_RETIRED;
  while (q->tail < q->head && job_at(q, q->tail)->state == IO_JOB_RETIRED)
    q->tail++;
  q->pending_bytes -= req->nbytes;
  file_request_retired(q, seq, req);
  platform_cond_broadcast(q->ready);
  platform_cond_broadcast(q->retired);
  platform_cond_broadcast(q->room);

Unlock:
  platform_mutex_unlock(q->mutex);
}

static void
worker_main(void* arg)
{
  struct io_scheduler* q = (struct io_scheduler*)arg;
  for (;;) {
    platform_mutex_lock(q->mutex);
    uint64_t seq;
    while ((seq = next_ready_seq(q)) == NO_SEQ) {
      if (q->shutdown && q->tail == q->head)
        break;
      platform_cond_wait(q->ready, q->mutex);
    }
    if (seq == NO_SEQ) {
      platform_mutex_unlock(q->mutex);
      return;
    }

    struct io_job* slot = job_at(q, seq);
    slot->state = IO_JOB_RUNNING;
    struct file_pending* file = file_find(q, slot->req.file);
    if (file) {
      file->in_flight++;
      if (is_barrier(&slot->req))
        file->barrier_running = 1;
    }
    const struct io_request req = slot->req;
    platform_mutex_unlock(q->mutex);

    q->backend.execute(q->backend.ctx, &req);
    retire_job(q, seq, &req);
  }
}

static void
scheduler_free(struct io_scheduler* q)
{
  if (!q)
    return;
  free(q->workers);
  free(q->ring);
  free(q->files);
  free(q->order);
  platform_cond_free(q->room);
  platform_cond_free(q->retired);
  platform_cond_free(q->ready);
  platform_mutex_free(q->mutex);
  free(q);
}

struct io_scheduler*
io_scheduler_create(struct io_backend backend,
                    struct io_scheduler_limits limits)
{
  if (!backend.execute) {
    log_error("io_scheduler: a backend must carry an execute");
    return NULL;
  }

  struct io_scheduler* q = (struct io_scheduler*)calloc(1, sizeof(*q));
  if (!q)
    return NULL;

  q->backend = backend;
  q->max_requests =
    limits.max_requests ? limits.max_requests : DEFAULT_MAX_REQUESTS;
  q->max_bytes = limits.max_bytes;
  q->max_in_flight_per_file =
    limits.max_in_flight_per_file ? limits.max_in_flight_per_file : 1;
  const uint64_t workers = limits.workers ? limits.workers : 1;
  if (q->max_requests > SIZE_MAX / sizeof(*q->ring) ||
      workers > SIZE_MAX / sizeof(*q->workers)) {
    log_error("io_scheduler: limits are too large");
    free(q);
    return NULL;
  }
  q->ring_cap = 1ull << ceil_log2(q->max_requests);
  if (q->ring_cap > SIZE_MAX / sizeof(*q->ring)) {
    log_error("io_scheduler: request ring is too large");
    free(q);
    return NULL;
  }
  q->head = 1;
  q->tail = 1;
  q->ring = (struct io_job*)calloc(q->ring_cap, sizeof(*q->ring));
  q->workers = (struct platform_thread**)calloc(workers, sizeof(*q->workers));
  q->mutex = platform_mutex_new();
  q->ready = platform_cond_new();
  q->retired = platform_cond_new();
  q->room = platform_cond_new();
  if (!q->ring || !q->workers || !q->mutex || !q->ready || !q->retired ||
      !q->room) {
    scheduler_free(q);
    return NULL;
  }

  while (q->worker_count < workers) {
    struct platform_thread* worker = platform_thread_start(worker_main, q);
    if (!worker)
      break;
    q->workers[q->worker_count++] = worker;
  }
  if (q->worker_count != workers) {
    log_error("io_scheduler: could only start %llu of %llu workers",
              (unsigned long long)q->worker_count,
              (unsigned long long)workers);
    io_scheduler_destroy(q);
    return NULL;
  }
  return q;
}

void
io_scheduler_destroy(struct io_scheduler* q)
{
  if (!q)
    return;

  platform_mutex_lock(q->mutex);
  q->shutdown = 1;
  platform_cond_broadcast(q->ready);
  platform_cond_broadcast(q->room);
  platform_cond_broadcast(q->retired);
  platform_mutex_unlock(q->mutex);

  for (uint64_t i = 0; i < q->worker_count; ++i)
    platform_thread_join(q->workers[i]);

  platform_mutex_lock(q->mutex);
  while (q->waiters)
    platform_cond_wait(q->retired, q->mutex);
  platform_mutex_unlock(q->mutex);
  scheduler_free(q);
}

uint64_t
io_scheduler_parked_threads(const struct io_scheduler* q)
{
  struct io_scheduler* mutable_q = (struct io_scheduler*)q;
  platform_mutex_lock(mutable_q->mutex);
  const uint64_t parked = mutable_q->waiters;
  platform_mutex_unlock(mutable_q->mutex);
  return parked;
}

static int
has_room(const struct io_scheduler* q, uint64_t nbytes)
{
  if (q->head - q->tail >= q->max_requests)
    return 0;
  if (q->max_bytes == 0)
    return 1;
  if (q->head == q->tail)
    return 1;
  if (q->pending_bytes > q->max_bytes)
    return 0;
  return nbytes <= q->max_bytes - q->pending_bytes;
}

static int
file_is_unavailable(struct io_scheduler* q, const struct io_request* req)
{
  if (req->file.generation == 0)
    return 0;
  const struct file_pending* file = &q->files[req->file.index];
  if (file->generation == 0 || file->outstanding == 0)
    return 0;
  if (file->generation == req->file.generation)
    return file->closing;

  return !file->closing || file->outstanding != 1 || file->in_flight != 1 ||
         !file->barrier_running || file->oldest_seq != file->newest_seq;
}

int
io_scheduler_post(struct io_scheduler* q, struct io_request req)
{
  platform_mutex_lock(q->mutex);
  while (!has_room(q, req.nbytes) && !q->shutdown)
    queue_wait(q, q->room);

  int refused = 0;
  if (q->shutdown) {
    log_error("io_scheduler: refused a post during shutdown");
    refused = 1;
  } else if (req.file.generation != 0 && files_make_room(q, req.file.index)) {
    log_error("io_scheduler: could not grow the open file table");
    refused = 1;
  } else if (req.file.generation != 0 && file_is_unavailable(q, &req)) {
    log_error("io_scheduler: refused a request naming a file still in use");
    refused = 1;
  }

  if (!refused) {
    const uint64_t seq = q->head;
    *job_at(q, seq) = (struct io_job){
      .req = req, .seq = seq, .older_on_file = NO_SEQ, .newer_on_file = NO_SEQ
    };
    q->head++;
    q->pending_bytes += req.nbytes;
    file_request_added(q, seq, &req);
    platform_cond_broadcast(q->ready);
  }
  platform_mutex_unlock(q->mutex);
  return refused;
}

uint64_t
io_scheduler_pending_bytes(const struct io_scheduler* q)
{
  struct io_scheduler* mutable_q = (struct io_scheduler*)q;
  platform_mutex_lock(mutable_q->mutex);
  const uint64_t pending = mutable_q->pending_bytes;
  platform_mutex_unlock(mutable_q->mutex);
  return pending;
}

struct io_event
io_scheduler_record(struct io_scheduler* q)
{
  platform_mutex_lock(q->mutex);
  const struct io_event event = { .seq = q->head - 1 };
  platform_mutex_unlock(q->mutex);
  return event;
}

void
io_event_wait(const struct io_scheduler* q, struct io_event event)
{
  struct io_scheduler* mutable_q = (struct io_scheduler*)q;
  platform_mutex_lock(mutable_q->mutex);
  while (mutable_q->tail - 1 < event.seq && !mutable_q->shutdown)
    queue_wait(mutable_q, mutable_q->retired);
  platform_mutex_unlock(mutable_q->mutex);
}

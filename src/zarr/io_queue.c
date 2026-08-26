#include "zarr/io_queue.h"
#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/io_queue_stats.h"

#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_REQUESTS 1024u
#define NO_SEQ 0u
// With nothing in flight there is nothing to wake a wait, so a refused
// request is offered again on a timer.
#define BUSY_RETRY_NS 100000LL

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
  // Requests naming the same file are linked oldest first by sequence
  // number, with NO_SEQ for an end.
  uint64_t older_on_file;
  uint64_t newer_on_file;
  uint8_t state;
};

// One open file, found by the index its token carries. An entry lives from
// the first request naming that file until the last one retires.
struct file_pending
{
  uint64_t generation;
  uint64_t outstanding; // unretired requests naming this file
  uint64_t writes;      // of those, the ones carrying a payload
  uint64_t in_flight;   // of those, the ones handed to the backend
  uint64_t oldest_seq;  // NO_SEQ when nothing is outstanding
  uint64_t newest_seq;
  uint64_t turn; // where this file sits in the round-robin order
  uint8_t barrier_running;
  uint8_t closing;
};

struct io_queue
{
  struct platform_thread** threads;
  uint64_t nthreads;

  struct platform_mutex* mutex;
  struct platform_cond* cond_not_empty;
  struct platform_cond* cond_retired;
  struct platform_cond* cond_slot_free;

  struct io_job* ring;
  uint64_t ring_cap; // power of two, fixed for the queue's life
  uint64_t head;     // next sequence number to hand out
  uint64_t tail;     // oldest sequence number that has not retired

  struct file_pending* files; // indexed by the token's file index
  uint64_t files_cap;
  uint32_t* order; // the live entries, in the order they take turns
  uint64_t norder;
  uint64_t next_turn; // where the next dispatch starts looking

  // A request naming no file is always ready, so it needs no entry. These
  // are kept on a list of their own and looked at first.
  uint64_t nofile_oldest;
  uint64_t nofile_newest;

  uint64_t max_requests;
  uint64_t max_bytes;
  uint64_t max_in_flight;
  uint64_t max_in_flight_per_file;

  uint64_t pending_bytes; // raised on commit, lowered when the job finishes
  uint64_t jobs_waiting;  // committed and not yet taken by a worker
  uint64_t in_flight;     // handed to the backend and not yet finished
  uint64_t files_with_writes;

  // This room is held for a reserve until it is committed or released.
  uint64_t reserved_requests;
  uint64_t reserved_bytes;

  // The lock cannot be freed until every parked thread has let go of it.
  uint64_t waiters;

  struct io_queue_counters counters;

  struct io_backend backend;

  int shutdown;
};

static struct io_job*
job_at(struct io_queue* q, uint64_t seq)
{
  return &q->ring[seq & (q->ring_cap - 1)];
}

static void
queue_wait(struct io_queue* q, struct platform_cond* cond)
{
  q->waiters++;
  platform_cond_wait(cond, q->mutex);
  if (--q->waiters == 0)
    platform_cond_broadcast(q->cond_retired);
}

static int
is_barrier(const struct io_request* req)
{
  return req->op == IO_OP_TRUNCATE || req->op == IO_OP_CLOSE;
}

// --- Lists threaded through the ring ---

static void
list_append(struct io_queue* q,
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
list_remove(struct io_queue* q,
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
  job->older_on_file = NO_SEQ;
  job->newer_on_file = NO_SEQ;
}

// --- Round-robin order over the live files ---

static void
order_add(struct io_queue* q, uint32_t index)
{
  q->files[index].turn = q->norder;
  q->order[q->norder++] = index;
}

static void
order_remove(struct io_queue* q, uint32_t index)
{
  const uint64_t turn = q->files[index].turn;
  q->order[turn] = q->order[--q->norder];
  q->files[q->order[turn]].turn = turn;
}

// --- The open file table ---

// Both tables are indexed by the token's file index, so both are grown to
// fit it. Non-zero is returned when they could not be.
static int
files_make_room(struct io_queue* q, uint32_t index)
{
  if (index < q->files_cap)
    return 0;

  uint64_t cap = q->files_cap ? q->files_cap : 16;
  while (cap <= index)
    cap *= 2;

  struct file_pending* files =
    (struct file_pending*)realloc(q->files, cap * sizeof(*files));
  if (!files)
    return 1;
  q->files = files;
  memset(q->files + q->files_cap, 0, (cap - q->files_cap) * sizeof(*files));

  uint32_t* order = (uint32_t*)realloc(q->order, cap * sizeof(*order));
  if (!order)
    return 1;
  q->order = order;

  q->files_cap = cap;
  return 0;
}

static struct file_pending*
file_find(struct io_queue* q, struct io_file_token file)
{
  if (file.generation == 0 || file.index >= q->files_cap)
    return NULL;
  struct file_pending* f = &q->files[file.index];
  return f->generation == file.generation ? f : NULL;
}

// The count of files with a write outstanding is what the depth available is
// measured from.
static void
writes_on_file_changed(struct io_queue* q, int64_t delta, int64_t now)
{
  q->files_with_writes = (uint64_t)((int64_t)q->files_with_writes + delta);
  io_queue_counters_files_waiting(&q->counters, q->files_with_writes, now);
}

static void
file_request_added(struct io_queue* q,
                   uint64_t seq,
                   const struct io_request* req,
                   int64_t now)
{
  if (req->file.generation == 0) {
    list_append(q, &q->nofile_oldest, &q->nofile_newest, seq);
    return;
  }

  struct file_pending* f = &q->files[req->file.index];

  // A backend hands an index back when the close naming it runs, which is
  // before that close retires, so a new file can claim the entry with the old
  // close still on its way out. That close is the only request the old
  // generation can have left, and retiring it finds a generation that no
  // longer matches.
  if (f->generation != req->file.generation) {
    if (f->outstanding == 0)
      order_add(q, req->file.index);
    *f = (struct file_pending){ .generation = req->file.generation,
                                .oldest_seq = NO_SEQ,
                                .newest_seq = NO_SEQ,
                                .turn = f->turn };
  }

  f->outstanding++;
  list_append(q, &f->oldest_seq, &f->newest_seq, seq);
  if (req->op == IO_OP_CLOSE)
    f->closing = 1;
  if (req->nbytes > 0 && f->writes++ == 0)
    writes_on_file_changed(q, +1, now);
}

static void
file_request_retired(struct io_queue* q,
                     uint64_t seq,
                     const struct io_request* req,
                     int64_t now)
{
  if (req->file.generation == 0) {
    list_remove(q, &q->nofile_oldest, &q->nofile_newest, seq);
    return;
  }

  struct file_pending* f = file_find(q, req->file);
  if (!f)
    return;

  // An entry is dropped as its last request retires, so one still naming
  // this generation and holding nothing means the counts below would wrap.
  if (f->outstanding == 0) {
    log_error("io_queue: an open file entry was already at zero");
    return;
  }

  list_remove(q, &f->oldest_seq, &f->newest_seq, seq);
  f->in_flight--;
  if (is_barrier(req))
    f->barrier_running = 0;
  if (req->nbytes > 0 && --f->writes == 0)
    writes_on_file_changed(q, -1, now);
  if (--f->outstanding == 0) {
    order_remove(q, req->file.index);
    f->generation = 0;
  }
}

// Writes to one file run together; a truncate or close is held behind
// everything posted ahead of it on that file, and holds back everything
// posted after it. Sound only for the oldest waiting request on the file:
// everything ahead of that one is running, so a running barrier is the only
// one that can be in the way.
static int
job_is_ready(const struct file_pending* f, const struct io_job* job)
{
  if (is_barrier(&job->req))
    return f->oldest_seq == job->seq;
  return !f->barrier_running;
}

// The request on this file to run next, or NO_SEQ. Everything ahead of the
// oldest waiting one is running, so if that one cannot run, nor can anything
// behind it.
static uint64_t
file_next_ready(struct io_queue* q, const struct file_pending* f)
{
  if (f->in_flight >= q->max_in_flight_per_file)
    return NO_SEQ;
  for (uint64_t s = f->oldest_seq; s != NO_SEQ;) {
    const struct io_job* job = job_at(q, s);
    if (job->state == IO_JOB_WAITING)
      return job_is_ready(f, job) ? s : NO_SEQ;
    s = job->newer_on_file;
  }
  return NO_SEQ;
}

// Files take turns, so a backlog on one of them cannot keep the others idle.
static uint64_t
next_ready_seq(struct io_queue* q)
{
  if (q->max_in_flight && q->in_flight >= q->max_in_flight)
    return NO_SEQ;

  for (uint64_t s = q->nofile_oldest; s != NO_SEQ;) {
    const struct io_job* job = job_at(q, s);
    if (job->state == IO_JOB_WAITING)
      return s;
    s = job->newer_on_file;
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

// Room is given back against the requested size, not the size written; a
// partial write would otherwise leak the difference.
static void
retire_job(struct io_queue* q, struct io_completion c, int64_t finished_ns)
{
  void* owned = NULL;
  void (*owned_free)(void*) = NULL;

  platform_mutex_lock(q->mutex);

  struct io_job* slot = job_at(q, c.seq);
  CHECK(Unlock, slot->seq == c.seq && slot->state == IO_JOB_RUNNING);

  const struct io_job job = *slot;
  owned = job.req.owned;
  owned_free = job.req.owned_free;

  slot->state = IO_JOB_RETIRED;
  while (q->tail < q->head && job_at(q, q->tail)->state == IO_JOB_RETIRED)
    q->tail++;
  q->pending_bytes -= job.req.nbytes;
  q->in_flight--;
  io_queue_counters_in_flight(&q->counters, q->in_flight, finished_ns);
  file_request_retired(q, c.seq, &job.req, finished_ns);
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

// A refused request is put back where it was, so nothing is reordered. Zero
// is returned when a newer open has taken its file entry over.
static int
requeue_job(struct io_queue* q, uint64_t seq, int64_t now)
{
  int requeued = 0;
  int waited_for_a_retirement = 0;

  platform_mutex_lock(q->mutex);

  struct io_job* slot = job_at(q, seq);
  CHECK(Unlock, slot->seq == seq && slot->state == IO_JOB_RUNNING);
  struct file_pending* f = file_find(q, slot->req.file);
  CHECK_SILENT(Unlock, f || slot->req.file.generation == 0);

  slot->state = IO_JOB_WAITING;
  q->jobs_waiting++;
  q->in_flight--;
  io_queue_counters_in_flight(&q->counters, q->in_flight, now);

  if (f) {
    f->in_flight--;
    if (is_barrier(&slot->req))
      f->barrier_running = 0;
  }
  requeued = 1;

  // The backend is full until something already handed over finishes.
  if (q->in_flight > 0) {
    waited_for_a_retirement = 1;
    platform_cond_wait(q->cond_not_empty, q->mutex);
  }

Unlock:
  platform_mutex_unlock(q->mutex);
  if (requeued && !waited_for_a_retirement)
    platform_sleep_ns(BUSY_RETRY_NS);
  return requeued;
}

static void
worker_thread(void* arg)
{
  struct io_queue* q = (struct io_queue*)arg;

  for (;;) {
    platform_mutex_lock(q->mutex);
    uint64_t seq;
    // A submitted request is still running, so only an empty window means
    // everything has drained.
    while ((seq = next_ready_seq(q)) == NO_SEQ) {
      if (q->shutdown && q->tail == q->head)
        break;
      platform_cond_wait(q->cond_not_empty, q->mutex);
    }

    if (seq == NO_SEQ) {
      platform_mutex_unlock(q->mutex);
      break;
    }

    struct io_job* slot = job_at(q, seq);
    slot->state = IO_JOB_RUNNING;
    const int64_t now = platform_monotonic_ns();
    // Only work with a payload is timed, so a truncate or close has no start.
    slot->started_ns = slot->req.nbytes > 0 ? now : 0;
    q->jobs_waiting--;
    q->in_flight++;
    io_queue_counters_in_flight(&q->counters, q->in_flight, now);
    struct file_pending* f = file_find(q, slot->req.file);
    if (f) {
      f->in_flight++;
      if (is_barrier(&slot->req))
        f->barrier_running = 1;
    }
    // The request has to outlive the call, so the slot is passed, not a copy.
    const struct io_request* req = &slot->req;
    struct io_completion done = {
      .seq = seq,
      .nbytes = req->nbytes,
      .status = IO_OK,
    };
    platform_mutex_unlock(q->mutex);

    const int dispatch = q->backend.execute(q->backend.ctx, req, seq, &done);

    if (dispatch == IO_DONE) {
      retire_job(q, done, platform_monotonic_ns());
    } else if (dispatch == IO_BUSY) {
      const int64_t now = platform_monotonic_ns();
      // A request on no list can never be picked up again, so it is finished
      // here instead.
      if (!requeue_job(q, seq, now)) {
        log_error("io_queue: a refused request named a file that was reopened");
        done.nbytes = 0;
        done.status = IO_CANCELLED;
        retire_job(q, done, now);
      }
    }
  }
}

static void
queue_free(struct io_queue* q)
{
  free(q->threads);
  free(q->ring);
  free(q->files);
  free(q->order);
  platform_cond_free(q->cond_slot_free);
  platform_cond_free(q->cond_retired);
  platform_cond_free(q->cond_not_empty);
  platform_mutex_free(q->mutex);
  free(q);
}

struct io_queue*
io_queue_create(struct io_backend backend, struct io_queue_limits limits)
{
  if (!backend.execute) {
    log_error("io_queue: a backend must carry an execute");
    return NULL;
  }

  struct io_queue* q = (struct io_queue*)calloc(1, sizeof(*q));
  if (!q)
    return NULL;

  q->backend = backend;
  q->max_requests =
    limits.max_requests ? limits.max_requests : DEFAULT_MAX_REQUESTS;
  q->max_bytes = limits.max_bytes;
  q->max_in_flight = limits.writes_in_flight;
  q->max_in_flight_per_file =
    limits.writes_in_flight_per_file ? limits.writes_in_flight_per_file : 1;
  // Masking the sequence number picks a slot, so the ring must be a power of
  // two even when the request limit is not.
  q->ring_cap = 1ull << ceil_log2(q->max_requests);
  // Zero means nothing was ever posted, so an event recorded before the first
  // post has nothing to wait for.
  q->head = 1;
  q->tail = 1;
  q->ring = (struct io_job*)calloc(q->ring_cap, sizeof(struct io_job));
  q->mutex = platform_mutex_new();
  q->cond_not_empty = platform_cond_new();
  q->cond_retired = platform_cond_new();
  q->cond_slot_free = platform_cond_new();

  const uint64_t workers = limits.workers ? limits.workers : 1;
  q->threads = (struct platform_thread**)calloc(workers, sizeof(*q->threads));

  if (!q->ring || !q->mutex || !q->cond_not_empty || !q->cond_retired ||
      !q->cond_slot_free || !q->threads) {
    queue_free(q);
    return NULL;
  }

  while (q->nthreads < workers) {
    struct platform_thread* t = platform_thread_start(worker_thread, q);
    if (!t)
      break;
    q->threads[q->nthreads++] = t;
  }

  // A queue short of workers still runs everything, but the depth a caller
  // asked for is one of the things being measured.
  if (q->nthreads != workers) {
    log_error("io_queue: could only start %llu of %llu workers",
              (unsigned long long)q->nthreads,
              (unsigned long long)workers);
    io_queue_destroy(q);
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

  for (uint64_t i = 0; i < q->nthreads; ++i)
    platform_thread_join(q->threads[i]);

  if (q->backend.stop)
    q->backend.stop(q->backend.ctx);

  // Shutdown is set, so nothing parks from here on.
  platform_mutex_lock(q->mutex);
  while (q->waiters)
    platform_cond_wait(q->cond_retired, q->mutex);
  platform_mutex_unlock(q->mutex);

  queue_free(q);
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
  // A write larger than the ceiling still has to go through, so it is
  // admitted when the queue is empty.
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

  const struct file_pending* f = file_find(q, req->file);
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
    // A close can be posted while the wait for room is going on.
    log_error("io_queue: refused a request naming a file that is closing");
    refused = 1;
  } else if (req.file.generation != 0 &&
             files_make_room(q, req.file.index)) {
    // Taken here because a commit cannot fail, and a request left off the
    // list is never run.
    log_error("io_queue: could not grow the open file table");
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
  *job_at(q, seq) = (struct io_job){
    .req = req,
    .seq = seq,
    .post_ns = now,
    .older_on_file = NO_SEQ,
    .newer_on_file = NO_SEQ,
  };
  q->head++;
  q->pending_bytes += req.nbytes;
  q->jobs_waiting++;
  file_request_added(q, seq, &req, now);

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
  struct io_queue* mq = (struct io_queue*)q;

  platform_mutex_lock(mq->mutex);
  while (mq->tail - 1 < ev.seq && !mq->shutdown)
    queue_wait(mq, mq->cond_retired);
  platform_mutex_unlock(mq->mutex);
}

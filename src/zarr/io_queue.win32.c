#include "zarr/io_queue.h"
#include "log/log.h"
#include "platform/platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdlib.h>
#include <string.h>

struct io_job
{
  struct io_work work;
  uint64_t seq;
  int64_t post_ns;
};

// One open file with work waiting on it.
struct file_waiting
{
  uint64_t file;
  uint64_t jobs;
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

  uint64_t next_seq;      // incremented on post
  uint64_t retired_seq;   // updated after each job completes
  uint64_t pending_bytes; // raised on post, lowered when the job finishes

  // Files with work waiting, and the time-weighted history of how many there
  // were. Weighting needs an absolute clock: the average has to be over time,
  // not over the number of times the count happened to change.
  struct file_waiting* files;
  uint64_t nfiles;
  uint64_t files_cap;
  int64_t start_ns;
  int64_t weighted_from_ns;
  double files_weighted_ns;

  struct io_queue_stats stats;

  int shutdown;
  int started;
};

// Fold the span since the last change into the time-weighted total. Call
// before every change to nfiles, and again when reading the average out.
static void
fold_files_waiting(struct io_queue* q)
{
  int64_t now = platform_monotonic_ns();
  q->files_weighted_ns +=
    (double)q->nfiles * (double)(now - q->weighted_from_ns);
  q->weighted_from_ns = now;
}

static void
file_work_added(struct io_queue* q, uint64_t file)
{
  if (file == 0)
    return;

  for (uint64_t i = 0; i < q->nfiles; ++i) {
    if (q->files[i].file == file) {
      q->files[i].jobs++;
      return;
    }
  }

  fold_files_waiting(q);

  if (q->nfiles == q->files_cap) {
    uint64_t cap = q->files_cap ? q->files_cap * 2 : 16;
    struct file_waiting* grown =
      (struct file_waiting*)realloc(q->files, cap * sizeof(*grown));
    // Losing this only costs accuracy in a counter, so carry on untracked
    // rather than failing a write.
    if (!grown)
      return;
    q->files = grown;
    q->files_cap = cap;
  }

  q->files[q->nfiles++] = (struct file_waiting){ .file = file, .jobs = 1 };
  if (q->nfiles > q->stats.files_waiting_peak)
    q->stats.files_waiting_peak = q->nfiles;
}

static void
file_work_finished(struct io_queue* q, uint64_t file)
{
  if (file == 0)
    return;

  for (uint64_t i = 0; i < q->nfiles; ++i) {
    if (q->files[i].file != file)
      continue;
    if (--q->files[i].jobs == 0) {
      fold_files_waiting(q);
      q->files[i] = q->files[q->nfiles - 1];
      q->nfiles--;
    }
    return;
  }
}

static uint64_t
size_bucket(uint64_t nbytes)
{
  uint64_t bucket = 0;
  while (nbytes > 1 && bucket + 1 < IO_SIZE_BUCKETS) {
    nbytes >>= 1;
    bucket++;
  }
  return bucket;
}

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

    int64_t started_ns = platform_monotonic_ns();
    job.work.fn(job.work.ctx);
    if (job.work.ctx_free)
      job.work.ctx_free(job.work.ctx);
    int64_t finished_ns = platform_monotonic_ns();

    AcquireSRWLockExclusive(&q->srw);
    q->retired_seq = job.seq;
    q->pending_bytes -= job.work.nbytes;
    file_work_finished(q, job.work.file);
    if (job.work.nbytes > 0) {
      double wait_ms = (double)(started_ns - job.post_ns) / 1e6;
      double run_ms = (double)(finished_ns - started_ns) / 1e6;
      q->stats.wait_ms_total += wait_ms;
      q->stats.run_ms_total += run_ms;
      if (wait_ms > q->stats.wait_ms_max)
        q->stats.wait_ms_max = wait_ms;
      if (run_ms > q->stats.run_ms_max)
        q->stats.run_ms_max = run_ms;
    }
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

  q->start_ns = platform_monotonic_ns();
  q->weighted_from_ns = q->start_ns;

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
  free(q->files);
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
  AcquireSRWLockExclusive(&q->srw);

  if (q->head - q->tail == q->ring_cap) {
    if (ring_grow(q)) {
      ReleaseSRWLockExclusive(&q->srw);
      return 1;
    }
  }

  q->next_seq++;
  q->ring[q->head & (q->ring_cap - 1)] = (struct io_job){
    .work = work,
    .seq = q->next_seq,
    .post_ns = platform_monotonic_ns(),
  };
  q->head++;
  q->pending_bytes += work.nbytes;

  file_work_added(q, work.file);

  uint64_t waiting = q->head - q->tail;
  if (waiting > q->stats.jobs_waiting_peak)
    q->stats.jobs_waiting_peak = waiting;
  if (q->pending_bytes > q->stats.bytes_waiting_peak)
    q->stats.bytes_waiting_peak = q->pending_bytes;
  if (work.nbytes > 0) {
    q->stats.writes++;
    q->stats.size_buckets[size_bucket(work.nbytes)]++;
    if (work.borrowed)
      q->stats.bytes_borrowed += work.nbytes;
    else
      q->stats.bytes_copied += work.nbytes;
  }

  WakeConditionVariable(&q->cond_not_empty);
  ReleaseSRWLockExclusive(&q->srw);
  return 0;
}

uint64_t
io_queue_pending_bytes(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  AcquireSRWLockExclusive(&mq->srw);
  uint64_t pending = mq->pending_bytes;
  ReleaseSRWLockExclusive(&mq->srw);
  return pending;
}

void
io_queue_get_stats(const struct io_queue* q, struct io_queue_stats* out)
{
  struct io_queue* mq = (struct io_queue*)q;
  AcquireSRWLockExclusive(&mq->srw);
  fold_files_waiting(mq);
  *out = mq->stats;
  int64_t observed_ns = mq->weighted_from_ns - mq->start_ns;
  out->files_waiting_mean =
    observed_ns > 0 ? mq->files_weighted_ns / (double)observed_ns : 0.0;
  ReleaseSRWLockExclusive(&mq->srw);
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

int
io_queue_is_shutdown(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  AcquireSRWLockExclusive(&mq->srw);
  int r = mq->shutdown;
  ReleaseSRWLockExclusive(&mq->srw);
  return r;
}

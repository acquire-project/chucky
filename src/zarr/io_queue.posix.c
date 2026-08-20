#include "zarr/io_queue.h"
#include "log/log.h"
#include "platform/platform.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct io_job
{
  struct io_work work;
  uint64_t seq;
  int64_t post_ns;
};

// One open file with a write waiting on it. Truncate and close carry no
// payload and cannot be run alongside anything, so they do not count.
struct file_waiting
{
  uint64_t file;
  uint64_t writes;
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

  // Files with a write waiting, and the time-weighted history of how many
  // there were. Weighting needs an absolute clock: the average has to be over
  // time, not over the number of times the count happened to change. The window
  // opens on the first post, so a slow startup cannot dilute the average.
  struct file_waiting* files;
  uint64_t nfiles;
  uint64_t files_cap;
  int64_t start_ns;
  int64_t weighted_from_ns;
  double files_weighted_ns;

  // Timings land when a write finishes, so they are averaged over finished
  // writes rather than over stats.writes, which counts every write posted.
  uint64_t writes_finished;
  double wait_ms_total;
  double run_ms_total;

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
      q->files[i].writes++;
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

  q->files[q->nfiles++] = (struct file_waiting){ .file = file, .writes = 1 };
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
    if (--q->files[i].writes == 0) {
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

    int64_t started_ns = platform_monotonic_ns();
    job.work.fn(job.work.ctx);
    if (job.work.ctx_free)
      job.work.ctx_free(job.work.ctx);
    int64_t finished_ns = platform_monotonic_ns();

    pthread_mutex_lock(&q->mutex);
    q->retired_seq = job.seq;
    q->pending_bytes -= job.work.nbytes;
    if (job.work.nbytes > 0) {
      file_work_finished(q, job.work.file);
      double wait_ms = (double)(started_ns - job.post_ns) / 1e6;
      double run_ms = (double)(finished_ns - started_ns) / 1e6;
      q->writes_finished++;
      q->wait_ms_total += wait_ms;
      q->run_ms_total += run_ms;
      if (wait_ms > q->stats.wait_ms_max)
        q->stats.wait_ms_max = wait_ms;
      if (run_ms > q->stats.run_ms_max)
        q->stats.run_ms_max = run_ms;
    }
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
  free(q->files);
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
  int64_t now = platform_monotonic_ns();
  if (q->start_ns == 0) {
    q->start_ns = now;
    q->weighted_from_ns = now;
  }
  q->ring[q->head & (q->ring_cap - 1)] = (struct io_job){
    .work = work,
    .seq = q->next_seq,
    .post_ns = now,
  };
  q->head++;
  q->pending_bytes += work.nbytes;

  uint64_t waiting = q->head - q->tail;
  if (waiting > q->stats.jobs_waiting_peak)
    q->stats.jobs_waiting_peak = waiting;
  if (q->pending_bytes > q->stats.bytes_waiting_peak)
    q->stats.bytes_waiting_peak = q->pending_bytes;
  if (work.nbytes > 0) {
    file_work_added(q, work.file);
    q->stats.writes++;
    q->stats.size_buckets[size_bucket(work.nbytes)]++;
    if (work.borrowed)
      q->stats.bytes_borrowed += work.nbytes;
    else
      q->stats.bytes_copied += work.nbytes;
  }

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
  fold_files_waiting(mq);
  *out = mq->stats;
  int64_t observed_ns = mq->start_ns ? mq->weighted_from_ns - mq->start_ns : 0;
  out->files_waiting_mean =
    observed_ns > 0 ? mq->files_weighted_ns / (double)observed_ns : 0.0;
  const double finished = (double)mq->writes_finished;
  out->wait_ms_mean = finished > 0 ? mq->wait_ms_total / finished : 0.0;
  out->run_ms_mean = finished > 0 ? mq->run_ms_total / finished : 0.0;
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

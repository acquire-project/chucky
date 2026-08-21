#include "zarr/io_queue_stats.h"

#include <stdlib.h>

// One open file with a write waiting on it.
struct file_waiting
{
  uint64_t file;
  uint64_t writes;
};

// Call before every change to the count, and again when reading the average
// out: the total is weighted by how long each count held.
static void
fold_files_waiting(struct io_queue_counters* c, int64_t now)
{
  c->files_weighted_ns +=
    (double)c->nfiles * (double)(now - c->weighted_from_ns);
  c->weighted_from_ns = now;
}

static void
file_write_added(struct io_queue_counters* c, uint64_t file, int64_t now)
{
  if (file == 0)
    return;

  for (uint64_t i = 0; i < c->nfiles; ++i) {
    if (c->files[i].file == file) {
      c->files[i].writes++;
      return;
    }
  }

  fold_files_waiting(c, now);

  if (c->nfiles == c->files_cap) {
    uint64_t cap = c->files_cap ? c->files_cap * 2 : 16;
    struct file_waiting* grown =
      (struct file_waiting*)realloc(c->files, cap * sizeof(*grown));
    // Losing this only costs accuracy in a counter, so carry on untracked
    // rather than failing a write.
    if (!grown)
      return;
    c->files = grown;
    c->files_cap = cap;
  }

  c->files[c->nfiles++] = (struct file_waiting){ .file = file, .writes = 1 };
  if (c->nfiles > c->published.files_waiting_peak)
    c->published.files_waiting_peak = c->nfiles;
}

static void
file_write_finished(struct io_queue_counters* c, uint64_t file, int64_t now)
{
  if (file == 0)
    return;

  for (uint64_t i = 0; i < c->nfiles; ++i) {
    if (c->files[i].file != file)
      continue;
    if (--c->files[i].writes == 0) {
      fold_files_waiting(c, now);
      c->files[i] = c->files[c->nfiles - 1];
      c->nfiles--;
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

void
io_queue_counters_free(struct io_queue_counters* c)
{
  free(c->files);
  c->files = NULL;
  c->nfiles = 0;
  c->files_cap = 0;
}

void
io_queue_counters_posted(struct io_queue_counters* c,
                         const struct io_work* work,
                         uint64_t jobs_waiting,
                         uint64_t bytes_waiting,
                         int64_t now)
{
  if (c->start_ns == 0) {
    c->start_ns = now;
    c->weighted_from_ns = now;
  }

  if (jobs_waiting > c->published.jobs_waiting_peak)
    c->published.jobs_waiting_peak = jobs_waiting;
  if (bytes_waiting > c->published.bytes_waiting_peak)
    c->published.bytes_waiting_peak = bytes_waiting;

  if (work->nbytes == 0)
    return;

  file_write_added(c, work->file, now);
  c->published.writes++;
  c->published.size_buckets[size_bucket(work->nbytes)]++;
  if (work->borrowed)
    c->published.bytes_borrowed += work->nbytes;
  else
    c->published.bytes_copied += work->nbytes;
}

void
io_queue_counters_finished(struct io_queue_counters* c,
                           const struct io_work* work,
                           int64_t post_ns,
                           int64_t started_ns,
                           int64_t finished_ns)
{
  if (work->nbytes == 0)
    return;

  file_write_finished(c, work->file, finished_ns);

  const double wait_ms = (double)(started_ns - post_ns) / 1e6;
  const double run_ms = (double)(finished_ns - started_ns) / 1e6;
  c->writes_finished++;
  c->wait_ms_total += wait_ms;
  c->run_ms_total += run_ms;
  if (wait_ms > c->published.wait_ms_max)
    c->published.wait_ms_max = wait_ms;
  if (run_ms > c->published.run_ms_max)
    c->published.run_ms_max = run_ms;
}

void
io_queue_counters_read(struct io_queue_counters* c,
                       struct io_queue_stats* out,
                       int64_t now)
{
  fold_files_waiting(c, now);
  *out = c->published;

  const int64_t observed_ns =
    c->start_ns ? c->weighted_from_ns - c->start_ns : 0;
  out->files_waiting_mean =
    observed_ns > 0 ? c->files_weighted_ns / (double)observed_ns : 0.0;

  const double finished = (double)c->writes_finished;
  out->wait_ms_mean = finished > 0 ? c->wait_ms_total / finished : 0.0;
  out->run_ms_mean = finished > 0 ? c->run_ms_total / finished : 0.0;
}

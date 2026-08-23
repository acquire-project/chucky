#include "zarr/io_queue_stats.h"

// Call before every change to the count, and again on read.
static void
fold_files_waiting(struct io_queue_counters* c, int64_t now)
{
  c->files_weighted_ns +=
    (double)c->files_waiting * (double)(now - c->weighted_from_ns);
  c->weighted_from_ns = now;
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
io_queue_counters_files_waiting(struct io_queue_counters* c,
                                uint64_t files,
                                int64_t now)
{
  fold_files_waiting(c, now);
  c->files_waiting = files;
  if (files > c->published.files_waiting_peak)
    c->published.files_waiting_peak = files;
}

void
io_queue_counters_posted(struct io_queue_counters* c,
                         const struct io_request* req,
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

  if (req->nbytes == 0)
    return;

  c->published.writes++;
  c->published.size_buckets[size_bucket(req->nbytes)]++;
  if (req->borrowed)
    c->published.bytes_borrowed += req->nbytes;
  else
    c->published.bytes_copied += req->nbytes;
}

void
io_queue_counters_finished(struct io_queue_counters* c,
                           const struct io_request* req,
                           int64_t post_ns,
                           int64_t started_ns,
                           int64_t finished_ns)
{
  if (req->nbytes == 0)
    return;

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

// The counters the io queue keeps about the writes it runs. None of it is
// platform-specific, and there are two queue implementations, so it lives
// here rather than in both.
//
// The queue owns a lock. Call all of this with that lock held.
#pragma once

#include "zarr/io_queue.h"
#include "zarr/types.io.h"

#include <stdint.h>

struct file_waiting;

struct io_queue_counters
{
  // Weighting needs an absolute clock: the average has to be over time, not
  // over the number of times the count happened to change.
  struct file_waiting* files;
  uint64_t nfiles;
  uint64_t files_cap;
  int64_t start_ns;
  int64_t weighted_from_ns;
  double files_weighted_ns;

  // Timings land when a write finishes, so their divisor is the finished
  // count rather than the posted count.
  uint64_t writes_finished;
  double wait_ms_total;
  double run_ms_total;

  // Everything a reader gets except the three averages, which are worked out
  // on read and are zero in here.
  struct io_queue_stats published;
};

void
io_queue_counters_free(struct io_queue_counters* c);

// jobs_waiting and bytes_waiting are the queue's totals after taking the
// work. now is the post time, which also opens the averaging window.
void
io_queue_counters_posted(struct io_queue_counters* c,
                         const struct io_work* work,
                         uint64_t jobs_waiting,
                         uint64_t bytes_waiting,
                         int64_t now);

// The three times are read only for work carrying a payload.
void
io_queue_counters_finished(struct io_queue_counters* c,
                           const struct io_work* work,
                           int64_t post_ns,
                           int64_t started_ns,
                           int64_t finished_ns);

void
io_queue_counters_read(struct io_queue_counters* c,
                       struct io_queue_stats* out,
                       int64_t now);

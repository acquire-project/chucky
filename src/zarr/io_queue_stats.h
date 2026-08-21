// The counters the io queue keeps about the writes it runs. Split out because
// none of it is platform-specific and there are two queue implementations;
// keeping one copy is what stops them drifting.
//
// The queue owns a lock. Call all of this with that lock held.
#pragma once

#include "zarr/io_queue.h"
#include "zarr/types.io.h"

#include <stdint.h>

struct file_waiting;

struct io_queue_counters
{
  // Files with a write waiting, and the time-weighted history of how many
  // there were. Weighting needs an absolute clock: the average has to be over
  // time, not over the number of times the count happened to change. The
  // window opens on the first post, so a slow startup cannot dilute it.
  struct file_waiting* files;
  uint64_t nfiles;
  uint64_t files_cap;
  int64_t start_ns;
  int64_t weighted_from_ns;
  double files_weighted_ns;

  // Timings land when a write finishes, so they are averaged over finished
  // writes rather than over `published.writes`, which counts every write
  // posted.
  uint64_t writes_finished;
  double wait_ms_total;
  double run_ms_total;

  // Everything a reader gets except the three averages, which are worked out
  // in io_queue_counters_read and are zero in here.
  struct io_queue_stats published;
};

void
io_queue_counters_free(struct io_queue_counters* c);

// Work was posted. jobs_waiting and bytes_waiting are the queue's totals
// after taking it; now is the post time, which also opens the window.
void
io_queue_counters_posted(struct io_queue_counters* c,
                         const struct io_work* work,
                         uint64_t jobs_waiting,
                         uint64_t bytes_waiting,
                         int64_t now);

// Work finished. The three times are only read for work carrying a payload.
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

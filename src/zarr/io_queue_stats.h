// Call all of this with the queue's lock held.
#pragma once

#include "zarr/io_queue.h"
#include "zarr/types.io.h"

#include <stdint.h>

struct io_queue_counters
{
  // Average is over time, not over how often the count changed.
  int64_t start_ns;
  int64_t weighted_from_ns;
  double files_weighted_ns;
  double in_flight_weighted_ns;
  uint64_t files_waiting;
  uint64_t in_flight;

  uint64_t writes_finished;
  double wait_ms_total;
  double run_ms_total;

  // Zero in here: the three averages, worked out on read.
  struct io_queue_stats published;
};

// The queue owns the table of open files and reports the count of those with
// a write outstanding.
void
io_queue_counters_files_waiting(struct io_queue_counters* c,
                                uint64_t files,
                                int64_t now);

// Requests handed to the backend and not yet finished.
void
io_queue_counters_in_flight(struct io_queue_counters* c,
                            uint64_t in_flight,
                            int64_t now);

// Totals as of after taking the work; now also opens the averaging window.
void
io_queue_counters_posted(struct io_queue_counters* c,
                         const struct io_request* req,
                         uint64_t jobs_waiting,
                         uint64_t bytes_waiting,
                         int64_t now);

// The three times are read only for work carrying a payload.
void
io_queue_counters_finished(struct io_queue_counters* c,
                           const struct io_request* req,
                           int64_t post_ns,
                           int64_t started_ns,
                           int64_t finished_ns);

void
io_queue_counters_read(struct io_queue_counters* c,
                       struct io_queue_stats* out,
                       int64_t now);

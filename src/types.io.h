// What the write path measured. Shared by the queue that runs the writes,
// the pool that owns the files, and the benchmarks that report both.
#pragma once

#include <stdint.h>

// Powers of two from one byte up. Bucket i counts requests of at least
// 2^i bytes and fewer than 2^(i+1). Writes today run from 4 KiB footers to
// 256 MiB shard bodies, so 40 buckets leaves plenty of room above.
#define IO_SIZE_BUCKETS 40

// Counts are for the queue's whole life. Peaks and the time-weighted average
// cover the span from the first posted work to the moment they are read, so a
// slow startup does not dilute the average.
struct io_queue_stats
{
  // How many distinct files had work waiting at once. This is the depth
  // available to a scheduler that could run writes at the same time. It is
  // not the depth achieved: one worker runs one write at a time, so the
  // achieved figure is always one and says nothing.
  uint64_t files_waiting_peak;
  double files_waiting_mean;

  uint64_t jobs_waiting_peak;
  uint64_t bytes_waiting_peak;

  uint64_t writes;         // jobs that carried a payload
  uint64_t bytes_copied;   // payload bytes the job owned
  uint64_t bytes_borrowed; // payload bytes owned by someone else

  // Time a write spent waiting to start, and time it spent running. The
  // averages cover writes that have finished, which is fewer than `writes`
  // whenever the stats are read with work still queued.
  double wait_ms_mean;
  double wait_ms_max;
  double run_ms_mean;
  double run_ms_max;

  uint64_t size_buckets[IO_SIZE_BUCKETS];
};

// The queue reports its own work; the file counts belong to the pool, because
// the queue only ever sees opaque file ids.
struct shard_pool_io_stats
{
  struct io_queue_stats queue;
  uint64_t files_opened;    // shard files opened over the whole run
  uint64_t files_open_peak; // most open at once
};

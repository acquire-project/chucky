// What the write path measured.
#pragma once

#include <stdint.h>

// Bucket i holds requests of at least 2^i bytes and fewer than 2^(i+1).
// Writes run from 4 KiB footers to 256 MiB shard bodies, so 40 leaves room.
#define IO_SIZE_BUCKETS 40

// Counts cover the queue's whole life. Peaks and the time-weighted average
// cover the first posted work up to the read, so a slow startup does not
// dilute them.
struct io_queue_stats
{
  // The depth available to a scheduler that could run writes at the same
  // time. Not the depth achieved: one worker runs one write at a time, so
  // that figure is always one and says nothing. Truncate and close carry no
  // payload, so a file that is only finalizing does not count.
  uint64_t files_waiting_peak;
  double files_waiting_mean;

  uint64_t jobs_waiting_peak;
  uint64_t bytes_waiting_peak;

  uint64_t writes;         // jobs that carried a payload
  uint64_t bytes_copied;   // payload bytes the job owned
  uint64_t bytes_borrowed; // payload bytes owned by someone else

  // Averaged over writes that have finished, which is fewer than the writes
  // posted whenever these are read with work still queued.
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

// What was measured on the write path.
#pragma once

#include <stdint.h>

// Bucket i: at least 2^i bytes, under 2^(i+1). 40 covers 4 KiB to 256 MiB.
#define IO_SIZE_BUCKETS 40

// Counts over the whole run; peaks and the mean only from the first post.
struct io_queue_stats
{
  // Room for writes in flight; not the number achieved, which is always one.
  uint64_t files_waiting_peak;
  double files_waiting_mean;

  uint64_t jobs_waiting_peak;
  uint64_t bytes_waiting_peak;

  uint64_t writes;         // jobs with a payload
  uint64_t bytes_copied;   // payload bytes owned by the job
  uint64_t bytes_borrowed; // payload bytes owned by someone else

  // Divisor is the finished count, not the posted count.
  double wait_ms_mean;
  double wait_ms_max;
  double run_ms_mean;
  double run_ms_max;

  uint64_t size_buckets[IO_SIZE_BUCKETS];
};

// File counts are the pool's: file ids are opaque to the queue.
struct shard_pool_io_stats
{
  struct io_queue_stats queue;
  uint64_t files_opened;    // shard files opened over the whole run
  uint64_t files_open_peak; // most open at once
};

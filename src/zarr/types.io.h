// The write path's measurements and its scheduling are carried in these
// types.
#pragma once

#include <stdint.h>

// The write requests are carried out by one of these.
enum io_backend_choice
{
  IO_BACKEND_THREADS = 0, // blocking writes on the queue's own workers
  IO_BACKEND_URING,       // io_uring, on Linux only
};

// How much of a filesystem sink's write backlog runs at once. A zero field
// takes the default.
struct io_scheduling
{
  uint64_t workers;
  uint64_t writes_in_flight;

  // Above one, shard files are pre-sized: on some filesystems a write that
  // extends a file takes the file's lock for itself.
  uint64_t writes_in_flight_per_file;

  enum io_backend_choice backend;
};

// Bucket i: at least 2^i bytes, under 2^(i+1). 40 covers 4 KiB to 256 MiB.
#define IO_SIZE_BUCKETS 40

// Counts over the whole run; peaks and the mean only from the first post.
struct io_queue_stats
{
  // Files with a write waiting: the depth available.
  uint64_t files_waiting_peak;
  double files_waiting_mean;

  // Requests the backend is running at once: the depth reached.
  uint64_t writes_in_flight_peak;
  double writes_in_flight_mean;

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

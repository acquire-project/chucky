// Filesystem-backed shard writer pool.
// Uses a command scheduler over asynchronous filesystem workers.
#pragma once

#include "zarr/io_backend.h"
#include "zarr/shard_pool.h"

#include <stdint.h>

struct io_scheduler;

// Create a filesystem shard pool with nslots writer slots.
// root: filesystem root path (keys are relative to this).
// unbuffered: use O_DIRECT for shard writes.
// io: how much of the write backlog runs at once; NULL takes the defaults.
// Returns NULL on error.
struct shard_pool*
shard_pool_fs_create(const char* root,
                     uint64_t nslots,
                     int unbuffered,
                     const struct io_scheduling* io);

// Fill zero fields with the write-scheduling defaults, so a caller that has
// to record what it used has the numbers rather than the zeros.
void
shard_pool_fs_scheduling_defaults(struct io_scheduling* io);

// A test's own backend can be called in place of the filesystem one, to make
// requests fail or block. Every field may be left null.
struct shard_pool_fs_wrapper
{
  void* ctx;
  // The pool's backend is the argument, and its executor calls the result.
  struct io_backend (*wrap)(void* ctx, struct io_backend inner);
  struct io_scheduler** queue; // receives the scheduler the pool built
};

// Create a filesystem shard pool with the wrapper between its queue and its
// filesystem backend.
struct shard_pool*
shard_pool_fs_create_wrapped(const char* root,
                             uint64_t nslots,
                             int unbuffered,
                             const struct io_scheduling* io,
                             struct shard_pool_fs_wrapper wrapper);

// Test helper: mark the pool errored so later deliveries fail and stay queued.
void
shard_pool_fs_set_error(struct shard_pool* pool);

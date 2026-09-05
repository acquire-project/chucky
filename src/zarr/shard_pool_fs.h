// Filesystem-backed shard writer pool.
// Uses a command scheduler over asynchronous filesystem workers.
#pragma once

#include "zarr/io_backend.h"
#include "zarr/shard_pool.h"

#include <stdint.h>

struct io_scheduler;
struct io_scheduler_limits;

// Handles belong to current slots or unfinished closes, bounding their count
// by nslots plus the scheduler's max_requests (1024 by default).
struct shard_pool*
shard_pool_fs_create(const char* root, uint64_t nslots, int unbuffered);

// A test's own backend can be called in place of the filesystem one, to make
// requests fail or block. Every field may be left null.
struct shard_pool_fs_wrapper
{
  void* ctx;
  // The pool's backend is the argument, and its executor calls the result.
  struct io_backend (*wrap)(void* ctx, struct io_backend inner);
  struct io_scheduler** queue; // receives the scheduler the pool built
};

// Create a filesystem shard pool with the wrapper between its scheduler and its
// filesystem backend.
struct shard_pool*
shard_pool_fs_create_wrapped(const char* root,
                             uint64_t nslots,
                             int unbuffered,
                             const struct io_scheduler_limits* limits,
                             struct shard_pool_fs_wrapper wrapper);

// Test helper: mark the pool errored so later deliveries fail and stay queued.
void
shard_pool_fs_set_error(struct shard_pool* pool);
